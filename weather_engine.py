"""
Weather Prediction Engine
==========================
Multi-model ensemble weather forecasting with Bayesian calibration.

REVOLUTIONARY APPROACH:
1. Fetch forecasts from 8+ global NWP models via Open-Meteo (free)
2. Compute forecast error distributions from historical data
3. Use Kernel Density Estimation to build probability distributions
4. Apply Bayesian updating as new model runs become available
5. Calibrate using isotonic regression on historical forecast-vs-actual pairs
"""

import numpy as np
import pandas as pd
import requests
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os
import time

import config


class WeatherEngine:
    """Multi-model ensemble weather prediction with probability calibration."""

    def __init__(self, station_id: str = "NYC"):
        self.station_id = station_id
        self.station = config.STATIONS[station_id]
        self.lat = self.station["lat"]
        self.lon = self.station["lon"]
        self.unit = self.station.get("unit", "fahrenheit")
        self.tz = self.station.get("tz", "America/New_York")

        # Calibration data (loaded from historical analysis)
        self.error_distributions = {}  # model_name -> {lead_time_hours -> error_stats}
        self.calibration_model = None
        self.historical_data = None
        self.model_weights = dict(config.MODEL_WEIGHTS)

    # ═══════════════════════════════════════════════════════════════
    # CORE: Multi-Model Forecast Fetching
    # ═══════════════════════════════════════════════════════════════

    # In-memory forecast cache: {(station_id, target_date, variable): {model: value}}
    _forecast_cache: Dict[tuple, Dict[str, float]] = {}
    _forecast_cache_time: Dict[tuple, float] = {}  # cache timestamps
    CACHE_TTL_SECONDS = 5400  # Refresh forecasts every 90 min (reduce API load)

    # Global rate-limit state shared across all WeatherEngine instances
    _global_rate_limited = False
    _global_rate_limit_until = 0.0

    def fetch_multi_model_forecasts(
        self, target_date: str, variable: str = "temperature_2m_max"
    ) -> Dict[str, float]:
        """
        Fetch forecasts from multiple NWP models for a target date.
        Uses BATCHED requests (all models in 1 call) to minimize API load.
        Results are cached in memory to avoid redundant API calls across scan cycles.

        Returns: {model_name: forecast_value}
        """
        cache_key = (self.station_id, target_date, variable)
        if cache_key in WeatherEngine._forecast_cache:
            cache_age = time.time() - WeatherEngine._forecast_cache_time.get(cache_key, 0)
            if cache_age < WeatherEngine.CACHE_TTL_SECONDS:
                return WeatherEngine._forecast_cache[cache_key]

        # If globally rate-limited, skip entirely until cooldown expires
        if WeatherEngine._global_rate_limited:
            if time.time() < WeatherEngine._global_rate_limit_until:
                return {}
            else:
                WeatherEngine._global_rate_limited = False

        forecasts = {}

        # ─── STRATEGY: Batch all non-default models in ONE API call ───
        # Open-Meteo accepts comma-separated model names, returning all in one response.
        # This cuts API calls from 8 per city to just 2 (1 batch + 1 best_match).

        non_default_models = [m for m in config.WEATHER_MODELS if m != "best_match"]

        # Call 1: Batch all named models in a single request
        if non_default_models:
            try:
                params = {
                    "latitude": self.lat,
                    "longitude": self.lon,
                    "daily": variable,
                    "timezone": self.tz,
                    "start_date": target_date,
                    "end_date": target_date,
                    "temperature_unit": self.unit,
                    "models": ",".join(non_default_models),
                }
                if config.OPEN_METEO_API_KEY:
                    params["apikey"] = config.OPEN_METEO_API_KEY

                resp = requests.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    # Batch response has model-specific keys: "daily" contains
                    # "temperature_2m_max_ecmwf_ifs025", etc.
                    if "daily" in data:
                        daily = data["daily"]
                        # Try model-specific keys first (batch response format)
                        for model in non_default_models:
                            model_key = f"{variable}_{model}"
                            if model_key in daily:
                                values = daily[model_key]
                                if values and values[0] is not None:
                                    forecasts[model] = float(values[0])
                        # Also check plain key (single-model response format)
                        if variable in daily and len(forecasts) == 0:
                            values = daily[variable]
                            if values and values[0] is not None:
                                # Assign to first model as fallback
                                forecasts[non_default_models[0]] = float(values[0])
                elif resp.status_code == 429:
                    wait_time = 120  # 2 minute global cooldown
                    print(f"  RATE LIMITED (429): Global cooldown {wait_time}s for {self.station_id}")
                    WeatherEngine._global_rate_limited = True
                    WeatherEngine._global_rate_limit_until = time.time() + wait_time
                    time.sleep(5)  # Brief sleep before returning
                    return {}

                time.sleep(1.0)  # Pause between calls
            except Exception as e:
                print(f"  Warning: Batch forecast failed for {self.station_id}: {e}")
                time.sleep(1.0)

        # Call 2: Fetch best_match (default model, no "models" param)
        if "best_match" in config.WEATHER_MODELS:
            try:
                params = {
                    "latitude": self.lat,
                    "longitude": self.lon,
                    "daily": variable,
                    "timezone": self.tz,
                    "start_date": target_date,
                    "end_date": target_date,
                    "temperature_unit": self.unit,
                }
                if config.OPEN_METEO_API_KEY:
                    params["apikey"] = config.OPEN_METEO_API_KEY
                resp = requests.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if "daily" in data and variable in data["daily"]:
                        values = data["daily"][variable]
                        if values and values[0] is not None:
                            forecasts["best_match"] = float(values[0])
                elif resp.status_code == 429:
                    WeatherEngine._global_rate_limited = True
                    WeatherEngine._global_rate_limit_until = time.time() + 120
                time.sleep(1.0)
            except Exception as e:
                print(f"  Warning: best_match forecast failed: {e}")

        if forecasts:
            n_total = len(config.WEATHER_MODELS)
            if len(forecasts) < n_total:
                print(f"  Forecasts for {self.station_id}/{target_date}: {len(forecasts)}/{n_total} models")
        else:
            print(f"  WARNING: No forecasts for {self.station_id}/{target_date}")

        # Cache successful results (even partial)
        if forecasts:
            WeatherEngine._forecast_cache[cache_key] = forecasts
            WeatherEngine._forecast_cache_time[cache_key] = time.time()

        return forecasts

    def fetch_hourly_forecasts(
        self, target_date: str
    ) -> Dict[str, List[float]]:
        """
        Fetch hourly temperature forecasts from all models for a target date.
        Uses batched requests to minimize API load.
        """
        hourly_forecasts = {}

        # Check global rate limit
        if WeatherEngine._global_rate_limited:
            if time.time() < WeatherEngine._global_rate_limit_until:
                return {}
            else:
                WeatherEngine._global_rate_limited = False

        non_default_models = [m for m in config.WEATHER_MODELS if m != "best_match"]

        # Batch call for all named models
        if non_default_models:
            try:
                params = {
                    "latitude": self.lat,
                    "longitude": self.lon,
                    "hourly": "temperature_2m",
                    "timezone": self.tz,
                    "start_date": target_date,
                    "end_date": target_date,
                    "temperature_unit": self.unit,
                    "models": ",".join(non_default_models),
                }
                if config.OPEN_METEO_API_KEY:
                    params["apikey"] = config.OPEN_METEO_API_KEY
                resp = requests.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    if "hourly" in data:
                        hourly = data["hourly"]
                        for model in non_default_models:
                            model_key = f"temperature_2m_{model}"
                            if model_key in hourly:
                                temps = hourly[model_key]
                                valid_temps = [t for t in temps if t is not None]
                                if valid_temps:
                                    hourly_forecasts[model] = valid_temps
                        # Fallback: plain key
                        if "temperature_2m" in hourly and not hourly_forecasts:
                            temps = hourly["temperature_2m"]
                            valid_temps = [t for t in temps if t is not None]
                            if valid_temps:
                                hourly_forecasts[non_default_models[0]] = valid_temps
                elif resp.status_code == 429:
                    WeatherEngine._global_rate_limited = True
                    WeatherEngine._global_rate_limit_until = time.time() + 120
                    return {}
                time.sleep(1.0)
            except Exception as e:
                print(f"  Warning: Batch hourly forecast failed: {e}")
                time.sleep(1.0)

        # best_match call
        if "best_match" in config.WEATHER_MODELS:
            try:
                params = {
                    "latitude": self.lat,
                    "longitude": self.lon,
                    "hourly": "temperature_2m",
                    "timezone": self.tz,
                    "start_date": target_date,
                    "end_date": target_date,
                    "temperature_unit": self.unit,
                }
                if config.OPEN_METEO_API_KEY:
                    params["apikey"] = config.OPEN_METEO_API_KEY
                resp = requests.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if "hourly" in data and "temperature_2m" in data["hourly"]:
                        temps = data["hourly"]["temperature_2m"]
                        valid_temps = [t for t in temps if t is not None]
                        if valid_temps:
                            hourly_forecasts["best_match"] = valid_temps
                elif resp.status_code == 429:
                    WeatherEngine._global_rate_limited = True
                    WeatherEngine._global_rate_limit_until = time.time() + 120
                time.sleep(1.0)
            except Exception as e:
                print(f"  Warning: best_match hourly failed: {e}")

        return hourly_forecasts

    # ═══════════════════════════════════════════════════════════════
    # HISTORICAL DATA: For Calibration & Error Analysis
    # ═══════════════════════════════════════════════════════════════

    def fetch_historical_actuals(
        self, start_date: str, end_date: str, variable: str = "temperature_2m_max"
    ) -> pd.DataFrame:
        """Fetch historical actual weather data from Open-Meteo archive."""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "daily": variable,
            "timezone": self.tz,
            "start_date": start_date,
            "end_date": end_date,
            "temperature_unit": self.unit,
        }

        resp = requests.get(config.OPEN_METEO_HISTORICAL_URL, params=params, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"Historical API error: {resp.status_code}")

        data = resp.json()
        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "actual": data["daily"][variable],
        })
        df = df.dropna()
        return df

    def fetch_historical_forecasts(
        self, start_date: str, end_date: str, variable: str = "temperature_2m_max"
    ) -> pd.DataFrame:
        """
        Fetch archived model forecasts (what models predicted historically).
        This is the KEY to calibration — comparing past forecasts to actuals.
        """
        all_forecasts = []

        for model in ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "best_match"]:
            try:
                params = {
                    "latitude": self.lat,
                    "longitude": self.lon,
                    "daily": variable,
                    "timezone": self.tz,
                    "start_date": start_date,
                    "end_date": end_date,
                    "temperature_unit": self.unit,
                }
                if model != "best_match":
                    params["models"] = model

                resp = requests.get(
                    config.OPEN_METEO_HISTORICAL_FORECAST_URL, params=params, timeout=30
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "daily" in data and variable in data["daily"]:
                        df = pd.DataFrame({
                            "date": pd.to_datetime(data["daily"]["time"]),
                            f"forecast_{model}": data["daily"][variable],
                        })
                        all_forecasts.append(df)
                time.sleep(0.5)
            except Exception as e:
                print(f"  Warning: Historical forecasts for {model} failed: {e}")

        if not all_forecasts:
            return pd.DataFrame()

        result = all_forecasts[0]
        for df in all_forecasts[1:]:
            result = result.merge(df, on="date", how="outer")

        return result

    # ═══════════════════════════════════════════════════════════════
    # CALIBRATION: Learn Error Distributions
    # ═══════════════════════════════════════════════════════════════

    def _default_calibration(self) -> Dict:
        """Default calibration stats when historical data is insufficient."""
        default = {
            "mean_bias": 0.0,
            "std": 3.5,  # ~3.5°F typical forecast error
            "mae": 2.8,
            "rmse": 3.5,
            "percentile_5": -6.0,
            "percentile_95": 6.0,
            "n_samples": 0,
        }
        for model in config.WEATHER_MODELS:
            self.error_distributions[model] = default.copy()
        return {"default": default}

    def _update_model_weights(self):
        """Update model weights based on inverse RMSE (better models get more weight)."""
        if not self.error_distributions:
            return

        inverse_rmse = {}
        for model, stats in self.error_distributions.items():
            if stats["rmse"] > 0:
                inverse_rmse[model] = 1.0 / stats["rmse"]

        if not inverse_rmse:
            return

        total = sum(inverse_rmse.values())
        for model in inverse_rmse:
            self.model_weights[model] = inverse_rmse[model] / total

        print(f"  Updated model weights: {json.dumps({k: round(v, 3) for k, v in self.model_weights.items()}, indent=2)}")

    # ═══════════════════════════════════════════════════════════════
    # PROBABILITY ENGINE: Monte Carlo Ensemble
    # ═══════════════════════════════════════════════════════════════

    def compute_probability_distribution(
        self,
        forecasts: Dict[str, float],
        bucket_edges: List[float],
        n_samples: int = None,
    ) -> Dict[str, float]:
        """
        REVOLUTIONARY APPROACH: Monte Carlo ensemble probability estimation.

        For each model:
        1. Take its point forecast
        2. Correct for known bias
        3. Sample from its calibrated error distribution
        4. Weight by model accuracy

        Then combine all samples into a probability distribution over buckets.
        """
        if n_samples is None:
            n_samples = config.MC_SAMPLES

        all_samples = []
        all_weights = []

        for model, forecast in forecasts.items():
            # Get calibration for this model
            error_stats = self.error_distributions.get(model)
            if error_stats is None:
                # Use default
                error_stats = {"mean_bias": 0, "std": 3.5}

            # Bias-corrected forecast
            corrected = forecast - error_stats["mean_bias"]

            # Generate samples from calibrated error distribution
            # Use a slightly heavy-tailed distribution (Student's t) to capture extreme errors
            df_t = 5  # degrees of freedom for t-distribution (heavier tails than normal)
            samples = corrected + stats.t.rvs(
                df=df_t, loc=0, scale=error_stats["std"], size=n_samples // len(forecasts)
            )

            model_weight = self.model_weights.get(model, 1.0 / len(forecasts))
            all_samples.extend(samples)
            all_weights.extend([model_weight] * len(samples))

        all_samples = np.array(all_samples)
        all_weights = np.array(all_weights)
        all_weights /= all_weights.sum()  # Normalize

        # Compute probabilities for each bucket
        is_fahrenheit = self.unit == "fahrenheit"
        bucket_probs = {}
        for i in range(len(bucket_edges) - 1):
            low = bucket_edges[i]
            high = bucket_edges[i + 1]

            # Weighted histogram
            mask = (all_samples >= low) & (all_samples < high)
            prob = all_weights[mask].sum()
            if is_fahrenheit:
                # °F: 2-degree range labels like "34-35°F"
                bucket_label = f"{int(low)}-{int(high-1)}°F"
            else:
                # °C: single-degree labels like "14°C", "-5°C"
                bucket_label = f"{int(low)}°C"
            bucket_probs[bucket_label] = max(prob, 0.001)  # Floor at 0.1%

        # Handle tail buckets
        # Polymarket labels: "31°F or below" means temp <= 31 (i.e. < 32 = bucket_edges[0])
        # For °F: tail label = edges[0] - 1 ("31°F or below" when edges start at 32)
        # For °C: tail label = edges[0] - 1 ("12°C or below" when edges start at 13)
        mask_low = all_samples < bucket_edges[0]
        low_tail_val = int(bucket_edges[0]) - 1
        if mask_low.any():
            label = f"{low_tail_val}°F or below" if is_fahrenheit else f"{low_tail_val}°C or below"
            bucket_probs[label] = all_weights[mask_low].sum()

        mask_high = all_samples >= bucket_edges[-1]
        if mask_high.any():
            label = f"{int(bucket_edges[-1])}°F or higher" if is_fahrenheit else f"{int(bucket_edges[-1])}°C or higher"
            bucket_probs[label] = all_weights[mask_high].sum()

        # Normalize to sum to 1
        total = sum(bucket_probs.values())
        if total > 0:
            bucket_probs = {k: v / total for k, v in bucket_probs.items()}

        return bucket_probs

    def compute_ensemble_stats(self, forecasts: Dict[str, float]) -> Dict:
        """Compute ensemble statistics for confidence assessment.

        Returns a dict with standardized keys:
            mean, std, range, spread, agreement, agreement_score,
            min, max, n_models
        """
        if not forecasts:
            return {
                "mean": 0, "std": 999, "range": 999, "spread": 999,
                "agreement": 0, "agreement_score": 0,
                "min": 0, "max": 0, "n_models": 0,
            }

        values = list(forecasts.values())
        mean = np.average(
            values,
            weights=[self.model_weights.get(m, 0.1) for m in forecasts.keys()]
        )
        std = np.std(values)
        model_range = max(values) - min(values)

        # Agreement score: 1 - (normalized spread)
        # If all models agree within 2°F, agreement = 1.0
        agreement = max(0, 1.0 - (model_range / 10.0))

        return {
            "mean": float(mean),
            "std": float(std),
            "range": float(model_range),
            "spread": float(model_range),  # alias for range
            "agreement": float(agreement),
            "agreement_score": float(agreement),  # alias for agreement
            "min": float(min(values)),
            "max": float(max(values)),
            "n_models": len(values),
        }

    # ═══════════════════════════════════════════════════════════════
    # ML MODEL INTEGRATION
    # ═══════════════════════════════════════════════════════════════

    def compute_ml_probability_distribution(
        self,
        forecasts: Dict[str, float],
        ensemble_stats: Dict,
        bucket_edges: List[float],
        target_date: str,
    ) -> Optional[Dict[str, float]]:
        """
        Compute probability distribution using the trained ML model.

        Falls back to None if the ML model is not available, allowing
        the caller to use the NWP ensemble method instead.

        The ML model (WeatherMLModel) produces better predictions by
        learning systematic biases between Open-Meteo forecasts and
        Weather Underground actuals across 20 stations.
        """
        try:
            from ml_model import WeatherMLModel
        except ImportError:
            return None

        # Load model from disk (cached after first load)
        if not hasattr(self, '_ml_model'):
            self._ml_model = None
            try:
                ml = WeatherMLModel()
                ml.load(config.ML_MODEL_PATH)
                if ml.is_trained:
                    self._ml_model = ml
            except (FileNotFoundError, Exception):
                pass  # Model not available

        if self._ml_model is None:
            return None

        # Build feature dict from current forecast data
        is_fahrenheit = self.unit == "fahrenheit"
        target = pd.Timestamp(target_date)

        # Get WU bias info for this station
        wu_bias_info = self._ml_model.station_biases.get(self.station_id, {})

        features = {
            "om_high_temp": ensemble_stats.get("mean", 0),
            "om_low_temp": ensemble_stats.get("mean", 0) - 10,  # Approximate
            "om_mean_temp": ensemble_stats.get("mean", 0) - 5,  # Approximate
            "om_precip": 0,  # Not available from ensemble
            "om_wind_max": 0,
            "om_rh_mean": 50,
            "om_pressure_mean": 1013,
            "station_lat": self.lat,
            "station_lon": self.lon,
            "is_fahrenheit": 1.0 if is_fahrenheit else 0.0,
            "wu_bias": wu_bias_info.get("mean_bias", 0),
            "wu_bias_std": wu_bias_info.get("std", 2.0),
            "sin_doy": np.sin(2 * np.pi * target.dayofyear / 365.25),
            "cos_doy": np.cos(2 * np.pi * target.dayofyear / 365.25),
            "month": target.month,
            "om_temp_lag1": ensemble_stats.get("mean", 0),  # Best estimate
            "om_temp_lag3_mean": ensemble_stats.get("mean", 0),
            "om_temp_lag7_mean": ensemble_stats.get("mean", 0),
            "om_temp_change": 0,
            "temp_anomaly": 0,
            "station_code": hash(self.station_id) % 1000,
            "station_id": self.station_id,
        }

        try:
            ml_probs = self._ml_model.predict_bucket_probs(
                features, bucket_edges, is_fahrenheit=is_fahrenheit
            )
            return ml_probs
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════
    # CLIMATOLOGICAL PRIOR: Bayesian base rate
    # ═══════════════════════════════════════════════════════════════

    def get_climatological_prior(
        self, target_date: str, bucket_edges: List[float]
    ) -> Dict[str, float]:
        """
        Get climatological base rates for temperature buckets.
        Uses 30+ years of historical data for the same calendar period.
        """
        target = pd.Timestamp(target_date)
        day_of_year = target.dayofyear

        # Fetch long-term historical data (use reanalysis)
        # Look at same 2-week window across years
        start = "1990-01-01"
        end = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

        try:
            actuals = self.fetch_historical_actuals(start, end)

            # Filter to same calendar window (±7 days of target day-of-year)
            actuals["doy"] = actuals["date"].dt.dayofyear
            window = actuals[
                (actuals["doy"] >= day_of_year - 7) & (actuals["doy"] <= day_of_year + 7)
            ]

            if len(window) < 10:
                return {}

            # Compute bucket probabilities from historical distribution
            is_fahrenheit = self.unit == "fahrenheit"
            prior_probs = {}
            for i in range(len(bucket_edges) - 1):
                low = bucket_edges[i]
                high = bucket_edges[i + 1]
                count = ((window["actual"] >= low) & (window["actual"] < high)).sum()
                if is_fahrenheit:
                    label = f"{int(low)}-{int(high-1)}°F"
                else:
                    label = f"{int(low)}°C"
                prior_probs[label] = count / len(window)

            return prior_probs

        except Exception as e:
            print(f"  Warning: Climatological prior failed: {e}")
            return {}

    # ═══════════════════════════════════════════════════════════════
    # V5: ENSEMBLE PROBABILISTIC FORECASTS (Professional API)
    # ═══════════════════════════════════════════════════════════════

    # Ensemble cache (separate from point-forecast cache)
    _ensemble_cache: Dict[tuple, Dict[str, List[float]]] = {}
    _ensemble_cache_time: Dict[tuple, float] = {}

    def fetch_ensemble_forecasts(
        self, target_date: str, variable: str = "temperature_2m_max"
    ) -> Dict[str, List[float]]:
        """
        V5: Fetch ensemble member forecasts from Open-Meteo Ensemble API.

        Calls customer-ensemble-api.open-meteo.com with models=ecmwf_ifs025,gfs025
        Each model returns control + memberNN keys giving 82+ independent scenarios.

        Returns: {model_name: [member_value_1, member_value_2, ...]}
        """
        cache_key = (self.station_id, target_date, variable, "ensemble")
        if cache_key in WeatherEngine._ensemble_cache:
            cache_age = time.time() - WeatherEngine._ensemble_cache_time.get(cache_key, 0)
            if cache_age < WeatherEngine.CACHE_TTL_SECONDS:
                return WeatherEngine._ensemble_cache[cache_key]

        # Check global rate limit
        if WeatherEngine._global_rate_limited:
            if time.time() < WeatherEngine._global_rate_limit_until:
                return {}
            else:
                WeatherEngine._global_rate_limited = False

        ensemble_data = {}
        ensemble_models = getattr(config, 'ENSEMBLE_MODELS', ["ecmwf_ifs025", "gfs025"])

        try:
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "daily": variable,
                "timezone": self.tz,
                "start_date": target_date,
                "end_date": target_date,
                "temperature_unit": self.unit,
                "models": ",".join(ensemble_models),
            }
            if config.OPEN_METEO_API_KEY:
                params["apikey"] = config.OPEN_METEO_API_KEY

            resp = requests.get(config.OPEN_METEO_ENSEMBLE_URL, params=params, timeout=25)

            if resp.status_code == 200:
                data = resp.json()
                if "daily" in data:
                    daily = data["daily"]
                    for model in ensemble_models:
                        members = []
                        # Control run (plain key with model suffix)
                        control_key = f"{variable}_{model}"
                        if control_key in daily:
                            val = daily[control_key]
                            if isinstance(val, list) and val and val[0] is not None:
                                members.append(float(val[0]))

                        # Ensemble members: variable_member01_model, variable_member02_model, etc.
                        # Or: variable_model_member01 format
                        for key, val in daily.items():
                            if not key.startswith(variable):
                                continue
                            if "member" not in key:
                                continue
                            if model not in key:
                                continue
                            if isinstance(val, list) and val and val[0] is not None:
                                members.append(float(val[0]))

                        # Also try format: temperature_2m_max_member01 (single-model response)
                        if not members:
                            for key, val in daily.items():
                                if key.startswith(f"{variable}_member"):
                                    if isinstance(val, list) and val and val[0] is not None:
                                        members.append(float(val[0]))

                        if members:
                            ensemble_data[model] = members
                            print(f"  V5 Ensemble {self.station_id}/{model}: {len(members)} members")

            elif resp.status_code == 429:
                print(f"  V5 Ensemble RATE LIMITED (429) for {self.station_id}")
                WeatherEngine._global_rate_limited = True
                WeatherEngine._global_rate_limit_until = time.time() + 120
                return {}
            else:
                print(f"  V5 Ensemble API error {resp.status_code} for {self.station_id}")

            time.sleep(1.0)

        except Exception as e:
            print(f"  V5 Ensemble fetch failed for {self.station_id}: {e}")

        # Cache results
        if ensemble_data:
            WeatherEngine._ensemble_cache[cache_key] = ensemble_data
            WeatherEngine._ensemble_cache_time[cache_key] = time.time()

        return ensemble_data

    def compute_ensemble_bucket_probabilities(
        self, ensemble_members: Dict[str, List[float]], bucket_edges: List[float]
    ) -> Dict[str, float]:
        """
        V5: Compute bucket probabilities by directly counting ensemble members.

        This is the PRIMARY V5 method — no Monte Carlo or KDE needed.
        P(bucket) = (count of members in bucket + 0.5) / (total_members + 0.5 * n_buckets)
        Uses Laplace smoothing to avoid zero probabilities.
        """
        # Collect all member values
        all_members = []
        for model, members in ensemble_members.items():
            all_members.extend(members)

        if not all_members:
            return {}

        total = len(all_members)
        is_fahrenheit = self.unit == "fahrenheit"

        # Count members in each bucket (with Laplace smoothing)
        n_buckets = len(bucket_edges) - 1 + 2  # regular buckets + 2 tails
        smoothing = 0.5

        bucket_probs = {}

        # Regular buckets
        for i in range(len(bucket_edges) - 1):
            low = bucket_edges[i]
            high = bucket_edges[i + 1]
            count = sum(1 for v in all_members if low <= v < high)
            prob = (count + smoothing) / (total + smoothing * n_buckets)

            if is_fahrenheit:
                label = f"{int(low)}-{int(high - 1)}°F"
            else:
                label = f"{int(low)}°C"
            bucket_probs[label] = prob

        # Low tail
        low_tail_count = sum(1 for v in all_members if v < bucket_edges[0])
        low_tail_val = int(bucket_edges[0]) - 1
        low_tail_label = f"{low_tail_val}°F or below" if is_fahrenheit else f"{low_tail_val}°C or below"
        bucket_probs[low_tail_label] = (low_tail_count + smoothing) / (total + smoothing * n_buckets)

        # High tail
        high_tail_count = sum(1 for v in all_members if v >= bucket_edges[-1])
        high_tail_label = f"{int(bucket_edges[-1])}°F or higher" if is_fahrenheit else f"{int(bucket_edges[-1])}°C or higher"
        bucket_probs[high_tail_label] = (high_tail_count + smoothing) / (total + smoothing * n_buckets)

        # Normalize to sum to 1
        total_prob = sum(bucket_probs.values())
        if total_prob > 0:
            bucket_probs = {k: v / total_prob for k, v in bucket_probs.items()}

        return bucket_probs

    def fetch_previous_runs(
        self, target_date: str, variable: str = "temperature_2m_max"
    ) -> Dict:
        """
        V5: Fetch previous model runs to detect forecast drift.

        Calls previous-runs API with past_days=2 to compare what models
        predicted 2 days ago, 1 day ago, and today for the target date.

        Returns: {
            "direction": "warmer"|"cooler"|"stable",
            "magnitude": float (degrees of drift),
            "consistency": float (0-1, how many models agree on drift direction)
        }
        """
        past_days = getattr(config, 'PREVIOUS_RUNS_DAYS', 2)
        drift_models = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]

        try:
            # Calculate start_date to include past_days before target
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            start_date = (target_dt - timedelta(days=past_days)).strftime("%Y-%m-%d")

            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "daily": variable,
                "timezone": self.tz,
                "start_date": start_date,
                "end_date": target_date,
                "temperature_unit": self.unit,
                "past_days": past_days,
                "models": ",".join(drift_models),
            }
            if config.OPEN_METEO_API_KEY:
                params["apikey"] = config.OPEN_METEO_API_KEY

            resp = requests.get(config.OPEN_METEO_PREVIOUS_RUNS_URL, params=params, timeout=20)

            if resp.status_code != 200:
                return {"direction": "stable", "magnitude": 0.0, "consistency": 0.0}

            data = resp.json()
            if "daily" not in data:
                return {"direction": "stable", "magnitude": 0.0, "consistency": 0.0}

            daily = data["daily"]
            times = daily.get("time", [])

            # Find the index for the target date
            target_idx = None
            for i, t in enumerate(times):
                if t == target_date:
                    target_idx = i
                    break

            if target_idx is None:
                return {"direction": "stable", "magnitude": 0.0, "consistency": 0.0}

            # Extract per-model forecasts for the target date across model runs
            model_drifts = []
            for model in drift_models:
                model_key = f"{variable}_{model}"
                if model_key not in daily:
                    continue

                values = daily[model_key]
                if not isinstance(values, list) or target_idx >= len(values):
                    continue

                current_val = values[target_idx]
                if current_val is None:
                    continue

                # Compare with earlier values if available (earlier runs)
                earlier_vals = []
                for j in range(max(0, target_idx - past_days), target_idx):
                    if j < len(values) and values[j] is not None:
                        earlier_vals.append(values[j])

                if earlier_vals:
                    avg_earlier = sum(earlier_vals) / len(earlier_vals)
                    drift = current_val - avg_earlier
                    model_drifts.append(drift)

            if not model_drifts:
                return {"direction": "stable", "magnitude": 0.0, "consistency": 0.0}

            avg_drift = sum(model_drifts) / len(model_drifts)
            # Consistency: fraction of models that agree on drift direction
            if avg_drift > 0:
                agree_count = sum(1 for d in model_drifts if d > 0)
            elif avg_drift < 0:
                agree_count = sum(1 for d in model_drifts if d < 0)
            else:
                agree_count = len(model_drifts)

            consistency = agree_count / len(model_drifts) if model_drifts else 0.0

            if abs(avg_drift) < 0.5:
                direction = "stable"
            elif avg_drift > 0:
                direction = "warmer"
            else:
                direction = "cooler"

            result = {
                "direction": direction,
                "magnitude": abs(avg_drift),
                "consistency": consistency,
            }
            print(f"  V5 Drift {self.station_id}: {direction} by {abs(avg_drift):.1f}° "
                  f"(consistency: {consistency:.0%})")
            return result

        except Exception as e:
            print(f"  V5 Previous runs failed for {self.station_id}: {e}")
            return {"direction": "stable", "magnitude": 0.0, "consistency": 0.0}

    def calibrate_v5(self, lookback_days: int = 60) -> Dict:
        """
        V5: Calibration using Historical Forecast API.

        Computes per-model bias and RMSE over last 60 days,
        plus ensemble calibration percentile data.
        Saves to disk as JSON for persistence between restarts.
        """
        end_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        print(f"  V5 Calibrating {self.station_id} ({start_date} to {end_date})...")

        try:
            # Fetch actuals
            actuals = self.fetch_historical_actuals(start_date, end_date)
            if actuals.empty or len(actuals) < 20:
                print(f"  V5 Calibration: insufficient actuals for {self.station_id}")
                return {}

            # Fetch historical forecasts
            forecasts = self.fetch_historical_forecasts(start_date, end_date)
            if forecasts.empty:
                print(f"  V5 Calibration: no historical forecasts for {self.station_id}")
                return {}

            merged = actuals.merge(forecasts, on="date", how="inner").dropna()
            if len(merged) < 20:
                print(f"  V5 Calibration: too few matched days ({len(merged)}) for {self.station_id}")
                return {}

            # Compute per-model stats
            v5_cal = {}
            forecast_cols = [c for c in merged.columns if c.startswith("forecast_")]

            for col in forecast_cols:
                model_name = col.replace("forecast_", "")
                errors = merged["actual"] - merged[col]
                valid = errors.dropna()

                if len(valid) < 10:
                    continue

                v5_cal[model_name] = {
                    "bias": float(valid.mean()),
                    "rmse": float(np.sqrt((valid ** 2).mean())),
                    "mae": float(valid.abs().mean()),
                    "std": float(valid.std()),
                    "n_samples": len(valid),
                }
                print(f"  V5 {self.station_id}/{model_name}: "
                      f"bias={v5_cal[model_name]['bias']:.2f}, "
                      f"RMSE={v5_cal[model_name]['rmse']:.2f}")

            # Store calibration
            self.v5_calibration = v5_cal

            # Also update regular error_distributions for compatibility
            for model_name, cal in v5_cal.items():
                self.error_distributions[model_name] = {
                    "mean_bias": cal["bias"],
                    "std": cal["std"],
                    "mae": cal["mae"],
                    "rmse": cal["rmse"],
                    "percentile_5": -2 * cal["std"],
                    "percentile_95": 2 * cal["std"],
                    "n_samples": cal["n_samples"],
                }

            # Update model weights based on V5 calibration
            self._update_model_weights()

            # Save to disk for persistence
            os.makedirs(config.DATA_DIR, exist_ok=True)
            cal_path = os.path.join(config.DATA_DIR, f"v5_calibration_{self.station_id}.json")
            with open(cal_path, "w") as f:
                json.dump(v5_cal, f, indent=2)
            print(f"  V5 Calibration saved to {cal_path}")

            return v5_cal

        except Exception as e:
            print(f"  V5 Calibration failed for {self.station_id}: {e}")
            return {}

    def compute_probability_distribution_v5(
        self,
        ensemble_data: Dict[str, List[float]],
        bucket_edges: List[float],
        drift_info: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        V5: Compute probability distribution using ensemble member counting.

        PRIMARY: Direct member counting from 82+ ensemble scenarios
        CORRECTION: Apply bias correction from v5_calibration
        ADJUSTMENT: Shift distribution if drift_info shows consistent trend
        FALLBACK: If ensemble unavailable, falls back to V4 Monte Carlo

        Returns bucket probabilities in same format as V4 methods.
        """
        if not ensemble_data:
            return {}

        # Step 1: Apply bias correction to ensemble members
        corrected_data = {}
        v5_cal = getattr(self, 'v5_calibration', {})

        for model, members in ensemble_data.items():
            bias = 0.0
            # Look up bias from V5 calibration
            if model in v5_cal:
                bias = v5_cal[model].get("bias", 0.0)
            elif model in self.error_distributions:
                bias = self.error_distributions[model].get("mean_bias", 0.0)

            # Correct each member by subtracting the model's bias
            corrected = [m - bias for m in members]
            corrected_data[model] = corrected

        # Step 2: Apply drift adjustment if consistent
        if drift_info and drift_info.get("consistency", 0) > 0.6:
            drift_mag = drift_info.get("magnitude", 0)
            drift_dir = drift_info.get("direction", "stable")
            if drift_dir == "warmer" and drift_mag > 0.5:
                shift = drift_mag * 0.3  # Apply 30% of drift as shift
                for model in corrected_data:
                    corrected_data[model] = [m + shift for m in corrected_data[model]]
            elif drift_dir == "cooler" and drift_mag > 0.5:
                shift = drift_mag * 0.3
                for model in corrected_data:
                    corrected_data[model] = [m - shift for m in corrected_data[model]]

        # Step 3: Compute bucket probabilities via member counting
        bucket_probs = self.compute_ensemble_bucket_probabilities(corrected_data, bucket_edges)

        return bucket_probs

    def calibrate(self, lookback_years: int = None) -> Dict:
        """
        Build calibration model. V5: tries calibrate_v5 first if API key is set.
        Falls back to V4 historical calibration.
        """
        # V5: Use Pro API calibration if available
        if config.OPEN_METEO_API_KEY:
            v5_result = self.calibrate_v5()
            if v5_result:
                # Also load persisted V5 calibration data if available
                cal_path = os.path.join(config.DATA_DIR, f"v5_calibration_{self.station_id}.json")
                if os.path.exists(cal_path):
                    try:
                        with open(cal_path) as f:
                            self.v5_calibration = json.load(f)
                    except Exception:
                        pass
                return v5_result

        # V4 fallback: original calibration
        return self._calibrate_v4(lookback_years)

    def _calibrate_v4(self, lookback_years: int = None) -> Dict:
        """V4 calibration (original method, kept as fallback)."""
        if lookback_years is None:
            lookback_years = config.CALIBRATION_YEARS

        end_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

        print(f"  Calibrating {self.station_id} from {start_date} to {end_date}...")

        # Fetch actual data
        actuals = self.fetch_historical_actuals(start_date, end_date)
        print(f"  Loaded {len(actuals)} days of actual data")

        # Fetch historical forecast data
        forecasts = self.fetch_historical_forecasts(start_date, end_date)
        print(f"  Loaded historical forecasts with {len(forecasts)} days")

        if forecasts.empty or actuals.empty:
            print("  Warning: Insufficient data for calibration, using defaults")
            return self._default_calibration()

        # Merge
        merged = actuals.merge(forecasts, on="date", how="inner")
        merged = merged.dropna()
        print(f"  Merged dataset: {len(merged)} days")

        if len(merged) < 30:
            print("  Warning: Too few matched days, using defaults")
            return self._default_calibration()

        # Compute error distributions per model
        calibration_stats = {}
        forecast_cols = [c for c in merged.columns if c.startswith("forecast_")]

        for col in forecast_cols:
            model_name = col.replace("forecast_", "")
            errors = merged["actual"] - merged[col]
            valid_errors = errors.dropna()

            if len(valid_errors) < 20:
                continue

            error_stats = {
                "mean_bias": float(valid_errors.mean()),
                "std": float(valid_errors.std()),
                "mae": float(valid_errors.abs().mean()),
                "rmse": float(np.sqrt((valid_errors ** 2).mean())),
                "percentile_5": float(valid_errors.quantile(0.05)),
                "percentile_95": float(valid_errors.quantile(0.95)),
                "n_samples": len(valid_errors),
            }
            self.error_distributions[model_name] = error_stats
            calibration_stats[model_name] = error_stats
            print(f"  {model_name}: bias={error_stats['mean_bias']:.2f}°F, "
                  f"RMSE={error_stats['rmse']:.2f}°F, MAE={error_stats['mae']:.2f}°F")

        # Update model weights based on inverse RMSE
        self._update_model_weights()

        # Store for probability computation
        self.historical_data = merged
        self.calibration_model = calibration_stats

        return calibration_stats


class PrecipitationEngine(WeatherEngine):
    """Specialized engine for precipitation markets."""

    def fetch_multi_model_forecasts(
        self, target_date: str, variable: str = "precipitation_sum"
    ) -> Dict[str, float]:
        return super().fetch_multi_model_forecasts(target_date, variable)

    def fetch_historical_actuals(
        self, start_date: str, end_date: str, variable: str = "precipitation_sum"
    ) -> pd.DataFrame:
        return super().fetch_historical_actuals(start_date, end_date, variable)

    def compute_probability_distribution(
        self,
        forecasts: Dict[str, float],
        bucket_edges: List[float],
        n_samples: int = None,
    ) -> Dict[str, float]:
        """
        Precipitation needs special handling:
        - Zero-inflated distribution (many dry days)
        - Log-normal for non-zero amounts
        - Heavier tails than temperature
        """
        if n_samples is None:
            n_samples = config.MC_SAMPLES

        all_samples = []
        all_weights = []

        for model, forecast in forecasts.items():
            error_stats = self.error_distributions.get(model, {"mean_bias": 0, "std": 0.3})

            n_model = n_samples // len(forecasts)

            if forecast < 0.01:
                # Dry forecast: mostly dry samples with small chance of light rain
                dry_frac = 0.85
                n_dry = int(n_model * dry_frac)
                n_wet = n_model - n_dry
                dry_samples = np.zeros(n_dry)
                wet_samples = np.abs(np.random.exponential(0.1, n_wet))
                samples = np.concatenate([dry_samples, wet_samples])
            else:
                # Wet forecast: use gamma distribution (natural for precipitation)
                shape = max(0.5, (forecast / max(error_stats["std"], 0.1)) ** 2)
                scale = max(0.01, forecast / shape)
                samples = np.random.gamma(shape, scale, n_model)

            model_weight = self.model_weights.get(model, 1.0 / len(forecasts))
            all_samples.extend(samples)
            all_weights.extend([model_weight] * len(samples))

        all_samples = np.array(all_samples)
        all_weights = np.array(all_weights)
        all_weights /= all_weights.sum()

        # Compute bucket probabilities
        bucket_probs = {}
        for i in range(len(bucket_edges) - 1):
            low = bucket_edges[i]
            high = bucket_edges[i + 1]
            mask = (all_samples >= low) & (all_samples < high)
            prob = all_weights[mask].sum()
            bucket_label = f"{low:.1f}-{high:.1f} in"
            bucket_probs[bucket_label] = max(prob, 0.001)

        # Normalize
        total = sum(bucket_probs.values())
        if total > 0:
            bucket_probs = {k: v / total for k, v in bucket_probs.items()}

        return bucket_probs
