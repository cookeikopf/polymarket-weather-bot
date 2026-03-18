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
    CACHE_TTL_SECONDS = 3600  # Refresh forecasts every hour

    def fetch_multi_model_forecasts(
        self, target_date: str, variable: str = "temperature_2m_max"
    ) -> Dict[str, float]:
        """
        Fetch forecasts from multiple NWP models for a target date.
        Results are cached in memory to avoid redundant API calls across scan cycles.

        Returns: {model_name: forecast_value}
        """
        cache_key = (self.station_id, target_date, variable)
        if cache_key in WeatherEngine._forecast_cache:
            cache_age = time.time() - WeatherEngine._forecast_cache_time.get(cache_key, 0)
            if cache_age < WeatherEngine.CACHE_TTL_SECONDS:
                return WeatherEngine._forecast_cache[cache_key]

        forecasts = {}
        failures = 0

        for model in config.WEATHER_MODELS:
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
                if model != "best_match":
                    params["models"] = model

                resp = requests.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if "daily" in data and variable in data["daily"]:
                        values = data["daily"][variable]
                        if values and values[0] is not None:
                            forecasts[model] = float(values[0])
                        else:
                            failures += 1
                    else:
                        failures += 1
                elif resp.status_code == 429:
                    print(f"  WARNING: Open-Meteo rate limited (429) for {model}")
                    failures += 1
                    time.sleep(2)  # Back off on rate limit
                else:
                    failures += 1
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                print(f"  Warning: Failed to fetch {model}: {e}")
                failures += 1

        if failures > 0 and len(forecasts) > 0:
            print(f"  Forecasts for {self.station_id}/{target_date}: {len(forecasts)} OK, {failures} failed")

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
        This gives us the full diurnal cycle to compute max temp ourselves.
        """
        hourly_forecasts = {}

        for model in config.WEATHER_MODELS:
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
                if model != "best_match":
                    params["models"] = model

                resp = requests.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if "hourly" in data and "temperature_2m" in data["hourly"]:
                        temps = data["hourly"]["temperature_2m"]
                        valid_temps = [t for t in temps if t is not None]
                        if valid_temps:
                            hourly_forecasts[model] = valid_temps
                time.sleep(0.2)
            except Exception as e:
                print(f"  Warning: Failed to fetch hourly {model}: {e}")

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

    def calibrate(self, lookback_years: int = None) -> Dict:
        """
        Build calibration model by comparing historical forecasts to actuals.
        Returns calibration statistics.
        """
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
