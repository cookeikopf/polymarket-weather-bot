"""
Weather Prediction Engine V7.1
================================
Multi-model ensemble weather forecasting via Open-Meteo APIs.
Supports: deterministic forecasts, ensemble members, historical forecasts, archive data.

V7.1 FIX: Balanced ensemble vs deterministic weighting in compute_bucket_probabilities().
"""

import time
import numpy as np
import requests
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import config as cfg
from utils import log


class WeatherEngine:
    """Fetches weather data and computes probability distributions for temperature buckets."""

    def __init__(self, station_id: str):
        self.station_id = station_id
        self.station = cfg.STATIONS[station_id]
        self.lat = self.station["lat"]
        self.lon = self.station["lon"]
        self.unit = self.station.get("unit", "fahrenheit")

    # ─── API helpers ──────────────────────────────────────────────

    def _api_params(self) -> dict:
        """Common API parameters."""
        p = {"latitude": self.lat, "longitude": self.lon}
        if cfg.OPEN_METEO_API_KEY:
            p["apikey"] = cfg.OPEN_METEO_API_KEY
        return p

    def _get_json(self, url: str, params: dict, retries: int = 3) -> Optional[dict]:
        """GET with exponential backoff."""
        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = min(60, 5 * (2 ** attempt))
                    log.warning(f"  Rate limited (429), waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 200:
                    return resp.json()
                log.debug(f"  API {resp.status_code}: {url}")
                return None
            except Exception as e:
                log.debug(f"  API error: {e}")
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
        return None

    # ─── Deterministic multi-model forecasts ──────────────────────

    def fetch_forecasts(self, target_date: str) -> Dict[str, float]:
        """Fetch max temperature forecasts from multiple NWP models.
        Returns {model_name: max_temp_value}.
        """
        temp_var = "temperature_2m_max"
        temp_unit = "fahrenheit" if self.unit == "fahrenheit" else "celsius"

        params = {
            **self._api_params(),
            "daily": temp_var,
            "temperature_unit": temp_unit,
            "timezone": "auto",
            "start_date": target_date,
            "end_date": target_date,
            "models": ",".join(cfg.WEATHER_MODELS),
        }

        data = self._get_json(cfg.FORECAST_URL, params)
        if not data:
            return {}

        results = {}
        # Try combined daily response first
        daily = data.get("daily", {})
        # The API returns separate keys per model when multiple models requested
        for model in cfg.WEATHER_MODELS:
            key = f"{temp_var}_{model}" if model != "best_match" else temp_var
            vals = daily.get(key, [])
            if not vals:
                # Try model-specific key
                key = f"{temp_var}_{model}"
                vals = daily.get(key, [])
            if vals and vals[0] is not None:
                results[model] = float(vals[0])

        # If combined call failed, try individual model calls
        if len(results) < 3:
            for model in cfg.WEATHER_MODELS:
                if model in results:
                    continue
                p = {
                    **self._api_params(),
                    "daily": temp_var,
                    "temperature_unit": temp_unit,
                    "timezone": "auto",
                    "start_date": target_date,
                    "end_date": target_date,
                    "models": model,
                }
                d = self._get_json(cfg.FORECAST_URL, p)
                if d:
                    vals = d.get("daily", {}).get(temp_var, [])
                    if vals and vals[0] is not None:
                        results[model] = float(vals[0])
                time.sleep(0.3)

        return results

    # ─── Ensemble forecasts (probabilistic) ───────────────────────

    def fetch_ensemble(self, target_date: str) -> Dict[str, List[float]]:
        """Fetch ensemble member forecasts.
        Returns {model_name: [member1_temp, member2_temp, ...]}.
        """
        temp_var = "temperature_2m_max"
        temp_unit = "fahrenheit" if self.unit == "fahrenheit" else "celsius"

        results = {}
        for model in cfg.ENSEMBLE_MODELS:
            params = {
                **self._api_params(),
                "daily": temp_var,
                "temperature_unit": temp_unit,
                "timezone": "auto",
                "start_date": target_date,
                "end_date": target_date,
                "models": model,
            }
            data = self._get_json(cfg.ENSEMBLE_URL, params)
            if not data:
                time.sleep(0.5)
                continue

            daily = data.get("daily", {})
            members = []
            # Ensemble API returns temperature_2m_max_member01, etc.
            for key, vals in daily.items():
                if key.startswith(f"{temp_var}_member") and vals and vals[0] is not None:
                    members.append(float(vals[0]))

            if members:
                results[model] = members
            time.sleep(0.5)

        return results

    # ─── Historical forecast (what models predicted in the past) ──

    def fetch_historical_forecast(self, target_date: str, forecast_date: str = None) -> Dict[str, float]:
        """Fetch what models predicted for target_date as of forecast_date.
        If forecast_date is None, uses 1 day before target_date.
        """
        if forecast_date is None:
            td = datetime.strptime(target_date, "%Y-%m-%d")
            forecast_date = (td - timedelta(days=1)).strftime("%Y-%m-%d")

        temp_var = "temperature_2m_max"
        temp_unit = "fahrenheit" if self.unit == "fahrenheit" else "celsius"

        hist_models = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "best_match"]

        params = {
            **self._api_params(),
            "daily": temp_var,
            "temperature_unit": temp_unit,
            "timezone": "auto",
            "start_date": forecast_date,
            "end_date": target_date,
            "models": ",".join(hist_models),
        }

        data = self._get_json(cfg.HIST_FORECAST_URL, params)
        if not data:
            return {}

        results = {}
        daily = data.get("daily", {})
        times = daily.get("time", [])

        # Find the index for our target date
        target_idx = None
        for i, t in enumerate(times):
            if t == target_date:
                target_idx = i
                break

        if target_idx is None:
            # Fallback: take last available value
            target_idx = len(times) - 1 if times else None

        if target_idx is None:
            return {}

        for model in hist_models:
            key = f"{temp_var}_{model}"
            vals = daily.get(key, [])
            if target_idx < len(vals) and vals[target_idx] is not None:
                results[model] = float(vals[target_idx])

        return results

    # ─── Archive (actual observed temperature) ────────────────────

    def fetch_actual_temp(self, date: str) -> Optional[float]:
        """Fetch actual observed max temperature for a past date."""
        temp_var = "temperature_2m_max"
        temp_unit = "fahrenheit" if self.unit == "fahrenheit" else "celsius"

        params = {
            **self._api_params(),
            "daily": temp_var,
            "temperature_unit": temp_unit,
            "timezone": "auto",
            "start_date": date,
            "end_date": date,
        }

        data = self._get_json(cfg.ARCHIVE_URL, params)
        if not data:
            return None

        vals = data.get("daily", {}).get(temp_var, [])
        if vals and vals[0] is not None:
            return float(vals[0])
        return None

    # ─── Probability distribution computation ─────────────────────

    def compute_ensemble_stats(self, forecasts: Dict[str, float],
                                ensemble_data: Dict[str, List[float]] = None) -> dict:
        """Compute ensemble statistics from forecast data."""
        if not forecasts:
            return {"mean": 0, "median": 0, "std": 5, "spread": 10,
                    "agreement": 0.5, "n_models": 0, "per_model_medians": {}}

        temps = list(forecasts.values())
        mean = np.mean(temps)
        median = np.median(temps)
        std = np.std(temps) if len(temps) > 1 else 3.0

        # If ensemble data available, use member stats
        member_temps = []
        per_model_medians = {}
        if ensemble_data:
            for model, members in ensemble_data.items():
                member_temps.extend(members)
                per_model_medians[model] = float(np.median(members))

        if member_temps:
            all_temps = member_temps
            member_std = float(np.std(all_temps))
        else:
            all_temps = temps
            member_std = std

        spread = max(all_temps) - min(all_temps) if all_temps else 10
        agreement = max(0, min(1, 1.0 - spread / 20.0))

        return {
            "mean": float(mean),
            "median": float(np.median(all_temps)),
            "std": float(std),
            "member_std": float(member_std),
            "spread": float(spread),
            "agreement": float(agreement),
            "n_models": len(forecasts),
            "per_model_medians": per_model_medians or {m: v for m, v in forecasts.items()},
        }

    def compute_bucket_probabilities(self, forecasts: Dict[str, float],
                                      ensemble_data: Dict[str, List[float]],
                                      bucket_edges: List[float],
                                      is_fahrenheit: bool) -> Dict[str, float]:
        """Compute probability for each temperature bucket using ensemble members + KDE.

        V7.1 FIX: Balanced weighting between ensemble and deterministic models.
        Previously ensemble members (3 models x ~50 members = ~150 samples at weight 1.0)
        overwhelmed deterministic models (8 models at weight ~0.04 each), giving ensemble
        ~99.8% influence. Ensemble models show systematic cold bias, causing the bot to
        bet on buckets that are too low.

        New approach: Each ensemble MODEL gets total weight = its model_weight, spread
        evenly across its members. Deterministic models keep their full model_weight.
        Result: ~40% ensemble, ~60% deterministic — both contribute meaningfully.

        Returns {bucket_label: probability}.
        """
        # Collect all temperature samples with balanced weights
        samples = []
        weights = []

        # Ensemble members: spread each model's weight across its members
        if ensemble_data:
            for model, members in ensemble_data.items():
                model_w = cfg.MODEL_WEIGHTS.get(model, 0.10)
                per_member_w = model_w / len(members) if members else 0
                for m in members:
                    samples.append(m)
                    weights.append(per_member_w)

        # Deterministic forecasts: full model weight each
        for model, temp in forecasts.items():
            samples.append(temp)
            w = cfg.MODEL_WEIGHTS.get(model, 0.10)
            weights.append(w)

        if not samples:
            return {}

        samples = np.array(samples)
        weights = np.array(weights)
        weights /= weights.sum()

        # Build probability distribution using weighted KDE
        try:
            # Use Scott's rule for bandwidth, adjusted by number of samples
            bw = max(0.5, 1.06 * np.std(samples) * len(samples) ** (-0.2))
            kde = stats.gaussian_kde(samples, bw_method=bw / np.std(samples) if np.std(samples) > 0.1 else 0.5,
                                     weights=weights)
        except Exception:
            # Fallback: simple histogram approach
            return self._histogram_probabilities(samples, bucket_edges, is_fahrenheit)

        # Integrate KDE over each bucket
        probs = {}
        step = 2 if is_fahrenheit else 1
        unit = "°F" if is_fahrenheit else "°C"

        sorted_edges = sorted(set(bucket_edges))

        for i in range(len(sorted_edges) - 1):
            low = sorted_edges[i]
            high = sorted_edges[i + 1]

            # Integrate using sampling
            x = np.linspace(low, high, 50)
            density = kde(x)
            prob = float(np.trapz(density, x))

            if is_fahrenheit:
                label = f"{int(low)}-{int(high - 1)}°F"
            else:
                label = f"{int(low)}°C"

            probs[label] = max(0, prob)

        # Add tail probabilities
        if sorted_edges:
            # Lower tail
            x_low = np.linspace(sorted_edges[0] - 20, sorted_edges[0], 50)
            p_low = float(np.trapz(kde(x_low), x_low))
            probs[f"{int(sorted_edges[0])}{unit} or below"] = max(0, p_low)

            # Upper tail
            x_high = np.linspace(sorted_edges[-1], sorted_edges[-1] + 20, 50)
            p_high = float(np.trapz(kde(x_high), x_high))
            probs[f"{int(sorted_edges[-1])}{unit} or higher"] = max(0, p_high)

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def _histogram_probabilities(self, samples, bucket_edges, is_fahrenheit):
        """Fallback: simple histogram-based probabilities."""
        probs = {}
        unit = "°F" if is_fahrenheit else "°C"
        sorted_edges = sorted(set(bucket_edges))
        n = len(samples)

        for i in range(len(sorted_edges) - 1):
            low = sorted_edges[i]
            high = sorted_edges[i + 1]
            count = np.sum((samples >= low) & (samples < high))
            if is_fahrenheit:
                label = f"{int(low)}-{int(high - 1)}°F"
            else:
                label = f"{int(low)}°C"
            probs[label] = count / n if n > 0 else 0

        if sorted_edges:
            below = np.sum(samples < sorted_edges[0])
            probs[f"{int(sorted_edges[0])}{unit} or below"] = below / n if n > 0 else 0
            above = np.sum(samples >= sorted_edges[-1])
            probs[f"{int(sorted_edges[-1])}{unit} or higher"] = above / n if n > 0 else 0

        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        return probs
