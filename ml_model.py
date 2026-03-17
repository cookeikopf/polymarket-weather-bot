"""
ML Weather Prediction Model
=============================
Gradient Boosted Trees trained on WU resolution data + Open-Meteo forecasts.

KEY INSIGHT: Polymarket resolves using WU data. Open-Meteo forecasts have 
systematic biases vs WU. This ML model learns those biases per station,
season, and weather pattern — giving us a systematic edge.

Features: Open-Meteo ensemble forecasts, station metadata, seasonality
Target: WU daily high temperature (the actual resolution value)
Output: Predicted WU high temp → compute bucket probabilities
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_LGBM = False

import config


class WeatherMLModel:
    """ML model that predicts WU resolution temperature from Open-Meteo forecasts."""

    def __init__(self):
        self.model = None
        self.feature_cols = []
        self.station_biases = {}  # station_id -> {mean_bias, std}
        self.is_trained = False

    # ═══════════════════════════════════════════════════════════════
    # Feature Engineering
    # ═══════════════════════════════════════════════════════════════

    def build_features(self, verbose: bool = True) -> pd.DataFrame:
        """
        Build feature matrix from WU historical data + Open-Meteo.
        Returns DataFrame with features and WU high temp as target.
        """
        all_features = []

        # Load WU comparison biases
        bias_path = os.path.join("data", "wu_comparison.json")
        if os.path.exists(bias_path):
            with open(bias_path) as f:
                comparison = json.load(f)
        else:
            comparison = {}

        for station_id in config.WU_STATIONS.keys():
            if verbose:
                print(f"  Building features for {station_id}...", end="", flush=True)

            # Load WU data
            wu_path = os.path.join(config.WU_DATA_DIR, f"{station_id.replace(' ', '_')}.csv")
            if not os.path.exists(wu_path):
                print(" no WU data")
                continue

            wu_df = pd.read_csv(wu_path)
            wu_df["date"] = pd.to_datetime(wu_df["date"])

            # Fetch Open-Meteo historical data for same period
            station = config.STATIONS[station_id]
            start = wu_df["date"].min().strftime("%Y-%m-%d")
            end = wu_df["date"].max().strftime("%Y-%m-%d")

            try:
                om_df = self._fetch_openmeteo_features(station_id, start, end)
            except Exception as e:
                print(f" OM fetch failed: {e}")
                continue

            if om_df.empty:
                print(" no OM data")
                continue

            # Merge WU and Open-Meteo
            merged = wu_df[["date", "high_temp", "low_temp", "mean_temp", "temp_range",
                            "mean_rh", "mean_pressure", "mean_wspd", "total_precip"]].merge(
                om_df, on="date", how="inner"
            )

            if len(merged) < 60:
                print(f" too few merged days ({len(merged)})")
                continue

            # Add station-level features
            merged["station_lat"] = station["lat"]
            merged["station_lon"] = station["lon"]
            merged["is_fahrenheit"] = 1 if station.get("unit") == "fahrenheit" else 0

            # Add WU bias features
            comp = comparison.get(station_id, {})
            merged["wu_bias"] = comp.get("mean_bias", 0)
            merged["wu_bias_std"] = comp.get("std_diff", 1)

            # Cyclical time features
            doy = merged["date"].dt.dayofyear
            merged["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
            merged["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
            merged["month"] = merged["date"].dt.month

            # Temperature trend features
            merged["om_temp_lag1"] = merged["om_high_temp"].shift(1)
            merged["om_temp_lag3_mean"] = merged["om_high_temp"].rolling(3, min_periods=1).mean()
            merged["om_temp_lag7_mean"] = merged["om_high_temp"].rolling(7, min_periods=1).mean()
            merged["om_temp_change"] = merged["om_high_temp"] - merged["om_high_temp"].shift(1)

            # Climatological anomaly
            monthly_mean = merged.groupby("month")["om_high_temp"].transform("mean")
            merged["temp_anomaly"] = merged["om_high_temp"] - monthly_mean

            # Station ID as category (for lightgbm)
            merged["station_code"] = list(config.WU_STATIONS.keys()).index(station_id)

            # Target: WU high temp
            merged["target_wu_high"] = merged["high_temp"]

            # Drop NaN rows
            merged = merged.dropna(subset=["target_wu_high", "om_high_temp"])

            all_features.append(merged)
            print(f" {len(merged)} samples")

        if not all_features:
            return pd.DataFrame()

        df = pd.concat(all_features, ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)

        # Save feature matrix
        os.makedirs("data", exist_ok=True)
        df.to_csv(config.ML_FEATURES_PATH, index=False)
        if verbose:
            print(f"\n  Total feature matrix: {len(df)} samples, {len(df.columns)} columns")

        return df

    def _fetch_openmeteo_features(self, station_id: str, start: str, end: str) -> pd.DataFrame:
        """Fetch Open-Meteo historical data as features."""
        station = config.STATIONS[station_id]
        unit = station.get("unit", "fahrenheit")

        params = {
            "latitude": station["lat"],
            "longitude": station["lon"],
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                     "precipitation_sum,windspeed_10m_max,relative_humidity_2m_mean,"
                     "pressure_msl_mean",
            "timezone": station.get("tz", "UTC"),
            "start_date": start,
            "end_date": end,
            "temperature_unit": unit,
        }

        resp = requests.get(config.OPEN_METEO_HISTORICAL_URL, params=params, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        if "daily" not in data:
            return pd.DataFrame()

        daily = data["daily"]
        df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})

        # Map available fields
        field_map = {
            "temperature_2m_max": "om_high_temp",
            "temperature_2m_min": "om_low_temp",
            "temperature_2m_mean": "om_mean_temp",
            "precipitation_sum": "om_precip",
            "windspeed_10m_max": "om_wind_max",
            "relative_humidity_2m_mean": "om_rh_mean",
            "pressure_msl_mean": "om_pressure_mean",
        }

        for src, dst in field_map.items():
            if src in daily:
                df[dst] = daily[src]

        time.sleep(0.2)
        return df.dropna(subset=["om_high_temp"])

    # ═══════════════════════════════════════════════════════════════
    # Model Training
    # ═══════════════════════════════════════════════════════════════

    def train(self, df: pd.DataFrame = None, verbose: bool = True) -> Dict:
        """Train the ML model with walk-forward validation."""
        if df is None:
            if os.path.exists(config.ML_FEATURES_PATH):
                df = pd.read_csv(config.ML_FEATURES_PATH)
                df["date"] = pd.to_datetime(df["date"])
            else:
                df = self.build_features(verbose)

        if df.empty:
            print("No training data available!")
            return {}

        # Define feature columns
        self.feature_cols = [
            "om_high_temp", "om_low_temp", "om_mean_temp", "om_precip",
            "om_wind_max", "om_rh_mean", "om_pressure_mean",
            "station_lat", "station_lon", "is_fahrenheit",
            "wu_bias", "wu_bias_std",
            "sin_doy", "cos_doy", "month",
            "om_temp_lag1", "om_temp_lag3_mean", "om_temp_lag7_mean",
            "om_temp_change", "temp_anomaly", "station_code",
        ]

        # Filter to available columns
        available = [c for c in self.feature_cols if c in df.columns]
        self.feature_cols = available

        # Drop rows with NaN in features or target
        train_df = df.dropna(subset=self.feature_cols + ["target_wu_high"])

        X = train_df[self.feature_cols].values
        y = train_df["target_wu_high"].values

        if verbose:
            print(f"\n  Training ML Model")
            print(f"  Features: {len(self.feature_cols)}")
            print(f"  Samples: {len(X)}")
            print(f"  Using: {'LightGBM' if HAS_LGBM else 'sklearn GBR'}")

        # Walk-forward cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_maes = []
        cv_rmses = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = self._create_model()
            if HAS_LGBM:
                model.fit(X_train, y_train,
                          eval_set=[(X_test, y_test)],
                          callbacks=[lgb.log_evaluation(0)])
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            cv_maes.append(mae)
            cv_rmses.append(rmse)

            if verbose:
                print(f"  Fold {fold+1}: MAE={mae:.3f}, RMSE={rmse:.3f}")

        # Train final model on all data
        self.model = self._create_model()
        if HAS_LGBM:
            self.model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
        else:
            self.model.fit(X, y)

        self.is_trained = True

        # Store station biases for runtime use
        for station_id in config.WU_STATIONS.keys():
            station_data = train_df[train_df["station_code"] == list(config.WU_STATIONS.keys()).index(station_id)]
            if not station_data.empty:
                residuals = station_data["target_wu_high"] - station_data["om_high_temp"]
                self.station_biases[station_id] = {
                    "mean_bias": float(residuals.mean()),
                    "std": float(residuals.std()),
                }

        results = {
            "cv_mae_mean": float(np.mean(cv_maes)),
            "cv_mae_std": float(np.std(cv_maes)),
            "cv_rmse_mean": float(np.mean(cv_rmses)),
            "cv_rmse_std": float(np.std(cv_rmses)),
            "n_features": len(self.feature_cols),
            "n_samples": len(X),
            "feature_importance": self._get_feature_importance(),
        }

        if verbose:
            print(f"\n  CV Results: MAE={results['cv_mae_mean']:.3f} ± {results['cv_mae_std']:.3f}")
            print(f"             RMSE={results['cv_rmse_mean']:.3f} ± {results['cv_rmse_std']:.3f}")
            print(f"\n  Top features:")
            imp = results["feature_importance"]
            for feat, score in sorted(imp.items(), key=lambda x: -x[1])[:10]:
                print(f"    {feat:25s}: {score:.4f}")

        return results

    def _create_model(self):
        """Create a fresh model instance."""
        if HAS_LGBM:
            return lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=-1,
            )
        else:
            return GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                min_samples_leaf=20,
                subsample=0.8,
                random_state=42,
            )

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        total = importances.sum()
        if total == 0:
            return {}
        return {
            feat: float(imp / total)
            for feat, imp in zip(self.feature_cols, importances)
        }

    # ═══════════════════════════════════════════════════════════════
    # Prediction
    # ═══════════════════════════════════════════════════════════════

    def predict_wu_temp(self, features: Dict) -> Tuple[float, float]:
        """
        Predict WU high temperature from features.
        Returns (predicted_temp, prediction_std).
        """
        if not self.is_trained or self.model is None:
            # Fallback: simple bias correction
            om_temp = features.get("om_high_temp", 0)
            bias = features.get("wu_bias", 0)
            return om_temp + bias, features.get("wu_bias_std", 2.0)

        # Build feature vector
        x = np.array([[features.get(col, 0) for col in self.feature_cols]])
        pred = self.model.predict(x)[0]

        # Estimate prediction uncertainty from station bias
        station_id = features.get("station_id")
        bias_info = self.station_biases.get(station_id, {"std": 2.0})
        pred_std = bias_info.get("std", 2.0)

        return float(pred), float(pred_std)

    def predict_bucket_probs(
        self,
        features: Dict,
        bucket_edges: List[float],
        is_fahrenheit: bool = True,
        n_samples: int = 5000,
    ) -> Dict[str, float]:
        """
        Predict probability distribution over temperature buckets.
        Uses ML prediction + uncertainty to generate samples.
        """
        pred_temp, pred_std = self.predict_wu_temp(features)

        # Generate samples from predicted distribution
        # Use t-distribution for heavier tails
        from scipy import stats
        samples = pred_temp + stats.t.rvs(df=5, loc=0, scale=pred_std, size=n_samples)

        # Compute bucket probabilities
        bucket_probs = {}
        unit_str = "°F" if is_fahrenheit else "°C"

        for i in range(len(bucket_edges) - 1):
            low = bucket_edges[i]
            high = bucket_edges[i + 1]
            mask = (samples >= low) & (samples < high)
            prob = mask.mean()
            if is_fahrenheit:
                label = f"{int(low)}-{int(high-1)}{unit_str}"
            else:
                label = f"{int(low)}{unit_str}"
            bucket_probs[label] = max(prob, 0.001)

        # Tail buckets
        low_tail = (samples < bucket_edges[0]).mean()
        tail_val = int(bucket_edges[0]) - 1
        if low_tail > 0:
            label = f"{tail_val}{unit_str} or below"
            bucket_probs[label] = low_tail

        high_tail = (samples >= bucket_edges[-1]).mean()
        if high_tail > 0:
            label = f"{int(bucket_edges[-1])}{unit_str} or higher"
            bucket_probs[label] = high_tail

        # Normalize
        total = sum(bucket_probs.values())
        if total > 0:
            bucket_probs = {k: v / total for k, v in bucket_probs.items()}

        return bucket_probs

    # ═══════════════════════════════════════════════════════════════
    # Persistence
    # ═══════════════════════════════════════════════════════════════

    def save(self, path: str = None):
        """Save trained model to disk."""
        if path is None:
            path = config.ML_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "station_biases": self.station_biases,
            "is_trained": self.is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Model saved to {path}")

    def load(self, path: str = None):
        """Load trained model from disk."""
        if path is None:
            path = config.ML_MODEL_PATH
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        self.station_biases = data["station_biases"]
        self.is_trained = data["is_trained"]
        print(f"  Model loaded from {path}")


def main():
    """Build features, train model, save."""
    print(f"\n{'='*60}")
    print(f"  ML Weather Prediction Model")
    print(f"{'='*60}")

    model = WeatherMLModel()

    print("\nStep 1: Building features...")
    df = model.build_features(verbose=True)

    if df.empty:
        print("No features built!")
        return

    print("\nStep 2: Training model...")
    results = model.train(df, verbose=True)

    print("\nStep 3: Saving model...")
    model.save()

    # Save training results
    results_path = os.path.join("results", "ml_training_results.json")
    os.makedirs("results", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Training results saved to {results_path}")

    return model, results


if __name__ == "__main__":
    main()
