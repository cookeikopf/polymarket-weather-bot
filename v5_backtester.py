#!/usr/bin/env python3
"""
V5 Advanced Backtester for Polymarket Weather Bot
===================================================
Uses REAL historical data from Open-Meteo APIs with ALL 7 V5 innovations:

  1. Bias-Corrected Ensemble
  2. Ensemble Spread Confidence
  3. Precipitation-Aware Temperature Adjustment
  4. Inter-Model Disagreement Detection
  5. Time-Decay Edge Optimization
  6. Adaptive Market Efficiency Scoring
  7. Dynamic Ladder Width + Bimodal Detection

Usage:
    python v5_backtester.py
"""

import os
import sys
import json
import time
import hashlib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats as sp_stats

# ═══════════════════════════════════════════════════════════════════
# API KEY — must be set before importing config
# ═══════════════════════════════════════════════════════════════════
os.environ["OPEN_METEO_API_KEY"] = "wjrcKzLOeLkcCnzx"

# Now import project modules (config reads the env var at import time)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ═══════════════════════════════════════════════════════════════════
# BACKTEST PARAMETERS
# ═══════════════════════════════════════════════════════════════════
STATION_ID = "NYC"
BACKTEST_DAYS = 90
BANKROLL_START = 100.0

# Strategy params — aligned with config.py V5-final
MIN_EDGE = 0.12          # V5-final: 12% minimum edge (was 0.08)
MAX_POSITION_PCT = 0.15  # V5-final: 15% (was 0.10)
KELLY_FRACTION = 0.15    # V5-final: 15% fractional Kelly (was 0.25)
TEMP_BUCKET_SIZE = 2

# Market simulation params — REALISTIC
MARKET_NOISE_STD = 1.5
OVERROUND_MIN = 1.05
OVERROUND_MAX = 1.10
SLIPPAGE = 0.01
SPREAD_COST = 0.02

# Ladder strategy params
LADDER_MAX_PRICE = 0.20         # V5-final: 20 cents max (was 0.18)
LADDER_BET_PER_BUCKET = 2.0     # V5-final: $2 per bucket
MAX_LADDER_SETS_PER_DAY = 1     # V5-final: 1 set per cycle
LADDER_MIN_PRICE = 0.02

# Conservative NO params
NO_MIN_ENTRY = 0.55     # Lowered to match config
NO_MAX_ENTRY = 0.85
MAX_NO_TRADES_PER_DAY = 3

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "backtest_cache")

# ═══════════════════════════════════════════════════════════════════
# STATION CONFIG
# ═══════════════════════════════════════════════════════════════════
STATION = config.STATIONS[STATION_ID]
LAT = STATION["lat"]
LON = STATION["lon"]
UNIT = STATION["unit"]
TZ = STATION["tz"]
IS_FAHRENHEIT = UNIT == "fahrenheit"

# API URLs (customer endpoints since we have API key)
ARCHIVE_URL = config.OPEN_METEO_HISTORICAL_URL
HIST_FORECAST_URL = config.OPEN_METEO_HISTORICAL_FORECAST_URL
API_KEY = os.environ["OPEN_METEO_API_KEY"]

# Models to use for forecasts
FORECAST_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "best_match"]

# Ensemble model mapping (same as weather_engine.py)
ENSEMBLE_MODEL_MAP = {
    "ecmwf_ifs025": "ecmwf_ifs025",
    "gfs_seamless": "gfs025",
    "icon_seamless": "icon_seamless",
}


# ═══════════════════════════════════════════════════════════════════
# CACHING LAYER
# ═══════════════════════════════════════════════════════════════════
def _cache_key(url: str, params: dict) -> str:
    raw = json.dumps({"url": url, **params}, sort_keys=True)
    h = hashlib.md5(raw.encode()).hexdigest()
    return h + ".json"


def cached_get(url: str, params: dict, timeout: int = 30) -> Optional[dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    fname = _cache_key(url, params)
    fpath = os.path.join(CACHE_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            return json.load(f)
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429:
            print(f"  RATE LIMITED (429) — waiting 60s...")
            time.sleep(60)
            resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            with open(fpath, "w") as f:
                json.dump(data, f)
            return data
        else:
            print(f"  API error {resp.status_code}: {url}")
            return None
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════
def fetch_actuals(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"  Fetching actuals {start_date} to {end_date}...")
    params = {
        "latitude": LAT, "longitude": LON,
        "daily": "temperature_2m_max",
        "timezone": TZ,
        "start_date": start_date, "end_date": end_date,
        "temperature_unit": UNIT,
        "apikey": API_KEY,
    }
    data = cached_get(ARCHIVE_URL, params)
    if not data or "daily" not in data:
        return pd.DataFrame()
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "actual": data["daily"]["temperature_2m_max"],
    }).dropna()
    print(f"    Got {len(df)} days of actuals")
    return df


def fetch_model_forecasts(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"  Fetching historical forecasts {start_date} to {end_date}...")
    all_dfs = []
    for model in FORECAST_MODELS:
        params = {
            "latitude": LAT, "longitude": LON,
            "daily": "temperature_2m_max",
            "timezone": TZ,
            "start_date": start_date, "end_date": end_date,
            "temperature_unit": UNIT,
            "apikey": API_KEY,
        }
        if model != "best_match":
            params["models"] = model
        data = cached_get(HIST_FORECAST_URL, params)
        if not data or "daily" not in data:
            print(f"    WARNING: No data for model {model}")
            continue
        daily = data["daily"]
        var_key = f"temperature_2m_max_{model}" if model != "best_match" else "temperature_2m_max"
        if var_key not in daily:
            var_key = "temperature_2m_max"
        if var_key not in daily:
            continue
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            f"forecast_{model}": daily[var_key],
        })
        all_dfs.append(df)
        print(f"    {model}: {len(df)} days")
        time.sleep(0.3)
    if not all_dfs:
        return pd.DataFrame()
    result = all_dfs[0]
    for df in all_dfs[1:]:
        result = result.merge(df, on="date", how="outer")
    return result


# ═══════════════════════════════════════════════════════════════════
# BUCKET HELPERS
# ═══════════════════════════════════════════════════════════════════
def make_bucket_edges(center_temp: float) -> List[float]:
    step = TEMP_BUCKET_SIZE
    center = round(center_temp / step) * step
    start = center - 14
    end = center + 16
    return list(range(int(start), int(end) + 1, step))


def make_bucket_labels(edges: List[float]) -> List[str]:
    labels = []
    low_tail_val = int(edges[0]) - 1
    labels.append(f"{low_tail_val}°F or below")
    for i in range(len(edges) - 1):
        low = int(edges[i])
        high = int(edges[i + 1]) - 1
        labels.append(f"{low}-{high}°F")
    labels.append(f"{int(edges[-1])}°F or higher")
    return labels


def bucket_center(label: str) -> Optional[float]:
    import re
    m = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)', label)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2.0
    m = re.search(r'(-?\d+).*or below', label, re.I)
    if m:
        return float(m.group(1)) - 0.5
    m = re.search(r'(-?\d+).*or higher', label, re.I)
    if m:
        return float(m.group(1)) + 0.5
    return None


def actual_to_winning_bucket(actual: float, edges: List[float], labels: List[str]) -> str:
    if actual < edges[0]:
        return labels[0]
    if actual >= edges[-1]:
        return labels[-1]
    for i in range(len(edges) - 1):
        if edges[i] <= actual < edges[i + 1]:
            return labels[i + 1]
    return labels[-1]


# ═══════════════════════════════════════════════════════════════════
# INNOVATION 1: BIAS-CORRECTED ENSEMBLE PROBABILITIES
# ═══════════════════════════════════════════════════════════════════
def load_calibration_data() -> Dict[str, Dict]:
    """Load per-model calibration from v5_calibration_NYC.json."""
    cal_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data",
        f"v5_calibration_{STATION_ID}.json"
    )
    try:
        with open(cal_path) as f:
            data = json.load(f)
        # Support both flat format and nested {"models": {...}} format
        return data.get("models", data) if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def compute_ensemble_probabilities(
    model_forecasts: Dict[str, float],
    model_errors: Dict[str, Dict],
    edges: List[float],
    labels: List[str],
    calibration: Dict[str, Dict] = None,
    precip_adjustment: float = 0.0,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    V5 Advanced ensemble probability estimation.
    Innovation 1: Apply bias correction — use in-sample errors as primary,
    calibration file as fallback for models not in in-sample data.
    Innovation 3: Apply precipitation temperature adjustment.
    Returns (probabilities dict, all_samples array for later analysis).
    """
    n_samples_per_model = 2500
    all_samples = []

    for model, forecast in model_forecasts.items():
        err = model_errors.get(model, {"bias": 0.0, "std": 3.5})
        bias = err.get("bias", err.get("mean_bias", 0.0))
        std = err.get("std", 3.5)

        # Innovation 1: If in-sample bias is near zero (unreliable), use calibration file
        if calibration and abs(bias) < 0.05 and err.get("n_samples", 0) < 20:
            cal_model = ENSEMBLE_MODEL_MAP.get(model, model)
            cal = calibration.get(cal_model, calibration.get(model, {}))
            cal_bias = cal.get("bias", 0.0)
            if cal_bias != 0.0:
                bias = cal_bias

        # bias = actual - forecast, so corrected = forecast + bias moves toward actual
        corrected = forecast + bias
        samples = corrected + sp_stats.t.rvs(df=5, loc=0, scale=std, size=n_samples_per_model)
        all_samples.extend(samples)

    all_samples = np.array(all_samples)

    # Innovation 3: Apply precipitation adjustment
    if precip_adjustment != 0.0:
        all_samples = all_samples + precip_adjustment

    total = len(all_samples)
    n_buckets = len(labels)
    smoothing = 0.5

    probs = {}
    count = np.sum(all_samples < edges[0])
    probs[labels[0]] = (count + smoothing) / (total + smoothing * n_buckets)

    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        count = np.sum((all_samples >= low) & (all_samples < high))
        probs[labels[i + 1]] = (count + smoothing) / (total + smoothing * n_buckets)

    count = np.sum(all_samples >= edges[-1])
    probs[labels[-1]] = (count + smoothing) / (total + smoothing * n_buckets)

    total_p = sum(probs.values())
    if total_p > 0:
        probs = {k: v / total_p for k, v in probs.items()}
    return probs, all_samples


# ═══════════════════════════════════════════════════════════════════
# INNOVATION 2+4+7: ENHANCED ENSEMBLE STATISTICS
# ═══════════════════════════════════════════════════════════════════
def compute_advanced_ensemble_stats(
    day_forecasts: Dict[str, float],
    all_samples: np.ndarray,
) -> Dict:
    """Compute comprehensive ensemble statistics for all innovations."""
    values = list(day_forecasts.values())
    ensemble_mean = np.mean(values)
    ensemble_std = np.std(values) if len(values) > 1 else 2.0
    model_range = max(values) - min(values)

    # Per-model medians (Innovation 4)
    per_model_medians = {}
    for model, val in day_forecasts.items():
        per_model_medians[model] = val

    # Member-level median from MC samples (for bucket centering)
    member_median = float(np.median(all_samples)) if len(all_samples) > 0 else ensemble_mean

    # Innovation 2: Use model-level spread for confidence scoring
    # (NOT sample-level std, which is dominated by sampling noise)
    # The model_range captures how much models disagree — this is the true uncertainty signal
    member_std = float(ensemble_std) if ensemble_std > 0.1 else 2.0

    # Skewness
    skewness = 0.0
    if len(all_samples) > 3:
        skewness = float(sp_stats.skew(all_samples))

    # Innovation 7: Bimodal detection
    is_bimodal = False
    bimodal_peaks = []
    if len(all_samples) > 20:
        is_bimodal, bimodal_peaks = detect_bimodality(all_samples)

    agreement = max(0, 1.0 - (model_range / 10.0))

    return {
        "mean": float(ensemble_mean),
        "median": float(member_median),
        "std": float(ensemble_std),
        "range": float(model_range),
        "spread": float(model_range),
        "agreement": float(agreement),
        "agreement_score": float(agreement),
        "min": float(min(values)),
        "max": float(max(values)),
        "n_models": len(values),
        "skewness": skewness,
        "per_model_medians": per_model_medians,
        "member_std": member_std,
        "member_spread": float(model_range),
        "is_bimodal": is_bimodal,
        "bimodal_peaks": bimodal_peaks,
    }


def detect_bimodality(members: np.ndarray) -> Tuple[bool, List[float]]:
    """Innovation 7: Simple bimodal detection via largest gap analysis."""
    sorted_m = np.sort(members)
    n = len(sorted_m)
    if n < 20:
        return False, []
    gap_threshold = getattr(config, 'BIMODAL_GAP_THRESHOLD_F', 4.0)
    gaps = np.diff(sorted_m)
    max_gap_idx = np.argmax(gaps)
    max_gap = gaps[max_gap_idx]
    if max_gap < gap_threshold:
        return False, []
    cluster1 = sorted_m[:max_gap_idx + 1]
    cluster2 = sorted_m[max_gap_idx + 1:]
    min_pct = 0.15
    if len(cluster1) / n < min_pct or len(cluster2) / n < min_pct:
        return False, []
    return True, [float(np.median(cluster1)), float(np.median(cluster2))]


# ═══════════════════════════════════════════════════════════════════
# INNOVATION 3: PRECIPITATION ADJUSTMENT (simulated)
# ═══════════════════════════════════════════════════════════════════
def simulate_precip_adjustment(day_forecasts: Dict[str, float], date_str: str) -> float:
    """
    Simulate precipitation-based temperature adjustment.
    In live, we'd fetch precip ensemble members. In backtest, we use a
    probabilistic model based on seasonal patterns.
    """
    # Use date to determine precipitation likelihood
    month = int(date_str.split("-")[1])
    # NYC: wetter months are spring/fall, drier in winter
    precip_probs = {
        1: 0.25, 2: 0.25, 3: 0.30, 4: 0.35, 5: 0.35, 6: 0.30,
        7: 0.25, 8: 0.25, 9: 0.30, 10: 0.30, 11: 0.30, 12: 0.30,
    }
    base_prob = precip_probs.get(month, 0.30)

    # Random draw (seeded by date for reproducibility within same seed)
    date_hash = hash(date_str) % 1000 / 1000.0
    if date_hash < base_prob * 0.4:  # ~10-14% chance of heavy rain
        return -getattr(config, 'PRECIP_TEMP_ADJUSTMENT_F', 1.5)
    elif date_hash > (1 - base_prob * 0.3):  # ~7-10% chance of very dry
        return getattr(config, 'PRECIP_DRY_TEMP_ADJUSTMENT_F', 0.5)
    return 0.0


# ═══════════════════════════════════════════════════════════════════
# INNOVATION 5: TIME-DECAY MULTIPLIER
# ═══════════════════════════════════════════════════════════════════
def compute_time_decay(days_to_resolution: float) -> float:
    """Return sizing multiplier based on days to market resolution."""
    if days_to_resolution <= 1:
        return 1.0
    elif days_to_resolution <= 3:
        return getattr(config, 'TIME_DECAY_MED_MULTIPLIER', 0.7)
    else:
        return getattr(config, 'TIME_DECAY_FAR_MULTIPLIER', 0.4)


# ═══════════════════════════════════════════════════════════════════
# INNOVATION 6: MARKET EFFICIENCY SCORING
# ═══════════════════════════════════════════════════════════════════
def score_market_efficiency(market_prices: Dict[str, float]) -> str:
    """Score simulated market as sharp/normal/soft."""
    prices_sum = sum(market_prices.values())
    expected = getattr(config, 'MARKET_EXPECTED_SUM', 1.05)
    deviation = abs(prices_sum - expected)
    if deviation < getattr(config, 'MARKET_SHARP_THRESHOLD', 0.03):
        return "sharp"
    elif deviation > getattr(config, 'MARKET_SOFT_THRESHOLD', 0.10):
        return "soft"
    return "normal"


# ═══════════════════════════════════════════════════════════════════
# MARKET PRICE SIMULATION (Realistic)
# ═══════════════════════════════════════════════════════════════════
def simulate_market_prices(
    best_match_forecast: float,
    all_model_forecasts: Dict[str, float],
    edges: List[float],
    labels: List[str],
) -> Dict[str, float]:
    if all_model_forecasts:
        if np.random.random() < 0.3 and "ecmwf_ifs025" in all_model_forecasts:
            vals = list(all_model_forecasts.values())
            ecmwf = all_model_forecasts["ecmwf_ifs025"]
            market_belief = 0.4 * ecmwf + 0.6 * np.mean(vals)
        else:
            market_belief = np.mean(list(all_model_forecasts.values()))
    else:
        market_belief = best_match_forecast
    market_belief += np.random.normal(0, MARKET_NOISE_STD)

    raw_prices = {}
    for label in labels:
        center = bucket_center(label)
        if center is None:
            raw_prices[label] = 0.01
            continue
        z_sharp = (center - market_belief) / 2.5
        z_broad = (center - market_belief) / 5.0
        raw_prices[label] = 0.5 * np.exp(-0.5 * z_sharp**2) + 0.5 * np.exp(-0.5 * z_broad**2)

    total = sum(raw_prices.values())
    if total <= 0:
        total = 1.0
    prices = {k: v / total for k, v in raw_prices.items()}
    overround = np.random.uniform(OVERROUND_MIN, OVERROUND_MAX)
    prices = {k: v * overround for k, v in prices.items()}
    prices = {k: np.clip(v, 0.01, 0.99) for k, v in prices.items()}
    return prices


# ═══════════════════════════════════════════════════════════════════
# V5 ADVANCED TRADE FINDER (all 7 innovations)
# ═══════════════════════════════════════════════════════════════════
@dataclass
class BacktestTrade:
    date: str
    bucket: str
    direction: str
    strategy: str
    entry_price: float
    size_usd: float
    shares: float
    our_prob: float
    market_price: float
    edge: float
    won: bool
    pnl: float
    actual_temp: float
    winning_bucket: str


def find_trades_v5_advanced(
    our_probs: Dict[str, float],
    market_prices: Dict[str, float],
    ensemble_stats: Dict,
    labels: List[str],
    bankroll: float,
    current_exposure: float,
    days_to_resolution: float = 1.0,
) -> List[dict]:
    """
    V5 Advanced trade finder with all 7 innovations.
    """
    trades = []
    n_models = ensemble_stats.get("n_models", len(FORECAST_MODELS))
    agreement = ensemble_stats.get("agreement", 0.5)
    ensemble_mean = ensemble_stats.get("median", ensemble_stats.get("mean", 0))
    member_std = ensemble_stats.get("member_std", ensemble_stats.get("std", 3.0))

    # ── Innovation 2: Ensemble Spread Confidence ──
    low_std = getattr(config, 'ENSEMBLE_SPREAD_LOW_STD', 2.0)
    high_std = getattr(config, 'ENSEMBLE_SPREAD_HIGH_STD', 4.0)
    if member_std < low_std:
        spread_mult = getattr(config, 'ENSEMBLE_HIGH_CONF_MULTIPLIER', 1.5)
    elif member_std > high_std:
        spread_mult = getattr(config, 'ENSEMBLE_LOW_CONF_MULTIPLIER', 0.5)
    else:
        frac = (member_std - low_std) / (high_std - low_std)
        spread_mult = 1.5 + frac * (0.5 - 1.5)

    # ── Innovation 4: Inter-Model Disagreement ──
    per_model_medians = ensemble_stats.get("per_model_medians", {})
    model_medians = list(per_model_medians.values()) if per_model_medians else []
    model_spread = (max(model_medians) - min(model_medians)) if len(model_medians) >= 2 else 0

    # For backtest with deterministic models, adjust thresholds upward
    # (real ensemble per-model medians have smaller spreads than point forecasts)
    disagree_thresh = getattr(config, 'MODEL_DISAGREEMENT_THRESHOLD_F', 4.0) + 1.0
    agree_thresh = getattr(config, 'MODEL_AGREEMENT_THRESHOLD_F', 2.0) + 0.5
    is_disagreement = model_spread > disagree_thresh
    is_agreement = model_spread < agree_thresh

    if is_disagreement:
        disagree_mult = getattr(config, 'DISAGREEMENT_SIZING_MULTIPLIER', 0.5)
    elif is_agreement:
        disagree_mult = getattr(config, 'AGREEMENT_SIZING_MULTIPLIER', 1.3)
    else:
        disagree_mult = 1.0

    # ── Innovation 5: Time-Decay ──
    time_mult = compute_time_decay(days_to_resolution)

    # ── Innovation 6: Market Efficiency ──
    market_eff = score_market_efficiency(market_prices)
    edge_multiplier = 1.0
    if market_eff == "sharp":
        edge_multiplier = getattr(config, 'MARKET_SHARP_EDGE_MULTIPLIER', 1.5)

    effective_min_edge = MIN_EDGE * edge_multiplier

    # Combined sizing multiplier
    sizing_mult = spread_mult * disagree_mult * time_mult

    # ── Innovation 7: Dynamic Ladder Width ──
    is_bimodal = ensemble_stats.get("is_bimodal", False)
    bimodal_peaks = ensemble_stats.get("bimodal_peaks", [])

    if is_bimodal:
        dynamic_buckets = 0  # use bimodal strategy instead
    elif member_std < getattr(config, 'NARROW_PEAK_STD_F', 2.0):
        dynamic_buckets = 2
    elif member_std > getattr(config, 'WIDE_PEAK_STD_F', 4.0):
        dynamic_buckets = 4
    else:
        dynamic_buckets = 3

    # ══════ STRATEGY 1: LADDER ══════
    if not is_bimodal and dynamic_buckets > 0:
        candidates = []
        for label in labels:
            center = bucket_center(label)
            if center is None:
                continue
            price = market_prices.get(label, 0)
            prob = our_probs.get(label, 0)
            if price < LADDER_MIN_PRICE or price > LADDER_MAX_PRICE:
                continue
            if prob < 0.01:
                continue
            dist = abs(center - ensemble_mean)
            candidates.append({
                "label": label, "center": center, "price": price,
                "prob": prob, "dist": dist,
            })

        candidates.sort(key=lambda c: c["dist"])
        ladder = candidates[:dynamic_buckets]

        for c in ladder[:dynamic_buckets * MAX_LADDER_SETS_PER_DAY]:
            entry = c["price"] + np.random.uniform(0, SLIPPAGE)
            entry = np.clip(entry, 0.01, 0.99)
            size = min(LADDER_BET_PER_BUCKET * sizing_mult, bankroll * MAX_POSITION_PCT)
            remaining = (bankroll * 0.60) - current_exposure
            if remaining < size or size < 1.0:
                continue
            shares = size / entry
            edge = c["prob"] - entry

            trades.append({
                "label": c["label"], "direction": "BUY_YES", "strategy": "ladder",
                "entry_price": entry, "size_usd": size, "shares": shares,
                "our_prob": c["prob"], "market_price": c["price"], "edge": edge,
            })
            current_exposure += size

    # ══════ STRATEGY 1b: BIMODAL (Innovation 7) ══════
    if is_bimodal and len(bimodal_peaks) >= 2:
        peak1, peak2 = bimodal_peaks[0], bimodal_peaks[1]
        valley_center = (peak1 + peak2) / 2.0

        for label in labels:
            center = bucket_center(label)
            if center is None:
                continue
            price = market_prices.get(label, 0)
            prob = our_probs.get(label, 0)

            near_peak1 = abs(center - peak1) <= 2
            near_peak2 = abs(center - peak2) <= 2
            near_valley = abs(center - valley_center) <= 2 and not near_peak1 and not near_peak2

            if (near_peak1 or near_peak2) and LADDER_MIN_PRICE <= price <= LADDER_MAX_PRICE and prob >= 0.01:
                entry = price + np.random.uniform(0, SLIPPAGE)
                entry = np.clip(entry, 0.01, 0.99)
                size = min(LADDER_BET_PER_BUCKET * sizing_mult, bankroll * MAX_POSITION_PCT)
                remaining = (bankroll * 0.60) - current_exposure
                if remaining < size or size < 1.0:
                    continue
                shares = size / entry
                edge = prob - entry

                trades.append({
                    "label": label, "direction": "BUY_YES", "strategy": "bimodal",
                    "entry_price": entry, "size_usd": size, "shares": shares,
                    "our_prob": prob, "market_price": price, "edge": edge,
                })
                current_exposure += size

            elif near_valley and price > 0.05:
                entry_no = 1.0 - price
                prob_no = 1.0 - prob
                edge = prob_no - entry_no
                if edge < MIN_EDGE:
                    continue
                entry_exec = entry_no + np.random.uniform(0, SLIPPAGE)
                entry_exec = np.clip(entry_exec, 0.01, 0.99)
                size = min(LADDER_BET_PER_BUCKET * sizing_mult * 0.5, bankroll * MAX_POSITION_PCT)
                remaining = (bankroll * 0.60) - current_exposure
                if remaining < size or size < 1.0:
                    continue
                shares = size / entry_exec

                trades.append({
                    "label": label, "direction": "BUY_NO", "strategy": "bimodal",
                    "entry_price": entry_exec, "size_usd": size, "shares": shares,
                    "our_prob": prob, "market_price": price, "edge": edge,
                })
                current_exposure += size

    # ══════ STRATEGY 2: CONSERVATIVE NO ══════
    no_count = 0
    for label in labels:
        if no_count >= MAX_NO_TRADES_PER_DAY:
            break
        prob_yes = our_probs.get(label, 0)
        price_yes = market_prices.get(label, 0)
        if price_yes <= 0.005:
            continue

        entry_no = 1.0 - price_yes
        if entry_no < NO_MIN_ENTRY or entry_no > NO_MAX_ENTRY:
            continue

        prob_no = 1.0 - prob_yes
        edge = prob_no - entry_no
        if edge < effective_min_edge:
            continue

        prob_uncertainty = min(0.30, member_std * 0.03)
        edge_significance = min(1.0, edge / max(prob_uncertainty, 0.01))
        confidence = 0.40 * agreement + 0.35 * edge_significance + 0.25 * min(1.0, n_models / 5)
        if confidence < 0.45:
            continue

        b = (1.0 - entry_no) / entry_no if 0 < entry_no < 1 else 0
        if b <= 0:
            continue
        kelly = max(0, (b * prob_no - (1.0 - prob_no)) / b)
        kelly *= KELLY_FRACTION
        kelly = min(kelly, MAX_POSITION_PCT)

        size = kelly * bankroll * confidence * sizing_mult
        size = min(size, bankroll * MAX_POSITION_PCT, 10.0)
        remaining = (bankroll * 0.60) - current_exposure
        if remaining < size or size < 2.0:
            continue

        entry_exec = entry_no + np.random.uniform(0, SLIPPAGE)
        entry_exec = np.clip(entry_exec, 0.01, 0.99)
        shares = size / entry_exec

        trades.append({
            "label": label, "direction": "BUY_NO", "strategy": "conservative_no",
            "entry_price": entry_exec, "size_usd": size, "shares": shares,
            "our_prob": prob_yes, "market_price": price_yes, "edge": edge,
        })
        current_exposure += size
        no_count += 1

    # ══════ STRATEGY 3: DISAGREEMENT NO (Innovation 4) ══════
    if is_disagreement and len(model_medians) >= 2:
        consensus_low = min(model_medians)
        consensus_high = max(model_medians)
        margin = 6.0

        for label in labels:
            prob_yes = our_probs.get(label, 0)
            if prob_yes > 0.05:
                continue
            center = bucket_center(label)
            if center is None:
                continue
            if consensus_low - margin < center < consensus_high + margin:
                continue

            price_yes = market_prices.get(label, 0)
            if price_yes <= 0.005:
                continue

            entry_no = 1.0 - price_yes
            if entry_no < 0.55 or entry_no > 0.90:
                continue

            prob_no = 1.0 - prob_yes
            edge = prob_no - entry_no
            if edge < MIN_EDGE:
                continue

            entry_exec = entry_no + np.random.uniform(0, SLIPPAGE)
            entry_exec = np.clip(entry_exec, 0.01, 0.99)
            size = min(LADDER_BET_PER_BUCKET * 0.5, bankroll * 0.03)
            remaining = (bankroll * 0.60) - current_exposure
            if remaining < size or size < 1.0:
                continue
            shares = size / entry_exec

            trades.append({
                "label": label, "direction": "BUY_NO", "strategy": "disagreement_no",
                "entry_price": entry_exec, "size_usd": size, "shares": shares,
                "our_prob": prob_yes, "market_price": price_yes, "edge": edge,
            })
            current_exposure += size

    # ── Innovation 5: Filter low-edge trades at long horizons ──
    if days_to_resolution >= 4:
        far_min_edge = getattr(config, 'TIME_DECAY_FAR_MIN_EDGE', 0.15)
        trades = [t for t in trades if abs(t["edge"]) >= far_min_edge or t["strategy"] == "bimodal"]

    return trades


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════
def compute_model_errors(
    actuals: pd.DataFrame, forecasts: pd.DataFrame
) -> Dict[str, Dict]:
    merged = actuals.merge(forecasts, on="date", how="inner").dropna()
    if len(merged) < 10:
        return {m: {"bias": 0.0, "std": 3.5, "rmse": 3.5} for m in FORECAST_MODELS}

    errors = {}
    for model in FORECAST_MODELS:
        col = f"forecast_{model}"
        if col not in merged.columns:
            errors[model] = {"bias": 0.0, "std": 3.5, "rmse": 3.5}
            continue
        diff = merged["actual"] - merged[col]
        valid = diff.dropna()
        if len(valid) < 10:
            errors[model] = {"bias": 0.0, "std": 3.5, "rmse": 3.5}
            continue
        errors[model] = {
            "bias": float(valid.mean()),
            "std": float(valid.std()),
            "rmse": float(np.sqrt((valid ** 2).mean())),
            "mae": float(valid.abs().mean()),
            "n_samples": len(valid),
        }
        print(f"    {model}: bias={errors[model]['bias']:.2f}°F, "
              f"RMSE={errors[model]['rmse']:.2f}°F, "
              f"std={errors[model]['std']:.2f}°F")
    return errors


# ═══════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════════
def run_backtest():
    """Run the full V5 Advanced backtest with all 7 innovations."""
    np.random.seed(42)

    end_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    start_dt = datetime.now() - timedelta(days=BACKTEST_DAYS + 2)
    start_date = start_dt.strftime("%Y-%m-%d")

    print("=" * 70)
    print("  V5 ADVANCED BACKTESTER — All 7 Innovations")
    print("=" * 70)
    print(f"  Station:    {STATION_ID} ({STATION['name']})")
    print(f"  Period:     {start_date} to {end_date} ({BACKTEST_DAYS} days)")
    print(f"  Bankroll:   ${BANKROLL_START:.2f}")
    print(f"  Strategy:   V5 Advanced (7 innovations)")
    print(f"  Innovations:")
    print(f"    1. Bias-Corrected Ensemble")
    print(f"    2. Ensemble Spread Confidence")
    print(f"    3. Precipitation-Aware Adjustment")
    print(f"    4. Inter-Model Disagreement")
    print(f"    5. Time-Decay Optimization")
    print(f"    6. Market Efficiency Scoring")
    print(f"    7. Dynamic Ladder + Bimodal Detection")
    print(f"  Cache:      {CACHE_DIR}")
    print("=" * 70)

    # ── Step 1: Fetch all data ──
    print("\n[1/5] Fetching historical data...")
    actuals = fetch_actuals(start_date, end_date)
    if actuals.empty:
        print("ERROR: No actuals data. Aborting.")
        return

    forecasts = fetch_model_forecasts(start_date, end_date)
    if forecasts.empty:
        print("ERROR: No forecast data. Aborting.")
        return

    # ── Step 2: Load calibration data (Innovation 1) ──
    print("\n[2/5] Loading calibration data...")
    calibration = load_calibration_data()
    if calibration:
        print(f"    Loaded calibration for {len(calibration)} models:")
        for model, cal in calibration.items():
            print(f"      {model}: bias={cal.get('bias', 0):.2f}°F, RMSE={cal.get('rmse', 0):.2f}°F")
    else:
        print("    No calibration data found — using raw ensemble (fallback)")

    # ── Step 3: Calibrate from historical data ──
    print("\n[3/5] Calibrating model errors (first 30 days as calibration window)...")
    merged = actuals.merge(forecasts, on="date", how="inner").dropna()
    print(f"    Total matched days: {len(merged)}")

    cal_window = min(30, len(merged) // 3)
    cal_data_actuals = merged.iloc[:cal_window][["date", "actual"]].copy()
    cal_data_forecasts = merged.iloc[:cal_window].drop(columns=["actual"]).copy()
    model_errors = compute_model_errors(cal_data_actuals, cal_data_forecasts)

    test_data = merged.iloc[cal_window:].copy()
    print(f"    Calibration window: {cal_window} days")
    print(f"    Test window: {len(test_data)} days")

    if len(test_data) < 10:
        print("ERROR: Too few test days. Aborting.")
        return

    # ── Step 4: Run daily simulation with all innovations ──
    print(f"\n[4/5] Running V5 Advanced trading simulation ({len(test_data)} days)...")
    bankroll = BANKROLL_START
    peak_bankroll = BANKROLL_START
    all_trades: List[BacktestTrade] = []
    daily_pnl_list = []
    equity_curve = [BANKROLL_START]
    days_tested = 0

    # Innovation tracking counters
    innovation_stats = {
        "bias_corrections_applied": 0,
        "precip_adjustments": 0,
        "spread_low_conf_days": 0,
        "spread_high_conf_days": 0,
        "disagreement_days": 0,
        "agreement_days": 0,
        "bimodal_days": 0,
        "sharp_market_days": 0,
        "soft_market_days": 0,
        "time_decay_filtered": 0,
    }

    for _, row in test_data.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        actual_temp = row["actual"]
        if pd.isna(actual_temp):
            daily_pnl_list.append(0.0)
            equity_curve.append(bankroll)
            continue

        days_tested += 1

        day_forecasts = {}
        for model in FORECAST_MODELS:
            col = f"forecast_{model}"
            if col in row and not pd.isna(row[col]):
                day_forecasts[model] = float(row[col])

        if not day_forecasts:
            daily_pnl_list.append(0.0)
            equity_curve.append(bankroll)
            continue

        # Innovation 3: Precipitation adjustment
        precip_adj = simulate_precip_adjustment(day_forecasts, date_str)
        if precip_adj != 0.0:
            innovation_stats["precip_adjustments"] += 1

        # Compute bias-corrected ensemble mean for bucket centering
        # This is key: center buckets on our CORRECTED forecast, not raw
        values = list(day_forecasts.values())
        raw_mean = np.mean(values)

        # Apply bias correction to get corrected mean
        # bias = actual - forecast, so corrected = forecast + bias moves toward actual
        corrected_values = []
        for model, forecast in day_forecasts.items():
            err = model_errors.get(model, {"bias": 0.0})
            bias = err.get("bias", 0.0)
            corrected_values.append(forecast + bias)
        corrected_mean = np.mean(corrected_values) + precip_adj

        # Build bucket structure centered on corrected mean
        edges = make_bucket_edges(corrected_mean)
        labels = make_bucket_labels(edges)
        winning_bucket = actual_to_winning_bucket(actual_temp, edges, labels)

        # Innovation 1: Bias-corrected probabilities
        our_probs, all_samples = compute_ensemble_probabilities(
            day_forecasts, model_errors, edges, labels,
            calibration=calibration,
            precip_adjustment=precip_adj,
        )
        if calibration:
            innovation_stats["bias_corrections_applied"] += 1

        # Innovation 2+4+7: Advanced ensemble stats
        ensemble_stats = compute_advanced_ensemble_stats(day_forecasts, all_samples)

        # Track innovation triggers
        member_std = ensemble_stats.get("member_std", 3.0)
        if member_std < getattr(config, 'ENSEMBLE_SPREAD_LOW_STD', 2.0):
            innovation_stats["spread_high_conf_days"] += 1
        elif member_std > getattr(config, 'ENSEMBLE_SPREAD_HIGH_STD', 4.0):
            innovation_stats["spread_low_conf_days"] += 1

        per_model = ensemble_stats.get("per_model_medians", {})
        if len(per_model) >= 2:
            ms = list(per_model.values())
            spread = max(ms) - min(ms)
            if spread > getattr(config, 'MODEL_DISAGREEMENT_THRESHOLD_F', 4.0):
                innovation_stats["disagreement_days"] += 1
            elif spread < getattr(config, 'MODEL_AGREEMENT_THRESHOLD_F', 2.0):
                innovation_stats["agreement_days"] += 1

        if ensemble_stats.get("is_bimodal", False):
            innovation_stats["bimodal_days"] += 1

        # Simulate market prices (market uses raw forecasts, not our corrected ones)
        best_match = day_forecasts.get("best_match", raw_mean)
        market_prices = simulate_market_prices(best_match, day_forecasts, edges, labels)

        # Innovation 6: Track market efficiency
        mkt_eff = score_market_efficiency(market_prices)
        if mkt_eff == "sharp":
            innovation_stats["sharp_market_days"] += 1
        elif mkt_eff == "soft":
            innovation_stats["soft_market_days"] += 1

        # Simulate time-to-resolution (backtest: mostly 1-day, some 2-day)
        days_to_resolution = 1.0 if np.random.random() < 0.8 else np.random.choice([2, 3])

        # Find trades with all innovations
        current_exposure = 0.0
        raw_trades = find_trades_v5_advanced(
            our_probs, market_prices, ensemble_stats,
            labels, bankroll, current_exposure,
            days_to_resolution=days_to_resolution,
        )

        # Resolve trades
        day_pnl = 0.0
        for t in raw_trades:
            label = t["label"]
            is_winning = (label == winning_bucket)

            if t["direction"] == "BUY_YES":
                won = is_winning
                if won:
                    pnl = t["shares"] * (1.0 - t["entry_price"])
                else:
                    pnl = -t["size_usd"]
            else:
                won = not is_winning
                if won:
                    pnl = t["shares"] * (1.0 - t["entry_price"])
                else:
                    pnl = -t["size_usd"]

            pnl -= t["size_usd"] * SPREAD_COST
            bankroll += pnl
            day_pnl += pnl

            trade = BacktestTrade(
                date=date_str, bucket=label, direction=t["direction"],
                strategy=t["strategy"], entry_price=t["entry_price"],
                size_usd=t["size_usd"], shares=t["shares"],
                our_prob=t["our_prob"], market_price=t["market_price"],
                edge=t["edge"], won=won, pnl=pnl,
                actual_temp=actual_temp, winning_bucket=winning_bucket,
            )
            all_trades.append(trade)

        daily_pnl_list.append(day_pnl)
        equity_curve.append(bankroll)
        peak_bankroll = max(peak_bankroll, bankroll)

        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if bankroll < BANKROLL_START * 0.25:
            print(f"    [{date_str}] DRAWDOWN HALT: {dd:.1%}, bankroll ${bankroll:.2f}")
            break

    # ── Step 5: Compile results ──
    print(f"\n[5/5] Compiling results...")

    wins = [t for t in all_trades if t.won]
    losses = [t for t in all_trades if not t.won]
    total_pnl = bankroll - BANKROLL_START
    total_return = total_pnl / BANKROLL_START * 100

    win_rate = len(wins) / len(all_trades) if all_trades else 0
    avg_edge = np.mean([abs(t.edge) for t in all_trades]) if all_trades else 0
    avg_pnl = np.mean([t.pnl for t in all_trades]) if all_trades else 0

    gross_wins = sum(t.pnl for t in wins) if wins else 0
    gross_losses = sum(abs(t.pnl) for t in losses) if losses else 1
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0

    # Max drawdown
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdowns = (peak - eq) / np.maximum(peak, 1)
    max_drawdown = float(np.max(drawdowns)) * 100

    # Sharpe ratio
    daily_arr = np.array(daily_pnl_list)
    daily_nonzero = daily_arr[daily_arr != 0] if np.any(daily_arr != 0) else daily_arr
    if len(daily_nonzero) > 1 and np.std(daily_nonzero) > 0:
        sharpe = (np.mean(daily_nonzero) / np.std(daily_nonzero)) * np.sqrt(252)
    else:
        sharpe = 0.0

    trades_per_day = len(all_trades) / days_tested if days_tested > 0 else 0

    # Per-strategy breakdown
    strategy_types = set(t.strategy for t in all_trades)
    strategy_breakdown = {}
    for strat in strategy_types:
        strat_trades = [t for t in all_trades if t.strategy == strat]
        strat_wins = [t for t in strat_trades if t.won]
        strategy_breakdown[strat] = {
            "total": len(strat_trades),
            "wins": len(strat_wins),
            "win_rate": round(len(strat_wins) / len(strat_trades), 4) if strat_trades else 0,
            "pnl": round(sum(t.pnl for t in strat_trades), 2),
        }

    sample_trades = []
    for t in all_trades[:20]:
        sample_trades.append({
            "date": t.date, "bucket": t.bucket, "direction": t.direction,
            "strategy": t.strategy, "entry_price": round(t.entry_price, 4),
            "size_usd": round(t.size_usd, 2), "our_prob": round(t.our_prob, 4),
            "market_price": round(t.market_price, 4), "edge": round(t.edge, 4),
            "won": t.won, "pnl": round(t.pnl, 2),
            "actual_temp": round(t.actual_temp, 1),
            "winning_bucket": t.winning_bucket,
        })

    results = {
        "version": "V5_Advanced_7_Innovations",
        "period": f"{start_date} to {end_date}",
        "station": STATION_ID,
        "days_tested": days_tested,
        "total_trades": len(all_trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "avg_edge_per_trade": round(avg_edge, 4),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "sharpe_ratio": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2),
        "trades_per_day": round(trades_per_day, 1),
        "bankroll_start": BANKROLL_START,
        "bankroll_end": round(bankroll, 2),
        "strategy_breakdown": strategy_breakdown,
        "innovation_stats": innovation_stats,
        "baseline_comparison": {
            "baseline_return_pct": 150.0,
            "baseline_sharpe": 7.20,
            "baseline_max_dd": 8.5,
            "baseline_win_rate": 0.338,
            "improvement_return_pct": round(total_return - 150.0, 2),
            "improvement_sharpe": round(sharpe - 7.20, 2),
            "improvement_max_dd": round(8.5 - max_drawdown, 2),
            "improvement_win_rate": round(win_rate - 0.338, 4),
        },
        "daily_pnl": [round(x, 2) for x in daily_pnl_list],
        "equity_curve": [round(x, 2) for x in equity_curve],
        "sample_trades": sample_trades,
    }

    # Save results
    output_path = "/home/user/workspace/v5_advanced_backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("  V5 ADVANCED BACKTEST RESULTS (7 Innovations)")
    print("=" * 70)
    print(f"  Period:            {start_date} to {end_date}")
    print(f"  Days tested:       {days_tested}")
    print(f"  {'─' * 50}")
    print(f"  Total trades:      {len(all_trades)}")
    print(f"  Winning trades:    {len(wins)}")
    print(f"  Losing trades:     {len(losses)}")
    print(f"  Win rate:          {win_rate:.1%}")
    print(f"  {'─' * 50}")
    print(f"  Total P&L:         ${total_pnl:+.2f}")
    print(f"  Total return:      {total_return:+.1f}%")
    print(f"  Bankroll:          ${BANKROLL_START:.2f} → ${bankroll:.2f}")
    print(f"  {'─' * 50}")
    print(f"  Max drawdown:      {max_drawdown:.1f}%")
    print(f"  Sharpe ratio:      {sharpe:.2f}")
    print(f"  Profit factor:     {profit_factor:.2f}")
    print(f"  Avg edge/trade:    {avg_edge:.1%}")
    print(f"  Avg P&L/trade:     ${avg_pnl:+.2f}")
    print(f"  Trades/day:        {trades_per_day:.1f}")
    print(f"  {'─' * 50}")

    print(f"\n  Strategy Breakdown:")
    for strat, stats in sorted(strategy_breakdown.items()):
        wr = f"{stats['win_rate']:.1%}" if stats['total'] > 0 else "N/A"
        print(f"    {strat:20s}: {stats['total']:3d} trades | Win rate: {wr} | P&L: ${stats['pnl']:+.2f}")

    print(f"\n  Innovation Activity:")
    print(f"    Bias corrections applied:     {innovation_stats['bias_corrections_applied']} days")
    print(f"    Precipitation adjustments:    {innovation_stats['precip_adjustments']} days")
    print(f"    High confidence (low spread): {innovation_stats['spread_high_conf_days']} days")
    print(f"    Low confidence (high spread): {innovation_stats['spread_low_conf_days']} days")
    print(f"    Model disagreement days:      {innovation_stats['disagreement_days']} days")
    print(f"    Model agreement days:         {innovation_stats['agreement_days']} days")
    print(f"    Bimodal distribution days:    {innovation_stats['bimodal_days']} days")
    print(f"    Sharp market days:            {innovation_stats['sharp_market_days']} days")
    print(f"    Soft market days:             {innovation_stats['soft_market_days']} days")

    print(f"\n  {'─' * 50}")
    print(f"  BASELINE COMPARISON:")
    print(f"    {'Metric':25s} {'Baseline':>10s} {'V5 Advanced':>12s} {'Change':>10s}")
    print(f"    {'─' * 57}")
    print(f"    {'Return':25s} {'150.0%':>10s} {total_return:>11.1f}% {total_return - 150.0:>+9.1f}%")
    print(f"    {'Sharpe':25s} {'7.20':>10s} {sharpe:>12.2f} {sharpe - 7.20:>+10.2f}")
    print(f"    {'Max Drawdown':25s} {'8.5%':>10s} {max_drawdown:>11.1f}% {8.5 - max_drawdown:>+9.1f}%")
    print(f"    {'Win Rate':25s} {'33.8%':>10s} {win_rate:>11.1%} {(win_rate - 0.338)*100:>+9.1f}%")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_backtest()
