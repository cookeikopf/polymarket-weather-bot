#!/usr/bin/env python3
"""
V5 Realistic Backtester for Polymarket Weather Bot
====================================================
Uses REAL historical data from Open-Meteo APIs:
  - Historical Forecast API: what models predicted
  - Archive API: what actually happened

Tests the V5 ensemble edge detection strategy over 90 days for NYC.
Simulates realistic market pricing (not random noise).

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

# Strategy params (from spec)
MIN_EDGE = 0.08          # 8% minimum edge
MAX_POSITION_PCT = 0.10  # 10% of bankroll per trade
KELLY_FRACTION = 0.25    # quarter-Kelly
TEMP_BUCKET_SIZE = 2     # 2°F buckets

# Market simulation params — REALISTIC
# Real Polymarket weather markets are fairly efficient. The "market" represents
# all other traders who also have access to NWP data. Our edge is small.
MARKET_NOISE_STD = 1.5   # Market belief noise (°F) — informed but not perfect
OVERROUND_MIN = 1.05     # Minimum vig (Polymarket weather markets)
OVERROUND_MAX = 1.10     # Maximum vig
SLIPPAGE = 0.01          # 1% slippage on execution
SPREAD_COST = 0.02       # 2% bid-ask spread cost per trade

# Ladder strategy params
LADDER_MAX_PRICE = 0.18      # Slightly below 20% — realistic given market vig
LADDER_BUCKETS = 5
LADDER_BET_PER_BUCKET = 2.0   # $2 per bucket = $10 max per ladder set
MAX_LADDER_SETS_PER_DAY = 1  # Only 1 ladder set per day (realistic)
LADDER_MIN_PRICE = 0.02      # Don't buy dust-priced buckets (no liquidity)

# Conservative NO params
NO_MIN_ENTRY = 0.65
NO_MAX_ENTRY = 0.85
MAX_NO_TRADES_PER_DAY = 3  # Cap NO trades per day

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


# ═══════════════════════════════════════════════════════════════════
# CACHING LAYER
# ═══════════════════════════════════════════════════════════════════
def _cache_key(url: str, params: dict) -> str:
    """Generate a deterministic cache filename from request params."""
    raw = json.dumps({"url": url, **params}, sort_keys=True)
    h = hashlib.md5(raw.encode()).hexdigest()
    return h + ".json"


def cached_get(url: str, params: dict, timeout: int = 30) -> Optional[dict]:
    """GET with filesystem cache. Returns parsed JSON or None."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fname = _cache_key(url, params)
    fpath = os.path.join(CACHE_DIR, fname)

    # Check cache
    if os.path.exists(fpath):
        with open(fpath) as f:
            return json.load(f)

    # Fetch from API
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
    """Fetch actual observed temperatures from Archive API."""
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
    """Fetch historical forecasts from all 4 models."""
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
        # Try model-specific key first, then plain key
        var_key = f"temperature_2m_max_{model}" if model != "best_match" else "temperature_2m_max"
        if var_key not in daily:
            var_key = "temperature_2m_max"
        if var_key not in daily:
            print(f"    WARNING: Key {var_key} not in response for {model}")
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
    """Create temperature bucket edges centered around a temperature.

    Polymarket uses 2°F buckets for NYC. We create ~15 buckets covering
    a realistic range around the forecast center.
    """
    step = TEMP_BUCKET_SIZE
    center = round(center_temp / step) * step
    start = center - 14
    end = center + 16
    return list(range(int(start), int(end) + 1, step))


def make_bucket_labels(edges: List[float]) -> List[str]:
    """Create bucket labels matching Polymarket format."""
    labels = []
    # Lower tail
    low_tail_val = int(edges[0]) - 1
    labels.append(f"{low_tail_val}°F or below")
    # Regular buckets
    for i in range(len(edges) - 1):
        low = int(edges[i])
        high = int(edges[i + 1]) - 1
        labels.append(f"{low}-{high}°F")
    # Upper tail
    labels.append(f"{int(edges[-1])}°F or higher")
    return labels


def bucket_center(label: str) -> Optional[float]:
    """Extract center temperature from a bucket label."""
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
    """Determine which bucket the actual temperature falls into."""
    if actual < edges[0]:
        return labels[0]
    if actual >= edges[-1]:
        return labels[-1]
    for i in range(len(edges) - 1):
        if edges[i] <= actual < edges[i + 1]:
            return labels[i + 1]  # +1 because labels[0] is the lower tail
    return labels[-1]


# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE PROBABILITY COMPUTATION (V5 method)
# ═══════════════════════════════════════════════════════════════════
def compute_ensemble_probabilities(
    model_forecasts: Dict[str, float],
    model_errors: Dict[str, Dict],
    edges: List[float],
    labels: List[str],
) -> Dict[str, float]:
    """
    V5 ensemble probability estimation.

    For each model, generate samples from its calibrated error distribution
    centered on the model's forecast. Then count bucket frequencies across
    all samples (Laplace-smoothed).

    This simulates what the V5 ensemble member counting does, using the
    4 deterministic model forecasts + their error distributions as a proxy
    for the full 122-member ensemble.
    """
    n_samples_per_model = 2500
    all_samples = []

    for model, forecast in model_forecasts.items():
        err = model_errors.get(model, {"bias": 0.0, "std": 3.5})
        bias = err.get("bias", err.get("mean_bias", 0.0))
        std = err.get("std", 3.5)

        # Bias-corrected center
        corrected = forecast - bias

        # Student's t with df=5 for slightly heavy tails (matches V4 approach)
        from scipy import stats as sp_stats
        samples = corrected + sp_stats.t.rvs(df=5, loc=0, scale=std, size=n_samples_per_model)
        all_samples.extend(samples)

    all_samples = np.array(all_samples)
    total = len(all_samples)
    n_buckets = len(labels)
    smoothing = 0.5

    probs = {}
    # Lower tail
    count = np.sum(all_samples < edges[0])
    probs[labels[0]] = (count + smoothing) / (total + smoothing * n_buckets)

    # Regular buckets
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        count = np.sum((all_samples >= low) & (all_samples < high))
        probs[labels[i + 1]] = (count + smoothing) / (total + smoothing * n_buckets)

    # Upper tail
    count = np.sum(all_samples >= edges[-1])
    probs[labels[-1]] = (count + smoothing) / (total + smoothing * n_buckets)

    # Normalize
    total_p = sum(probs.values())
    if total_p > 0:
        probs = {k: v / total_p for k, v in probs.items()}
    return probs


# ═══════════════════════════════════════════════════════════════════
# MARKET PRICE SIMULATION (Realistic)
# ═══════════════════════════════════════════════════════════════════
def simulate_market_prices(
    best_match_forecast: float,
    all_model_forecasts: Dict[str, float],
    edges: List[float],
    labels: List[str],
) -> Dict[str, float]:
    """
    Simulate realistic Polymarket prices.

    The market is WELL-INFORMED — other sophisticated traders also use NWP models.
    The market consensus reflects a blend of public models, not just best_match.
    Our edge comes from better calibration and ensemble weighting, NOT from having
    access to data the market doesn't have.

    Key realism features:
    - Market belief = weighted average of public models + small noise
    - Price distribution = Gaussian with std=2.5°F (tighter than raw model uncertainty)
    - Overround: prices sum to 1.04-1.10 (realistic Polymarket vig)
    - The most-likely bucket gets priced at 15-40% (realistic for weather markets)
    """
    # Market uses a weighted blend of public models. Other sophisticated traders
    # (like neobrother) also have NWP model access. The market is well-informed.
    if all_model_forecasts:
        # 70% of the time, market uses simple avg; 30% it happens to weight
        # ECMWF more heavily (the best model) — simulating smart money
        if np.random.random() < 0.3 and "ecmwf_ifs025" in all_model_forecasts:
            vals = list(all_model_forecasts.values())
            ecmwf = all_model_forecasts["ecmwf_ifs025"]
            market_belief = 0.4 * ecmwf + 0.6 * np.mean(vals)
        else:
            market_belief = np.mean(list(all_model_forecasts.values()))
    else:
        market_belief = best_match_forecast
    # Add noise — market isn't perfectly on model consensus
    market_belief += np.random.normal(0, MARKET_NOISE_STD)

    raw_prices = {}
    for label in labels:
        center = bucket_center(label)
        if center is None:
            raw_prices[label] = 0.01
            continue

        # Market uses a mixture distribution: sharp consensus + uncertainty tail
        # This produces realistic peak prices (~22-28%) while keeping tails low.
        # Sharp component (0.5 weight, std=2.5°F): strong consensus near peak
        # Broad component (0.5 weight, std=5.0°F): acknowledges uncertainty
        z_sharp = (center - market_belief) / 2.5
        z_broad = (center - market_belief) / 5.0
        raw_prices[label] = 0.5 * np.exp(-0.5 * z_sharp**2) + 0.5 * np.exp(-0.5 * z_broad**2)

    # Normalize to sum to 1, then apply vig
    total = sum(raw_prices.values())
    if total <= 0:
        total = 1.0
    prices = {k: v / total for k, v in raw_prices.items()}

    # Apply overround (vig)
    overround = np.random.uniform(OVERROUND_MIN, OVERROUND_MAX)
    prices = {k: v * overround for k, v in prices.items()}

    # Clip to valid range
    prices = {k: np.clip(v, 0.01, 0.99) for k, v in prices.items()}

    return prices


# ═══════════════════════════════════════════════════════════════════
# EDGE DETECTION (V5 dual strategy)
# ═══════════════════════════════════════════════════════════════════
@dataclass
class BacktestTrade:
    """A single backtested trade."""
    date: str
    bucket: str
    direction: str        # "BUY_YES" or "BUY_NO"
    strategy: str         # "ladder" or "conservative_no"
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


def find_trades(
    our_probs: Dict[str, float],
    market_prices: Dict[str, float],
    ensemble_mean: float,
    ensemble_std: float,
    labels: List[str],
    bankroll: float,
    current_exposure: float,
) -> List[dict]:
    """
    Find trading opportunities using V5 dual strategy.

    Strategy 1 (LADDER): Buy YES in buckets near ensemble median at low prices.
    Strategy 2 (CONSERVATIVE NO): Buy NO on unlikely outcomes at high entry prices.

    Returns list of trade dicts (not yet resolved).
    """
    trades = []
    n_models = len(FORECAST_MODELS)
    model_range = ensemble_std * 2  # approximate
    agreement = max(0, 1.0 - model_range / 10.0)

    # ── STRATEGY 1: LADDER ──
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
    ladder = candidates[:LADDER_BUCKETS]  # Up to 5 buckets per ladder set

    for c in ladder[:LADDER_BUCKETS * MAX_LADDER_SETS_PER_DAY]:
        entry = c["price"] + np.random.uniform(0, SLIPPAGE)
        entry = np.clip(entry, 0.01, 0.99)
        size = min(LADDER_BET_PER_BUCKET, bankroll * MAX_POSITION_PCT)
        remaining = (bankroll * 0.60) - current_exposure
        if remaining < size:
            continue
        if size < 1.0:
            continue

        shares = size / entry
        edge = c["prob"] - entry
        # EV
        ev = c["prob"] * (1.0 - entry) - (1.0 - c["prob"]) * entry

        trades.append({
            "label": c["label"], "direction": "BUY_YES", "strategy": "ladder",
            "entry_price": entry, "size_usd": size, "shares": shares,
            "our_prob": c["prob"], "market_price": c["price"], "edge": edge,
        })
        current_exposure += size

    # ── STRATEGY 2: CONSERVATIVE NO ──
    no_count = 0
    for label in labels:
        if no_count >= MAX_NO_TRADES_PER_DAY:
            break
        prob_yes = our_probs.get(label, 0)
        price_yes = market_prices.get(label, 0)
        if price_yes <= 0.005:
            continue

        # NO entry: we pay (1 - price_yes) for NO shares
        entry_no = 1.0 - price_yes
        if entry_no < NO_MIN_ENTRY or entry_no > NO_MAX_ENTRY:
            continue

        prob_no = 1.0 - prob_yes
        edge = prob_no - entry_no
        if edge < MIN_EDGE:
            continue

        # Confidence check
        prob_uncertainty = min(0.30, ensemble_std * 0.03)
        edge_significance = min(1.0, edge / max(prob_uncertainty, 0.01))
        confidence = 0.40 * agreement + 0.35 * edge_significance + 0.25 * min(1.0, n_models / 5)
        if confidence < 0.45:
            continue

        # Kelly sizing
        b = (1.0 - entry_no) / entry_no if entry_no > 0 and entry_no < 1 else 0
        if b <= 0:
            continue
        kelly = max(0, (b * prob_no - (1.0 - prob_no)) / b)
        kelly *= KELLY_FRACTION
        kelly = min(kelly, MAX_POSITION_PCT)

        size = kelly * bankroll * confidence
        size = min(size, bankroll * MAX_POSITION_PCT, 10.0)
        remaining = (bankroll * 0.60) - current_exposure
        if remaining < size:
            continue
        if size < 2.0:
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

    return trades


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════
def compute_model_errors(
    actuals: pd.DataFrame, forecasts: pd.DataFrame
) -> Dict[str, Dict]:
    """Compute per-model bias and std from historical forecast vs actual."""
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
    """Run the full V5 backtest."""
    np.random.seed(42)  # Reproducible results

    end_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    start_dt = datetime.now() - timedelta(days=BACKTEST_DAYS + 2)
    start_date = start_dt.strftime("%Y-%m-%d")

    print("=" * 70)
    print("  V5 REALISTIC BACKTESTER — Polymarket Weather Bot")
    print("=" * 70)
    print(f"  Station:    {STATION_ID} ({STATION['name']})")
    print(f"  Period:     {start_date} to {end_date} ({BACKTEST_DAYS} days)")
    print(f"  Bankroll:   ${BANKROLL_START:.2f}")
    print(f"  Strategy:   V5 Dual (Ladder + Conservative NO)")
    print(f"  Data:       Real Open-Meteo historical forecasts + actuals")
    print(f"  Cache:      {CACHE_DIR}")
    print("=" * 70)

    # ── Step 1: Fetch all data ──
    print("\n[1/4] Fetching historical data...")
    actuals = fetch_actuals(start_date, end_date)
    if actuals.empty:
        print("ERROR: No actuals data. Aborting.")
        return

    forecasts = fetch_model_forecasts(start_date, end_date)
    if forecasts.empty:
        print("ERROR: No forecast data. Aborting.")
        return

    # ── Step 2: Calibrate (using first 30 days to avoid look-ahead bias) ──
    print("\n[2/4] Calibrating model errors (first 30 days as calibration window)...")
    merged = actuals.merge(forecasts, on="date", how="inner").dropna()
    print(f"    Total matched days: {len(merged)}")

    # Split: first 30 days for calibration, rest for testing
    cal_window = min(30, len(merged) // 3)
    cal_data_actuals = merged.iloc[:cal_window][["date", "actual"]].copy()
    cal_data_forecasts = merged.iloc[:cal_window].drop(columns=["actual"]).copy()
    model_errors = compute_model_errors(cal_data_actuals, cal_data_forecasts)

    # Test on remaining days
    test_data = merged.iloc[cal_window:].copy()
    print(f"    Calibration window: {cal_window} days")
    print(f"    Test window: {len(test_data)} days")

    if len(test_data) < 10:
        print("ERROR: Too few test days. Aborting.")
        return

    # ── Step 3: Run daily simulation ──
    print(f"\n[3/4] Running daily trading simulation ({len(test_data)} days)...")
    bankroll = BANKROLL_START
    peak_bankroll = BANKROLL_START
    all_trades: List[BacktestTrade] = []
    daily_pnl_list = []
    equity_curve = [BANKROLL_START]
    days_tested = 0

    for _, row in test_data.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        actual_temp = row["actual"]
        if pd.isna(actual_temp):
            daily_pnl_list.append(0.0)
            equity_curve.append(bankroll)
            continue

        days_tested += 1

        # Get model forecasts for this day
        day_forecasts = {}
        for model in FORECAST_MODELS:
            col = f"forecast_{model}"
            if col in row and not pd.isna(row[col]):
                day_forecasts[model] = float(row[col])

        if not day_forecasts:
            daily_pnl_list.append(0.0)
            equity_curve.append(bankroll)
            continue

        # Ensemble stats
        values = list(day_forecasts.values())
        ensemble_mean = np.mean(values)
        ensemble_std = np.std(values) if len(values) > 1 else 2.0

        # Build bucket structure centered on ensemble mean
        edges = make_bucket_edges(ensemble_mean)
        labels = make_bucket_labels(edges)
        winning_bucket = actual_to_winning_bucket(actual_temp, edges, labels)

        # Our V5 ensemble probabilities
        our_probs = compute_ensemble_probabilities(
            day_forecasts, model_errors, edges, labels
        )

        # Simulate market prices (market knows public models but not our calibration)
        best_match = day_forecasts.get("best_match", ensemble_mean)
        market_prices = simulate_market_prices(best_match, day_forecasts, edges, labels)

        # Find trading opportunities
        current_exposure = 0.0
        raw_trades = find_trades(
            our_probs, market_prices, ensemble_mean, ensemble_std,
            labels, bankroll, current_exposure,
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
            else:  # BUY_NO
                won = not is_winning
                if won:
                    pnl = t["shares"] * (1.0 - t["entry_price"])
                else:
                    pnl = -t["size_usd"]

            # Apply spread cost (bid-ask friction)
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

        # Track drawdown (no halt for backtest — we want full-period stats)
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd > 0.25 and bankroll < BANKROLL_START * 0.5:
            # Only halt if we've lost more than 50% of starting bankroll
            print(f"    [{date_str}] DRAWDOWN HALT: {dd:.1%}, bankroll ${bankroll:.2f}")
            break

    # ── Step 4: Compile results ──
    print(f"\n[4/4] Compiling results...")

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

    # Sharpe ratio (annualized from daily)
    daily_arr = np.array(daily_pnl_list)
    daily_nonzero = daily_arr[daily_arr != 0] if np.any(daily_arr != 0) else daily_arr
    if len(daily_nonzero) > 1 and np.std(daily_nonzero) > 0:
        sharpe = (np.mean(daily_nonzero) / np.std(daily_nonzero)) * np.sqrt(252)
    else:
        sharpe = 0.0

    trades_per_day = len(all_trades) / days_tested if days_tested > 0 else 0

    # Separate stats by strategy
    ladder_trades = [t for t in all_trades if t.strategy == "ladder"]
    no_trades = [t for t in all_trades if t.strategy == "conservative_no"]
    ladder_wins = [t for t in ladder_trades if t.won]
    no_wins = [t for t in no_trades if t.won]

    # ── Build results dict ──
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
        "strategy_breakdown": {
            "ladder": {
                "total": len(ladder_trades),
                "wins": len(ladder_wins),
                "win_rate": round(len(ladder_wins) / len(ladder_trades), 4) if ladder_trades else 0,
                "pnl": round(sum(t.pnl for t in ladder_trades), 2),
            },
            "conservative_no": {
                "total": len(no_trades),
                "wins": len(no_wins),
                "win_rate": round(len(no_wins) / len(no_trades), 4) if no_trades else 0,
                "pnl": round(sum(t.pnl for t in no_trades), 2),
            },
        },
        "daily_pnl": [round(x, 2) for x in daily_pnl_list],
        "equity_curve": [round(x, 2) for x in equity_curve],
        "sample_trades": sample_trades,
    }

    # Save results
    output_path = "/home/user/workspace/v5_backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("  V5 BACKTEST RESULTS")
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
    print(f"  LADDER strategy:")
    print(f"    Trades: {len(ladder_trades)} | "
          f"Win rate: {len(ladder_wins)/len(ladder_trades):.1%}" if ladder_trades else "    Trades: 0")
    if ladder_trades:
        print(f"    P&L: ${sum(t.pnl for t in ladder_trades):+.2f}")
    print(f"  CONSERVATIVE NO strategy:")
    print(f"    Trades: {len(no_trades)} | "
          f"Win rate: {len(no_wins)/len(no_trades):.1%}" if no_trades else "    Trades: 0")
    if no_trades:
        print(f"    P&L: ${sum(t.pnl for t in no_trades):+.2f}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_backtest()
