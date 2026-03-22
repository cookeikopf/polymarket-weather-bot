"""
Real-Data Backtester for Polymarket Weather Bot V6
====================================================
Collects REAL market data from Polymarket + REAL weather forecasts from Open-Meteo,
then simulates the bot's strategy against actual historical prices and resolutions.

Usage: python real_backtest.py [days_back]
"""

import os
import json
import time
import re
import sys
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy import stats as sp_stats
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION (mirrors config.py but standalone for backtest)
# ═══════════════════════════════════════════════════════════════════

OPEN_METEO_API_KEY = os.getenv("OPEN_METEO_API_KEY", "wjrcKzLOeLkcCnzx")
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

_PREFIX = "customer-" if OPEN_METEO_API_KEY else ""
HIST_FORECAST_URL = f"https://{_PREFIX}historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = f"https://{_PREFIX}archive-api.open-meteo.com/v1/archive"

STATIONS = {
    "NYC":     {"lat": 40.7769, "lon": -73.8740, "unit": "fahrenheit", "slug": "nyc"},
    "Chicago": {"lat": 41.9742, "lon": -87.9073, "unit": "fahrenheit", "slug": "chicago"},
    "Miami":   {"lat": 25.7959, "lon": -80.2870, "unit": "fahrenheit", "slug": "miami"},
    "London":  {"lat": 51.4700, "lon": -0.4543,  "unit": "celsius",    "slug": "london"},
    "Tokyo":   {"lat": 35.5494, "lon": 139.7798, "unit": "celsius",    "slug": "tokyo"},
    "Paris":   {"lat": 49.0097, "lon": 2.5479,   "unit": "celsius",    "slug": "paris"},
    "Seoul":   {"lat": 37.4602, "lon": 126.4407, "unit": "celsius",    "slug": "seoul"},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "unit": "fahrenheit", "slug": "atlanta"},
    "Dallas":  {"lat": 32.8998, "lon": -97.0403, "unit": "fahrenheit", "slug": "dallas"},
    "Toronto": {"lat": 43.6777, "lon": -79.6248, "unit": "celsius",    "slug": "toronto"},
}

HIST_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "best_match"]

# ═══════════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS (to be optimized)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StrategyParams:
    """All tunable strategy parameters."""
    # Ladder
    ladder_enabled: bool = True
    ladder_max_entry: float = 0.20
    ladder_buckets: int = 3
    ladder_bet_per_bucket: float = 2.0

    # Conservative NO
    no_enabled: bool = True
    no_min_entry: float = 0.55
    no_max_entry: float = 0.85
    no_min_edge: float = 0.12

    # Kelly & sizing
    kelly_fraction: float = 0.15
    max_position_pct: float = 0.15
    min_trade_usd: float = 5.0
    max_trade_usd: float = 10.0
    max_exposure: float = 0.60

    # Bankroll
    initial_bankroll: float = 100.0

    # Forecast KDE bandwidth
    kde_bw_mult: float = 1.06

    # Entry timing: "early", "mid", "late"
    entry_timing: str = "mid"

    # Minimum forecast model count
    min_models: int = 2

    # Confidence threshold for NO
    no_min_confidence: float = 0.45


# ═══════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════

def api_get(url, params, timeout=15, retries=3):
    """GET with retries and rate limit handling."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = min(60, 5 * (2 ** attempt))
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            return resp
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    return None


def collect_all_data(cities: List[str], days_back: int = 21) -> List[dict]:
    """Collect real market data + weather forecasts for backtesting."""
    all_records = []
    total_attempts = 0
    found = 0

    for days_ago in range(3, days_back + 1):
        date = datetime.now() - timedelta(days=days_ago)
        target_date = date.strftime("%Y-%m-%d")
        month = date.strftime("%B").lower()
        day = date.day
        year = date.year

        for sid, info in [(s, STATIONS[s]) for s in cities]:
            slug = f"highest-temperature-in-{info['slug']}-on-{month}-{day}-{year}"
            total_attempts += 1

            # 1) Fetch Gamma event
            resp = api_get(f"{GAMMA_API}/events", {"slug": slug})
            if not resp or resp.status_code != 200:
                time.sleep(0.1)
                continue

            data = resp.json()
            if not data or not isinstance(data, list) or not data:
                time.sleep(0.1)
                continue

            ev = data[0]
            sub_markets = ev.get("markets", [])
            if not sub_markets:
                continue

            # 2) Parse all outcomes with real prices
            outcomes = []
            winning_bucket = None

            for m in sub_markets:
                q = m.get("question", "")
                tokens = json.loads(m.get("clobTokenIds", "[]"))
                prices = json.loads(m.get("outcomePrices", "[]"))

                yes_token = tokens[0] if tokens else ""
                no_token = tokens[1] if len(tokens) > 1 else ""
                yes_final = float(prices[0]) if prices else 0

                # Bucket name
                name = extract_bucket_name(q, info["unit"])

                # Price history from CLOB
                price_history = []
                if yes_token:
                    hr = api_get(f"{CLOB_API}/prices-history",
                                 {"market": yes_token, "interval": "max", "fidelity": 60})
                    if hr and hr.status_code == 200:
                        price_history = hr.json().get("history", [])
                    time.sleep(0.12)

                entry_prices = compute_entry_prices(price_history)
                is_winner = yes_final > 0.5

                if is_winner:
                    winning_bucket = name

                outcomes.append({
                    "name": name,
                    "yes_token": yes_token,
                    "final_price": yes_final,
                    "is_winner": is_winner,
                    "entry_early": entry_prices["early"],
                    "entry_mid": entry_prices["mid"],
                    "entry_late": entry_prices["late"],
                    "n_price_points": len(price_history),
                })

            # 3) Fetch weather forecasts (what models predicted the day before)
            forecast_date = (date - timedelta(days=1)).strftime("%Y-%m-%d")
            forecasts = fetch_historical_forecast(
                info["lat"], info["lon"], target_date, forecast_date, info["unit"]
            )
            time.sleep(0.3)

            # 4) Fetch actual temperature
            actual_temp = fetch_actual_temp(info["lat"], info["lon"], target_date, info["unit"])
            time.sleep(0.2)

            # 5) Compute forecast stats
            if forecasts:
                temps = list(forecasts.values())
                forecast_mean = float(np.mean(temps))
                forecast_std = float(np.std(temps)) if len(temps) > 1 else 3.0
                forecast_median = float(np.median(temps))
            else:
                forecast_mean = forecast_std = forecast_median = None

            record = {
                "slug": slug,
                "station_id": sid,
                "target_date": target_date,
                "title": ev.get("title", ""),
                "winning_bucket": winning_bucket,
                "n_outcomes": len(outcomes),
                "outcomes": outcomes,
                "forecasts": forecasts,
                "actual_temp": actual_temp,
                "forecast_mean": forecast_mean,
                "forecast_std": forecast_std,
                "forecast_median": forecast_median,
            }
            all_records.append(record)
            found += 1
            winner_str = winning_bucket or "unresolved"
            fc_str = f"mean={forecast_mean:.1f}" if forecast_mean else "no-fc"
            actual_str = f"actual={actual_temp:.1f}" if actual_temp else "no-actual"
            print(f"  [{found}] {target_date} {sid:10s} | {len(outcomes)} outcomes | {winner_str} | {fc_str} | {actual_str}")

    print(f"\nData collection: {found}/{total_attempts} markets found")
    return all_records


def extract_bucket_name(question: str, unit: str) -> str:
    """Extract bucket label from market question."""
    m = re.search(r'between\s+(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*F', question)
    if m:
        return f"{m.group(1)}-{m.group(2)}°F"
    m = re.search(r'be\s+(-?\d+)\s*°\s*C', question)
    if m:
        return f"{m.group(1)}°C"
    m = re.search(r'(-?\d+)\s*°\s*([FC])\s+or\s+higher', question)
    if m:
        return f"{m.group(1)}°{m.group(2)} or higher"
    m = re.search(r'(-?\d+)\s*°\s*([FC])\s+or\s+(?:below|lower)', question)
    if m:
        return f"{m.group(1)}°{m.group(2)} or below"
    m = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*F', question)
    if m:
        return f"{m.group(1)}-{m.group(2)}°F"
    m = re.search(r'(-?\d+)\s*°\s*C', question)
    if m:
        return f"{m.group(1)}°C"
    return question[:40]


def compute_entry_prices(history: list) -> dict:
    """Compute entry prices at different time points."""
    if not history:
        return {"early": 0, "mid": 0, "late": 0}
    prices = [float(h.get("p", 0)) for h in history if h.get("p")]
    if not prices:
        return {"early": 0, "mid": 0, "late": 0}
    n = len(prices)
    return {
        "early": prices[min(2, n - 1)],
        "mid": prices[n // 2],
        "late": prices[max(0, n - max(1, n // 4))],
    }


def fetch_historical_forecast(lat, lon, target_date, forecast_date, unit) -> Dict[str, float]:
    """Fetch what models predicted for target_date as of forecast_date."""
    temp_unit = "fahrenheit" if unit == "fahrenheit" else "celsius"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": temp_unit,
        "timezone": "auto",
        "start_date": forecast_date,
        "end_date": target_date,
        "models": ",".join(HIST_MODELS),
    }
    if OPEN_METEO_API_KEY:
        params["apikey"] = OPEN_METEO_API_KEY

    resp = api_get(HIST_FORECAST_URL, params)
    if not resp or resp.status_code != 200:
        return {}

    data = resp.json()
    daily = data.get("daily", {})
    times = daily.get("time", [])

    target_idx = None
    for i, t in enumerate(times):
        if t == target_date:
            target_idx = i
            break
    if target_idx is None:
        target_idx = len(times) - 1 if times else None
    if target_idx is None:
        return {}

    results = {}
    for model in HIST_MODELS:
        key = f"temperature_2m_max_{model}"
        vals = daily.get(key, [])
        if target_idx < len(vals) and vals[target_idx] is not None:
            results[model] = float(vals[target_idx])
    return results


def fetch_actual_temp(lat, lon, date, unit) -> Optional[float]:
    """Fetch actual observed max temperature."""
    temp_unit = "fahrenheit" if unit == "fahrenheit" else "celsius"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": temp_unit,
        "timezone": "auto",
        "start_date": date, "end_date": date,
    }
    if OPEN_METEO_API_KEY:
        params["apikey"] = OPEN_METEO_API_KEY

    resp = api_get(ARCHIVE_URL, params)
    if not resp or resp.status_code != 200:
        return None
    vals = resp.json().get("daily", {}).get("temperature_2m_max", [])
    if vals and vals[0] is not None:
        return float(vals[0])
    return None


# ═══════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """A single trade in the backtest."""
    date: str
    station: str
    outcome_name: str
    direction: str       # "BUY_YES" or "BUY_NO"
    strategy: str        # "ladder" or "conservative_no"
    entry_price: float
    size_usd: float
    our_prob: float
    edge: float
    confidence: float
    is_winner: bool      # Did this outcome actually win?
    pnl: float = 0.0
    roi: float = 0.0


def extract_bucket_temp(name: str, is_fahrenheit: bool) -> Optional[float]:
    """Extract center temperature from bucket name."""
    m = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F', name)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2.0
    m = re.search(r'(-?\d+)\s*°\s*C', name)
    if m:
        return float(m.group(1))
    m = re.search(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*below', name, re.I)
    if m:
        return float(m.group(1)) - (1 if is_fahrenheit else 0.5)
    m = re.search(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*higher', name, re.I)
    if m:
        return float(m.group(1)) + (1 if is_fahrenheit else 0.5)
    return None


def compute_bucket_probabilities(forecasts: Dict[str, float], outcomes: List[dict],
                                  is_fahrenheit: bool, kde_bw_mult: float = 1.06) -> Dict[str, float]:
    """Compute probability for each bucket using KDE on forecast models."""
    if not forecasts:
        return {}

    samples = np.array(list(forecasts.values()))
    if len(samples) < 2:
        # With only 1 model, use wider gaussian
        mean = samples[0]
        std = 3.0 if is_fahrenheit else 1.5
    else:
        mean = np.mean(samples)
        std = max(np.std(samples), 0.5)

    # Generate probability for each bucket
    probs = {}
    for outcome in outcomes:
        name = outcome["name"]
        temp = extract_bucket_temp(name, is_fahrenheit)
        if temp is None:
            probs[name] = 0.01
            continue

        # Check for tail buckets
        is_lower_tail = bool(re.search(r'or\s+below', name, re.I))
        is_upper_tail = bool(re.search(r'or\s+higher', name, re.I))

        if is_lower_tail:
            # P(X < threshold)
            prob = sp_stats.norm.cdf(temp + 0.5, loc=mean, scale=max(std, 1.0))
        elif is_upper_tail:
            # P(X >= threshold)
            prob = 1.0 - sp_stats.norm.cdf(temp - 0.5, loc=mean, scale=max(std, 1.0))
        else:
            # Range bucket
            if is_fahrenheit:
                # "56-57°F" means 56 to 57.999
                m2 = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)', name)
                if m2:
                    low = int(m2.group(1))
                    high = int(m2.group(2)) + 1  # inclusive upper
                else:
                    low = temp - 1
                    high = temp + 1
            else:
                # "14°C" typically means 14.0 to 14.999
                low = temp
                high = temp + 1

            prob = sp_stats.norm.cdf(high, loc=mean, scale=max(std, 1.0)) - \
                   sp_stats.norm.cdf(low, loc=mean, scale=max(std, 1.0))

        probs[name] = max(prob, 0.001)

    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs


def run_backtest(markets: List[dict], params: StrategyParams) -> Tuple[List[Trade], dict]:
    """Run complete backtest with given parameters against real data."""
    trades = []
    bankroll = params.initial_bankroll
    peak_bankroll = bankroll
    max_drawdown = 0
    daily_pnl = defaultdict(float)
    exposure = 0.0

    for market in markets:
        if not market.get("forecasts") or not market.get("winning_bucket"):
            continue
        if market.get("forecast_mean") is None:
            continue

        sid = market["station_id"]
        is_f = STATIONS[sid]["unit"] == "fahrenheit"
        target_date = market["target_date"]
        forecasts = market["forecasts"]
        outcomes = market["outcomes"]

        if len(forecasts) < params.min_models:
            continue

        # Compute our probability distribution
        our_probs = compute_bucket_probabilities(forecasts, outcomes, is_f, params.kde_bw_mult)
        if not our_probs:
            continue

        forecast_mean = market["forecast_mean"]
        forecast_std = market.get("forecast_std", 3.0)
        forecast_median = market.get("forecast_median", forecast_mean)

        # === Strategy 1: LADDER (BUY YES near median) ===
        if params.ladder_enabled:
            # Find candidates near median
            candidates = []
            for outcome in outcomes:
                name = outcome["name"]
                prob = our_probs.get(name, 0)
                if prob < 0.01:
                    continue

                # Get entry price at chosen timing
                entry = outcome.get(f"entry_{params.entry_timing}", 0)
                if entry <= 0.005 or entry > params.ladder_max_entry:
                    continue

                bucket_temp = extract_bucket_temp(name, is_f)
                if bucket_temp is None:
                    continue

                dist = abs(bucket_temp - forecast_median)
                candidates.append((outcome, prob, entry, bucket_temp, dist, name))

            candidates.sort(key=lambda c: c[4])  # Nearest to median first

            for outcome, prob, entry, temp, dist, name in candidates[:params.ladder_buckets]:
                if entry <= 0 or entry >= 1:
                    continue

                edge = prob - entry
                win_pnl = 1.0 - entry
                ev_per_dollar = (prob * win_pnl - (1 - prob) * entry) / entry

                # Check if this outcome actually won
                is_winner = outcome.get("is_winner", False)
                size = min(params.ladder_bet_per_bucket,
                          bankroll * params.max_exposure - exposure)
                if size < 1.0:
                    continue

                # PnL calculation
                if is_winner:
                    pnl = size * (1.0 / entry - 1)  # Profit on YES win
                else:
                    pnl = -size  # Lost entire stake

                trade = Trade(
                    date=target_date, station=sid, outcome_name=name,
                    direction="BUY_YES", strategy="ladder",
                    entry_price=entry, size_usd=size,
                    our_prob=prob, edge=edge,
                    confidence=0.5,
                    is_winner=is_winner, pnl=pnl,
                    roi=pnl / size if size > 0 else 0,
                )
                trades.append(trade)
                bankroll += pnl
                daily_pnl[target_date] += pnl
                exposure += size if not is_winner else 0

                peak_bankroll = max(peak_bankroll, bankroll)
                dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                max_drawdown = max(max_drawdown, dd)

        # === Strategy 2: CONSERVATIVE NO (BUY NO on unlikely outcomes) ===
        if params.no_enabled:
            for outcome in outcomes:
                name = outcome["name"]
                prob_yes = our_probs.get(name, 0)
                prob_no = 1.0 - prob_yes

                entry_yes = outcome.get(f"entry_{params.entry_timing}", 0)
                if entry_yes <= 0:
                    continue

                # NO entry price = 1 - YES price
                entry_no = 1.0 - entry_yes

                if entry_no < params.no_min_entry or entry_no > params.no_max_entry:
                    continue

                edge = prob_no - entry_no
                if edge < params.no_min_edge:
                    continue

                # Confidence
                model_std = forecast_std
                prob_unc = min(0.30, model_std * 0.03)
                edge_sig = min(1.0, edge / max(prob_unc, 0.01))
                n_models = len(forecasts)
                agreement = max(0, min(1, 1.0 - (forecast_std / 10.0)))
                confidence = 0.4 * agreement + 0.35 * edge_sig + 0.25 * min(1.0, n_models / 5)

                if confidence < params.no_min_confidence:
                    continue

                # Kelly sizing
                if entry_no > 0 and entry_no < 1:
                    b = (1.0 - entry_no) / entry_no
                    kelly = max(0, (b * prob_no - (1 - prob_no)) / b)
                    kelly = min(kelly * params.kelly_fraction, params.max_position_pct)
                else:
                    kelly = 0

                size = min(kelly * bankroll * confidence, params.max_trade_usd)
                remaining = bankroll * params.max_exposure - exposure
                size = min(size, remaining)
                if size < params.min_trade_usd:
                    size = params.min_trade_usd if params.min_trade_usd <= remaining else 0
                if size <= 0:
                    continue

                # Check actual resolution: NO wins if this outcome did NOT win
                is_winner_no = not outcome.get("is_winner", False)

                if is_winner_no:
                    pnl = size * ((1.0 - entry_no) / entry_no)  # Profit on NO win
                else:
                    pnl = -size  # Lost entire stake

                trade = Trade(
                    date=target_date, station=sid, outcome_name=name,
                    direction="BUY_NO", strategy="conservative_no",
                    entry_price=entry_no, size_usd=size,
                    our_prob=prob_no, edge=edge,
                    confidence=confidence,
                    is_winner=is_winner_no, pnl=pnl,
                    roi=pnl / size if size > 0 else 0,
                )
                trades.append(trade)
                bankroll += pnl
                daily_pnl[target_date] += pnl
                exposure += size if not is_winner_no else 0

                peak_bankroll = max(peak_bankroll, bankroll)
                dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                max_drawdown = max(max_drawdown, dd)

    # Compute summary stats
    if not trades:
        return trades, {"error": "No trades generated"}

    total_trades = len(trades)
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    win_rate = len(winners) / total_trades if total_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_win = np.mean([t.pnl for t in winners]) if winners else 0
    avg_loss = np.mean([t.pnl for t in losers]) if losers else 0

    # Strategy breakdown
    ladder_trades = [t for t in trades if t.strategy == "ladder"]
    no_trades = [t for t in trades if t.strategy == "conservative_no"]

    ladder_pnl = sum(t.pnl for t in ladder_trades)
    no_pnl = sum(t.pnl for t in no_trades)
    ladder_wr = len([t for t in ladder_trades if t.pnl > 0]) / len(ladder_trades) if ladder_trades else 0
    no_wr = len([t for t in no_trades if t.pnl > 0]) / len(no_trades) if no_trades else 0

    # Daily stats
    daily_vals = list(daily_pnl.values())
    winning_days = len([d for d in daily_vals if d > 0])
    losing_days = len([d for d in daily_vals if d <= 0])

    # Per-city stats
    city_pnl = defaultdict(float)
    city_trades = defaultdict(int)
    for t in trades:
        city_pnl[t.station] += t.pnl
        city_trades[t.station] += 1

    summary = {
        "total_trades": total_trades,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "final_bankroll": bankroll,
        "roi_pct": (bankroll - params.initial_bankroll) / params.initial_bankroll * 100,
        "max_drawdown": max_drawdown,
        "peak_bankroll": peak_bankroll,
        "sharpe": np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(365) if daily_vals and np.std(daily_vals) > 0 else 0,

        "ladder_trades": len(ladder_trades),
        "ladder_pnl": ladder_pnl,
        "ladder_win_rate": ladder_wr,
        "no_trades": len(no_trades),
        "no_pnl": no_pnl,
        "no_win_rate": no_wr,

        "winning_days": winning_days,
        "losing_days": losing_days,

        "city_pnl": dict(city_pnl),
        "city_trades": dict(city_trades),

        "params": asdict(params),
    }

    return trades, summary


def print_report(trades: List[Trade], summary: dict, label: str = ""):
    """Print formatted backtest report."""
    print(f"\n{'='*70}")
    print(f"  REAL BACKTEST RESULTS {label}")
    print(f"{'='*70}")

    print(f"\n  Total Trades:     {summary['total_trades']}")
    print(f"  Winners:          {summary['winners']} ({summary['win_rate']:.1%})")
    print(f"  Losers:           {summary['losers']}")

    print(f"\n  Total P&L:        ${summary['total_pnl']:+.2f}")
    print(f"  Gross Profit:     ${summary['gross_profit']:.2f}")
    print(f"  Gross Loss:       ${summary['gross_loss']:.2f}")
    print(f"  Profit Factor:    {summary['profit_factor']:.2f}")

    print(f"\n  Avg Win:          ${summary['avg_win']:.2f}")
    print(f"  Avg Loss:         ${summary['avg_loss']:.2f}")

    print(f"\n  Initial Bankroll: ${summary['params']['initial_bankroll']:.2f}")
    print(f"  Final Bankroll:   ${summary['final_bankroll']:.2f}")
    print(f"  ROI:              {summary['roi_pct']:+.1f}%")
    print(f"  Max Drawdown:     {summary['max_drawdown']:.1%}")
    print(f"  Annualized Sharpe:{summary['sharpe']:.2f}")

    print(f"\n  --- Strategy Breakdown ---")
    print(f"  LADDER:  {summary['ladder_trades']} trades | P&L ${summary['ladder_pnl']:+.2f} | WR {summary['ladder_win_rate']:.1%}")
    print(f"  NO:      {summary['no_trades']} trades | P&L ${summary['no_pnl']:+.2f} | WR {summary['no_win_rate']:.1%}")

    print(f"\n  --- By City ---")
    for city in sorted(summary['city_pnl'].keys(), key=lambda c: summary['city_pnl'][c], reverse=True):
        n = summary['city_trades'][city]
        pnl = summary['city_pnl'][city]
        print(f"  {city:12s}: {n:3d} trades | P&L ${pnl:+.2f}")

    print(f"\n  Winning Days: {summary['winning_days']}  |  Losing Days: {summary['losing_days']}")

    # Show worst/best trades
    if trades:
        sorted_trades = sorted(trades, key=lambda t: t.pnl)
        print(f"\n  --- Worst 5 Trades ---")
        for t in sorted_trades[:5]:
            print(f"  {t.date} {t.station:8s} {t.direction:7s} {t.outcome_name:25s} entry={t.entry_price:.3f} size=${t.size_usd:.2f} P&L=${t.pnl:+.2f}")
        print(f"\n  --- Best 5 Trades ---")
        for t in sorted_trades[-5:]:
            print(f"  {t.date} {t.station:8s} {t.direction:7s} {t.outcome_name:25s} entry={t.entry_price:.3f} size=${t.size_usd:.2f} P&L=${t.pnl:+.2f}")

    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

def optimize_strategy(markets: List[dict]) -> Tuple[StrategyParams, dict]:
    """Grid search over key parameters to find optimal strategy."""
    best_params = None
    best_summary = None
    best_metric = float('-inf')
    all_results = []

    # Parameter grid
    ladder_max_entries = [0.12, 0.15, 0.20, 0.25]
    ladder_buckets_opts = [2, 3, 4]
    no_min_edges = [0.08, 0.10, 0.12, 0.15, 0.18]
    no_min_entries = [0.50, 0.55, 0.60, 0.65]
    no_max_entries = [0.80, 0.85, 0.90]
    entry_timings = ["early", "mid", "late"]
    kelly_fracs = [0.10, 0.15, 0.20]
    no_min_confs = [0.35, 0.40, 0.45, 0.50]

    # Quick scan: test entry timing + ladder max entry + NO min edge
    print("\n=== Phase 1: Scanning entry timing & thresholds ===")
    for timing in entry_timings:
        for lme in ladder_max_entries:
            for nme in no_min_edges:
                p = StrategyParams(
                    entry_timing=timing,
                    ladder_max_entry=lme,
                    no_min_edge=nme,
                )
                _, s = run_backtest(markets, p)
                if "error" in s:
                    continue

                # Metric: weighted combination of profit factor, win rate, total PnL, drawdown
                # We want: high PF, high WR, positive PnL, low drawdown
                metric = (
                    s["total_pnl"] * 0.4 +
                    s["profit_factor"] * 5.0 +
                    s["win_rate"] * 20.0 -
                    s["max_drawdown"] * 30.0 +
                    min(s["sharpe"], 5) * 2.0
                )

                all_results.append({
                    "timing": timing, "lme": lme, "nme": nme,
                    "pnl": s["total_pnl"], "pf": s["profit_factor"],
                    "wr": s["win_rate"], "dd": s["max_drawdown"],
                    "trades": s["total_trades"], "metric": metric,
                })

                if metric > best_metric and s["total_trades"] >= 5:
                    best_metric = metric
                    best_params = StrategyParams(
                        entry_timing=timing,
                        ladder_max_entry=lme,
                        no_min_edge=nme,
                    )
                    best_summary = s

    # Report phase 1
    all_results.sort(key=lambda r: r["metric"], reverse=True)
    print(f"\nTop 10 Phase 1 configs:")
    print(f"  {'Timing':6s} {'LME':5s} {'NME':5s} | {'Trades':6s} {'PnL':8s} {'PF':6s} {'WR':6s} {'DD':6s} {'Metric':8s}")
    for r in all_results[:10]:
        print(f"  {r['timing']:6s} {r['lme']:5.2f} {r['nme']:5.2f} | {r['trades']:6d} ${r['pnl']:+7.2f} {r['pf']:6.2f} {r['wr']:5.1%} {r['dd']:5.1%} {r['metric']:+8.2f}")

    if not best_params:
        return StrategyParams(), {"error": "No profitable config found"}

    # Phase 2: Fine-tune around best params
    print(f"\n=== Phase 2: Fine-tuning around best config ===")
    print(f"  Base: timing={best_params.entry_timing}, lme={best_params.ladder_max_entry}, nme={best_params.no_min_edge}")

    phase2_results = []
    for lb in ladder_buckets_opts:
        for nmin in no_min_entries:
            for nmax in no_max_entries:
                for kf in kelly_fracs:
                    for nconf in no_min_confs:
                        p = StrategyParams(
                            entry_timing=best_params.entry_timing,
                            ladder_max_entry=best_params.ladder_max_entry,
                            no_min_edge=best_params.no_min_edge,
                            ladder_buckets=lb,
                            no_min_entry=nmin,
                            no_max_entry=nmax,
                            kelly_fraction=kf,
                            no_min_confidence=nconf,
                        )
                        _, s = run_backtest(markets, p)
                        if "error" in s:
                            continue

                        metric = (
                            s["total_pnl"] * 0.4 +
                            s["profit_factor"] * 5.0 +
                            s["win_rate"] * 20.0 -
                            s["max_drawdown"] * 30.0 +
                            min(s["sharpe"], 5) * 2.0
                        )

                        phase2_results.append({
                            "lb": lb, "nmin": nmin, "nmax": nmax,
                            "kf": kf, "nconf": nconf,
                            "pnl": s["total_pnl"], "pf": s["profit_factor"],
                            "wr": s["win_rate"], "dd": s["max_drawdown"],
                            "trades": s["total_trades"], "metric": metric,
                            "params": p, "summary": s,
                        })

    phase2_results.sort(key=lambda r: r["metric"], reverse=True)
    print(f"\nTop 10 Phase 2 configs:")
    print(f"  {'Bkts':4s} {'NMin':5s} {'NMax':5s} {'KF':5s} {'NConf':5s} | {'Trades':6s} {'PnL':8s} {'PF':6s} {'WR':6s} {'DD':6s}")
    for r in phase2_results[:10]:
        print(f"  {r['lb']:4d} {r['nmin']:5.2f} {r['nmax']:5.2f} {r['kf']:5.2f} {r['nconf']:5.2f} | {r['trades']:6d} ${r['pnl']:+7.2f} {r['pf']:6.2f} {r['wr']:5.1%} {r['dd']:5.1%}")

    if phase2_results:
        # Pick best from phase 2 (must have at least 5 trades and positive PnL)
        for r in phase2_results:
            if r["trades"] >= 5 and r["pnl"] > 0:
                best_params = r["params"]
                best_summary = r["summary"]
                best_metric = r["metric"]
                break

    return best_params, best_summary


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    cities = list(STATIONS.keys())

    # ─── Step 1: Collect real data ───
    data_file = "data/real_markets.json"
    if os.path.exists(data_file) and "--no-cache" not in sys.argv:
        print(f"Loading cached data from {data_file}...")
        with open(data_file) as f:
            markets = json.load(f)
        print(f"Loaded {len(markets)} markets")
    else:
        print(f"Collecting real data: {len(cities)} cities, {days} days back...")
        markets = collect_all_data(cities, days_back=days)

        # Save
        os.makedirs("data", exist_ok=True)
        with open(data_file, "w") as f:
            json.dump(markets, f, indent=2, default=str)
        print(f"Saved {len(markets)} markets to {data_file}")

    # Filter: only markets with forecasts and resolutions
    valid = [m for m in markets if m.get("forecasts") and m.get("winning_bucket")]
    print(f"\nValid markets (with forecasts + resolution): {len(valid)}")

    if not valid:
        print("ERROR: No valid markets found. Check data collection.")
        sys.exit(1)

    # ─── Step 2: Run initial backtest with current params ───
    print("\n" + "="*70)
    print("  PHASE 1: INITIAL BACKTEST (current V6 params)")
    print("="*70)
    initial_params = StrategyParams()
    trades_initial, summary_initial = run_backtest(valid, initial_params)
    print_report(trades_initial, summary_initial, "(INITIAL - V6 DEFAULTS)")

    # ─── Step 3: Optimize ───
    print("\n" + "="*70)
    print("  PHASE 2: PARAMETER OPTIMIZATION")
    print("="*70)
    best_params, best_summary = optimize_strategy(valid)

    # ─── Step 4: Run final backtest with optimized params ───
    print("\n" + "="*70)
    print("  PHASE 3: OPTIMIZED BACKTEST")
    print("="*70)
    trades_opt, summary_opt = run_backtest(valid, best_params)
    print_report(trades_opt, summary_opt, "(OPTIMIZED)")

    # ─── Step 5: Save results ───
    results = {
        "initial": {
            "summary": summary_initial,
            "n_trades": len(trades_initial),
        },
        "optimized": {
            "summary": summary_opt,
            "n_trades": len(trades_opt),
            "params": asdict(best_params),
        },
        "data_stats": {
            "total_markets": len(markets),
            "valid_markets": len(valid),
            "cities": cities,
            "days_back": days,
        },
    }

    with open("data/backtest_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to data/backtest_results.json")
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"  Initial P&L: ${summary_initial.get('total_pnl', 0):+.2f}  →  Optimized P&L: ${summary_opt.get('total_pnl', 0):+.2f}")
    print(f"  Initial WR:  {summary_initial.get('win_rate', 0):.1%}   →  Optimized WR:  {summary_opt.get('win_rate', 0):.1%}")
    print(f"  Initial PF:  {summary_initial.get('profit_factor', 0):.2f}  →  Optimized PF:  {summary_opt.get('profit_factor', 0):.2f}")
    print(f"{'='*70}")
