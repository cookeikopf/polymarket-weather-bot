"""
Real-Data Backtester V2 — Polymarket Weather Bot V6.2
======================================================
Includes all 3 strategies:
  1. LADDER (BUY YES near median at early prices)
  2. CONSERVATIVE NO (BUY NO on unlikely outcomes)
  3. LATE SNIPER (BUY YES/NO using late prices when forecast confidence is high)

Uses REAL Polymarket prices + REAL weather forecasts.
"""

import os, json, re, sys, time
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy import stats as sp_stats
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
OPEN_METEO_API_KEY = os.getenv("OPEN_METEO_API_KEY", "wjrcKzLOeLkcCnzx")
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
_P = "customer-" if OPEN_METEO_API_KEY else ""
HIST_FORECAST_URL = f"https://{_P}historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = f"https://{_P}archive-api.open-meteo.com/v1/archive"
HIST_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "best_match"]

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


@dataclass
class StrategyParams:
    # Ladder
    ladder_enabled: bool = True
    ladder_max_entry: float = 0.25
    ladder_buckets: int = 2
    ladder_bet_per_bucket: float = 2.0
    # Conservative NO
    no_enabled: bool = True
    no_min_entry: float = 0.65
    no_max_entry: float = 0.90
    no_min_edge: float = 0.12
    # Late Sniper
    sniper_enabled: bool = True
    sniper_min_edge: float = 0.10         # Min edge for sniper trades
    sniper_max_yes_entry: float = 0.35    # Max price to buy YES
    sniper_min_no_entry: float = 0.60     # Min NO entry for sniper
    sniper_max_no_entry: float = 0.92     # Max NO entry for sniper
    sniper_confidence_mult: float = 1.5   # Size multiplier (we're more confident late)
    sniper_bet_size: float = 3.0          # Base bet per sniper trade
    sniper_max_bets: int = 3              # Max sniper trades per market
    # General
    kelly_fraction: float = 0.20
    max_position_pct: float = 0.15
    min_trade_usd: float = 5.0
    max_trade_usd: float = 10.0
    max_exposure: float = 0.60
    initial_bankroll: float = 100.0
    entry_timing: str = "early"
    min_models: int = 2
    no_min_confidence: float = 0.35


@dataclass
class Trade:
    date: str
    station: str
    outcome_name: str
    direction: str
    strategy: str
    entry_price: float
    size_usd: float
    our_prob: float
    edge: float
    confidence: float
    is_winner: bool
    pnl: float = 0.0
    roi: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# DATA COLLECTION (same as V1 but also stores full price history timing)
# ═══════════════════════════════════════════════════════════════════

def api_get(url, params, timeout=15, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(min(60, 5 * (2 ** attempt)))
                continue
            return resp
        except:
            if attempt < retries - 1: time.sleep(2 * (attempt + 1))
    return None


def extract_bucket_name(q, unit):
    m = re.search(r'between\s+(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*F', q)
    if m: return f"{m.group(1)}-{m.group(2)}°F"
    m = re.search(r'be\s+(-?\d+)\s*°\s*C', q)
    if m: return f"{m.group(1)}°C"
    m = re.search(r'(-?\d+)\s*°\s*([FC])\s+or\s+higher', q)
    if m: return f"{m.group(1)}°{m.group(2)} or higher"
    m = re.search(r'(-?\d+)\s*°\s*([FC])\s+or\s+(?:below|lower)', q)
    if m: return f"{m.group(1)}°{m.group(2)} or below"
    m = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*F', q)
    if m: return f"{m.group(1)}-{m.group(2)}°F"
    m = re.search(r'(-?\d+)\s*°\s*C', q)
    if m: return f"{m.group(1)}°C"
    return q[:40]


def compute_entry_prices(history):
    if not history: return {"early": 0, "mid": 0, "late": 0, "sniper": 0}
    prices = [float(h.get("p", 0)) for h in history if h.get("p")]
    if not prices: return {"early": 0, "mid": 0, "late": 0, "sniper": 0}
    n = len(prices)
    return {
        "early": prices[min(2, n-1)],
        "mid": prices[n // 2],
        "late": prices[max(0, n - max(1, n // 4))],
        # Sniper: price ~6h before resolution (last ~25% of price history)
        "sniper": prices[max(0, int(n * 0.80))],
    }


def fetch_forecasts(lat, lon, target_date, forecast_date, unit):
    tu = "fahrenheit" if unit == "fahrenheit" else "celsius"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max", "temperature_unit": tu,
        "timezone": "auto", "start_date": forecast_date, "end_date": target_date,
        "models": ",".join(HIST_MODELS),
    }
    if OPEN_METEO_API_KEY: params["apikey"] = OPEN_METEO_API_KEY
    resp = api_get(HIST_FORECAST_URL, params)
    if not resp or resp.status_code != 200: return {}
    daily = resp.json().get("daily", {})
    times = daily.get("time", [])
    ti = None
    for i, t in enumerate(times):
        if t == target_date: ti = i; break
    if ti is None: ti = len(times) - 1 if times else None
    if ti is None: return {}
    results = {}
    for mdl in HIST_MODELS:
        k = f"temperature_2m_max_{mdl}"
        v = daily.get(k, [])
        if ti < len(v) and v[ti] is not None:
            results[mdl] = float(v[ti])
    return results


def fetch_actual(lat, lon, date, unit):
    tu = "fahrenheit" if unit == "fahrenheit" else "celsius"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max", "temperature_unit": tu,
        "timezone": "auto", "start_date": date, "end_date": date,
    }
    if OPEN_METEO_API_KEY: params["apikey"] = OPEN_METEO_API_KEY
    resp = api_get(ARCHIVE_URL, params)
    if not resp or resp.status_code != 200: return None
    v = resp.json().get("daily", {}).get("temperature_2m_max", [])
    return float(v[0]) if v and v[0] is not None else None


def collect_data(cities, days_back=18):
    all_markets = []
    found = 0
    for days_ago in range(3, days_back + 1):
        date = datetime.now() - timedelta(days=days_ago)
        target = date.strftime("%Y-%m-%d")
        month = date.strftime("%B").lower()
        day, year = date.day, date.year
        for sid, info in [(s, STATIONS[s]) for s in cities]:
            slug = f"highest-temperature-in-{info['slug']}-on-{month}-{day}-{year}"
            try:
                resp = api_get(f"{GAMMA_API}/events", {"slug": slug})
                if not resp or resp.status_code != 200 or not resp.json():
                    time.sleep(0.08); continue
                ev = resp.json()[0]
                subs = ev.get("markets", [])
                if not subs: continue
                outcomes, winner = [], None
                for m in subs:
                    q = m.get("question", "")
                    tokens = json.loads(m.get("clobTokenIds", "[]"))
                    prices = json.loads(m.get("outcomePrices", "[]"))
                    yt = tokens[0] if tokens else ""
                    yf = float(prices[0]) if prices else 0
                    name = extract_bucket_name(q, info["unit"])
                    ph = []
                    if yt:
                        hr = api_get(f"{CLOB_API}/prices-history",
                            {"market": yt, "interval": "max", "fidelity": 60})
                        if hr and hr.status_code == 200:
                            ph = hr.json().get("history", [])
                        time.sleep(0.08)
                    ep = compute_entry_prices(ph)
                    iw = yf > 0.5
                    if iw: winner = name
                    outcomes.append({
                        "name": name, "final_price": yf, "is_winner": iw,
                        "entry_early": ep["early"], "entry_mid": ep["mid"],
                        "entry_late": ep["late"], "entry_sniper": ep["sniper"],
                        "n_price_points": len(ph),
                    })
                fc_date = (date - timedelta(days=1)).strftime("%Y-%m-%d")
                forecasts = fetch_forecasts(info["lat"], info["lon"], target, fc_date, info["unit"])
                time.sleep(0.15)
                actual = fetch_actual(info["lat"], info["lon"], target, info["unit"])
                time.sleep(0.1)
                fm = fs = fmed = None
                if forecasts:
                    temps = list(forecasts.values())
                    fm, fs = float(np.mean(temps)), float(np.std(temps)) if len(temps) > 1 else 3.0
                    fmed = float(np.median(temps))
                rec = {
                    "slug": slug, "station_id": sid, "target_date": target,
                    "title": ev.get("title",""), "winning_bucket": winner,
                    "n_outcomes": len(outcomes), "outcomes": outcomes,
                    "forecasts": forecasts, "actual_temp": actual,
                    "forecast_mean": fm, "forecast_std": fs, "forecast_median": fmed,
                }
                all_markets.append(rec)
                found += 1
                w = winner or "unresolved"
                print(f"  [{found}] {target} {sid:8s} | {len(outcomes)} oc | {w}")
            except Exception as e:
                continue
    return all_markets


# ═══════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════

def extract_bucket_temp(name, is_f):
    m = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F', name)
    if m: return (int(m.group(1)) + int(m.group(2))) / 2.0
    m = re.search(r'(-?\d+)\s*°\s*C', name)
    if m: return float(m.group(1))
    m = re.search(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*below', name, re.I)
    if m: return float(m.group(1)) - (1 if is_f else 0.5)
    m = re.search(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*higher', name, re.I)
    if m: return float(m.group(1)) + (1 if is_f else 0.5)
    return None


def compute_probs(forecasts, outcomes, is_f):
    if not forecasts: return {}
    samples = np.array(list(forecasts.values()))
    mean = np.mean(samples) if len(samples) > 1 else samples[0]
    std = max(np.std(samples), 0.5) if len(samples) > 1 else (3.0 if is_f else 1.5)
    probs = {}
    for o in outcomes:
        name = o["name"]
        temp = extract_bucket_temp(name, is_f)
        if temp is None: probs[name] = 0.01; continue
        is_low = bool(re.search(r'or\s+below', name, re.I))
        is_high = bool(re.search(r'or\s+higher', name, re.I))
        if is_low:
            prob = sp_stats.norm.cdf(temp + 0.5, loc=mean, scale=max(std, 1.0))
        elif is_high:
            prob = 1.0 - sp_stats.norm.cdf(temp - 0.5, loc=mean, scale=max(std, 1.0))
        else:
            if is_f:
                m2 = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)', name)
                low, high = (int(m2.group(1)), int(m2.group(2)) + 1) if m2 else (temp-1, temp+1)
            else:
                low, high = temp, temp + 1
            prob = sp_stats.norm.cdf(high, loc=mean, scale=max(std, 1.0)) - \
                   sp_stats.norm.cdf(low, loc=mean, scale=max(std, 1.0))
        probs[name] = max(prob, 0.001)
    total = sum(probs.values())
    if total > 0: probs = {k: v/total for k, v in probs.items()}
    return probs


def compute_sniper_probs(forecasts, actual_temp, outcomes, is_f):
    """Late sniper uses narrower distribution — forecast is more certain close to resolution.
    We simulate this by using a tighter std (based on actual MAE ~1°)."""
    if not forecasts: return {}
    samples = np.array(list(forecasts.values()))
    mean = np.mean(samples) if len(samples) > 1 else samples[0]
    # Key insight: close to resolution, forecast error is ~1° for most cities
    # Use tighter std for more concentrated probability
    std = 1.2 if is_f else 0.7  # Much tighter than early forecast
    
    probs = {}
    for o in outcomes:
        name = o["name"]
        temp = extract_bucket_temp(name, is_f)
        if temp is None: probs[name] = 0.01; continue
        is_low = bool(re.search(r'or\s+below', name, re.I))
        is_high = bool(re.search(r'or\s+higher', name, re.I))
        if is_low:
            prob = sp_stats.norm.cdf(temp + 0.5, loc=mean, scale=std)
        elif is_high:
            prob = 1.0 - sp_stats.norm.cdf(temp - 0.5, loc=mean, scale=std)
        else:
            if is_f:
                m2 = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)', name)
                low, high = (int(m2.group(1)), int(m2.group(2)) + 1) if m2 else (temp-1, temp+1)
            else:
                low, high = temp, temp + 1
            prob = sp_stats.norm.cdf(high, loc=mean, scale=std) - \
                   sp_stats.norm.cdf(low, loc=mean, scale=std)
        probs[name] = max(prob, 0.001)
    total = sum(probs.values())
    if total > 0: probs = {k: v/total for k, v in probs.items()}
    return probs


def run_backtest(markets, params):
    trades = []
    bankroll = params.initial_bankroll
    peak = bankroll
    max_dd = 0
    daily_pnl = defaultdict(float)
    exposure = 0.0

    for market in markets:
        if not market.get("forecasts") or not market.get("winning_bucket"): continue
        if market.get("forecast_mean") is None: continue
        sid = market["station_id"]
        is_f = STATIONS[sid]["unit"] == "fahrenheit"
        td = market["target_date"]
        forecasts = market["forecasts"]
        outcomes = market["outcomes"]
        if len(forecasts) < params.min_models: continue

        our_probs = compute_probs(forecasts, outcomes, is_f)
        if not our_probs: continue
        fmean = market["forecast_mean"]
        fstd = market.get("forecast_std", 3.0)
        fmed = market.get("forecast_median", fmean)

        # ─── Strategy 1: LADDER ───
        if params.ladder_enabled:
            cands = []
            for o in outcomes:
                name = o["name"]
                prob = our_probs.get(name, 0)
                if prob < 0.01: continue
                entry = o.get(f"entry_{params.entry_timing}", 0)
                if entry <= 0.005 or entry > params.ladder_max_entry: continue
                bt = extract_bucket_temp(name, is_f)
                if bt is None: continue
                dist = abs(bt - fmed)
                cands.append((o, prob, entry, bt, dist, name))
            cands.sort(key=lambda c: c[4])
            for o, prob, entry, temp, dist, name in cands[:params.ladder_buckets]:
                if entry <= 0 or entry >= 1: continue
                is_w = o.get("is_winner", False)
                size = min(params.ladder_bet_per_bucket, bankroll * params.max_exposure - exposure)
                if size < 1.0: continue
                pnl = size * (1.0/entry - 1) if is_w else -size
                edge = prob - entry
                trades.append(Trade(td, sid, name, "BUY_YES", "ladder", entry, size,
                                   prob, edge, 0.5, is_w, pnl, pnl/size if size else 0))
                bankroll += pnl; daily_pnl[td] += pnl
                peak = max(peak, bankroll)
                dd = (peak - bankroll)/peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

        # ─── Strategy 2: CONSERVATIVE NO ───
        if params.no_enabled:
            for o in outcomes:
                name = o["name"]
                prob_yes = our_probs.get(name, 0)
                prob_no = 1.0 - prob_yes
                entry_yes = o.get(f"entry_{params.entry_timing}", 0)
                if entry_yes <= 0: continue
                entry_no = 1.0 - entry_yes
                if entry_no < params.no_min_entry or entry_no > params.no_max_entry: continue
                edge = prob_no - entry_no
                if edge < params.no_min_edge: continue
                # Confidence
                model_std = fstd
                prob_unc = min(0.30, model_std * 0.03)
                edge_sig = min(1.0, edge / max(prob_unc, 0.01))
                n_m = len(forecasts)
                agr = max(0, min(1, 1.0 - (fstd / 10.0)))
                conf = 0.4 * agr + 0.35 * edge_sig + 0.25 * min(1.0, n_m / 5)
                if conf < params.no_min_confidence: continue
                # Kelly
                if 0 < entry_no < 1:
                    b = (1.0 - entry_no) / entry_no
                    kelly = max(0, (b * prob_no - (1 - prob_no)) / b)
                    kelly = min(kelly * params.kelly_fraction, params.max_position_pct)
                else: kelly = 0
                size = min(kelly * bankroll * conf, params.max_trade_usd)
                rem = bankroll * params.max_exposure - exposure
                size = min(size, rem)
                if size < params.min_trade_usd:
                    size = params.min_trade_usd if params.min_trade_usd <= rem else 0
                if size <= 0: continue
                is_w_no = not o.get("is_winner", False)
                pnl = size * ((1.0 - entry_no) / entry_no) if is_w_no else -size
                trades.append(Trade(td, sid, name, "BUY_NO", "conservative_no", entry_no, size,
                                   prob_no, edge, conf, is_w_no, pnl, pnl/size if size else 0))
                bankroll += pnl; daily_pnl[td] += pnl
                peak = max(peak, bankroll)
                dd = (peak - bankroll)/peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

        # ─── Strategy 3: LATE SNIPER ───
        if params.sniper_enabled:
            sniper_probs = compute_sniper_probs(forecasts, market.get("actual_temp"), outcomes, is_f)
            if not sniper_probs: continue
            sniper_count = 0

            # 3a: Sniper BUY YES — forecast confident, late price still cheap
            best_sniper_yes = []
            for o in outcomes:
                name = o["name"]
                prob = sniper_probs.get(name, 0)
                sniper_entry = o.get("entry_sniper", 0)
                if sniper_entry <= 0.005 or sniper_entry > params.sniper_max_yes_entry: continue
                if prob < 0.10: continue
                edge = prob - sniper_entry
                if edge < params.sniper_min_edge: continue
                bt = extract_bucket_temp(name, is_f)
                if bt is None: continue
                ev_per_dollar = (prob * (1/sniper_entry - 1) - (1-prob)) if sniper_entry > 0 else 0
                best_sniper_yes.append((o, prob, sniper_entry, edge, ev_per_dollar, name))

            best_sniper_yes.sort(key=lambda x: x[4], reverse=True)
            for o, prob, entry, edge, ev, name in best_sniper_yes[:params.sniper_max_bets]:
                if sniper_count >= params.sniper_max_bets: break
                size = min(params.sniper_bet_size * params.sniper_confidence_mult,
                          bankroll * params.max_exposure - exposure)
                if size < 1.0: continue
                is_w = o.get("is_winner", False)
                pnl = size * (1.0/entry - 1) if is_w else -size
                trades.append(Trade(td, sid, name, "BUY_YES", "late_sniper", entry, size,
                                   prob, edge, 0.8, is_w, pnl, pnl/size if size else 0))
                bankroll += pnl; daily_pnl[td] += pnl
                sniper_count += 1
                peak = max(peak, bankroll)
                dd = (peak - bankroll)/peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            # 3b: Sniper BUY NO — very confident the outcome is wrong, late price mispriced
            for o in outcomes:
                if sniper_count >= params.sniper_max_bets: break
                name = o["name"]
                prob_yes = sniper_probs.get(name, 0)
                prob_no = 1.0 - prob_yes
                sniper_entry_yes = o.get("entry_sniper", 0)
                if sniper_entry_yes <= 0: continue
                entry_no = 1.0 - sniper_entry_yes
                if entry_no < params.sniper_min_no_entry or entry_no > params.sniper_max_no_entry: continue
                edge = prob_no - entry_no
                if edge < params.sniper_min_edge: continue
                size = min(params.sniper_bet_size * params.sniper_confidence_mult,
                          bankroll * params.max_exposure - exposure)
                if size < params.min_trade_usd: continue
                is_w_no = not o.get("is_winner", False)
                pnl = size * ((1.0 - entry_no) / entry_no) if is_w_no else -size
                trades.append(Trade(td, sid, name, "BUY_NO", "late_sniper", entry_no, size,
                                   prob_no, edge, 0.8, is_w_no, pnl, pnl/size if size else 0))
                bankroll += pnl; daily_pnl[td] += pnl
                sniper_count += 1
                peak = max(peak, bankroll)
                dd = (peak - bankroll)/peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

    if not trades: return trades, {"error": "No trades"}

    # Summary
    total = len(trades)
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    wr = len(winners) / total
    pf = gp / gl if gl > 0 else float('inf')
    dv = list(daily_pnl.values())
    
    strats = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
    for t in trades:
        strats[t.strategy]["trades"] += 1
        strats[t.strategy]["pnl"] += t.pnl
        if t.pnl > 0: strats[t.strategy]["wins"] += 1

    city_pnl = defaultdict(float)
    city_n = defaultdict(int)
    for t in trades:
        city_pnl[t.station] += t.pnl
        city_n[t.station] += 1

    return trades, {
        "total_trades": total, "winners": len(winners), "losers": len(losers),
        "win_rate": wr, "total_pnl": total_pnl, "gross_profit": gp, "gross_loss": gl,
        "profit_factor": pf,
        "avg_win": np.mean([t.pnl for t in winners]) if winners else 0,
        "avg_loss": np.mean([t.pnl for t in losers]) if losers else 0,
        "final_bankroll": bankroll, "roi_pct": (bankroll - params.initial_bankroll) / params.initial_bankroll * 100,
        "max_drawdown": max_dd, "peak_bankroll": peak,
        "sharpe": np.mean(dv) / np.std(dv) * np.sqrt(365) if dv and np.std(dv) > 0 else 0,
        "strategies": {k: {"trades": v["trades"], "pnl": v["pnl"],
                           "win_rate": v["wins"]/v["trades"] if v["trades"] else 0}
                      for k, v in strats.items()},
        "city_pnl": dict(city_pnl), "city_trades": dict(city_n),
        "winning_days": len([d for d in dv if d > 0]),
        "losing_days": len([d for d in dv if d <= 0]),
        "params": asdict(params),
    }


def print_report(trades, s, label=""):
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS {label}")
    print(f"{'='*70}")
    print(f"\n  Trades: {s['total_trades']} | Win: {s['winners']} ({s['win_rate']:.1%}) | Loss: {s['losers']}")
    print(f"  P&L: ${s['total_pnl']:+.2f} | GP: ${s['gross_profit']:.2f} | GL: ${s['gross_loss']:.2f}")
    print(f"  Profit Factor: {s['profit_factor']:.2f} | Avg Win: ${s['avg_win']:.2f} | Avg Loss: ${s['avg_loss']:.2f}")
    print(f"  Bankroll: ${s['params']['initial_bankroll']:.0f} → ${s['final_bankroll']:.2f} ({s['roi_pct']:+.1f}%)")
    print(f"  Max DD: {s['max_drawdown']:.1%} | Sharpe: {s['sharpe']:.1f}")
    print(f"\n  --- Strategies ---")
    for k, v in s["strategies"].items():
        print(f"  {k:20s}: {v['trades']:3d} trades | ${v['pnl']:+8.2f} | WR {v['win_rate']:.1%}")
    print(f"\n  --- Cities ---")
    for c in sorted(s["city_pnl"].keys(), key=lambda x: s["city_pnl"][x], reverse=True):
        print(f"  {c:12s}: {s['city_trades'][c]:3d} trades | ${s['city_pnl'][c]:+.2f}")
    print(f"\n  Days: {s['winning_days']} winning | {s['losing_days']} losing")
    if trades:
        st = sorted(trades, key=lambda t: t.pnl)
        print(f"\n  Worst: {st[0].date} {st[0].station} {st[0].strategy} {st[0].outcome_name} ${st[0].pnl:+.2f}")
        print(f"  Best:  {st[-1].date} {st[-1].station} {st[-1].strategy} {st[-1].outcome_name} ${st[-1].pnl:+.2f}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

def optimize(markets):
    best_params = None
    best_metric = float('-inf')
    results = []

    # Phase 1: Sniper parameters — focused grid (432 combos vs 2160)
    print("\n=== OPTIMIZING LATE SNIPER PARAMS ===")
    tested = 0
    for s_edge in [0.06, 0.08, 0.10, 0.13]:
        for s_yes_max in [0.28, 0.35, 0.42]:
            for s_no_min in [0.55, 0.62, 0.68]:
                for s_no_max in [0.88, 0.93]:
                    for s_bet in [2.0, 3.5, 5.0]:
                        for s_max in [2, 3, 4]:
                            p = StrategyParams(
                                sniper_min_edge=s_edge,
                                sniper_max_yes_entry=s_yes_max,
                                sniper_min_no_entry=s_no_min,
                                sniper_max_no_entry=s_no_max,
                                sniper_bet_size=s_bet,
                                sniper_max_bets=s_max,
                            )
                            _, s = run_backtest(markets, p)
                            if "error" in s: continue
                            metric = (
                                s["total_pnl"] * 0.35 +
                                s["profit_factor"] * 5.0 +
                                s["win_rate"] * 20.0 -
                                s["max_drawdown"] * 40.0 +
                                min(s["sharpe"], 5) * 2.0
                            )
                            tested += 1
                            results.append({"metric": metric, "params": p, "summary": s,
                                           "s_edge": s_edge, "s_yes": s_yes_max,
                                           "s_no_min": s_no_min, "s_no_max": s_no_max,
                                           "s_bet": s_bet, "s_max": s_max})
                            if metric > best_metric and s["total_trades"] >= 10 and s["total_pnl"] > 0:
                                best_metric = metric
                                best_params = p
    print(f"  Tested {tested} configurations")

    results.sort(key=lambda r: r["metric"], reverse=True)
    print(f"\nTop 5 configs:")
    for r in results[:5]:
        s = r["summary"]
        print(f"  edge={r['s_edge']:.2f} yes={r['s_yes']:.2f} no={r['s_no_min']:.2f}-{r['s_no_max']:.2f} bet=${r['s_bet']:.0f} max={r['s_max']} "
              f"| {s['total_trades']} trades ${s['total_pnl']:+.0f} PF={s['profit_factor']:.2f} WR={s['win_rate']:.1%} DD={s['max_drawdown']:.1%}")

    return best_params or StrategyParams()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_file = "data/real_markets_v2.json"
    cities = list(STATIONS.keys())

    if os.path.exists(data_file) and "--no-cache" not in sys.argv:
        print(f"Loading cached data from {data_file}...")
        with open(data_file) as f: markets = json.load(f)
    else:
        print(f"Collecting data: {len(cities)} cities...")
        markets = collect_data(cities, days_back=18)
        os.makedirs("data", exist_ok=True)
        with open(data_file, "w") as f: json.dump(markets, f, indent=2, default=str)

    valid = [m for m in markets if m.get("forecasts") and m.get("winning_bucket")]
    print(f"Valid markets: {len(valid)}")

    # Phase 1: V6.1 baseline (no sniper)
    print("\n" + "="*70)
    print("  V6.1 BASELINE (Ladder + Conservative NO)")
    p1 = StrategyParams(sniper_enabled=False)
    t1, s1 = run_backtest(valid, p1)
    print_report(t1, s1, "V6.1 BASELINE")

    # Phase 2: V6.2 with sniper (default params)
    print("  V6.2 DEFAULT (All 3 strategies)")
    p2 = StrategyParams()
    t2, s2 = run_backtest(valid, p2)
    print_report(t2, s2, "V6.2 DEFAULT SNIPER")

    # Phase 3: Optimize sniper params
    print("  OPTIMIZING...")
    best_p = optimize(valid)
    t3, s3 = run_backtest(valid, best_p)
    print_report(t3, s3, "V6.2 OPTIMIZED")

    # Save
    res = {
        "v61_baseline": {"summary": s1},
        "v62_default": {"summary": s2},
        "v62_optimized": {"summary": s3, "params": asdict(best_p)},
        "data_stats": {"total": len(markets), "valid": len(valid)},
    }
    with open("data/backtest_v2_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  COMPARISON")
    print(f"  V6.1: ${s1['total_pnl']:+.0f} | PF {s1['profit_factor']:.2f} | WR {s1['win_rate']:.1%} | DD {s1['max_drawdown']:.1%}")
    print(f"  V6.2: ${s3['total_pnl']:+.0f} | PF {s3['profit_factor']:.2f} | WR {s3['win_rate']:.1%} | DD {s3['max_drawdown']:.1%}")
    print(f"{'='*70}")
