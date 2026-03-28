"""
Fast data collector — collects market data in parallel-ish fashion with minimal delays.
Saves incrementally so we don't lose progress on timeout.
"""
import os, json, re, sys, time
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def api_get(url, params, timeout=12, retries=2):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            return resp
        except:
            if attempt < retries - 1: time.sleep(1)
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


def collect_one_market(sid, info, target_date_str, date_obj):
    """Collect a single market. Returns dict or None."""
    month = date_obj.strftime("%B").lower()
    day, year = date_obj.day, date_obj.year
    slug = f"highest-temperature-in-{info['slug']}-on-{month}-{day}-{year}"
    
    try:
        resp = api_get(f"{GAMMA_API}/events", {"slug": slug})
        if not resp or resp.status_code != 200 or not resp.json():
            return None
        ev = resp.json()[0]
        subs = ev.get("markets", [])
        if not subs: return None
        
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
            ep = compute_entry_prices(ph)
            iw = yf > 0.5
            if iw: winner = name
            outcomes.append({
                "name": name, "final_price": yf, "is_winner": iw,
                "entry_early": ep["early"], "entry_mid": ep["mid"],
                "entry_late": ep["late"], "entry_sniper": ep["sniper"],
                "n_price_points": len(ph),
            })
        
        fc_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
        forecasts = fetch_forecasts(info["lat"], info["lon"], target_date_str, fc_date, info["unit"])
        actual = fetch_actual(info["lat"], info["lon"], target_date_str, info["unit"])
        
        fm = fs = fmed = None
        if forecasts:
            temps = list(forecasts.values())
            fm, fs = float(np.mean(temps)), float(np.std(temps)) if len(temps) > 1 else 3.0
            fmed = float(np.median(temps))
        
        return {
            "slug": slug, "station_id": sid, "target_date": target_date_str,
            "title": ev.get("title",""), "winning_bucket": winner,
            "n_outcomes": len(outcomes), "outcomes": outcomes,
            "forecasts": forecasts, "actual_temp": actual,
            "forecast_mean": fm, "forecast_std": fs, "forecast_median": fmed,
        }
    except Exception as e:
        return None


def main():
    data_file = "data/real_markets_v2.json"
    os.makedirs("data", exist_ok=True)
    
    # Load existing data if any
    existing = []
    existing_keys = set()
    if os.path.exists(data_file):
        with open(data_file) as f:
            existing = json.load(f)
        for m in existing:
            existing_keys.add(f"{m['station_id']}_{m['target_date']}")
        print(f"Loaded {len(existing)} existing markets")
    
    # Build task list
    tasks = []
    for days_ago in range(3, 21):  # 18 days of data
        date = datetime.now() - timedelta(days=days_ago)
        target = date.strftime("%Y-%m-%d")
        for sid, info in STATIONS.items():
            key = f"{sid}_{target}"
            if key not in existing_keys:
                tasks.append((sid, info, target, date))
    
    print(f"Need to collect: {len(tasks)} markets")
    
    all_markets = list(existing)
    collected = 0
    
    # Collect with thread pool (3 threads to avoid rate limits)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for sid, info, target, date in tasks:
            f = executor.submit(collect_one_market, sid, info, target, date)
            futures[f] = (sid, target)
        
        for future in as_completed(futures):
            sid, target = futures[future]
            try:
                result = future.result(timeout=30)
                if result:
                    all_markets.append(result)
                    collected += 1
                    w = result.get("winning_bucket") or "unresolved"
                    print(f"  [{collected}] {target} {sid:8s} | {result['n_outcomes']} oc | {w}")
                    
                    # Save every 10 markets
                    if collected % 10 == 0:
                        with open(data_file, "w") as f:
                            json.dump(all_markets, f, indent=2, default=str)
                        print(f"  ... saved ({len(all_markets)} total)")
            except Exception as e:
                pass
    
    # Final save
    with open(data_file, "w") as f:
        json.dump(all_markets, f, indent=2, default=str)
    
    valid = [m for m in all_markets if m.get("forecasts") and m.get("winning_bucket")]
    print(f"\nDone! Total: {len(all_markets)} markets, {len(valid)} valid (with forecasts + winner)")


if __name__ == "__main__":
    main()
