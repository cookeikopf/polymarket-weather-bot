"""
Collect real Polymarket weather market data + Open-Meteo forecasts for backtesting.
Saves complete dataset to data/real_markets.json
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config as cfg
from utils import log

DATA_DIR = "data"


def collect_market_data(cities: List[str] = None, days_back: int = 30,
                        api_delay: float = 0.15) -> List[dict]:
    """Collect resolved weather market data from Polymarket Gamma + CLOB APIs."""
    if cities is None:
        cities = list(cfg.STATIONS.keys())

    city_slugs = {sid: cfg.STATIONS[sid].get("slug", sid.lower()) for sid in cities}
    all_markets = []

    for days_ago in range(3, days_back + 1):
        date = datetime.now() - timedelta(days=days_ago)
        target_date = date.strftime("%Y-%m-%d")
        month = date.strftime("%B").lower()
        day = date.day
        year = date.year

        for station_id in cities:
            slug_name = city_slugs[station_id]
            slug = f"highest-temperature-in-{slug_name}-on-{month}-{day}-{year}"

            try:
                resp = requests.get("https://gamma-api.polymarket.com/events",
                                    params={"slug": slug}, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if not data or not isinstance(data, list) or len(data) == 0:
                    continue

                ev = data[0]
                sub_markets = ev.get("markets", [])
                if not sub_markets:
                    continue

                # Parse all outcomes
                outcomes = []
                winning_bucket = None

                for m in sub_markets:
                    q = m.get("question", "")
                    tokens = json.loads(m.get("clobTokenIds", "[]"))
                    prices = json.loads(m.get("outcomePrices", "[]"))

                    yes_token = tokens[0] if tokens else ""
                    no_token = tokens[1] if len(tokens) > 1 else ""
                    yes_final = float(prices[0]) if prices else 0

                    # Extract bucket name
                    import re
                    name = _extract_bucket_name(q, station_id)

                    # Get price history from CLOB
                    price_history = []
                    if yes_token:
                        try:
                            hr = requests.get("https://clob.polymarket.com/prices-history",
                                              params={"market": yes_token, "interval": "max",
                                                       "fidelity": 60},
                                              timeout=10)
                            if hr.status_code == 200:
                                price_history = hr.json().get("history", [])
                            time.sleep(api_delay)
                        except Exception:
                            pass

                    # Determine entry price (earliest available or ~24h before)
                    entry_prices = _compute_entry_prices(price_history)

                    is_winner = yes_final > 0.5
                    if is_winner:
                        winning_bucket = name

                    outcomes.append({
                        "name": name,
                        "yes_token": yes_token,
                        "no_token": no_token,
                        "final_price": yes_final,
                        "is_winner": is_winner,
                        "price_history": price_history,
                        "entry_early": entry_prices.get("early", 0),
                        "entry_mid": entry_prices.get("mid", 0),
                        "entry_late": entry_prices.get("late", 0),
                    })

                market_record = {
                    "slug": slug,
                    "station_id": station_id,
                    "target_date": target_date,
                    "title": ev.get("title", ""),
                    "winning_bucket": winning_bucket,
                    "n_outcomes": len(outcomes),
                    "outcomes": outcomes,
                }
                all_markets.append(market_record)
                log.info(f"  {target_date} {station_id:10s} | {len(outcomes)} outcomes | winner: {winning_bucket or 'none'}")

            except Exception as e:
                log.debug(f"  Error {slug}: {e}")
                continue

    return all_markets


def _extract_bucket_name(question: str, station_id: str) -> str:
    """Extract bucket label from market question."""
    import re
    station = cfg.STATIONS.get(station_id, {})
    is_f = station.get("unit") == "fahrenheit"
    unit = "°F" if is_f else "°C"

    # "between 56-57°F"
    m = re.search(r'between\s+(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*F', question)
    if m:
        return f"{m.group(1)}-{m.group(2)}°F"
    # "be 14°C"
    m = re.search(r'be\s+(-?\d+)\s*°\s*C', question)
    if m:
        return f"{m.group(1)}°C"
    # "or higher"
    m = re.search(r'(-?\d+)\s*°\s*([FC])\s+or\s+higher', question)
    if m:
        return f"{m.group(1)}°{m.group(2)} or higher"
    # "or below"
    m = re.search(r'(-?\d+)\s*°\s*([FC])\s+or\s+(?:below|lower)', question)
    if m:
        return f"{m.group(1)}°{m.group(2)} or below"
    # Fallback
    m = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*F', question)
    if m:
        return f"{m.group(1)}-{m.group(2)}°F"
    m = re.search(r'(-?\d+)\s*°\s*C', question)
    if m:
        return f"{m.group(1)}°C"
    return question[:40]


def _compute_entry_prices(history: list) -> dict:
    """Compute entry prices at different time points from price history."""
    if not history:
        return {"early": 0, "mid": 0, "late": 0}

    prices = [float(h.get("p", 0)) for h in history if h.get("p")]
    if not prices:
        return {"early": 0, "mid": 0, "late": 0}

    n = len(prices)
    return {
        "early": prices[min(2, n - 1)],           # Early in trading (a few hours in)
        "mid": prices[n // 2],                      # Mid-point (~12h before resolution)
        "late": prices[max(0, n - n // 4)],         # Late (~6h before resolution)
    }


def collect_weather_forecasts(markets: List[dict], api_delay: float = 0.4) -> List[dict]:
    """Add weather forecast data to each market record."""
    from weather import WeatherEngine

    engines = {}
    for station_id in cfg.STATIONS:
        engines[station_id] = WeatherEngine(station_id)

    for market in markets:
        sid = market["station_id"]
        target_date = market["target_date"]
        engine = engines.get(sid)
        if not engine:
            continue

        # Get what models predicted the day before
        forecast_date = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

        forecasts = engine.fetch_historical_forecast(target_date, forecast_date)
        time.sleep(api_delay)

        actual_temp = engine.fetch_actual_temp(target_date)
        time.sleep(api_delay)

        market["forecasts"] = forecasts
        market["actual_temp"] = actual_temp

        if forecasts:
            import numpy as np
            temps = list(forecasts.values())
            market["forecast_mean"] = float(np.mean(temps))
            market["forecast_std"] = float(np.std(temps)) if len(temps) > 1 else 3.0
            market["forecast_median"] = float(np.median(temps))
        else:
            market["forecast_mean"] = None
            market["forecast_std"] = None
            market["forecast_median"] = None

        log.info(f"  Forecast {target_date} {sid}: mean={market.get('forecast_mean','?')} actual={actual_temp}")

    return markets


def save_dataset(markets: List[dict], filename: str = "real_markets.json"):
    """Save collected dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)

    # Strip large price_history arrays to save space (keep summary prices)
    for m in markets:
        for o in m.get("outcomes", []):
            o.pop("price_history", None)

    with open(path, "w") as f:
        json.dump(markets, f, indent=2, default=str)

    log.info(f"Saved {len(markets)} markets to {path}")
    return path


if __name__ == "__main__":
    import sys

    cities = ["NYC", "Chicago", "London", "Miami", "Tokyo"]
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 21

    log.info(f"Collecting real market data: {len(cities)} cities, {days} days back...")
    markets = collect_market_data(cities=cities, days_back=days)
    log.info(f"Collected {len(markets)} markets")

    log.info("Adding weather forecasts...")
    markets = collect_weather_forecasts(markets)

    path = save_dataset(markets)
    log.info(f"Done. Dataset: {path}")

    # Quick stats
    with_winner = [m for m in markets if m.get("winning_bucket")]
    with_forecast = [m for m in markets if m.get("forecast_mean") is not None]
    log.info(f"  Markets with winner: {len(with_winner)}")
    log.info(f"  Markets with forecast: {len(with_forecast)}")
