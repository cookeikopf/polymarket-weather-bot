"""
Weather Underground Historical Data Collector
===============================================
Fetches historical weather observations from WU (the Polymarket resolution source).
Caches responses to avoid re-fetching. Extracts daily high temperatures.
"""

import requests
import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import config


class WUDataCollector:
    """Collect historical weather data from Weather Underground API."""

    def __init__(self):
        self.api_key = config.WU_API_KEY
        self.api_base = config.WU_API_BASE
        self.cache_dir = Path(config.WU_CACHE_DIR)
        self.data_dir = Path(config.WU_DATA_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = 0.35  # seconds between requests

    # ═══════════════════════════════════════════════════════════════
    # API Fetching
    # ═══════════════════════════════════════════════════════════════

    def _cache_path(self, station_id: str, date_str: str) -> Path:
        """Cache file path for a station+date combo."""
        return self.cache_dir / f"{station_id}_{date_str}.json"

    def _fetch_raw(self, station_id: str, start_date: str, end_date: str = None) -> Optional[dict]:
        """
        Fetch raw observations from WU API.
        start_date/end_date in YYYYMMDD format.
        """
        wu_cfg = config.WU_STATIONS.get(station_id)
        if not wu_cfg:
            print(f"  [WU] Unknown station: {station_id}")
            return None

        wu_station = wu_cfg["wu"]
        units = wu_cfg["units"]

        url = f"{self.api_base}/{wu_station}/observations/historical.json"
        params = {
            "apiKey": self.api_key,
            "units": units,
            "startDate": start_date,
        }
        if end_date:
            params["endDate"] = end_date

        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=20)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 204:
                    return {"observations": []}  # No data
                elif resp.status_code == 429:
                    wait = (attempt + 1) * 5
                    print(f"  [WU] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [WU] API error {resp.status_code} for {station_id} {start_date}")
                    return None
            except Exception as e:
                print(f"  [WU] Request failed: {e}")
                time.sleep(2)

        return None

    def fetch_day(self, station_id: str, date: str) -> Optional[Dict]:
        """
        Fetch one day of WU data for a station.
        date: YYYY-MM-DD format.
        Returns dict with daily summary or None.
        """
        date_compact = date.replace("-", "")

        # Check cache
        cache_file = self._cache_path(station_id, date_compact)
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
                return cached.get("daily_summary")

        # Fetch from API
        data = self._fetch_raw(station_id, date_compact)
        time.sleep(self.request_delay)

        if not data or not data.get("observations"):
            return None

        # Process observations into daily summary
        summary = self._process_observations(data["observations"], date, station_id)

        # Cache
        cache_data = {
            "station_id": station_id,
            "date": date,
            "n_observations": len(data["observations"]),
            "daily_summary": summary,
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        return summary

    def fetch_range(
        self, station_id: str, start_date: str, end_date: str, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Fetch a date range for one station. Returns DataFrame with daily data.
        Dates in YYYY-MM-DD format.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_days = []
        current = start
        chunk_size = 5  # Fetch 5 days at a time

        while current <= end:
            chunk_end = min(current + timedelta(days=chunk_size - 1), end)
            start_compact = current.strftime("%Y%m%d")
            end_compact = chunk_end.strftime("%Y%m%d")

            # Check if all days in chunk are cached
            all_cached = True
            cached_days = []
            check_date = current
            while check_date <= chunk_end:
                ds = check_date.strftime("%Y%m%d")
                cache_file = self._cache_path(station_id, ds)
                if cache_file.exists():
                    with open(cache_file) as f:
                        cached = json.load(f)
                        if cached.get("daily_summary"):
                            cached_days.append(cached["daily_summary"])
                else:
                    all_cached = False
                    break
                check_date += timedelta(days=1)

            if all_cached:
                all_days.extend(cached_days)
            else:
                # Fetch chunk from API
                data = self._fetch_raw(station_id, start_compact, end_compact)
                time.sleep(self.request_delay)

                if data and data.get("observations"):
                    # Group observations by date
                    obs_by_date = self._group_by_date(data["observations"])

                    check_date = current
                    while check_date <= chunk_end:
                        date_str = check_date.strftime("%Y-%m-%d")
                        date_compact = check_date.strftime("%Y%m%d")
                        day_obs = obs_by_date.get(date_str, [])

                        if day_obs:
                            summary = self._process_observations(day_obs, date_str, station_id)
                        else:
                            summary = None

                        # Cache individual day
                        cache_file = self._cache_path(station_id, date_compact)
                        cache_data = {
                            "station_id": station_id,
                            "date": date_str,
                            "n_observations": len(day_obs),
                            "daily_summary": summary,
                        }
                        with open(cache_file, "w") as f:
                            json.dump(cache_data, f)

                        if summary:
                            all_days.append(summary)

                        check_date += timedelta(days=1)

            current = chunk_end + timedelta(days=1)

            if verbose and len(all_days) % 50 == 0 and len(all_days) > 0:
                print(f"  [WU] {station_id}: {len(all_days)} days collected...")

        if not all_days:
            return pd.DataFrame()

        df = pd.DataFrame(all_days)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def fetch_all_stations(
        self, start_date: str, end_date: str, verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for all 20 stations. Returns {station_id: DataFrame}."""
        results = {}

        for i, station_id in enumerate(config.WU_STATIONS.keys()):
            if verbose:
                print(f"\n[{i+1}/20] Fetching {station_id}...")

            df = self.fetch_range(station_id, start_date, end_date, verbose=verbose)
            if not df.empty:
                # Save to CSV
                out_path = self.data_dir / f"{station_id.replace(' ', '_')}.csv"
                df.to_csv(out_path, index=False)
                results[station_id] = df
                if verbose:
                    print(f"  {station_id}: {len(df)} days, "
                          f"temp range: {df['high_temp'].min():.0f}-{df['high_temp'].max():.0f}")
            else:
                print(f"  {station_id}: NO DATA")

        return results

    # ═══════════════════════════════════════════════════════════════
    # Data Processing
    # ═══════════════════════════════════════════════════════════════

    def _group_by_date(self, observations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group hourly observations by date."""
        by_date = {}
        for obs in observations:
            ts = obs.get("valid_time_gmt")
            if ts:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                date_str = dt.strftime("%Y-%m-%d")
                by_date.setdefault(date_str, []).append(obs)
        return by_date

    def _process_observations(
        self, observations: List[Dict], date_str: str, station_id: str
    ) -> Optional[Dict]:
        """Process hourly observations into a daily summary."""
        if not observations:
            return None

        temps = [o["temp"] for o in observations if o.get("temp") is not None]
        if not temps:
            return None

        # Extract all available fields
        dew_pts = [o["dewPt"] for o in observations if o.get("dewPt") is not None]
        rh_vals = [o["rh"] for o in observations if o.get("rh") is not None]
        pressure_vals = [o["pressure"] for o in observations if o.get("pressure") is not None]
        wspd_vals = [o["wspd"] for o in observations if o.get("wspd") is not None]
        precip_vals = [o["precip_hrly"] for o in observations if o.get("precip_hrly") is not None]

        wu_units = config.WU_STATIONS[station_id]["units"]

        summary = {
            "date": date_str,
            "station_id": station_id,
            "high_temp": max(temps),
            "low_temp": min(temps),
            "mean_temp": np.mean(temps),
            "temp_range": max(temps) - min(temps),
            "n_observations": len(observations),
            "units": "F" if wu_units == "e" else "C",
        }

        if dew_pts:
            summary["mean_dewpt"] = np.mean(dew_pts)
        if rh_vals:
            summary["mean_rh"] = np.mean(rh_vals)
        if pressure_vals:
            summary["mean_pressure"] = np.mean(pressure_vals)
        if wspd_vals:
            summary["mean_wspd"] = np.mean(wspd_vals)
            summary["max_wspd"] = max(wspd_vals)
        if precip_vals:
            summary["total_precip"] = sum(p for p in precip_vals if p > 0)
        else:
            summary["total_precip"] = 0.0

        return summary

    def load_station_data(self, station_id: str) -> pd.DataFrame:
        """Load previously collected station data from CSV."""
        path = self.data_dir / f"{station_id.replace(' ', '_')}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            return df
        return pd.DataFrame()


def main():
    """Collect WU data for all 20 Polymarket stations."""
    collector = WUDataCollector()

    # Collect from 2024-06-01 to 2026-03-15
    start = "2024-06-01"
    end = "2026-03-15"

    print(f"{'='*60}")
    print(f"  Weather Underground Data Collection")
    print(f"  Period: {start} to {end}")
    print(f"  Stations: {len(config.WU_STATIONS)}")
    print(f"{'='*60}")

    results = collector.fetch_all_stations(start, end, verbose=True)

    print(f"\n{'='*60}")
    print(f"  COLLECTION SUMMARY")
    print(f"{'='*60}")
    total_days = 0
    for station_id, df in sorted(results.items()):
        total_days += len(df)
        print(f"  {station_id:15s}: {len(df):4d} days | "
              f"High: {df['high_temp'].min():.0f}-{df['high_temp'].max():.0f} | "
              f"Missing: {(pd.Timestamp(end) - pd.Timestamp(start)).days + 1 - len(df)} days")

    print(f"\n  Total: {total_days} station-days collected")
    print(f"  Data saved to: {config.WU_DATA_DIR}/")


if __name__ == "__main__":
    main()
