"""
WU vs Open-Meteo Comparison
============================
Compare Weather Underground actuals (Polymarket resolution source)
with Open-Meteo reanalysis data (our forecast source).
Measures systematic biases that our ML model can exploit.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict

import config


def fetch_openmeteo_actuals(station_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical actual temperatures from Open-Meteo."""
    station = config.STATIONS[station_id]
    unit = station.get("unit", "fahrenheit")

    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "daily": "temperature_2m_max",
        "timezone": station.get("tz", "UTC"),
        "start_date": start_date,
        "end_date": end_date,
        "temperature_unit": unit,
    }

    resp = requests.get(config.OPEN_METEO_HISTORICAL_URL, params=params, timeout=30)
    if resp.status_code != 200:
        return pd.DataFrame()

    data = resp.json()
    if "daily" not in data:
        return pd.DataFrame()

    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "om_high_temp": data["daily"]["temperature_2m_max"],
    })
    return df.dropna()


def compare_station(station_id: str, wu_df: pd.DataFrame) -> Dict:
    """Compare WU vs Open-Meteo for one station."""
    if wu_df.empty:
        return None

    start = wu_df["date"].min().strftime("%Y-%m-%d")
    end = wu_df["date"].max().strftime("%Y-%m-%d")

    om_df = fetch_openmeteo_actuals(station_id, start, end)
    time.sleep(0.3)

    if om_df.empty:
        return None

    # Merge on date
    wu_daily = wu_df[["date", "high_temp"]].copy()
    wu_daily["date"] = pd.to_datetime(wu_daily["date"])
    merged = wu_daily.merge(om_df, on="date", how="inner")

    if len(merged) < 30:
        return None

    # Compute differences (WU - OpenMeteo)
    diff = merged["high_temp"] - merged["om_high_temp"]

    unit = config.WU_STATIONS[station_id]["units"]
    unit_label = "°F" if unit == "e" else "°C"

    result = {
        "station_id": station_id,
        "n_days": len(merged),
        "unit": unit_label,
        "mean_bias": float(diff.mean()),        # Positive = WU reads higher
        "std_diff": float(diff.std()),
        "mae": float(diff.abs().mean()),
        "max_abs_diff": float(diff.abs().max()),
        "median_diff": float(diff.median()),
        "pct_wu_higher": float((diff > 0).mean() * 100),
        "pct_wu_lower": float((diff < 0).mean() * 100),
        "pct_equal": float((diff == 0).mean() * 100),
        "correlation": float(merged["high_temp"].corr(merged["om_high_temp"])),
        "p5_diff": float(diff.quantile(0.05)),
        "p95_diff": float(diff.quantile(0.95)),
    }
    return result


def run_comparison():
    """Run full comparison for all stations."""
    print(f"\n{'='*70}")
    print(f"  WU vs Open-Meteo Comparison")
    print(f"  Measuring resolution data discrepancies")
    print(f"{'='*70}\n")

    results = {}
    wu_data_dir = config.WU_DATA_DIR

    for i, station_id in enumerate(config.WU_STATIONS.keys()):
        csv_path = os.path.join(wu_data_dir, f"{station_id.replace(' ', '_')}.csv")
        if not os.path.exists(csv_path):
            print(f"  [{i+1}/20] {station_id}: No WU data")
            continue

        wu_df = pd.read_csv(csv_path)
        wu_df["date"] = pd.to_datetime(wu_df["date"])

        print(f"  [{i+1}/20] {station_id}...", end="", flush=True)
        result = compare_station(station_id, wu_df)

        if result:
            results[station_id] = result
            bias = result["mean_bias"]
            mae = result["mae"]
            unit = result["unit"]
            direction = "WU höher" if bias > 0 else "WU niedriger"
            print(f" Bias: {bias:+.2f}{unit} ({direction}), MAE: {mae:.2f}{unit}, "
                  f"Korr: {result['correlation']:.4f}")
        else:
            print(f" FAILED")

    # Summary
    if results:
        print(f"\n{'='*70}")
        print(f"  ZUSAMMENFASSUNG: WU vs Open-Meteo Diskrepanzen")
        print(f"{'='*70}")
        print(f"\n  {'Station':15s} {'Bias':>8s} {'MAE':>8s} {'Std':>8s} {'MaxDiff':>8s} {'Korr':>8s} {'WU>OM':>7s}")
        print(f"  {'─'*62}")

        f_biases = []
        c_biases = []

        for sid, r in sorted(results.items()):
            u = r["unit"]
            print(f"  {sid:15s} {r['mean_bias']:+7.2f}{u} {r['mae']:6.2f}{u} "
                  f"{r['std_diff']:6.2f}{u} {r['max_abs_diff']:6.1f}{u} "
                  f"{r['correlation']:7.4f} {r['pct_wu_higher']:5.1f}%")

            if u == "°F":
                f_biases.append(r["mean_bias"])
            else:
                c_biases.append(r["mean_bias"])

        if f_biases:
            print(f"\n  °F stations avg bias: {np.mean(f_biases):+.2f}°F")
        if c_biases:
            print(f"  °C stations avg bias: {np.mean(c_biases):+.2f}°C")

        all_maes = [r["mae"] for r in results.values()]
        print(f"  Overall avg MAE: {np.mean(all_maes):.2f}")

        # Save results
        out_path = os.path.join("data", "wu_comparison.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_comparison()
