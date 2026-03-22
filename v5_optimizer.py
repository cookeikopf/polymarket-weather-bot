#!/usr/bin/env python3
"""
V5 Parameter Optimizer
=======================
Runs the V5 backtester with different parameter combinations to find
the most profitable settings. Uses cached API data so each run takes ~2s.

Usage: python3 v5_optimizer.py
"""

import os
import sys
import json
import itertools
import time
import numpy as np
from copy import deepcopy

os.environ['OPEN_METEO_API_KEY'] = 'wjrcKzLOeLkcCnzx'

# Import the backtester module
import v5_backtester as bt

# ═══════════════════════════════════════════════════════════════
# PARAMETER GRID
# ═══════════════════════════════════════════════════════════════

PARAM_GRID = {
    # Edge thresholds
    "MIN_EDGE": [0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    
    # Position sizing
    "KELLY_FRACTION": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "MAX_POSITION_PCT": [0.08, 0.10, 0.15, 0.20],
    
    # Ladder strategy
    "LADDER_MAX_PRICE": [0.15, 0.18, 0.22, 0.25],
    "LADDER_BUCKETS": [3, 5, 7],
    "LADDER_BET_PER_BUCKET": [1.5, 2.0, 3.0, 4.0],
    "MAX_LADDER_SETS_PER_DAY": [1, 2],
    
    # Conservative NO
    "NO_MIN_ENTRY": [0.60, 0.65, 0.70, 0.75],
    "MAX_NO_TRADES_PER_DAY": [2, 3, 5],
}

# Total combinations would be enormous — we do focused sweeps instead.

def run_backtest_with_params(**params):
    """Run a single backtest with given parameter overrides."""
    # Save original values
    originals = {}
    for key, val in params.items():
        originals[key] = getattr(bt, key)
        setattr(bt, key, val)
    
    try:
        # Run backtest silently
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            result = bt.run_backtest()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        
        return result
    finally:
        # Restore originals
        for key, val in originals.items():
            setattr(bt, key, val)


def extract_metrics(result):
    """Extract key metrics from backtest result."""
    if result is None:
        return None
    return {
        "total_pnl": result.get("total_pnl", 0),
        "total_return_pct": result.get("total_return_pct", 0),
        "max_drawdown_pct": result.get("max_drawdown_pct", 100),
        "sharpe_ratio": result.get("sharpe_ratio", 0),
        "profit_factor": result.get("profit_factor", 0),
        "win_rate": result.get("win_rate", 0),
        "total_trades": result.get("total_trades", 0),
        "bankroll_end": result.get("bankroll_end", 100),
    }


def score(m):
    """
    Composite score: maximize risk-adjusted returns.
    Penalize high drawdowns and low trade counts.
    """
    if m is None or m["total_trades"] < 20:
        return -999
    
    # Primary: Sharpe ratio (risk-adjusted return)
    s = m["sharpe_ratio"] * 40
    
    # Bonus for absolute return
    s += min(m["total_return_pct"], 300) * 0.3
    
    # Penalty for drawdown > 25%
    if m["max_drawdown_pct"] > 25:
        s -= (m["max_drawdown_pct"] - 25) * 2
    if m["max_drawdown_pct"] > 40:
        s -= (m["max_drawdown_pct"] - 40) * 5
    
    # Bonus for decent win rate
    if m["win_rate"] > 0.3:
        s += 10
    
    # Bonus for profit factor
    s += min(m["profit_factor"], 3) * 10
    
    return s


def sweep_focused(name, param_dict, fixed_overrides=None):
    """Run a focused parameter sweep on a few params."""
    keys = list(param_dict.keys())
    combos = list(itertools.product(*[param_dict[k] for k in keys]))
    
    print(f"\n{'='*70}")
    print(f"SWEEP: {name} ({len(combos)} combinations)")
    print(f"{'='*70}")
    
    results = []
    t0 = time.time()
    
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        if fixed_overrides:
            params.update(fixed_overrides)
        
        try:
            result = run_backtest_with_params(**params)
            metrics = extract_metrics(result)
            s = score(metrics)
            results.append({
                "params": params,
                "metrics": metrics,
                "score": s,
            })
        except Exception as e:
            results.append({
                "params": params,
                "metrics": None,
                "score": -999,
                "error": str(e),
            })
        
        # Progress
        if (i + 1) % 10 == 0 or i == len(combos) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(combos) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(combos)}] {rate:.1f} runs/sec, ~{remaining:.0f}s remaining")
    
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Print top 10
    print(f"\nTOP 10 RESULTS:")
    print(f"{'Rank':<5} {'Score':<8} {'Return%':<10} {'Sharpe':<8} {'MaxDD%':<8} {'PF':<6} {'WinR%':<7} {'Trades':<7} | Params")
    print("-" * 120)
    
    for i, r in enumerate(results[:10]):
        m = r["metrics"]
        if m is None:
            continue
        param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"  {i+1:<3} {r['score']:<8.1f} {m['total_return_pct']:<10.1f} {m['sharpe_ratio']:<8.2f} "
              f"{m['max_drawdown_pct']:<8.1f} {m['profit_factor']:<6.2f} {m['win_rate']*100:<7.1f} "
              f"{m['total_trades']:<7} | {param_str}")
    
    return results


def main():
    print("V5 PARAMETER OPTIMIZER")
    print("=" * 70)
    print(f"Using cached backtest data — each run takes ~2-3 seconds")
    print(f"Baseline: +119.6% return, 2.49 Sharpe, 31.7% MaxDD")
    
    all_results = {}
    
    # ── SWEEP 1: Edge Threshold + Kelly Fraction ──
    # These are the most important risk/reward parameters
    r1 = sweep_focused("Edge + Kelly", {
        "MIN_EDGE": [0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
        "KELLY_FRACTION": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    })
    all_results["edge_kelly"] = r1
    
    # Get best edge/kelly from sweep 1
    best1 = r1[0]["params"]
    best_edge = best1["MIN_EDGE"]
    best_kelly = best1["KELLY_FRACTION"]
    print(f"\n>>> Best Edge/Kelly: MIN_EDGE={best_edge}, KELLY_FRACTION={best_kelly}")
    
    # ── SWEEP 2: Position Sizing + Ladder Params ──
    r2 = sweep_focused("Ladder Strategy", {
        "LADDER_MAX_PRICE": [0.15, 0.18, 0.22, 0.25, 0.30],
        "LADDER_BUCKETS": [3, 5, 7, 9],
        "LADDER_BET_PER_BUCKET": [1.5, 2.0, 3.0, 4.0, 5.0],
    }, fixed_overrides={"MIN_EDGE": best_edge, "KELLY_FRACTION": best_kelly})
    all_results["ladder"] = r2
    
    best2 = r2[0]["params"]
    best_ladder_price = best2["LADDER_MAX_PRICE"]
    best_ladder_buckets = best2["LADDER_BUCKETS"]
    best_ladder_bet = best2["LADDER_BET_PER_BUCKET"]
    print(f"\n>>> Best Ladder: MAX_PRICE={best_ladder_price}, BUCKETS={best_ladder_buckets}, BET={best_ladder_bet}")
    
    # ── SWEEP 3: Conservative NO params ──
    r3 = sweep_focused("Conservative NO", {
        "NO_MIN_ENTRY": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        "MAX_NO_TRADES_PER_DAY": [1, 2, 3, 5, 7],
        "MAX_POSITION_PCT": [0.08, 0.10, 0.15, 0.20],
    }, fixed_overrides={
        "MIN_EDGE": best_edge, "KELLY_FRACTION": best_kelly,
        "LADDER_MAX_PRICE": best_ladder_price, "LADDER_BUCKETS": best_ladder_buckets,
        "LADDER_BET_PER_BUCKET": best_ladder_bet,
    })
    all_results["conservative_no"] = r3
    
    best3 = r3[0]["params"]
    print(f"\n>>> Best NO: NO_MIN_ENTRY={best3['NO_MIN_ENTRY']}, MAX_NO_TRADES={best3['MAX_NO_TRADES_PER_DAY']}, MAX_POS={best3['MAX_POSITION_PCT']}")
    
    # ── SWEEP 4: Ladder Sets per Day ──
    r4 = sweep_focused("Ladder Frequency", {
        "MAX_LADDER_SETS_PER_DAY": [1, 2, 3],
        "LADDER_MIN_PRICE": [0.01, 0.02, 0.03, 0.05],
    }, fixed_overrides={
        "MIN_EDGE": best_edge, "KELLY_FRACTION": best_kelly,
        "LADDER_MAX_PRICE": best_ladder_price, "LADDER_BUCKETS": best_ladder_buckets,
        "LADDER_BET_PER_BUCKET": best_ladder_bet,
        "NO_MIN_ENTRY": best3["NO_MIN_ENTRY"], 
        "MAX_NO_TRADES_PER_DAY": best3["MAX_NO_TRADES_PER_DAY"],
        "MAX_POSITION_PCT": best3["MAX_POSITION_PCT"],
    })
    all_results["ladder_freq"] = r4
    
    # ── FINAL: Combine all best params ──
    print("\n" + "=" * 70)
    print("FINAL OPTIMIZED RUN")
    print("=" * 70)
    
    best_params = {}
    best_params.update(r1[0]["params"])
    best_params.update({k: v for k, v in r2[0]["params"].items() if k not in ("MIN_EDGE", "KELLY_FRACTION")})
    best_params.update({k: v for k, v in r3[0]["params"].items() if k not in best_params})
    best_params.update({k: v for k, v in r4[0]["params"].items() if k not in best_params})
    
    print(f"\nOptimized Parameters:")
    for k, v in sorted(best_params.items()):
        default_val = getattr(bt, k)
        changed = " ← CHANGED" if v != default_val else ""
        print(f"  {k}: {default_val} → {v}{changed}")
    
    # Run final backtest with all optimized params
    result = run_backtest_with_params(**best_params)
    metrics = extract_metrics(result)
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'Optimized':<15} {'Delta':<15}")
    print("-" * 70)
    baseline = {
        "total_return_pct": 119.6, "sharpe_ratio": 2.49, "max_drawdown_pct": 31.7,
        "profit_factor": 1.24, "win_rate": 0.264, "total_trades": 295,
    }
    for key in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "profit_factor", "win_rate", "total_trades"]:
        b = baseline[key]
        o = metrics[key]
        if key == "win_rate":
            print(f"  {key:<23} {b*100:<15.1f} {o*100:<15.1f} {(o-b)*100:+.1f}%")
        elif key == "max_drawdown_pct":
            print(f"  {key:<23} {b:<15.1f} {o:<15.1f} {o-b:+.1f} {'(better)' if o < b else '(worse)'}")
        else:
            print(f"  {key:<23} {b:<15.2f} {o:<15.2f} {o-b:+.2f}")
    
    # Save results
    output = {
        "optimized_params": best_params,
        "baseline_metrics": baseline,
        "optimized_metrics": metrics,
        "improvement": {
            "return_pct_delta": metrics["total_return_pct"] - baseline["total_return_pct"],
            "sharpe_delta": metrics["sharpe_ratio"] - baseline["sharpe_ratio"],
            "drawdown_delta": metrics["max_drawdown_pct"] - baseline["max_drawdown_pct"],
        },
        "sweep_results_summary": {
            name: {
                "best_params": r[0]["params"],
                "best_score": r[0]["score"],
                "best_metrics": r[0]["metrics"],
                "n_combinations": len(r),
            }
            for name, r in all_results.items()
        }
    }
    
    with open("/home/user/workspace/v5_optimization_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to /home/user/workspace/v5_optimization_results.json")
    print(f"\nDone! Copy the optimized params to config.py for live trading.")


if __name__ == "__main__":
    main()
