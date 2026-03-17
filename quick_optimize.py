#!/usr/bin/env python3
"""
Quick Parameter Optimizer
==========================
Calibrates 2 representative cities, then runs 600 param combos
using cached calibration. Total time: ~5-8 minutes.
"""

import numpy as np
import json
import os
import sys
import itertools
import copy
import time
from datetime import datetime
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from backtester import Backtester
from weather_engine import WeatherEngine

TEST_CITIES = ["NYC", "London"]  # 1 Fahrenheit, 1 Celsius — fast calibration
BACKTEST_START = "2025-06-01"
BACKTEST_END = "2026-03-15"
BASE_BANKROLL = 50.0

start_time = time.time()


def calibrate_and_cache():
    """Calibrate cities once, return cached engine data + actuals."""
    cache = {}
    for city in TEST_CITIES:
        print(f"  Calibrating {city}...", end=" ", flush=True)
        t0 = time.time()
        engine = WeatherEngine(city)
        engine.calibrate()
        actuals = engine.fetch_historical_actuals(BACKTEST_START, BACKTEST_END)
        cache[city] = {
            "engine": engine,
            "actuals": actuals,
        }
        print(f"done ({time.time()-t0:.0f}s, {len(actuals)} days)")
    return cache


def fast_backtest(city, cache, params):
    """Run backtest reusing cached engine — NO API calls."""
    config.MIN_EDGE_PCT = params["min_edge"]
    config.KELLY_FRACTION = params["kelly"]
    config.MAX_POSITION_PCT = params["max_pos"]
    config.MIN_ENSEMBLE_AGREEMENT = params["min_agreement"]
    config.MAX_TRADE_SIZE_USDC = params.get("max_trade_usd", 25.0)
    config.SIM_SPREAD = params.get("spread", 0.06)
    config.SIM_SLIPPAGE = params.get("slippage", 0.02)
    config.MAX_DRAWDOWN_PCT = params.get("max_dd", 0.20)

    cached = cache[city]
    src_engine = cached["engine"]
    actuals = cached["actuals"]

    bt = Backtester(city, initial_bankroll=BASE_BANKROLL)
    # Copy calibration data
    bt.engine.error_distributions = src_engine.error_distributions
    bt.engine.model_weights = src_engine.model_weights
    bt.engine.is_calibrated = True
    if hasattr(src_engine, 'historical_biases'):
        bt.engine.historical_biases = src_engine.historical_biases

    if actuals.empty:
        return None

    for _, row in actuals.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        actual_temp = row["actual"]
        if pd.isna(actual_temp):
            bt.equity_curve.append(bt.bankroll)
            bt.daily_pnl.append(0)
            continue
        if bt.drawdown_halt:
            bt.equity_curve.append(bt.bankroll)
            bt.daily_pnl.append(0)
            continue
        day_pnl = bt._simulate_day(date_str, actual_temp, False)
        bt.daily_pnl.append(day_pnl)
        bt.equity_curve.append(bt.bankroll)
        bt.peak_bankroll = max(bt.peak_bankroll, bt.bankroll)
        dd = (bt.peak_bankroll - bt.bankroll) / bt.peak_bankroll
        if dd >= config.MAX_DRAWDOWN_PCT:
            bt.drawdown_halt = True

    result = bt._compile_results()
    return result, bt


def main():
    print(f"\n{'='*70}")
    print(f"  QUICK PARAMETER OPTIMIZATION")
    print(f"  Cities: {', '.join(TEST_CITIES)}")
    print(f"  Period: {BACKTEST_START} to {BACKTEST_END}")
    print(f"{'='*70}\n")

    # PHASE 0: One-time calibration
    cache = calibrate_and_cache()

    # PHASE 1: Grid search (pure computation, no API calls)
    grid = {
        "kelly":         [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        "min_edge":      [0.03, 0.05, 0.07, 0.10, 0.12, 0.15],
        "min_agreement": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        "max_pos":       [0.05, 0.08, 0.10, 0.15, 0.20],
    }
    fixed = {"max_trade_usd": 25.0, "spread": 0.06, "slippage": 0.02, "max_dd": 0.20}

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    total = len(combos)
    print(f"\nPhase 1: Grid search over {total} combos...\n")

    results = []
    best_score = -999
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params.update(fixed)

        trades_all = []
        returns = []
        drawdowns = []

        for city in TEST_CITIES:
            out = fast_backtest(city, cache, params)
            if out is None:
                continue
            res, bt = out
            returns.append(res.total_return_pct)
            drawdowns.append(res.max_drawdown_pct)
            for t in res.trades:
                trades_all.append({"pnl": t.pnl, "won": t.won, "edge": t.edge})

        if not trades_all or not returns:
            results.append({"params": params, "score": -999})
            continue

        n = len(trades_all)
        wr = len([t for t in trades_all if t["won"]]) / n * 100
        tp = sum(t["pnl"] for t in trades_all)
        tw = sum(t["pnl"] for t in trades_all if t["pnl"] > 0)
        tl = abs(sum(t["pnl"] for t in trades_all if t["pnl"] <= 0))
        pf = tw / tl if tl > 0 else 999
        avg_ret = np.mean(returns)
        min_ret = min(returns)
        worst_dd = max(drawdowns)
        tpnl = [t["pnl"] for t in trades_all]
        sharpe = np.mean(tpnl) / np.std(tpnl) if np.std(tpnl) > 0 else 0
        cal_days = (datetime.strptime(BACKTEST_END, "%Y-%m-%d") -
                   datetime.strptime(BACKTEST_START, "%Y-%m-%d")).days
        tpw = n / (cal_days / 7)

        score = (avg_ret * 0.30 + min_ret * 0.25 + pf * 5 * 0.20 +
                 sharpe * 50 * 0.15 - worst_dd * 0.10)

        r = {
            "params": params, "score": round(score, 2),
            "total_trades": n, "trades_per_week": round(tpw, 1),
            "win_rate": round(wr, 1), "avg_return_pct": round(avg_ret, 1),
            "min_return_pct": round(min_ret, 1), "total_pnl": round(tp, 2),
            "profit_factor": round(pf, 1), "trade_sharpe": round(sharpe, 4),
            "worst_drawdown": round(worst_dd, 1),
        }
        results.append(r)

        if score > best_score:
            best_score = score
            p = params
            print(f"  ★ NEW BEST [{i+1}/{total}] score={score:.0f} | "
                  f"ret={avg_ret:+.0f}% | WR={wr:.0f}% | PF={pf:.1f} | "
                  f"T/wk={tpw:.1f} | DD={worst_dd:.0f}% | "
                  f"k={p['kelly']} e={p['min_edge']} a={p['min_agreement']} p={p['max_pos']}")

        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (total - i - 1)
            print(f"  ... {i+1}/{total} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    results.sort(key=lambda r: r.get("score", -999), reverse=True)

    # Filter valid results
    valid = [r for r in results if r.get("total_trades", 0) > 0]

    print(f"\n{'='*70}")
    print(f"  TOP 15 CONFIGURATIONS (of {len(valid)} valid)")
    print(f"{'='*70}")
    print(f"  {'#':>3} {'Score':>7} {'Ret%':>7} {'MinR%':>7} {'WR%':>5} {'PF':>5} "
          f"{'T/wk':>5} {'DD%':>5} | {'K':>5} {'E':>5} {'A':>5} {'P':>5}")
    print(f"  {'─'*74}")

    for rank, r in enumerate(valid[:15], 1):
        p = r["params"]
        print(f"  {rank:>3} {r['score']:>7.0f} {r['avg_return_pct']:>+6.0f}% "
              f"{r['min_return_pct']:>+6.0f}% {r['win_rate']:>4.0f}% {r['profit_factor']:>5.1f} "
              f"{r['trades_per_week']:>5.1f} {r['worst_drawdown']:>4.0f}% | "
              f"{p['kelly']:>4.2f} {p['min_edge']:>4.2f} "
              f"{p['min_agreement']:>4.2f} {p['max_pos']:>4.2f}")

    # PHASE 2: Sensitivity
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Friction Sensitivity")
    print(f"{'='*70}\n")

    best_p = valid[0]["params"].copy()
    sens = []
    for spread, slip, label in [
        (0.02, 0.005, "Very optimistic (weather: 0 fees)"),
        (0.04, 0.01, "Optimistic"),
        (0.06, 0.02, "Baseline"),
        (0.08, 0.03, "Pessimistic"),
        (0.10, 0.04, "Very pessimistic"),
    ]:
        p = best_p.copy()
        p["spread"] = spread
        p["slippage"] = slip
        trades = []
        for city in TEST_CITIES:
            out = fast_backtest(city, cache, p)
            if out:
                for t in out[0].trades:
                    trades.append({"pnl": t.pnl, "won": t.won})
        if trades:
            wr = len([t for t in trades if t["won"]]) / len(trades) * 100
            tp = sum(t["pnl"] for t in trades)
            tw = sum(t["pnl"] for t in trades if t["pnl"] > 0)
            tl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
            pf = tw / tl if tl > 0 else 999
            print(f"  {label:35s} PnL=${tp:>+9.2f} | WR={wr:.0f}% | PF={pf:.1f} | N={len(trades)}")
            sens.append({"label": label, "spread": spread, "slippage": slip,
                        "pnl": round(tp, 2), "wr": round(wr, 1), "pf": round(pf, 1)})

    # Save
    os.makedirs("results", exist_ok=True)
    bp = valid[0]["params"]

    output = {
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": round(time.time() - start_time),
        "configs_tested": total,
        "valid_configs": len(valid),
        "optimal_params": {
            "KELLY_FRACTION": bp["kelly"],
            "MIN_EDGE_PCT": bp["min_edge"],
            "MIN_ENSEMBLE_AGREEMENT": bp["min_agreement"],
            "MAX_POSITION_PCT": bp["max_pos"],
        },
        "current_params": {
            "KELLY_FRACTION": 0.25,
            "MIN_EDGE_PCT": 0.05,
            "MIN_ENSEMBLE_AGREEMENT": 0.60,
            "MAX_POSITION_PCT": 0.10,
        },
        "optimal_metrics": {
            "avg_return_pct": valid[0]["avg_return_pct"],
            "win_rate": valid[0]["win_rate"],
            "profit_factor": valid[0]["profit_factor"],
            "trades_per_week": valid[0]["trades_per_week"],
            "worst_drawdown": valid[0]["worst_drawdown"],
        },
        "top_15": [{
            "rank": i+1, **{k: v for k, v in r.items() if k != "params"},
            "kelly": r["params"]["kelly"], "min_edge": r["params"]["min_edge"],
            "min_agreement": r["params"]["min_agreement"], "max_pos": r["params"]["max_pos"],
        } for i, r in enumerate(valid[:15])],
        "sensitivity": sens,
    }

    with open("results/optimization_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Parameter Optimization — Polymarket Weather Bot",
                 fontsize=15, fontweight="bold", y=0.98)

    # 1. Top 20 returns
    ax1 = axes[0, 0]
    top20 = valid[:20]
    rets = [r["avg_return_pct"] for r in top20]
    cols = ["#4CAF50" if r > 0 else "#FF5252" for r in rets]
    ax1.bar(range(1, len(rets)+1), rets, color=cols, alpha=0.85)
    ax1.set_xlabel("Config Rank"); ax1.set_ylabel("Avg Return (%)")
    ax1.set_title("Top 20 Configs: Average Return", fontweight="bold")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Sensitivity
    ax2 = axes[0, 1]
    if sens:
        labs = [s["label"].split("(")[0].strip() for s in sens]
        pnls = [s["pnl"] for s in sens]
        pfs = [s["pf"] for s in sens]
        x = range(len(labs))
        w = 0.35
        ax2.bar([i-w/2 for i in x], pnls, w, label="PnL ($)", color="#2196F3", alpha=0.85)
        ax2t = ax2.twinx()
        ax2t.bar([i+w/2 for i in x], pfs, w, label="PF", color="#FF9800", alpha=0.85)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(labs, rotation=20, ha="right", fontsize=9)
        ax2.set_ylabel("PnL ($)", color="#2196F3")
        ax2t.set_ylabel("Profit Factor", color="#FF9800")
        ax2.set_title("Market Friction Sensitivity", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

    # 3. Kelly heatmap
    ax3 = axes[1, 0]
    ks = [r["params"]["kelly"] for r in valid[:50]]
    es = [r["params"]["min_edge"] for r in valid[:50]]
    sc_vals = [r["avg_return_pct"] for r in valid[:50]]
    sc = ax3.scatter(ks, es, c=sc_vals, cmap="RdYlGn", s=80, edgecolors="black", linewidth=0.5)
    ax3.set_xlabel("Kelly Fraction"); ax3.set_ylabel("Min Edge")
    ax3.set_title("Kelly × Edge → Return (Top 50)", fontweight="bold")
    plt.colorbar(sc, ax=ax3, label="Return (%)")
    ax3.grid(True, alpha=0.3)

    # 4. Win Rate vs Profit Factor
    ax4 = axes[1, 1]
    wrs = [r["win_rate"] for r in valid[:50]]
    pfs2 = [r["profit_factor"] for r in valid[:50]]
    rets2 = [r["avg_return_pct"] for r in valid[:50]]
    sc2 = ax4.scatter(wrs, pfs2, c=rets2, cmap="RdYlGn", s=80, edgecolors="black", linewidth=0.5)
    ax4.set_xlabel("Win Rate (%)"); ax4.set_ylabel("Profit Factor")
    ax4.set_title("Win Rate × PF → Return (Top 50)", fontweight="bold")
    plt.colorbar(sc2, ax=ax4, label="Return (%)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/optimization_charts.png", dpi=150, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Current:  kelly=0.25, edge=0.05, agreement=0.60, pos=0.10")
    print(f"  Optimal:  kelly={bp['kelly']}, edge={bp['min_edge']}, "
          f"agreement={bp['min_agreement']}, pos={bp['max_pos']}")
    print(f"  Return:   {valid[0]['avg_return_pct']:+.0f}% avg | "
          f"WR={valid[0]['win_rate']:.0f}% | PF={valid[0]['profit_factor']:.1f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
