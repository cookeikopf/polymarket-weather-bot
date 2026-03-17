#!/usr/bin/env python3
"""
Fast Parameter Optimizer
=========================
Pre-calibrates cities ONCE, then runs rapid parameter sweeps
by reusing cached calibration data.
"""

import numpy as np
import json
import os
import sys
import itertools
import copy
from datetime import datetime
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from backtester import Backtester, BacktestResult
from weather_engine import WeatherEngine


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
TEST_CITIES = ["NYC", "Chicago", "Miami", "London", "Tokyo", "Buenos Aires"]
BACKTEST_START = "2025-06-01"
BACKTEST_END = "2026-03-15"
BASE_BANKROLL = 50.0


def pre_calibrate_cities(cities):
    """Calibrate all cities once and cache engine state."""
    print("Pre-calibrating weather engines (one-time)...")
    engines = {}
    city_actuals = {}

    for city in cities:
        print(f"  Calibrating {city}...")
        engine = WeatherEngine(city)
        engine.calibrate()
        engines[city] = {
            "error_distributions": copy.deepcopy(engine.error_distributions),
            "model_weights": copy.deepcopy(engine.model_weights),
            "historical_biases": copy.deepcopy(getattr(engine, 'historical_biases', {})),
        }

        # Also fetch actuals once
        actuals = engine.fetch_historical_actuals(BACKTEST_START, BACKTEST_END)
        city_actuals[city] = actuals
        print(f"    {city}: {len(actuals)} days, error_dists for {len(engine.error_distributions)} models")

    return engines, city_actuals


def run_fast_backtest(city, cached_engine, cached_actuals, params):
    """Run backtest using pre-cached calibration data — no API calls."""
    # Apply params
    config.MIN_EDGE_PCT = params["min_edge"]
    config.KELLY_FRACTION = params["kelly"]
    config.MAX_POSITION_PCT = params["max_pos"]
    config.MIN_ENSEMBLE_AGREEMENT = params["min_agreement"]
    config.MAX_TRADE_SIZE_USDC = params.get("max_trade_usd", 25.0)
    config.SIM_SPREAD = params.get("spread", 0.06)
    config.SIM_SLIPPAGE = params.get("slippage", 0.02)
    config.MAX_DRAWDOWN_PCT = params.get("max_dd", 0.20)

    bt = Backtester(city, initial_bankroll=BASE_BANKROLL)
    # Inject cached calibration
    bt.engine.error_distributions = cached_engine["error_distributions"]
    bt.engine.model_weights = cached_engine["model_weights"]
    if hasattr(bt.engine, 'historical_biases'):
        bt.engine.historical_biases = cached_engine.get("historical_biases", {})
    bt.engine.is_calibrated = True

    # Run simulation using cached actuals
    import pandas as pd
    actuals = cached_actuals

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

    return bt._compile_results(), bt


def run_optimization():
    """Main optimization routine."""
    print(f"\n{'='*70}")
    print(f"  FAST PARAMETER OPTIMIZATION")
    print(f"  Cities: {', '.join(TEST_CITIES)}")
    print(f"  Period: {BACKTEST_START} to {BACKTEST_END}")
    print(f"  Bankroll: ${BASE_BANKROLL}")
    print(f"{'='*70}\n")

    # Phase 0: Pre-calibrate
    engines, city_actuals = pre_calibrate_cities(TEST_CITIES)

    # Phase 1: Grid search
    param_grid = {
        "kelly":         [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        "min_edge":      [0.03, 0.05, 0.07, 0.10, 0.12],
        "min_agreement": [0.45, 0.50, 0.55, 0.60, 0.65],
        "max_pos":       [0.05, 0.08, 0.10, 0.15],
    }
    fixed = {"max_trade_usd": 25.0, "spread": 0.06, "slippage": 0.02, "max_dd": 0.20}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)

    print(f"\nPhase 1: Testing {total} parameter combos (cached, fast)...\n")

    results = []
    best_score = -999

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        params.update(fixed)

        all_trades = []
        city_results = {}

        for city in TEST_CITIES:
            try:
                out = run_fast_backtest(city, engines[city], city_actuals[city], params)
                if out is None:
                    continue
                result, bt = out

                city_results[city] = {
                    "return_pct": result.total_return_pct,
                    "max_drawdown": result.max_drawdown_pct,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                    "profit_factor": result.profit_factor,
                    "sharpe": result.sharpe_ratio,
                    "avg_edge": result.avg_edge,
                }

                for t in result.trades:
                    all_trades.append({
                        "pnl": t.pnl, "won": t.won, "edge": t.edge,
                        "date": t.date, "city": city,
                    })
            except Exception as e:
                pass

        if not all_trades:
            results.append({"params": params, "score": -999, "total_trades": 0})
            continue

        # Compute aggregate metrics
        total_trades = len(all_trades)
        wins = [t for t in all_trades if t["won"]]
        win_rate = len(wins) / total_trades * 100
        total_pnl = sum(t["pnl"] for t in all_trades)

        returns = [r["return_pct"] for r in city_results.values() if r.get("total_trades", 0) > 0]
        avg_return = np.mean(returns) if returns else 0
        median_return = np.median(returns) if returns else 0
        min_return = min(returns) if returns else 0
        max_dds = [r["max_drawdown"] for r in city_results.values() if r.get("total_trades", 0) > 0]
        worst_dd = max(max_dds) if max_dds else 100

        total_wins_usd = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
        total_losses_usd = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0))
        pf = total_wins_usd / total_losses_usd if total_losses_usd > 0 else 999

        trade_pnls = [t["pnl"] for t in all_trades]
        trade_sharpe = np.mean(trade_pnls) / np.std(trade_pnls) if np.std(trade_pnls) > 0 else 0

        cal_days = (datetime.strptime(BACKTEST_END, "%Y-%m-%d") -
                   datetime.strptime(BACKTEST_START, "%Y-%m-%d")).days
        tpw = total_trades / (cal_days / 7)

        score = (
            avg_return * 0.30 +
            median_return * 0.20 +
            min_return * 0.15 +
            pf * 5 * 0.15 +
            trade_sharpe * 50 * 0.10 +
            -worst_dd * 0.10
        )

        r = {
            "params": params,
            "score": round(score, 2),
            "total_trades": total_trades,
            "trades_per_week": round(tpw, 1),
            "win_rate": round(win_rate, 1),
            "avg_return_pct": round(avg_return, 2),
            "median_return_pct": round(median_return, 2),
            "min_return_pct": round(min_return, 2),
            "total_pnl": round(total_pnl, 2),
            "profit_factor": round(pf, 2),
            "trade_sharpe": round(trade_sharpe, 4),
            "worst_drawdown": round(worst_dd, 1),
            "per_city": city_results,
        }
        results.append(r)

        if score > best_score:
            best_score = score
            print(f"  ★ NEW BEST [{i+1}/{total}]: score={score:.1f} | "
                  f"return={avg_return:+.1f}% | WR={win_rate:.0f}% | "
                  f"PF={pf:.1f} | T/wk={tpw:.1f} | DD={worst_dd:.0f}% | "
                  f"kelly={params['kelly']} edge={params['min_edge']} "
                  f"agree={params['min_agreement']} pos={params['max_pos']}")

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{total} done")

    results.sort(key=lambda r: r["score"], reverse=True)

    # Print top 15
    print(f"\n{'='*70}")
    print(f"  TOP 15 CONFIGURATIONS")
    print(f"{'='*70}")
    print(f"  {'#':>3} {'Score':>7} {'Return':>8} {'Win%':>6} {'PF':>6} "
          f"{'T/wk':>6} {'DD':>5} | {'Kelly':>6} {'Edge':>6} {'Agree':>6} {'Pos%':>5}")
    print(f"  {'─'*3} {'─'*7} {'─'*8} {'─'*6} {'─'*6} "
          f"{'─'*6} {'─'*5} | {'─'*6} {'─'*6} {'─'*6} {'─'*5}")

    for rank, r in enumerate(results[:15], 1):
        p = r["params"]
        print(f"  {rank:>3} {r['score']:>7.1f} {r['avg_return_pct']:>+7.1f}% "
              f"{r['win_rate']:>5.0f}% {r['profit_factor']:>5.1f}  "
              f"{r['trades_per_week']:>5.1f} {r['worst_drawdown']:>4.0f}% | "
              f"{p['kelly']:>5.2f} {p['min_edge']:>5.2f} "
              f"{p['min_agreement']:>5.2f} {p['max_pos']:>5.2f}")

    # ═══════════════════════════════════════════════════
    # PHASE 2: Spread/Slippage Sensitivity
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Spread/Slippage Sensitivity (Best Config)")
    print(f"{'='*70}\n")

    best_p = results[0]["params"].copy()
    sensitivity = []

    for spread, slip, label in [
        (0.03, 0.01, "Optimistic"),
        (0.06, 0.02, "Baseline"),
        (0.08, 0.03, "Pessimistic"),
        (0.10, 0.04, "Very pessimistic"),
        (0.12, 0.05, "Worst case"),
    ]:
        params = best_p.copy()
        params["spread"] = spread
        params["slippage"] = slip

        all_t = []
        for city in TEST_CITIES:
            try:
                out = run_fast_backtest(city, engines[city], city_actuals[city], params)
                if out:
                    res, _ = out
                    for t in res.trades:
                        all_t.append({"pnl": t.pnl, "won": t.won})
            except:
                pass

        if all_t:
            wr = len([t for t in all_t if t["won"]]) / len(all_t) * 100
            tp = sum(t["pnl"] for t in all_t)
            tw = sum(t["pnl"] for t in all_t if t["pnl"] > 0)
            tl = abs(sum(t["pnl"] for t in all_t if t["pnl"] <= 0))
            pf = tw / tl if tl > 0 else 999
            print(f"  {label:20s}: PnL=${tp:+8.2f} | WR={wr:.0f}% | PF={pf:.1f} | Trades={len(all_t)}")
            sensitivity.append({"label": label, "spread": spread, "slippage": slip,
                               "pnl": round(tp, 2), "win_rate": round(wr, 1),
                               "profit_factor": round(pf, 1), "trades": len(all_t)})

    # ═══════════════════════════════════════════════════
    # PHASE 3: All 20 cities with best config
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 3: All 20 Cities (Best Config)")
    print(f"{'='*70}\n")

    all_cities = list(config.STATIONS.keys())
    all_engines, all_actuals = pre_calibrate_cities(
        [c for c in all_cities if c not in engines]
    )
    # Merge
    engines.update(all_engines)
    city_actuals.update(all_actuals)

    best_p_for_all = results[0]["params"].copy()
    city_20_results = {}
    profitable = []
    unprofitable = []

    for city in all_cities:
        try:
            out = run_fast_backtest(city, engines[city], city_actuals[city], best_p_for_all)
            if out:
                res, bt = out
                city_20_results[city] = {
                    "return_pct": round(res.total_return_pct, 1),
                    "max_drawdown": round(res.max_drawdown_pct, 1),
                    "win_rate": round(res.win_rate, 0),
                    "total_trades": res.total_trades,
                    "profit_factor": round(res.profit_factor, 1),
                    "avg_edge": round(res.avg_edge, 1),
                    "final_bankroll": round(bt.bankroll, 2),
                }
                if res.total_return_pct > 0:
                    profitable.append(city)
                else:
                    unprofitable.append(city)
        except Exception as e:
            city_20_results[city] = {"error": str(e)}

    print(f"  {'City':<16} {'Trades':>7} {'Return':>10} {'Win%':>7} {'PF':>6} {'DD':>6} {'Final$':>8}")
    print(f"  {'─'*16} {'─'*7} {'─'*10} {'─'*7} {'─'*6} {'─'*6} {'─'*8}")

    for city in sorted(city_20_results.keys(),
                      key=lambda c: city_20_results[c].get("return_pct", -999), reverse=True):
        r = city_20_results[city]
        if r.get("total_trades", 0) > 0:
            print(f"  {city:<16} {r['total_trades']:>7} {r['return_pct']:>+9.1f}% "
                  f"{r['win_rate']:>6.0f}% {r['profit_factor']:>5.1f} "
                  f"{r['max_drawdown']:>5.0f}% ${r['final_bankroll']:>7.2f}")

    print(f"\n  Profitable: {len(profitable)}/20 | Unprofitable: {len(unprofitable)}/20")

    # ═══════════════════════════════════════════════════
    # SAVE & CHART
    # ═══════════════════════════════════════════════════
    os.makedirs("results", exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "configs_tested": total,
        "best_params": results[0]["params"],
        "best_score": results[0]["score"],
        "best_metrics": {
            "avg_return_pct": results[0]["avg_return_pct"],
            "win_rate": results[0]["win_rate"],
            "profit_factor": results[0]["profit_factor"],
            "trades_per_week": results[0]["trades_per_week"],
            "worst_drawdown": results[0]["worst_drawdown"],
        },
        "top_15": [{
            "rank": i+1, "params": r["params"], "score": r["score"],
            "avg_return_pct": r["avg_return_pct"], "win_rate": r["win_rate"],
            "profit_factor": r["profit_factor"], "trades_per_week": r["trades_per_week"],
        } for i, r in enumerate(results[:15])],
        "sensitivity": sensitivity,
        "per_city_20": city_20_results,
        "profitable_cities": profitable,
        "unprofitable_cities": unprofitable,
    }

    with open("results/optimization_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Chart
    create_charts(results[:20], sensitivity, city_20_results)

    # Final summary
    bp = results[0]["params"]
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Optimal config.py values:")
    print(f"    KELLY_FRACTION         = {bp['kelly']}")
    print(f"    MIN_EDGE_PCT           = {bp['min_edge']}")
    print(f"    MIN_ENSEMBLE_AGREEMENT = {bp['min_agreement']}")
    print(f"    MAX_POSITION_PCT       = {bp['max_pos']}")
    print(f"")
    print(f"  vs Current config.py:")
    print(f"    KELLY_FRACTION         = 0.25")
    print(f"    MIN_EDGE_PCT           = 0.05")
    print(f"    MIN_ENSEMBLE_AGREEMENT = 0.60")
    print(f"    MAX_POSITION_PCT       = 0.10")
    print(f"{'='*70}\n")

    return output


def create_charts(top_results, sensitivity, city_results):
    """Create optimization charts."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Parameter Optimization Results — Polymarket Weather Bot",
                 fontsize=15, fontweight="bold", y=0.98)

    # 1. Top configs
    ax1 = axes[0, 0]
    rets = [r["avg_return_pct"] for r in top_results]
    colors = ["#4CAF50" if r > 0 else "#FF5252" for r in rets]
    ax1.bar(range(1, len(rets)+1), rets, color=colors, alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Config Rank")
    ax1.set_ylabel("Avg Return (%)")
    ax1.set_title("Top 20 Configs by Composite Score", fontweight="bold")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Sensitivity
    ax2 = axes[0, 1]
    if sensitivity:
        labels = [s["label"] for s in sensitivity]
        pnls = [s["pnl"] for s in sensitivity]
        pfs = [s["profit_factor"] for s in sensitivity]
        x = range(len(labels))
        w = 0.35
        ax2.bar([i-w/2 for i in x], pnls, w, label="Total PnL ($)", color="#2196F3", alpha=0.85)
        ax2t = ax2.twinx()
        ax2t.bar([i+w/2 for i in x], pfs, w, label="Profit Factor", color="#FF9800", alpha=0.85)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax2.set_ylabel("Total PnL ($)", color="#2196F3")
        ax2t.set_ylabel("Profit Factor", color="#FF9800")
        ax2.set_title("Sensitivity to Market Friction", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

    # 3. Per-city returns
    ax3 = axes[1, 0]
    cd = [(c, r.get("return_pct", 0)) for c, r in city_results.items() if r.get("total_trades", 0) > 0]
    cd.sort(key=lambda x: x[1], reverse=True)
    if cd:
        ns, rs = zip(*cd)
        cols = ["#4CAF50" if r > 0 else "#FF5252" for r in rs]
        ax3.barh(range(len(ns)), rs, color=cols, alpha=0.85)
        ax3.set_yticks(range(len(ns)))
        ax3.set_yticklabels(ns, fontsize=8)
        ax3.set_xlabel("Return (%)")
        ax3.set_title("Per-City Returns (20 Cities, Optimized)", fontweight="bold")
        ax3.axvline(x=0, color="black", linewidth=0.5)
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis="x")

    # 4. Kelly × Edge scatter
    ax4 = axes[1, 1]
    ks = [r["params"]["kelly"] for r in top_results]
    es = [r["params"]["min_edge"] for r in top_results]
    rs2 = [r["avg_return_pct"] for r in top_results]
    sc = ax4.scatter(ks, es, c=rs2, cmap="RdYlGn", s=120, edgecolors="black", linewidth=0.5)
    ax4.set_xlabel("Kelly Fraction")
    ax4.set_ylabel("Min Edge Threshold")
    ax4.set_title("Kelly × Edge → Return (Top 20)", fontweight="bold")
    plt.colorbar(sc, ax=ax4, label="Avg Return (%)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/optimization_charts.png", dpi=150, bbox_inches="tight")
    print(f"\nCharts saved to results/optimization_charts.png")
    plt.close()


if __name__ == "__main__":
    run_optimization()
