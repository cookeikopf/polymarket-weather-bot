#!/usr/bin/env python3
"""
Parameter Optimizer
====================
Systematic grid search over key trading parameters to find the optimal
configuration for maximum profitability.

Tests combinations of:
- Kelly Fraction (position sizing aggressiveness)
- MIN_EDGE_PCT (minimum edge to trade)
- MIN_ENSEMBLE_AGREEMENT (confidence threshold)
- MAX_POSITION_PCT (max bet per trade)
- MAX_CONCURRENT_POSITIONS
- SIM_SPREAD / SIM_SLIPPAGE (market friction sensitivity)

Runs backtests on a representative set of cities and aggregates results.
"""

import numpy as np
import json
import os
import sys
import itertools
from datetime import datetime
from collections import Counter
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from backtester import Backtester, BacktestResult


# ═══════════════════════════════════════════════════════════════
# TEST CITIES: Representative sample (3 US °F + 3 EU/Asia °C)
# Keeps runtime manageable while covering different climates
# ═══════════════════════════════════════════════════════════════
TEST_CITIES = ["NYC", "Chicago", "Miami", "London", "Tokyo", "Buenos Aires"]
BACKTEST_START = "2025-06-01"  # 9 months of data
BACKTEST_END = "2026-03-15"
BASE_BANKROLL = 50.0  # Match real bankroll


def run_single_config(params: dict, cities: list = None, verbose: bool = False) -> dict:
    """Run backtest with given parameters across test cities."""
    cities = cities or TEST_CITIES

    # Apply parameters
    config.MIN_EDGE_PCT = params["min_edge"]
    config.KELLY_FRACTION = params["kelly"]
    config.MAX_POSITION_PCT = params["max_pos"]
    config.MIN_ENSEMBLE_AGREEMENT = params["min_agreement"]
    config.MAX_CONCURRENT_POSITIONS = params.get("max_concurrent", 8)
    config.MAX_TRADE_SIZE_USDC = params.get("max_trade_usd", 25.0)
    config.SIM_SPREAD = params.get("spread", 0.06)
    config.SIM_SLIPPAGE = params.get("slippage", 0.02)
    config.MAX_DRAWDOWN_PCT = params.get("max_dd", 0.20)

    all_trades = []
    city_results = {}
    city_equities = {}

    for city in cities:
        try:
            bt = Backtester(city, initial_bankroll=BASE_BANKROLL)
            result = bt.run_backtest(
                start_date=BACKTEST_START,
                end_date=BACKTEST_END,
                verbose=False,
            )

            city_results[city] = {
                "return_pct": result.total_return_pct,
                "max_drawdown": result.max_drawdown_pct,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "profit_factor": result.profit_factor,
                "sharpe": result.sharpe_ratio,
                "avg_edge": result.avg_edge,
                "final_bankroll": bt.bankroll,
            }
            city_equities[city] = result.equity_curve

            for t in result.trades:
                all_trades.append({
                    "city": city,
                    "date": t.date,
                    "pnl": t.pnl,
                    "won": t.won,
                    "edge": t.edge,
                    "size_usd": t.size_usd,
                    "direction": t.direction,
                })

        except Exception as e:
            if verbose:
                print(f"  ERROR {city}: {e}")
            city_results[city] = {"return_pct": 0, "total_trades": 0, "error": str(e)}

    # Aggregate metrics
    if not all_trades:
        return {"params": params, "total_trades": 0, "score": -999}

    total_trades = len(all_trades)
    wins = [t for t in all_trades if t["won"]]
    win_rate = len(wins) / total_trades * 100
    total_pnl = sum(t["pnl"] for t in all_trades)
    avg_pnl_per_trade = total_pnl / total_trades

    # Per-city average return (more robust than total PnL)
    returns = [r["return_pct"] for r in city_results.values() if r.get("total_trades", 0) > 0]
    avg_return = np.mean(returns) if returns else 0
    median_return = np.median(returns) if returns else 0
    min_return = min(returns) if returns else 0
    max_drawdowns = [r["max_drawdown"] for r in city_results.values() if r.get("total_trades", 0) > 0]
    avg_drawdown = np.mean(max_drawdowns) if max_drawdowns else 100
    worst_drawdown = max(max_drawdowns) if max_drawdowns else 100

    # Profit factor
    total_wins_usd = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
    total_losses_usd = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0))
    profit_factor = total_wins_usd / total_losses_usd if total_losses_usd > 0 else 999

    # Sharpe-like ratio of per-trade PnL
    trade_pnls = [t["pnl"] for t in all_trades]
    trade_sharpe = np.mean(trade_pnls) / np.std(trade_pnls) if np.std(trade_pnls) > 0 else 0

    # Calendar days
    d1 = datetime.strptime(BACKTEST_START, "%Y-%m-%d")
    d2 = datetime.strptime(BACKTEST_END, "%Y-%m-%d")
    cal_days = (d2 - d1).days
    trades_per_week = total_trades / (cal_days / 7)

    # Composite score: balance profitability, consistency, and risk
    # Penalize: high drawdown, low trade count, negative min return
    score = (
        avg_return * 0.30 +              # Average return across cities
        median_return * 0.20 +            # Median (robustness)
        min_return * 0.15 +               # Worst city (don't blow up anywhere)
        profit_factor * 5 * 0.15 +        # Profit factor (scaled)
        trade_sharpe * 50 * 0.10 +        # Risk-adjusted PnL
        -worst_drawdown * 0.10            # Drawdown penalty
    )

    result = {
        "params": params,
        "total_trades": total_trades,
        "trades_per_week": round(trades_per_week, 1),
        "win_rate": round(win_rate, 1),
        "avg_return_pct": round(avg_return, 2),
        "median_return_pct": round(median_return, 2),
        "min_return_pct": round(min_return, 2),
        "max_return_pct": round(max(returns) if returns else 0, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(avg_pnl_per_trade, 4),
        "profit_factor": round(profit_factor, 2),
        "trade_sharpe": round(trade_sharpe, 4),
        "avg_drawdown": round(avg_drawdown, 1),
        "worst_drawdown": round(worst_drawdown, 1),
        "score": round(score, 2),
        "per_city": city_results,
    }

    if verbose:
        print(f"  Score: {score:.1f} | Return: {avg_return:+.1f}% | "
              f"WR: {win_rate:.0f}% | PF: {profit_factor:.1f} | "
              f"Trades/wk: {trades_per_week:.1f} | DD: {worst_drawdown:.0f}%")

    return result


def run_grid_search():
    """Run systematic grid search over parameter space."""

    print(f"\n{'='*70}")
    print(f"  PARAMETER OPTIMIZATION — Grid Search")
    print(f"  Cities: {', '.join(TEST_CITIES)}")
    print(f"  Period: {BACKTEST_START} to {BACKTEST_END}")
    print(f"  Bankroll: ${BASE_BANKROLL}")
    print(f"{'='*70}\n")

    # Parameter grid
    param_grid = {
        "kelly": [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        "min_edge": [0.03, 0.05, 0.07, 0.10, 0.12],
        "min_agreement": [0.45, 0.50, 0.55, 0.60, 0.65],
        "max_pos": [0.05, 0.08, 0.10, 0.15],
    }

    # Fixed params for grid search
    fixed = {
        "max_concurrent": 8,
        "max_trade_usd": 25.0,
        "spread": 0.06,
        "slippage": 0.02,
        "max_dd": 0.20,
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)

    print(f"Testing {total} parameter combinations...\n")

    results = []
    best_score = -999
    best_config = None

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        params.update(fixed)

        label = (f"kelly={params['kelly']:.2f} edge={params['min_edge']:.2f} "
                 f"agree={params['min_agreement']:.2f} pos={params['max_pos']:.2f}")

        if (i + 1) % 20 == 0 or i == 0:
            print(f"[{i+1}/{total}] {label}")

        result = run_single_config(params, verbose=False)
        results.append(result)

        if result["score"] > best_score:
            best_score = result["score"]
            best_config = result
            print(f"  ★ NEW BEST [{i+1}]: score={best_score:.1f} | "
                  f"return={result['avg_return_pct']:+.1f}% | "
                  f"WR={result['win_rate']:.0f}% | "
                  f"PF={result['profit_factor']:.1f} | "
                  f"trades/wk={result['trades_per_week']:.1f}")

    # Sort by score
    results.sort(key=lambda r: r["score"], reverse=True)

    # Print top 10
    print(f"\n{'='*70}")
    print(f"  TOP 10 CONFIGURATIONS")
    print(f"{'='*70}")
    print(f"  {'#':>3} {'Score':>7} {'Return':>8} {'Win%':>6} {'PF':>5} "
          f"{'T/wk':>6} {'DD':>5} | {'Kelly':>6} {'Edge':>6} {'Agree':>6} {'Pos%':>5}")
    print(f"  {'─'*3} {'─'*7} {'─'*8} {'─'*6} {'─'*5} "
          f"{'─'*6} {'─'*5} | {'─'*6} {'─'*6} {'─'*6} {'─'*5}")

    for rank, r in enumerate(results[:10], 1):
        p = r["params"]
        print(f"  {rank:>3} {r['score']:>7.1f} {r['avg_return_pct']:>+7.1f}% "
              f"{r['win_rate']:>5.0f}% {r['profit_factor']:>5.1f} "
              f"{r['trades_per_week']:>5.1f} {r['worst_drawdown']:>4.0f}% | "
              f"{p['kelly']:>5.2f} {p['min_edge']:>5.2f} "
              f"{p['min_agreement']:>5.2f} {p['max_pos']:>5.2f}")

    # ═══════════════════════════════════════════════════
    # PHASE 2: Sensitivity analysis around best config
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Spread/Slippage Sensitivity (Best Config)")
    print(f"{'='*70}\n")

    best_params = results[0]["params"].copy()
    sensitivity_results = []

    spread_slippage_combos = [
        (0.03, 0.01, "Optimistic (tight spread)"),
        (0.06, 0.02, "Baseline (current)"),
        (0.08, 0.03, "Pessimistic (wide spread)"),
        (0.10, 0.04, "Very pessimistic"),
        (0.12, 0.05, "Worst case"),
    ]

    for spread, slippage, label in spread_slippage_combos:
        params = best_params.copy()
        params["spread"] = spread
        params["slippage"] = slippage
        print(f"  {label}: spread={spread:.0%}, slippage={slippage:.0%}")
        result = run_single_config(params, verbose=True)
        result["scenario"] = label
        sensitivity_results.append(result)

    # ═══════════════════════════════════════════════════
    # PHASE 3: Per-city analysis with best config
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Per-City Profitability (All 20 Cities, Best Config)")
    print(f"{'='*70}\n")

    all_cities = list(config.STATIONS.keys())
    full_result = run_single_config(best_params, cities=all_cities, verbose=False)

    print(f"  {'City':<16} {'Trades':>7} {'Return':>10} {'Win%':>7} {'PF':>6} {'DD':>6}")
    print(f"  {'─'*16} {'─'*7} {'─'*10} {'─'*7} {'─'*6} {'─'*6}")

    profitable_cities = []
    unprofitable_cities = []

    for city in sorted(full_result["per_city"].keys(),
                      key=lambda c: full_result["per_city"][c].get("return_pct", 0),
                      reverse=True):
        r = full_result["per_city"][city]
        if r.get("total_trades", 0) > 0:
            print(f"  {city:<16} {r['total_trades']:>7} {r['return_pct']:>+9.1f}% "
                  f"{r['win_rate']:>6.0f}% {r['profit_factor']:>5.1f} "
                  f"{r['max_drawdown']:>5.0f}%")
            if r["return_pct"] > 0:
                profitable_cities.append(city)
            else:
                unprofitable_cities.append(city)

    print(f"\n  Profitable cities: {len(profitable_cities)}/20")
    print(f"  Unprofitable cities: {len(unprofitable_cities)}/20")
    if unprofitable_cities:
        print(f"  Consider excluding: {', '.join(unprofitable_cities)}")

    # ═══════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════
    os.makedirs("results", exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "test_cities": TEST_CITIES,
        "period": f"{BACKTEST_START} to {BACKTEST_END}",
        "bankroll": BASE_BANKROLL,
        "total_configs_tested": total,
        "best_config": {
            "params": results[0]["params"],
            "score": results[0]["score"],
            "avg_return_pct": results[0]["avg_return_pct"],
            "win_rate": results[0]["win_rate"],
            "profit_factor": results[0]["profit_factor"],
            "trades_per_week": results[0]["trades_per_week"],
            "worst_drawdown": results[0]["worst_drawdown"],
        },
        "top_10": [
            {
                "rank": i + 1,
                "params": r["params"],
                "score": r["score"],
                "avg_return_pct": r["avg_return_pct"],
                "win_rate": r["win_rate"],
                "profit_factor": r["profit_factor"],
                "trades_per_week": r["trades_per_week"],
                "worst_drawdown": r["worst_drawdown"],
            }
            for i, r in enumerate(results[:10])
        ],
        "sensitivity": [
            {
                "scenario": s["scenario"],
                "spread": s["params"]["spread"],
                "slippage": s["params"]["slippage"],
                "avg_return_pct": s["avg_return_pct"],
                "profit_factor": s["profit_factor"],
                "win_rate": s["win_rate"],
            }
            for s in sensitivity_results
        ],
        "per_city_20": full_result["per_city"],
        "profitable_cities": profitable_cities,
        "unprofitable_cities": unprofitable_cities,
    }

    with open("results/optimization_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════
    # CHARTS
    # ═══════════════════════════════════════════════════
    create_optimization_charts(results[:20], sensitivity_results, full_result)

    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Best params:")
    bp = results[0]["params"]
    print(f"    KELLY_FRACTION     = {bp['kelly']}")
    print(f"    MIN_EDGE_PCT       = {bp['min_edge']}")
    print(f"    MIN_ENSEMBLE_AGREEMENT = {bp['min_agreement']}")
    print(f"    MAX_POSITION_PCT   = {bp['max_pos']}")
    print(f"  Score: {results[0]['score']:.1f}")
    print(f"  Avg Return: {results[0]['avg_return_pct']:+.1f}%")
    print(f"  Win Rate: {results[0]['win_rate']:.0f}%")
    print(f"  Profit Factor: {results[0]['profit_factor']:.1f}")
    print(f"  Trades/Week: {results[0]['trades_per_week']:.1f}")
    print(f"{'='*70}\n")

    return output


def create_optimization_charts(top_results, sensitivity_results, full_result):
    """Create optimization result visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Parameter Optimization Results", fontsize=16, fontweight="bold", y=0.98)

    # 1. Top configs comparison
    ax1 = axes[0, 0]
    ranks = range(1, len(top_results) + 1)
    scores = [r["score"] for r in top_results]
    returns = [r["avg_return_pct"] for r in top_results]
    colors = ["#4CAF50" if r > 0 else "#FF5252" for r in returns]

    bars = ax1.bar(ranks, returns, color=colors, alpha=0.8, edgecolor="white")
    ax1.set_xlabel("Configuration Rank")
    ax1.set_ylabel("Avg Return (%)")
    ax1.set_title("Top 20 Configurations by Return", fontweight="bold")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Sensitivity to spread/slippage
    ax2 = axes[0, 1]
    scenarios = [s["scenario"] for s in sensitivity_results]
    sens_returns = [s["avg_return_pct"] for s in sensitivity_results]
    sens_pf = [s["profit_factor"] for s in sensitivity_results]

    x = range(len(scenarios))
    width = 0.35
    bars1 = ax2.bar([i - width/2 for i in x], sens_returns, width, label="Return (%)", color="#2196F3", alpha=0.8)
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar([i + width/2 for i in x], sens_pf, width, label="Profit Factor", color="#FF9800", alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([s.split(" (")[0] for s in scenarios], rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Return (%)", color="#2196F3")
    ax2_twin.set_ylabel("Profit Factor", color="#FF9800")
    ax2.set_title("Sensitivity to Market Friction", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Per-city returns (20 cities)
    ax3 = axes[1, 0]
    city_data = [(c, r.get("return_pct", 0)) for c, r in full_result["per_city"].items()
                 if r.get("total_trades", 0) > 0]
    city_data.sort(key=lambda x: x[1], reverse=True)
    if city_data:
        names, rets = zip(*city_data)
        colors = ["#4CAF50" if r > 0 else "#FF5252" for r in rets]
        ax3.barh(range(len(names)), rets, color=colors, alpha=0.8)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names, fontsize=7)
        ax3.set_xlabel("Return (%)")
        ax3.set_title("Per-City Returns (Optimized Config)", fontweight="bold")
        ax3.axvline(x=0, color="black", linewidth=0.5)
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis="x")

    # 4. Kelly vs Return heatmap-style scatter
    ax4 = axes[1, 1]
    kellys = [r["params"]["kelly"] for r in top_results]
    edges = [r["params"]["min_edge"] for r in top_results]
    rets_scatter = [r["avg_return_pct"] for r in top_results]

    scatter = ax4.scatter(kellys, edges, c=rets_scatter, cmap="RdYlGn", s=100,
                         edgecolors="black", linewidth=0.5, alpha=0.9)
    ax4.set_xlabel("Kelly Fraction")
    ax4.set_ylabel("Min Edge")
    ax4.set_title("Kelly × Edge → Return (Top 20)", fontweight="bold")
    plt.colorbar(scatter, ax=ax4, label="Return (%)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/optimization_charts.png", dpi=150, bbox_inches="tight")
    print(f"\nCharts saved to results/optimization_charts.png")
    plt.close()


if __name__ == "__main__":
    run_grid_search()
