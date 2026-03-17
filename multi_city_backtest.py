#!/usr/bin/env python3
"""
Multi-City Backtest
====================
Runs backtests across ALL 20 Polymarket weather cities and computes
expected daily trade volume with the expanded market set.
"""

import numpy as np
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

import config
from backtester import Backtester, BacktestResult

# Apply optimal parameters
config.MIN_EDGE_PCT = 0.07
config.KELLY_FRACTION = 0.20
config.MAX_POSITION_PCT = 0.08
config.MIN_ENSEMBLE_AGREEMENT = 0.55
config.SIM_SPREAD = 0.06
config.SIM_SLIPPAGE = 0.02
config.SIM_MARKET_NOISE = 0.015
config.MAX_TRADE_SIZE_USDC = 100.0
config.MAX_DRAWDOWN_PCT = 0.20  # slightly higher for multi-city

# Shorter backtest period per city (API rate limits)
BACKTEST_START = "2025-01-01"
BACKTEST_END = "2026-03-15"


def run_multi_city_backtest():
    """Run backtest for each city and aggregate results."""
    
    all_cities = list(config.STATIONS.keys())
    print(f"\n{'='*70}")
    print(f"  MULTI-CITY BACKTEST: {len(all_cities)} Cities")
    print(f"  Period: {BACKTEST_START} to {BACKTEST_END}")
    print(f"{'='*70}\n")
    
    city_results = {}
    all_trades = []
    total_trading_days = set()
    
    for i, city in enumerate(all_cities):
        print(f"\n[{i+1}/{len(all_cities)}] {city} ({config.STATIONS[city]['name']})")
        print(f"{'─'*50}")
        
        try:
            bt = Backtester(city, initial_bankroll=1000.0)
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
                "avg_edge": result.avg_edge,
                "unit": config.STATIONS[city]["unit"],
            }
            
            # Collect trades with city label
            for t in result.trades:
                all_trades.append({
                    "city": city,
                    "date": t.date,
                    "pnl": t.pnl,
                    "won": t.won,
                    "edge": t.edge,
                })
                total_trading_days.add(t.date)
            
            unit_sym = "°F" if config.STATIONS[city]["unit"] == "fahrenheit" else "°C"
            print(f"  Return: {result.total_return_pct:+.1f}% | "
                  f"Trades: {result.total_trades} | "
                  f"Win Rate: {result.win_rate:.0f}% | "
                  f"PF: {result.profit_factor:.1f} | "
                  f"Unit: {unit_sym}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            city_results[city] = {"return_pct": 0, "total_trades": 0, "error": str(e)}
    
    # ═══════════════════════════════════════════════════
    # AGGREGATE ANALYSIS
    # ═══════════════════════════════════════════════════
    
    total_trades = len(all_trades)
    
    # Calendar days in backtest
    from datetime import datetime as dt
    d1 = dt.strptime(BACKTEST_START, "%Y-%m-%d")
    d2 = dt.strptime(BACKTEST_END, "%Y-%m-%d")
    calendar_days = (d2 - d1).days
    
    # Trading day counts
    date_counts = Counter(t["date"] for t in all_trades)
    days_with_trades = len(date_counts)
    
    # Per-day stats
    trades_per_day_values = list(date_counts.values())
    
    print(f"\n\n{'='*70}")
    print(f"  MULTI-CITY AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"  Cities tested:              {len(all_cities)}")
    print(f"  Calendar days:              {calendar_days}")
    print(f"  Total trades (all cities):  {total_trades}")
    print(f"  Days with at least 1 trade: {days_with_trades}")
    print(f"  Days without trades:        {calendar_days - days_with_trades}")
    print(f"{'─'*70}")
    print(f"  Trades per CALENDAR day:    {total_trades / calendar_days:.1f}")
    print(f"  Trades per TRADING day:     {np.mean(trades_per_day_values):.1f}" if trades_per_day_values else "  N/A")
    print(f"  Trading frequency:          {days_with_trades / calendar_days * 100:.0f}% of days")
    print(f"  Trades per WEEK:            {total_trades / (calendar_days / 7):.1f}")
    print(f"  Trades per MONTH:           {total_trades / (calendar_days / 30):.1f}")
    print(f"{'─'*70}")
    
    # P&L stats
    if all_trades:
        wins = [t for t in all_trades if t["won"]]
        losses = [t for t in all_trades if not t["won"]]
        total_pnl = sum(t["pnl"] for t in all_trades)
        win_rate = len(wins) / len(all_trades) * 100
        avg_pnl_per_trade = total_pnl / len(all_trades)
        
        print(f"  Total P&L (all cities):     ${total_pnl:+,.2f}")
        print(f"  Win rate:                   {win_rate:.1f}%")
        print(f"  Avg P&L per trade:          ${avg_pnl_per_trade:+.2f}")
        print(f"  Avg edge:                   {np.mean([t['edge'] for t in all_trades])*100:.1f}%")
    
    print(f"\n{'─'*70}")
    print(f"  PER-CITY BREAKDOWN:")
    print(f"{'─'*70}")
    print(f"  {'City':<16} {'Trades':>7} {'Return':>10} {'Win%':>7} {'PF':>6} {'Edge':>7}")
    print(f"  {'─'*16} {'─'*7} {'─'*10} {'─'*7} {'─'*6} {'─'*7}")
    
    sorted_cities = sorted(city_results.items(), key=lambda x: x[1].get("total_trades", 0), reverse=True)
    for city, res in sorted_cities:
        if res.get("total_trades", 0) > 0:
            print(f"  {city:<16} {res['total_trades']:>7} {res['return_pct']:>+9.1f}% "
                  f"{res['win_rate']:>6.0f}% {res['profit_factor']:>5.1f} {res['avg_edge']:>6.1f}%")
        else:
            print(f"  {city:<16} {'0':>7} {'N/A':>10} {'N/A':>7} {'N/A':>6} {'N/A':>7}")
    
    print(f"{'='*70}\n")
    
    # ═══════════════════════════════════════════════════
    # TRADE DISTRIBUTION CHART
    # ═══════════════════════════════════════════════════
    create_multi_city_charts(city_results, all_trades, date_counts, calendar_days)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/multi_city_results.json", "w") as f:
        json.dump({
            "summary": {
                "total_cities": len(all_cities),
                "total_trades": total_trades,
                "calendar_days": calendar_days,
                "trades_per_day": round(total_trades / calendar_days, 1),
                "trades_per_week": round(total_trades / (calendar_days / 7), 1),
                "trades_per_month": round(total_trades / (calendar_days / 30), 1),
                "trading_frequency_pct": round(days_with_trades / calendar_days * 100, 1),
                "total_pnl": round(sum(t["pnl"] for t in all_trades), 2) if all_trades else 0,
                "win_rate": round(len([t for t in all_trades if t["won"]]) / max(len(all_trades), 1) * 100, 1),
            },
            "per_city": city_results,
        }, f, indent=2)
    
    print(f"Results saved to results/multi_city_results.json")
    return city_results, all_trades


def create_multi_city_charts(city_results, all_trades, date_counts, calendar_days):
    """Create multi-city performance visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Multi-City Weather Bot — 20 Cities Backtest", fontsize=16, fontweight="bold")
    
    # 1. Trades per city
    ax1 = axes[0, 0]
    cities_sorted = sorted(
        [(c, r["total_trades"]) for c, r in city_results.items() if r.get("total_trades", 0) > 0],
        key=lambda x: x[1], reverse=True
    )
    if cities_sorted:
        names, counts = zip(*cities_sorted)
        colors = ["#2196F3" if config.STATIONS[n]["unit"] == "fahrenheit" else "#FF9800" for n in names]
        ax1.barh(range(len(names)), counts, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel("Total Trades")
        ax1.set_title("Trades per City (Blue=°F, Orange=°C)", fontweight="bold")
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis="x")
    
    # 2. Daily trade distribution
    ax2 = axes[0, 1]
    if date_counts:
        daily_values = list(date_counts.values())
        max_trades = max(daily_values) if daily_values else 1
        bins = range(0, max_trades + 2)
        ax2.hist(daily_values, bins=bins, color="#4CAF50", alpha=0.7, edgecolor="white")
        ax2.axvline(x=np.mean(daily_values), color="red", linestyle="--", label=f"Avg: {np.mean(daily_values):.1f}")
        ax2.set_xlabel("Trades per Day")
        ax2.set_ylabel("Number of Days")
        ax2.set_title("Daily Trade Volume Distribution", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Return per city
    ax3 = axes[1, 0]
    city_returns = sorted(
        [(c, r["return_pct"]) for c, r in city_results.items() if r.get("total_trades", 0) > 0],
        key=lambda x: x[1], reverse=True
    )
    if city_returns:
        names, returns = zip(*city_returns)
        colors = ["#4CAF50" if r > 0 else "#FF5252" for r in returns]
        ax3.barh(range(len(names)), returns, color=colors, alpha=0.8)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names, fontsize=8)
        ax3.set_xlabel("Return (%)")
        ax3.set_title("Return per City", fontweight="bold")
        ax3.axvline(x=0, color="black", linewidth=0.5)
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis="x")
    
    # 4. Cumulative trades over time
    ax4 = axes[1, 1]
    if all_trades:
        dates = sorted(set(t["date"] for t in all_trades))
        cumulative = []
        running = 0
        for d in dates:
            running += date_counts.get(d, 0)
            cumulative.append(running)
        ax4.plot(range(len(cumulative)), cumulative, color="#2196F3", linewidth=2)
        ax4.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.1, color="#2196F3")
        ax4.set_xlabel("Trading Days (chronological)")
        ax4.set_ylabel("Cumulative Trades")
        ax4.set_title("Cumulative Trade Count Over Time", fontweight="bold")
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/multi_city_chart.png", dpi=150, bbox_inches="tight")
    print(f"\nChart saved to results/multi_city_chart.png")
    plt.close()


if __name__ == "__main__":
    run_multi_city_backtest()
