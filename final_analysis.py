#!/usr/bin/env python3
"""
Final Analysis & Visualization
================================
Runs the optimized backtest and creates performance charts.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import config
from backtester import Backtester


def run_final_analysis():
    """Run final optimized backtest and generate report."""
    
    # Apply optimal parameters (Moderate profile - best risk/reward)
    config.MIN_EDGE_PCT = 0.07
    config.KELLY_FRACTION = 0.20
    config.MAX_POSITION_PCT = 0.08
    config.MIN_ENSEMBLE_AGREEMENT = 0.55
    config.SIM_SPREAD = 0.06
    config.SIM_SLIPPAGE = 0.02
    config.SIM_MARKET_NOISE = 0.015
    config.MAX_TRADE_SIZE_USDC = 100.0
    config.MAX_DRAWDOWN_PCT = 0.15
    
    print("\n" + "="*70)
    print("  FINAL OPTIMIZED BACKTEST")
    print("  Optimal Profile: Moderate-Balanced")
    print("="*70)
    
    # Run backtest
    bt = Backtester("NYC", initial_bankroll=1000.0)
    result = bt.run_backtest(
        start_date="2024-01-01",
        end_date="2026-03-15",
        verbose=True,
    )
    bt.print_results(result)
    
    # Generate charts
    os.makedirs("results", exist_ok=True)
    create_performance_charts(result, bt)
    
    # Generate detailed report
    generate_report(result, bt)
    
    return result


def create_performance_charts(result, bt):
    """Create performance visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Polymarket Weather Bot - Backtest Results", fontsize=16, fontweight="bold")
    
    # 1. Equity Curve
    ax1 = axes[0, 0]
    equity = result.equity_curve
    ax1.plot(equity, color="#2196F3", linewidth=1.5, label="Equity")
    ax1.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Starting Capital")
    ax1.fill_between(range(len(equity)), 1000, equity, alpha=0.1, color="#2196F3")
    ax1.set_title("Equity Curve", fontweight="bold")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / np.maximum(peak, 1) * 100
    ax2.fill_between(range(len(drawdown)), 0, -drawdown, color="#FF5252", alpha=0.5)
    ax2.axhline(y=-config.MAX_DRAWDOWN_PCT * 100, color="red", linestyle="--", alpha=0.7, label=f"Max DD Limit ({config.MAX_DRAWDOWN_PCT*100:.0f}%)")
    ax2.set_title("Drawdown", fontweight="bold")
    ax2.set_xlabel("Trading Days")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Trade P&L Distribution
    ax3 = axes[1, 0]
    if result.trades:
        pnls = [t.pnl for t in result.trades]
        colors = ["#4CAF50" if p > 0 else "#FF5252" for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax3.axhline(y=0, color="black", linewidth=0.5)
        ax3.set_title(f"Individual Trade P&L (n={len(pnls)})", fontweight="bold")
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("P&L ($)")
        ax3.grid(True, alpha=0.3)
    
    # 4. Edge Distribution
    ax4 = axes[1, 1]
    if result.trades:
        edges = [t.edge * 100 for t in result.trades]
        winning_edges = [t.edge * 100 for t in result.trades if t.won]
        losing_edges = [t.edge * 100 for t in result.trades if not t.won]
        
        bins = np.linspace(0, max(edges) + 1, 15)
        ax4.hist(winning_edges, bins=bins, alpha=0.6, color="#4CAF50", label=f"Wins ({len(winning_edges)})")
        ax4.hist(losing_edges, bins=bins, alpha=0.6, color="#FF5252", label=f"Losses ({len(losing_edges)})")
        ax4.set_title("Edge Distribution by Outcome", fontweight="bold")
        ax4.set_xlabel("Edge at Entry (%)")
        ax4.set_ylabel("Number of Trades")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/performance_chart.png", dpi=150, bbox_inches="tight")
    print(f"\nChart saved to results/performance_chart.png")
    plt.close()


def generate_report(result, bt):
    """Generate detailed text report."""
    
    report = {
        "title": "Polymarket Weather Prediction Bot - Final Analysis",
        "timestamp": datetime.now().isoformat(),
        "strategy": {
            "approach": "Multi-Model Ensemble Weather Forecasting + Monte Carlo Probability Estimation",
            "models_used": config.WEATHER_MODELS,
            "calibration": {
                "ecmwf_ifs025": {"bias": "1.08°F", "RMSE": "2.25°F", "MAE": "1.63°F"},
                "icon_seamless": {"bias": "-0.03°F", "RMSE": "2.25°F", "MAE": "1.75°F"},
                "best_match": {"bias": "-0.77°F", "RMSE": "2.56°F", "MAE": "1.98°F"},
            },
            "edge_detection": "Compare ML probability distribution vs market prices",
            "position_sizing": "Fractional Kelly Criterion (0.20x)",
        },
        "parameters": {
            "min_edge": f"{config.MIN_EDGE_PCT:.0%}",
            "kelly_fraction": config.KELLY_FRACTION,
            "max_position_pct": f"{config.MAX_POSITION_PCT:.0%}",
            "min_ensemble_agreement": f"{config.MIN_ENSEMBLE_AGREEMENT:.0%}",
            "max_trade_size": f"${config.MAX_TRADE_SIZE_USDC}",
            "max_drawdown": f"{config.MAX_DRAWDOWN_PCT:.0%}",
        },
        "results": {
            "total_return_pct": round(result.total_return_pct, 2),
            "total_pnl": round(result.total_pnl, 2),
            "max_drawdown_pct": round(result.max_drawdown_pct, 2),
            "win_rate": round(result.win_rate, 1),
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "avg_edge": round(result.avg_edge, 1),
            "avg_win": round(result.avg_win, 2),
            "avg_loss": round(result.avg_loss, 2),
            "profit_factor": round(result.profit_factor, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 2),
            "calmar_ratio": round(result.calmar_ratio, 2),
        },
        "recommendations": {
            "readiness": "READY FOR PAPER TRADING",
            "next_steps": [
                "1. Run paper trading for 2-4 weeks with real Polymarket data",
                "2. Monitor edge accuracy (predicted edge vs actual P&L)",
                "3. Start live trading with $50-100 initial capital",
                "4. Scale position sizes gradually as confidence builds",
                "5. Re-calibrate model weights monthly",
            ],
            "risks": [
                "Weather market liquidity is limited ($2.5K-$15K typical)",
                "Market efficiency may reduce edges over time",
                "Model calibration may degrade in unusual weather patterns",
                "Slippage on multi-outcome markets can be significant",
            ],
        },
    }
    
    with open("results/final_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to results/final_report.json")


if __name__ == "__main__":
    run_final_analysis()
