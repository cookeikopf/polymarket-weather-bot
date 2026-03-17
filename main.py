#!/usr/bin/env python3
"""
Polymarket Weather Prediction Bot - Main Entry Point
=====================================================

REVOLUTIONARY APPROACH:
1. Multi-model ensemble from 8+ global NWP models (ECMWF, GFS, ICON, etc.)
2. Bayesian probability calibration using historical forecast errors
3. Monte Carlo simulation for probability distribution estimation
4. Kelly Criterion position sizing with fractional Kelly for safety
5. Edge detection: our probabilities vs. market consensus

Usage:
    python main.py backtest          Run full backtest
    python main.py optimize          Optimize parameters
    python main.py scan              Scan live markets (paper mode)
    python main.py paper [minutes]   Run paper trading
    python main.py live [minutes]    Run live trading (requires .env setup)
    python main.py calibrate         Run calibration only
    python main.py status            Show bot configuration
"""

import sys
import json
import os
from datetime import datetime

import config


def cmd_backtest():
    """Run full backtest."""
    from backtester import Backtester

    bt = Backtester("NYC")
    result = bt.run_backtest(verbose=True)
    bt.print_results(result)

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "station": "NYC",
        "total_return_pct": result.total_return_pct,
        "total_pnl": result.total_pnl,
        "max_drawdown_pct": result.max_drawdown_pct,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "profit_factor": result.profit_factor,
        "sharpe_ratio": result.sharpe_ratio,
        "calmar_ratio": result.calmar_ratio,
        "equity_curve": result.equity_curve[-20:],  # Last 20 points
    }
    with open(f"{config.RESULTS_DIR}/backtest_latest.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to {config.RESULTS_DIR}/backtest_latest.json")

    return result


def cmd_optimize():
    """Run parameter optimization."""
    from optimizer import ParameterOptimizer

    opt = ParameterOptimizer("NYC")
    results = opt.run_profile_optimization(verbose=True)

    # Save optimization results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    opt_data = {}
    for name, res in results.items():
        opt_data[name] = {
            "params": res.params,
            "score": res.score,
            "return": res.result.total_return_pct,
            "max_drawdown": res.result.max_drawdown_pct,
            "win_rate": res.result.win_rate,
            "trades": res.result.total_trades,
            "sharpe": res.result.sharpe_ratio,
        }

    with open(f"{config.RESULTS_DIR}/optimization_latest.json", "w") as f:
        json.dump(opt_data, f, indent=2)
    print(f"Optimization results saved to {config.RESULTS_DIR}/optimization_latest.json")


def cmd_scan():
    """Scan live markets (one cycle, paper mode)."""
    from live_trader import LiveTrader

    trader = LiveTrader(paper_mode=True)
    trader.initialize()
    signals = trader.run_scan_cycle()

    if signals:
        print(f"\nFound {len(signals)} signals:")
        for s in signals:
            print(f"  {s.direction} '{s.outcome.name}' | "
                  f"Edge: {s.edge:.1%} | Size: ${s.suggested_size_usd:.2f}")
    else:
        print("\nNo actionable signals found in current markets")


def cmd_paper(duration: int = 60):
    """Run paper trading."""
    from live_trader import LiveTrader

    trader = LiveTrader(paper_mode=True)
    trader.initialize()
    trader.run_continuous(duration_minutes=duration)


def cmd_live(duration: int = 60):
    """Run live trading."""
    from live_trader import LiveTrader

    if not config.PRIVATE_KEY:
        print("ERROR: Set POLYMARKET_PRIVATE_KEY in .env file")
        print("  echo 'POLYMARKET_PRIVATE_KEY=0x...' > .env")
        sys.exit(1)

    print("⚠️  LIVE TRADING MODE - Real money at risk!")
    print(f"  Duration: {duration} minutes")
    print(f"  Bankroll: ${config.BACKTEST_INITIAL_BANKROLL}")
    confirm = input("  Type 'YES' to confirm: ")
    if confirm != "YES":
        print("Aborted.")
        return

    trader = LiveTrader(paper_mode=False)
    trader.initialize()
    trader.run_continuous(duration_minutes=duration)


def cmd_calibrate():
    """Run weather model calibration."""
    from weather_engine import WeatherEngine

    for station_id in config.STATIONS:
        print(f"\nCalibrating {station_id}...")
        engine = WeatherEngine(station_id)
        stats = engine.calibrate()

        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        with open(f"{config.RESULTS_DIR}/calibration_{station_id}.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Calibration saved to {config.RESULTS_DIR}/calibration_{station_id}.json")


def cmd_status():
    """Show bot configuration and status."""
    print(f"\n{'='*50}")
    print(f"  Polymarket Weather Bot - Configuration")
    print(f"{'='*50}")
    print(f"  Paper Mode:           {config.PAPER_TRADING}")
    print(f"  Stations:             {list(config.STATIONS.keys())}")
    print(f"  Weather Models:       {len(config.WEATHER_MODELS)}")
    print(f"  Min Edge:             {config.MIN_EDGE_PCT:.0%}")
    print(f"  Kelly Fraction:       {config.KELLY_FRACTION}")
    print(f"  Max Position:         {config.MAX_POSITION_PCT:.0%}")
    print(f"  Max Drawdown:         {config.MAX_DRAWDOWN_PCT:.0%}")
    print(f"  Scan Interval:        {config.SCAN_INTERVAL_SECONDS}s")
    print(f"  Initial Bankroll:     ${config.BACKTEST_INITIAL_BANKROLL}")
    print(f"  Private Key Set:      {'Yes' if config.PRIVATE_KEY else 'No'}")
    print(f"{'='*50}\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "backtest":
        cmd_backtest()
    elif command == "optimize":
        cmd_optimize()
    elif command == "scan":
        cmd_scan()
    elif command == "paper":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        cmd_paper(duration)
    elif command == "live":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        cmd_live(duration)
    elif command == "calibrate":
        cmd_calibrate()
    elif command == "status":
        cmd_status()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
