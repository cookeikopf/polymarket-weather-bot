#!/usr/bin/env python3
"""
Polymarket Weather Prediction Bot V3 - Main Entry Point
=========================================================

V3: Live-trading-ready with data-driven strategy from paper trading analysis.

Strategy: BUY_NO only, min entry 0.55, min edge 10%, daily loss limit $20.

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
        "equity_curve": result.equity_curve[-20:],
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
    """Run live trading with V3 strategy."""
    from live_trader import LiveTrader

    if not config.PRIVATE_KEY:
        print("ERROR: Set POLYMARKET_PRIVATE_KEY in .env file")
        print("  1. Export your key from https://reveal.polymarket.com")
        print("  2. Copy .env.example to .env and fill in your values")
        sys.exit(1)

    if not config.FUNDER_ADDRESS:
        print("WARNING: POLYMARKET_FUNDER_ADDRESS not set in .env")
        print("  Your funder address is your Polymarket profile address")
        print("  (NOT your MetaMask deposit address!)")
        print("  Find it at: polymarket.com/profile/<YOUR_ADDRESS>")
        print("  Continuing in EOA mode...\n")

    print("\n" + "=" * 60)
    print("  LIVE TRADING MODE V3 — Real money at risk!")
    print("=" * 60)
    print(f"  Duration:       {duration} minutes")
    print(f"  Bankroll:       ${config.LIVE_BANKROLL:.2f}")
    print(f"  Strategy:       {'BUY_YES+BUY_NO' if config.ALLOW_BUY_YES else 'BUY_NO ONLY'}")
    print(f"  Min Edge:       {config.MIN_EDGE_PCT:.0%}")
    print(f"  Min Entry:      {config.MIN_ENTRY_PRICE}")
    print(f"  Max per trade:  ${config.MAX_TRADE_SIZE_USDC:.2f}")
    print(f"  Max positions:  {config.MAX_CONCURRENT_POSITIONS}")
    print(f"  Max exposure:   {config.MAX_TOTAL_EXPOSURE:.0%} = ${config.LIVE_BANKROLL * config.MAX_TOTAL_EXPOSURE:.2f}")
    print(f"  Max drawdown:   {config.MAX_DRAWDOWN_PCT:.0%} = ${config.LIVE_BANKROLL * config.MAX_DRAWDOWN_PCT:.2f}")
    print(f"  Daily loss cap: ${config.MAX_DAILY_LOSS_USDC:.2f}")
    print(f"  Funder:         {config.FUNDER_ADDRESS[:10] + '...' if config.FUNDER_ADDRESS else 'EOA'}")
    print("=" * 60)
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
    print(f"\n{'='*55}")
    print(f"  Polymarket Weather Bot V3 - Configuration")
    print(f"{'='*55}")
    print(f"  Paper Mode:           {config.PAPER_TRADING}")
    print(f"  Strategy:             {'BUY_YES+BUY_NO' if config.ALLOW_BUY_YES else 'BUY_NO ONLY'}")
    print(f"  Stations:             {list(config.STATIONS.keys())}")
    print(f"  Weather Models:       {len(config.WEATHER_MODELS)}")
    print(f"  Min Edge:             {config.MIN_EDGE_PCT:.0%}")
    print(f"  Min Entry Price:      {config.MIN_ENTRY_PRICE}")
    print(f"  Max Entry Price:      {config.MAX_ENTRY_PRICE}")
    print(f"  Kelly Fraction:       {config.KELLY_FRACTION}")
    print(f"  Max Position:         {config.MAX_POSITION_PCT:.0%}")
    print(f"  Max Concurrent:       {config.MAX_CONCURRENT_POSITIONS}")
    print(f"  Max Drawdown:         {config.MAX_DRAWDOWN_PCT:.0%}")
    print(f"  Daily Loss Limit:     ${config.MAX_DAILY_LOSS_USDC}")
    print(f"  Scan Interval:        {config.SCAN_INTERVAL_SECONDS}s")
    print(f"  Live Bankroll:        ${config.LIVE_BANKROLL}")
    print(f"  Max Trade Size:       ${config.MAX_TRADE_SIZE_USDC}")
    print(f"  Scan Days Ahead:      {config.MARKET_SCAN_DAYS_AHEAD}")
    print(f"  Private Key Set:      {'Yes' if config.PRIVATE_KEY else 'No'}")
    print(f"  Funder Address Set:   {'Yes' if config.FUNDER_ADDRESS else 'No'}")
    print(f"  Signature Type:       {config.SIGNATURE_TYPE}")
    print(f"{'='*55}\n")


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
