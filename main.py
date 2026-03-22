#!/usr/bin/env python3
"""
Polymarket Weather Prediction Bot V5 - Main Entry Point
=========================================================

V5: Professional API upgrade — ensemble probabilities + historical calibration + drift detection.
Dual strategy: LADDER BUY_YES (primary) + CONSERVATIVE BUY_NO (secondary).

Usage:
    python main.py backtest          Run full backtest
    python main.py optimize          Optimize parameters
    python main.py scan              Scan live markets (paper mode)
    python main.py paper [minutes]   Run paper trading
    python main.py live [minutes]    Run live trading (requires .env setup)
    python main.py calibrate         Run calibration (V5 if API key set, else V4)
    python main.py calibrate_v5      Run V5 calibration for all 20 stations
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
    print("  LIVE TRADING MODE V4 — Real money at risk!")
    print("=" * 60)
    print(f"  Duration:       {duration} minutes")
    print(f"  Bankroll:       ${config.LIVE_BANKROLL:.2f}")
    print(f"  Strategy:       LADDER + CONSERVATIVE NO")
    print(f"  Ladder:         {config.LADDER_BUCKETS} buckets x ${config.LADDER_BET_PER_BUCKET}/ea (max ${config.LADDER_MAX_ENTRY_PRICE})")
    print(f"  Conserv. NO:    entry {config.CONSERVATIVE_NO_MIN_ENTRY}-{config.CONSERVATIVE_NO_MAX_ENTRY}, edge >= {config.MIN_EDGE_PCT:.0%}")
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


def cmd_calibrate_v5():
    """Run V5 calibration (ensemble + historical forecast API) for all stations."""
    from weather_engine import WeatherEngine

    if not config.OPEN_METEO_API_KEY:
        print("WARNING: OPEN_METEO_API_KEY not set. V5 calibration requires Pro API access.")
        print("Set OPEN_METEO_API_KEY in .env to enable V5 features.")
        print("Falling back to V4 calibration...\n")
        cmd_calibrate()
        return

    print(f"\n{'='*60}")
    print(f"  V5 Calibration — Professional API")
    print(f"  Stations: {len(config.STATIONS)}")
    print(f"  Ensemble models: {getattr(config, 'ENSEMBLE_MODELS', [])}")
    print(f"{'='*60}\n")

    for station_id in config.STATIONS:
        print(f"\nV5 Calibrating {station_id}...")
        engine = WeatherEngine(station_id)
        stats = engine.calibrate_v5()

        if stats:
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            with open(f"{config.RESULTS_DIR}/v5_calibration_{station_id}.json", "w") as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"  V5 calibration saved to {config.RESULTS_DIR}/v5_calibration_{station_id}.json")
        else:
            print(f"  V5 calibration failed for {station_id}, falling back to V4...")
            engine.calibrate()


def cmd_status():
    """Show bot configuration and status."""
    v5_active = bool(config.OPEN_METEO_API_KEY)
    version = "V5 (Pro API)" if v5_active else "V4"

    print(f"\n{'='*60}")
    print(f"  Polymarket Weather Bot {version} - Dual Strategy")
    print(f"{'='*60}")
    print(f"  Paper Mode:           {config.PAPER_TRADING}")
    print(f"  Live Bankroll:        ${config.LIVE_BANKROLL}")
    print(f"")
    if v5_active:
        print(f"  --- V5 PROFESSIONAL API ---")
        print(f"  API Key:              Set")
        print(f"  Ensemble Models:      {getattr(config, 'ENSEMBLE_MODELS', [])}")
        print(f"  Previous Runs Days:   {getattr(config, 'PREVIOUS_RUNS_DAYS', 2)}")
        print(f"  Ensemble URL:         {config.OPEN_METEO_ENSEMBLE_URL}")
        print(f"")
    print(f"  --- LADDER STRATEGY (Primary) ---")
    print(f"  Enabled:              {config.LADDER_ENABLED}")
    print(f"  Buckets per ladder:   {config.LADDER_BUCKETS}")
    print(f"  Bet per bucket:       ${config.LADDER_BET_PER_BUCKET}")
    print(f"  Max entry price:      ${config.LADDER_MAX_ENTRY_PRICE}")
    print(f"  Max sets per cycle:   {config.LADDER_MAX_SETS_PER_CYCLE}")
    print(f"")
    print(f"  --- CONSERVATIVE NO (Secondary) ---")
    print(f"  Enabled:              {config.ALLOW_BUY_NO}")
    print(f"  Entry range:          {config.CONSERVATIVE_NO_MIN_ENTRY} - {config.CONSERVATIVE_NO_MAX_ENTRY}")
    print(f"  Min edge:             {config.MIN_EDGE_PCT:.0%}")
    print(f"")
    print(f"  --- RISK MANAGEMENT ---")
    print(f"  Max Concurrent Pos:   {config.MAX_CONCURRENT_POSITIONS}")
    print(f"  Max Exposure:         {config.MAX_TOTAL_EXPOSURE:.0%}")
    print(f"  Max Drawdown:         {config.MAX_DRAWDOWN_PCT:.0%}")
    print(f"  Daily Loss Limit:     ${config.MAX_DAILY_LOSS_USDC}")
    print(f"  Max Trade Size:       ${config.MAX_TRADE_SIZE_USDC}")
    print(f"")
    print(f"  --- OPERATIONAL ---")
    print(f"  Stations:             {len(config.STATIONS)} cities")
    print(f"  Weather Models:       {len(config.WEATHER_MODELS)}")
    print(f"  Scan Interval:        {config.SCAN_INTERVAL_SECONDS}s")
    print(f"  Scan Days Ahead:      {config.MARKET_SCAN_DAYS_AHEAD}")
    print(f"  Order Strategy:       {config.ORDER_STRATEGY}")
    print(f"  Private Key Set:      {'Yes' if config.PRIVATE_KEY else 'No'}")
    print(f"  Funder Address Set:   {'Yes' if config.FUNDER_ADDRESS else 'No'}")
    print(f"  Builder API Set:      {'Yes' if config.BUILDER_API_KEY else 'No'}")
    print(f"  Open-Meteo API Key:   {'Yes' if config.OPEN_METEO_API_KEY else 'No'}")
    print(f"{'='*60}\n")


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
    elif command == "calibrate_v5":
        cmd_calibrate_v5()
    elif command == "status":
        cmd_status()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
