#!/usr/bin/env python3
"""
Polymarket Weather Bot V6 — CLI Entry Point
=============================================

Usage:
  python main.py paper              # Paper trading (continuous)
  python main.py live               # Live trading (real orders)
  python main.py scan               # Single scan cycle (paper)
  python main.py backtest           # Quick backtest (14 days, 5 stations)
  python main.py backtest full 60   # Full backtest (60 days, all stations)
  python main.py status             # Show current positions & P&L
"""

import sys
import os
import signal

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import log
import config as cfg


def cmd_paper(duration: int = 1440):
    """Run paper trading continuously."""
    from trader import LiveTrader
    trader = LiveTrader(paper_mode=True)
    trader.initialize()
    trader.run_continuous(duration_minutes=duration)


def cmd_live(duration: int = 1440):
    """Run live trading (real USDC)."""
    from trader import LiveTrader

    # Pre-flight checks
    if not cfg.PRIVATE_KEY:
        log.error("POLYMARKET_PRIVATE_KEY not set in .env")
        sys.exit(1)
    if not cfg.OPEN_METEO_API_KEY:
        log.warning("OPEN_METEO_API_KEY not set — using free tier (rate limited)")

    log.info("="*55)
    log.info("  ⚠  LIVE MODE — Real USDC at risk")
    log.info(f"  Bankroll: ${cfg.BANKROLL}")
    log.info(f"  Max daily loss: ${cfg.MAX_DAILY_LOSS_USDC}")
    log.info(f"  Max drawdown: {cfg.MAX_DRAWDOWN_PCT:.0%}")
    log.info(f"  Order strategy: {cfg.ORDER_STRATEGY}")
    log.info("="*55)
    log.info("  Starting in 10 seconds... Ctrl+C to abort.")

    import time
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        log.info("  Aborted.")
        sys.exit(0)

    trader = LiveTrader(paper_mode=False)
    trader.initialize()
    trader.run_continuous(duration_minutes=duration)


def cmd_scan():
    """Single scan cycle (paper mode)."""
    from trader import LiveTrader
    trader = LiveTrader(paper_mode=True)
    trader.initialize()
    signals = trader.run_scan_cycle()
    trader.print_report()
    log.info(f"\nFound {len(signals)} signals this cycle.")


def cmd_backtest(mode: str = "quick", days: int = 14, stations: list = None):
    """Run backtest."""
    from backtest import quick_backtest, full_backtest, Backtester

    if mode == "full":
        result = full_backtest(days=days)
    elif stations:
        bt = Backtester(stations=stations, api_delay=0.3)
        from datetime import datetime, timedelta
        end = datetime.now() - timedelta(days=3)
        start = end - timedelta(days=days)
        result = bt.run(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    else:
        result = quick_backtest(days=days)

    log.info(f"\nBacktest complete: {result.total_trades} trades, ROI: {result.roi_pct:+.1f}%")


def cmd_status():
    """Show current bot status from saved state."""
    import json

    for mode in ["paper", "live"]:
        path = f"{cfg.RESULTS_DIR}/{mode}_state.json"
        if not os.path.exists(path):
            continue

        with open(path) as f:
            state = json.load(f)

        log.info(f"\n{'='*55}")
        log.info(f"  {mode.upper()} MODE STATUS")
        log.info(f"{'='*55}")
        log.info(f"  Last update:  {state.get('timestamp', '?')}")
        log.info(f"  Bankroll:     ${state.get('bankroll', 0):.2f}")
        log.info(f"  Peak:         ${state.get('peak_bankroll', 0):.2f}")
        log.info(f"  Daily P&L:    ${state.get('daily_pnl', 0):+.2f}")
        log.info(f"  Daily trades: {state.get('daily_trades', 0)}")

        positions = state.get("positions", [])
        if positions:
            log.info(f"  Open positions: {len(positions)}")
            for p in positions:
                log.info(f"    {p['direction']} {p['outcome_name']} @ {p['entry_price']:.3f} | ${p['size_usd']:.2f}")

        trades = state.get("trades", [])
        if trades:
            recent = trades[-10:]
            total_pnl = sum(t.get("pnl", 0) for t in trades)
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            wr = wins / len(trades) * 100 if trades else 0
            log.info(f"  Total trades: {len(trades)} (WR: {wr:.0f}%) | Total P&L: ${total_pnl:+.2f}")
            log.info(f"  Last {len(recent)} trades:")
            for t in recent:
                log.info(f"    {t['direction']} {t['outcome_name']} | ${t.get('pnl', 0):+.2f} | {t.get('exit_reason', '?')}")


def print_help():
    log.info("""
╔═══════════════════════════════════════════════════╗
║  Polymarket Weather Bot V6                        ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  Commands:                                        ║
║    python main.py paper [minutes]                 ║
║      Run paper trading (default: 1440 = 24h)      ║
║                                                   ║
║    python main.py live [minutes]                  ║
║      Run live trading with real USDC              ║
║                                                   ║
║    python main.py scan                            ║
║      Single scan cycle (paper mode)               ║
║                                                   ║
║    python main.py backtest [days]                 ║
║      Quick backtest (5 stations, default 14d)     ║
║                                                   ║
║    python main.py backtest full [days]            ║
║      Full backtest (all 20 stations, default 60d) ║
║                                                   ║
║    python main.py status                          ║
║      Show saved positions & P&L                   ║
║                                                   ║
║  Environment (.env):                              ║
║    POLYMARKET_PRIVATE_KEY     (required for live)  ║
║    POLYMARKET_FUNDER_ADDRESS                      ║
║    POLYMARKET_SIGNATURE_TYPE  (default: 2)        ║
║    POLYMARKET_BANKROLL        (default: 100)      ║
║    POLY_BUILDER_API_KEY       (gasless trading)   ║
║    POLY_BUILDER_SECRET                            ║
║    POLY_BUILDER_PASSPHRASE                        ║
║    OPEN_METEO_API_KEY         (commercial tier)   ║
║    ORDER_STRATEGY             (taker|maker|adapt) ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
""")


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "paper":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 1440
        cmd_paper(duration)

    elif cmd == "live":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 1440
        cmd_live(duration)

    elif cmd == "scan":
        cmd_scan()

    elif cmd == "backtest":
        if len(sys.argv) > 2 and sys.argv[2] == "full":
            days = int(sys.argv[3]) if len(sys.argv) > 3 else 60
            cmd_backtest("full", days)
        else:
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
            cmd_backtest("quick", days)

    elif cmd == "status":
        cmd_status()

    elif cmd in ["help", "--help", "-h"]:
        print_help()

    else:
        log.error(f"Unknown command: {cmd}")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
