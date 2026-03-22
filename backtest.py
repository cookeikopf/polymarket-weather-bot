"""
Historical Backtester V6
=========================
Backtests the dual-strategy (Ladder + Conservative NO) against real historical data.

Uses Open-Meteo APIs:
  - historical-forecast-api → what models predicted N days before target
  - archive-api → actual observed temperature (ground truth)
  - ensemble-api → not available historically, so we simulate spread from model disagreement

Simulates Polymarket-style temperature bucket markets with realistic spreads.
"""

import os
import json
import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict

import config as cfg
from weather import WeatherEngine
from strategy import EdgeDetector, TradingSignal, extract_bucket_temp
from markets import WeatherMarket, MarketOutcome, parse_bucket_edges
from utils import log

# ═══════════════════════════════════════════════════════════════════
# Simulated market generation
# ═══════════════════════════════════════════════════════════════════

def generate_bucket_edges(station_id: str, target_date: str) -> List[float]:
    """Generate realistic Polymarket-style bucket edges for a city/date."""
    station = cfg.STATIONS.get(station_id, {})
    is_f = station.get("unit") == "fahrenheit"

    # Estimate typical range from month + location
    month = int(target_date.split("-")[1])
    lat = station.get("lat", 40)

    # Rough climate baseline (°F)
    if abs(lat) < 25:  # tropical
        base = 85 if month in [6, 7, 8] else 78
        spread = 12
    elif lat > 0:  # northern hemisphere
        seasonal = [35, 38, 48, 58, 68, 78, 83, 82, 74, 62, 48, 38]
        base = seasonal[month - 1]
        spread = 16
    else:  # southern hemisphere (flip seasons)
        seasonal = [82, 80, 74, 62, 50, 42, 40, 44, 52, 62, 72, 80]
        base = seasonal[month - 1]
        spread = 16

    if not is_f:
        base = (base - 32) * 5 / 9
        spread = spread * 5 / 9

    step = 2 if is_f else 1
    n_buckets = 8 if is_f else 10

    # Center buckets around expected temperature
    center = round(base / step) * step
    start = center - (n_buckets // 2) * step

    edges = [start + i * step for i in range(n_buckets + 1)]
    return edges


def generate_simulated_market(station_id: str, target_date: str, our_probs: Dict[str, float],
                               actual_temp: float) -> Optional[WeatherMarket]:
    """Create a simulated market with prices derived from a naive model + noise."""
    station = cfg.STATIONS.get(station_id, {})
    is_f = station.get("unit") == "fahrenheit"
    unit = "°F" if is_f else "°C"

    edges = generate_bucket_edges(station_id, target_date)
    if not edges or len(edges) < 3:
        return None

    step = 2 if is_f else 1
    outcomes = []

    # Generate "market maker" probabilities — slightly different from our model
    # Simulates an imperfect market that we can find edges against
    noise_scale = cfg.SIM_SPREAD / 2
    mm_probs = {}

    for i in range(len(edges) - 1):
        low = edges[i]
        high = edges[i + 1]
        if is_f:
            label = f"{int(low)}-{int(high - 1)}{unit}"
        else:
            label = f"{int(low)}{unit}"

        # MM uses actual temp with large noise (simulating less accurate public model)
        mm_center = actual_temp + random.gauss(0, 2.5 if is_f else 1.4)
        bucket_center = (low + high) / 2
        dist = abs(bucket_center - mm_center)
        mm_prob = max(0.01, np.exp(-0.5 * (dist / (3.5 if is_f else 2.0)) ** 2))
        mm_probs[label] = mm_prob

    # Tail buckets
    tail_low_label = f"{int(edges[0])}{unit} or below"
    tail_high_label = f"{int(edges[-1])}{unit} or higher"

    tail_low_dist = max(0, edges[0] - actual_temp + random.gauss(0, 2))
    tail_high_dist = max(0, actual_temp - edges[-1] + random.gauss(0, 2))
    mm_probs[tail_low_label] = max(0.005, np.exp(-0.3 * tail_low_dist))
    mm_probs[tail_high_label] = max(0.005, np.exp(-0.3 * tail_high_dist))

    # Normalize MM probs
    total = sum(mm_probs.values())
    if total > 0:
        mm_probs = {k: v / total for k, v in mm_probs.items()}

    # Convert to prices (with overround)
    overround = 1.05 + random.uniform(-0.02, 0.03)
    prices = {k: min(0.98, max(0.01, v * overround)) for k, v in mm_probs.items()}

    # Build outcomes
    for label, price in prices.items():
        spread = cfg.SIM_SPREAD * (0.5 + random.random())
        half = spread / 2
        bid = max(0.005, price - half)
        ask = min(0.995, price + half)

        outcomes.append(MarketOutcome(
            token_id=f"sim_{station_id}_{target_date}_{label}",
            name=label,
            price=price,
            no_token_id=f"sim_no_{station_id}_{target_date}_{label}",
            clob_bid=round(bid, 4),
            clob_ask=round(ask, 4),
            clob_spread=round(ask - bid, 4),
        ))

    return WeatherMarket(
        event_id=f"sim_{station_id}_{target_date}",
        condition_id=f"sim_cond_{station_id}_{target_date}",
        slug=f"sim-temp-{station_id}-{target_date}",
        question=f"Simulated temperature market {station_id} {target_date}",
        description="",
        market_type="temperature_max",
        station_id=station_id,
        target_date=target_date,
        end_date=f"{target_date}T23:59:59Z",
        volume=random.uniform(1000, 50000),
        liquidity=random.uniform(500, 10000),
        outcomes=outcomes,
        neg_risk=True,
        tick_size="0.01",
        order_min_size=5.0,
        unit=unit,
    )


def determine_winning_bucket(market: WeatherMarket, actual_temp: float) -> Optional[str]:
    """Determine which outcome 'wins' based on actual temperature."""
    station = cfg.STATIONS.get(market.station_id, {})
    is_f = station.get("unit") == "fahrenheit"

    for o in market.outcomes:
        temp = extract_bucket_temp(o.name, is_f)
        if temp is None:
            continue

        # Check "or below" tail
        if "below" in o.name.lower() or "lower" in o.name.lower():
            nums = [int(x) for x in __import__("re").findall(r'-?\d+', o.name)]
            if nums and actual_temp < nums[0]:
                return o.name
            continue

        # Check "or higher" tail
        if "higher" in o.name.lower():
            nums = [int(x) for x in __import__("re").findall(r'-?\d+', o.name)]
            if nums and actual_temp >= nums[0]:
                return o.name
            continue

        # Check range bucket (e.g., "34-35°F")
        m = __import__("re").search(r'(-?\d+)\s*[-–]\s*(-?\d+)', o.name)
        if m:
            low = int(m.group(1))
            high = int(m.group(2))
            if low <= actual_temp <= high:
                return o.name
            continue

        # Single value bucket (e.g., "14°C")
        m = __import__("re").search(r'(-?\d+)\s*°\s*C', o.name)
        if m:
            val = int(m.group(1))
            if is_f:
                continue
            if val <= actual_temp < val + 1:
                return o.name
            continue

    return None


# ═══════════════════════════════════════════════════════════════════
# Backtest Position Tracking
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BacktestTrade:
    date: str
    station_id: str
    direction: str  # BUY_YES or BUY_NO
    strategy: str   # ladder or conservative_no
    outcome_name: str
    entry_price: float
    size_usd: float
    our_prob: float
    edge: float
    confidence: float
    # Resolved at settlement
    won: bool = False
    pnl: float = 0.0
    actual_temp: float = 0.0
    winning_bucket: str = ""


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    stations: List[str]
    initial_bankroll: float
    final_bankroll: float
    peak_bankroll: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    profit_factor: float
    sharpe_ratio: float
    ladder_trades: int
    ladder_pnl: float
    ladder_win_rate: float
    no_trades: int
    no_pnl: float
    no_win_rate: float
    roi_pct: float
    annualized_roi_pct: float
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    trades: List[BacktestTrade] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Main Backtester
# ═══════════════════════════════════════════════════════════════════

class Backtester:
    """Historical backtester using Open-Meteo historical forecast + archive APIs."""

    def __init__(self, stations: List[str] = None, initial_bankroll: float = None,
                 api_delay: float = 0.4):
        self.stations = stations or list(cfg.STATIONS.keys())
        self.bankroll = initial_bankroll or cfg.BACKTEST_INITIAL_BANKROLL
        self.initial_bankroll = self.bankroll
        self.peak_bankroll = self.bankroll

        self.engines: Dict[str, WeatherEngine] = {}
        self.edge_detector = EdgeDetector()
        self.api_delay = api_delay

        self.trades: List[BacktestTrade] = []
        self.daily_pnl: Dict[str, float] = {}
        self.daily_exposure: float = 0.0
        self.equity_curve: List[float] = [self.bankroll]

        for sid in self.stations:
            self.engines[sid] = WeatherEngine(sid)

    def run(self, start_date: str, end_date: str, forecast_lead_days: int = 1) -> BacktestResult:
        """Run backtest over date range.

        Args:
            start_date: YYYY-MM-DD start (inclusive)
            end_date: YYYY-MM-DD end (inclusive)
            forecast_lead_days: How many days before target we 'see' the forecast (1 = day before)
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        n_days = (end - start).days + 1

        log.info(f"\n{'='*60}")
        log.info(f"  BACKTEST V6")
        log.info(f"  {start_date} → {end_date} ({n_days} days)")
        log.info(f"  Stations: {len(self.stations)} | Bankroll: ${self.bankroll:.2f}")
        log.info(f"  Lead: {forecast_lead_days}d | Slippage: {cfg.SIM_SLIPPAGE}")
        log.info(f"{'='*60}\n")

        current = start
        day_count = 0

        while current <= end:
            target_date = current.strftime("%Y-%m-%d")
            forecast_date = (current - timedelta(days=forecast_lead_days)).strftime("%Y-%m-%d")
            day_count += 1

            if day_count % 7 == 0:
                log.info(f"  [{day_count}/{n_days}] {target_date} | Bankroll: ${self.bankroll:.2f} | Trades: {len(self.trades)}")

            daily_trades = self._process_day(target_date, forecast_date)
            self.daily_pnl[target_date] = sum(t.pnl for t in daily_trades)
            self.equity_curve.append(self.bankroll)

            # Safety: stop if bankroll depleted
            if self.bankroll < cfg.MIN_TRADE_SIZE_USDC:
                log.warning(f"  BANKROLL DEPLETED at {target_date}: ${self.bankroll:.2f}")
                break

            current += timedelta(days=1)

        return self._compile_results(start_date, end_date)

    def _process_day(self, target_date: str, forecast_date: str) -> List[BacktestTrade]:
        """Process one day: fetch forecasts, generate markets, find edges, settle."""
        day_trades = []
        daily_exposure = 0.0

        for station_id in self.stations:
            engine = self.engines[station_id]
            station = cfg.STATIONS[station_id]
            is_f = station.get("unit") == "fahrenheit"

            # 1. Fetch what models predicted
            forecasts = engine.fetch_historical_forecast(target_date, forecast_date)
            if not forecasts:
                continue
            time.sleep(self.api_delay)

            # 2. Fetch actual temperature (ground truth)
            actual = engine.fetch_actual_temp(target_date)
            if actual is None:
                continue
            time.sleep(self.api_delay)

            # 3. Compute our probability distribution
            stats = engine.compute_ensemble_stats(forecasts)

            edges = generate_bucket_edges(station_id, target_date)
            if not edges:
                continue

            # Simulate ensemble data from model disagreement
            sim_ensemble = {}
            for model, temp in forecasts.items():
                members = [temp + random.gauss(0, 1.5 if is_f else 0.8) for _ in range(10)]
                sim_ensemble[model] = members

            our_probs = engine.compute_bucket_probabilities(
                forecasts, sim_ensemble, edges, is_f
            )
            if not our_probs:
                continue

            # 4. Generate simulated market
            market = generate_simulated_market(station_id, target_date, our_probs, actual)
            if not market:
                continue

            # 5. Find edges
            signals = self.edge_detector.find_edges(
                market, our_probs, stats, self.bankroll, daily_exposure, days_to_res=1.0
            )
            if not signals:
                continue

            # 6. Settle trades immediately (we know the actual temperature)
            winning_bucket = determine_winning_bucket(market, actual)

            for signal in signals:
                # Check bankroll
                if self.bankroll < signal.suggested_size_usd:
                    continue
                if daily_exposure + signal.suggested_size_usd > self.bankroll * cfg.MAX_TOTAL_EXPOSURE:
                    continue

                # Apply slippage to entry
                slippage = cfg.SIM_SLIPPAGE * random.uniform(0, 1)
                if signal.direction == "BUY_YES":
                    entry = (signal.outcome.clob_ask if signal.outcome.clob_ask > 0 else signal.market_price) + slippage
                else:
                    entry = (1.0 - signal.outcome.clob_bid if signal.outcome.clob_bid > 0 else 1.0 - signal.market_price) + slippage

                entry = max(0.01, min(0.99, entry))

                # Determine outcome
                if signal.direction == "BUY_YES":
                    won = (winning_bucket is not None and
                           winning_bucket.lower().strip() == signal.outcome.name.lower().strip())
                    pnl = (1.0 - entry) * signal.suggested_size_usd / entry if won else -signal.suggested_size_usd
                else:
                    won = (winning_bucket is None or
                           winning_bucket.lower().strip() != signal.outcome.name.lower().strip())
                    pnl = (1.0 - entry) * signal.suggested_size_usd / entry if won else -signal.suggested_size_usd

                trade = BacktestTrade(
                    date=target_date, station_id=station_id,
                    direction=signal.direction, strategy=signal.strategy,
                    outcome_name=signal.outcome.name,
                    entry_price=entry, size_usd=signal.suggested_size_usd,
                    our_prob=signal.our_probability, edge=signal.edge,
                    confidence=signal.confidence, won=won, pnl=pnl,
                    actual_temp=actual, winning_bucket=winning_bucket or "none",
                )

                self.trades.append(trade)
                day_trades.append(trade)
                self.bankroll += pnl
                daily_exposure += signal.suggested_size_usd
                self.peak_bankroll = max(self.peak_bankroll, self.bankroll)

        return day_trades

    def _compile_results(self, start_date: str, end_date: str) -> BacktestResult:
        """Compile backtest statistics."""
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1

        # Max drawdown
        max_dd = 0.0
        peak = self.initial_bankroll
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio (daily returns)
        daily_returns = list(self.daily_pnl.values())
        if daily_returns and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)
        else:
            sharpe = 0.0

        # Strategy breakdown
        ladder = [t for t in self.trades if t.strategy == "ladder"]
        no_trades = [t for t in self.trades if t.strategy == "conservative_no"]

        n_days = max(1, (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days)
        roi = (self.bankroll - self.initial_bankroll) / self.initial_bankroll
        ann_roi = (1 + roi) ** (365 / n_days) - 1 if roi > -1 else -1

        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            stations=self.stations,
            initial_bankroll=self.initial_bankroll,
            final_bankroll=self.bankroll,
            peak_bankroll=self.peak_bankroll,
            max_drawdown_pct=max_dd,
            total_trades=len(self.trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(self.trades) if self.trades else 0,
            total_pnl=total_pnl,
            avg_pnl_per_trade=total_pnl / len(self.trades) if self.trades else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            sharpe_ratio=sharpe,
            ladder_trades=len(ladder),
            ladder_pnl=sum(t.pnl for t in ladder),
            ladder_win_rate=len([t for t in ladder if t.pnl > 0]) / len(ladder) if ladder else 0,
            no_trades=len(no_trades),
            no_pnl=sum(t.pnl for t in no_trades),
            no_win_rate=len([t for t in no_trades if t.pnl > 0]) / len(no_trades) if no_trades else 0,
            roi_pct=roi * 100,
            annualized_roi_pct=ann_roi * 100,
            daily_pnl=self.daily_pnl,
            trades=self.trades,
        )

        self._print_report(result)
        self._save_results(result)
        return result

    def _print_report(self, r: BacktestResult):
        log.info(f"\n{'='*60}")
        log.info(f"  BACKTEST RESULTS V6")
        log.info(f"{'='*60}")
        log.info(f"  Period:          {r.start_date} → {r.end_date}")
        log.info(f"  Stations:        {len(r.stations)}")
        log.info(f"  Initial:         ${r.initial_bankroll:.2f}")
        log.info(f"  Final:           ${r.final_bankroll:.2f}")
        log.info(f"  Peak:            ${r.peak_bankroll:.2f}")
        log.info(f"  Total P&L:       ${r.total_pnl:+.2f}")
        log.info(f"  ROI:             {r.roi_pct:+.1f}%")
        log.info(f"  Annualized ROI:  {r.annualized_roi_pct:+.1f}%")
        log.info(f"  Max Drawdown:    {r.max_drawdown_pct:.1%}")
        log.info(f"  Sharpe Ratio:    {r.sharpe_ratio:.2f}")
        log.info(f"  {'─'*50}")
        log.info(f"  Total Trades:    {r.total_trades}")
        log.info(f"  Win Rate:        {r.win_rate:.1%}")
        log.info(f"  Avg P&L/Trade:   ${r.avg_pnl_per_trade:+.3f}")
        log.info(f"  Profit Factor:   {r.profit_factor:.2f}")
        log.info(f"  {'─'*50}")
        log.info(f"  LADDER:  {r.ladder_trades} trades | ${r.ladder_pnl:+.2f} | WR {r.ladder_win_rate:.1%}")
        log.info(f"  NO:      {r.no_trades} trades | ${r.no_pnl:+.2f} | WR {r.no_win_rate:.1%}")
        log.info(f"{'='*60}\n")

        # Best/worst days
        if r.daily_pnl:
            sorted_days = sorted(r.daily_pnl.items(), key=lambda x: x[1])
            log.info("  Worst 5 days:")
            for d, p in sorted_days[:5]:
                if p != 0:
                    log.info(f"    {d}: ${p:+.2f}")
            log.info("  Best 5 days:")
            for d, p in sorted_days[-5:]:
                if p != 0:
                    log.info(f"    {d}: ${p:+.2f}")

        # Per-station breakdown
        by_station = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
        for t in r.trades:
            by_station[t.station_id]["trades"] += 1
            by_station[t.station_id]["pnl"] += t.pnl
            if t.pnl > 0:
                by_station[t.station_id]["wins"] += 1

        log.info(f"\n  Per-station:")
        for sid in sorted(by_station, key=lambda s: by_station[s]["pnl"], reverse=True):
            s = by_station[sid]
            wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
            log.info(f"    {sid:12s} | {s['trades']:3d} trades | ${s['pnl']:+8.2f} | WR {wr:.0f}%")

    def _save_results(self, r: BacktestResult):
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

        # Save summary
        summary = {
            "start_date": r.start_date,
            "end_date": r.end_date,
            "n_stations": len(r.stations),
            "stations": r.stations,
            "initial_bankroll": r.initial_bankroll,
            "final_bankroll": r.final_bankroll,
            "peak_bankroll": r.peak_bankroll,
            "total_pnl": r.total_pnl,
            "roi_pct": r.roi_pct,
            "annualized_roi_pct": r.annualized_roi_pct,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "total_trades": r.total_trades,
            "win_rate": r.win_rate,
            "avg_pnl_per_trade": r.avg_pnl_per_trade,
            "profit_factor": r.profit_factor,
            "ladder_trades": r.ladder_trades,
            "ladder_pnl": r.ladder_pnl,
            "ladder_win_rate": r.ladder_win_rate,
            "no_trades": r.no_trades,
            "no_pnl": r.no_pnl,
            "no_win_rate": r.no_win_rate,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{cfg.RESULTS_DIR}/backtest_{ts}.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save trade log
        trade_log = [asdict(t) for t in r.trades]
        with open(f"{cfg.RESULTS_DIR}/trades_{ts}.json", "w") as f:
            json.dump(trade_log, f, indent=2, default=str)

        # Save equity curve
        with open(f"{cfg.RESULTS_DIR}/equity_{ts}.json", "w") as f:
            json.dump(self.equity_curve, f)

        log.info(f"  Results saved to {cfg.RESULTS_DIR}/backtest_{ts}.json")


# ═══════════════════════════════════════════════════════════════════
# Quick Backtest (limited stations, shorter period)
# ═══════════════════════════════════════════════════════════════════

def quick_backtest(days: int = 14, stations: List[str] = None) -> BacktestResult:
    """Run a quick backtest with limited scope for validation.
    Uses fewer stations and recent dates for fast iteration.
    """
    if stations is None:
        stations = ["NYC", "Chicago", "London", "Tokyo", "Miami"]

    end = datetime.now() - timedelta(days=3)  # 3 days ago (archive needs delay)
    start = end - timedelta(days=days)

    bt = Backtester(stations=stations, api_delay=0.3)
    return bt.run(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


def full_backtest(days: int = 60) -> BacktestResult:
    """Full backtest across all stations."""
    end = datetime.now() - timedelta(days=3)
    start = end - timedelta(days=days)

    bt = Backtester(stations=list(cfg.STATIONS.keys()), api_delay=0.4)
    return bt.run(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        full_backtest(days=days)
    else:
        days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
        quick_backtest(days=days)
