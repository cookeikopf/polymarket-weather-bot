"""
Backtesting Engine
===================
Simulates the full trading pipeline using historical weather data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import os
import re

import config
from weather_engine import WeatherEngine


@dataclass
class Trade:
    """A completed trade."""
    trade_id: int
    date: str
    outcome_name: str
    direction: str
    entry_price: float
    size_usd: float
    shares: float
    our_probability: float
    market_price: float
    edge: float
    confidence: float
    pnl: float
    won: bool


@dataclass
class BacktestResult:
    """Full backtest results."""
    total_return_pct: float
    total_pnl: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_edge: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    calmar_ratio: float
    kelly_growth_rate: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_pnl: List[float] = field(default_factory=list)


class Backtester:
    """Comprehensive backtesting engine for weather prediction markets."""

    def __init__(self, station_id: str = "NYC", initial_bankroll: float = None):
        self.station_id = station_id
        self.initial_bankroll = initial_bankroll or config.BACKTEST_INITIAL_BANKROLL
        self.engine = WeatherEngine(station_id)
        station_cfg = config.STATIONS.get(station_id, {})
        self.unit = "°F" if station_cfg.get("unit") == "fahrenheit" else "°C"
        self.is_fahrenheit = station_cfg.get("unit") == "fahrenheit"

        # State
        self.bankroll = self.initial_bankroll
        self.peak_bankroll = self.initial_bankroll
        self.trades: List[Trade] = []
        self.equity_curve = [self.initial_bankroll]
        self.daily_pnl = []
        self.trade_counter = 0
        self.drawdown_halt = False

    def _make_bucket_edges(self, actual_temp: float) -> List[float]:
        """Create temperature bucket edges centered around the actual temperature."""
        step = config.TEMP_BUCKET_SIZE_F
        center = round(actual_temp / step) * step
        start = center - 14
        end = center + 16
        return list(range(int(start), int(end) + 1, step))

    def _make_bucket_labels(self, edges: List[float]) -> List[str]:
        """Create bucket labels matching Polymarket format."""
        labels = []
        # Lower tail
        labels.append(f"{int(edges[0])}{self.unit} or below")
        # Regular buckets
        for i in range(len(edges) - 1):
            low = int(edges[i])
            high = int(edges[i + 1]) - 1
            labels.append(f"{low}-{high}{self.unit}")
        # Upper tail
        labels.append(f"{int(edges[-1])}{self.unit} or higher")
        return labels

    def _actual_to_winning_bucket(self, actual_temp: float, edges: List[float], labels: List[str]) -> str:
        """Determine which bucket the actual temperature falls into."""
        if actual_temp < edges[0]:
            return labels[0]  # Lower tail
        if actual_temp >= edges[-1]:
            return labels[-1]  # Upper tail
        for i in range(len(edges) - 1):
            if edges[i] <= actual_temp < edges[i + 1]:
                return labels[i + 1]  # +1 because labels[0] is tail
        return labels[-1]

    def _simulate_market_prices(self, actual_temp: float, edges: List[float], labels: List[str]) -> Dict[str, float]:
        """
        Simulate realistic market prices.
        Markets are informed but imperfect - centered roughly near the actual
        with noise and some systematic biases.
        """
        # Market's "best guess" is actual + some noise (market doesn't know actual perfectly)
        market_belief = actual_temp + np.random.normal(0, 3.0)
        
        prices = {}
        for label in labels:
            # Parse bucket bounds
            if "or below" in label:
                mid = edges[0] - 1
            elif "or higher" in label:
                mid = edges[-1] + 1
            else:
                nums = re.findall(r'(\d+)', label)
                if len(nums) >= 2:
                    mid = (float(nums[0]) + float(nums[1])) / 2
                else:
                    mid = float(nums[0])

            # True-ish probability based on market belief
            prob = np.exp(-0.5 * ((mid - market_belief) / 3.0) ** 2)
            # Add noise
            noise = np.random.normal(0, config.SIM_MARKET_NOISE)
            price = np.clip(prob + noise, 0.01, 0.99)
            prices[label] = price

        # Normalize so prices sum roughly to 1 (market constraint)
        total = sum(prices.values())
        prices = {k: v / total for k, v in prices.items()}
        
        return prices

    def run_backtest(
        self,
        start_date: str = None,
        end_date: str = None,
        verbose: bool = True,
    ) -> BacktestResult:
        """Run full backtest over historical period."""
        if start_date is None:
            start_date = config.BACKTEST_START_DATE
        if end_date is None:
            end_date = config.BACKTEST_END_DATE

        print(f"\n{'='*60}")
        print(f"  BACKTEST: {self.station_id} Weather Bot")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Initial Bankroll: ${self.initial_bankroll:.2f}")
        print(f"{'='*60}\n")

        # Step 1: Calibrate
        print("Step 1: Calibrating weather engine...")
        self.engine.calibrate()

        # Step 2: Fetch historical actuals
        print("\nStep 2: Loading historical data...")
        actuals = self.engine.fetch_historical_actuals(start_date, end_date)
        print(f"  Loaded {len(actuals)} days of actual data")

        if actuals.empty:
            print("  ERROR: No historical data")
            return self._compile_results()

        # Step 3: Simulate daily
        print("\nStep 3: Running daily trading simulation...")
        
        for _, row in actuals.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            actual_temp = row["actual"]

            if pd.isna(actual_temp):
                self.equity_curve.append(self.bankroll)
                self.daily_pnl.append(0)
                continue

            if self.drawdown_halt:
                self.equity_curve.append(self.bankroll)
                self.daily_pnl.append(0)
                continue

            day_pnl = self._simulate_day(date_str, actual_temp, verbose)
            self.daily_pnl.append(day_pnl)
            self.equity_curve.append(self.bankroll)

            # Check drawdown
            self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
            dd = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
            if dd >= config.MAX_DRAWDOWN_PCT:
                self.drawdown_halt = True
                if verbose:
                    print(f"  *** MAX DRAWDOWN HIT: {dd:.1%} ***")

        print("\nStep 4: Compiling results...")
        return self._compile_results()

    def _simulate_day(self, date_str: str, actual_temp: float, verbose: bool) -> float:
        """Simulate one trading day."""
        # 1. Create bucket structure
        edges = self._make_bucket_edges(actual_temp)
        labels = self._make_bucket_labels(edges)
        winning_bucket = self._actual_to_winning_bucket(actual_temp, edges, labels)

        # 2. Simulate market prices (what traders think)
        market_prices = self._simulate_market_prices(actual_temp, edges, labels)

        # 3. Simulate our forecasts (actual + calibrated model errors)
        forecasts = {}
        for model in config.WEATHER_MODELS:
            error_stats = self.engine.error_distributions.get(model, {"mean_bias": 0, "std": 3.5})
            error = np.random.normal(error_stats["mean_bias"], error_stats["std"])
            forecasts[model] = actual_temp + error

        # 4. Compute our probability distribution
        our_probs = self.engine.compute_probability_distribution(forecasts, edges)
        ensemble_stats = self.engine.compute_ensemble_stats(forecasts)

        # 5. Find edges (our_prob vs market_price)
        day_pnl = 0
        trades_today = 0
        current_exposure = sum(1 for _ in range(0))  # placeholder

        for label in labels:
            if trades_today >= 3:
                break

            our_prob = our_probs.get(label, 0)
            market_price = market_prices.get(label, 0)

            if market_price < 0.02 or market_price > 0.95:
                continue
            if our_prob < config.MIN_PROBABILITY:
                continue

            # Edge calculation
            edge = our_prob - market_price

            if abs(edge) < config.MIN_EDGE_PCT:
                continue

            # Confidence from ensemble
            confidence = ensemble_stats.get("agreement", 0.5)
            if confidence < config.MIN_ENSEMBLE_AGREEMENT:
                continue

            # Determine direction
            if edge > 0:
                direction = "BUY_YES"
                entry_price = market_price
                win_prob = our_prob
            else:
                direction = "BUY_NO"
                entry_price = 1.0 - market_price
                win_prob = 1.0 - our_prob
                edge = abs(edge)

            # Kelly sizing
            if entry_price <= 0 or entry_price >= 1:
                continue
            b = (1.0 - entry_price) / entry_price
            kelly = max(0, (b * win_prob - (1 - win_prob)) / b)
            kelly *= config.KELLY_FRACTION

            size_usd = min(
                kelly * self.bankroll,
                self.bankroll * config.MAX_POSITION_PCT,
                config.MAX_TRADE_SIZE_USDC,
            )
            if size_usd < config.MIN_TRADE_SIZE_USDC:
                continue

            # Simulate execution with slippage
            slippage = np.random.uniform(0, config.SIM_SLIPPAGE)
            exec_price = np.clip(entry_price + slippage, 0.01, 0.99)
            shares = size_usd / exec_price

            # Determine outcome
            is_correct_bucket = (label == winning_bucket)
            if direction == "BUY_YES":
                won = is_correct_bucket
            else:
                won = not is_correct_bucket

            # P&L calculation
            if won:
                pnl = shares * (1.0 - exec_price)  # Net win
            else:
                pnl = -size_usd  # Lose stake

            # Spread cost
            pnl -= size_usd * config.SIM_SPREAD * 0.5

            self.bankroll += pnl
            day_pnl += pnl
            self.trade_counter += 1
            trades_today += 1

            trade = Trade(
                trade_id=self.trade_counter,
                date=date_str,
                outcome_name=label,
                direction=direction,
                entry_price=exec_price,
                size_usd=size_usd,
                shares=shares,
                our_probability=our_prob,
                market_price=market_price,
                edge=edge,
                confidence=confidence,
                pnl=pnl,
                won=won,
            )
            self.trades.append(trade)

            if verbose and (trades_today <= 1 or abs(pnl) > 5):
                emoji = "+" if won else "-"
                print(
                    f"  [{date_str}] {emoji} {direction} '{label}' "
                    f"@ {exec_price:.3f} | Edge: {edge:.1%} | "
                    f"PnL: ${pnl:+.2f} | Bank: ${self.bankroll:.2f}"
                )

        return day_pnl

    def _compile_results(self) -> BacktestResult:
        """Compile all trading results."""
        if not self.trades:
            return BacktestResult(
                total_return_pct=0, total_pnl=0, max_drawdown_pct=0,
                win_rate=0, total_trades=0, winning_trades=0, losing_trades=0,
                avg_edge=0, avg_win=0, avg_loss=0, profit_factor=0,
                sharpe_ratio=0, calmar_ratio=0, kelly_growth_rate=0,
                trades=[], equity_curve=self.equity_curve, daily_pnl=self.daily_pnl,
            )

        total_pnl = self.bankroll - self.initial_bankroll
        total_return = total_pnl / self.initial_bankroll

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0
        total_wins = sum(t.pnl for t in wins) if wins else 0
        total_losses = sum(abs(t.pnl) for t in losses) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 999

        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.maximum(peak, 1)
        max_drawdown = float(np.max(drawdown))

        daily = np.array(self.daily_pnl) if self.daily_pnl else np.array([0])
        daily_nonzero = daily[daily != 0] if np.any(daily != 0) else daily
        sharpe = (np.mean(daily_nonzero) / np.std(daily_nonzero)) * np.sqrt(252) if np.std(daily_nonzero) > 0 else 0

        calmar = total_return / max_drawdown if max_drawdown > 0 else 999
        avg_edge = np.mean([t.edge for t in self.trades])
        kelly_growth = np.mean([np.log(max(0.001, 1 + t.pnl / max(self.initial_bankroll, 1))) for t in self.trades])

        return BacktestResult(
            total_return_pct=total_return * 100,
            total_pnl=total_pnl,
            max_drawdown_pct=max_drawdown * 100,
            win_rate=win_rate * 100,
            total_trades=len(self.trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            avg_edge=avg_edge * 100,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            kelly_growth_rate=kelly_growth,
            trades=self.trades,
            equity_curve=self.equity_curve,
            daily_pnl=self.daily_pnl,
        )

    def print_results(self, result: BacktestResult):
        """Pretty print backtest results."""
        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS: {self.station_id} Weather Bot")
        print(f"{'='*60}")
        print(f"  Total Return:      {result.total_return_pct:+.2f}%")
        print(f"  Total P&L:         ${result.total_pnl:+.2f}")
        print(f"  Max Drawdown:      {result.max_drawdown_pct:.2f}%")
        print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
        print(f"  Calmar Ratio:      {result.calmar_ratio:.2f}")
        print(f"  Profit Factor:     {result.profit_factor:.2f}")
        print(f"{'─'*60}")
        print(f"  Total Trades:      {result.total_trades}")
        print(f"  Win Rate:          {result.win_rate:.1f}%")
        print(f"  Avg Edge:          {result.avg_edge:.1f}%")
        print(f"  Avg Win:           ${result.avg_win:.2f}")
        print(f"  Avg Loss:          ${result.avg_loss:.2f}")
        print(f"{'─'*60}")
        print(f"  Final Bankroll:    ${self.bankroll:.2f}")
        print(f"  Kelly Growth Rate: {result.kelly_growth_rate:.4f}")
        print(f"{'='*60}\n")
