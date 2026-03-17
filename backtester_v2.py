"""
Backtester V2 — WU Resolution + ML Model
==========================================
Upgraded backtester that:
1. Uses Weather Underground data as the resolution source (like Polymarket)
2. Uses ML model predictions instead of just Monte Carlo
3. Simulates market prices based on Open-Meteo consensus (more realistic)
4. Compares ML vs non-ML approaches

KEY DIFFERENCE from V1: Resolution is determined by WU high temp,
not Open-Meteo actuals. Our edge comes from the ML model predicting
WU values better than the "market" (Open-Meteo consensus).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
import json
import os
import re

import config
from ml_model import WeatherMLModel


@dataclass
class TradeV2:
    """A completed trade."""
    trade_id: int
    date: str
    station_id: str
    outcome_name: str
    direction: str
    entry_price: float
    size_usd: float
    shares: float
    our_probability: float
    market_price: float
    edge: float
    wu_high_temp: float
    om_high_temp: float
    pnl: float
    won: bool


@dataclass
class BacktestResultV2:
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
    trades: List[TradeV2] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_pnl: List[float] = field(default_factory=list)
    station_stats: Dict = field(default_factory=dict)


class BacktesterV2:
    """WU-resolution backtester with ML model support."""

    def __init__(self, initial_bankroll: float = None, use_ml: bool = True):
        self.initial_bankroll = initial_bankroll or config.BACKTEST_INITIAL_BANKROLL
        self.use_ml = use_ml

        # ML model will be trained walk-forward during backtest
        self.ml_model = None

        # Load WU comparison biases
        bias_path = os.path.join("data", "wu_comparison.json")
        self.wu_biases = {}
        if os.path.exists(bias_path):
            with open(bias_path) as f:
                self.wu_biases = json.load(f)

        # State
        self.bankroll = self.initial_bankroll
        self.peak_bankroll = self.initial_bankroll
        self.trades: List[TradeV2] = []
        self.equity_curve = [self.initial_bankroll]
        self.daily_pnl = []
        self.trade_counter = 0
        self.drawdown_halt = False

    def run_backtest(
        self,
        stations: List[str] = None,
        start_date: str = "2024-09-01",
        end_date: str = "2026-03-01",
        verbose: bool = True,
    ) -> BacktestResultV2:
        """
        Run backtest across multiple stations using WU resolution data.
        """
        if stations is None:
            stations = list(config.WU_STATIONS.keys())

        print(f"\n{'='*65}")
        print(f"  BACKTEST V2: WU Resolution + {'ML Model' if self.use_ml else 'Baseline'}")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Stations: {len(stations)}")
        print(f"  Initial Bankroll: ${self.initial_bankroll:.2f}")
        print(f"{'='*65}\n")

        # Load all data
        print("Loading data...")
        station_data = {}
        for sid in stations:
            wu_path = os.path.join(config.WU_DATA_DIR, f"{sid.replace(' ', '_')}.csv")
            if not os.path.exists(wu_path):
                continue
            wu_df = pd.read_csv(wu_path)
            wu_df["date"] = pd.to_datetime(wu_df["date"])

            # Load feature data
            feat_path = config.ML_FEATURES_PATH
            if os.path.exists(feat_path):
                feat_df = pd.read_csv(feat_path)
                feat_df["date"] = pd.to_datetime(feat_df["date"])
                station_code = list(config.WU_STATIONS.keys()).index(sid)
                feat_df = feat_df[feat_df["station_code"] == station_code]
                station_data[sid] = {
                    "wu": wu_df,
                    "features": feat_df,
                }
            else:
                station_data[sid] = {"wu": wu_df, "features": pd.DataFrame()}

        print(f"  Loaded {len(station_data)} stations")

        # Walk-forward ML training: train on data BEFORE the backtest period
        if self.use_ml:
            print("  Training ML model (walk-forward: training on pre-backtest data only)...")
            feat_path = config.ML_FEATURES_PATH
            if os.path.exists(feat_path):
                all_feat = pd.read_csv(feat_path)
                all_feat["date"] = pd.to_datetime(all_feat["date"])
                train_data = all_feat[all_feat["date"] < start_date]
                if len(train_data) >= 200:
                    self.ml_model = WeatherMLModel()
                    self.ml_model.train(train_data, verbose=True)
                    print(f"  ML model trained on {len(train_data)} pre-backtest samples")
                else:
                    print(f"  Not enough pre-backtest data ({len(train_data)}), using bias-only")
                    self.ml_model = None

        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Simulate day by day
        station_stats = {sid: {"trades": 0, "wins": 0, "pnl": 0} for sid in stations}

        for date in dates:
            if self.drawdown_halt:
                self.equity_curve.append(self.bankroll)
                self.daily_pnl.append(0)
                continue

            day_pnl = 0
            trades_today = 0

            for sid in stations:
                if trades_today >= config.MAX_CONCURRENT_POSITIONS:
                    break

                data = station_data.get(sid)
                if data is None:
                    continue

                wu_df = data["wu"]
                feat_df = data["features"]

                # Get WU actual for this date (resolution)
                date_str = date.strftime("%Y-%m-%d")
                wu_row = wu_df[wu_df["date"] == date]
                if wu_row.empty:
                    continue

                wu_high = wu_row["high_temp"].iloc[0]
                is_f = config.WU_STATIONS[sid]["units"] == "e"

                # Get features for this date
                feat_row = feat_df[feat_df["date"] == date] if not feat_df.empty else pd.DataFrame()

                if feat_row.empty:
                    continue

                om_high = feat_row["om_high_temp"].iloc[0]
                if pd.isna(om_high) or pd.isna(wu_high):
                    continue

                # Build bucket structure
                edges = self._make_bucket_edges(wu_high, is_f)
                labels = self._make_bucket_labels(edges, is_f)
                winning_bucket = self._find_winning_bucket(wu_high, edges, labels)

                # Simulate market prices (based on Open-Meteo consensus = "market belief")
                market_prices = self._simulate_market_prices(om_high, edges, labels)

                # Our predictions
                if self.use_ml and self.ml_model and self.ml_model.is_trained:
                    # Use ML model
                    features = feat_row.iloc[0].to_dict()
                    features["station_id"] = sid
                    our_probs = self.ml_model.predict_bucket_probs(
                        features, edges, is_fahrenheit=is_f
                    )
                else:
                    # Baseline: use Open-Meteo with simple bias correction
                    bias = self.wu_biases.get(sid, {}).get("mean_bias", 0)
                    our_probs = self._baseline_probs(om_high + bias, edges, labels, is_f)

                # Find and execute trades
                for label in labels:
                    if trades_today >= config.MAX_CONCURRENT_POSITIONS:
                        break

                    our_prob = our_probs.get(label, 0)
                    market_price = market_prices.get(label, 0)

                    if market_price < 0.02 or market_price > 0.95:
                        continue
                    if our_prob < config.MIN_PROBABILITY:
                        continue

                    edge = our_prob - market_price

                    if abs(edge) < config.MIN_EDGE_PCT:
                        continue

                    # Direction
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

                    # Realistic position sizing - cap at MAX_TRADE_SIZE
                    # regardless of bankroll growth (liquidity constraint)
                    size_usd = min(
                        kelly * self.bankroll,
                        self.bankroll * config.MAX_POSITION_PCT,
                        config.MAX_TRADE_SIZE_USDC,  # Hard cap (market liquidity)
                    )
                    if size_usd < config.MIN_TRADE_SIZE_USDC:
                        continue

                    # Execution with slippage
                    slippage = np.random.uniform(0, config.SIM_SLIPPAGE)
                    exec_price = np.clip(entry_price + slippage, 0.01, 0.99)
                    shares = size_usd / exec_price

                    # Resolution
                    is_winning_bucket = (label == winning_bucket)
                    if direction == "BUY_YES":
                        won = is_winning_bucket
                    else:
                        won = not is_winning_bucket

                    # P&L
                    if won:
                        pnl = shares * (1.0 - exec_price)
                    else:
                        pnl = -size_usd

                    # Spread cost
                    pnl -= size_usd * config.SIM_SPREAD * 0.5

                    self.bankroll += pnl
                    day_pnl += pnl
                    self.trade_counter += 1
                    trades_today += 1

                    trade = TradeV2(
                        trade_id=self.trade_counter,
                        date=date_str,
                        station_id=sid,
                        outcome_name=label,
                        direction=direction,
                        entry_price=exec_price,
                        size_usd=size_usd,
                        shares=shares,
                        our_probability=our_prob,
                        market_price=market_price,
                        edge=edge,
                        wu_high_temp=wu_high,
                        om_high_temp=om_high,
                        pnl=pnl,
                        won=won,
                    )
                    self.trades.append(trade)

                    station_stats[sid]["trades"] += 1
                    station_stats[sid]["pnl"] += pnl
                    if won:
                        station_stats[sid]["wins"] += 1

            self.daily_pnl.append(day_pnl)
            self.equity_curve.append(self.bankroll)

            # Check drawdown
            self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
            dd = (self.peak_bankroll - self.bankroll) / self.peak_bankroll if self.peak_bankroll > 0 else 0
            if dd >= config.MAX_DRAWDOWN_PCT:
                self.drawdown_halt = True
                if verbose:
                    print(f"  *** DRAWDOWN HALT: {dd:.1%} on {date_str} ***")

        return self._compile_results(station_stats, verbose)

    def _make_bucket_edges(self, ref_temp: float, is_fahrenheit: bool) -> List[float]:
        """Create bucket edges centered on reference temperature."""
        step = 2 if is_fahrenheit else 1
        center = round(ref_temp / step) * step
        n_buckets = 7 if is_fahrenheit else 8
        start = center - n_buckets * step
        end = center + (n_buckets + 1) * step
        return list(range(int(start), int(end) + 1, step))

    def _make_bucket_labels(self, edges: List[float], is_fahrenheit: bool) -> List[str]:
        """Create bucket labels."""
        labels = []
        unit = "°F" if is_fahrenheit else "°C"
        labels.append(f"{int(edges[0]) - 1}{unit} or below")
        for i in range(len(edges) - 1):
            if is_fahrenheit:
                labels.append(f"{int(edges[i])}-{int(edges[i+1]) - 1}{unit}")
            else:
                labels.append(f"{int(edges[i])}{unit}")
        labels.append(f"{int(edges[-1])}{unit} or higher")
        return labels

    def _find_winning_bucket(self, temp: float, edges: List[float], labels: List[str]) -> str:
        """Find which bucket the WU temperature falls into."""
        if temp < edges[0]:
            return labels[0]
        if temp >= edges[-1]:
            return labels[-1]
        for i in range(len(edges) - 1):
            if edges[i] <= temp < edges[i + 1]:
                return labels[i + 1]
        return labels[-1]

    def _simulate_market_prices(
        self, om_belief: float, edges: List[float], labels: List[str]
    ) -> Dict[str, float]:
        """
        Simulate REALISTIC market prices based on Open-Meteo consensus.
        Real Polymarket weather markets are fairly efficient — most bettors
        use similar weather data. Our edge is the WU bias correction.
        Market sigma should be close to the actual forecast error (~3°F / ~1.5°C).
        """
        # Market's belief: Open-Meteo + small noise (markets are fairly smart)
        market_temp = om_belief + np.random.normal(0, 0.8)

        prices = {}
        for label in labels:
            # Parse bucket midpoint
            if "or below" in label:
                mid = edges[0] - 1
            elif "or higher" in label:
                mid = edges[-1] + 1
            else:
                nums = re.findall(r'-?\d+', label)
                if len(nums) >= 2:
                    mid = (float(nums[0]) + float(nums[1])) / 2
                else:
                    mid = float(nums[0]) + 0.5

            # Market uses its own probability model (tighter than naive Gaussian)
            is_f = "°F" in label
            sigma = 3.0 if is_f else 1.8  # Realistic forecast uncertainty
            prob = np.exp(-0.5 * ((mid - market_temp) / sigma) ** 2)
            # Small noise (market isn't perfectly efficient)
            noise = np.random.normal(0, 0.005)
            prices[label] = np.clip(prob + noise, 0.005, 0.99)

        # Normalize so prices sum to ~1 (market constraint)
        total = sum(prices.values())
        prices = {k: v / total for k, v in prices.items()}

        # Add realistic spread: market prices deviate from true prob
        # by up to 2-3 cents (bid-ask spread in weather markets)
        spread_prices = {}
        for label, price in prices.items():
            spread_noise = np.random.uniform(-0.015, 0.015)
            spread_prices[label] = np.clip(price + spread_noise, 0.005, 0.995)

        # Re-normalize
        total = sum(spread_prices.values())
        spread_prices = {k: v / total for k, v in spread_prices.items()}
        return spread_prices

    def _baseline_probs(
        self, predicted_temp: float, edges: List[float],
        labels: List[str], is_fahrenheit: bool
    ) -> Dict[str, float]:
        """Baseline probability calculation (without ML)."""
        sigma = 3.5 if is_fahrenheit else 2.0
        samples = predicted_temp + stats.t.rvs(df=5, loc=0, scale=sigma, size=3000)

        probs = {}
        for label in labels:
            if "or below" in label:
                mask = samples < edges[0]
            elif "or higher" in label:
                mask = samples >= edges[-1]
            else:
                nums = re.findall(r'-?\d+', label)
                if len(nums) >= 2:
                    low, high = float(nums[0]), float(nums[1])
                    if is_fahrenheit:
                        mask = (samples >= low) & (samples < high + 1)
                    else:
                        mask = (samples >= low) & (samples < low + 1)
                else:
                    low = float(nums[0])
                    mask = (samples >= low) & (samples < low + 1)
            probs[label] = max(mask.mean(), 0.001)

        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        return probs

    def _compile_results(self, station_stats: Dict, verbose: bool) -> BacktestResultV2:
        """Compile backtest results."""
        if not self.trades:
            return BacktestResultV2(
                total_return_pct=0, total_pnl=0, max_drawdown_pct=0,
                win_rate=0, total_trades=0, winning_trades=0, losing_trades=0,
                avg_edge=0, avg_win=0, avg_loss=0, profit_factor=0,
                sharpe_ratio=0, calmar_ratio=0,
            )

        total_pnl = self.bankroll - self.initial_bankroll
        total_return = total_pnl / self.initial_bankroll

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0
        total_wins = sum(t.pnl for t in wins)
        total_losses = sum(abs(t.pnl) for t in losses)
        profit_factor = total_wins / total_losses if total_losses > 0 else 999

        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.maximum(peak, 1)
        max_drawdown = float(np.max(drawdown))

        daily = np.array(self.daily_pnl) if self.daily_pnl else np.array([0])
        daily_nz = daily[daily != 0] if np.any(daily != 0) else daily
        sharpe = (np.mean(daily_nz) / np.std(daily_nz)) * np.sqrt(252) if np.std(daily_nz) > 0 else 0

        calmar = total_return / max_drawdown if max_drawdown > 0 else 999
        avg_edge = np.mean([t.edge for t in self.trades])

        result = BacktestResultV2(
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
            trades=self.trades,
            equity_curve=self.equity_curve,
            daily_pnl=self.daily_pnl,
            station_stats=station_stats,
        )

        if verbose:
            self._print_results(result)

        return result

    def _print_results(self, r: BacktestResultV2):
        """Pretty print results."""
        mode = "ML Model" if self.use_ml else "Baseline (Bias-Corrected)"
        print(f"\n{'='*65}")
        print(f"  BACKTEST V2 RESULTS: {mode}")
        print(f"{'='*65}")
        print(f"  Total Return:    {r.total_return_pct:+.2f}%")
        print(f"  Total P&L:       ${r.total_pnl:+.2f}")
        print(f"  Max Drawdown:    {r.max_drawdown_pct:.2f}%")
        print(f"  Sharpe Ratio:    {r.sharpe_ratio:.2f}")
        print(f"  Calmar Ratio:    {r.calmar_ratio:.2f}")
        print(f"  Profit Factor:   {r.profit_factor:.2f}")
        print(f"{'─'*65}")
        print(f"  Total Trades:    {r.total_trades}")
        print(f"  Win Rate:        {r.win_rate:.1f}%")
        print(f"  Avg Edge:        {r.avg_edge:.1f}%")
        print(f"  Avg Win:         ${r.avg_win:.2f}")
        print(f"  Avg Loss:        ${r.avg_loss:.2f}")
        print(f"{'─'*65}")
        print(f"  Final Bankroll:  ${self.bankroll:.2f}")

        # Per-station breakdown
        if r.station_stats:
            print(f"\n  {'Station':15s} {'Trades':>7s} {'Wins':>6s} {'WR':>7s} {'P&L':>10s}")
            print(f"  {'─'*48}")
            for sid, st in sorted(r.station_stats.items(), key=lambda x: -x[1]["pnl"]):
                if st["trades"] == 0:
                    continue
                wr = st["wins"] / st["trades"] * 100 if st["trades"] > 0 else 0
                print(f"  {sid:15s} {st['trades']:7d} {st['wins']:6d} {wr:6.1f}% ${st['pnl']:+9.2f}")

        print(f"{'='*65}\n")


def main():
    """Run comparison: ML vs Baseline backtest."""
    os.makedirs("results", exist_ok=True)

    # Use start date that allows 3+ months training data
    # Data starts 2024-06-01, so start backtesting from 2025-01-01
    bt_start = "2025-01-01"
    bt_end = "2026-03-01"

    # Run ML backtest
    print("\n" + "="*65)
    print("  RUNNING ML MODEL BACKTEST")
    print("="*65)
    bt_ml = BacktesterV2(initial_bankroll=50.0, use_ml=True)
    result_ml = bt_ml.run_backtest(start_date=bt_start, end_date=bt_end)

    # Run baseline backtest
    print("\n" + "="*65)
    print("  RUNNING BASELINE BACKTEST (Bias-Corrected Only)")
    print("="*65)
    bt_base = BacktesterV2(initial_bankroll=50.0, use_ml=False)
    result_base = bt_base.run_backtest(start_date=bt_start, end_date=bt_end)

    # Comparison
    print("\n" + "="*65)
    print("  COMPARISON: ML vs Baseline")
    print("="*65)
    print(f"\n  {'Metric':25s} {'ML Model':>12s} {'Baseline':>12s} {'Delta':>12s}")
    print(f"  {'─'*64}")

    metrics = [
        ("Total Return", f"{result_ml.total_return_pct:+.1f}%", f"{result_base.total_return_pct:+.1f}%",
         f"{result_ml.total_return_pct - result_base.total_return_pct:+.1f}%"),
        ("Total P&L", f"${result_ml.total_pnl:+.2f}", f"${result_base.total_pnl:+.2f}",
         f"${result_ml.total_pnl - result_base.total_pnl:+.2f}"),
        ("Win Rate", f"{result_ml.win_rate:.1f}%", f"{result_base.win_rate:.1f}%",
         f"{result_ml.win_rate - result_base.win_rate:+.1f}%"),
        ("Sharpe Ratio", f"{result_ml.sharpe_ratio:.2f}", f"{result_base.sharpe_ratio:.2f}",
         f"{result_ml.sharpe_ratio - result_base.sharpe_ratio:+.2f}"),
        ("Profit Factor", f"{result_ml.profit_factor:.2f}", f"{result_base.profit_factor:.2f}",
         f"{result_ml.profit_factor - result_base.profit_factor:+.2f}"),
        ("Max Drawdown", f"{result_ml.max_drawdown_pct:.1f}%", f"{result_base.max_drawdown_pct:.1f}%",
         f"{result_ml.max_drawdown_pct - result_base.max_drawdown_pct:+.1f}%"),
        ("Total Trades", f"{result_ml.total_trades}", f"{result_base.total_trades}",
         f"{result_ml.total_trades - result_base.total_trades:+d}"),
    ]

    for name, ml_val, base_val, delta in metrics:
        print(f"  {name:25s} {ml_val:>12s} {base_val:>12s} {delta:>12s}")

    # Save results
    def serialize_result(r):
        return {
            "total_return_pct": r.total_return_pct,
            "total_pnl": r.total_pnl,
            "max_drawdown_pct": r.max_drawdown_pct,
            "win_rate": r.win_rate,
            "total_trades": r.total_trades,
            "winning_trades": r.winning_trades,
            "losing_trades": r.losing_trades,
            "avg_edge": r.avg_edge,
            "avg_win": r.avg_win,
            "avg_loss": r.avg_loss,
            "profit_factor": r.profit_factor,
            "sharpe_ratio": r.sharpe_ratio,
            "calmar_ratio": r.calmar_ratio,
            "equity_curve": r.equity_curve,
            "station_stats": r.station_stats,
        }

    output = {
        "ml_model": serialize_result(result_ml),
        "baseline": serialize_result(result_base),
        "comparison": {
            "return_delta": result_ml.total_return_pct - result_base.total_return_pct,
            "pnl_delta": result_ml.total_pnl - result_base.total_pnl,
            "wr_delta": result_ml.win_rate - result_base.win_rate,
            "sharpe_delta": result_ml.sharpe_ratio - result_base.sharpe_ratio,
        }
    }

    out_path = os.path.join("results", "wu_backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
