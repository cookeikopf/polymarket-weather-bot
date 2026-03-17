"""
Parameter Optimizer
====================
Systematically optimizes all bot parameters using grid search
over backtesting results.
"""

import numpy as np
import json
from itertools import product
from typing import Dict, List, Tuple
from dataclasses import dataclass

import config
from backtester import Backtester, BacktestResult


@dataclass
class OptimizationResult:
    """Result of one parameter combination."""
    params: Dict
    result: BacktestResult
    score: float  # Combined optimization objective


class ParameterOptimizer:
    """Grid search optimizer for bot parameters."""

    # Parameter grid
    PARAM_GRID = {
        "min_edge": [0.03, 0.05, 0.07, 0.10, 0.15],
        "kelly_fraction": [0.10, 0.15, 0.20, 0.25, 0.35],
        "max_position_pct": [0.05, 0.08, 0.10, 0.15],
        "min_ensemble_agreement": [0.40, 0.50, 0.60, 0.70],
    }

    # Pre-defined profiles for quick testing
    PROFILES = {
        "Conservative": {
            "min_edge": 0.10,
            "kelly_fraction": 0.15,
            "max_position_pct": 0.05,
            "min_ensemble_agreement": 0.70,
        },
        "Moderate": {
            "min_edge": 0.07,
            "kelly_fraction": 0.20,
            "max_position_pct": 0.08,
            "min_ensemble_agreement": 0.60,
        },
        "Balanced": {
            "min_edge": 0.05,
            "kelly_fraction": 0.25,
            "max_position_pct": 0.10,
            "min_ensemble_agreement": 0.50,
        },
        "Aggressive": {
            "min_edge": 0.03,
            "kelly_fraction": 0.35,
            "max_position_pct": 0.15,
            "min_ensemble_agreement": 0.40,
        },
        "Ultra-Selective": {
            "min_edge": 0.15,
            "kelly_fraction": 0.10,
            "max_position_pct": 0.05,
            "min_ensemble_agreement": 0.70,
        },
    }

    def __init__(self, station_id: str = "NYC"):
        self.station_id = station_id
        self.results: List[OptimizationResult] = []

    def run_profile_optimization(self, verbose: bool = True) -> Dict[str, OptimizationResult]:
        """Run backtests with pre-defined profiles."""
        print(f"\n{'='*60}")
        print(f"  PARAMETER OPTIMIZATION")
        print(f"  Station: {self.station_id}")
        print(f"  Testing {len(self.PROFILES)} profiles")
        print(f"{'='*60}\n")

        profile_results = {}

        for name, params in self.PROFILES.items():
            print(f"\n{'─'*40}")
            print(f"  Testing profile: {name}")
            print(f"  {json.dumps(params, indent=2)}")
            print(f"{'─'*40}")

            # Apply parameters
            self._apply_params(params)

            # Run backtest
            backtester = Backtester(self.station_id)
            result = backtester.run_backtest(verbose=False)

            # Score: maximize return/drawdown ratio (Calmar) with penalty for few trades
            trade_penalty = max(0, 1 - result.total_trades / 50)  # Penalize < 50 trades
            score = (
                0.35 * result.calmar_ratio +
                0.25 * result.sharpe_ratio +
                0.20 * result.profit_factor +
                0.10 * (result.win_rate / 100) +
                0.10 * (1 - trade_penalty)
            )

            opt_result = OptimizationResult(
                params=params, result=result, score=score
            )
            self.results.append(opt_result)
            profile_results[name] = opt_result

            if verbose:
                print(f"\n  Results for {name}:")
                print(f"    Return:        {result.total_return_pct:+.2f}%")
                print(f"    Max Drawdown:  {result.max_drawdown_pct:.2f}%")
                print(f"    Win Rate:      {result.win_rate:.1f}%")
                print(f"    Trades:        {result.total_trades}")
                print(f"    Profit Factor: {result.profit_factor:.2f}")
                print(f"    Sharpe:        {result.sharpe_ratio:.2f}")
                print(f"    Calmar:        {result.calmar_ratio:.2f}")
                print(f"    Score:         {score:.3f}")

        # Find best profile
        best_name = max(profile_results, key=lambda n: profile_results[n].score)
        best = profile_results[best_name]

        print(f"\n{'='*60}")
        print(f"  BEST PROFILE: {best_name}")
        print(f"  Score: {best.score:.3f}")
        print(f"  Return: {best.result.total_return_pct:+.2f}%")
        print(f"  Max DD: {best.result.max_drawdown_pct:.2f}%")
        print(f"  Win Rate: {best.result.win_rate:.1f}%")
        print(f"{'='*60}\n")

        # Apply best parameters
        self._apply_params(best.params)

        return profile_results

    def _apply_params(self, params: Dict):
        """Apply parameter set to config."""
        if "min_edge" in params:
            config.MIN_EDGE_PCT = params["min_edge"]
        if "kelly_fraction" in params:
            config.KELLY_FRACTION = params["kelly_fraction"]
        if "max_position_pct" in params:
            config.MAX_POSITION_PCT = params["max_position_pct"]
        if "min_ensemble_agreement" in params:
            config.MIN_ENSEMBLE_AGREEMENT = params["min_ensemble_agreement"]

    def get_ranking(self) -> List[Tuple[str, float, Dict]]:
        """Get ranked list of all tested configurations."""
        ranked = sorted(self.results, key=lambda r: r.score, reverse=True)
        return [(f"Config_{i}", r.score, r.params) for i, r in enumerate(ranked)]
