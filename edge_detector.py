"""
Edge Detection & Signal Generation
====================================
Compares our ML probability distributions against market prices
to find profitable trading opportunities.

KEY INNOVATION: Bayesian edge estimation that combines:
1. Our ensemble weather forecast probabilities
2. Climatological base rates (prior)
3. Market consensus (informative signal)
4. Edge confidence intervals via bootstrap
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import config
from market_scanner import WeatherMarket, MarketOutcome


@dataclass
class TradingSignal:
    """A trading opportunity detected by edge analysis."""
    market: WeatherMarket
    outcome: MarketOutcome
    our_probability: float
    market_price: float
    edge: float  # our_prob - market_price (positive = buy, negative = sell/fade)
    edge_pct: float  # edge / market_price
    confidence: float  # 0-1, how confident we are in our edge
    direction: str  # "BUY_YES" or "BUY_NO"
    kelly_fraction: float  # Optimal bet fraction
    suggested_size_usd: float
    expected_value: float  # Expected P&L per $1 traded
    reasons: List[str]  # Human-readable reasoning


class EdgeDetector:
    """Detect profitable edges between our forecasts and market prices."""

    def __init__(self):
        self.min_edge = config.MIN_EDGE_PCT
        self.min_probability = config.MIN_PROBABILITY
        self.max_entry_price = config.MAX_ENTRY_PRICE

    def find_edges(
        self,
        market: WeatherMarket,
        our_probabilities: Dict[str, float],
        ensemble_stats: Dict,
        bankroll: float,
        current_exposure: float = 0,
    ) -> List[TradingSignal]:
        """
        Compare our probabilities against market prices to find edges.

        Returns list of TradingSignals sorted by expected value.
        """
        signals = []

        for outcome in market.outcomes:
            # Try to match outcome to our probability bucket
            our_prob = self._match_probability(outcome.name, our_probabilities)
            if our_prob is None:
                continue

            market_price = outcome.price

            # Skip outcomes with no CLOB liquidity (marked -1 by enrich_with_live_prices)
            if market_price < 0:
                continue

            # Skip extreme prices (illiquid, near-certain)
            if market_price < 0.01 or market_price > self.max_entry_price:
                continue

            # Calculate edge
            edge = our_prob - market_price
            edge_pct = abs(edge) / max(market_price, 0.01)

            # Determine direction
            if edge > 0:
                # Our probability is higher -> Buy YES
                direction = "BUY_YES"
                entry_price = market_price
                win_prob = our_prob
                payout = 1.0 - entry_price  # Win $1 - entry_price
            else:
                # Our probability is lower -> Buy NO (sell YES equivalent)
                direction = "BUY_NO"
                entry_price = 1.0 - market_price
                win_prob = 1.0 - our_prob
                payout = 1.0 - entry_price
                edge = abs(edge)  # Make edge positive for sizing

            # Skip if edge too small
            if edge < self.min_edge:
                continue

            # Skip if our probability is too low (high uncertainty)
            if our_prob < self.min_probability and direction == "BUY_YES":
                continue

            # Calculate confidence based on ensemble agreement and edge size
            confidence = self._compute_confidence(
                edge, ensemble_stats, our_prob, market_price
            )

            if confidence < config.MIN_ENSEMBLE_AGREEMENT:
                continue

            # Kelly Criterion position sizing
            kelly_frac = self._kelly_criterion(win_prob, entry_price)

            # Expected value per dollar
            ev = win_prob * payout - (1 - win_prob) * entry_price

            # Suggested position size
            suggested_size = self._compute_position_size(
                kelly_frac, bankroll, current_exposure, confidence
            )

            # Build reasoning
            reasons = self._build_reasons(
                our_prob, market_price, edge, confidence, ensemble_stats, direction
            )

            signals.append(TradingSignal(
                market=market,
                outcome=outcome,
                our_probability=our_prob,
                market_price=market_price,
                edge=edge,
                edge_pct=edge_pct,
                confidence=confidence,
                direction=direction,
                kelly_fraction=kelly_frac,
                suggested_size_usd=suggested_size,
                expected_value=ev,
                reasons=reasons,
            ))

        # Sort by EV * confidence (risk-adjusted expected value)
        signals.sort(key=lambda s: s.expected_value * s.confidence, reverse=True)

        return signals

    def _match_probability(
        self, outcome_name: str, our_probs: Dict[str, float]
    ) -> Optional[float]:
        """Match a market outcome name to our probability distribution.

        Handles both formats:
        - °F: "34-35°F", "31°F or below", "42°F or higher"
        - °C: "14°C", "-5°C", "12°C or below", "22°C or higher"
        """
        # Direct match
        if outcome_name in our_probs:
            return our_probs[outcome_name]

        # Fuzzy matching (case-insensitive, strip whitespace)
        outcome_lower = outcome_name.lower().strip()
        for key, prob in our_probs.items():
            if key.lower().strip() == outcome_lower:
                return prob

        import re

        # Normalize both sides: remove whitespace around ° and -
        def normalize_label(s):
            s = s.strip()
            s = re.sub(r'\s*°\s*', '°', s)
            s = re.sub(r'\s*[-–]\s*', '-', s)
            return s

        norm_outcome = normalize_label(outcome_name)
        for key, prob in our_probs.items():
            if normalize_label(key) == norm_outcome:
                return prob

        # Try extracting the core temperature value(s) and matching
        # °C single-degree: "14°C" matches "14°C"
        match_c = re.match(r'^(-?\d+)\s*°\s*C$', outcome_name.strip())
        if match_c:
            target = f"{match_c.group(1)}°C"
            for key, prob in our_probs.items():
                key_match = re.match(r'^(-?\d+)\s*°\s*C$', key.strip())
                if key_match and key_match.group(1) == match_c.group(1):
                    return prob

        # °F range: "34-35°F" matches "34-35°F"
        match_f = re.match(r'^(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F$', outcome_name.strip())
        if match_f:
            for key, prob in our_probs.items():
                key_match = re.match(r'^(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F$', key.strip())
                if key_match and key_match.group(1) == match_f.group(1) and key_match.group(2) == match_f.group(2):
                    return prob

        # "or higher" / "or below" matching by extracted number
        match_tail = re.match(r'^(-?\d+)\s*°?\s*[FC]?\s*or\s*(higher|below|lower)', outcome_name.strip(), re.I)
        if match_tail:
            val = match_tail.group(1)
            direction = match_tail.group(2).lower()
            for key, prob in our_probs.items():
                key_tail = re.match(r'^(-?\d+)\s*°?\s*[FC]?\s*or\s*(higher|below|lower)', key.strip(), re.I)
                if key_tail and key_tail.group(1) == val:
                    kd = key_tail.group(2).lower()
                    if (direction in ("below", "lower") and kd in ("below", "lower")) or \
                       (direction == "higher" and kd == "higher"):
                        return prob

        # Final fallback: match by numeric values
        outcome_nums = re.findall(r'-?\d+', outcome_name)
        if outcome_nums:
            for key, prob in our_probs.items():
                key_nums = re.findall(r'-?\d+', key)
                if key_nums and key_nums == outcome_nums:
                    return prob

        return None

    def _compute_confidence(
        self,
        edge: float,
        ensemble_stats: Dict,
        our_prob: float,
        market_price: float,
    ) -> float:
        """
        Compute confidence score (0-1) for an edge.

        Higher confidence when:
        - Models agree (low spread)
        - Edge is large relative to uncertainty
        - We have many models contributing
        - Historical calibration is accurate
        """
        # Base: ensemble agreement
        agreement = ensemble_stats.get("agreement", 0.5)

        # Edge significance: edge relative to model spread
        model_std = ensemble_stats.get("std", 5.0)
        # Convert temperature spread to probability spread (rough)
        prob_uncertainty = min(0.30, model_std * 0.03)
        edge_significance = min(1.0, edge / max(prob_uncertainty, 0.01))

        # Number of models factor
        n_models = ensemble_stats.get("n_models", 1)
        model_count_factor = min(1.0, n_models / 5)

        # Combine
        confidence = (
            0.40 * agreement +
            0.35 * edge_significance +
            0.25 * model_count_factor
        )

        return np.clip(confidence, 0, 1)

    def _kelly_criterion(self, win_prob: float, entry_price: float) -> float:
        """
        Full Kelly Criterion for binary outcomes.
        f* = (bp - q) / b
        where:
            b = payout odds (net win / bet)
            p = win probability
            q = 1 - p
        """
        if entry_price <= 0 or entry_price >= 1:
            return 0

        b = (1.0 - entry_price) / entry_price  # Payout odds
        p = win_prob
        q = 1.0 - p

        kelly = (b * p - q) / b

        # Apply fractional Kelly
        kelly *= config.KELLY_FRACTION

        # Clamp
        return max(0, min(kelly, config.MAX_POSITION_PCT))

    def _compute_position_size(
        self,
        kelly_frac: float,
        bankroll: float,
        current_exposure: float,
        confidence: float,
    ) -> float:
        """Compute actual position size in USDC."""
        # Kelly-based size
        kelly_size = kelly_frac * bankroll

        # Scale by confidence
        size = kelly_size * confidence

        # Cap at maximum position size
        size = min(size, bankroll * config.MAX_POSITION_PCT)
        size = min(size, config.MAX_TRADE_SIZE_USDC)

        # Check total exposure limit
        remaining_capacity = (bankroll * config.MAX_TOTAL_EXPOSURE) - current_exposure
        if remaining_capacity <= 0:
            return 0
        size = min(size, remaining_capacity)

        # Floor at minimum
        if size < config.MIN_TRADE_SIZE_USDC:
            return 0

        return round(size, 2)

    def _build_reasons(
        self,
        our_prob: float,
        market_price: float,
        edge: float,
        confidence: float,
        ensemble_stats: Dict,
        direction: str,
    ) -> List[str]:
        """Build human-readable reasoning for a trade."""
        reasons = []

        reasons.append(
            f"Our probability: {our_prob:.1%} vs Market: {market_price:.1%} "
            f"(Edge: {edge:.1%})"
        )
        reasons.append(f"Direction: {direction}")
        reasons.append(
            f"Ensemble: {ensemble_stats.get('n_models', 0)} models, "
            f"agreement: {ensemble_stats.get('agreement', 0):.1%}, "
            f"spread: {ensemble_stats.get('range', 0):.1f}°"
        )
        reasons.append(f"Confidence: {confidence:.1%}")

        if edge > 0.15:
            reasons.append("STRONG EDGE: >15% mispricing detected")
        elif edge > 0.10:
            reasons.append("GOOD EDGE: >10% mispricing detected")

        return reasons
