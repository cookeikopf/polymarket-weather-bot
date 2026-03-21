"""
Edge Detection & Signal Generation (V3 — Live Trading)
========================================================
Compares our ML probability distributions against market prices
to find profitable trading opportunities.

V3 CHANGES (based on retro-analysis of 20 paper positions):
- Direction filter: BUY_YES disabled (0% win rate), BUY_NO only (58%)
- Entry price filter: min 0.55, max 0.85
- Edge threshold: 10% minimum (was 3%)
- Ensemble agreement: 50% minimum (was 40%)
- Forecast bias correction: conservative shift toward market consensus
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
        self.max_entry_price = getattr(config, 'MAX_ENTRY_PRICE', 0.85)
        self.min_entry_price = getattr(config, 'MIN_ENTRY_PRICE', 0.55)
        self.allow_buy_yes = getattr(config, 'ALLOW_BUY_YES', False)
        self.allow_buy_no = getattr(config, 'ALLOW_BUY_NO', True)

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

            market_price = outcome.price  # Mid-price (enriched from CLOB)

            # Skip outcomes with no CLOB liquidity (marked -1 by enrich_with_live_prices)
            if market_price < 0:
                continue

            # Skip extreme prices (illiquid, near-certain)
            if market_price < 0.01 or market_price > 0.99:
                continue

            # Use spread-aware prices if CLOB data available
            # _clob_ask = effective ask (what we pay for BUY_YES)
            # _clob_bid = effective bid (what we receive for SELL_YES → relates to BUY_NO cost)
            clob_ask = getattr(outcome, '_clob_ask', None)
            clob_bid = getattr(outcome, '_clob_bid', None)

            # ─── V3: FORECAST BIAS CORRECTION ───
            # Retro-analysis showed our forecasts systematically overestimate
            # extreme/tail probabilities. Apply conservative shrinkage toward
            # the market consensus (Bayesian shrinkage).
            # Blend: 70% our forecast + 30% market price
            bias_corrected_prob = 0.70 * our_prob + 0.30 * market_price

            # Calculate edge accounting for spread
            if clob_ask and clob_bid:
                # Spread-aware edge: use worst-case execution price
                buy_yes_cost = clob_ask   # What we actually pay for YES
                buy_no_cost = 1.0 - clob_bid  # What we actually pay for NO
                edge_yes = bias_corrected_prob - buy_yes_cost
                edge_no = (1.0 - bias_corrected_prob) - buy_no_cost
            else:
                # Fallback: mid-price (backtesting)
                edge_yes = bias_corrected_prob - market_price
                edge_no = market_price - bias_corrected_prob

            edge_pct = abs(bias_corrected_prob - market_price) / max(market_price, 0.01)

            # Determine direction based on which side has positive edge
            if edge_yes > edge_no and edge_yes > 0:
                # BUY YES
                direction = "BUY_YES"
                entry_price = clob_ask if clob_ask else market_price
                win_prob = bias_corrected_prob
                payout = 1.0 - entry_price
                edge = edge_yes
            elif edge_no > 0:
                # BUY NO
                direction = "BUY_NO"
                entry_price = (1.0 - clob_bid) if clob_bid else (1.0 - market_price)
                win_prob = 1.0 - bias_corrected_prob
                payout = 1.0 - entry_price
                edge = edge_no
            else:
                # No positive edge on either side
                continue

            # ─── V3: DIRECTION FILTER ───
            # BUY_YES had 0% win rate (0/8 positions lost $79.84)
            # BUY_NO had 58% win rate (7/12 positions, slight loss due to sizing)
            if direction == "BUY_YES" and not self.allow_buy_yes:
                continue
            if direction == "BUY_NO" and not self.allow_buy_no:
                continue

            # ─── V3: ENTRY PRICE FILTER ───
            # Positions with entry price < 0.50 had 0% win rate
            # Winning BUY_NO trades had avg entry = 0.664
            if entry_price < self.min_entry_price:
                continue
            if entry_price > self.max_entry_price:
                continue

            # Skip if edge too small
            if edge < self.min_edge:
                continue

            # Skip if our probability is too low (high uncertainty)
            if our_prob < self.min_probability and direction == "BUY_YES":
                continue

            # Calculate confidence based on ensemble agreement and edge size
            confidence = self._compute_confidence(
                edge, ensemble_stats, bias_corrected_prob, market_price
            )

            if confidence < config.MIN_ENSEMBLE_AGREEMENT:
                continue

            # Kelly Criterion position sizing
            kelly_frac = self._kelly_criterion(win_prob, entry_price)

            # Expected value per dollar
            ev = win_prob * payout - (1 - win_prob) * entry_price

            # ─── V3: Require positive EV after fees ───
            # Polymarket has ~2% round-trip cost (spread + slippage)
            FEE_ESTIMATE = 0.02
            if ev < FEE_ESTIMATE:
                continue

            # Suggested position size
            suggested_size = self._compute_position_size(
                kelly_frac, bankroll, current_exposure, confidence, entry_price
            )

            # Build reasoning
            reasons = self._build_reasons(
                our_prob, bias_corrected_prob, market_price, edge,
                confidence, ensemble_stats, direction
            )

            signals.append(TradingSignal(
                market=market,
                outcome=outcome,
                our_probability=bias_corrected_prob,
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
        entry_price: float = 0.5,
    ) -> float:
        """Compute actual position size in USDC.

        Args:
            entry_price: Used to verify the order produces >= 5 shares
                         (Polymarket minimum order size).
        """
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

        # Ensure minimum order produces >= 5 shares (Polymarket requirement)
        min_viable_usd = 5.0 * entry_price  # 5 shares * price per share
        min_required = max(config.MIN_TRADE_SIZE_USDC, min_viable_usd)

        # If size is below minimum but we have capacity, bump to minimum
        if size < min_required:
            if min_required <= remaining_capacity and min_required <= config.MAX_TRADE_SIZE_USDC:
                size = min_required
            else:
                return 0  # Can't meet minimum

        return round(size, 2)

    def _build_reasons(
        self,
        raw_prob: float,
        bias_corrected_prob: float,
        market_price: float,
        edge: float,
        confidence: float,
        ensemble_stats: Dict,
        direction: str,
    ) -> List[str]:
        """Build human-readable reasoning for a trade."""
        reasons = []

        reasons.append(
            f"Raw forecast: {raw_prob:.1%}, Bias-corrected: {bias_corrected_prob:.1%} "
            f"vs Market: {market_price:.1%} (Edge: {edge:.1%})"
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
