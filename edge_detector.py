"""
Edge Detection & Signal Generation (V4 — Dual Strategy)
=========================================================
Two independent profit strategies:

1. LADDER STRATEGY (primary profit driver):
   - Identify the most-likely temperature bucket from ensemble median
   - Buy YES shares in 3-5 buckets around the median at LOW prices (<$0.20)
   - 85%+ hit rate on 1-day forecasts → massive asymmetric payouts
   - Win: $10-25+ per hit | Loss: $2-10 per miss
   - This is how neobrother ($20k+) and the $24k bot work

2. CONSERVATIVE BUY_NO (steady income):
   - Buy NO on outcomes unlikely to happen (high entry price = high probability of NO)
   - Only at entry >= 0.65 (80% win rate from retro-analysis)
   - Win: $1-3 per trade | Loss: $5 per trade
   - Requires high win rate but provides steady income stream

KEY INSIGHT: Weather markets on Polymarket have NO taker fees.
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
    edge: float
    edge_pct: float
    confidence: float
    direction: str  # "BUY_YES" or "BUY_NO"
    kelly_fraction: float
    suggested_size_usd: float
    expected_value: float
    reasons: List[str]
    strategy: str = "ladder"  # "ladder" or "conservative_no"


class EdgeDetector:
    """Detect profitable edges using dual strategy: Ladder + Conservative NO."""

    def __init__(self):
        self.min_edge = config.MIN_EDGE_PCT
        self.min_probability = config.MIN_PROBABILITY
        self.max_entry_price = getattr(config, 'MAX_ENTRY_PRICE', 0.85)
        self.min_entry_price = getattr(config, 'MIN_ENTRY_PRICE', 0.55)

        # Ladder strategy params
        self.ladder_max_price = getattr(config, 'LADDER_MAX_ENTRY_PRICE', 0.20)
        self.ladder_buckets = getattr(config, 'LADDER_BUCKETS', 5)
        self.ladder_bet_per_bucket = getattr(config, 'LADDER_BET_PER_BUCKET', 2.0)

        # Conservative NO params
        self.conservative_no_min_entry = getattr(config, 'CONSERVATIVE_NO_MIN_ENTRY', 0.65)
        self.conservative_no_max_entry = getattr(config, 'CONSERVATIVE_NO_MAX_ENTRY', 0.85)

    def find_edges(
        self,
        market: WeatherMarket,
        our_probabilities: Dict[str, float],
        ensemble_stats: Dict,
        bankroll: float,
        current_exposure: float = 0,
    ) -> List[TradingSignal]:
        """
        Find trading opportunities using both strategies.
        Returns combined list of signals sorted by expected value.
        """
        signals = []

        # ─── STRATEGY 1: LADDER (BUY_YES around ensemble median) ───
        ladder_signals = self._find_ladder_signals(
            market, our_probabilities, ensemble_stats, bankroll, current_exposure
        )
        signals.extend(ladder_signals)

        # ─── STRATEGY 2: CONSERVATIVE BUY_NO ───
        if getattr(config, 'ALLOW_BUY_NO', True):
            no_signals = self._find_conservative_no_signals(
                market, our_probabilities, ensemble_stats, bankroll, current_exposure
            )
            signals.extend(no_signals)

        # Sort by EV * confidence
        signals.sort(key=lambda s: s.expected_value * s.confidence, reverse=True)

        return signals

    def _find_ladder_signals(
        self,
        market: WeatherMarket,
        our_probabilities: Dict[str, float],
        ensemble_stats: Dict,
        bankroll: float,
        current_exposure: float,
    ) -> List[TradingSignal]:
        """
        LADDER STRATEGY: Buy YES shares in 3-5 buckets around the ensemble median.

        Logic:
        1. Get ensemble median temperature forecast
        2. Find the market bucket that contains this median
        3. Buy YES in this bucket + adjacent buckets (the "ladder")
        4. Only buy if market price is LOW (< $0.20) = massive upside if correct
        5. 1 correct bucket → $1/share payout → huge return on 5-15 cent investment

        Why this works:
        - 1-day NWP forecasts are 85-90% accurate (within a few degrees)
        - Market often underprices the most-likely bucket
        - Payout asymmetry: 700-1900% return on winning bucket vs small loss on losers
        """
        signals = []

        # Get ensemble median
        ensemble_mean = ensemble_stats.get("mean", None)
        if ensemble_mean is None:
            return signals

        # Determine temperature unit
        station = config.STATIONS.get(market.station_id, {})
        is_fahrenheit = station.get("unit", "fahrenheit") == "fahrenheit"

        # Find which outcomes are near the ensemble median
        # and score each by distance from median + market price
        import re
        candidates = []

        for outcome in market.outcomes:
            our_prob = self._match_probability(outcome.name, our_probabilities)
            if our_prob is None:
                continue

            market_price = outcome.price
            if market_price < 0 or market_price > self.ladder_max_price:
                continue  # Only buy cheap shares

            # Skip outcomes with no CLOB data
            if market_price <= 0.005:
                continue

            # Determine the temperature value for this bucket
            bucket_temp = self._extract_bucket_temp(outcome.name, is_fahrenheit)
            if bucket_temp is None:
                continue

            # Distance from ensemble median (in degrees)
            dist = abs(bucket_temp - ensemble_mean)

            # Score: prefer buckets close to median AND with low market price
            # Close to median → high chance of hitting
            # Low price → high payout multiplier
            candidates.append({
                "outcome": outcome,
                "our_prob": our_prob,
                "market_price": market_price,
                "bucket_temp": bucket_temp,
                "dist_from_median": dist,
            })

        if not candidates:
            return signals

        # Sort by distance from median (closest first)
        candidates.sort(key=lambda c: c["dist_from_median"])

        # Take top N closest to median (the ladder)
        ladder = candidates[:self.ladder_buckets]

        if not ladder:
            return signals

        # Calculate combined ladder statistics
        total_cost = sum(c["market_price"] for c in ladder)
        total_prob_hit = sum(c["our_prob"] for c in ladder)  # prob at least one hits

        # Confidence from ensemble agreement
        agreement = ensemble_stats.get("agreement", 0.5)
        n_models = ensemble_stats.get("n_models", 1)

        for c in ladder:
            outcome = c["outcome"]
            our_prob = c["our_prob"]
            market_price = c["market_price"]

            # Use CLOB spread-aware prices if available
            clob_ask = getattr(outcome, '_clob_ask', None)
            entry_price = clob_ask if clob_ask else market_price

            if entry_price <= 0 or entry_price >= 1:
                continue

            # Edge: our probability minus what we pay
            edge = our_prob - entry_price

            # For ladder strategy, we don't require positive edge on each bucket
            # The COMBINED ladder has positive EV because the winning bucket pays 700%+
            # But skip if our model gives 0% probability
            if our_prob < 0.01:
                continue

            # Expected value for this single bucket (standalone)
            win_pnl = (1.0 - entry_price)  # payout per share if correct
            loss_pnl = entry_price  # cost per share if wrong
            ev_per_share = our_prob * win_pnl - (1 - our_prob) * loss_pnl
            ev_per_dollar = ev_per_share / entry_price

            # Confidence: higher if models agree and bucket is close to median
            dist = c["dist_from_median"]
            max_dist = (4 if is_fahrenheit else 2)  # reasonable ladder width
            proximity_score = max(0, 1.0 - dist / (max_dist * 2))
            confidence = (0.40 * agreement + 0.30 * proximity_score + 0.30 * min(1.0, n_models / 5))

            # Position size: fixed per ladder bucket
            suggested_size = self.ladder_bet_per_bucket

            # Check remaining capacity
            remaining = (bankroll * config.MAX_TOTAL_EXPOSURE) - current_exposure
            if remaining < suggested_size:
                continue

            reasons = [
                f"LADDER: Bucket {outcome.name} near ensemble median {ensemble_mean:.1f}°",
                f"Distance from median: {dist:.1f}° | Market price: {market_price:.3f}",
                f"Our probability: {our_prob:.1%} | Payout if correct: {1.0/entry_price:.0f}x",
                f"Ensemble: {n_models} models, agreement {agreement:.1%}",
            ]

            if dist <= (2 if is_fahrenheit else 1):
                reasons.append("CORE LADDER: Within 1 bucket of median")
            else:
                reasons.append("EDGE LADDER: Extended range bucket")

            signals.append(TradingSignal(
                market=market,
                outcome=outcome,
                our_probability=our_prob,
                market_price=market_price,
                edge=max(edge, 0),
                edge_pct=abs(edge) / max(market_price, 0.01),
                confidence=confidence,
                direction="BUY_YES",
                kelly_fraction=0,  # Fixed sizing for ladder
                suggested_size_usd=suggested_size,
                expected_value=ev_per_dollar,
                reasons=reasons,
                strategy="ladder",
            ))

        return signals

    def _find_conservative_no_signals(
        self,
        market: WeatherMarket,
        our_probabilities: Dict[str, float],
        ensemble_stats: Dict,
        bankroll: float,
        current_exposure: float,
    ) -> List[TradingSignal]:
        """
        CONSERVATIVE BUY_NO: Buy NO on outcomes that are unlikely.
        Only at high entry prices (= high NO probability = safe bets).
        """
        signals = []

        for outcome in market.outcomes:
            our_prob = self._match_probability(outcome.name, our_probabilities)
            if our_prob is None:
                continue

            market_price = outcome.price
            if market_price < 0:
                continue

            # Use spread-aware prices
            clob_ask = getattr(outcome, '_clob_ask', None)
            clob_bid = getattr(outcome, '_clob_bid', None)

            # BUY_NO: we bet that this outcome does NOT happen
            # Entry price for NO = 1 - bid (what we pay for NO shares)
            if clob_bid:
                entry_price_no = 1.0 - clob_bid
            else:
                entry_price_no = 1.0 - market_price

            # Conservative NO filter: only high-entry (high probability of NO winning)
            if entry_price_no < self.conservative_no_min_entry:
                continue
            if entry_price_no > self.conservative_no_max_entry:
                continue

            # Edge calculation
            our_prob_no = 1.0 - our_prob  # our probability of NO
            if clob_bid:
                edge = our_prob_no - entry_price_no
            else:
                edge = (1.0 - our_prob) - (1.0 - market_price)

            # Require minimum edge
            if edge < self.min_edge:
                continue

            # Confidence
            agreement = ensemble_stats.get("agreement", 0.5)
            n_models = ensemble_stats.get("n_models", 1)
            model_std = ensemble_stats.get("std", 5.0)
            prob_uncertainty = min(0.30, model_std * 0.03)
            edge_significance = min(1.0, edge / max(prob_uncertainty, 0.01))
            confidence = (0.40 * agreement + 0.35 * edge_significance + 0.25 * min(1.0, n_models / 5))

            if confidence < config.MIN_ENSEMBLE_AGREEMENT:
                continue

            # EV
            win_pnl = 1.0 - entry_price_no  # payout per share
            loss_pnl = entry_price_no
            ev_per_share = our_prob_no * win_pnl - (1 - our_prob_no) * loss_pnl
            ev_per_dollar = ev_per_share / entry_price_no

            # Position sizing via Kelly
            kelly_frac = self._kelly_criterion(our_prob_no, entry_price_no)
            suggested_size = self._compute_position_size(
                kelly_frac, bankroll, current_exposure, confidence, entry_price_no
            )

            reasons = [
                f"CONSERVATIVE NO: {outcome.name} unlikely (our P(NO)={our_prob_no:.1%})",
                f"Entry NO: {entry_price_no:.3f} | Edge: {edge:.1%}",
                f"Ensemble: {ensemble_stats.get('n_models', 0)} models, agreement {agreement:.1%}",
            ]

            signals.append(TradingSignal(
                market=market,
                outcome=outcome,
                our_probability=our_prob,
                market_price=market_price,
                edge=edge,
                edge_pct=abs(edge) / max(market_price, 0.01),
                confidence=confidence,
                direction="BUY_NO",
                kelly_fraction=kelly_frac,
                suggested_size_usd=suggested_size,
                expected_value=ev_per_dollar,
                reasons=reasons,
                strategy="conservative_no",
            ))

        return signals

    def _extract_bucket_temp(self, outcome_name: str, is_fahrenheit: bool) -> Optional[float]:
        """Extract the center temperature of a bucket from its name."""
        import re

        # °F range: "34-35°F" → center = 34.5
        match = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F', outcome_name)
        if match:
            return (int(match.group(1)) + int(match.group(2))) / 2.0

        # °C single degree: "14°C" → center = 14
        match = re.search(r'(-?\d+)\s*°\s*C', outcome_name)
        if match:
            return float(match.group(1))

        # "X or below" → treat as X - step/2
        match = re.search(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*below', outcome_name, re.I)
        if match:
            return float(match.group(1)) - (1 if is_fahrenheit else 0.5)

        # "X or higher" → treat as X + step/2
        match = re.search(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*higher', outcome_name, re.I)
        if match:
            return float(match.group(1)) + (1 if is_fahrenheit else 0.5)

        return None

    def _match_probability(
        self, outcome_name: str, our_probs: Dict[str, float]
    ) -> Optional[float]:
        """Match a market outcome name to our probability distribution."""
        # Direct match
        if outcome_name in our_probs:
            return our_probs[outcome_name]

        # Fuzzy matching (case-insensitive, strip whitespace)
        outcome_lower = outcome_name.lower().strip()
        for key, prob in our_probs.items():
            if key.lower().strip() == outcome_lower:
                return prob

        import re

        # Normalize both sides
        def normalize_label(s):
            s = s.strip()
            s = re.sub(r'\s*°\s*', '°', s)
            s = re.sub(r'\s*[-–]\s*', '-', s)
            return s

        norm_outcome = normalize_label(outcome_name)
        for key, prob in our_probs.items():
            if normalize_label(key) == norm_outcome:
                return prob

        # °C single-degree
        match_c = re.match(r'^(-?\d+)\s*°\s*C$', outcome_name.strip())
        if match_c:
            for key, prob in our_probs.items():
                key_match = re.match(r'^(-?\d+)\s*°\s*C$', key.strip())
                if key_match and key_match.group(1) == match_c.group(1):
                    return prob

        # °F range
        match_f = re.match(r'^(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F$', outcome_name.strip())
        if match_f:
            for key, prob in our_probs.items():
                key_match = re.match(r'^(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F$', key.strip())
                if key_match and key_match.group(1) == match_f.group(1) and key_match.group(2) == match_f.group(2):
                    return prob

        # "or higher" / "or below"
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

    def _kelly_criterion(self, win_prob: float, entry_price: float) -> float:
        """Kelly Criterion for binary outcomes."""
        if entry_price <= 0 or entry_price >= 1:
            return 0

        b = (1.0 - entry_price) / entry_price
        p = win_prob
        q = 1.0 - p

        kelly = (b * p - q) / b
        kelly *= config.KELLY_FRACTION

        return max(0, min(kelly, config.MAX_POSITION_PCT))

    def _compute_position_size(
        self,
        kelly_frac: float,
        bankroll: float,
        current_exposure: float,
        confidence: float,
        entry_price: float = 0.5,
    ) -> float:
        """Compute position size in USDC for conservative NO strategy."""
        kelly_size = kelly_frac * bankroll
        size = kelly_size * confidence

        size = min(size, bankroll * config.MAX_POSITION_PCT)
        size = min(size, config.MAX_TRADE_SIZE_USDC)

        remaining_capacity = (bankroll * config.MAX_TOTAL_EXPOSURE) - current_exposure
        if remaining_capacity <= 0:
            return 0
        size = min(size, remaining_capacity)

        min_viable_usd = 5.0 * entry_price
        min_required = max(config.MIN_TRADE_SIZE_USDC, min_viable_usd)

        if size < min_required:
            if min_required <= remaining_capacity and min_required <= config.MAX_TRADE_SIZE_USDC:
                size = min_required
            else:
                return 0

        return round(size, 2)
