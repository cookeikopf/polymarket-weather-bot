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
    """Detect profitable edges using dual strategy: Ladder + Conservative NO.

    V5 Advanced Innovations:
    - Innovation 2: Ensemble spread confidence scaling
    - Innovation 4: Inter-model disagreement detection
    - Innovation 5: Time-decay edge optimization
    - Innovation 6: Market efficiency scoring
    - Innovation 7: Dynamic ladder width + bimodal detection
    """

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
        days_to_resolution: float = 1.0,
    ) -> List[TradingSignal]:
        """
        Find trading opportunities using both strategies + all 7 V5 innovations.
        Returns combined list of signals sorted by expected value.
        """
        signals = []

        # Innovation 6: Market Efficiency Scoring
        market_efficiency = self._compute_market_efficiency(market)
        effective_min_edge = self.min_edge
        if market_efficiency == "sharp":
            multiplier = getattr(config, 'MARKET_SHARP_EDGE_MULTIPLIER', 1.5)
            effective_min_edge = self.min_edge * multiplier

        # Innovation 2: Ensemble Spread Confidence
        spread_multiplier = self._compute_spread_multiplier(ensemble_stats)

        # Innovation 4: Inter-Model Disagreement
        disagreement_info = self._compute_model_disagreement(ensemble_stats)

        # Innovation 5: Time-Decay
        time_decay_multiplier = self._compute_time_decay(
            days_to_resolution, ensemble_stats
        )

        # Innovation 7: Dynamic Ladder Width
        dynamic_buckets = self._compute_dynamic_ladder_width(ensemble_stats)

        # Combined sizing multiplier
        sizing_multiplier = spread_multiplier * disagreement_info["sizing_multiplier"] * time_decay_multiplier

        # ─── STRATEGY 1: LADDER (BUY_YES around ensemble median) ───
        ladder_signals = self._find_ladder_signals(
            market, our_probabilities, ensemble_stats, bankroll, current_exposure,
            sizing_multiplier=sizing_multiplier,
            dynamic_buckets=dynamic_buckets,
        )
        signals.extend(ladder_signals)

        # ─── Innovation 7: BIMODAL strategy ───
        if ensemble_stats.get("is_bimodal", False):
            bimodal_signals = self._find_bimodal_signals(
                market, our_probabilities, ensemble_stats, bankroll, current_exposure,
                sizing_multiplier=sizing_multiplier,
            )
            signals.extend(bimodal_signals)

        # ─── STRATEGY 2: CONSERVATIVE BUY_NO ───
        if getattr(config, 'ALLOW_BUY_NO', True):
            no_signals = self._find_conservative_no_signals(
                market, our_probabilities, ensemble_stats, bankroll, current_exposure,
                effective_min_edge=effective_min_edge,
                sizing_multiplier=sizing_multiplier,
                disagreement_day=disagreement_info["is_disagreement"],
            )
            signals.extend(no_signals)

        # Innovation 4: On disagreement days, add conservative NO on extremes
        if disagreement_info["is_disagreement"]:
            extreme_no_signals = self._find_disagreement_no_signals(
                market, our_probabilities, ensemble_stats, bankroll, current_exposure,
            )
            signals.extend(extreme_no_signals)

        # Innovation 5: Filter low-edge trades at long horizons
        if days_to_resolution >= getattr(config, 'TIME_DECAY_MED_CONF_DAYS', 3) + 1:
            far_min_edge = getattr(config, 'TIME_DECAY_FAR_MIN_EDGE', 0.15)
            signals = [s for s in signals if s.edge_pct >= far_min_edge or s.strategy == "bimodal"]

        # Sort by EV * confidence
        signals.sort(key=lambda s: s.expected_value * s.confidence, reverse=True)

        return signals

    # ─── Innovation 2: Ensemble Spread Confidence ───
    def _compute_spread_multiplier(self, ensemble_stats: Dict) -> float:
        """Scale position sizes based on ensemble spread (std)."""
        member_std = ensemble_stats.get("member_std", ensemble_stats.get("std", 3.0))
        low_std = getattr(config, 'ENSEMBLE_SPREAD_LOW_STD', 2.0)
        high_std = getattr(config, 'ENSEMBLE_SPREAD_HIGH_STD', 4.0)
        high_mult = getattr(config, 'ENSEMBLE_HIGH_CONF_MULTIPLIER', 1.5)
        low_mult = getattr(config, 'ENSEMBLE_LOW_CONF_MULTIPLIER', 0.5)

        if member_std < low_std:
            return high_mult
        elif member_std > high_std:
            return low_mult
        else:
            # Linear interpolation between high and low
            frac = (member_std - low_std) / (high_std - low_std)
            return high_mult + frac * (low_mult - high_mult)

    # ─── Innovation 4: Inter-Model Disagreement ───
    def _compute_model_disagreement(self, ensemble_stats: Dict) -> Dict:
        """Detect inter-model disagreement from per-model medians."""
        per_model_medians = ensemble_stats.get("per_model_medians", {})
        if len(per_model_medians) < 2:
            return {"is_disagreement": False, "is_agreement": False,
                    "model_spread": 0, "sizing_multiplier": 1.0}

        medians = list(per_model_medians.values())
        model_spread = max(medians) - min(medians)
        disagree_thresh = getattr(config, 'MODEL_DISAGREEMENT_THRESHOLD_F', 4.0)
        agree_thresh = getattr(config, 'MODEL_AGREEMENT_THRESHOLD_F', 2.0)

        is_disagreement = model_spread > disagree_thresh
        is_agreement = model_spread < agree_thresh

        if is_disagreement:
            sizing_mult = getattr(config, 'DISAGREEMENT_SIZING_MULTIPLIER', 0.5)
        elif is_agreement:
            sizing_mult = getattr(config, 'AGREEMENT_SIZING_MULTIPLIER', 1.3)
        else:
            sizing_mult = 1.0

        return {
            "is_disagreement": is_disagreement,
            "is_agreement": is_agreement,
            "model_spread": model_spread,
            "sizing_multiplier": sizing_mult,
        }

    # ─── Innovation 5: Time-Decay ───
    def _compute_time_decay(self, days_to_resolution: float, ensemble_stats: Dict) -> float:
        """Calculate time-decay sizing multiplier."""
        full_days = getattr(config, 'TIME_DECAY_FULL_CONF_DAYS', 1)
        med_days = getattr(config, 'TIME_DECAY_MED_CONF_DAYS', 3)
        med_mult = getattr(config, 'TIME_DECAY_MED_MULTIPLIER', 0.7)
        far_mult = getattr(config, 'TIME_DECAY_FAR_MULTIPLIER', 0.4)

        if days_to_resolution <= full_days:
            return 1.0
        elif days_to_resolution <= med_days:
            return med_mult
        else:
            return far_mult

    # ─── Innovation 6: Market Efficiency Scoring ───
    def _compute_market_efficiency(self, market: WeatherMarket) -> str:
        """Score market as 'sharp', 'soft', or 'normal'."""
        prices_sum = sum(o.price for o in market.outcomes if o.price > 0)
        expected_sum = getattr(config, 'MARKET_EXPECTED_SUM', 1.05)
        sharp_thresh = getattr(config, 'MARKET_SHARP_THRESHOLD', 0.03)
        soft_thresh = getattr(config, 'MARKET_SOFT_THRESHOLD', 0.10)

        deviation = abs(prices_sum - expected_sum)
        if deviation < sharp_thresh:
            return "sharp"
        elif deviation > soft_thresh:
            return "soft"
        return "normal"

    # ─── Innovation 7: Dynamic Ladder Width ───
    def _compute_dynamic_ladder_width(self, ensemble_stats: Dict) -> int:
        """Determine number of ladder buckets based on ensemble shape."""
        if ensemble_stats.get("is_bimodal", False):
            return 0  # Bimodal uses separate strategy

        member_std = ensemble_stats.get("member_std", ensemble_stats.get("std", 3.0))
        narrow_std = getattr(config, 'NARROW_PEAK_STD_F', 2.0)
        wide_std = getattr(config, 'WIDE_PEAK_STD_F', 4.0)

        if member_std < narrow_std:
            return 2  # Tight ladder
        elif member_std > wide_std:
            return 4  # Wide ladder
        else:
            return 3  # Standard ladder

    def _find_ladder_signals(
        self,
        market: WeatherMarket,
        our_probabilities: Dict[str, float],
        ensemble_stats: Dict,
        bankroll: float,
        current_exposure: float,
        sizing_multiplier: float = 1.0,
        dynamic_buckets: Optional[int] = None,
    ) -> List[TradingSignal]:
        """
        LADDER STRATEGY: Buy YES shares in N buckets around the ensemble median.

        Innovation 2: sizing_multiplier from ensemble spread confidence.
        Innovation 7: dynamic_buckets from ensemble shape analysis.
        """
        signals = []

        # Get ensemble median (prefer member median if available)
        ensemble_mean = ensemble_stats.get("median", ensemble_stats.get("mean", None))
        if ensemble_mean is None:
            return signals

        # Innovation 7: Use dynamic bucket count if provided
        n_buckets = dynamic_buckets if dynamic_buckets and dynamic_buckets > 0 else self.ladder_buckets

        # Determine temperature unit
        station = config.STATIONS.get(market.station_id, {})
        is_fahrenheit = station.get("unit", "fahrenheit") == "fahrenheit"

        import re
        candidates = []

        for outcome in market.outcomes:
            our_prob = self._match_probability(outcome.name, our_probabilities)
            if our_prob is None:
                continue

            market_price = outcome.price
            if market_price < 0 or market_price > self.ladder_max_price:
                continue

            if market_price <= 0.005:
                continue

            bucket_temp = self._extract_bucket_temp(outcome.name, is_fahrenheit)
            if bucket_temp is None:
                continue

            dist = abs(bucket_temp - ensemble_mean)
            candidates.append({
                "outcome": outcome,
                "our_prob": our_prob,
                "market_price": market_price,
                "bucket_temp": bucket_temp,
                "dist_from_median": dist,
            })

        if not candidates:
            return signals

        candidates.sort(key=lambda c: c["dist_from_median"])
        ladder = candidates[:n_buckets]

        if not ladder:
            return signals

        agreement = ensemble_stats.get("agreement", 0.5)
        n_models = ensemble_stats.get("n_models", 1)

        for c in ladder:
            outcome = c["outcome"]
            our_prob = c["our_prob"]
            market_price = c["market_price"]

            clob_ask = getattr(outcome, '_clob_ask', None)
            entry_price = clob_ask if clob_ask else market_price

            if entry_price <= 0 or entry_price >= 1:
                continue

            edge = our_prob - entry_price
            if our_prob < 0.01:
                continue

            win_pnl = (1.0 - entry_price)
            loss_pnl = entry_price
            ev_per_share = our_prob * win_pnl - (1 - our_prob) * loss_pnl
            ev_per_dollar = ev_per_share / entry_price

            dist = c["dist_from_median"]
            max_dist = (4 if is_fahrenheit else 2)
            proximity_score = max(0, 1.0 - dist / (max_dist * 2))
            confidence = (0.40 * agreement + 0.30 * proximity_score + 0.30 * min(1.0, n_models / 5))

            # Innovation 2+4+5: Apply combined sizing multiplier
            suggested_size = self.ladder_bet_per_bucket * sizing_multiplier

            remaining = (bankroll * config.MAX_TOTAL_EXPOSURE) - current_exposure
            if remaining < suggested_size:
                continue

            reasons = [
                f"LADDER: Bucket {outcome.name} near ensemble median {ensemble_mean:.1f}°",
                f"Distance from median: {dist:.1f}° | Market price: {market_price:.3f}",
                f"Our probability: {our_prob:.1%} | Payout if correct: {1.0/entry_price:.0f}x",
                f"Ensemble: {n_models} models, agreement {agreement:.1%}",
                f"Sizing multiplier: {sizing_multiplier:.2f}x | Ladder width: {n_buckets} buckets",
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
                kelly_fraction=0,
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
        effective_min_edge: float = None,
        sizing_multiplier: float = 1.0,
        disagreement_day: bool = False,
    ) -> List[TradingSignal]:
        """
        CONSERVATIVE BUY_NO: Buy NO on outcomes that are unlikely.
        Innovation 4: On disagreement days, prioritize conservative NO trades.
        Innovation 6: Use effective_min_edge (higher for sharp markets).
        """
        signals = []
        min_edge = effective_min_edge if effective_min_edge is not None else self.min_edge

        for outcome in market.outcomes:
            our_prob = self._match_probability(outcome.name, our_probabilities)
            if our_prob is None:
                continue

            market_price = outcome.price
            if market_price < 0:
                continue

            clob_ask = getattr(outcome, '_clob_ask', None)
            clob_bid = getattr(outcome, '_clob_bid', None)

            if clob_bid:
                entry_price_no = 1.0 - clob_bid
            else:
                entry_price_no = 1.0 - market_price

            if entry_price_no < self.conservative_no_min_entry:
                continue
            if entry_price_no > self.conservative_no_max_entry:
                continue

            our_prob_no = 1.0 - our_prob
            if clob_bid:
                edge = our_prob_no - entry_price_no
            else:
                edge = (1.0 - our_prob) - (1.0 - market_price)

            if edge < min_edge:
                continue

            agreement = ensemble_stats.get("agreement", 0.5)
            n_models = ensemble_stats.get("n_models", 1)
            model_std = ensemble_stats.get("std", 5.0)
            prob_uncertainty = min(0.30, model_std * 0.03)
            edge_significance = min(1.0, edge / max(prob_uncertainty, 0.01))
            confidence = (0.40 * agreement + 0.35 * edge_significance + 0.25 * min(1.0, n_models / 5))

            if confidence < config.MIN_ENSEMBLE_AGREEMENT:
                continue

            win_pnl = 1.0 - entry_price_no
            loss_pnl = entry_price_no
            ev_per_share = our_prob_no * win_pnl - (1 - our_prob_no) * loss_pnl
            ev_per_dollar = ev_per_share / entry_price_no

            kelly_frac = self._kelly_criterion(our_prob_no, entry_price_no)
            suggested_size = self._compute_position_size(
                kelly_frac, bankroll, current_exposure, confidence, entry_price_no
            )
            # Apply innovation multipliers
            suggested_size *= sizing_multiplier

            reasons = [
                f"CONSERVATIVE NO: {outcome.name} unlikely (our P(NO)={our_prob_no:.1%})",
                f"Entry NO: {entry_price_no:.3f} | Edge: {edge:.1%}",
                f"Ensemble: {ensemble_stats.get('n_models', 0)} models, agreement {agreement:.1%}",
                f"Sizing multiplier: {sizing_multiplier:.2f}x",
            ]
            if disagreement_day:
                reasons.append("DISAGREEMENT DAY: Conservative NO prioritized")

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

    # ─── Innovation 7: Bimodal Strategy ───
    def _find_bimodal_signals(
        self,
        market: WeatherMarket,
        our_probabilities: Dict[str, float],
        ensemble_stats: Dict,
        bankroll: float,
        current_exposure: float,
        sizing_multiplier: float = 1.0,
    ) -> List[TradingSignal]:
        """
        Innovation 7: When ensemble is bimodal, buy YES on both peaks and NO on valley.
        """
        signals = []
        peaks = ensemble_stats.get("bimodal_peaks", [])
        if len(peaks) < 2:
            return signals

        station = config.STATIONS.get(market.station_id, {})
        is_fahrenheit = station.get("unit", "fahrenheit") == "fahrenheit"

        peak1, peak2 = peaks[0], peaks[1]
        valley_center = (peak1 + peak2) / 2.0

        for outcome in market.outcomes:
            our_prob = self._match_probability(outcome.name, our_probabilities)
            if our_prob is None:
                continue

            market_price = outcome.price
            if market_price <= 0.005:
                continue

            bucket_temp = self._extract_bucket_temp(outcome.name, is_fahrenheit)
            if bucket_temp is None:
                continue

            # Buy YES on buckets near peaks (within 2°F)
            near_peak1 = abs(bucket_temp - peak1) <= 2
            near_peak2 = abs(bucket_temp - peak2) <= 2
            near_valley = abs(bucket_temp - valley_center) <= 2 and not near_peak1 and not near_peak2

            if near_peak1 or near_peak2:
                if market_price > self.ladder_max_price:
                    continue
                entry_price = market_price
                edge = our_prob - entry_price
                if our_prob < 0.01:
                    continue

                win_pnl = 1.0 - entry_price
                ev_per_share = our_prob * win_pnl - (1 - our_prob) * entry_price
                ev_per_dollar = ev_per_share / entry_price if entry_price > 0 else 0

                size = self.ladder_bet_per_bucket * sizing_multiplier
                remaining = (bankroll * config.MAX_TOTAL_EXPOSURE) - current_exposure
                if remaining < size:
                    continue

                peak_label = "peak1" if near_peak1 else "peak2"
                signals.append(TradingSignal(
                    market=market, outcome=outcome,
                    our_probability=our_prob, market_price=market_price,
                    edge=max(edge, 0),
                    edge_pct=abs(edge) / max(market_price, 0.01),
                    confidence=0.6,
                    direction="BUY_YES", kelly_fraction=0,
                    suggested_size_usd=size,
                    expected_value=ev_per_dollar,
                    reasons=[
                        f"BIMODAL {peak_label}: Bucket {outcome.name} near peak {peak1 if near_peak1 else peak2:.1f}°",
                        f"Bimodal distribution: peaks at {peak1:.1f}° and {peak2:.1f}°",
                    ],
                    strategy="bimodal",
                ))

            elif near_valley and market_price > 0.05:
                # Buy NO on valley bucket
                entry_price_no = 1.0 - market_price
                our_prob_no = 1.0 - our_prob
                edge = our_prob_no - entry_price_no
                if edge < self.min_edge:
                    continue

                win_pnl = 1.0 - entry_price_no
                ev_per_share = our_prob_no * win_pnl - (1 - our_prob_no) * entry_price_no
                ev_per_dollar = ev_per_share / entry_price_no if entry_price_no > 0 else 0

                size = self.ladder_bet_per_bucket * sizing_multiplier * 0.5
                remaining = (bankroll * config.MAX_TOTAL_EXPOSURE) - current_exposure
                if remaining < size:
                    continue

                signals.append(TradingSignal(
                    market=market, outcome=outcome,
                    our_probability=our_prob, market_price=market_price,
                    edge=edge,
                    edge_pct=abs(edge) / max(market_price, 0.01),
                    confidence=0.55,
                    direction="BUY_NO", kelly_fraction=0,
                    suggested_size_usd=size,
                    expected_value=ev_per_dollar,
                    reasons=[
                        f"BIMODAL VALLEY NO: {outcome.name} in valley between peaks",
                        f"Valley center: {valley_center:.1f}° | Peaks: {peak1:.1f}°, {peak2:.1f}°",
                    ],
                    strategy="bimodal",
                ))

        return signals

    # ─── Innovation 4: Disagreement Day NO Strategy ───
    def _find_disagreement_no_signals(
        self,
        market: WeatherMarket,
        our_probabilities: Dict[str, float],
        ensemble_stats: Dict,
        bankroll: float,
        current_exposure: float,
    ) -> List[TradingSignal]:
        """
        Innovation 4: On disagreement days, buy NO on extreme tail outcomes.
        When models disagree, the truth is usually somewhere in the middle.
        """
        signals = []
        per_model_medians = ensemble_stats.get("per_model_medians", {})
        if not per_model_medians:
            return signals

        # Find the consensus range (middle zone where 2+ models agree)
        medians = sorted(per_model_medians.values())
        consensus_low = medians[0]
        consensus_high = medians[-1]

        station = config.STATIONS.get(market.station_id, {})
        is_fahrenheit = station.get("unit", "fahrenheit") == "fahrenheit"

        for outcome in market.outcomes:
            our_prob = self._match_probability(outcome.name, our_probabilities)
            if our_prob is None or our_prob > 0.05:
                continue  # Only target very unlikely outcomes

            market_price = outcome.price
            if market_price <= 0.005:
                continue

            bucket_temp = self._extract_bucket_temp(outcome.name, is_fahrenheit)
            if bucket_temp is None:
                continue

            # Only target buckets far outside ALL model medians
            margin = 6.0 if is_fahrenheit else 3.5
            if consensus_low - margin < bucket_temp < consensus_high + margin:
                continue  # Not extreme enough

            entry_price_no = 1.0 - market_price
            if entry_price_no < 0.55 or entry_price_no > 0.90:
                continue

            our_prob_no = 1.0 - our_prob
            edge = our_prob_no - entry_price_no
            if edge < self.min_edge:
                continue

            win_pnl = 1.0 - entry_price_no
            ev_per_share = our_prob_no * win_pnl - (1 - our_prob_no) * entry_price_no
            ev_per_dollar = ev_per_share / entry_price_no if entry_price_no > 0 else 0

            size = min(self.ladder_bet_per_bucket, bankroll * 0.03)
            remaining = (bankroll * config.MAX_TOTAL_EXPOSURE) - current_exposure
            if remaining < size or size < 1.0:
                continue

            signals.append(TradingSignal(
                market=market, outcome=outcome,
                our_probability=our_prob, market_price=market_price,
                edge=edge,
                edge_pct=abs(edge) / max(market_price, 0.01),
                confidence=0.5,
                direction="BUY_NO", kelly_fraction=0,
                suggested_size_usd=size,
                expected_value=ev_per_dollar,
                reasons=[
                    f"DISAGREEMENT NO: {outcome.name} extreme tail, all models disagree away",
                    f"Model medians: {[f'{v:.1f}' for v in medians]} | Bucket: {bucket_temp:.1f}°",
                ],
                strategy="disagreement_no",
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
