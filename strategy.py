"""
Edge Detection & Signal Generation V6
========================================
Dual strategy: Ladder (BUY YES) + Conservative NO (BUY NO).
Includes ensemble confidence, model disagreement, time decay, market efficiency scoring.
"""

import re
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

import config as cfg
from markets import WeatherMarket, MarketOutcome


@dataclass
class TradingSignal:
    """A detected trading opportunity."""
    market: WeatherMarket
    outcome: MarketOutcome
    our_probability: float
    market_price: float
    edge: float
    confidence: float
    direction: str          # "BUY_YES" or "BUY_NO"
    suggested_size_usd: float
    expected_value: float   # EV per dollar
    strategy: str           # "ladder" or "conservative_no"
    reasons: List[str]


class EdgeDetector:
    """Detect profitable edges using dual strategy."""

    def find_edges(self, market: WeatherMarket, our_probs: Dict[str, float],
                   ensemble_stats: dict, bankroll: float,
                   current_exposure: float = 0, days_to_res: float = 1.0) -> List[TradingSignal]:
        """Find all trading opportunities for a market."""
        signals = []

        # Compute modifiers
        sizing_mult = self._sizing_multiplier(ensemble_stats, days_to_res)
        eff_min_edge = cfg.MIN_EDGE_PCT
        if self._market_efficiency(market) == "sharp":
            eff_min_edge *= cfg.MARKET_SHARP_EDGE_MULT

        # Strategy 1: Ladder
        if cfg.LADDER_ENABLED:
            signals.extend(self._ladder_signals(
                market, our_probs, ensemble_stats, bankroll, current_exposure, sizing_mult
            ))

        # Strategy 2: Conservative NO
        if cfg.ALLOW_BUY_NO:
            signals.extend(self._conservative_no_signals(
                market, our_probs, ensemble_stats, bankroll, current_exposure,
                eff_min_edge, sizing_mult
            ))

        # Filter low-edge trades at long horizons
        if days_to_res >= cfg.TIME_DECAY_MED_DAYS + 1:
            signals = [s for s in signals if s.edge >= cfg.TIME_DECAY_FAR_MIN_EDGE]

        signals.sort(key=lambda s: s.expected_value * s.confidence, reverse=True)
        return signals

    # ─── Strategy 1: Ladder ───────────────────────────────────────

    def _ladder_signals(self, market, our_probs, stats, bankroll,
                         exposure, sizing_mult) -> List[TradingSignal]:
        """Buy YES on 3 buckets nearest to ensemble median at low prices."""
        signals = []
        median = stats.get("median", stats.get("mean"))
        if median is None:
            return signals

        station = cfg.STATIONS.get(market.station_id, {})
        is_f = station.get("unit") == "fahrenheit"

        # Find candidates near median
        candidates = []
        for outcome in market.outcomes:
            prob = match_probability(outcome.name, our_probs)
            if prob is None or prob < 0.01:
                continue
            price = outcome.price
            if price < 0 or price > cfg.LADDER_MAX_ENTRY_PRICE or price <= 0.005:
                continue
            bucket_temp = extract_bucket_temp(outcome.name, is_f)
            if bucket_temp is None:
                continue
            dist = abs(bucket_temp - median)
            candidates.append((outcome, prob, price, bucket_temp, dist))

        candidates.sort(key=lambda c: c[4])  # Sort by distance from median

        agreement = stats.get("agreement", 0.5)
        n_models = stats.get("n_models", 1)

        for outcome, prob, price, temp, dist in candidates[:cfg.LADDER_BUCKETS]:
            entry = outcome.clob_ask if outcome.clob_ask > 0 else price
            if entry <= 0 or entry >= 1:
                continue

            edge = prob - entry
            win_pnl = 1.0 - entry
            ev_per_share = prob * win_pnl - (1 - prob) * entry
            ev_per_dollar = ev_per_share / entry

            max_dist = 4 if is_f else 2
            prox = max(0, 1.0 - dist / (max_dist * 2))
            confidence = 0.4 * agreement + 0.3 * prox + 0.3 * min(1.0, n_models / 5)

            size = cfg.LADDER_BET_PER_BUCKET * sizing_mult
            remaining = bankroll * cfg.MAX_TOTAL_EXPOSURE - exposure
            if remaining < size:
                continue

            signals.append(TradingSignal(
                market=market, outcome=outcome, our_probability=prob,
                market_price=price, edge=max(edge, 0), confidence=confidence,
                direction="BUY_YES", suggested_size_usd=size,
                expected_value=ev_per_dollar, strategy="ladder",
                reasons=[
                    f"LADDER: {outcome.name} near median {median:.1f}°",
                    f"Dist {dist:.1f}° | Price {price:.3f} | P(our)={prob:.1%} | Payout {1/entry:.0f}x",
                ],
            ))

        return signals

    # ─── Strategy 2: Conservative NO ──────────────────────────────

    def _conservative_no_signals(self, market, our_probs, stats, bankroll,
                                  exposure, min_edge, sizing_mult) -> List[TradingSignal]:
        """Buy NO on unlikely outcomes at high entry prices."""
        signals = []
        agreement = stats.get("agreement", 0.5)
        n_models = stats.get("n_models", 1)

        for outcome in market.outcomes:
            prob = match_probability(outcome.name, our_probs)
            if prob is None:
                continue
            price = outcome.price
            if price < 0:
                continue

            # NO entry price
            if outcome.clob_bid > 0:
                entry_no = 1.0 - outcome.clob_bid
            else:
                entry_no = 1.0 - price

            if entry_no < cfg.CONSERVATIVE_NO_MIN_ENTRY or entry_no > cfg.CONSERVATIVE_NO_MAX_ENTRY:
                continue

            prob_no = 1.0 - prob
            edge = prob_no - entry_no
            if edge < min_edge:
                continue

            # Confidence
            model_std = stats.get("std", 5.0)
            prob_unc = min(0.30, model_std * 0.03)
            edge_sig = min(1.0, edge / max(prob_unc, 0.01))
            confidence = 0.4 * agreement + 0.35 * edge_sig + 0.25 * min(1.0, n_models / 5)
            if confidence < 0.45:
                continue

            # EV and sizing
            ev_per_share = prob_no * (1 - entry_no) - (1 - prob_no) * entry_no
            ev_per_dollar = ev_per_share / entry_no

            kelly = self._kelly(prob_no, entry_no)
            size = min(kelly * bankroll * confidence, cfg.MAX_TRADE_SIZE_USDC) * sizing_mult
            remaining = bankroll * cfg.MAX_TOTAL_EXPOSURE - exposure
            size = min(size, remaining)
            if size < cfg.MIN_TRADE_SIZE_USDC:
                size = cfg.MIN_TRADE_SIZE_USDC if cfg.MIN_TRADE_SIZE_USDC <= remaining else 0
            if size <= 0:
                continue

            signals.append(TradingSignal(
                market=market, outcome=outcome, our_probability=prob,
                market_price=price, edge=edge, confidence=confidence,
                direction="BUY_NO", suggested_size_usd=round(size, 2),
                expected_value=ev_per_dollar, strategy="conservative_no",
                reasons=[
                    f"NO: {outcome.name} unlikely (P(NO)={prob_no:.1%})",
                    f"Entry {entry_no:.3f} | Edge {edge:.1%}",
                ],
            ))

        return signals

    # ─── Helpers ──────────────────────────────────────────────────

    def _sizing_multiplier(self, stats: dict, days_to_res: float) -> float:
        """Combined sizing multiplier from spread, disagreement, and time decay."""
        # Spread confidence
        std = stats.get("member_std", stats.get("std", 3.0))
        if std < cfg.ENSEMBLE_SPREAD_LOW_STD:
            spread_m = cfg.ENSEMBLE_HIGH_CONF_MULT
        elif std > cfg.ENSEMBLE_SPREAD_HIGH_STD:
            spread_m = cfg.ENSEMBLE_LOW_CONF_MULT
        else:
            frac = (std - cfg.ENSEMBLE_SPREAD_LOW_STD) / (cfg.ENSEMBLE_SPREAD_HIGH_STD - cfg.ENSEMBLE_SPREAD_LOW_STD)
            spread_m = cfg.ENSEMBLE_HIGH_CONF_MULT + frac * (cfg.ENSEMBLE_LOW_CONF_MULT - cfg.ENSEMBLE_HIGH_CONF_MULT)

        # Model disagreement
        medians = list(stats.get("per_model_medians", {}).values())
        if len(medians) >= 2:
            model_spread = max(medians) - min(medians)
            if model_spread > cfg.MODEL_DISAGREEMENT_THRESH_F:
                disagree_m = cfg.DISAGREEMENT_SIZING_MULT
            elif model_spread < cfg.MODEL_AGREEMENT_THRESH_F:
                disagree_m = cfg.AGREEMENT_SIZING_MULT
            else:
                disagree_m = 1.0
        else:
            disagree_m = 1.0

        # Time decay
        if days_to_res <= cfg.TIME_DECAY_FULL_DAYS:
            time_m = 1.0
        elif days_to_res <= cfg.TIME_DECAY_MED_DAYS:
            time_m = cfg.TIME_DECAY_MED_MULT
        else:
            time_m = cfg.TIME_DECAY_FAR_MULT

        return spread_m * disagree_m * time_m

    def _market_efficiency(self, market: WeatherMarket) -> str:
        prices_sum = sum(o.price for o in market.outcomes if o.price > 0)
        dev = abs(prices_sum - cfg.MARKET_EXPECTED_SUM)
        if dev < cfg.MARKET_SHARP_THRESHOLD:
            return "sharp"
        if dev > cfg.MARKET_SOFT_THRESHOLD:
            return "soft"
        return "normal"

    def _kelly(self, win_prob: float, entry_price: float) -> float:
        if entry_price <= 0 or entry_price >= 1:
            return 0
        b = (1.0 - entry_price) / entry_price
        kelly = (b * win_prob - (1 - win_prob)) / b
        return max(0, min(kelly * cfg.KELLY_FRACTION, cfg.MAX_POSITION_PCT))


# ─── Probability matching ─────────────────────────────────────────

def match_probability(outcome_name: str, probs: Dict[str, float]) -> Optional[float]:
    """Match a market outcome name to our probability distribution."""
    if outcome_name in probs:
        return probs[outcome_name]

    # Case-insensitive
    low = outcome_name.lower().strip()
    for k, v in probs.items():
        if k.lower().strip() == low:
            return v

    # Normalize whitespace around symbols
    def norm(s):
        s = re.sub(r'\s*°\s*', '°', s.strip())
        s = re.sub(r'\s*[-–]\s*', '-', s)
        return s

    n = norm(outcome_name)
    for k, v in probs.items():
        if norm(k) == n:
            return v

    # Match by numbers
    nums = re.findall(r'-?\d+', outcome_name)
    if nums:
        for k, v in probs.items():
            if re.findall(r'-?\d+', k) == nums:
                # Also check same tail direction
                is_tail_out = bool(re.search(r'or\s+(higher|below|lower)', outcome_name, re.I))
                is_tail_k = bool(re.search(r'or\s+(higher|below|lower)', k, re.I))
                if is_tail_out == is_tail_k:
                    return v

    return None


def extract_bucket_temp(name: str, is_fahrenheit: bool) -> Optional[float]:
    """Extract center temperature from bucket name."""
    m = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F', name)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2.0
    m = re.search(r'(-?\d+)\s*°\s*C', name)
    if m:
        return float(m.group(1))
    m = re.search(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*below', name, re.I)
    if m:
        return float(m.group(1)) - (1 if is_fahrenheit else 0.5)
    m = re.search(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*higher', name, re.I)
    if m:
        return float(m.group(1)) + (1 if is_fahrenheit else 0.5)
    return None
