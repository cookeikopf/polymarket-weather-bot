"""
Polymarket Weather Market Scanner
==================================
Discovers and monitors active weather prediction markets.
Parses market structure (outcomes, prices, resolution criteria).

Uses slug-based discovery: constructs known weather market slugs
and queries the Gamma API directly for each city/date combination.
"""

import numpy as np
import requests
import re
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("weather_bot")

import config


@dataclass
class MarketOutcome:
    """Single outcome in a multi-outcome weather market."""
    token_id: str       # YES token ID
    name: str
    price: float        # Current market price (0-1)
    no_token_id: str = ""  # NO token ID (for BUY_NO trades)
    low_bound: Optional[float] = None
    high_bound: Optional[float] = None


@dataclass
class WeatherMarket:
    """Parsed weather market from Polymarket."""
    event_id: str
    condition_id: str
    slug: str
    question: str
    description: str
    market_type: str  # "temperature_max", "precipitation", etc.
    station_id: str  # "NYC", "London", etc.
    target_date: str  # YYYY-MM-DD
    resolution_source: str
    end_date: str
    volume: float
    liquidity: float
    outcomes: List[MarketOutcome] = field(default_factory=list)
    neg_risk: bool = False
    minimum_tick_size: str = "0.001"  # Default for weather markets
    order_min_size: float = 5.0       # Minimum order size in shares
    unit: str = "°F"


class MarketScanner:
    """Discovers and parses Polymarket weather markets via slug-based lookup."""

    def __init__(self):
        self.gamma_api = config.GAMMA_API_HOST
        self.active_markets: List[WeatherMarket] = []

    def scan_weather_markets(self) -> List[WeatherMarket]:
        """Scan Polymarket for active weather markets using slug-based discovery."""
        markets = []

        today = datetime.now()
        days_ahead = getattr(config, "MARKET_SCAN_DAYS_AHEAD", 7)

        for station_id, station in config.STATIONS.items():
            slug_name = station.get("slug_name")
            if not slug_name:
                continue

            for day_offset in range(1, days_ahead + 1):
                target = today + timedelta(days=day_offset)
                month_name = target.strftime("%B").lower()  # e.g. "march"
                day = target.day
                year = target.year
                target_date = target.strftime("%Y-%m-%d")

                # Temperature market slug
                temp_slug = f"highest-temperature-in-{slug_name}-on-{month_name}-{day}-{year}"
                market = self._fetch_event_by_slug(temp_slug, station_id, target_date, "temperature_max")
                if market:
                    markets.append(market)

                # Precipitation market slug (only some cities have these)
                precip_slug = f"precipitation-in-{slug_name}-on-{month_name}-{day}-{year}"
                precip_market = self._fetch_event_by_slug(precip_slug, station_id, target_date, "precipitation")
                if precip_market:
                    markets.append(precip_market)

        # Deduplicate by condition_id
        seen = set()
        unique_markets = []
        for m in markets:
            key = m.condition_id or m.slug
            if key not in seen:
                seen.add(key)
                unique_markets.append(m)

        self.active_markets = unique_markets
        return unique_markets

    def _fetch_event_by_slug(
        self,
        slug: str,
        station_id: str,
        target_date: str,
        market_type: str,
    ) -> Optional[WeatherMarket]:
        """Fetch a single event from Gamma API by its slug."""
        try:
            resp = requests.get(
                f"{self.gamma_api}/events",
                params={"slug": slug},
                timeout=15,
            )
            if resp.status_code != 200:
                return None

            events = resp.json()
            if not events:
                return None

            # Take the first (should be only) matching event
            event = events[0] if isinstance(events, list) else events

            return self._parse_weather_event(event, station_id, target_date, market_type)

        except Exception as e:
            # Silently skip — most slug combinations won't have active markets
            return None

    def _parse_weather_event(
        self,
        event: dict,
        station_id: str,
        target_date: str,
        market_type: str,
    ) -> Optional[WeatherMarket]:
        """Parse a Polymarket event into a WeatherMarket."""
        sub_markets = event.get("markets", [])
        if not sub_markets:
            return None

        station = config.STATIONS.get(station_id, {})
        unit = "°F" if station.get("unit") == "fahrenheit" else "°C"

        # Parse first sub-market for common trading details
        first_market = sub_markets[0]

        # Parse outcomes from all sub-markets
        outcomes = self._parse_outcomes(sub_markets, market_type, unit)

        # Get tick size and min order size from API
        tick_size = str(first_market.get("orderPriceMinTickSize") or first_market.get("minimum_tick_size") or "0.001")
        order_min_size = float(first_market.get("orderMinSize", 5))

        return WeatherMarket(
            event_id=str(event.get("id", "")),
            condition_id=first_market.get("conditionId", ""),
            slug=event.get("slug", ""),
            question=event.get("title", ""),
            description=event.get("description", ""),
            market_type=market_type,
            station_id=station_id,
            target_date=target_date,
            resolution_source=first_market.get("resolutionSource", ""),
            end_date=first_market.get("endDate", ""),
            volume=float(event.get("volume", 0)),
            liquidity=float(event.get("liquidity", 0)),
            outcomes=outcomes,
            neg_risk=first_market.get("negRisk", False),
            minimum_tick_size=tick_size,
            order_min_size=order_min_size,
            unit=unit,
        )

    def _parse_outcomes(self, sub_markets: list, market_type: str, unit: str) -> List[MarketOutcome]:
        """Parse multi-outcome market into structured outcomes."""
        outcomes = []

        for sm in sub_markets:
            try:
                token_ids = json.loads(sm.get("clobTokenIds", "[]"))
                outcome_names = json.loads(sm.get("outcomes", "[]"))
                outcome_prices = json.loads(sm.get("outcomePrices", "[]"))

                # For neg-risk multi-outcome markets, each sub-market is a separate outcome
                question = sm.get("question", "")

                if "Yes" in outcome_names and len(token_ids) >= 2:
                    # Extract the specific outcome from the question
                    outcome_name = self._extract_outcome_name(question, market_type, unit)
                    low, high = self._parse_bucket_bounds(outcome_name, unit)

                    yes_price = float(outcome_prices[0]) if outcome_prices else 0.5

                    outcomes.append(MarketOutcome(
                        token_id=token_ids[0],    # YES token
                        name=outcome_name,
                        price=yes_price,
                        no_token_id=token_ids[1],  # NO token
                        low_bound=low,
                        high_bound=high,
                    ))
            except (json.JSONDecodeError, ValueError, IndexError):
                continue

        return outcomes

    def _extract_outcome_name(self, question: str, market_type: str, unit: str) -> str:
        """Extract the specific outcome (bucket label) from a sub-market question."""
        # °F range pattern: "between 34-35°F" or "be 34-35°F"
        match = re.search(r'(?:between\s+|be\s+)?(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*F', question)
        if match:
            return f"{match.group(1)}-{match.group(2)}°F"

        # °C single-degree pattern: "be 14°C" or "be -5°C"
        match = re.search(r'(?:be\s+)(-?\d+)\s*°\s*C(?:\s+on\b)', question)
        if match:
            return f"{match.group(1)}°C"

        # "or higher" pattern: "42°F or higher" / "22°C or higher"
        match = re.search(r'(-?\d+)\s*°\s*([FC])\s+or\s+higher', question)
        if match:
            return f"{match.group(1)}°{match.group(2)} or higher"

        # "or below" / "or lower" pattern
        match = re.search(r'(-?\d+)\s*°\s*([FC])\s+or\s+(?:below|lower)', question)
        if match:
            return f"{match.group(1)}°{match.group(2)} or below"

        # Fallback: try broader patterns
        # °F range anywhere
        match = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*F', question)
        if match:
            return f"{match.group(1)}-{match.group(2)}°F"

        # °C single-degree anywhere
        match = re.search(r'(-?\d+)\s*°\s*C', question)
        if match:
            return f"{match.group(1)}°C"

        # Generic number extraction
        match = re.search(r'(\d+[-–]\d+\s*°?[FC]?|\d+\s*°?[FC]?\s+or\s+(?:higher|lower|below|above))', question)
        if match:
            return match.group(1)

        return question

    def _parse_bucket_bounds(self, name: str, unit: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse bucket boundaries from outcome name."""
        # °F range: "34-35°F" -> (34, 36)
        match = re.match(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F?', name)
        if match:
            return float(match.group(1)), float(match.group(2)) + 1

        # °C single degree: "14°C" -> (14, 15)
        match = re.match(r'(-?\d+)\s*°\s*C$', name)
        if match:
            val = float(match.group(1))
            return val, val + 1

        # "X°F or higher" / "X°C or higher"
        match = re.match(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*higher', name, re.I)
        if match:
            return float(match.group(1)), 200.0

        # "X°F or below" / "X°C or below"
        match = re.match(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*(?:below|lower)', name, re.I)
        if match:
            return -200.0, float(match.group(1)) + 1

        return None, None

    def get_market_prices(self, market: WeatherMarket) -> Dict[str, float]:
        """Fetch current orderbook prices for a market."""
        prices = {}
        for outcome in market.outcomes:
            prices[outcome.name] = outcome.price
        return prices

    def enrich_with_live_prices(self, markets: List[WeatherMarket]) -> List[WeatherMarket]:
        """Replace stale Gamma API prices with live CLOB orderbook prices.

        The Gamma API `outcomePrices` field can return cached/stale prices.
        This method fetches the real effective price from the CLOB /price
        endpoint for each outcome.

        CRITICAL: Must be called BEFORE edge detection to avoid phantom trades
        at prices that don't exist in the orderbook.
        """
        enriched = 0
        skipped = 0

        for market in markets:
            for outcome in market.outcomes:
                # Skip simulated tokens (backtesting)
                if outcome.token_id.startswith("sim_"):
                    continue

                # Fetch live orderbook for the YES token
                depth = self.fetch_orderbook_depth(outcome.token_id)

                if not depth["has_liquidity"]:
                    # No liquidity — mark price as -1 so edge detector skips it
                    outcome.price = -1.0
                    skipped += 1
                    continue

                # Use mid-price between effective bid and ask as the live price
                live_mid = (depth["best_bid"] + depth["best_ask"]) / 2.0

                # Sanity check: price must be between 0.01 and 0.99
                if 0.01 <= live_mid <= 0.99:
                    old_price = outcome.price
                    outcome.price = round(live_mid, 4)
                    if abs(old_price - live_mid) > 0.03:
                        logger.info(f"PRICE FIX: {outcome.name} | "
                              f"Gamma: {old_price:.3f} -> CLOB: {live_mid:.3f} "
                              f"(delta: {abs(old_price - live_mid):.3f})")
                    enriched += 1
                else:
                    skipped += 1

                # Store CLOB bid/ask for spread-aware edge calculation
                outcome._clob_spread = depth["spread"]
                outcome._clob_bid = depth["best_bid"]
                outcome._clob_ask = depth["best_ask"]

                # Rate limit: avoid CLOB API burst
                time.sleep(0.1)

        logger.info(f"CLOB price enrichment: {enriched} updated, {skipped} skipped (no liquidity)")
        return markets

    def fetch_orderbook(self, token_id: str) -> Dict:
        """Fetch orderbook for a specific token from CLOB API."""
        try:
            resp = requests.get(
                f"{config.POLYMARKET_HOST}/book",
                params={"token_id": token_id},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {"bids": [], "asks": []}

    def fetch_orderbook_depth(self, token_id: str) -> Dict:
        """Fetch orderbook and compute depth metrics for liquidity checks.

        For neg-risk tokens, the raw orderbook has unusual structure
        (bids near 0, asks near 1), so we also check the CLOB price
        endpoints which give effective market prices.
        """
        book = self.fetch_orderbook(token_id)
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        bid_depth = sum(float(b.get("size", 0)) for b in bids)
        ask_depth = sum(float(a.get("size", 0)) for a in asks)
        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 1
        spread = best_ask - best_bid if bids and asks else 1.0

        # For neg-risk markets, also check effective CLOB prices
        effective_bid = best_bid
        effective_ask = best_ask
        try:
            buy_resp = requests.get(
                f"{config.POLYMARKET_HOST}/price",
                params={"token_id": token_id, "side": "BUY"},
                timeout=5,
            )
            sell_resp = requests.get(
                f"{config.POLYMARKET_HOST}/price",
                params={"token_id": token_id, "side": "SELL"},
                timeout=5,
            )
            if buy_resp.status_code == 200 and sell_resp.status_code == 200:
                effective_bid = float(buy_resp.json().get("price", best_bid))
                effective_ask = float(sell_resp.json().get("price", best_ask))
        except Exception:
            pass

        effective_spread = abs(effective_ask - effective_bid) if effective_ask != effective_bid else spread

        # For neg-risk markets, raw book always has bids/asks (at 0.001/0.999)
        # so checking raw depth is useless. Instead check:
        # 1. /price endpoint returned real values (not still the raw defaults)
        # 2. Effective spread is reasonable (<50% = real market interest)
        price_endpoint_worked = (
            effective_bid != best_bid or effective_ask != best_ask
        )
        has_real_liquidity = (
            price_endpoint_worked
            and effective_spread < 0.50
            and effective_bid > 0.001
            and effective_ask < 0.999
        )

        return {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "best_bid": effective_bid,
            "best_ask": effective_ask,
            "spread": effective_spread,
            "has_liquidity": has_real_liquidity,
        }


def create_simulated_market(
    target_date: str,
    station_id: str = "NYC",
    actual_temp: Optional[float] = None,
    market_noise: float = None,
) -> WeatherMarket:
    """
    Create a simulated weather market for backtesting.
    Generates realistic market prices based on true probability + noise.
    """
    if market_noise is None:
        market_noise = config.SIM_MARKET_NOISE

    station = config.STATIONS.get(station_id, {})
    is_fahrenheit = station.get("unit", "fahrenheit") == "fahrenheit"
    unit = "°F" if is_fahrenheit else "°C"

    # Generate bucket edges based on season
    month = int(target_date.split("-")[1])
    if is_fahrenheit:
        if month in [12, 1, 2]:
            center = 38
        elif month in [3, 4, 5]:
            center = 55
        elif month in [6, 7, 8]:
            center = 82
        else:
            center = 60

        bucket_start = center - 12
        bucket_end = center + 14
        step = config.TEMP_BUCKET_SIZE_F  # 2
    else:
        if month in [12, 1, 2]:
            center = 7
        elif month in [3, 4, 5]:
            center = 14
        elif month in [6, 7, 8]:
            center = 23
        else:
            center = 14

        bucket_start = center - 6
        bucket_end = center + 8
        step = 1  # °C uses single-degree buckets

    edges = list(range(bucket_start, bucket_end, step))

    # Generate "true" probability distribution
    if actual_temp is not None:
        # Centered on actual with small noise (market participants have good but not perfect info)
        true_dist = {}
        for i in range(len(edges) - 1):
            low = edges[i]
            high = edges[i + 1]
            # Probability based on distance from actual
            mid = (low + high) / 2
            prob = np.exp(-0.5 * ((mid - actual_temp) / 2.5) ** 2)
            if is_fahrenheit:
                label = f"{int(low)}-{int(high-1)}{unit}"
            else:
                label = f"{int(low)}{unit}"
            true_dist[label] = prob

        # Add tails
        label_low = f"{int(edges[0])}{unit} or below"
        true_dist[label_low] = np.exp(-0.5 * ((edges[0] - 2 - actual_temp) / 2.5) ** 2)
        label_high = f"{int(edges[-1])}{unit} or higher"
        true_dist[label_high] = np.exp(-0.5 * ((edges[-1] + 2 - actual_temp) / 2.5) ** 2)

        total = sum(true_dist.values())
        true_dist = {k: v / total for k, v in true_dist.items()}
    else:
        from scipy import stats as sp_stats
        # Uniform-ish with slight center bias
        n_buckets = len(edges) - 1 + 2  # + tails
        true_dist = {f"bucket_{i}": 1.0 / n_buckets for i in range(n_buckets)}

    # Add market noise
    outcomes = []
    for name, true_prob in true_dist.items():
        noise = np.random.normal(0, market_noise)
        market_price = np.clip(true_prob + noise, 0.01, 0.99)
        outcomes.append(MarketOutcome(
            token_id=f"sim_{name}",
            name=name,
            price=market_price,
            low_bound=None,
            high_bound=None,
        ))

    return WeatherMarket(
        event_id="sim",
        condition_id="sim",
        slug=f"sim-weather-{target_date}",
        question=f"Simulated: Highest temperature on {target_date}?",
        description="Simulated market for backtesting",
        market_type="temperature_max",
        station_id=station_id,
        target_date=target_date,
        resolution_source="simulation",
        end_date=target_date + "T23:59:00Z",
        volume=5000,
        liquidity=2000,
        outcomes=outcomes,
        unit=unit,
    )
