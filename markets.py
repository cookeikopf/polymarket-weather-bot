"""
Polymarket Market Scanner & CLOB Integration V6
=================================================
Discovers weather markets via Gamma API (slug-based).
Fetches live CLOB orderbook prices.
"""

import re
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import config as cfg
from utils import log


@dataclass
class MarketOutcome:
    """Single outcome in a multi-outcome weather market."""
    token_id: str
    name: str
    price: float
    no_token_id: str = ""
    low_bound: Optional[float] = None
    high_bound: Optional[float] = None
    # Live CLOB data (set by enrich_with_live_prices)
    clob_bid: float = 0.0
    clob_ask: float = 0.0
    clob_spread: float = 1.0


@dataclass
class WeatherMarket:
    """Parsed weather market from Polymarket."""
    event_id: str
    condition_id: str
    slug: str
    question: str
    description: str
    market_type: str
    station_id: str
    target_date: str
    end_date: str
    volume: float
    liquidity: float
    outcomes: List[MarketOutcome] = field(default_factory=list)
    neg_risk: bool = False
    tick_size: str = "0.01"
    order_min_size: float = 5.0
    unit: str = "°F"


class MarketScanner:
    """Discovers and parses Polymarket weather markets via slug-based lookup."""

    def __init__(self):
        self.gamma_api = cfg.GAMMA_API_HOST

    def scan_weather_markets(self) -> List[WeatherMarket]:
        """Scan for active weather markets across all stations."""
        markets = []
        today = datetime.now()

        for station_id, station in cfg.STATIONS.items():
            slug_name = station.get("slug")
            if not slug_name:
                continue

            for day_offset in range(1, cfg.SCAN_DAYS_AHEAD + 1):
                target = today + timedelta(days=day_offset)
                month = target.strftime("%B").lower()
                day = target.day
                year = target.year
                target_date = target.strftime("%Y-%m-%d")

                # Temperature market
                slug = f"highest-temperature-in-{slug_name}-on-{month}-{day}-{year}"
                market = self._fetch_event_by_slug(slug, station_id, target_date, "temperature_max")
                if market:
                    markets.append(market)

        # Deduplicate
        seen = set()
        unique = []
        for m in markets:
            key = m.condition_id or m.slug
            if key not in seen:
                seen.add(key)
                unique.append(m)

        return unique

    def _fetch_event_by_slug(self, slug: str, station_id: str,
                              target_date: str, market_type: str) -> Optional[WeatherMarket]:
        """Fetch a single event from Gamma API by slug."""
        try:
            resp = requests.get(f"{self.gamma_api}/events",
                                params={"slug": slug}, timeout=15)
            if resp.status_code != 200:
                return None
            events = resp.json()
            if not events:
                return None
            event = events[0] if isinstance(events, list) else events
            return self._parse_event(event, station_id, target_date, market_type)
        except Exception:
            return None

    def _parse_event(self, event: dict, station_id: str,
                      target_date: str, market_type: str) -> Optional[WeatherMarket]:
        """Parse Gamma API event into WeatherMarket."""
        sub_markets = event.get("markets", [])
        if not sub_markets:
            return None

        station = cfg.STATIONS.get(station_id, {})
        unit = "°F" if station.get("unit") == "fahrenheit" else "°C"
        first = sub_markets[0]
        tick_size = str(first.get("orderPriceMinTickSize") or "0.01")
        order_min = float(first.get("orderMinSize", 5))

        outcomes = self._parse_outcomes(sub_markets, market_type, unit)

        return WeatherMarket(
            event_id=str(event.get("id", "")),
            condition_id=first.get("conditionId", ""),
            slug=event.get("slug", ""),
            question=event.get("title", ""),
            description=event.get("description", ""),
            market_type=market_type,
            station_id=station_id,
            target_date=target_date,
            end_date=first.get("endDate", ""),
            volume=float(event.get("volume", 0)),
            liquidity=float(event.get("liquidity", 0)),
            outcomes=outcomes,
            neg_risk=first.get("negRisk", False),
            tick_size=tick_size,
            order_min_size=order_min,
            unit=unit,
        )

    def _parse_outcomes(self, sub_markets: list, market_type: str, unit: str) -> List[MarketOutcome]:
        """Parse sub-markets into structured outcomes."""
        outcomes = []
        for sm in sub_markets:
            try:
                token_ids = json.loads(sm.get("clobTokenIds", "[]"))
                outcome_names = json.loads(sm.get("outcomes", "[]"))
                outcome_prices = json.loads(sm.get("outcomePrices", "[]"))
                question = sm.get("question", "")

                if "Yes" in outcome_names and len(token_ids) >= 2:
                    name = self._extract_outcome_name(question, unit)
                    yes_price = float(outcome_prices[0]) if outcome_prices else 0.5

                    outcomes.append(MarketOutcome(
                        token_id=token_ids[0],
                        name=name,
                        price=yes_price,
                        no_token_id=token_ids[1],
                    ))
            except (json.JSONDecodeError, ValueError, IndexError):
                continue
        return outcomes

    def _extract_outcome_name(self, question: str, unit: str) -> str:
        """Extract bucket label from sub-market question."""
        # °F range: "between 34-35°F"
        m = re.search(r'(?:between\s+|be\s+)?(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*F', question)
        if m:
            return f"{m.group(1)}-{m.group(2)}°F"
        # °C single: "be 14°C"
        m = re.search(r'(?:be\s+)(-?\d+)\s*°\s*C(?:\s+on\b)', question)
        if m:
            return f"{m.group(1)}°C"
        # "or higher"
        m = re.search(r'(-?\d+)\s*°\s*([FC])\s+or\s+higher', question)
        if m:
            return f"{m.group(1)}°{m.group(2)} or higher"
        # "or below"
        m = re.search(r'(-?\d+)\s*°\s*([FC])\s+or\s+(?:below|lower)', question)
        if m:
            return f"{m.group(1)}°{m.group(2)} or below"
        # Broad °F range
        m = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*F', question)
        if m:
            return f"{m.group(1)}-{m.group(2)}°F"
        # Broad °C
        m = re.search(r'(-?\d+)\s*°\s*C', question)
        if m:
            return f"{m.group(1)}°C"
        return question

    # ─── CLOB Price Enrichment ────────────────────────────────────

    def enrich_with_live_prices(self, markets: List[WeatherMarket]) -> List[WeatherMarket]:
        """Replace Gamma prices with live CLOB orderbook prices."""
        enriched = 0
        for market in markets:
            for outcome in market.outcomes:
                if outcome.token_id.startswith("sim_"):
                    continue
                depth = self.fetch_orderbook_depth(outcome.token_id)
                if not depth["has_liquidity"]:
                    outcome.price = -1.0
                    continue
                live_mid = (depth["best_bid"] + depth["best_ask"]) / 2.0
                if 0.01 <= live_mid <= 0.99:
                    outcome.price = round(live_mid, 4)
                    outcome.clob_bid = depth["best_bid"]
                    outcome.clob_ask = depth["best_ask"]
                    outcome.clob_spread = depth["spread"]
                    enriched += 1
                time.sleep(0.1)
        log.info(f"  CLOB enrichment: {enriched} outcomes updated")
        return markets

    def fetch_orderbook_depth(self, token_id: str) -> Dict:
        """Fetch orderbook depth and effective prices for a token."""
        book = self._fetch_raw_orderbook(token_id)
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 1
        bid_depth = sum(float(b.get("size", 0)) for b in bids)
        ask_depth = sum(float(a.get("size", 0)) for a in asks)

        # Try effective CLOB prices (for neg-risk markets)
        eff_bid, eff_ask = best_bid, best_ask
        try:
            buy_r = requests.get(f"{cfg.POLYMARKET_HOST}/price",
                                 params={"token_id": token_id, "side": "BUY"}, timeout=5)
            sell_r = requests.get(f"{cfg.POLYMARKET_HOST}/price",
                                  params={"token_id": token_id, "side": "SELL"}, timeout=5)
            if buy_r.status_code == 200 and sell_r.status_code == 200:
                eff_ask = float(buy_r.json().get("price", best_bid))
                eff_bid = float(sell_r.json().get("price", best_ask))
        except Exception:
            pass

        spread = abs(eff_ask - eff_bid)
        has_liq = (
            (eff_bid != best_bid or eff_ask != best_ask)
            and spread < 0.50
            and eff_bid > 0.001
            and eff_ask < 0.999
        )

        return {
            "best_bid": eff_bid, "best_ask": eff_ask,
            "bid_depth": bid_depth, "ask_depth": ask_depth,
            "spread": spread, "has_liquidity": has_liq,
        }

    def _fetch_raw_orderbook(self, token_id: str) -> dict:
        try:
            resp = requests.get(f"{cfg.POLYMARKET_HOST}/book",
                                params={"token_id": token_id}, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {"bids": [], "asks": []}


def parse_bucket_edges(market: WeatherMarket) -> List[float]:
    """Extract bucket edges from market outcomes for probability mapping."""
    is_f = market.unit == "°F"
    range_lows = []

    for o in market.outcomes:
        m = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F', o.name)
        if m:
            range_lows.append(int(m.group(1)))
            continue
        if re.search(r'or\s+(?:below|lower|higher)', o.name, re.I):
            continue
        m = re.search(r'(-?\d+)\s*°\s*C', o.name)
        if m:
            range_lows.append(int(m.group(1)))
            continue
        nums = re.findall(r'-?\d+', o.name)
        if nums:
            range_lows.append(int(nums[0]))

    if not range_lows:
        return []

    range_lows.sort()
    step = 2 if is_f else 1
    edges = sorted(set(range_lows + [range_lows[-1] + step]))
    return edges
