"""
Live Trading Module
====================
Handles real order execution on Polymarket CLOB.
Supports both paper trading (simulation) and live trading.

Improvements:
- Smart entry/exit with take-profit, edge decay, gradual position reduction
- Orderbook liquidity checks before placing orders
- Paper trading P&L tracking with disk persistence
- Position correlation management across climate zones
"""

import os
import json
import time
import logging
import signal as sig
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

import config
from weather_engine import WeatherEngine
from market_scanner import MarketScanner, WeatherMarket
from edge_detector import EdgeDetector, TradingSignal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("weather_bot")


@dataclass
class Position:
    """Active trading position."""
    market_slug: str
    outcome_name: str
    token_id: str
    direction: str
    entry_price: float
    shares: float
    size_usd: float
    edge_at_entry: float
    confidence: float
    entry_time: str
    resolution_time: str
    station_id: str = ""
    current_edge: float = 0.0
    peak_unrealized_pnl: float = 0.0
    status: str = "open"  # open, closed, resolved


@dataclass
class TradeRecord:
    """Completed trade record for P&L tracking."""
    market_slug: str
    outcome_name: str
    station_id: str
    direction: str
    entry_price: float
    exit_price: float
    shares: float
    size_usd: float
    pnl: float
    edge_at_entry: float
    entry_time: str
    exit_time: str
    exit_reason: str  # "take_profit", "edge_decay", "time_exit", "resolution", "stop_loss"


class LiveTrader:
    """Live trading execution engine with smart entry/exit and P&L tracking."""

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.scanner = MarketScanner()
        self.engines = {}  # station_id -> WeatherEngine
        self.edge_detector = EdgeDetector()

        # Portfolio state
        self.bankroll = config.LIVE_BANKROLL if not paper_mode else config.BACKTEST_INITIAL_BANKROLL
        self.positions: List[Position] = []
        self.trade_history: List[TradeRecord] = []
        self.peak_bankroll = self.bankroll

        # CLOB client (initialized on first live trade)
        self.clob_client = None

        # Load saved state if available
        self._load_state()

        print(f"\n{'='*50}")
        print(f"  Weather Prediction Bot")
        print(f"  Mode: {'PAPER' if paper_mode else 'LIVE'}")
        print(f"  Bankroll: ${self.bankroll:.2f}")
        print(f"  Open Positions: {len(self.positions)}")
        print(f"  Total Trades: {len(self.trade_history)}")
        print(f"{'='*50}\n")

    def initialize(self):
        """Initialize engines and calibrate models."""
        for station_id in config.STATIONS:
            print(f"Initializing {station_id} engine...")
            engine = WeatherEngine(station_id)
            engine.calibrate()
            self.engines[station_id] = engine

        if not self.paper_mode:
            self._init_clob_client()

    def _init_clob_client(self):
        """Initialize Polymarket CLOB client for live trading.

        Authentication setup:
        - Uses PRIVATE_KEY exported from Polymarket (reveal.polymarket.com)
        - FUNDER_ADDRESS = your Polymarket proxy wallet (from profile URL)
        - SIGNATURE_TYPE: 0=EOA, 1=POLY_PROXY (email login), 2=GNOSIS_SAFE (browser wallet)

        Builder API (optional but recommended):
        - Gasless transactions (no POL needed for gas)
        - Order attribution → volume tracking → weekly USDC rewards
        - Get keys at: polymarket.com/settings?tab=builder
        """
        if not config.PRIVATE_KEY:
            raise ValueError("PRIVATE_KEY not set in .env file")

        try:
            from py_clob_client.client import ClobClient

            # Build init kwargs
            init_kwargs = {
                "host": config.POLYMARKET_HOST,
                "key": config.PRIVATE_KEY,
                "chain_id": config.CHAIN_ID,
            }

            # Add funder/signature type if configured
            if config.FUNDER_ADDRESS:
                init_kwargs["funder"] = config.FUNDER_ADDRESS
                init_kwargs["signature_type"] = config.SIGNATURE_TYPE

            # Add Builder API config if available (gasless + order attribution)
            builder_config = self._create_builder_config()
            if builder_config:
                init_kwargs["builder_config"] = builder_config
                print("  Builder API: ENABLED (gasless + attribution)")
            else:
                print("  Builder API: disabled (set POLY_BUILDER_API_KEY for gasless trading)")

            # Create client and derive API creds
            temp_client = ClobClient(**init_kwargs)
            temp_client.set_api_creds(temp_client.create_or_derive_api_creds())

            self.clob_client = temp_client
            print("  CLOB client initialized successfully")
            print(f"  Funder: {config.FUNDER_ADDRESS[:10]}..." if config.FUNDER_ADDRESS else "  Funder: (EOA mode)")
            print(f"  Order strategy: {config.ORDER_STRATEGY}")

        except Exception as e:
            print(f"  ERROR: Failed to initialize CLOB client: {e}")
            print("  Falling back to paper trading mode")
            self.paper_mode = True

    def _create_builder_config(self):
        """Create Builder API config if credentials are available."""
        if not config.BUILDER_API_KEY:
            return None

        try:
            from py_builder_signing_sdk.config import BuilderConfig
            from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds

            return BuilderConfig(
                local_builder_creds=BuilderApiKeyCreds(
                    key=config.BUILDER_API_KEY,
                    secret=config.BUILDER_SECRET,
                    passphrase=config.BUILDER_PASSPHRASE,
                )
            )
        except ImportError:
            print("  WARNING: py-builder-signing-sdk not installed, skipping Builder API")
            return None
        except Exception as e:
            print(f"  WARNING: Builder API config failed: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════
    # SCAN CYCLE
    # ═══════════════════════════════════════════════════════════════

    def run_scan_cycle(self) -> List[TradingSignal]:
        """Run one full scan cycle: discover markets, analyze, generate signals.

        TWO-PHASE approach to minimize CLOB API calls:
        Phase 1: Scan with Gamma prices (cheap) → find candidate edges
        Phase 2: Enrich candidates with CLOB prices (expensive) → verify & execute
        """

        # ─── PHASE 1: Scan & screen with Gamma prices ───
        logger.info("[1/5] Scanning for weather markets...")
        markets = self.scanner.scan_weather_markets()
        logger.info(f"  Found {len(markets)} active weather markets")

        if not markets:
            logger.info("  No weather markets found. Waiting...")
            return []

        # Screen for candidate edges using Gamma prices (cheap, no CLOB calls)
        logger.info("[2/5] Screening for candidate edges (Gamma prices)...")
        candidate_markets = []
        relaxed_threshold = max(0.02, config.MIN_EDGE_PCT - 0.02)  # Slightly relaxed

        skip_reasons = {"no_engine": 0, "no_forecasts": 0, "no_bucket_edges": 0,
                        "no_prob_match": 0, "edges_too_small": 0}

        for market in markets:
            engine = self.engines.get(market.station_id)
            if not engine:
                skip_reasons["no_engine"] += 1
                logger.debug(f"  SKIP {market.station_id}/{market.target_date}: no engine")
                continue

            forecasts = engine.fetch_multi_model_forecasts(market.target_date)
            if not forecasts:
                skip_reasons["no_forecasts"] += 1
                logger.warning(f"  SKIP {market.station_id}/{market.target_date}: no forecasts returned")
                continue

            ensemble_stats = engine.compute_ensemble_stats(forecasts)
            bucket_edges = self._market_to_bucket_edges(market)
            if not bucket_edges:
                skip_reasons["no_bucket_edges"] += 1
                logger.warning(f"  SKIP {market.station_id}/{market.target_date}: no bucket edges parsed from {len(market.outcomes)} outcomes")
                continue

            our_probs = engine.compute_probability_distribution(forecasts, bucket_edges)

            # Try ML model (better predictions if model file available)
            ml_probs = engine.compute_ml_probability_distribution(
                forecasts, ensemble_stats, bucket_edges, market.target_date
            )
            if ml_probs:
                our_probs = ml_probs
                logger.debug(f"  {market.station_id}/{market.target_date}: using ML probabilities")

            # ── BUG FIX: Validate outcome label matching (phantom prevention) ──
            # Build a validated probability dict that only includes entries
            # whose keys can be matched against REAL market outcomes.
            # This prevents phantom labels (e.g. our engine generating "63°F or below"
            # when the actual market only has "59°F or below").
            actual_outcome_names = {o.name for o in market.outcomes}
            validated_probs = {}
            phantom_labels = []
            for label, prob in our_probs.items():
                # Check: can ANY real market outcome match this label via _match_probability?
                # We need to check both directions:
                # 1) Does this label appear as a key that a market outcome can find?
                found_match = False
                for outcome in market.outcomes:
                    result = self.edge_detector._match_probability(outcome.name, {label: prob})
                    if result is not None:
                        found_match = True
                        break
                if found_match:
                    validated_probs[label] = prob
                else:
                    phantom_labels.append(label)
            if phantom_labels:
                logger.warning(f"  PHANTOM FILTER: Removed {len(phantom_labels)} phantom labels "
                              f"for {market.station_id}/{market.target_date}: {phantom_labels}")
                logger.debug(f"    Market outcome names: {list(actual_outcome_names)}")
            # Renormalize to sum to 1
            total_vp = sum(validated_probs.values())
            if total_vp > 0:
                validated_probs = {k: v / total_vp for k, v in validated_probs.items()}
            our_probs = validated_probs
            market._our_probs = our_probs

            # Quick screen: any outcome with potential edge?
            has_candidate = False
            best_edge_this_market = 0
            matched_count = 0
            for outcome in market.outcomes:
                our_prob = self.edge_detector._match_probability(outcome.name, our_probs)
                if our_prob is None:
                    continue
                matched_count += 1
                gamma_price = outcome.price
                if gamma_price <= 0 or gamma_price >= 1:
                    continue
                edge = abs(our_prob - gamma_price)
                best_edge_this_market = max(best_edge_this_market, edge)
                if edge >= relaxed_threshold:
                    has_candidate = True
                    break

            if matched_count == 0:
                skip_reasons["no_prob_match"] += 1
                logger.warning(f"  SKIP {market.station_id}/{market.target_date}: 0/{len(market.outcomes)} outcomes matched probability labels")
                logger.warning(f"    Outcome names: {[o.name for o in market.outcomes[:3]]}...")
                logger.warning(f"    Our prob keys: {list(our_probs.keys())[:3]}...")
            elif not has_candidate:
                skip_reasons["edges_too_small"] += 1
                logger.info(f"  {market.station_id}/{market.target_date}: best edge {best_edge_this_market:.3f} < threshold {relaxed_threshold} ({matched_count}/{len(market.outcomes)} matched)")

            if has_candidate:
                # Store forecast data for Phase 2
                market._forecasts = forecasts
                market._ensemble_stats = ensemble_stats
                market._our_probs = our_probs
                candidate_markets.append(market)

        logger.info(f"  {len(candidate_markets)}/{len(markets)} markets have candidate edges")
        if any(v > 0 for v in skip_reasons.values()):
            logger.info(f"  Skip reasons: {skip_reasons}")

        if not candidate_markets:
            logger.info("[5/5] No candidate edges found. Skipping CLOB enrichment.")
            return []

        # ─── PHASE 2: Enrich candidates with CLOB prices ───
        logger.info(f"[3/5] Enriching {len(candidate_markets)} candidate markets with CLOB prices...")
        candidate_markets = self.scanner.enrich_with_live_prices(candidate_markets)

        # Re-run edge detection with real CLOB prices
        logger.info("[4/5] Detecting edges with live CLOB prices...")
        all_signals = []
        current_exposure = sum(p.size_usd for p in self.positions)

        for market in candidate_markets:
            forecasts = market._forecasts
            ensemble_stats = market._ensemble_stats
            our_probs = market._our_probs

            logger.info(f"  Analyzing: {market.question}")
            logger.info(f"  Ensemble: mean={ensemble_stats['mean']:.1f} deg, "
                  f"spread={ensemble_stats['spread']:.1f} deg, "
                  f"agreement={ensemble_stats['agreement']:.1%}")

            signals = self.edge_detector.find_edges(
                market, our_probs, ensemble_stats, self.bankroll, current_exposure
            )

            if signals:
                logger.info(f"  Found {len(signals)} trading signals:")
                for s in signals[:5]:
                    logger.info(f"    {s.direction} '{s.outcome.name}' | "
                          f"Edge: {s.edge:.1%} | "
                          f"Size: ${s.suggested_size_usd:.2f} | "
                          f"EV: ${s.expected_value:.4f}/$ | "
                          f"Conf: {s.confidence:.1%}")

            all_signals.extend(signals)

        # ─── PHASE 3: Execute ───
        if all_signals:
            logger.info(f"[5/5] Executing top signals...")
            self._execute_signals(all_signals)
        else:
            logger.info("[5/5] No trades to execute")

        return all_signals

    def _market_to_bucket_edges(self, market: WeatherMarket) -> List[float]:
        """Convert market outcomes to bucket edges.

        Parses real Polymarket outcome names to extract the exact bucket boundaries.
        - °F markets: "32-33°F" → edges at 32, 34; "31°F or below" → lower tail at 32
        - °C markets: "14°C" → edges at 14, 15; "12°C or below" → lower tail at 13

        Returns sorted list of edges where:
        - Everything < edges[0] = lower tail ("X or below")
        - edges[i] to edges[i+1] = regular bucket
        - Everything >= edges[-1] = upper tail ("X or higher")
        """
        import re

        station = config.STATIONS.get(market.station_id, {})
        is_fahrenheit = station.get("unit", "fahrenheit") == "fahrenheit"

        range_lows = []   # lower bounds of regular range buckets
        tail_low = None    # boundary of "X or below" tail
        tail_high = None   # boundary of "X or higher" tail

        for outcome in market.outcomes:
            name = outcome.name

            # Check for tail labels first
            match_below = re.search(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*below', name, re.IGNORECASE)
            match_higher = re.search(r'(-?\d+)\s*°?\s*[FC]?\s*or\s*higher', name, re.IGNORECASE)

            if match_below:
                tail_low = int(match_below.group(1))
                continue
            if match_higher:
                tail_high = int(match_higher.group(1))
                continue

            # °F range: "32-33°F" -> low bound is 32
            match = re.search(r'(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F', name)
            if match:
                range_lows.append(int(match.group(1)))
                continue

            # °C single degree: "14°C" or "-5°C" -> that degree is the bucket
            match = re.search(r'(-?\d+)\s*°\s*C', name)
            if match:
                range_lows.append(int(match.group(1)))
                continue

            # Fallback: extract first integer
            nums = re.findall(r'-?\d+', name)
            if nums:
                range_lows.append(int(nums[0]))

        if not range_lows:
            return []

        range_lows.sort()
        step = 2 if is_fahrenheit else 1

        # Build edges: each range_low starts a bucket of width `step`
        # The edges array defines boundaries between buckets:
        # [range_lows[0], range_lows[0]+step, range_lows[1]+step, ...]
        # For °C: [13, 14, 15, 16, ...] (each bucket is 1°C)
        # For °F: [32, 34, 36, 38, ...] (each bucket is 2°F)
        edges = []
        for low in range_lows:
            edges.append(low)
        # Add the upper bound of the last range bucket
        edges.append(range_lows[-1] + step)

        # Deduplicate and sort
        edges = sorted(set(edges))
        return edges

    # ═══════════════════════════════════════════════════════════════
    # EXECUTION
    # ═══════════════════════════════════════════════════════════════

    def _execute_signals(self, signals: List[TradingSignal]):
        """Execute trading signals (paper or live) with correlation checks."""
        # Sort by risk-adjusted EV
        signals.sort(key=lambda s: s.expected_value * s.confidence, reverse=True)

        executed = 0
        for signal in signals:
            if executed >= 3:  # Max trades per cycle
                break

            if signal.suggested_size_usd < config.MIN_TRADE_SIZE_USDC:
                continue

            # Check position limits
            if len(self.positions) >= config.MAX_CONCURRENT_POSITIONS:
                print("  Max concurrent positions reached")
                break

            # ── BUG FIX: Prevent duplicate positions on same outcome ──
            # Check if we already hold a position on this market_slug + outcome_name + direction
            already_held = any(
                p.market_slug == signal.market.slug
                and p.outcome_name == signal.outcome.name
                and p.direction == signal.direction
                and p.status == "open"
                for p in self.positions
            )
            if already_held:
                logger.info(f"  DEDUP: Skipping '{signal.outcome.name}' on {signal.market.slug} — "
                            f"already have open {signal.direction} position")
                continue

            # Check climate zone correlation limits
            if not self._check_zone_capacity(signal.market.station_id):
                print(f"  Skipping {signal.market.station_id}: zone position limit reached")
                continue

            # Check orderbook liquidity (for live markets)
            if not self.paper_mode and signal.outcome.token_id and not signal.outcome.token_id.startswith("sim_"):
                depth = self.scanner.fetch_orderbook_depth(signal.outcome.token_id)
                if not depth["has_liquidity"]:
                    print(f"  Skipping '{signal.outcome.name}': insufficient liquidity")
                    continue
                # Adjust size if liquidity is thin
                available = depth["ask_depth"] if signal.direction == "BUY_YES" else depth["bid_depth"]
                if available > 0 and signal.suggested_size_usd > available * 0.5:
                    signal.suggested_size_usd = min(signal.suggested_size_usd, available * 0.5)
                    print(f"  Reduced size to ${signal.suggested_size_usd:.2f} due to thin liquidity")

            if self.paper_mode:
                self._paper_execute(signal)
            else:
                self._live_execute(signal)

            executed += 1

    def _check_zone_capacity(self, station_id: str) -> bool:
        """Check if we can open another position in the same climate zone."""
        max_per_zone = getattr(config, "MAX_POSITIONS_PER_ZONE", 3)
        zones = getattr(config, "CLIMATE_ZONES", {})

        # Find which zone this station belongs to
        station_zone = None
        for zone_name, stations in zones.items():
            if station_id in stations:
                station_zone = zone_name
                break

        if station_zone is None:
            return True  # Unknown zone, allow

        # Count open positions in the same zone
        zone_stations = set(zones.get(station_zone, []))
        zone_positions = sum(1 for p in self.positions if p.station_id in zone_stations)

        return zone_positions < max_per_zone

    def _paper_execute(self, signal: TradingSignal):
        """Execute a paper trade.

        Uses spread-aware CLOB prices for realistic paper execution:
        - BUY_YES: entry at _clob_ask (what we'd actually pay for YES shares)
        - BUY_NO: entry at 1 - _clob_bid (what we'd actually pay for NO shares)
        Falls back to mid-price if CLOB data not available (backtesting).
        """
        clob_ask = getattr(signal.outcome, '_clob_ask', None)
        clob_bid = getattr(signal.outcome, '_clob_bid', None)

        if signal.direction == "BUY_YES":
            # For BUY_YES: we pay the ask price
            entry_price = clob_ask if clob_ask else signal.market_price
            token_id = signal.outcome.token_id
        else:
            # For BUY_NO: we pay 1 - bid price for the NO share
            entry_price = (1.0 - clob_bid) if clob_bid else (1.0 - signal.market_price)
            token_id = signal.outcome.no_token_id or signal.outcome.token_id

        if entry_price <= 0 or entry_price >= 1:
            print(f"  [PAPER] Skipping '{signal.outcome.name}': invalid entry_price={entry_price:.3f}")
            return

        shares = signal.suggested_size_usd / entry_price

        position = Position(
            market_slug=signal.market.slug,
            outcome_name=signal.outcome.name,
            token_id=token_id,
            direction=signal.direction,
            entry_price=entry_price,
            shares=shares,
            size_usd=signal.suggested_size_usd,
            edge_at_entry=signal.edge,
            confidence=signal.confidence,
            entry_time=datetime.now().isoformat(),
            resolution_time=signal.market.end_date,
            station_id=signal.market.station_id,
            current_edge=signal.edge,
        )
        self.positions.append(position)
        self._save_state()

        print(f"  [PAPER] {signal.direction} '{signal.outcome.name}' "
              f"@ {entry_price:.3f} (YES={signal.market_price:.3f}) | "
              f"Size: ${signal.suggested_size_usd:.2f} | "
              f"Edge: {signal.edge:.1%}")

    def _live_execute(self, signal: TradingSignal):
        """Execute a live trade on Polymarket CLOB.

        Supports three order strategies:
        - 'taker':    FOK/FAK market orders for instant fill
        - 'maker':    Post-only limit orders (earns liquidity rewards)
        - 'adaptive': Maker for small edges, taker for large edges (>15%)

        Key details for neg-risk weather markets:
        - Must pass PartialCreateOrderOptions(neg_risk=True, tick_size=...)
        - BUY_YES: buy the YES token at market price
        - BUY_NO: buy the NO token at (1 - market_price)
        """
        if not self.clob_client:
            print("  ERROR: CLOB client not initialized")
            return

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
            from py_clob_client.order_builder.constants import BUY, SELL

            # Determine token, side, and price based on direction
            if signal.direction == "BUY_YES":
                token_id = signal.outcome.token_id  # YES token
                side = BUY
            else:
                # BUY_NO: buy the NO token directly
                token_id = signal.outcome.no_token_id  # NO token
                if not token_id:
                    print(f"  ERROR: No NO token ID for '{signal.outcome.name}'")
                    return
                side = BUY

            # CRITICAL: Fetch LIVE orderbook price right before execution
            # to prevent trading at stale/phantom prices
            live_depth = self.scanner.fetch_orderbook_depth(token_id)
            if not live_depth["has_liquidity"]:
                print(f"  Skipping '{signal.outcome.name}': no liquidity in orderbook")
                return

            # Use best_ask for BUY orders (that's the price we'd actually pay)
            live_ask = live_depth["best_ask"]
            live_bid = live_depth["best_bid"]
            base_price = live_ask  # We're buying, so we pay the ask

            # Verify the live price hasn't diverged too far from signal price
            if signal.direction == "BUY_YES":
                expected_price = signal.market_price
            else:
                expected_price = 1.0 - signal.market_price

            price_divergence = abs(base_price - expected_price)
            if price_divergence > 0.10:  # >10 cent divergence = stale signal
                print(f"  PRICE MISMATCH: '{signal.outcome.name}' "
                      f"signal={expected_price:.3f} vs orderbook={base_price:.3f} "
                      f"(divergence={price_divergence:.3f}). Skipping.")
                return
            elif price_divergence > 0.03:
                print(f"  PRICE DRIFT: '{signal.outcome.name}' "
                      f"signal={expected_price:.3f} vs orderbook={base_price:.3f} "
                      f"(drift={price_divergence:.3f}). Proceeding with live price.")

            # Determine tick_size from market data
            tick_size = signal.market.minimum_tick_size or "0.001"
            tick = float(tick_size)

            # Choose order strategy
            strategy = config.ORDER_STRATEGY
            if strategy == "adaptive":
                strategy = "taker" if signal.edge > config.TAKER_EDGE_THRESHOLD else "maker"

            # Adjust price based on strategy
            if strategy == "maker":
                # Post-only: place slightly inside the spread to be first in queue
                # but DON'T cross the spread (or the order gets rejected)
                price = base_price - config.MAKER_PRICE_OFFSET
                order_type_label = "MAKER (post-only)"
            else:
                # Taker: match the current best price for instant fill
                price = base_price
                order_type_label = "TAKER (market)"

            # Round price to tick_size
            price = round(round(price / tick) * tick, 4)
            price = max(tick, min(price, 1.0 - tick))

            # Calculate number of shares
            size = signal.suggested_size_usd / price

            # Enforce minimum order size
            min_size = getattr(signal.market, 'order_min_size', 5.0)
            if size < min_size:
                print(f"  Skipping: order size {size:.1f} shares < minimum {min_size}")
                return

            size = round(size, 2)

            order_args = OrderArgs(
                price=price,
                size=size,
                side=side,
                token_id=token_id,
            )

            # CRITICAL: pass neg_risk and tick_size for weather markets
            options = PartialCreateOrderOptions(
                neg_risk=signal.market.neg_risk if signal.market.neg_risk else False,
                tick_size=tick_size,
            )

            print(f"  [LIVE] {order_type_label}: {signal.direction} '{signal.outcome.name}' "
                  f"@ {price:.4f} x {size:.2f} shares (${signal.suggested_size_usd:.2f})")
            print(f"         edge={signal.edge:.1%}, neg_risk={options.neg_risk}, tick={tick_size}")

            # Execute based on strategy
            signed_order = self.clob_client.create_order(order_args, options)

            if strategy == "maker":
                # Post-only: rejected if it would cross the spread (good - prevents bad fills)
                resp = self.clob_client.post_order(signed_order, OrderType.GTC, post_only=True)
            else:
                # Taker: GTC limit at market price (effectively a market order)
                resp = self.clob_client.post_order(signed_order, OrderType.GTC)

            print(f"  [LIVE] Response: {resp}")

            if resp.get("success"):
                position = Position(
                    market_slug=signal.market.slug,
                    outcome_name=signal.outcome.name,
                    token_id=token_id,
                    direction=signal.direction,
                    entry_price=price,
                    shares=size,
                    size_usd=signal.suggested_size_usd,
                    edge_at_entry=signal.edge,
                    confidence=signal.confidence,
                    entry_time=datetime.now().isoformat(),
                    resolution_time=signal.market.end_date,
                    station_id=signal.market.station_id,
                    current_edge=signal.edge,
                )
                self.positions.append(position)
                self._save_state()
                print(f"  [LIVE] Position opened ({order_type_label})")
            else:
                error_msg = resp.get("errorMsg", "Unknown error")
                print(f"  [LIVE] Order rejected: {error_msg}")

                # If maker order rejected, fall back to taker
                if strategy == "maker" and "post only" in str(error_msg).lower():
                    print(f"  [LIVE] Maker rejected (would cross spread), retrying as taker...")
                    resp = self.clob_client.post_order(signed_order, OrderType.GTC)
                    if resp.get("success"):
                        position = Position(
                            market_slug=signal.market.slug,
                            outcome_name=signal.outcome.name,
                            token_id=token_id,
                            direction=signal.direction,
                            entry_price=price,
                            shares=size,
                            size_usd=signal.suggested_size_usd,
                            edge_at_entry=signal.edge,
                            confidence=signal.confidence,
                            entry_time=datetime.now().isoformat(),
                            resolution_time=signal.market.end_date,
                            station_id=signal.market.station_id,
                            current_edge=signal.edge,
                        )
                        self.positions.append(position)
                        self._save_state()
                        print(f"  [LIVE] Position opened (fallback taker)")

        except Exception as e:
            import traceback
            print(f"  ERROR: Live execution failed: {e}")
            traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════
    # SMART EXIT STRATEGY
    # ═══════════════════════════════════════════════════════════════

    def check_positions(self):
        """Check and manage existing positions with smart exit logic."""
        for pos in self.positions[:]:
            try:
                resolution_time = datetime.fromisoformat(pos.resolution_time.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                # Can't parse resolution time, use a default far future
                resolution_time = datetime.now() + timedelta(days=7)
                if resolution_time.tzinfo is None:
                    from datetime import timezone
                    resolution_time = resolution_time.replace(tzinfo=timezone.utc)

            now = datetime.now(resolution_time.tzinfo) if resolution_time.tzinfo else datetime.now()
            hours_to_resolution = (resolution_time - now).total_seconds() / 3600

            # --- Take Profit: if market price matches our probability (edge -> 0) ---
            if self._check_take_profit(pos):
                self._exit_position(pos, "take_profit")
                continue

            # --- Edge Decay: re-evaluate if new forecasts change our edge ---
            if self._check_edge_decay(pos):
                self._exit_position(pos, "edge_decay")
                continue

            # --- Smart Time-Based Exit ---
            # Gradual position reduction instead of hard 2h cutoff
            if hours_to_resolution < 6:
                if hours_to_resolution < 2:
                    # Hard exit unless deep in profit
                    unrealized_pnl = self._estimate_unrealized_pnl(pos)
                    if unrealized_pnl > pos.size_usd * 0.3:
                        # Deep in profit - let it ride to resolution
                        print(f"  Letting {pos.outcome_name} ride (profit: ${unrealized_pnl:.2f})")
                    else:
                        print(f"  Exiting {pos.outcome_name} - 2h to resolution")
                        self._exit_position(pos, "time_exit")
                elif hours_to_resolution < 6:
                    # Reduce 50% of position at 6h mark
                    if pos.shares > 0 and not hasattr(pos, '_reduced'):
                        print(f"  Reducing {pos.outcome_name} by 50% - 6h to resolution")
                        self._reduce_position(pos, 0.5)

    def _check_take_profit(self, pos: Position) -> bool:
        """Check if market price has moved to match our probability (edge -> 0)."""
        take_profit_threshold = getattr(config, "TAKE_PROFIT_EDGE_PCT", 0.02)
        if abs(pos.current_edge) < take_profit_threshold and pos.current_edge < pos.edge_at_entry * 0.3:
            return True
        return False

    def _check_edge_decay(self, pos: Position) -> bool:
        """Check if our edge has significantly decayed based on new forecast data."""
        engine = self.engines.get(pos.station_id)
        if not engine:
            return False

        try:
            # Extract target_date from slug
            target_date = self._slug_to_target_date(pos.market_slug)
            if not target_date:
                return False

            forecasts = engine.fetch_multi_model_forecasts(target_date)
            if not forecasts:
                return False

            # Recompute our probability for this bucket
            station = config.STATIONS.get(pos.station_id, {})
            is_fahrenheit = station.get("unit", "fahrenheit") == "fahrenheit"
            ensemble_mean = sum(forecasts.values()) / len(forecasts)

            # Simple edge decay check: if forecast moved significantly away from entry assumption
            # More sophisticated: recompute full probability distribution
            # For now, flag if edge might have reversed
            if pos.edge_at_entry > 0.10 and abs(pos.current_edge) < 0.02:
                return True
        except Exception:
            pass

        return False

    def _estimate_unrealized_pnl(self, pos: Position) -> float:
        """Estimate unrealized P&L for a position."""
        # In paper mode, approximate based on entry edge
        # Positive edge at entry suggests the position should be profitable
        if pos.direction == "BUY_YES":
            estimated_current_price = pos.entry_price + pos.current_edge * 0.5
            return pos.shares * (estimated_current_price - pos.entry_price)
        else:
            estimated_current_price = pos.entry_price - pos.current_edge * 0.5
            return pos.shares * (pos.entry_price - estimated_current_price)

    def _reduce_position(self, pos: Position, fraction: float):
        """Reduce a position by a fraction (e.g., 0.5 = close half)."""
        exit_shares = pos.shares * fraction
        exit_size = pos.size_usd * fraction

        # Record partial exit
        pnl = self._estimate_unrealized_pnl(pos) * fraction
        self.trade_history.append(TradeRecord(
            market_slug=pos.market_slug,
            outcome_name=pos.outcome_name,
            station_id=pos.station_id,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=pos.entry_price,  # approximate
            shares=exit_shares,
            size_usd=exit_size,
            pnl=pnl,
            edge_at_entry=pos.edge_at_entry,
            entry_time=pos.entry_time,
            exit_time=datetime.now().isoformat(),
            exit_reason="partial_time_exit",
        ))

        # Reduce position
        pos.shares -= exit_shares
        pos.size_usd -= exit_size
        self.bankroll += exit_size + pnl

        self._save_state()

    def _exit_position(self, position: Position, reason: str = "time_exit"):
        """Exit a position (paper or live) and record trade."""
        pnl = self._estimate_unrealized_pnl(position)

        record = TradeRecord(
            market_slug=position.market_slug,
            outcome_name=position.outcome_name,
            station_id=position.station_id,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=position.entry_price,  # approximate for paper
            shares=position.shares,
            size_usd=position.size_usd,
            pnl=pnl,
            edge_at_entry=position.edge_at_entry,
            entry_time=position.entry_time,
            exit_time=datetime.now().isoformat(),
            exit_reason=reason,
        )
        self.trade_history.append(record)
        self.bankroll += position.size_usd + pnl
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)

        if position in self.positions:
            self.positions.remove(position)

        self._save_state()

        print(f"  [{'PAPER' if self.paper_mode else 'LIVE'}] Exited {position.outcome_name} "
              f"(reason: {reason}, P&L: ${pnl:+.2f})")

    def _slug_to_target_date(self, slug: str) -> Optional[str]:
        """Extract target date from a market slug."""
        import re
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        match = re.search(r'on-(\w+)-(\d+)-(\d{4})', slug)
        if match:
            month = months.get(match.group(1).lower())
            if month:
                return f"{match.group(3)}-{month:02d}-{int(match.group(2)):02d}"
        return None

    # ═══════════════════════════════════════════════════════════════
    # P&L TRACKING & PERSISTENCE
    # ═══════════════════════════════════════════════════════════════

    def _save_state(self):
        """Save positions and trade history to disk."""
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        state = {
            "timestamp": datetime.now().isoformat(),
            "bankroll": self.bankroll,
            "peak_bankroll": self.peak_bankroll,
            "positions": [asdict(p) for p in self.positions],
            "trade_history": [asdict(t) for t in self.trade_history],
        }
        try:
            with open(f"{config.RESULTS_DIR}/paper_trading_state.json", "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            print(f"  Warning: Failed to save state: {e}")

    def _load_state(self):
        """Load positions and trade history from disk."""
        state_path = f"{config.RESULTS_DIR}/paper_trading_state.json"
        if not os.path.exists(state_path):
            return

        try:
            with open(state_path, "r") as f:
                state = json.load(f)

            self.bankroll = state.get("bankroll", self.bankroll)
            self.peak_bankroll = state.get("peak_bankroll", self.peak_bankroll)

            self.positions = []
            for p in state.get("positions", []):
                self.positions.append(Position(**{
                    k: v for k, v in p.items()
                    if k in Position.__dataclass_fields__
                }))

            self.trade_history = []
            for t in state.get("trade_history", []):
                self.trade_history.append(TradeRecord(**{
                    k: v for k, v in t.items()
                    if k in TradeRecord.__dataclass_fields__
                }))

            print(f"  Loaded state: ${self.bankroll:.2f} bankroll, "
                  f"{len(self.positions)} positions, {len(self.trade_history)} trades")
        except Exception as e:
            print(f"  Warning: Failed to load state: {e}")

    def compute_pnl_report(self) -> Dict:
        """Compute running P&L and generate status report."""
        total_realized_pnl = sum(t.pnl for t in self.trade_history)
        total_unrealized_pnl = sum(self._estimate_unrealized_pnl(p) for p in self.positions)

        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl <= 0]

        win_rate = len(wins) / len(self.trade_history) * 100 if self.trade_history else 0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float("inf")

        current_drawdown = (self.peak_bankroll - self.bankroll) / self.peak_bankroll * 100 if self.peak_bankroll > 0 else 0

        # Per-station breakdown
        station_pnl = {}
        for t in self.trade_history:
            if t.station_id not in station_pnl:
                station_pnl[t.station_id] = {"pnl": 0, "trades": 0, "wins": 0}
            station_pnl[t.station_id]["pnl"] += t.pnl
            station_pnl[t.station_id]["trades"] += 1
            if t.pnl > 0:
                station_pnl[t.station_id]["wins"] += 1

        report = {
            "timestamp": datetime.now().isoformat(),
            "mode": "PAPER" if self.paper_mode else "LIVE",
            "bankroll": self.bankroll,
            "peak_bankroll": self.peak_bankroll,
            "current_drawdown_pct": current_drawdown,
            "total_realized_pnl": total_realized_pnl,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_pnl": total_realized_pnl + total_unrealized_pnl,
            "open_positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "station_breakdown": station_pnl,
        }

        return report

    def print_pnl_report(self):
        """Print a formatted P&L report."""
        report = self.compute_pnl_report()

        print(f"\n{'='*55}")
        print(f"  P&L Report ({report['mode']} Mode)")
        print(f"{'='*55}")
        print(f"  Bankroll:          ${report['bankroll']:.2f}")
        print(f"  Peak Bankroll:     ${report['peak_bankroll']:.2f}")
        print(f"  Drawdown:          {report['current_drawdown_pct']:.1f}%")
        print(f"  Realized P&L:      ${report['total_realized_pnl']:+.2f}")
        print(f"  Unrealized P&L:    ${report['total_unrealized_pnl']:+.2f}")
        print(f"  Total P&L:         ${report['total_pnl']:+.2f}")
        print(f"  Open Positions:    {report['open_positions']}")
        print(f"  Total Trades:      {report['total_trades']}")
        print(f"  Win Rate:          {report['win_rate']:.1f}%")
        print(f"  Avg Win:           ${report['avg_win']:+.2f}")
        print(f"  Avg Loss:          ${report['avg_loss']:+.2f}")
        print(f"  Profit Factor:     {report['profit_factor']:.2f}")

        if report['station_breakdown']:
            print(f"\n  Station Breakdown:")
            for station, stats in sorted(report['station_breakdown'].items(), key=lambda x: x[1]['pnl'], reverse=True):
                wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] else 0
                print(f"    {station:12s}: ${stats['pnl']:+8.2f} ({stats['trades']} trades, {wr:.0f}% win)")

        print(f"{'='*55}\n")

        # Save report to disk
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        with open(f"{config.RESULTS_DIR}/pnl_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════
    # CONTINUOUS OPERATION
    # ═══════════════════════════════════════════════════════════════

    def run_continuous(self, duration_minutes: int = 60):
        """Run the bot continuously."""
        logger.info(f"Starting continuous mode for {duration_minutes} minutes...")

        # Graceful shutdown handler
        def handle_shutdown(signum, frame):
            logger.info("Shutdown signal received, saving state...")
            self._save_state()
            self.print_pnl_report()
            raise SystemExit(0)

        sig.signal(sig.SIGTERM, handle_shutdown)
        sig.signal(sig.SIGINT, handle_shutdown)

        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        cycle = 0
        while datetime.now() < end_time:
            cycle += 1
            print(f"\n{'─'*40}")
            print(f"  Cycle {cycle} | {datetime.now().strftime('%H:%M:%S')}")
            print(f"  Bankroll: ${self.bankroll:.2f} | Positions: {len(self.positions)}")
            print(f"{'─'*40}")

            # Check drawdown halt
            if self.peak_bankroll > 0:
                dd = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
                if dd > config.MAX_DRAWDOWN_PCT:
                    print(f"  DRAWDOWN HALT: {dd:.1%} exceeds {config.MAX_DRAWDOWN_PCT:.0%} limit")
                    break

            try:
                self.run_scan_cycle()
                self.check_positions()
            except Exception as e:
                print(f"  Error in cycle: {e}")

            # Print report every 5 cycles
            if cycle % 5 == 0:
                self.print_pnl_report()

            # Wait for next scan
            sleep_time = min(config.SCAN_INTERVAL_SECONDS,
                           (end_time - datetime.now()).total_seconds())
            if sleep_time > 0:
                print(f"\n  Sleeping {sleep_time:.0f}s until next scan...")
                time.sleep(sleep_time)

        # Final report
        self.print_pnl_report()

    def get_status(self) -> Dict:
        """Get current bot status."""
        return {
            "mode": "PAPER" if self.paper_mode else "LIVE",
            "bankroll": self.bankroll,
            "positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "peak_bankroll": self.peak_bankroll,
            "current_drawdown": (self.peak_bankroll - self.bankroll) / self.peak_bankroll * 100 if self.peak_bankroll > 0 else 0,
        }
