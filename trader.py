"""
Trade Execution & Position Management V6
==========================================
Handles live/paper order execution, position tracking, P&L, state persistence.
"""

import os
import json
import time
import signal as sig
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import config as cfg
from weather import WeatherEngine
from markets import MarketScanner, WeatherMarket, parse_bucket_edges
from strategy import EdgeDetector, TradingSignal, match_probability
from utils import log, utcnow, utcnow_iso
import requests


@dataclass
class Position:
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
    status: str = "open"


@dataclass
class Trade:
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
    exit_reason: str


class LiveTrader:
    """Live/paper trading engine."""

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.scanner = MarketScanner()
        self.engines: Dict[str, WeatherEngine] = {}
        self.edge_detector = EdgeDetector()

        self.bankroll = cfg.BANKROLL
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.peak_bankroll = self.bankroll

        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.trading_day = utcnow().strftime("%Y-%m-%d")
        self.daily_halt = False

        self.clob_client = None
        self._load_state()
        self._check_daily_reset()

        log.info(f"{'='*55}")
        log.info(f"  Weather Bot V6 | {'PAPER' if paper_mode else 'LIVE'}")
        log.info(f"  Bankroll: ${self.bankroll:.2f} | Positions: {len(self.positions)}")
        log.info(f"  Trades: {len(self.trades)} | Daily P&L: ${self.daily_pnl:+.2f}")
        log.info(f"{'='*55}")

    def initialize(self):
        """Initialize weather engines for all stations."""
        for sid in cfg.STATIONS:
            self.engines[sid] = WeatherEngine(sid)
            log.info(f"  Engine ready: {sid}")

        if not self.paper_mode:
            self._init_clob()

    # ─── CLOB client ──────────────────────────────────────────────

    def _check_geoblock(self) -> bool:
        """Check if IP is geoblocked by Polymarket."""
        if os.getenv("SKIP_GEOBLOCK_CHECK", "0") == "1":
            log.info("  Geoblock: SKIPPED (env override)")
            return True
        try:
            resp = requests.get("https://polymarket.com/api/geoblock", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                blocked = data.get("blocked", False)
                country = data.get("country", "?")
                ip = data.get("ip", "?")
                log.info(f"  Geoblock: blocked={blocked}, IP={ip}, country={country}")
                if blocked:
                    log.error(f"  GEOBLOCKED from {country}! Change VPS to allowed region.")
                    log.error(f"  Allowed: FI, SE, IE, ES, CA(excl ON), BR, JP, KR, etc.")
                    return False
                return True
        except Exception as e:
            log.warning(f"  Geoblock check failed: {e}")
        return True  # Proceed if check fails

    def _init_clob(self):
        """Initialize CLOB client for live trading."""
        if not cfg.PRIVATE_KEY:
            raise ValueError("POLYMARKET_PRIVATE_KEY not set")

        if not self._check_geoblock():
            log.warning("  Falling back to paper mode due to geoblocking")
            self.paper_mode = True
            return

        try:
            from py_clob_client.client import ClobClient
            kwargs = {
                "host": cfg.POLYMARKET_HOST,
                "key": cfg.PRIVATE_KEY,
                "chain_id": cfg.CHAIN_ID,
            }
            if cfg.FUNDER_ADDRESS:
                kwargs["funder"] = cfg.FUNDER_ADDRESS
                kwargs["signature_type"] = cfg.SIGNATURE_TYPE

            builder_cfg = self._builder_config()
            if builder_cfg:
                kwargs["builder_config"] = builder_cfg
                log.info("  Builder API: ENABLED (gasless)")

            client = ClobClient(**kwargs)
            client.set_api_creds(client.create_or_derive_api_creds())
            self.clob_client = client
            log.info(f"  CLOB client: OK | Strategy: {cfg.ORDER_STRATEGY}")
        except Exception as e:
            log.error(f"  CLOB init failed: {e}")
            self.paper_mode = True

    def _builder_config(self):
        if not cfg.BUILDER_API_KEY:
            return None
        try:
            from py_builder_signing_sdk.config import BuilderConfig
            from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
            return BuilderConfig(
                local_builder_creds=BuilderApiKeyCreds(
                    key=cfg.BUILDER_API_KEY,
                    secret=cfg.BUILDER_SECRET,
                    passphrase=cfg.BUILDER_PASSPHRASE,
                )
            )
        except Exception:
            return None

    # ─── Daily tracking ───────────────────────────────────────────

    def _check_daily_reset(self):
        today = utcnow().strftime("%Y-%m-%d")
        if today != self.trading_day:
            log.info(f"  New day: {today} | Yesterday P&L: ${self.daily_pnl:+.2f}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.trading_day = today
            self.daily_halt = False

    def _check_safety(self) -> List[str]:
        """Pre-trade safety checks. Returns list of blocking reasons."""
        issues = []
        if self.daily_pnl <= -cfg.MAX_DAILY_LOSS_USDC:
            issues.append(f"Daily loss limit: ${self.daily_pnl:+.2f}")
            self.daily_halt = True
        if self.peak_bankroll > 0:
            dd = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
            if dd > cfg.MAX_DRAWDOWN_PCT:
                issues.append(f"Max drawdown: {dd:.1%}")
        open_n = sum(1 for p in self.positions if p.status == "open")
        if open_n >= cfg.MAX_CONCURRENT_POSITIONS:
            issues.append(f"Max positions: {open_n}")
        total_exp = sum(p.size_usd for p in self.positions if p.status == "open")
        max_exp = self.bankroll * cfg.MAX_TOTAL_EXPOSURE
        if total_exp >= max_exp:
            issues.append(f"Max exposure: ${total_exp:.2f}")
        if self.bankroll < cfg.MIN_TRADE_SIZE_USDC * 2:
            issues.append(f"Low bankroll: ${self.bankroll:.2f}")
        return issues

    # ─── Main scan cycle ──────────────────────────────────────────

    def run_scan_cycle(self) -> List[TradingSignal]:
        """One full scan: discover markets → analyze → execute."""
        self._check_daily_reset()
        blocking = self._check_safety()
        if blocking:
            for b in blocking:
                log.warning(f"  SAFETY: {b}")
            self._check_positions()
            return []

        # Phase 1: Discover markets
        log.info("[1/4] Scanning weather markets...")
        markets = self.scanner.scan_weather_markets()
        log.info(f"  Found {len(markets)} markets")
        if not markets:
            return []

        # Phase 2: Compute probabilities and screen
        log.info("[2/4] Computing forecasts & probabilities...")
        candidates = []
        skip = {"no_forecast": 0, "no_edges": 0, "no_match": 0}

        for market in markets:
            engine = self.engines.get(market.station_id)
            if not engine:
                continue

            forecasts = engine.fetch_forecasts(market.target_date)
            if not forecasts:
                skip["no_forecast"] += 1
                continue

            ensemble_data = engine.fetch_ensemble(market.target_date)
            stats = engine.compute_ensemble_stats(forecasts, ensemble_data)

            edges = parse_bucket_edges(market)
            if not edges:
                continue

            is_f = market.unit == "°F"
            our_probs = engine.compute_bucket_probabilities(
                forecasts, ensemble_data, edges, is_f
            )
            if not our_probs:
                continue

            # Validate label matching
            validated = {}
            for label, prob in our_probs.items():
                for o in market.outcomes:
                    if match_probability(o.name, {label: prob}) is not None:
                        validated[label] = prob
                        break
            if not validated:
                skip["no_match"] += 1
                continue
            total_v = sum(validated.values())
            if total_v > 0:
                validated = {k: v / total_v for k, v in validated.items()}

            # Quick edge screen (relaxed threshold)
            has_edge = False
            thresh = max(0.02, cfg.MIN_EDGE_PCT - 0.02)
            for o in market.outcomes:
                p = match_probability(o.name, validated)
                if p and 0 < o.price < 1:
                    if abs(p - o.price) >= thresh:
                        has_edge = True
                        break

            if not has_edge:
                skip["no_edges"] += 1
                continue

            market._forecasts = forecasts
            market._ensemble = ensemble_data
            market._stats = stats
            market._probs = validated
            candidates.append(market)

        log.info(f"  {len(candidates)}/{len(markets)} candidates | Skip: {skip}")
        if not candidates:
            return []

        # Phase 3: Enrich with CLOB prices & detect edges
        log.info(f"[3/4] CLOB enrichment for {len(candidates)} candidates...")
        candidates = self.scanner.enrich_with_live_prices(candidates)

        all_signals = []
        exposure = sum(p.size_usd for p in self.positions if p.status == "open")
        for m in candidates:
            # Compute hours to resolution for Late Sniper timing
            hours_to_res = 24.0
            try:
                if m.end_date:
                    res_time = datetime.fromisoformat(m.end_date.replace("Z", "+00:00"))
                    hours_to_res = max(0, (res_time - utcnow()).total_seconds() / 3600)
            except Exception:
                pass
            days_to_res = hours_to_res / 24.0
            signals = self.edge_detector.find_edges(
                m, m._probs, m._stats, self.bankroll, exposure,
                days_to_res=days_to_res, hours_to_res=hours_to_res
            )
            if signals:
                log.info(f"  {m.station_id}/{m.target_date}: {len(signals)} signals")
                for s in signals[:3]:
                    log.info(f"    {s.direction} {s.outcome.name} | Edge {s.edge:.1%} | ${s.suggested_size_usd:.2f}")
            all_signals.extend(signals)

        # Phase 4: Execute
        if all_signals:
            log.info(f"[4/4] Executing {len(all_signals)} signals...")
            self._execute_signals(all_signals)
        else:
            log.info("[4/4] No actionable trades this cycle")

        self._check_positions()
        return all_signals

    # ─── Execution ────────────────────────────────────────────────

    def _execute_signals(self, signals: List[TradingSignal]):
        blocking = self._check_safety()
        if blocking:
            return

        ladder = [s for s in signals if s.strategy == "ladder"]
        no_sigs = [s for s in signals if s.strategy == "conservative_no"]

        # Execute ladder sets
        sets_done = 0
        by_market = defaultdict(list)
        for s in ladder:
            by_market[s.market.slug].append(s)

        for slug, msigs in by_market.items():
            if sets_done >= cfg.LADDER_MAX_SETS_PER_CYCLE:
                break
            msigs.sort(key=lambda s: s.expected_value * s.confidence, reverse=True)
            executed = 0
            for sig_item in msigs[:cfg.LADDER_BUCKETS]:
                if self._can_trade(sig_item):
                    self._execute_one(sig_item)
                    executed += 1
            if executed > 0:
                sets_done += 1
                log.info(f"  LADDER SET: {slug} — {executed} buckets")

        # Execute NO signals
        no_sigs.sort(key=lambda s: s.expected_value * s.confidence, reverse=True)
        no_done = 0
        for s in no_sigs:
            if no_done >= 5:
                break
            if self.daily_halt:
                break
            if self._can_trade(s):
                self._execute_one(s)
                no_done += 1

    def _can_trade(self, signal: TradingSignal) -> bool:
        """Pre-trade check for a single signal."""
        min_sz = cfg.MIN_TRADE_SIZE_USDC if signal.direction == "BUY_NO" else 1.0
        if signal.suggested_size_usd < min_sz:
            return False
        if len([p for p in self.positions if p.status == "open"]) >= cfg.MAX_CONCURRENT_POSITIONS:
            return False
        # Dedup
        if any(p.market_slug == signal.market.slug and p.outcome_name == signal.outcome.name
               and p.direction == signal.direction and p.status == "open" for p in self.positions):
            return False
        # Zone capacity
        zones = cfg.CLIMATE_ZONES
        for zone, stations in zones.items():
            if signal.market.station_id in stations:
                zone_pos = sum(1 for p in self.positions if p.station_id in stations and p.status == "open")
                if zone_pos >= cfg.MAX_POSITIONS_PER_ZONE:
                    return False
        return True

    def _execute_one(self, signal: TradingSignal):
        """Execute a single trade (paper or live)."""
        if self.paper_mode:
            self._paper_execute(signal)
        else:
            self._live_execute(signal)

    def _paper_execute(self, signal: TradingSignal):
        if signal.direction == "BUY_YES":
            entry = signal.outcome.clob_ask if signal.outcome.clob_ask > 0 else signal.market_price
            token_id = signal.outcome.token_id
        else:
            entry = (1.0 - signal.outcome.clob_bid) if signal.outcome.clob_bid > 0 else (1.0 - signal.market_price)
            token_id = signal.outcome.no_token_id or signal.outcome.token_id

        if entry <= 0 or entry >= 1:
            return

        pos = Position(
            market_slug=signal.market.slug, outcome_name=signal.outcome.name,
            token_id=token_id, direction=signal.direction,
            entry_price=entry, shares=signal.suggested_size_usd / entry,
            size_usd=signal.suggested_size_usd, edge_at_entry=signal.edge,
            confidence=signal.confidence, entry_time=utcnow_iso(),
            resolution_time=signal.market.end_date, station_id=signal.market.station_id,
        )
        self.positions.append(pos)
        self._save_state()
        log.info(f"  [PAPER] {signal.direction} {signal.outcome.name} @ {entry:.3f} | ${signal.suggested_size_usd:.2f}")

    def _live_execute(self, signal: TradingSignal):
        if not self.clob_client:
            return
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
            from py_clob_client.order_builder.constants import BUY

            if signal.direction == "BUY_YES":
                token_id = signal.outcome.token_id
            else:
                token_id = signal.outcome.no_token_id
                if not token_id:
                    return

            # Fetch live price
            depth = self.scanner.fetch_orderbook_depth(token_id)
            if not depth["has_liquidity"]:
                return

            base_price = depth["best_ask"]
            tick = float(signal.market.tick_size or "0.01")

            strategy = cfg.ORDER_STRATEGY
            if strategy == "adaptive":
                strategy = "taker" if signal.edge > cfg.TAKER_EDGE_THRESHOLD else "maker"

            price = base_price - cfg.MAKER_PRICE_OFFSET if strategy == "maker" else base_price
            price = round(round(price / tick) * tick, 4)
            price = max(tick, min(price, 1.0 - tick))

            size = round(signal.suggested_size_usd / price, 2)
            if size < signal.market.order_min_size:
                return

            order_args = OrderArgs(price=price, size=size, side=BUY, token_id=token_id)
            options = PartialCreateOrderOptions(
                neg_risk=signal.market.neg_risk, tick_size=signal.market.tick_size,
            )

            log.info(f"  [LIVE] {signal.direction} {signal.outcome.name} @ {price:.4f} x {size}")

            signed = self.clob_client.create_order(order_args, options)
            resp = self.clob_client.post_order(signed, OrderType.GTC)

            if resp.get("success"):
                pos = Position(
                    market_slug=signal.market.slug, outcome_name=signal.outcome.name,
                    token_id=token_id, direction=signal.direction,
                    entry_price=price, shares=size, size_usd=signal.suggested_size_usd,
                    edge_at_entry=signal.edge, confidence=signal.confidence,
                    entry_time=utcnow_iso(), resolution_time=signal.market.end_date,
                    station_id=signal.market.station_id,
                )
                self.positions.append(pos)
                self._save_state()
                log.info(f"  [LIVE] Position opened")
            else:
                log.warning(f"  [LIVE] Rejected: {resp.get('errorMsg', 'unknown')}")

        except Exception as e:
            log.error(f"  [LIVE] Error: {e}")

    # ─── Position management ──────────────────────────────────────

    def _check_positions(self):
        for pos in self.positions[:]:
            if pos.status != "open":
                continue
            try:
                res_time = datetime.fromisoformat(pos.resolution_time.replace("Z", "+00:00"))
                if res_time.tzinfo is None:
                    res_time = res_time.replace(tzinfo=timezone.utc)
            except Exception:
                res_time = utcnow() + timedelta(days=7)

            hours_left = (res_time - utcnow()).total_seconds() / 3600
            if hours_left < cfg.EXIT_HOURS_BEFORE_RESOLUTION:
                self._exit_position(pos, "time_exit")

    def _exit_position(self, pos: Position, reason: str):
        exit_price = pos.entry_price
        pnl = 0.0

        if not pos.token_id.startswith("sim_") and not self.paper_mode:
            try:
                depth = self.scanner.fetch_orderbook_depth(pos.token_id)
                if depth["has_liquidity"]:
                    if pos.direction == "BUY_YES":
                        exit_price = depth["best_bid"]
                    else:
                        exit_price = 1.0 - depth["best_ask"]
            except Exception:
                pass

        pnl = pos.shares * (exit_price - pos.entry_price)
        self.trades.append(Trade(
            market_slug=pos.market_slug, outcome_name=pos.outcome_name,
            station_id=pos.station_id, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price,
            shares=pos.shares, size_usd=pos.size_usd, pnl=pnl,
            edge_at_entry=pos.edge_at_entry, entry_time=pos.entry_time,
            exit_time=utcnow_iso(), exit_reason=reason,
        ))

        self.daily_pnl += pnl
        self.daily_trades += 1
        self.bankroll += pos.size_usd + pnl
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
        self.positions.remove(pos)
        self._save_state()
        log.info(f"  EXIT {pos.outcome_name} @ {exit_price:.3f} | P&L: ${pnl:+.2f} | Reason: {reason}")

    # ─── State persistence ────────────────────────────────────────

    def _save_state(self):
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        mode = "paper" if self.paper_mode else "live"
        state = {
            "timestamp": utcnow_iso(), "version": "V6", "mode": mode.upper(),
            "bankroll": self.bankroll, "peak_bankroll": self.peak_bankroll,
            "daily_pnl": self.daily_pnl, "daily_trades": self.daily_trades,
            "trading_day": self.trading_day, "daily_halt": self.daily_halt,
            "positions": [asdict(p) for p in self.positions],
            "trades": [asdict(t) for t in self.trades[-200:]],  # Keep last 200
        }
        try:
            with open(f"{cfg.RESULTS_DIR}/{mode}_state.json", "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            log.error(f"  Save state failed: {e}")

    def _load_state(self):
        mode = "paper" if self.paper_mode else "live"
        path = f"{cfg.RESULTS_DIR}/{mode}_state.json"
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                state = json.load(f)
            # Never load mismatched mode
            if state.get("mode", "PAPER") != mode.upper():
                return
            self.bankroll = state.get("bankroll", self.bankroll)
            self.peak_bankroll = state.get("peak_bankroll", self.peak_bankroll)
            self.daily_pnl = state.get("daily_pnl", 0)
            self.daily_trades = state.get("daily_trades", 0)
            self.trading_day = state.get("trading_day", self.trading_day)
            self.daily_halt = state.get("daily_halt", False)
            for p in state.get("positions", []):
                self.positions.append(Position(**{k: v for k, v in p.items() if k in Position.__dataclass_fields__}))
            for t in state.get("trades", []):
                self.trades.append(Trade(**{k: v for k, v in t.items() if k in Trade.__dataclass_fields__}))
            log.info(f"  Loaded: ${self.bankroll:.2f}, {len(self.positions)} pos, {len(self.trades)} trades")
        except Exception as e:
            log.error(f"  Load state failed: {e}")

    # ─── P&L report ───────────────────────────────────────────────

    def print_report(self):
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        wr = len(wins) / len(self.trades) * 100 if self.trades else 0
        pf = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float("inf")
        dd = (self.peak_bankroll - self.bankroll) / self.peak_bankroll * 100 if self.peak_bankroll > 0 else 0

        log.info(f"\n{'='*55}")
        log.info(f"  P&L Report V6 ({'PAPER' if self.paper_mode else 'LIVE'})")
        log.info(f"{'='*55}")
        log.info(f"  Bankroll:    ${self.bankroll:.2f} (peak ${self.peak_bankroll:.2f})")
        log.info(f"  Drawdown:    {dd:.1f}%")
        log.info(f"  Realized:    ${sum(t.pnl for t in self.trades):+.2f}")
        log.info(f"  Positions:   {len([p for p in self.positions if p.status == 'open'])}")
        log.info(f"  Trades:      {len(self.trades)} (WR: {wr:.0f}%, PF: {pf:.2f})")
        log.info(f"  Daily P&L:   ${self.daily_pnl:+.2f} ({self.daily_trades} trades)")
        log.info(f"{'='*55}")

    # ─── Continuous loop ──────────────────────────────────────────

    def run_continuous(self, duration_minutes: int = 60):
        log.info(f"Starting continuous mode for {duration_minutes} minutes...")

        def shutdown(signum, frame):
            log.info("Shutdown signal, saving...")
            self._save_state()
            self.print_report()
            raise SystemExit(0)

        sig.signal(sig.SIGTERM, shutdown)
        sig.signal(sig.SIGINT, shutdown)

        end = datetime.now() + timedelta(minutes=duration_minutes)
        cycle = 0

        while datetime.now() < end:
            cycle += 1
            log.info(f"\n{'─'*40}")
            log.info(f"  Cycle {cycle} | {datetime.now().strftime('%H:%M:%S')} | ${self.bankroll:.2f}")
            log.info(f"{'─'*40}")

            if self.peak_bankroll > 0:
                dd = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
                if dd > cfg.MAX_DRAWDOWN_PCT:
                    log.warning(f"  DRAWDOWN HALT: {dd:.1%}")
                    break

            try:
                self.run_scan_cycle()
            except Exception as e:
                log.error(f"  Cycle error: {e}")
                import traceback
                traceback.print_exc()

            if cycle % 5 == 0:
                self.print_report()

            sleep = min(cfg.SCAN_INTERVAL_SECONDS, (end - datetime.now()).total_seconds())
            if sleep > 0:
                log.info(f"  Sleeping {sleep:.0f}s...")
                time.sleep(sleep)

        self.print_report()
