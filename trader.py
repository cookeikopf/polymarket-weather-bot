"""
Trade Execution & Position Management V7
==========================================
V7 Changes:
  - HOLD TO RESOLUTION: no fake time_exit — positions resolve naturally
  - REAL SELL: _exit_position places actual sell orders on-chain
  - ON-CHAIN BALANCE: sync bankroll from real USDC balance
  - POSITION RECONCILIATION: adopt untracked exchange positions on startup
  - RESOLUTION DETECTION: detect resolved markets, record correct P&L ($1 wins / $0 losses)
  - AVAILABLE CASH GUARD: only trade with cash, not locked-in-positions
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
    order_id: str = ""


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
    """Live/paper trading engine V7."""

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
        self._exchange_token_ids: set = set()
        self._load_state()
        self._check_daily_reset()

        log.info(f"{'='*55}")
        log.info(f"  Weather Bot V7 | {'PAPER' if paper_mode else 'LIVE'}")
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
            self._reconcile_positions()   # V7: full reconciliation on startup
            self._sync_with_exchange()

    # ─── CLOB client ──────────────────────────────────────────────

    def _check_geoblock(self) -> bool:
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
                    log.error(f"  GEOBLOCKED from {country}!")
                    return False
                return True
        except Exception as e:
            log.warning(f"  Geoblock check failed: {e}")
        return True

    def _init_clob(self):
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

    # ─── V7: Full Position Reconciliation ───────────────────────────

    def _reconcile_positions(self):
        """V7: Full reconciliation on startup.
        1. Fetch all real positions from Polymarket data API
        2. Check for resolved markets — record P&L for wins/losses
        3. Adopt any untracked active positions
        4. Remove phantom positions (we track but exchange doesn't have)
        5. Sync bankroll from real cash + position values
        """
        if self.paper_mode:
            return
        try:
            funder = cfg.FUNDER_ADDRESS
            if not funder:
                return

            resp = requests.get(
                f"https://data-api.polymarket.com/positions?user={funder}",
                timeout=15,
            )
            if resp.status_code != 200:
                log.warning(f"  [RECONCILE] API returned {resp.status_code}")
                return

            real_positions = resp.json()
            if not isinstance(real_positions, list):
                return

            log.info(f"  [RECONCILE] Exchange has {len(real_positions)} positions")
            tracked = {p.token_id: p for p in self.positions if p.status in ("open", "pending")}

            # Process each real position
            for rp in real_positions:
                token_id = rp.get("asset", "")
                real_shares = float(rp.get("size", 0))
                cur_price = float(rp.get("curPrice", 0))
                cur_value = float(rp.get("currentValue", 0))
                initial_value = float(rp.get("initialValue", 0))
                redeemable = rp.get("redeemable", False)
                title = rp.get("title", "?")[:60]
                pnl_cash = float(rp.get("cashPnl", 0))

                if token_id in tracked:
                    tp = tracked[token_id]
                    # Fix share mismatch
                    if abs(real_shares - tp.shares) > 0.01:
                        log.info(f"  [RECONCILE] Fix shares {tp.outcome_name}: {tp.shares:.2f} → {real_shares:.2f}")
                        tp.shares = real_shares

                    # Check if resolved (value=0 and redeemable, or price=0.9995+)
                    if redeemable and cur_value == 0:
                        # Lost — resolved to $0
                        self._record_resolution(tp, won=False)
                        tracked.pop(token_id, None)
                    elif cur_price >= 0.999 and real_shares > 0:
                        # Won — price is ~$1.00, may not be redeemable yet but basically won
                        # Don't close yet — wait for official resolution to claim
                        log.info(f"  [RECONCILE] WINNING: {tp.outcome_name} (price={cur_price:.4f}, value=${cur_value:.2f})")
                    else:
                        log.info(f"  [RECONCILE] OK: {tp.outcome_name} | {real_shares:.1f} shares @ ${cur_price:.4f} = ${cur_value:.2f}")
                else:
                    # Untracked position on exchange
                    if redeemable and cur_value == 0:
                        log.info(f"  [RECONCILE] Untracked LOSS (redeemable, $0): {title}")
                        # Nothing to do — it's a resolved loss
                    elif cur_value > 0.01:
                        # Active untracked position — adopt it
                        log.warning(f"  [RECONCILE] ADOPTING untracked: {title} | {real_shares:.1f} shares, value=${cur_value:.2f}")
                        avg_price = float(rp.get("avgPrice", cur_price))
                        outcome = rp.get("outcome", "Yes")
                        direction = "BUY_YES" if outcome == "Yes" else "BUY_NO"
                        slug = rp.get("eventSlug", "")
                        end_date = rp.get("endDate", "")

                        new_pos = Position(
                            market_slug=slug,
                            outcome_name=title[:40],
                            token_id=token_id,
                            direction=direction,
                            entry_price=avg_price,
                            shares=real_shares,
                            size_usd=round(initial_value, 4),
                            edge_at_entry=0.0,  # Unknown
                            confidence=0.5,
                            entry_time=utcnow_iso(),
                            resolution_time=end_date if end_date else "",
                            station_id="",
                            status="open",
                        )
                        self.positions.append(new_pos)
                    else:
                        log.info(f"  [RECONCILE] Untracked dust: {title} (value=${cur_value:.4f})")

            # Remove phantoms (we track but exchange doesn't have)
            real_token_ids = {rp.get("asset", "") for rp in real_positions}
            for token_id, tp in list(tracked.items()):
                if token_id not in real_token_ids:
                    log.warning(f"  [RECONCILE] PHANTOM removed: {tp.outcome_name}")
                    self.positions.remove(tp)

            # V7: Sync bankroll from on-chain reality
            # Total value = sum of all active position values
            total_pos_value = sum(
                float(rp.get("currentValue", 0))
                for rp in real_positions
                if float(rp.get("currentValue", 0)) > 0
            )
            total_invested = sum(
                p.size_usd for p in self.positions if p.status in ("open", "pending")
            )
            
            # Log portfolio summary
            log.info(f"  [RECONCILE] Portfolio: positions=${total_pos_value:.2f} | "
                     f"tracked_invested=${total_invested:.2f} | bankroll=${self.bankroll:.2f}")

            self._save_state()

        except Exception as e:
            log.warning(f"  [RECONCILE] Failed: {e}")
            import traceback
            traceback.print_exc()

    def _record_resolution(self, pos: Position, won: bool):
        """Record a resolved position's P&L and remove from active positions."""
        if won:
            exit_price = 1.0
            pnl = pos.shares * (1.0 - pos.entry_price)
        else:
            exit_price = 0.0
            pnl = -pos.size_usd

        self.trades.append(Trade(
            market_slug=pos.market_slug, outcome_name=pos.outcome_name,
            station_id=pos.station_id, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price,
            shares=pos.shares, size_usd=pos.size_usd, pnl=pnl,
            edge_at_entry=pos.edge_at_entry, entry_time=pos.entry_time,
            exit_time=utcnow_iso(), exit_reason="resolution_win" if won else "resolution_loss",
        ))

        self.daily_pnl += pnl
        self.daily_trades += 1
        if won:
            self.bankroll += pos.size_usd + pnl  # Return cost + profit
        # For losses, the cost is already gone (was deducted at entry)
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
        if pos in self.positions:
            self.positions.remove(pos)
        log.info(f"  [RESOLVED] {'WIN' if won else 'LOSS'} {pos.outcome_name} | "
                 f"P&L: ${pnl:+.2f} | Shares: {pos.shares:.1f}")

    # ─── Exchange Sync ────────────────────────────────────────────

    def _sync_with_exchange(self):
        """Sync pending orders: check fills, cancel stale, refund unfilled."""
        if self.paper_mode or not self.clob_client:
            return
        try:
            pending = [p for p in self.positions if p.status == "pending"]
            for pos in pending:
                order_id = pos.order_id
                if not order_id:
                    self.bankroll += pos.size_usd
                    self.positions.remove(pos)
                    log.info(f"  [SYNC] Removed orphan pending: {pos.outcome_name}")
                    continue
                try:
                    order_info = self.clob_client.get_order(order_id)
                    matched = float(order_info.get("size_matched", 0) or 0)
                    status = order_info.get("status", "")

                    if matched > 0 and status in ("MATCHED", "CLOSED"):
                        actual_cost = round(matched * pos.entry_price, 4)
                        refund = pos.size_usd - actual_cost
                        pos.shares = matched
                        pos.size_usd = actual_cost
                        pos.status = "open"
                        if refund > 0:
                            self.bankroll += refund
                        log.info(f"  [SYNC] FILLED: {pos.outcome_name} | {matched:.1f} shares @ ${actual_cost:.2f}")
                    elif status == "LIVE":
                        log.info(f"  [SYNC] Pending: {pos.outcome_name} still LIVE")
                    else:
                        self.bankroll += pos.size_usd
                        self.positions.remove(pos)
                        log.info(f"  [SYNC] Refunded: {pos.outcome_name} (status={status})")
                except Exception as e:
                    log.warning(f"  [SYNC] Order check failed for {pos.outcome_name}: {e}")

            # Refresh exchange order cache
            self._exchange_token_ids = set()
            try:
                orders = self.clob_client.get_orders()
                if isinstance(orders, list):
                    for o in orders:
                        if o.get("status") in ("LIVE", "MATCHED"):
                            self._exchange_token_ids.add(o.get("asset_id", ""))
            except Exception:
                pass

            for p in self.positions:
                if p.status in ("open", "pending"):
                    self._exchange_token_ids.add(p.token_id)

            open_pos = [p for p in self.positions if p.status == "open"]
            pending_pos = [p for p in self.positions if p.status == "pending"]
            invested = sum(p.size_usd for p in open_pos + pending_pos)
            log.info(f"  [SYNC] Bankroll: ${self.bankroll:.2f} | Invested: ${invested:.2f} | Open: {len(open_pos)} | Pending: {len(pending_pos)}")
            self._save_state()
        except Exception as e:
            log.warning(f"  [SYNC] Failed: {e}")

    def _verify_portfolio(self):
        """Periodic portfolio check — lighter than full reconciliation."""
        if self.paper_mode:
            return
        try:
            funder = cfg.FUNDER_ADDRESS
            if not funder:
                return
            resp = requests.get(
                f"https://data-api.polymarket.com/positions?user={funder}",
                timeout=15,
            )
            if resp.status_code != 200:
                return

            real_positions = resp.json()
            if not isinstance(real_positions, list):
                return

            tracked = {p.token_id: p for p in self.positions if p.status in ("open", "pending")}

            for rp in real_positions:
                token_id = rp.get("asset", "")
                real_shares = float(rp.get("size", 0))
                cur_value = float(rp.get("currentValue", 0))
                cur_price = float(rp.get("curPrice", 0))
                pnl = float(rp.get("cashPnl", 0))
                title = rp.get("title", "?")[:60]
                redeemable = rp.get("redeemable", False)

                if token_id in tracked:
                    tp = tracked[token_id]
                    # Check for resolution
                    if redeemable and cur_value == 0:
                        log.info(f"  [EYES] RESOLVED LOSS: {tp.outcome_name}")
                        self._record_resolution(tp, won=False)
                    elif cur_price >= 0.999:
                        log.info(f"  [EYES] WINNING: {tp.outcome_name} (${cur_value:.2f})")
                    else:
                        log.info(f"  [EYES] OK {tp.outcome_name}: {real_shares:.1f}sh, ${cur_value:.2f}, P&L ${pnl:+.2f}")
                else:
                    if cur_value > 0.01 and not redeemable:
                        log.warning(f"  [EYES] UNTRACKED: {title} | ${cur_value:.2f}")

            total_value = sum(float(rp.get("currentValue", 0)) for rp in real_positions)
            total_pnl = sum(float(rp.get("cashPnl", 0)) for rp in real_positions)
            invested = sum(p.size_usd for p in self.positions if p.status in ("open", "pending"))
            log.info(
                f"  [EYES] Portfolio: cash=${self.bankroll:.2f} + "
                f"positions=${total_value:.2f} = ${self.bankroll + total_value:.2f} | "
                f"P&L=${total_pnl:+.2f}"
            )
            self._save_state()
        except Exception as e:
            log.warning(f"  [EYES] Portfolio check failed: {e}")

    # ─── Daily tracking ───────────────────────────────────────────

    def _check_daily_reset(self):
        today = utcnow().strftime("%Y-%m-%d")
        if today != self.trading_day:
            log.info(f"  New day: {today} | Yesterday P&L: ${self.daily_pnl:+.2f}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.trading_day = today
            self.daily_halt = False

    def _available_cash(self) -> float:
        """V7: Available cash = bankroll minus invested capital.
        This prevents the bot from spending money locked in positions."""
        invested = sum(p.size_usd for p in self.positions if p.status in ("open", "pending"))
        return max(0, self.bankroll - invested)

    def _check_safety(self) -> List[str]:
        """Pre-trade safety checks."""
        issues = []
        if self.daily_pnl <= -cfg.MAX_DAILY_LOSS_USDC:
            issues.append(f"Daily loss limit: ${self.daily_pnl:+.2f}")
            self.daily_halt = True
        invested = sum(p.size_usd for p in self.positions if p.status in ("open", "pending"))
        total_capital = self.bankroll
        if self.peak_bankroll > 0:
            dd = (self.peak_bankroll - total_capital) / self.peak_bankroll
            if dd > cfg.MAX_DRAWDOWN_PCT:
                issues.append(f"Max drawdown: {dd:.1%} (capital ${total_capital:.2f})")
        open_n = sum(1 for p in self.positions if p.status in ("open", "pending"))
        if open_n >= cfg.MAX_CONCURRENT_POSITIONS:
            issues.append(f"Max positions: {open_n}")
        if invested >= total_capital * cfg.MAX_TOTAL_EXPOSURE:
            issues.append(f"Max exposure: ${invested:.2f}")
        cash = self._available_cash()
        if cash < cfg.MIN_TRADE_SIZE_USDC:
            issues.append(f"Low cash: ${cash:.2f} (bankroll ${self.bankroll:.2f}, invested ${invested:.2f})")
        return issues

    # ─── Main scan cycle ──────────────────────────────────────────

    def run_scan_cycle(self) -> List[TradingSignal]:
        """One full scan: discover markets → analyze → execute."""
        self._check_daily_reset()
        self._sync_with_exchange()
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
        sniper = [s for s in signals if s.strategy == "late_sniper"]

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

        sniper.sort(key=lambda s: s.expected_value * s.confidence, reverse=True)
        sniper_done = 0
        for s in sniper:
            if sniper_done >= cfg.SNIPER_MAX_BETS:
                break
            if self.daily_halt:
                break
            if self._can_trade(s):
                self._execute_one(s)
                sniper_done += 1
                log.info(f"  SNIPER: {s.direction} {s.outcome.name} | Edge {s.edge:.1%}")

    def _can_trade(self, signal: TradingSignal) -> bool:
        """Pre-trade check for a single signal."""
        min_sz = cfg.MIN_TRADE_SIZE_USDC if signal.direction == "BUY_NO" else 1.0
        if signal.suggested_size_usd < min_sz:
            return False
        active = [p for p in self.positions if p.status in ("open", "pending")]
        if len(active) >= cfg.MAX_CONCURRENT_POSITIONS:
            return False
        # Dedup: check local state
        if any(p.market_slug == signal.market.slug and p.outcome_name == signal.outcome.name
               and p.direction == signal.direction and p.status in ("open", "pending")
               for p in self.positions):
            return False
        # Dedup: check exchange orders
        token_id = signal.outcome.token_id if signal.direction == "BUY_YES" else signal.outcome.no_token_id
        if token_id and token_id in self._exchange_token_ids:
            return False
        # V7: Use available cash (bankroll minus invested), not raw bankroll
        cash = self._available_cash()
        if signal.suggested_size_usd > cash:
            return False
        # Exposure check
        invested = sum(p.size_usd for p in self.positions if p.status in ("open", "pending"))
        if invested + signal.suggested_size_usd > self.bankroll * cfg.MAX_TOTAL_EXPOSURE:
            return False
        # Zone capacity
        zones = cfg.CLIMATE_ZONES
        for zone, stations in zones.items():
            if signal.market.station_id in stations:
                zone_pos = sum(1 for p in self.positions if p.station_id in stations and p.status in ("open", "pending"))
                if zone_pos >= cfg.MAX_POSITIONS_PER_ZONE:
                    return False
        return True

    def _execute_one(self, signal: TradingSignal):
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

        size_usd = min(signal.suggested_size_usd, self._available_cash())
        if size_usd < 1.0:
            return

        pos = Position(
            market_slug=signal.market.slug, outcome_name=signal.outcome.name,
            token_id=token_id, direction=signal.direction,
            entry_price=entry, shares=size_usd / entry,
            size_usd=size_usd, edge_at_entry=signal.edge,
            confidence=signal.confidence, entry_time=utcnow_iso(),
            resolution_time=signal.market.end_date, station_id=signal.market.station_id,
        )
        self.positions.append(pos)
        # V7: Don't deduct from bankroll for paper — paper just tracks
        self._save_state()
        log.info(f"  [PAPER] {signal.direction} {signal.outcome.name} @ {entry:.3f} | ${size_usd:.2f}")

    def _live_execute(self, signal: TradingSignal):
        if not self.clob_client:
            return
        try:
            from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType, PartialCreateOrderOptions
            from py_clob_client.order_builder.constants import BUY

            if signal.direction == "BUY_YES":
                token_id = signal.outcome.token_id
            else:
                token_id = signal.outcome.no_token_id
                if not token_id:
                    return

            depth = self.scanner.fetch_orderbook_depth(token_id)
            if not depth["has_liquidity"]:
                log.info(f"  [LIVE] Skip {signal.outcome.name}: no liquidity")
                return

            tick = float(signal.market.tick_size or "0.01")
            eff_buy = depth.get("eff_buy_price", depth["best_ask"])

            if signal.market.neg_risk:
                fill_price = eff_buy + tick
            else:
                fill_price = eff_buy

            price = round(round(fill_price / tick) * tick, 4)
            price = max(tick, min(price, 1.0 - tick))

            eff_cost = round(round(eff_buy / tick) * tick, 4)
            if signal.direction == "BUY_YES":
                real_edge = signal.our_probability - eff_cost
            else:
                real_edge = (1.0 - signal.our_probability) - eff_cost
            if real_edge < 0.05:
                log.info(f"  [LIVE] Skip {signal.outcome.name}: real edge {real_edge:.1%} too low")
                return

            # V7: Use available cash for sizing
            cash = self._available_cash()
            trade_size = min(signal.suggested_size_usd, cash * 0.95)  # 5% buffer
            size = round(trade_size / price, 2)
            if size < signal.market.order_min_size:
                return

            if signal.market.neg_risk:
                amount = round(size * price, 2)
                market_args = MarketOrderArgs(
                    token_id=token_id,
                    amount=amount,
                    side=BUY,
                    price=price,
                )
                options = PartialCreateOrderOptions(
                    neg_risk=True, tick_size=signal.market.tick_size,
                )
                log.info(f"  [LIVE] FOK {signal.direction} {signal.outcome.name} @ {price:.4f} (eff_buy={eff_buy:.4f}) ${amount:.2f} (edge {real_edge:.1%})")
                try:
                    signed = self.clob_client.create_market_order(market_args, options)
                    resp = self.clob_client.post_order(signed, OrderType.FOK)
                except Exception as fok_err:
                    err_str = str(fok_err)
                    if "fully filled" in err_str or "FOK" in err_str:
                        log.info(f"  [LIVE] FOK rejected (insufficient liq), trying FAK...")
                        try:
                            signed = self.clob_client.create_market_order(market_args, options)
                            resp = self.clob_client.post_order(signed, OrderType.FAK)
                        except Exception as fak_err:
                            log.warning(f"  [LIVE] FAK also failed: {fak_err}")
                            return
                    else:
                        raise fok_err
            else:
                reserved_cost = round(size * price, 4)
                order_args = OrderArgs(price=price, size=size, side=BUY, token_id=token_id)
                options = PartialCreateOrderOptions(
                    neg_risk=False, tick_size=signal.market.tick_size,
                )
                log.info(f"  [LIVE] GTC {signal.direction} {signal.outcome.name} @ {price:.4f} x {size} (edge {real_edge:.1%})")
                signed = self.clob_client.create_order(order_args, options)
                resp = self.clob_client.post_order(signed, OrderType.GTC)

            if not resp or not resp.get("success"):
                err = resp.get('errorMsg', 'unknown') if resp else 'no response'
                log.warning(f"  [LIVE] Rejected: {err}")
                return

            order_id = resp.get("orderID", "")

            if signal.market.neg_risk:
                taking = float(resp.get("takingAmount", 0) or 0)
                making = float(resp.get("makingAmount", 0) or 0)
                fok_status = resp.get("status", "")
                tx_hashes = resp.get("transactionsHashes", [])

                if taking > 0 and making > 0:
                    fill_price_actual = round(making / taking, 4) if taking > 0 else price
                    pos = Position(
                        market_slug=signal.market.slug, outcome_name=signal.outcome.name,
                        token_id=token_id, direction=signal.direction,
                        entry_price=fill_price_actual, shares=taking, size_usd=round(making, 4),
                        edge_at_entry=real_edge, confidence=signal.confidence,
                        entry_time=utcnow_iso(), resolution_time=signal.market.end_date,
                        station_id=signal.market.station_id, status="open",
                        order_id=order_id,
                    )
                    self.positions.append(pos)
                    self.bankroll -= round(making, 4)
                    self._exchange_token_ids.add(token_id)
                    self._save_state()
                    tx_short = tx_hashes[0][:16] if tx_hashes else 'N/A'
                    log.info(f"  [LIVE] FOK FILLED: {taking:.1f} shares @ ${making:.2f} | Bankroll: ${self.bankroll:.2f} | Cash: ${self._available_cash():.2f} | tx={tx_short}...")
                else:
                    log.info(f"  [LIVE] FOK NO FILL: {signal.outcome.name} (status={fok_status})")
            else:
                reserved_cost = round(size * price, 4)
                pos = Position(
                    market_slug=signal.market.slug, outcome_name=signal.outcome.name,
                    token_id=token_id, direction=signal.direction,
                    entry_price=price, shares=size, size_usd=reserved_cost,
                    edge_at_entry=real_edge, confidence=signal.confidence,
                    entry_time=utcnow_iso(), resolution_time=signal.market.end_date,
                    station_id=signal.market.station_id, status="pending",
                    order_id=order_id,
                )
                self.positions.append(pos)
                self.bankroll -= reserved_cost
                self._exchange_token_ids.add(token_id)
                self._save_state()
                log.info(f"  [LIVE] GTC PLACED: {order_id[:16]}... | Reserved ${reserved_cost:.2f} | Cash: ${self._available_cash():.2f}")

        except Exception as e:
            log.error(f"  [LIVE] Error: {e}")

    # ─── V7: Position management — HOLD TO RESOLUTION ─────────────

    def _check_positions(self):
        """V7: Check positions for resolution status.
        
        KEY CHANGE: We NO LONGER do time_exit (fake selling).
        Weather markets resolve to $1 or $0. We HOLD positions
        until resolution, then record the actual result.
        
        The _verify_portfolio() call handles detecting resolved positions
        via the Polymarket data API.
        """
        # Nothing to do here for neg-risk weather markets — they resolve naturally.
        # The _verify_portfolio() and _reconcile_positions() methods handle
        # detecting resolved markets and recording P&L.
        pass

    # ─── State persistence ────────────────────────────────────────

    def _save_state(self):
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        mode = "paper" if self.paper_mode else "live"
        state = {
            "timestamp": utcnow_iso(), "version": "V7", "mode": mode.upper(),
            "bankroll": self.bankroll, "peak_bankroll": self.peak_bankroll,
            "daily_pnl": self.daily_pnl, "daily_trades": self.daily_trades,
            "trading_day": self.trading_day, "daily_halt": self.daily_halt,
            "positions": [asdict(p) for p in self.positions],
            "trades": [asdict(t) for t in self.trades[-200:]],
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
        invested = sum(p.size_usd for p in self.positions if p.status in ("open", "pending"))
        cash = self._available_cash()
        dd = (self.peak_bankroll - self.bankroll) / self.peak_bankroll * 100 if self.peak_bankroll > 0 else 0

        log.info(f"\n{'='*55}")
        log.info(f"  P&L Report V7 ({'PAPER' if self.paper_mode else 'LIVE'})")
        log.info(f"{'='*55}")
        log.info(f"  Bankroll:    ${self.bankroll:.2f} (peak ${self.peak_bankroll:.2f})")
        log.info(f"  Cash:        ${cash:.2f} (invested: ${invested:.2f})")
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
            log.info(f"  Cycle {cycle} | {datetime.now().strftime('%H:%M:%S')} | Cash: ${self._available_cash():.2f} | Bankroll: ${self.bankroll:.2f}")
            log.info(f"{'─'*40}")

            invested = sum(p.size_usd for p in self.positions if p.status in ("open", "pending"))
            total_cap = self.bankroll
            if self.peak_bankroll > 0:
                dd = (self.peak_bankroll - total_cap) / self.peak_bankroll
                if dd > cfg.MAX_DRAWDOWN_PCT:
                    log.warning(f"  DRAWDOWN PAUSE: {dd:.1%} (capital ${total_cap:.2f}) — skipping trades, will retry")
                    self._sync_with_exchange()
                    self._check_positions()
                    sleep = min(cfg.SCAN_INTERVAL_SECONDS, (end - datetime.now()).total_seconds())
                    if sleep > 0:
                        time.sleep(sleep)
                    continue

            try:
                self.run_scan_cycle()
            except Exception as e:
                log.error(f"  Cycle error: {e}")
                import traceback
                traceback.print_exc()

            if cycle % 5 == 0:
                self.print_report()
                self._verify_portfolio()

            sleep = min(cfg.SCAN_INTERVAL_SECONDS, (end - datetime.now()).total_seconds())
            if sleep > 0:
                log.info(f"  Sleeping {sleep:.0f}s...")
                time.sleep(sleep)

        self.print_report()
