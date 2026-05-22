"""
Trade executor — paper mode and live mode.

Paper mode: records virtual trades to SQLite. No real money moves.
Live mode:  uses py-clob-client to place real limit orders on Polymarket.
            Requires POLY_PRIVATE_KEY in .env and a funded Polygon wallet.
"""

import logging
import math
import os
import random
import time
from typing import Optional

from .logger import log_trade, log_trade_prediction, get_realized_pnl, log_fill
from .risk_manager import RiskManager, RiskVetoError
from .strategy import Opportunity

log = logging.getLogger(__name__)

# Aggressive limit: place this many cents above best ask to guarantee fill
_LIMIT_SLIP = 0.01

# Passive fill probability: posting at mid, not crossing spread
_PASSIVE_FILL_PROB = 0.65


def _mid_price(opportunity, ask_price: float, spread: float) -> float:
    """
    Compute the true mid-price for a passive limit order.

    YES side: mid = (YES_bid + YES_ask) / 2
    NO  side: mid = 1 - YES_mid = 1 - (YES_bid + YES_ask) / 2

    Falls back to ask * (1 - spread/2) when orderbook data is unavailable.
    Without this, NO-side passive orders were computed as (YES_bid + NO_ask)/2
    which always equals 0.5 regardless of where the market is trading.
    """
    bid     = getattr(opportunity, "best_bid", None)   # YES best bid
    yes_ask = getattr(opportunity, "yes_ask",  None)   # YES best ask
    if bid is not None and yes_ask is not None:
        yes_mid = (bid + yes_ask) / 2.0
        if opportunity.side == "NO":
            return max(0.01, min(0.99, 1.0 - yes_mid))
        return max(0.01, min(0.99, yes_mid))
    # Fallback: approximate mid from ask and fractional spread
    return max(0.01, min(0.99, ask_price * (1.0 - spread / 2.0)))


class TradeExecutor:
    def __init__(self, config: dict):
        self.paper: bool = config["trading"].get("paper_mode", True)
        self.risk = RiskManager(config)
        self._clob_client = None
        self._bankroll: float = float(os.getenv("BANKROLL_USDC", "500"))
        slip_cfg = config.get("slippage", {})
        self._slip_base: float       = slip_cfg.get("base", 0.005)
        self._slip_max_impact: float = slip_cfg.get("max_impact", 0.02)

        if not self.paper:
            self._init_live_client()

    # ── Live client setup ──────────────────────────────────────────────────────

    def _init_live_client(self) -> None:
        pk = os.getenv("POLY_PRIVATE_KEY", "")
        if not pk:
            raise EnvironmentError(
                "POLY_PRIVATE_KEY must be set in .env for live trading"
            )
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.constants import POLYGON

            host = "https://clob.polymarket.com"
            chain_id = POLYGON  # 137

            self._clob_client = ClobClient(host, key=pk, chain_id=chain_id)

            # Try to load stored API credentials; generate fresh ones if absent
            api_key    = os.getenv("POLY_API_KEY", "")
            api_secret = os.getenv("POLY_API_SECRET", "")
            api_pass   = os.getenv("POLY_API_PASSPHRASE", "")

            if api_key and api_secret and api_pass:
                from py_clob_client.clob_types import ApiCreds
                creds = ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_pass,
                )
                self._clob_client.set_api_creds(creds)
            else:
                log.info("Generating new Polymarket API credentials…")
                creds = self._clob_client.create_api_key()
                log.info(
                    "NEW CREDENTIALS — save these to .env:\n"
                    "  POLY_API_KEY=%s\n  POLY_API_SECRET=%s\n  POLY_API_PASSPHRASE=%s",
                    creds.api_key, creds.api_secret, creds.api_passphrase,
                )
                self._clob_client.set_api_creds(creds)

            log.info("Live CLOB client initialized")
        except ImportError:
            raise ImportError(
                "py-clob-client is not installed. Run: pip install py-clob-client"
            )

    # ── Slippage estimation ────────────────────────────────────────────────────

    def _estimate_slippage(self, size_usdc: float, volume_usdc: float = 0.0) -> float:
        """
        Square-root market impact model: impact = base × sqrt(size / depth).
        For paper mode, depth is estimated from market volume (10% proxy).
        Returns the computed impact (capped at max_impact).
        """
        depth_estimate = max(200.0, volume_usdc * 0.10)
        impact = self._slip_base * math.sqrt(size_usdc / depth_estimate)
        return min(impact, self._slip_max_impact)

    def _estimate_slippage_live(self, size_usdc: float, token_id: str) -> float:
        """
        Fetch top-5 ask levels from CLOB to compute actual market depth,
        then apply sqrt impact model. Falls back to volume proxy on error.
        """
        try:
            from .market_scanner import CLOB_API, _safe_get as _scanner_get
            data = _scanner_get(f"{CLOB_API}/orderbook/{token_id}")
            if data:
                asks = sorted(
                    [
                        (float(a["price"]), float(a.get("size", 0)))
                        for a in data.get("asks", [])
                        if "price" in a and "size" in a
                    ]
                )[:5]
                depth_usdc = sum(p * s for p, s in asks) if asks else 0.0
                if depth_usdc > 0:
                    impact = self._slip_base * math.sqrt(size_usdc / depth_usdc)
                    log.debug(
                        "Slippage model: size=%.2f  depth=%.2f  impact=%.4f",
                        size_usdc, depth_usdc, impact,
                    )
                    return min(impact, self._slip_max_impact)
        except Exception as exc:
            log.debug("Slippage depth fetch failed: %s", exc)
        return self._slip_base  # fallback: flat base slippage

    # ── Order mode selection ───────────────────────────────────────────────────

    def _choose_order_mode(self, opportunity: Opportunity) -> str:
        """
        PASSIVE: post limit at mid-price (inside spread) — better price, ~65% fill.
        AGGRESSIVE: take liquidity at ask + slippage — guaranteed fill.

        Use aggressive when:
          • market expiry < 24h  (urgency)
          • EV > 12%             (high-conviction, prioritise fill)
          • spread > 3%          (mid-post still saves meaningful cost)

        Adapts fill probability based on learned fill rate history.
        """
        hours = getattr(opportunity, "hours_to_expiry", 72.0)
        ev    = opportunity.ev
        spread = getattr(opportunity, "spread", 0.0)

        # Check learned fill rate — switch to aggressive if passive fill rate is poor
        try:
            learned_passive_rate = self._get_learned_fill_rate()
            if learned_passive_rate is not None and learned_passive_rate < 0.40:
                # Passive fills are too rare — prefer aggressive
                log.debug("Learned passive fill rate %.0f%% — switching to aggressive",
                          learned_passive_rate * 100)
                return "AGGRESSIVE"
        except Exception:
            pass

        if hours < 24 or ev > 0.12 or spread > 0.03:
            return "AGGRESSIVE"
        return "PASSIVE"

    def _get_learned_fill_rate(self) -> Optional[float]:
        """Get the recent passive fill rate from trade history."""
        try:
            import sqlite3
            from .logger import DB_PATH
            with sqlite3.connect(str(DB_PATH)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT COUNT(*) as total FROM fill_tracking "
                    "WHERE mode='PASSIVE' AND timestamp > datetime('now', '-30 days')"
                ).fetchone()
                if not row or row["total"] < 5:
                    return None
                fill_row = conn.execute(
                    "SELECT AVG(filled) as rate FROM fill_tracking "
                    "WHERE mode='PASSIVE' AND timestamp > datetime('now', '-30 days')"
                ).fetchone()
                return float(fill_row["rate"]) if fill_row else None
        except Exception:
            return None

    # ── Main entry point ───────────────────────────────────────────────────────

    def execute(
        self,
        opportunity: Opportunity,
        weather_age_seconds: float = 0,
    ) -> Optional[int]:
        """
        Evaluate risk and place trade (paper or live).
        Returns trade_id on success, None if vetoed or failed.
        """
        try:
            approval = self.risk.approve(
                opportunity=opportunity,
                bankroll=self._bankroll,
                weather_age_seconds=weather_age_seconds,
            )
        except RiskVetoError as exc:
            log.info("Risk veto: %s", exc)
            # Notify Telegram on critical halts (loss streak, drawdown, monthly limit)
            veto_msg = str(exc)
            if any(kw in veto_msg for kw in ("streak halt", "Drawdown halt", "Monthly loss")):
                try:
                    from .notifier import notify_risk_halt
                    notify_risk_halt(veto_msg)
                except Exception:
                    pass
            return None

        size = approval.size_usdc
        price = opportunity.market_price

        if self.paper:
            return self._execute_paper(opportunity, size, price)
        else:
            return self._execute_live(opportunity, size, price)

    # ── Paper execution ────────────────────────────────────────────────────────

    def _execute_paper(
        self, opportunity: Opportunity, size: float, price: float
    ) -> Optional[int]:
        # Slippage guard: skip if market is too thin for this order size
        volume_usdc = getattr(opportunity, "volume_usdc", 0.0) or 0.0
        slippage = self._estimate_slippage(size, volume_usdc)
        if slippage >= self._slip_max_impact:
            log.info(
                "[PAPER] Slippage %.3f >= max %.3f — market too thin for %.2f USDC on %s",
                slippage, self._slip_max_impact, size, opportunity.market_title[:45],
            )
            return None

        mode = self._choose_order_mode(opportunity)
        spread = getattr(opportunity, "spread", 0.0)

        if mode == "PASSIVE":
            # Simulate limit order posted at mid: ~65% fill probability
            if random.random() > _PASSIVE_FILL_PROB:
                log.info(
                    "[PAPER] Passive miss on %s — retrying aggressive",
                    opportunity.market_title[:45],
                )
                log_fill(None, "PASSIVE", filled=False)
                mode = "AGGRESSIVE"
                price = min(0.99, price + _LIMIT_SLIP)
            else:
                passive_price = _mid_price(opportunity, price, spread)
                price_saved = price - passive_price
                log.info(
                    "[PAPER] Passive fill at mid %.3f (saved %.4f vs ask %.3f)",
                    passive_price, price_saved, price,
                )
                log_fill(None, "PASSIVE", filled=True, price_saved=price_saved)
                price = passive_price
        else:
            log_fill(None, "AGGRESSIVE", filled=True)

        trade_id = log_trade(
            market_id=opportunity.market_id,
            market_title=opportunity.market_title,
            side=opportunity.side,
            size_usdc=size,
            entry_price=price,
            our_prob=opportunity.our_prob,
            confidence=opportunity.confidence,
            ev=opportunity.ev,
            paper=True,
            city=opportunity.city,
            metric=opportunity.metric,
            expiry=opportunity.target_dt,
        )
        log_trade_prediction(
            trade_id, opportunity.raw_prob, opportunity.our_prob,
            opportunity.side, opportunity.city, opportunity.metric,
        )
        # Record trade features for meta model training
        try:
            from .model_tracker import get_tracker
            get_tracker().record_meta_features(
                trade_id,
                ev=opportunity.ev,
                spread=spread,
                confidence=opportunity.confidence,
                regime=getattr(opportunity, "regime", "NORMAL"),
                hours_to_expiry=getattr(opportunity, "hours_to_expiry", 72.0),
                our_prob=opportunity.our_prob,
            )
        except Exception as exc:
            log.debug("record_meta_features failed (non-fatal): %s", exc)

        log.info(
            "[PAPER] Trade #%d placed: %s %s  size=%.2f  price=%.3f  "
            "spread=%.1f%%  mode=%s",
            trade_id, opportunity.side, opportunity.market_title[:50],
            size, price, spread * 100, mode,
        )
        return trade_id

    # ── Live execution ─────────────────────────────────────────────────────────

    def _execute_live(
        self, opportunity: Opportunity, size: float, price: float
    ) -> Optional[int]:
        """Attempt live order with up to 3 retries on transient failures."""
        if self._clob_client is None:
            log.error("Live client not initialized")
            return None

        max_retries = 3
        for attempt in range(max_retries):
            result = self._attempt_live_order(opportunity, size, price)
            if result is not None:
                return result
            if attempt < max_retries - 1:
                wait = 2.0 ** attempt  # 1s, 2s
                log.warning(
                    "[LIVE] Order attempt %d/%d failed for %s — retrying in %.0fs",
                    attempt + 1, max_retries, opportunity.market_title[:40], wait,
                )
                time.sleep(wait)

        log.error("[LIVE] All %d order attempts failed for %s",
                  max_retries, opportunity.market_title[:50])
        return None

    def _attempt_live_order(
        self, opportunity: Opportunity, size: float, price: float
    ) -> Optional[int]:
        try:
            from py_clob_client.clob_types import OrderArgs, Side

            is_yes = opportunity.side == "YES"
            token_id_for_slip = opportunity.yes_token_id if is_yes else opportunity.no_token_id
            if token_id_for_slip:
                slippage = self._estimate_slippage_live(size, token_id_for_slip)
                if slippage >= self._slip_max_impact:
                    log.info(
                        "[LIVE] Slippage %.3f >= max %.3f — market too thin for %.2f USDC on %s",
                        slippage, self._slip_max_impact, size, opportunity.market_title[:45],
                    )
                    return None

            mode = self._choose_order_mode(opportunity)
            spread = getattr(opportunity, "spread", 0.0)

            if mode == "PASSIVE":
                # Post limit at true mid — rests inside the spread
                limit_price = _mid_price(opportunity, price, spread)
                log.info("[LIVE] Passive limit at mid %.3f (ask=%.3f)", limit_price, price)
            else:
                limit_price = min(0.99, price + _LIMIT_SLIP)

            if is_yes:
                token_id = opportunity.yes_token_id or opportunity.market_id
                if not opportunity.yes_token_id:
                    log.warning("yes_token_id missing for %s — using condition_id",
                                opportunity.market_title[:50])
            else:
                token_id = opportunity.no_token_id or opportunity.market_id
                if not opportunity.no_token_id:
                    log.warning("no_token_id missing for %s — using condition_id",
                                opportunity.market_title[:50])

            token_count = round(size / limit_price, 2)
            actual_usdc = round(token_count * limit_price, 2)

            order_args = OrderArgs(
                token_id=token_id,
                price=limit_price,
                size=token_count,
                side=Side.BUY,
            )
            resp = self._clob_client.create_and_post_order(order_args)

            if resp and resp.get("success"):
                order_id = resp.get("orderID", "unknown")
                log.info(
                    "[LIVE] Order placed: %s  order_id=%s  tokens=%.2f  price=%.3f  usdc=%.2f",
                    opportunity.market_title[:50], order_id, token_count, limit_price, actual_usdc,
                )
                trade_id = log_trade(
                    market_id=opportunity.market_id,
                    market_title=opportunity.market_title,
                    side=opportunity.side,
                    size_usdc=actual_usdc,
                    entry_price=limit_price,
                    our_prob=opportunity.our_prob,
                    confidence=opportunity.confidence,
                    ev=opportunity.ev,
                    paper=False,
                    city=opportunity.city,
                    metric=opportunity.metric,
                    expiry=opportunity.target_dt,
                )
                log_trade_prediction(
                    trade_id, opportunity.raw_prob, opportunity.our_prob,
                    opportunity.side, opportunity.city, opportunity.metric,
                )
                try:
                    from .model_tracker import get_tracker
                    get_tracker().record_meta_features(
                        trade_id,
                        ev=opportunity.ev,
                        spread=spread,
                        confidence=opportunity.confidence,
                        regime=getattr(opportunity, "regime", "NORMAL"),
                        hours_to_expiry=getattr(opportunity, "hours_to_expiry", 72.0),
                        our_prob=opportunity.our_prob,
                    )
                except Exception as exc:
                    log.debug("record_meta_features failed (non-fatal): %s", exc)
                return trade_id

            log.error("Order rejected by CLOB: %s", resp)
            return None

        except Exception as exc:
            log.error("Live order exception: %s", exc, exc_info=True)
            return None

    # ── Bankroll update ────────────────────────────────────────────────────────

    def refresh_bankroll(self) -> float:
        """Recalculate bankroll as initial USDC + all realized PnL to date."""
        initial = float(os.getenv("BANKROLL_USDC", "500"))
        pnl = get_realized_pnl(paper=self.paper)
        self._bankroll = max(1.0, initial + pnl)
        log.info(
            "Bankroll refreshed: %.2f USDC  (initial=%.2f  realized_pnl=%+.2f)",
            self._bankroll, initial, pnl,
        )
        return self._bankroll

    def get_bankroll(self) -> float:
        return self._bankroll
