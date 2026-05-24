"""
Risk management: Kelly criterion sizing, daily/weekly/monthly loss limits,
position caps, correlation blocking, portfolio exposure cap,
drawdown-based protection, and adversarial market blocking.
"""

import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from .logger import (
    get_daily_loss, get_weekly_loss, get_monthly_loss,
    get_open_positions, DB_PATH,
)

log = logging.getLogger(__name__)


_TEMP_METRICS = {"TEMP_ABOVE", "TEMP_BELOW"}

# Minimum bankroll to continue trading — prevents ruin spirals
_MIN_BANKROLL_USDC = 50.0


class RiskVetoError(Exception):
    """Raised when risk checks block a trade."""


@dataclass
class TradeApproval:
    size_usdc: float
    reason: str


class RiskManager:
    def __init__(self, config: dict):
        t = config["trading"]
        r = config.get("risk", {})

        self.kelly_fraction: float          = t.get("kelly_fraction", 0.25)
        self.max_position: float            = t.get("max_position_usdc", 50)
        self.max_open: int                  = t.get("max_open_positions", 5)
        ll = config.get("loss_limits", {})
        self.daily_limit: float             = ll.get("daily_usdc", t.get("daily_loss_limit_usdc", 200))
        self.weekly_limit: float            = ll.get("weekly_usdc", 500)
        self.monthly_limit: float           = ll.get("monthly_usdc", 1000)
        self.min_confidence: float          = t.get("min_confidence", 0.50)
        self.stale_hours: float             = r.get("stale_forecast_hours", 3)
        self.correlation_block: bool        = r.get("correlation_block", True)
        self.max_city_exposure_pct: float   = r.get("max_city_exposure_pct", 0.30)
        self.max_portfolio_exp_pct: float   = r.get("max_portfolio_exposure_pct", 0.60)
        self.drawdown_halt_pct: float       = r.get("drawdown_halt_pct", 0.40)
        self.paper: bool                    = t.get("paper_mode", True)
        self._loss_streak_pause: int        = t.get("loss_streak_pause", 5)

        # Crypto-specific risk limits (much tighter than weather)
        crypto_cfg = config.get("crypto", {})
        self._crypto_max_position: float    = crypto_cfg.get("max_position_usdc", 5.0)
        self._crypto_daily_limit: float     = crypto_cfg.get("daily_loss_limit_usdc", 15.0)

        # Per-scan caches (TTL = 120s) to avoid repeated DB queries within one scan batch
        self._streak_cache: int    = 0
        self._streak_ts: float     = 0.0
        self._peak_cache: float    = 0.0
        self._peak_ts: float       = 0.0
        self._cache_ttl: float     = 120.0

    # ── Kelly sizing ───────────────────────────────────────────────────────────

    def kelly_size(
        self,
        prob: float,
        market_price: float,
        bankroll: float,
        confidence: float = 1.0,
        regime: str = "NORMAL",
    ) -> float:
        """
        Dynamic fractional Kelly bet size in USDC.

        Full Kelly: f* = (p*b - q) / b
        Dynamic fraction = base_fraction × confidence_adj × regime_multiplier

        Regime multipliers:
          NORMAL:    1.0  — full dynamic fraction
          UNCERTAIN: 0.60 — reduce size when models disagree
          EXTREME:   0.0  — already vetoed before reaching here
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0
        b = (1.0 / market_price) - 1.0
        if b <= 0:
            return 0.0
        full_kelly = (prob * b - (1.0 - prob)) / b
        if full_kelly <= 0:
            return 0.0

        regime_mult = {"NORMAL": 1.0, "UNCERTAIN": 0.60, "EXTREME": 0.0}.get(regime, 1.0)
        conf_adj    = max(0.5, min(1.0, confidence))  # clamp to [0.5, 1.0]
        dyn_fraction = self.kelly_fraction * conf_adj * regime_mult

        return min(bankroll * full_kelly * dyn_fraction, self.max_position)

    # ── Exposure helpers ───────────────────────────────────────────────────────

    def _city_exposure(self, city: str, open_positions: List[dict]) -> float:
        return sum(
            pos["size_usdc"]
            for pos in open_positions
            if (pos.get("city") or "").lower() == city.lower()
        )

    def _portfolio_exposure(self, open_positions: List[dict]) -> float:
        """Total USDC across all open positions."""
        return sum(pos["size_usdc"] for pos in open_positions)

    # ── Drawdown & streak protection ───────────────────────────────────────────

    def _compute_peak_bankroll(self, initial: float) -> float:
        """Peak bankroll = highest running total across all resolved trades. Cached 120s."""
        import time as _t
        now = _t.time()
        if now - self._peak_ts < self._cache_ttl and self._peak_cache > 0:
            return self._peak_cache
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT pnl_usdc FROM trades WHERE paper=? AND outcome IN ('WIN','LOSS') "
                "ORDER BY timestamp ASC",
                (int(self.paper),),
            ).fetchall()
        except sqlite3.Error:
            return initial
        finally:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
        running = initial
        peak = initial
        for r in rows:
            running += (r["pnl_usdc"] or 0)
            if running > peak:
                peak = running
        self._peak_cache = peak
        self._peak_ts = now
        return peak

    def _consecutive_losses(self) -> int:
        """Count most recent consecutive LOSS outcomes. Cached for 120s."""
        import time as _t
        now = _t.time()
        if now - self._streak_ts < self._cache_ttl:
            return self._streak_cache
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT outcome FROM trades WHERE paper=? AND outcome IN ('WIN','LOSS') "
                "ORDER BY timestamp DESC LIMIT 10",
                (int(self.paper),),
            ).fetchall()
        except sqlite3.Error:
            return 0
        finally:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
        count = 0
        for r in rows:
            if r["outcome"] == "LOSS":
                count += 1
            else:
                break
        self._streak_cache = count
        self._streak_ts = now
        return count

    def invalidate_caches(self) -> None:
        """Call after a trade resolves to force fresh DB reads."""
        self._streak_ts = 0.0
        self._peak_ts = 0.0

    def _drawdown_kelly_multiplier(self, bankroll: float, initial: float) -> float:
        """
        Scale Kelly fraction by current drawdown from peak.
        Returns 0.0 when drawdown exceeds halt threshold (signals trade veto).
        Graduated reduction:  10% dd → 0.75×,  20% → 0.50×,  30% → 0.25×
        """
        peak = self._compute_peak_bankroll(initial)
        if peak <= 0:
            return 1.0
        drawdown = (peak - bankroll) / peak
        if drawdown <= 0:
            return 1.0
        if drawdown >= self.drawdown_halt_pct:
            return 0.0
        # Graduated reduction — starts reducing at 5% drawdown
        for threshold, mult in [(0.15, 0.25), (0.10, 0.50), (0.05, 0.75)]:
            if drawdown >= threshold:
                return mult
        return 1.0

    def _volatility_multiplier(self) -> float:
        """
        Reduce position sizing when recent PnL volatility is high.
        Uses the last 20 trades' PnL standard deviation:
          stdev < $5   -> 1.0  (normal sizing)
          stdev $5-$10 -> 0.80 (slightly reduce)
          stdev $10-$20-> 0.60 (moderate reduction)
          stdev > $20  -> 0.40 (heavy reduction — volatile period)
        """
        try:
            from .logger import get_pnl_volatility
            vol = get_pnl_volatility(paper=self.paper, last_n=20)
        except Exception:
            return 1.0
        if vol <= 5.0:
            return 1.0
        if vol <= 10.0:
            return 0.80
        if vol <= 20.0:
            return 0.60
        return 0.40

    def _streak_kelly_multiplier(self) -> float:
        """Reduce Kelly during consecutive losing streaks.
        With loss_streak_pause config, fully halt trading after N consecutive losses."""
        streak = self._consecutive_losses()
        if streak >= self._loss_streak_pause:
            return 0.0   # full halt — triggers RiskVetoError
        if streak >= 4:
            return 0.40
        if streak >= 3:
            return 0.60
        if streak >= 2:
            return 0.80
        return 1.0

    # ── Portfolio VaR ─────────────────────────────────────────────────────────

    def estimate_portfolio_var(
        self, open_positions: List[dict], bankroll: float
    ) -> dict:
        """
        Simple parametric VaR for the current open position book.
        Conservative: assumes full correlation (all positions lose together).

        Returns dict with total_exposure, expected_loss, var_95, var_99,
        and pct_of_bankroll.
        """
        if not open_positions:
            return {
                "total_exposure": 0.0, "expected_loss": 0.0,
                "var_95": 0.0, "var_99": 0.0, "pct_of_bankroll": 0.0,
            }

        total_exp = sum(p["size_usdc"] for p in open_positions)
        # Expected loss = Σ size × P(loss for that side)
        exp_loss = 0.0
        for p in open_positions:
            prob = p.get("our_prob", 0.5)
            side = p.get("side", "YES")
            p_loss = (1.0 - prob) if side == "YES" else prob
            exp_loss += p["size_usdc"] * p_loss

        # 95% VaR: worst correlated scenario for independent positions
        # Using normal approximation with full-correlation assumption
        import math
        # var_95 = expected_loss + 1.645 × sqrt(Σ variance_i)
        variance_sum = sum(
            (p["size_usdc"] ** 2)
            * (
                (1.0 - p.get("our_prob", 0.5)) if p.get("side") == "YES"
                else p.get("our_prob", 0.5)
            )
            * (
                (p.get("our_prob", 0.5)) if p.get("side") == "YES"
                else (1.0 - p.get("our_prob", 0.5))
            )
            for p in open_positions
        )
        std_portfolio = math.sqrt(variance_sum) if variance_sum > 0 else 0.0

        return {
            "total_exposure": round(total_exp, 2),
            "expected_loss":  round(exp_loss, 2),
            "var_95":         round(exp_loss + 1.645 * std_portfolio, 2),
            "var_99":         round(exp_loss + 2.326 * std_portfolio, 2),
            "pct_of_bankroll": round(total_exp / bankroll * 100, 1) if bankroll > 0 else 0.0,
        }

    # ── Correlation check ──────────────────────────────────────────────────────

    def _is_correlated(self, opportunity, open_positions: List[dict]) -> bool:
        """
        Block if we already have an open position on the same city + date that
        uses the same metric or a closely related metric (both temp directions).
        """
        if not self.correlation_block:
            return False
        opp_date = (opportunity.target_dt or "")[:10]
        if not opp_date:
            return False  # no date → can't confirm correlation, allow trade
        for pos in open_positions:
            if pos.get("city") != opportunity.city:
                continue
            pos_expiry = (pos.get("expiry") or "")[:10]
            if not pos_expiry or pos_expiry != opp_date:
                continue
            pos_metric = pos.get("metric", "")
            if pos_metric == opportunity.metric:
                return True
            # Block opposite temp directions on same city/date — highly correlated
            if pos_metric in _TEMP_METRICS and opportunity.metric in _TEMP_METRICS:
                return True
        return False

    # ── Main approval gate ─────────────────────────────────────────────────────

    def approve(
        self,
        opportunity,          # strategy.Opportunity dataclass
        bankroll: float,
        weather_age_seconds: float = 0,
    ) -> TradeApproval:
        """
        Run all risk checks. Returns TradeApproval with final size.
        Raises RiskVetoError with a reason if any check fails.
        """
        # 0. Bankroll floor — halt if capital is critically low
        if bankroll < _MIN_BANKROLL_USDC:
            raise RiskVetoError(
                f"Bankroll {bankroll:.2f} USDC below floor {_MIN_BANKROLL_USDC:.2f} — trading halted"
            )

        # 0.1. Adversarial market check — block flagged markets for cooldown period
        if getattr(opportunity, "adversarial", False):
            raise RiskVetoError(
                f"Market {opportunity.market_id[:30]} flagged adversarial — entry blocked"
            )

        # 1. Confidence floor
        if opportunity.confidence < self.min_confidence:
            raise RiskVetoError(
                f"Confidence {opportunity.confidence:.2f} below minimum {self.min_confidence}"
            )

        # 2. Stale forecast
        max_stale = self.stale_hours * 3600
        if weather_age_seconds > max_stale:
            raise RiskVetoError(
                f"Forecast is {weather_age_seconds/3600:.1f}h old "
                f"(max {self.stale_hours}h)"
            )

        # 3. Daily loss limit
        daily_loss = get_daily_loss(paper=self.paper)
        if daily_loss >= self.daily_limit:
            raise RiskVetoError(
                f"Daily loss limit reached: {daily_loss:.2f} USDC "
                f"(limit {self.daily_limit:.2f})"
            )

        # 3b. Weekly loss limit — soft pause until Monday 00:00 UTC
        weekly_loss = get_weekly_loss(paper=self.paper)
        if weekly_loss >= self.weekly_limit:
            raise RiskVetoError(
                f"Weekly loss limit reached: {weekly_loss:.2f} USDC "
                f"(limit {self.weekly_limit:.2f}) — resumes Monday 00:00 UTC"
            )

        # 3c. Monthly loss limit — hard stop
        monthly_loss = get_monthly_loss(paper=self.paper)
        if monthly_loss >= self.monthly_limit:
            raise RiskVetoError(
                f"Monthly loss limit reached: {monthly_loss:.2f} USDC "
                f"(limit {self.monthly_limit:.2f}) — manual restart required"
            )

        # 4. Max open positions
        open_positions = get_open_positions(paper=self.paper)
        if len(open_positions) >= self.max_open:
            raise RiskVetoError(
                f"Already at max open positions ({self.max_open})"
            )

        # 5. Correlation block
        if self._is_correlated(opportunity, open_positions):
            raise RiskVetoError(
                f"Correlated position already open for "
                f"{opportunity.city} / {opportunity.metric}"
            )

        # 5.5 City portfolio exposure cap
        city = opportunity.city or ""
        if city:
            city_exp = self._city_exposure(city, open_positions)
            max_city = bankroll * self.max_city_exposure_pct
            if city_exp >= max_city:
                raise RiskVetoError(
                    f"City exposure cap for {city}: "
                    f"{city_exp:.2f}/{max_city:.2f} USDC ({self.max_city_exposure_pct:.0%})"
                )

        # 5.6 Drawdown protection — scale or halt based on current drawdown
        initial = float(os.getenv("BANKROLL_USDC", "500"))
        dd_mult = self._drawdown_kelly_multiplier(bankroll, initial)
        if dd_mult == 0.0:
            raise RiskVetoError(
                f"Drawdown halt: bankroll {bankroll:.2f} has exceeded "
                f"{self.drawdown_halt_pct:.0%} drawdown from peak"
            )
        streak_mult = self._streak_kelly_multiplier()
        if streak_mult == 0.0:
            streak_count = self._consecutive_losses()
            raise RiskVetoError(
                f"Loss streak halt: {streak_count} consecutive losses "
                f"(pause after {self._loss_streak_pause}) — wait for a win to resume"
            )

        # 6. Dynamic Kelly sizing (regime + confidence aware)
        regime = getattr(opportunity, "regime", "NORMAL")
        size = self.kelly_size(
            prob=opportunity.our_prob,
            market_price=opportunity.market_price,
            bankroll=bankroll,
            confidence=opportunity.confidence,
            regime=regime,
        )
        size = size * dd_mult * streak_mult

        # 6.05 Crypto position cap — much smaller than weather
        is_crypto = opportunity.metric in ("CRYPTO_ABOVE", "CRYPTO_BELOW")
        if is_crypto:
            size = min(size, self._crypto_max_position)

        # 6.1 Volatility-adjusted sizing — reduce during high-variance periods
        vol_mult = self._volatility_multiplier()
        size = size * vol_mult

        if size < 1.0:
            raise RiskVetoError(
                f"Kelly size too small ({size:.2f} USDC) — edge may be marginal "
                f"(dd_mult={dd_mult:.2f} streak_mult={streak_mult:.2f} vol_mult={vol_mult:.2f})"
            )

        # 6.5 Total portfolio exposure cap — checked AFTER Kelly so we know actual size
        total_exp = self._portfolio_exposure(open_positions)
        max_portfolio = bankroll * self.max_portfolio_exp_pct
        if total_exp + size > max_portfolio:
            raise RiskVetoError(
                f"Portfolio exposure cap: adding {size:.2f} would bring total to "
                f"{total_exp + size:.2f}/{max_portfolio:.2f} USDC ({self.max_portfolio_exp_pct:.0%})"
            )

        log.info(
            "Risk approved: %s %.2f USDC  (daily_loss=%.2f  open=%d  "
            "dd_mult=%.2f  streak_mult=%.2f  vol_mult=%.2f)",
            opportunity.market_title[:50], size, daily_loss, len(open_positions),
            dd_mult, streak_mult, vol_mult,
        )
        return TradeApproval(size_usdc=round(size, 2), reason="OK")
