"""
Probability engine and expected value calculator.

Converts ensemble weather forecasts into outcome probabilities,
then computes EV for both the YES and NO sides of a market.

Supported metrics:
  RAIN       — precip_prob directly from ensemble
  SNOW       — precip_prob weighted by temperature-based snow fraction
  TEMP_ABOVE — P(temp > threshold) via normal distribution tail
  TEMP_BELOW — P(temp < threshold) via normal distribution tail
  WIND_ABOVE — P(wind > threshold) via normal distribution tail
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Polymarket charges 2% fee on net winnings (applied to profit, not stake).
# All EV and PnL calculations account for this.
_POLY_FEE = 0.02

# Maximum allowable bid/ask spread — wider spreads eat the edge
_MAX_SPREAD = 0.05

# Time decay constant for forecast uncertainty degradation
# k=0.006 gives: 24h→0.87  48h→0.75  72h→0.65  120h→0.49  168h→0.37
_DECAY_K = 0.006

# Estimated market-impact slippage on entry (added to execution price)
_SLIPPAGE = 0.003

try:
    from scipy.stats import norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from .market_scanner import (
    RAIN, SNOW, TEMP_ABOVE, TEMP_BELOW, WIND_ABOVE,
    TEMP_RANGE, TEMP_ABOVE_MAX, TEMP_BELOW_MAX,
    CRYPTO_ABOVE, CRYPTO_BELOW,
)

log = logging.getLogger(__name__)

# Wind forecast uncertainty (fixed; wind is less lead-time sensitive than temperature)
_WIND_STD_KPH = 8.0

def _temp_std_for_horizon(hours: float) -> float:
    """
    Temperature forecast uncertainty grows with lead time.
    Based on typical NWP ensemble spread growth:
      ≤6h → ±0.8°C,  ≤24h → ±1.8°C,  ≤48h → ±2.8°C,  ≤96h → ±3.8°C,  >96h → ±5.0°C
    """
    if hours <= 6:
        return 0.8
    if hours <= 24:
        return 1.8
    if hours <= 48:
        return 2.8
    if hours <= 96:
        return 3.8
    return 5.0


def _calibration_quality(tracker) -> float:
    """
    Returns a [0.75, 1.0] quality multiplier for the edge score.
    Penalises trading when calibration is absent or uses small-sample Platt.
    """
    try:
        info = tracker.get_calibration_info()
        if not info.get("fitted"):
            return 0.80   # no calibration: 20% discount
        if info.get("method") == "isotonic":
            return 1.00   # PAV isotonic: most accurate
        return 0.92       # Platt: decent but fewer samples
    except Exception:
        return 0.80


# ── Temperature unit conversion ────────────────────────────────────────────────
def _to_celsius(value: float, unit: str) -> float:
    unit = (unit or "C").upper()
    if unit == "F":
        return (value - 32) * 5 / 9
    if unit == "K":
        return value - 273.15
    return value
# ── Normal CDF helpers ─────────────────────────────────────────────────────────

def _erf_approx(x: float) -> float:
    """Abramowitz & Stegun rational approximation of erf. Max error < 1.5e-7."""
    if not math.isfinite(x):
        return 1.0 if x > 0 else -1.0
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
           + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * math.exp(-x * x))

def _prob_above(mean: float, std: float, threshold: float) -> float:
    """P(X > threshold) for X ~ N(mean, std)."""
    if _HAS_SCIPY:
        return max(0.001, min(0.999, float(norm.sf(threshold, loc=mean, scale=std))))
    z = (threshold - mean) / (std * 1.4142135623730951)  # std * sqrt(2)
    return max(0.001, min(0.999, 0.5 * (1.0 - _erf_approx(z))))
def _prob_below(mean: float, std: float, threshold: float) -> float:
    """P(X < threshold) for X ~ N(mean, std)."""
    return 1.0 - _prob_above(mean, std, threshold)

def _prob_range(mean: float, std: float, lo: float, hi: float) -> float:
    """P(lo <= X <= hi) for X ~ N(mean, std)."""
    return max(0.001, min(0.999, _prob_below(mean, std, hi) - _prob_below(mean, std, lo)))


def time_decay_factor(hours_to_expiry: float) -> float:
    """
    Weather forecast accuracy degrades with lead time.
    Returns a multiplier in (0, 1] that reduces edge confidence for distant markets.
    Starts at 1.0 for <2h markets, decays exponentially thereafter.
    """
    return math.exp(-_DECAY_K * max(0.0, hours_to_expiry - 2.0))


def _ev_with_spread(win_prob: float, entry_price: float, spread: float) -> float:
    """
    Real EV using actual entry price + slippage + spread penalty.
    entry_price: actual ask price (what we pay)
    spread:      bid/ask spread as a fraction of mid (0.03 = 3%)
    """
    entry = min(0.99, entry_price + _SLIPPAGE)
    keep  = 1.0 - _POLY_FEE
    ev    = win_prob * (1.0 / entry - 1.0) * keep - (1.0 - win_prob)
    return round(ev - spread * 0.5, 4)  # half-spread paid on entry


# ── Core probability conversion ────────────────────────────────────────────────

def weather_to_probability(
    weather: dict, market: dict, hours_to_expiry: float = 48.0
) -> Optional[float]:
    """
    Convert ensemble weather dict + market metadata into a YES probability.
    hours_to_expiry is used to scale temperature forecast uncertainty.
    Returns None if the metric is unknown or data is insufficient.
    """
    metric = market.get("metric")
    threshold = market.get("threshold")
    unit = market.get("threshold_unit", "C")

    precip = weather.get("precip_prob")
    temp_c = weather.get("temp_c")
    wind_kph = weather.get("wind_kph")

    if metric == RAIN:
        if precip is None:
            return None
        return float(precip)

    if metric == SNOW:
        if precip is None:
            return None
        precip_f = float(precip)
        # P(snow) = P(precip occurs) × P(snow | precip, temp)
        if temp_c is None:
            snow_frac = 0.40
        elif temp_c >= 4.0:
            snow_frac = 0.02
        elif temp_c >= 2.0:
            snow_frac = 0.10
        elif temp_c >= 0.0:
            snow_frac = 0.10 + (2.0 - temp_c) / 2.0 * 0.65  # 10% → 75%
        elif temp_c >= -5.0:
            snow_frac = 0.75 + (-temp_c / 5.0) * 0.20        # 75% → 95%
        else:
            snow_frac = 0.95
        return max(0.01, precip_f * snow_frac)

    if metric in (TEMP_ABOVE, TEMP_BELOW):
        if temp_c is None or threshold is None:
            return None
        threshold_c = _to_celsius(threshold, unit)
        temp_std = _temp_std_for_horizon(hours_to_expiry)
        if metric == TEMP_ABOVE:
            return _prob_above(temp_c, temp_std, threshold_c)
        else:
            return _prob_below(temp_c, temp_std, threshold_c)

    # Temperature bucket (neg-risk daily max series) — prefer temp_max_c if available
    if metric in (TEMP_ABOVE_MAX, TEMP_BELOW_MAX, TEMP_RANGE):
        # Daily max is typically 2-4°C above the midday hourly temperature.
        # Use temp_max_c if the fetcher provides it; otherwise apply a conservative +2°C offset.
        temp_max_c = weather.get("temp_max_c")
        if temp_max_c is None and temp_c is not None:
            temp_max_c = temp_c + 2.0
        if temp_max_c is None:
            return None

        # Daily max forecast uncertainty is slightly higher than point-in-time hourly
        temp_std = _temp_std_for_horizon(hours_to_expiry) * 1.2

        if metric == TEMP_ABOVE_MAX:
            if threshold is None:
                return None
            return _prob_above(temp_max_c, temp_std, _to_celsius(threshold, unit))

        if metric == TEMP_BELOW_MAX:
            if threshold is None:
                return None
            return _prob_below(temp_max_c, temp_std, _to_celsius(threshold, unit))

        if metric == TEMP_RANGE:
            lo = market.get("threshold_low")
            hi = market.get("threshold_high")
            if lo is None or hi is None:
                return None
            return _prob_range(
                temp_max_c, temp_std,
                _to_celsius(lo, unit),
                _to_celsius(hi, unit),
            )

    if metric == WIND_ABOVE:
        if wind_kph is None or threshold is None:
            return None
        threshold_kph = threshold * 1.60934 if unit.lower() == "mph" else threshold
        return _prob_above(wind_kph, _WIND_STD_KPH, threshold_kph)

    # ── Crypto price metrics ──────────────────────────────────────────────────
    if metric in (CRYPTO_ABOVE, CRYPTO_BELOW):
        crypto_prob = weather.get("crypto_prob")
        if crypto_prob is not None:
            return float(crypto_prob)
        return None

    log.debug("Unknown metric '%s' -- cannot compute probability", metric)
    return None
# ── EV calculation ─────────────────────────────────────────────────────────────
@dataclass
class Opportunity:
    market_id: str
    market_title: str
    side: str           # "YES" or "NO"
    our_prob: float     # calibrated win probability
    market_price: float # actual entry price (best_ask or implied NO ask)
    ev: float           # real EV after fee + spread
    confidence: float   # time-decay-adjusted ensemble confidence
    score: float        # composite score for ranking
    city: str
    metric: str
    target_dt: str
    lat: float
    lon: float
    yes_token_id: Optional[str] = None
    no_token_id: Optional[str] = None
    # ── Extended fields (self-learning + regime + execution)
    raw_prob: float = 0.0         # pre-calibration probability
    regime: str = "NORMAL"        # NORMAL / UNCERTAIN / EXTREME
    hours_to_expiry: float = 72.0 # used for time decay and sizing
    spread: float = 0.0           # bid/ask spread at time of evaluation
    best_bid: Optional[float] = None  # YES best bid for passive order pricing
    yes_ask: Optional[float] = None  # YES best ask — needed for correct NO passive price
    ev_raw: float = 0.0           # EV before opportunity cost deduction (display only)
    adversarial: bool = False     # True if market was flagged by adversarial detector
    volume_usdc: float = 0.0      # market volume (used for slippage depth estimate)

def calculate_ev(our_prob: float, market_price: float) -> Tuple[float, float]:
    """
    Return (ev_yes, ev_no) after Polymarket's 2% fee on net winnings.

    Derivation (YES side, price p, fee f=0.02):
      If win:  profit = (1/p - 1) * (1 - f)   [fee on net profit only]
      If lose: profit = -1
      EV_yes = p_win * (1/p - 1) * (1-f) - (1 - p_win)

    For NO (effective price = 1-p):
      EV_no = (1 - p_win) * (1/(1-p) - 1) * (1-f) - p_win
    """
    p    = max(0.01, min(0.99, market_price))
    keep = 1.0 - _POLY_FEE
    ev_yes = our_prob * (1.0 / p - 1.0) * keep - (1.0 - our_prob)
    ev_no  = (1.0 - our_prob) * (1.0 / (1.0 - p) - 1.0) * keep - our_prob
    return round(ev_yes, 4), round(ev_no, 4)


def _liquidity_factor(volume_usdc: float) -> float:
    """Scale 0-1: 5k → 0.5, 50k → 1.0. Penalizes thin books."""
    return min(1.0, max(0.1, (volume_usdc - 5000) / 45000 + 0.5))
# ── Public interface ───────────────────────────────────────────────────────────

class ProbabilityEngine:
    def __init__(self, config: dict):
        self._config = config
        self.ev_threshold    = config["trading"]["ev_threshold"]
        self.min_confidence  = config["trading"]["min_confidence"]
        self.min_edge_score  = config["trading"].get("min_edge_score", 0.02)
        occ = config.get("opportunity_cost", {})
        self.opp_cost_enabled: bool  = occ.get("enabled", False)
        self.opp_cost_apy: float     = occ.get("apy", 0.05)
        from .model_tracker import get_tracker
        self.tracker = get_tracker()

    def evaluate(self, market: dict, weather: dict) -> Optional[Opportunity]:
        """
        Evaluate a single market. Pipeline:
          raw_prob → calibrate → time_decay → spread_check → real_EV → Opportunity
        """
        # 5-pre. Resolve hours_to_expiry first (needed for temp std + time decay)
        hours_to_expiry = 72.0
        target_dt_str = market.get("target_dt") or market.get("expiry_dt")
        if target_dt_str:
            try:
                from datetime import datetime, timezone as _tz
                target_dt = datetime.fromisoformat(target_dt_str.replace("Z", "+00:00"))
                hours_to_expiry = max(0.0,
                    (target_dt - datetime.now(_tz.utc)).total_seconds() / 3600)
            except (ValueError, TypeError):
                pass

        # 1. Raw probability from weather model (with lead-time-aware temp std)
        raw_prob = weather_to_probability(weather, market, hours_to_expiry=hours_to_expiry)
        if raw_prob is None:
            log.debug("No probability for %s", market["title"][:60])
            return None
        raw_prob = max(0.02, min(0.98, raw_prob))

        # 2. Calibrated probability (Platt scaling from historical outcomes)
        our_prob = self.tracker.calibrate(raw_prob)

        # 3. Regime guard — don't trade when models wildly disagree
        regime = weather.get("regime", "NORMAL")
        if regime == "EXTREME":
            log.debug("Extreme regime — skipping %s", market["title"][:60])
            return None

        # 4. Base confidence check
        confidence = weather.get("confidence", 0.5)
        if confidence < self.min_confidence:
            log.debug("Low confidence (%.2f) for %s", confidence, market["title"][:60])
            return None

        # 5. Time decay: reduce effective confidence for distant-expiry markets
        decay = time_decay_factor(hours_to_expiry)
        effective_confidence = confidence * decay

        # Use full min_confidence threshold (not 0.75×) — the 0.75 factor was
        # too permissive, allowing distant low-confidence markets through
        if effective_confidence < self.min_confidence:
            log.debug("Confidence after decay (%.2f) below minimum (%.2f) for %s",
                      effective_confidence, self.min_confidence, market["title"][:60])
            return None

        # 6. Spread check — skip markets where crossing costs eat the edge
        spread   = market.get("spread", 0.0)
        best_ask = market.get("best_ask") or market.get("yes_price")
        best_bid = market.get("best_bid") or (best_ask * (1.0 - spread) if best_ask else None)

        if best_ask is None:
            log.debug("Missing yes_price/best_ask for %s", market.get("title", "")[:60])
            return None

        if spread > _MAX_SPREAD:
            log.debug("Spread %.1f%% too wide for %s", spread * 100, market["title"][:60])
            return None

        # 7. Real EV with spread + slippage penalties
        # NO entry price = 1 - YES best_bid (implied NO ask via arbitrage)
        no_entry = max(0.01, min(0.99,
            1.0 - (best_bid if best_bid is not None else best_ask * (1 - spread))))

        ev_yes = _ev_with_spread(our_prob, best_ask, spread)
        ev_no  = _ev_with_spread(1.0 - our_prob, no_entry, spread)

        # 7.5. Opportunity cost: deduct time value of locked capital
        if self.opp_cost_enabled:
            days_locked = hours_to_expiry / 24.0
            opp_cost = (days_locked / 365.0) * self.opp_cost_apy
            ev_yes_adj = round(ev_yes - opp_cost, 4)
            ev_no_adj  = round(ev_no  - opp_cost, 4)
        else:
            ev_yes_adj, ev_no_adj = ev_yes, ev_no

        # 8. Side selection (threshold applied to opportunity-cost-adjusted EV)
        if ev_yes_adj >= ev_no_adj and ev_yes_adj >= self.ev_threshold:
            side, ev, ev_raw = "YES", ev_yes_adj, ev_yes
            market_price = best_ask
        elif ev_no_adj > ev_yes_adj and ev_no_adj >= self.ev_threshold:
            side, ev, ev_raw = "NO", ev_no_adj, ev_no
            market_price = no_entry
        else:
            log.debug("No edge on %s  EV_yes=%.3f(adj=%.3f)  EV_no=%.3f(adj=%.3f)",
                      market["title"][:60], ev_yes, ev_yes_adj, ev_no, ev_no_adj)
            return None

        # 9. Edge Quality Score: EV × confidence × (1−spread) × liquidity × cal_quality
        liq_factor = _liquidity_factor(market.get("volume_usdc", 0))
        cal_quality = _calibration_quality(self.tracker)
        edge_score = ev * effective_confidence * (1.0 - spread) * liq_factor * cal_quality

        if edge_score < self.min_edge_score:
            log.debug(
                "Edge score %.4f below threshold %.4f for %s",
                edge_score, self.min_edge_score, market["title"][:60],
            )
            return None

        # 9.5. Error pattern penalty — reduce EV for historically weak patterns
        try:
            from .learner import get_learner
            penalty = get_learner(self._config).get_pattern_penalty(
                city=market.get("city", ""),
                metric=market.get("metric", ""),
            )
            if penalty > 0:
                ev_before_penalty = ev
                ev = round(ev * (1.0 - penalty), 4)
                if ev < self.ev_threshold:
                    log.debug(
                        "Pattern penalty %.2f killed edge on %s (EV %.3f → %.3f)",
                        penalty, market["title"][:60], ev_before_penalty, ev,
                    )
                    return None
                log.info(
                    "Pattern penalty %.2f applied to %s (EV %.3f → %.3f)",
                    penalty, market["title"][:60], ev_before_penalty, ev,
                )
        except Exception:
            pass

        # 10. Meta model filter — secondary P(win) check on trade features
        meta_prob = self.tracker.meta_predict(
            ev=ev,
            spread=spread,
            confidence=effective_confidence,
            regime=regime,
            hours_to_expiry=hours_to_expiry,
            our_prob=our_prob,
        )
        from .model_tracker import META_VETO_THRESHOLD
        if self.tracker._meta_fitted and meta_prob < META_VETO_THRESHOLD:
            log.debug(
                "Meta model veto (P_win=%.2f < %.2f) for %s",
                meta_prob, META_VETO_THRESHOLD, market["title"][:60],
            )
            return None

        opp_cost_note = (
            f"  opp_cost={ev_raw - ev:+.4f}" if self.opp_cost_enabled else ""
        )
        log.info(
            "OPPORTUNITY  %s %s  raw=%.2f  cal=%.2f  price=%.3f  "
            "EV=%.3f(raw=%.3f)%s  conf=%.2f  decay=%.2f  spread=%.1f%%  "
            "regime=%s  edge=%.4f  meta=%.2f",
            side, market["title"][:40], raw_prob, our_prob, market_price,
            ev, ev_raw, opp_cost_note, confidence, decay,
            spread * 100, regime, edge_score, meta_prob,
        )

        return Opportunity(
            market_id=market["condition_id"],
            market_title=market["title"],
            side=side,
            our_prob=our_prob,
            market_price=market_price,
            ev=ev,
            confidence=effective_confidence,
            score=edge_score,
            city=market.get("city", ""),
            metric=market.get("metric", ""),
            target_dt=market.get("target_dt", ""),
            lat=market.get("lat", 0),
            lon=market.get("lon", 0),
            yes_token_id=market.get("yes_token_id"),
            no_token_id=market.get("no_token_id"),
            raw_prob=raw_prob,
            regime=regime,
            hours_to_expiry=hours_to_expiry,
            spread=spread,
            best_bid=best_bid,
            yes_ask=best_ask,
            ev_raw=ev_raw,
            adversarial=market.get("adversarial", False),
            volume_usdc=market.get("volume_usdc", 0.0),
        )

    def rank(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """Sort by composite score descending."""
        return sorted(opportunities, key=lambda o: o.score, reverse=True)