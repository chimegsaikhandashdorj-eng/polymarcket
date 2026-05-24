"""
Advanced crypto signal analysis — multi-timeframe, sentiment, regime, and macro.

Provides signals:
  1. Multi-Timeframe Alignment — confirms trend across 1h, 4h, daily (TA-Lib accelerated)
  2. Fear & Greed Index — market sentiment (0=Extreme Fear, 100=Extreme Greed)
  3. Regime Detection — trending / ranging / volatile / crash (TA-Lib ADX)
  4. Correlation Guard — prevents correlated BTC+ETH positions
  5. Volume Anomaly — unusual volume as conviction booster/reducer
  6. Macro Signals (OpenBB) — DXY, VIX for macro-aware trading decisions
"""

import logging
import math
import statistics
import time
from typing import Dict, List, Optional, Tuple

import requests

try:
    import numpy as np
    import talib
    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False

try:
    from openbb import obb as _obb
    _HAS_OPENBB = True
except ImportError:
    _HAS_OPENBB = False

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "polymarket-crypto-bot/1.0"})
_TIMEOUT = 10


# ── 1. Multi-Timeframe Trend Alignment ────────────────────────────────────────

def multi_timeframe_trend(hourly_prices: List[float]) -> dict:
    """
    Analyze trend alignment across multiple timeframes.
    Uses TA-Lib EMA/MACD when available for more accurate signals.

    Returns alignment score [-1, +1]:
      +1.0 = all timeframes bullish (strong buy signal)
      -1.0 = all timeframes bearish (strong sell signal)
       0.0 = mixed signals (no trade)

    Timeframes:
      - Short:  EMA6 vs EMA12 (momentum)
      - Medium: EMA24 vs EMA48 (trend)
      - Long:   MACD histogram or 7d slope (macro direction)
    """
    if len(hourly_prices) < 168:
        if len(hourly_prices) < 25:
            return {"alignment": 0.0, "timeframes": {}, "confidence": 0.3}

    if _HAS_TALIB:
        return _multi_timeframe_talib(hourly_prices)
    return _multi_timeframe_fallback(hourly_prices)


def _multi_timeframe_talib(hourly_prices: List[float]) -> dict:
    """TA-Lib accelerated multi-timeframe analysis with EMA and MACD."""
    close = np.array(hourly_prices, dtype=np.float64)
    signals = {}

    # Short-term: EMA6 vs EMA12
    if len(close) >= 12:
        ema_6 = talib.EMA(close, timeperiod=6)
        ema_12 = talib.EMA(close, timeperiod=12)
        if not np.isnan(ema_6[-1]) and not np.isnan(ema_12[-1]):
            signals["short"] = 1 if ema_6[-1] > ema_12[-1] else -1

    # Medium-term: EMA24 vs EMA48
    if len(close) >= 48:
        ema_24 = talib.EMA(close, timeperiod=24)
        ema_48 = talib.EMA(close, timeperiod=48)
        if not np.isnan(ema_24[-1]) and not np.isnan(ema_48[-1]):
            signals["medium"] = 1 if ema_24[-1] > ema_48[-1] else -1

    # Long-term: MACD histogram direction (more responsive than linear slope)
    if len(close) >= 168:
        _macd, _signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        if not np.isnan(hist[-1]):
            signals["long"] = 1 if hist[-1] > 0 else -1

    if not signals:
        return {"alignment": 0.0, "timeframes": signals, "confidence": 0.3}

    alignment = sum(signals.values()) / len(signals)
    agree_pct = abs(alignment)
    confidence = 0.5 + agree_pct * 0.3

    return {
        "alignment": round(alignment, 3),
        "timeframes": signals,
        "confidence": round(confidence, 2),
    }


def _multi_timeframe_fallback(hourly_prices: List[float]) -> dict:
    """Pure-Python multi-timeframe analysis (no TA-Lib)."""
    signals = {}

    if len(hourly_prices) >= 12:
        sma_6 = statistics.mean(hourly_prices[-6:])
        sma_12 = statistics.mean(hourly_prices[-12:])
        signals["short"] = 1 if sma_6 > sma_12 else -1

    if len(hourly_prices) >= 48:
        sma_24 = statistics.mean(hourly_prices[-24:])
        sma_48 = statistics.mean(hourly_prices[-48:])
        signals["medium"] = 1 if sma_24 > sma_48 else -1

    if len(hourly_prices) >= 168:
        week_data = hourly_prices[-168:]
        slope = _linear_slope(week_data)
        signals["long"] = 1 if slope > 0 else -1

    if not signals:
        return {"alignment": 0.0, "timeframes": signals, "confidence": 0.3}

    alignment = sum(signals.values()) / len(signals)
    agree_pct = abs(alignment)
    confidence = 0.5 + agree_pct * 0.3

    return {
        "alignment": round(alignment, 3),
        "timeframes": signals,
        "confidence": round(confidence, 2),
    }


def _linear_slope(prices: List[float]) -> float:
    """Simple linear regression slope (least squares)."""
    n = len(prices)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(prices)
    numerator = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator > 0 else 0.0


# ── 2. Fear & Greed Index ─────────────────────────────────────────────────────

_FG_CACHE: Dict[str, Tuple[float, dict]] = {}
_FG_CACHE_TTL = 3600  # 1 hour cache


def get_fear_greed_index() -> Optional[dict]:
    """
    Fetch Bitcoin Fear & Greed Index from alternative.me (free, no key).
    Returns:
      value: 0-100 (0=Extreme Fear, 100=Extreme Greed)
      classification: "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed"
      signal: trading signal based on contrarian logic
    """
    cached = _FG_CACHE.get("latest")
    if cached and (time.time() - cached[0]) < _FG_CACHE_TTL:
        return cached[1]

    try:
        r = _SESSION.get(
            "https://api.alternative.me/fng/?limit=1&format=json",
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        entry = data.get("data", [{}])[0]
        value = int(entry.get("value", 50))
        classification = entry.get("value_classification", "Neutral")

        # Contrarian signal: extreme fear = buy opportunity, extreme greed = caution
        if value <= 20:
            signal = "strong_buy"    # Extreme fear — market oversold
            signal_mult = 1.3        # Boost probability of "above" bets
        elif value <= 35:
            signal = "buy"
            signal_mult = 1.1
        elif value >= 80:
            signal = "strong_sell"   # Extreme greed — market likely to correct
            signal_mult = 0.7        # Reduce probability of "above" bets
        elif value >= 65:
            signal = "sell"
            signal_mult = 0.9
        else:
            signal = "neutral"
            signal_mult = 1.0

        result = {
            "value": value,
            "classification": classification,
            "signal": signal,
            "signal_mult": signal_mult,
        }
        _FG_CACHE["latest"] = (time.time(), result)
        log.info("Fear & Greed Index: %d (%s) -> %s", value, classification, signal)
        return result
    except Exception as exc:
        log.debug("Fear & Greed fetch failed: %s", exc)
        return None


# ── 3. Market Regime Detection ────────────────────────────────────────────────

def detect_crypto_regime(hourly_prices: List[float]) -> str:
    """
    Classify current market regime using TA-Lib ATR and ADX when available:
      TRENDING   — strong directional move (good for momentum trades)
      RANGING    — sideways chop (good for mean-reversion / range bets)
      VOLATILE   — high vol, no clear direction (reduce position size)
      CRASH      — sharp drop (avoid new entries, consider exits)
    """
    if len(hourly_prices) < 48:
        return "UNKNOWN"

    recent = hourly_prices[-48:]
    current = recent[-1]

    # Crash detection first: >10% drop in 24h
    if len(hourly_prices) >= 24:
        price_24h_ago = hourly_prices[-24]
        drop_24h = (current - price_24h_ago) / price_24h_ago
        if drop_24h < -0.10:
            return "CRASH"

    # TA-Lib enhanced regime detection using ADX
    if _HAS_TALIB and len(hourly_prices) >= 48:
        close = np.array(hourly_prices[-48:], dtype=np.float64)
        high = np.array([max(hourly_prices[max(0, i - 1):i + 1]) for i in range(len(hourly_prices) - 48, len(hourly_prices))], dtype=np.float64)
        low = np.array([min(hourly_prices[max(0, i - 1):i + 1]) for i in range(len(hourly_prices) - 48, len(hourly_prices))], dtype=np.float64)
        adx = talib.ADX(high, low, close, timeperiod=14)
        adx_val = float(adx[-1]) if not np.isnan(adx[-1]) else None
        if adx_val is not None:
            # ADX > 25 = trending, ADX < 20 = ranging
            returns = [(recent[i] - recent[i - 1]) / recent[i - 1] for i in range(1, len(recent))]
            vol = statistics.stdev(returns) if len(returns) >= 2 else 0
            if adx_val > 30 and vol > 0.02:
                return "TRENDING"
            if adx_val < 20 and vol < 0.015:
                return "RANGING"
            if vol > 0.03 and adx_val < 25:
                return "VOLATILE"
            if adx_val > 25:
                return "TRENDING"
            return "RANGING"

    # Fallback: pure-Python regime detection
    returns = [(recent[i] - recent[i - 1]) / recent[i - 1] for i in range(1, len(recent))]
    vol = statistics.stdev(returns) if len(returns) >= 2 else 0

    price_range = max(recent) - min(recent)
    if price_range <= 0:
        return "RANGING"
    direction_strength = abs(recent[-1] - recent[0]) / price_range

    if vol > 0.03:
        if direction_strength < 0.3:
            return "VOLATILE"
        return "TRENDING"

    if direction_strength > 0.6:
        return "TRENDING"
    if direction_strength < 0.2 and vol < 0.015:
        return "RANGING"

    return "RANGING"


# ── 4. Correlation Guard ──────────────────────────────────────────────────────

# Historical correlation matrix (approximate)
_CRYPTO_CORRELATIONS = {
    ("bitcoin", "ethereum"): 0.85,
    ("bitcoin", "solana"): 0.75,
    ("bitcoin", "xrp"): 0.70,
    ("bitcoin", "dogecoin"): 0.65,
    ("bitcoin", "cardano"): 0.72,
    ("ethereum", "solana"): 0.80,
    ("ethereum", "xrp"): 0.65,
    ("ethereum", "dogecoin"): 0.60,
    ("ethereum", "cardano"): 0.75,
    ("solana", "cardano"): 0.70,
}

# Correlation threshold above which we block
_CORRELATION_BLOCK_THRESHOLD = 0.70


def check_crypto_correlation(
    new_asset: str, new_direction: str, open_crypto_positions: List[dict]
) -> Optional[str]:
    """
    Check if a new crypto trade is too correlated with existing positions.
    Returns blocking reason or None if OK.

    Rules:
      - Same asset + same direction = always blocked
      - Correlated assets (>0.70) + same direction = blocked
      - Opposite directions on correlated assets = allowed (natural hedge)
    """
    for pos in open_crypto_positions:
        pos_asset = pos.get("crypto_asset", "")
        pos_side = pos.get("side", "")

        # Same asset, same direction
        if pos_asset == new_asset:
            pos_direction = "above" if pos_side == "YES" else "below"
            if pos_direction == new_direction:
                return f"Already have {new_asset} {new_direction} position open"

        # Check correlation between different assets.
        # Sort the pair so the lookup key is order-independent, then bind it
        # to a 2-tuple so the type matches our correlation table.
        _sorted = sorted([new_asset, pos_asset])
        pair: Tuple[str, str] = (_sorted[0], _sorted[1])
        corr = _CRYPTO_CORRELATIONS.get(pair, 0.0)
        if corr >= _CORRELATION_BLOCK_THRESHOLD:
            # Same direction on correlated assets = blocked
            pos_direction = "above" if pos_side == "YES" else "below"
            if pos_direction == new_direction:
                return (
                    f"Correlated position: {pos_asset} ({corr:.0%} corr with {new_asset}) "
                    f"— same direction blocked"
                )

    return None


# ── 5. Volume Anomaly Detection ───────────────────────────────────────────────

def detect_volume_anomaly(volumes: List[float]) -> dict:
    """
    Detect unusual volume patterns.
    Returns:
      anomaly: bool — True if volume significantly above average
      multiplier: float — how much above average (e.g., 3.5x)
      signal: "high_conviction" / "normal" / "low_conviction"
    """
    if not volumes or len(volumes) < 24:
        return {"anomaly": False, "multiplier": 1.0, "signal": "normal"}

    recent_vol = statistics.mean(volumes[-6:])  # last 6 hours
    avg_vol = statistics.mean(volumes[-168:] if len(volumes) >= 168 else volumes)

    if avg_vol <= 0:
        return {"anomaly": False, "multiplier": 1.0, "signal": "normal"}

    multiplier = recent_vol / avg_vol

    if multiplier >= 3.0:
        return {"anomaly": True, "multiplier": round(multiplier, 1), "signal": "high_conviction"}
    if multiplier <= 0.3:
        return {"anomaly": True, "multiplier": round(multiplier, 1), "signal": "low_conviction"}
    return {"anomaly": False, "multiplier": round(multiplier, 1), "signal": "normal"}


# ── 6. Composite Signal Score ─────────────────────────────────────────────────

def compute_composite_signal(
    prob_above: float,
    direction: str,
    hourly_prices: List[float],
    volumes: Optional[List[float]] = None,
) -> dict:
    """
    Combine all signals into a single composite score for decision-making.
    Returns adjusted probability and confidence.

    Process:
      1. Multi-timeframe alignment -> direction confirmation
      2. Fear & Greed -> contrarian bias
      3. Regime -> confidence adjustment
      4. Volume -> conviction boost/reduction
    """
    # Base values
    adjusted_prob = prob_above
    confidence_adj = 1.0
    signals_used = []

    # 1. Multi-timeframe
    mtf = multi_timeframe_trend(hourly_prices)
    alignment = mtf["alignment"]
    if direction == "above":
        # Bullish alignment boosts "above" probability
        adjusted_prob *= (1.0 + alignment * 0.10)  # +/-10% max
    else:
        # Bearish alignment boosts "below" probability (= 1 - above)
        adjusted_prob *= (1.0 - alignment * 0.10)
    signals_used.append(f"MTF={alignment:+.2f}")

    # 2. Fear & Greed
    fg = get_fear_greed_index()
    if fg:
        if direction == "above":
            adjusted_prob *= fg["signal_mult"]
        else:
            # Inverse for "below" bets
            inv_mult = 2.0 - fg["signal_mult"]  # 0.7 -> 1.3 and vice versa
            adjusted_prob *= inv_mult
        signals_used.append(f"F&G={fg['value']}({fg['signal']})")

    # 3. Regime
    regime = detect_crypto_regime(hourly_prices)
    regime_conf = {
        "TRENDING": 1.0,    # Good for directional bets
        "RANGING": 0.85,    # OK but less predictable
        "VOLATILE": 0.60,   # High uncertainty
        "CRASH": 0.40,      # Very dangerous
        "UNKNOWN": 0.70,
    }
    confidence_adj *= regime_conf.get(regime, 0.70)
    signals_used.append(f"Regime={regime}")

    # 4. Volume anomaly
    if volumes:
        vol_signal = detect_volume_anomaly(volumes)
        if vol_signal["signal"] == "high_conviction":
            confidence_adj *= 1.15  # Volume confirms move
        elif vol_signal["signal"] == "low_conviction":
            confidence_adj *= 0.80  # Low volume = less confidence
        signals_used.append(f"Vol={vol_signal['multiplier']:.1f}x")

    # 5. TA-Lib extended indicators (MACD, Bollinger %B) if available
    if _HAS_TALIB and len(hourly_prices) >= 30:
        close = np.array(hourly_prices, dtype=np.float64)
        # MACD histogram as trend confirmation
        _, _, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        if not np.isnan(macd_hist[-1]):
            macd_val = float(macd_hist[-1])
            if direction == "above" and macd_val > 0:
                adjusted_prob *= 1.05
                signals_used.append("MACD+")
            elif direction == "below" and macd_val < 0:
                adjusted_prob *= 1.05
                signals_used.append("MACD+")
            elif (direction == "above" and macd_val < 0) or (direction == "below" and macd_val > 0):
                confidence_adj *= 0.90
                signals_used.append("MACD-")

        # Bollinger %B — overbought/oversold confirmation
        bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]):
            bb_range = bb_upper[-1] - bb_lower[-1]
            if bb_range > 0:
                pct_b = (close[-1] - bb_lower[-1]) / bb_range
                if direction == "above" and pct_b < 0.2:
                    adjusted_prob *= 1.05  # oversold bounce
                    signals_used.append("BB_oversold")
                elif direction == "below" and pct_b > 0.8:
                    adjusted_prob *= 1.05  # overbought reversal
                    signals_used.append("BB_overbought")

    # 6. Macro economic signals (OpenBB / fallback)
    macro = get_macro_signals()
    if macro:
        crypto_bias = macro.get("crypto_bias", 1.0)
        if direction == "above":
            adjusted_prob *= crypto_bias
        else:
            adjusted_prob *= (2.0 - crypto_bias)  # inverse for below
        vix_str = f"VIX={macro.get('vix_value', '?')}({macro.get('vix_level', '?')})"
        dxy_str = f"DXY={macro.get('dxy_trend', '?')}"
        signals_used.append(f"Macro:{dxy_str},{vix_str}")

    # Clamp
    adjusted_prob = max(0.01, min(0.99, adjusted_prob))
    confidence_adj = max(0.3, min(1.2, confidence_adj))

    return {
        "adjusted_prob": round(adjusted_prob, 4),
        "confidence_multiplier": round(confidence_adj, 3),
        "regime": regime,
        "signals": signals_used,
        "mtf_alignment": alignment,
        "fear_greed": fg.get("value") if fg else None,
        "macro_bias": macro.get("crypto_bias") if macro else None,
    }


# ── 7. Option Pricing Model & Volatility Safeguard (IV vs HV) ──────────────────

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


def confirm_crypto_option_edge(
    current_price: float,
    threshold: float,
    direction: str,
    hours_to_expiry: float,
    daily_volatility: float,
    yes_market_price: float,
    min_edge_threshold: float = 0.02,
) -> dict:
    """
    Quant Option Pricing Model to verify underpriced YES shares.
    Compares Implied Volatility (IV) from Polymarket YES price
    with Historical Volatility (HV) from recent asset returns.

    Returns:
        dict: {
            "theoretical_yes_price": float,
            "implied_vol": float,
            "historical_vol": float,
            "has_underpriced_edge": bool,
            "vol_ratio": float,
            "price_edge": float
        }
    """
    if current_price <= 0 or threshold <= 0 or hours_to_expiry <= 0 or daily_volatility <= 0:
        return {
            "theoretical_yes_price": yes_market_price,
            "implied_vol": daily_volatility,
            "historical_vol": daily_volatility,
            "has_underpriced_edge": False,
            "vol_ratio": 1.0,
            "price_edge": 0.0,
        }

    # Time to expiry in days
    T = max(0.01, hours_to_expiry / 24.0)
    sigma_hv = daily_volatility

    # 1. Calculate theoretical price under Black-Scholes binary option model (r = 0)
    # Price of Binary Call (above) = N(d2), where d2 = (ln(S0/K) - 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    # We define log_ratio = ln(K/S0) = -ln(S0/K), so d2 = (-log_ratio - 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    log_ratio = math.log(threshold / current_price)
    sigma_T = sigma_hv * math.sqrt(T)

    if sigma_T > 0:
        d2 = (-log_ratio - 0.5 * (sigma_hv ** 2) * T) / sigma_T
        p_theo_above = 0.5 * (1.0 + _erf_approx(d2 / 1.4142135623730951))
    else:
        p_theo_above = 0.99 if current_price > threshold else 0.01

    p_theo_above = max(0.01, min(0.99, p_theo_above))
    theoretical_yes_price = p_theo_above if direction == "above" else (1.0 - p_theo_above)

    # 2. Bisection Solver to estimate Implied Volatility (IV) from yes_market_price
    # We solve for vol that yields yes_market_price
    target_price = max(0.001, min(0.999, yes_market_price))

    def price_from_vol(vol: float) -> float:
        vol = max(0.0001, vol)
        s_T = vol * math.sqrt(T)
        d2_val = (-log_ratio - 0.5 * (vol ** 2) * T) / s_T
        p_above = 0.5 * (1.0 + _erf_approx(d2_val / 1.4142135623730951))
        return p_above if direction == "above" else (1.0 - p_above)

    low_vol, high_vol = 0.0001, 10.0
    sigma_iv = daily_volatility  # fallback
    best_diff = float("inf")

    p_low = price_from_vol(low_vol)
    p_high = price_from_vol(high_vol)

    if target_price <= min(p_low, p_high):
        sigma_iv = low_vol
    elif target_price >= max(p_low, p_high):
        sigma_iv = high_vol
    else:
        for _ in range(30):
            mid_vol = (low_vol + high_vol) / 2.0
            p_mid = price_from_vol(mid_vol)
            diff = abs(p_mid - target_price)
            if diff < best_diff:
                best_diff = diff
                sigma_iv = mid_vol

            # Since binary option price is monotonic with respect to volatility
            # (depending on whether S0 > K or S0 < K), we can safely use the range endpoints
            if (p_mid > target_price) == (p_high > p_low):
                high_vol = mid_vol
                p_high = p_mid
            else:
                low_vol = mid_vol
                p_low = p_mid

    # 3. Verify underpriced edge and cheap volatility (IV vs HV)
    price_edge = theoretical_yes_price - yes_market_price
    vol_ratio = sigma_iv / sigma_hv if sigma_hv > 0 else 1.0

    # Underpriced YES shares condition:
    # A) Theoretical YES price is greater than actual YES market price by the minimum edge
    # B) AND Implied Volatility (IV) <= Historical Volatility (HV) * 1.05 (confirms cheap volatility safeguard)
    has_underpriced_edge = (price_edge >= min_edge_threshold) and (sigma_iv <= sigma_hv * 1.05)

    log.debug(
        "Option pricing check: S0=%.2f K=%.2f T=%.2fd HV=%.2f%% IV=%.2f%% "
        "TheoYES=%.3f MktYES=%.3f Edge=%.3f Underpriced=%s",
        current_price, threshold, T, sigma_hv * 100, sigma_iv * 100,
        theoretical_yes_price, yes_market_price, price_edge, has_underpriced_edge
    )

    return {
        "theoretical_yes_price": round(theoretical_yes_price, 4),
        "implied_vol": round(sigma_iv, 4),
        "historical_vol": round(sigma_hv, 4),
        "has_underpriced_edge": has_underpriced_edge,
        "vol_ratio": round(vol_ratio, 3),
        "price_edge": round(price_edge, 4),
    }


# ── 8. Macro Economic Signals (OpenBB) ──────────────────────────────────────

_MACRO_CACHE: Dict[str, Tuple[float, dict]] = {}
_MACRO_CACHE_TTL = 3600  # 1 hour cache


def get_macro_signals() -> Optional[dict]:
    """
    Fetch macro economic indicators via OpenBB Platform.
    Falls back to free API sources when OpenBB is unavailable.

    Returns:
      dxy_trend: "strengthening" / "weakening" / "neutral"
      vix_level: "low" / "normal" / "elevated" / "extreme"
      crypto_bias: multiplier [0.8, 1.2] for crypto probability adjustment
        - Strong dollar + high VIX = bearish for crypto (0.85)
        - Weak dollar + low VIX = bullish for crypto (1.15)
    """
    cached = _MACRO_CACHE.get("latest")
    if cached and (time.time() - cached[0]) < _MACRO_CACHE_TTL:
        return cached[1]

    result = None
    if _HAS_OPENBB:
        result = _fetch_macro_openbb()
    if result is None:
        result = _fetch_macro_fallback()

    if result:
        _MACRO_CACHE["latest"] = (time.time(), result)
    return result


def _fetch_macro_openbb() -> Optional[dict]:
    """Fetch DXY and VIX from OpenBB Platform."""
    try:
        # DXY (US Dollar Index) — recent close data
        dxy_data = _obb.index.price.historical(symbol="DXY", provider="yfinance", period="1mo")
        dxy_df = dxy_data.to_df() if hasattr(dxy_data, "to_df") else None

        dxy_trend = "neutral"
        if dxy_df is not None and len(dxy_df) >= 5:
            dxy_recent = dxy_df["close"].iloc[-5:].tolist()
            dxy_5d_change = (dxy_recent[-1] - dxy_recent[0]) / dxy_recent[0]
            if dxy_5d_change > 0.005:
                dxy_trend = "strengthening"
            elif dxy_5d_change < -0.005:
                dxy_trend = "weakening"

        # VIX (Volatility Index) — current level
        vix_data = _obb.index.price.historical(symbol="^VIX", provider="yfinance", period="5d")
        vix_df = vix_data.to_df() if hasattr(vix_data, "to_df") else None

        vix_level = "normal"
        vix_value = 20.0
        if vix_df is not None and len(vix_df) >= 1:
            vix_value = float(vix_df["close"].iloc[-1])
            if vix_value < 15:
                vix_level = "low"
            elif vix_value < 25:
                vix_level = "normal"
            elif vix_value < 35:
                vix_level = "elevated"
            else:
                vix_level = "extreme"

        # Crypto bias: inverse relationship with dollar strength and volatility
        crypto_bias = 1.0
        if dxy_trend == "strengthening":
            crypto_bias *= 0.93
        elif dxy_trend == "weakening":
            crypto_bias *= 1.07
        if vix_level == "extreme":
            crypto_bias *= 0.85
        elif vix_level == "elevated":
            crypto_bias *= 0.93
        elif vix_level == "low":
            crypto_bias *= 1.08

        crypto_bias = max(0.8, min(1.2, crypto_bias))

        log.info("OpenBB Macro: DXY=%s VIX=%.1f(%s) crypto_bias=%.2f",
                 dxy_trend, vix_value, vix_level, crypto_bias)

        return {
            "dxy_trend": dxy_trend,
            "vix_level": vix_level,
            "vix_value": round(vix_value, 1),
            "crypto_bias": round(crypto_bias, 3),
            "source": "openbb",
        }
    except Exception as exc:
        log.debug("OpenBB macro fetch failed: %s", exc)
        return None


def _fetch_macro_fallback() -> Optional[dict]:
    """Lightweight fallback: VIX from Yahoo Finance public chart API."""
    try:
        r = _SESSION.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
            params={"interval": "1d", "range": "5d"},
            headers={"User-Agent": "polymarket-bot/1.0"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        closes = (
            data.get("chart", {}).get("result", [{}])[0]
            .get("indicators", {}).get("quote", [{}])[0]
            .get("close", [])
        )
        closes = [c for c in closes if c is not None]
        if not closes:
            return None

        vix_value = closes[-1]
        if vix_value < 15:
            vix_level = "low"
        elif vix_value < 25:
            vix_level = "normal"
        elif vix_value < 35:
            vix_level = "elevated"
        else:
            vix_level = "extreme"

        crypto_bias = 1.0
        if vix_level == "extreme":
            crypto_bias = 0.85
        elif vix_level == "elevated":
            crypto_bias = 0.93
        elif vix_level == "low":
            crypto_bias = 1.08

        log.info("Macro fallback: VIX=%.1f(%s) crypto_bias=%.2f",
                 vix_value, vix_level, crypto_bias)

        return {
            "dxy_trend": "neutral",
            "vix_level": vix_level,
            "vix_value": round(vix_value, 1),
            "crypto_bias": round(crypto_bias, 3),
            "source": "yahoo_fallback",
        }
    except Exception as exc:
        log.debug("Macro fallback fetch failed: %s", exc)
        return None

