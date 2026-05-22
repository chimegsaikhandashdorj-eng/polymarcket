"""
Advanced crypto signal analysis — multi-timeframe, sentiment, and regime detection.

Provides three key signals:
  1. Multi-Timeframe Alignment — confirms trend across 1h, 4h, daily
  2. Fear & Greed Index — market sentiment (0=Extreme Fear, 100=Extreme Greed)
  3. Regime Detection — trending / ranging / volatile / crash
  4. Correlation Guard — prevents correlated BTC+ETH positions
  5. Volume Anomaly — unusual volume as conviction booster/reducer
"""

import logging
import math
import statistics
import time
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "polymarket-crypto-bot/1.0"})
_TIMEOUT = 10


# ── 1. Multi-Timeframe Trend Alignment ────────────────────────────────────────

def multi_timeframe_trend(hourly_prices: List[float]) -> dict:
    """
    Analyze trend alignment across multiple timeframes.
    Returns alignment score [-1, +1]:
      +1.0 = all timeframes bullish (strong buy signal)
      -1.0 = all timeframes bearish (strong sell signal)
       0.0 = mixed signals (no trade)

    Timeframes:
      - Short:  12h SMA vs 6h SMA (momentum)
      - Medium: 48h SMA vs 24h SMA (trend)
      - Long:   168h (7d) slope (macro direction)
    """
    if len(hourly_prices) < 168:
        # Not enough data for full analysis — use what we have
        if len(hourly_prices) < 25:
            return {"alignment": 0.0, "timeframes": {}, "confidence": 0.3}

    signals = {}

    # Short-term: 6h vs 12h
    if len(hourly_prices) >= 12:
        sma_6 = statistics.mean(hourly_prices[-6:])
        sma_12 = statistics.mean(hourly_prices[-12:])
        signals["short"] = 1 if sma_6 > sma_12 else -1

    # Medium-term: 24h vs 48h
    if len(hourly_prices) >= 48:
        sma_24 = statistics.mean(hourly_prices[-24:])
        sma_48 = statistics.mean(hourly_prices[-48:])
        signals["medium"] = 1 if sma_24 > sma_48 else -1

    # Long-term: 7-day linear regression slope
    if len(hourly_prices) >= 168:
        week_data = hourly_prices[-168:]
        slope = _linear_slope(week_data)
        signals["long"] = 1 if slope > 0 else -1

    if not signals:
        return {"alignment": 0.0, "timeframes": signals, "confidence": 0.3}

    # Alignment = average of all timeframe signals
    alignment = sum(signals.values()) / len(signals)
    # Confidence increases with more agreeing timeframes
    agree_pct = abs(alignment)
    confidence = 0.5 + agree_pct * 0.3  # [0.5, 0.8]

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
    Classify current market regime:
      TRENDING   — strong directional move (good for momentum trades)
      RANGING    — sideways chop (good for mean-reversion / range bets)
      VOLATILE   — high vol, no clear direction (reduce position size)
      CRASH      — sharp drop (avoid new entries, consider exits)

    Uses ATR-normalized price range + directional strength.
    """
    if len(hourly_prices) < 48:
        return "UNKNOWN"

    recent = hourly_prices[-48:]
    current = recent[-1]

    # 1. Volatility: std of hourly returns
    returns = [(recent[i] - recent[i - 1]) / recent[i - 1] for i in range(1, len(recent))]
    vol = statistics.stdev(returns) if len(returns) >= 2 else 0

    # 2. Directional strength: (end - start) / (max - min)
    price_range = max(recent) - min(recent)
    if price_range <= 0:
        return "RANGING"
    direction_strength = abs(recent[-1] - recent[0]) / price_range

    # 3. Crash detection: >10% drop in 24h
    if len(hourly_prices) >= 24:
        price_24h_ago = hourly_prices[-24]
        drop_24h = (current - price_24h_ago) / price_24h_ago
        if drop_24h < -0.10:
            return "CRASH"

    # 4. Classify
    if vol > 0.03:  # Very high hourly volatility
        if direction_strength < 0.3:
            return "VOLATILE"  # High vol, no direction
        return "TRENDING"      # High vol with direction = strong trend

    if direction_strength > 0.6:
        return "TRENDING"
    if direction_strength < 0.2 and vol < 0.015:
        return "RANGING"

    return "RANGING"  # default to cautious


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

        # Check correlation between different assets
        pair = tuple(sorted([new_asset, pos_asset]))
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
    }
