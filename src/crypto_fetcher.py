"""
Crypto price data fetcher — multi-source ensemble for price predictions.

Sources (via CCXT unified interface + direct API fallbacks):
  CCXT exchanges (configurable, default: binance, kraken, coinbasepro, okx):
    - Unified OHLCV, ticker, and orderbook via ccxt library
    - 100+ exchanges supported through a single interface
  Direct API fallbacks (used when CCXT is unavailable):
    1. CoinGecko API    — free, no key needed (30 req/min)
    2. Binance API      — free, no key needed (high-rate)
    3. CoinPaprika API  — free, no key needed

Returns consensus price data including:
  - current_price (USD)
  - price_24h_ago
  - volatility_24h (std dev of hourly returns)
  - momentum (trend direction & strength)
  - support/resistance levels
"""

import logging
import math
import statistics
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "polymarket-crypto-bot/1.0"})
_TIMEOUT = 10

try:
    import ccxt
    _HAS_CCXT = True
except ImportError:
    _HAS_CCXT = False

try:
    import numpy as np
    import talib
    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False


# ── Supported assets ──────────────────────────────────────────────────────────

SUPPORTED_ASSETS = {
    "bitcoin":  {"symbol": "BTC", "ccxt_symbol": "BTC/USDT", "coingecko_id": "bitcoin", "binance": "BTCUSDT"},
    "ethereum": {"symbol": "ETH", "ccxt_symbol": "ETH/USDT", "coingecko_id": "ethereum", "binance": "ETHUSDT"},
    "solana":   {"symbol": "SOL", "ccxt_symbol": "SOL/USDT", "coingecko_id": "solana", "binance": "SOLUSDT"},
    "xrp":      {"symbol": "XRP", "ccxt_symbol": "XRP/USDT", "coingecko_id": "ripple", "binance": "XRPUSDT"},
    "dogecoin": {"symbol": "DOGE", "ccxt_symbol": "DOGE/USDT", "coingecko_id": "dogecoin", "binance": "DOGEUSDT"},
    "cardano":  {"symbol": "ADA", "ccxt_symbol": "ADA/USDT", "coingecko_id": "cardano", "binance": "ADAUSDT"},
}

# Symbol aliases (for parsing market titles)
SYMBOL_ALIASES = {
    "btc": "bitcoin", "bitcoin": "bitcoin",
    "eth": "ethereum", "ethereum": "ethereum", "ether": "ethereum",
    "sol": "solana", "solana": "solana",
    "xrp": "xrp", "ripple": "xrp",
    "doge": "dogecoin", "dogecoin": "dogecoin",
    "ada": "cardano", "cardano": "cardano",
}

# Default CCXT exchanges (order = priority for OHLCV data)
_DEFAULT_CCXT_EXCHANGES = ["binance", "kraken", "okx"]


# ── CCXT unified fetcher ─────────────────────────────────────────────────────

def _init_ccxt_exchange(exchange_id: str) -> Optional[object]:
    if not _HAS_CCXT:
        return None
    try:
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            log.debug("CCXT exchange not found: %s", exchange_id)
            return None
        ex = exchange_class({"enableRateLimit": True, "timeout": _TIMEOUT * 1000})
        return ex
    except Exception as exc:
        log.debug("CCXT init failed for %s: %s", exchange_id, exc)
        return None


def _fetch_ccxt_ohlcv(
    exchange, ccxt_symbol: str, hours: int = 72
) -> Optional[dict]:
    """Fetch OHLCV data from any CCXT-compatible exchange."""
    try:
        since_ms = int((time.time() - hours * 3600) * 1000)
        ohlcv = exchange.fetch_ohlcv(
            ccxt_symbol, timeframe="1h", since=since_ms, limit=min(hours + 1, 1000)
        )
        if not ohlcv or len(ohlcv) < 2:
            return None
        # ohlcv: [[timestamp, open, high, low, close, volume], ...]
        closes = [float(c[4]) for c in ohlcv]
        highs = [float(c[2]) for c in ohlcv]
        lows = [float(c[3]) for c in ohlcv]
        volumes = [float(c[5]) for c in ohlcv]
        return {
            "source": f"ccxt_{exchange.id}",
            "current_price": closes[-1],
            "hourly_prices": closes,
            "highs": highs,
            "lows": lows,
            "volumes": volumes,
            "timestamp": time.time(),
        }
    except Exception as exc:
        log.debug("CCXT OHLCV fetch failed (%s): %s", exchange.id, exc)
        return None


def _fetch_ccxt_ticker(exchange, ccxt_symbol: str) -> Optional[dict]:
    """Fetch current ticker from any CCXT-compatible exchange."""
    try:
        ticker = exchange.fetch_ticker(ccxt_symbol)
        if not ticker or not ticker.get("last"):
            return None
        return {
            "source": f"ccxt_{exchange.id}",
            "current_price": float(ticker["last"]),
            "volume_24h": float(ticker.get("quoteVolume") or ticker.get("baseVolume") or 0),
            "pct_change_24h": float(ticker.get("percentage") or 0),
            "bid": float(ticker.get("bid") or 0),
            "ask": float(ticker.get("ask") or 0),
            "timestamp": time.time(),
        }
    except Exception as exc:
        log.debug("CCXT ticker fetch failed (%s): %s", exchange.id, exc)
        return None


# ── Source 1: CoinGecko ───────────────────────────────────────────────────────

def _fetch_coingecko(asset_id: str, hours: int = 72) -> Optional[dict]:
    """Fetch price history from CoinGecko (free, no key)."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{asset_id}/market_chart"
        params = {"vs_currency": "usd", "days": str(max(1, hours // 24 + 1))}
        r = _SESSION.get(url, params=params, timeout=_TIMEOUT)
        if r.status_code == 429:
            log.debug("CoinGecko rate limited")
            return None
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        if not prices:
            return None
        # prices = [[timestamp_ms, price], ...]
        current = prices[-1][1]
        hourly = [p[1] for p in prices[-(hours + 1):]]
        return {
            "source": "coingecko",
            "current_price": current,
            "hourly_prices": hourly,
            "timestamp": time.time(),
        }
    except Exception as exc:
        log.debug("CoinGecko fetch failed: %s", exc)
        return None


# ── Source 2: Binance ─────────────────────────────────────────────────────────

def _fetch_binance(symbol: str, hours: int = 72) -> Optional[dict]:
    """Fetch hourly klines from Binance (free, no key)."""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": "1h",
            "limit": min(hours + 1, 1000),
        }
        r = _SESSION.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        klines = r.json()
        if not klines:
            return None
        # klines: [open_time, open, high, low, close, volume, ...]
        closes = [float(k[4]) for k in klines]
        current = closes[-1]
        return {
            "source": "binance",
            "current_price": current,
            "hourly_prices": closes,
            "highs": [float(k[2]) for k in klines],
            "lows": [float(k[3]) for k in klines],
            "volumes": [float(k[5]) for k in klines],
            "timestamp": time.time(),
        }
    except Exception as exc:
        log.debug("Binance fetch failed: %s", exc)
        return None


# ── Source 3: CoinPaprika ─────────────────────────────────────────────────────

def _fetch_coinpaprika(asset_id: str) -> Optional[dict]:
    """Fetch current ticker from CoinPaprika (free)."""
    # CoinPaprika uses different IDs: "btc-bitcoin", "eth-ethereum" etc.
    paprika_map = {
        "bitcoin": "btc-bitcoin",
        "ethereum": "eth-ethereum",
        "solana": "sol-solana",
        "ripple": "xrp-xrp",
        "dogecoin": "doge-dogecoin",
        "cardano": "ada-cardano",
    }
    pid = paprika_map.get(asset_id)
    if not pid:
        return None
    try:
        url = f"https://api.coinpaprika.com/v1/tickers/{pid}"
        r = _SESSION.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        quotes = data.get("quotes", {}).get("USD", {})
        current = quotes.get("price")
        if current is None:
            return None
        return {
            "source": "coinpaprika",
            "current_price": float(current),
            "pct_change_24h": quotes.get("percent_change_24h", 0),
            "pct_change_7d": quotes.get("percent_change_7d", 0),
            "volume_24h": quotes.get("volume_24h", 0),
            "timestamp": time.time(),
        }
    except Exception as exc:
        log.debug("CoinPaprika fetch failed: %s", exc)
        return None


# ── Technical Analysis (TA-Lib accelerated with Python fallbacks) ─────────────

def _compute_volatility(hourly_prices: List[float]) -> float:
    """Daily volatility from hourly returns."""
    if len(hourly_prices) < 2:
        return 0.0
    returns = [
        (hourly_prices[i] - hourly_prices[i - 1]) / hourly_prices[i - 1]
        for i in range(1, len(hourly_prices))
        if hourly_prices[i - 1] > 0
    ]
    if len(returns) < 2:
        return 0.0
    hourly_std = statistics.stdev(returns)
    daily_vol = hourly_std * math.sqrt(24)
    return daily_vol


def _compute_momentum(hourly_prices: List[float]) -> dict:
    """
    Momentum indicators via TA-Lib (C-optimized) with Python fallback.
    Returns trend, strength, rsi, and extended indicators when TA-Lib is available:
      macd_signal, bollinger_pct, atr
    """
    if len(hourly_prices) < 25:
        return {"trend": 0, "strength": 0.5, "rsi": 50.0}

    if _HAS_TALIB:
        return _compute_momentum_talib(hourly_prices)
    return _compute_momentum_fallback(hourly_prices)


def _compute_momentum_talib(hourly_prices: List[float]) -> dict:
    """TA-Lib powered momentum with MACD, Bollinger Bands, ATR."""
    close = np.array(hourly_prices, dtype=np.float64)

    # SMA crossover
    sma_12 = talib.SMA(close, timeperiod=12)
    sma_24 = talib.SMA(close, timeperiod=24)
    sma_short = sma_12[-1]
    sma_long = sma_24[-1]

    if np.isnan(sma_short) or np.isnan(sma_long):
        trend, strength = 0, 0.0
    elif sma_short > sma_long:
        trend = 1
        strength = min(1.0, (sma_short - sma_long) / sma_long * 50)
    elif sma_short < sma_long:
        trend = -1
        strength = min(1.0, (sma_long - sma_short) / sma_long * 50)
    else:
        trend, strength = 0, 0.0

    # RSI (14-period)
    rsi_arr = talib.RSI(close, timeperiod=14)
    rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0

    # MACD (12, 26, 9)
    macd_line, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_val = float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0

    # Bollinger Bands (20, 2)
    bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_pct = 0.5
    if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]):
        bb_range = bb_upper[-1] - bb_lower[-1]
        if bb_range > 0:
            bb_pct = (close[-1] - bb_lower[-1]) / bb_range

    # EMA 50 for longer-term trend context
    ema_50 = talib.EMA(close, timeperiod=min(50, len(close) - 1))
    above_ema50 = bool(not np.isnan(ema_50[-1]) and close[-1] > ema_50[-1])

    return {
        "trend": trend,
        "strength": round(strength, 3),
        "rsi": round(rsi, 1),
        "macd_signal": round(macd_val, 4),
        "bollinger_pct": round(bb_pct, 3),
        "above_ema50": above_ema50,
    }


def _compute_momentum_fallback(hourly_prices: List[float]) -> dict:
    """Pure-Python momentum (no TA-Lib)."""
    sma_short = statistics.mean(hourly_prices[-12:])
    sma_long = statistics.mean(hourly_prices[-24:])

    if sma_short > sma_long:
        trend = 1
        strength = min(1.0, (sma_short - sma_long) / sma_long * 50)
    elif sma_short < sma_long:
        trend = -1
        strength = min(1.0, (sma_long - sma_short) / sma_long * 50)
    else:
        trend, strength = 0, 0.0

    rsi = _compute_rsi(hourly_prices[-15:])
    return {"trend": trend, "strength": round(strength, 3), "rsi": round(rsi, 1)}


def _compute_rsi(prices: List[float], period: int = 14) -> float:
    """Pure-Python RSI fallback (used when TA-Lib is unavailable)."""
    if len(prices) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = statistics.mean(gains[-period:]) if gains else 0
    avg_loss = statistics.mean(losses[-period:]) if losses else 0

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Average True Range — TA-Lib accelerated with Python fallback."""
    if _HAS_TALIB and len(highs) >= period + 1:
        h = np.array(highs, dtype=np.float64)
        l = np.array(lows, dtype=np.float64)
        c = np.array(closes, dtype=np.float64)
        atr = talib.ATR(h, l, c, timeperiod=period)
        val = atr[-1]
        return float(val) if not np.isnan(val) else 0.0
    # Python fallback
    if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
        return 0.0
    trs = []
    for i in range(1, min(len(highs), len(lows), len(closes))):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if not trs:
        return 0.0
    return statistics.mean(trs[-period:])


def _find_support_resistance(highs: List[float], lows: List[float]) -> dict:
    """Support/resistance from recent highs and lows."""
    if not highs or not lows:
        return {"support": None, "resistance": None}

    recent_highs = highs[-48:]
    recent_lows = lows[-48:]

    resistance = max(recent_highs) if recent_highs else None
    support = min(recent_lows) if recent_lows else None

    return {"support": support, "resistance": resistance}


# ── Probability estimation ────────────────────────────────────────────────────

def estimate_prob_above(
    current_price: float,
    threshold: float,
    hours_to_expiry: float,
    daily_volatility: float,
    momentum: dict,
) -> float:
    """
    Estimate P(price > threshold) at expiry using log-normal model + momentum bias.

    Based on geometric Brownian motion:
      ln(S_T/S_0) ~ N(mu*T, sigma*sqrt(T))
    where mu includes drift from momentum.
    """
    if current_price <= 0 or threshold <= 0:
        return 0.5

    # Time in days
    T = max(0.01, hours_to_expiry / 24.0)

    # Daily vol -> period vol
    sigma = daily_volatility * math.sqrt(T)
    if sigma <= 0:
        # No volatility data — use distance to threshold
        return 0.7 if current_price > threshold else 0.3

    # Drift: slight positive bias from momentum
    # momentum trend: +1/-1, strength: 0-1
    drift_daily = momentum.get("trend", 0) * momentum.get("strength", 0) * daily_volatility * 0.5
    mu = drift_daily * T

    # Log-normal: P(S_T > K) = P(ln(S_T/S_0) > ln(K/S_0))
    # = P(Z > (ln(K/S_0) - mu) / sigma)
    log_ratio = math.log(threshold / current_price)
    z = (log_ratio - mu) / sigma

    # Normal CDF approximation
    prob_above = 0.5 * (1.0 - _erf_approx(z / 1.4142135623730951))
    return max(0.01, min(0.99, prob_above))


def _erf_approx(x: float) -> float:
    """Abramowitz & Stegun erf approximation."""
    if not math.isfinite(x):
        return 1.0 if x > 0 else -1.0
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
           + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * math.exp(-x * x))


# ── Public API ────────────────────────────────────────────────────────────────

class CryptoEnsemble:
    """
    Multi-source crypto price fetcher with technical analysis.
    Uses CCXT unified interface for exchange data + direct API fallbacks.
    """

    def __init__(self, config: Optional[dict] = None):
        self._cache: Dict[str, dict] = {}
        self._cache_ttl = 300  # 5 minute cache
        self._ccxt_exchanges: List = []
        self._ccxt_initialized = False
        self._config = config or {}

    def _ensure_ccxt(self) -> None:
        """Lazy-init CCXT exchanges on first use."""
        if self._ccxt_initialized:
            return
        self._ccxt_initialized = True
        if not _HAS_CCXT:
            log.info("CCXT not installed — using direct API fallbacks only")
            return
        exchange_ids = (
            self._config.get("crypto", {}).get("ccxt_exchanges", _DEFAULT_CCXT_EXCHANGES)
        )
        for eid in exchange_ids:
            ex = _init_ccxt_exchange(eid)
            if ex:
                self._ccxt_exchanges.append(ex)
                log.debug("CCXT exchange ready: %s", eid)
        if self._ccxt_exchanges:
            log.info("CCXT initialized with %d exchanges: %s",
                     len(self._ccxt_exchanges),
                     ", ".join(e.id for e in self._ccxt_exchanges))

    def fetch(self, asset: str, hours_ahead: int = 72) -> Optional[dict]:
        """
        Fetch price data for an asset. Returns dict with:
          current_price, volatility, momentum, support, resistance, confidence
        """
        asset_lower = asset.lower()
        asset_key = SYMBOL_ALIASES.get(asset_lower, asset_lower)

        if asset_key not in SUPPORTED_ASSETS:
            log.debug("Unsupported crypto asset: %s", asset)
            return None

        # Check cache
        cached = self._cache.get(asset_key)
        if cached and (time.time() - cached.get("_ts", 0)) < self._cache_ttl:
            return cached

        self._ensure_ccxt()
        info = SUPPORTED_ASSETS[asset_key]
        sources = []

        # ── CCXT exchanges (primary — unified interface) ──
        ccxt_symbol = info.get("ccxt_symbol", f"{info['symbol']}/USDT")
        for ex in self._ccxt_exchanges:
            ohlcv_data = _fetch_ccxt_ohlcv(ex, ccxt_symbol, hours=hours_ahead)
            if ohlcv_data:
                sources.append(ohlcv_data)
                break  # one OHLCV source is enough for hourly data
        # Ticker from additional CCXT exchanges for price consensus
        for ex in self._ccxt_exchanges:
            if any(s["source"] == f"ccxt_{ex.id}" for s in sources):
                continue  # already have OHLCV from this exchange
            ticker = _fetch_ccxt_ticker(ex, ccxt_symbol)
            if ticker:
                sources.append(ticker)

        # ── Direct API fallbacks (when CCXT misses or unavailable) ──
        if not any(s.get("hourly_prices") for s in sources):
            # Need hourly data — try direct APIs
            cg = _fetch_coingecko(info["coingecko_id"], hours=hours_ahead)
            if cg:
                sources.append(cg)
            bn = _fetch_binance(info["binance"], hours=hours_ahead)
            if bn:
                sources.append(bn)
        else:
            # Already have hourly from CCXT — just add price sources for consensus
            cg = _fetch_coingecko(info["coingecko_id"], hours=hours_ahead)
            if cg:
                sources.append(cg)

        cp = _fetch_coinpaprika(info["coingecko_id"])
        if cp:
            sources.append(cp)

        if not sources:
            log.warning("All crypto sources failed for %s", asset)
            return None

        # Aggregate current price (median of sources)
        prices = [s["current_price"] for s in sources if s.get("current_price")]
        current_price = statistics.median(prices) if prices else None
        if current_price is None:
            return None

        # Use best hourly data (prefer CCXT OHLCV, then Binance, then CoinGecko)
        hourly = None
        highs = None
        lows = None
        for s in sources:
            if s.get("hourly_prices") and len(s["hourly_prices"]) > 20:
                hourly = s["hourly_prices"]
                highs = s.get("highs")
                lows = s.get("lows")
                break

        # Compute technical indicators
        volatility = _compute_volatility(hourly) if hourly else 0.03  # default 3% daily
        momentum = _compute_momentum(hourly) if hourly else {"trend": 0, "strength": 0, "rsi": 50}
        sr = _find_support_resistance(highs or [], lows or [])

        # Confidence: more sources = higher confidence
        confidence = min(1.0, 0.5 + len(sources) * 0.15)

        # Extract volumes (prefer CCXT source)
        volumes = None
        for s in sources:
            if s.get("volumes") and len(s["volumes"]) > 20:
                volumes = s["volumes"]
                break

        result = {
            "asset": asset_key,
            "symbol": info["symbol"],
            "current_price": round(current_price, 2),
            "volatility_daily": round(volatility, 4),
            "momentum": momentum,
            "support": sr.get("support"),
            "resistance": sr.get("resistance"),
            "confidence": round(confidence, 2),
            "sources_used": [s["source"] for s in sources],
            "source_count": len(sources),
            "hourly_prices": hourly or [],
            "volumes": volumes or [],
            "_ts": time.time(),
        }
        self._cache[asset_key] = result
        return result

    def get_probability(
        self, asset: str, threshold: float, direction: str, hours_to_expiry: float
    ) -> Optional[Tuple[float, float]]:
        """
        Returns (probability, confidence) for "will asset be above/below threshold?"

        direction: "above" or "below"
        """
        data = self.fetch(asset, hours_ahead=int(hours_to_expiry) + 24)
        if not data:
            return None

        prob_above = estimate_prob_above(
            current_price=data["current_price"],
            threshold=threshold,
            hours_to_expiry=hours_to_expiry,
            daily_volatility=data["volatility_daily"],
            momentum=data["momentum"],
        )

        if direction == "below":
            prob = 1.0 - prob_above
        else:
            prob = prob_above

        return (prob, data["confidence"])
