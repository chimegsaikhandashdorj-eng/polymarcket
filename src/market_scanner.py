"""
Polymarket market scanner.

Uses two Polymarket endpoints:
  - Gamma API  (https://gamma-api.polymarket.com/markets)  — metadata, volume
  - CLOB API   (https://clob.polymarket.com/markets)       — YES/NO prices

Filters for weather-related markets, drops thin liquidity, and parses
the market title to extract: city, event type (RAIN/TEMP_ABOVE/etc.),
and threshold value.
"""

import collections
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "polymarket-weather-bot/1.0"})

GAMMA_API    = "https://gamma-api.polymarket.com"
CLOB_API     = "https://clob.polymarket.com"

_TIMEOUT     = 15
_MAX_RETRIES = 3

# Canonical UTC-safe ISO parser is exposed at the package root so every
# submodule normalizes Polymarket / CLOB timestamps the same way.
from . import parse_utc_isoformat  # noqa: E402


def _safe_get(url: str, params: dict = None) -> Optional[Union[dict, list]]:
    """HTTP GET with exponential-backoff retry. 4xx errors fail immediately."""
    for attempt in range(_MAX_RETRIES):
        try:
            r = _SESSION.get(url, params=params, timeout=_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status in (400, 401, 403, 404):
                log.warning("MarketScanner HTTP %d for %s — not retrying", status, url.split("?")[0])
                return None
            if attempt < _MAX_RETRIES - 1:
                wait = 2.0 ** attempt
                log.warning("MarketScanner HTTP %d, retry in %.0fs (%s)", status, wait, url.split("?")[0])
                time.sleep(wait)
            else:
                log.warning("MarketScanner: all retries exhausted for %s", url.split("?")[0])
                return None
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2.0 ** attempt)
            else:
                log.warning("MarketScanner request failed %s: %s", url.split("?")[0], type(exc).__name__)
                return None
    return None


# ── City name -> lat/lon lookup built from config ───────────────────────────────

_CITY_ALIASES: Dict[str, str] = {
    # Unambiguous multi-char aliases only — avoid short strings like "la", "ny", "chi"
    # that appear as substrings in common English words (landfall, any, chicago).
    "nyc": "new york",
    "new york city": "new york",
    "manhattan": "new york",
    "los angeles ca": "los angeles",
    "chicago il": "chicago",
    "miami fl": "miami",
    "south beach": "miami",
    "houston tx": "houston",
    "phoenix az": "phoenix",
    "philly": "philadelphia",
    "san fran": "san francisco",
    "san francisco ca": "san francisco",
    "dallas tx": "dallas",
    "dfw": "dallas",
    "austin tx": "austin",
    "san diego ca": "san diego",
    "seattle wa": "seattle",
    "denver co": "denver",
    "washington dc": "washington",
    "washington d.c.": "washington",
    "boston ma": "boston",
    "atlanta ga": "atlanta",
    "las vegas nv": "las vegas",
    "vegas": "las vegas",
    "portland or": "portland",
    "minneapolis mn": "minneapolis",
    "detroit mi": "detroit",
    "nashville tn": "nashville",
    "charlotte nc": "charlotte",
    "orlando fl": "orlando",
    "london uk": "london",
    "london england": "london",
    "paris france": "paris",
    "toronto canada": "toronto",
}


class CityIndex:
    def __init__(self, cities: List[dict]):
        # Build lookup: lowercase city name -> {lat, lon, country}
        self._index: Dict[str, dict] = {}
        for c in cities:
            key = c["name"].lower()
            self._index[key] = {"lat": c["lat"], "lon": c["lon"], "country": c["country"]}
        # Register aliases that point to known canonical names
        for alias, canonical in _CITY_ALIASES.items():
            if canonical in self._index and alias not in self._index:
                self._index[alias] = self._index[canonical]

    def match(self, title: str) -> Optional[Tuple[str, dict]]:
        """Return (city_name, {lat, lon, country}) for the first city found in title."""
        title_lower = title.lower()
        # Try multi-word names first (longer match wins), then aliases
        for name in sorted(self._index.keys(), key=len, reverse=True):
            if name in title_lower:
                # Return the canonical city name (strip alias suffixes like "nyc" -> "new york")
                canonical = _CITY_ALIASES.get(name, name)
                coords = self._index.get(canonical, self._index[name])
                return canonical, coords
        return None


# ── Market title parsing ───────────────────────────────────────────────────────

# Event types we handle
RAIN            = "RAIN"
SNOW            = "SNOW"
TEMP_ABOVE      = "TEMP_ABOVE"       # legacy: "will it exceed X°F today?"
TEMP_BELOW      = "TEMP_BELOW"       # legacy: "will it stay below X°F today?"
WIND_ABOVE      = "WIND_ABOVE"
# Temperature bucket market types (Polymarket neg-risk daily max series)
TEMP_ABOVE_MAX  = "TEMP_ABOVE_MAX"   # "X°F or higher" — top bucket (daily max)
TEMP_BELOW_MAX  = "TEMP_BELOW_MAX"   # "X°F or below"  -- bottom bucket (daily max)
TEMP_RANGE      = "TEMP_RANGE"       # "between X-Y°F" -- middle bucket (daily max)

# Crypto metric types
CRYPTO_ABOVE    = "CRYPTO_ABOVE"     # "Will BTC be above $X?"
CRYPTO_BELOW    = "CRYPTO_BELOW"     # "Will ETH be below $X?"
CRYPTO_RANGE    = "CRYPTO_RANGE"     # "Will BTC close between $X-$Y?"

# Crypto title patterns
_CRYPTO_ABOVE_RE = re.compile(
    r"\b(bitcoin|btc|ethereum|eth|solana|sol|xrp|ripple|dogecoin|doge|cardano|ada)\b"
    r".*?\b(?:above|over|exceed|hit|reach|surpass|higher than|at least|or more|or higher)"
    r".*?\$?([\d,]+(?:\.\d+)?)",
    re.I,
)
_CRYPTO_BELOW_RE = re.compile(
    r"\b(bitcoin|btc|ethereum|eth|solana|sol|xrp|ripple|dogecoin|doge|cardano|ada)\b"
    r".*?\b(?:below|under|less than|lower than|at most|or less|or lower|not reach|not exceed|drop)"
    r".*?\$?([\d,]+(?:\.\d+)?)",
    re.I,
)
# Reverse pattern: "$100,000" before coin name
_CRYPTO_PRICE_FIRST_RE = re.compile(
    r"\$?([\d,]+(?:\.\d+)?)\s*"
    r".*?\b(bitcoin|btc|ethereum|eth|solana|sol|xrp|ripple|dogecoin|doge|cardano|ada)\b",
    re.I,
)


_RAIN_RE  = re.compile(r"\brain\b|\bprecip|\brainfall|\bshower", re.I)
_SNOW_RE  = re.compile(r"\bsnow\b|\bblizzard\b|\bsnowfall\b|\bsnowstorm\b", re.I)
_WIND_RE  = re.compile(r"\bwind.*?(\d+)\s*(?:mph|km/h|kmh|kph)\b", re.I)

# Bucket-market patterns (checked before generic temp patterns)
# "between 17-18°F" or "between 17 and 18°F" or "between 17-18 F"
_TEMP_RANGE_RE = re.compile(
    r"between\s+(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*[°]?\s*([FC])\b"
    r"|between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*[°]?\s*([FC])\b",
    re.I,
)
# "27°F or higher" / "52°F or above" / "52 F or more"
_TEMP_ABOVE_MAX_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[°]?\s*([FC])\s+or\s+(?:higher|above|more)\b",
    re.I,
)
# "16°F or below" / "41 F or lower" / "X°F or less"
_TEMP_BELOW_MAX_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[°]?\s*([FC])\s+or\s+(?:below|lower|less)\b",
    re.I,
)
# Context indicator: "highest temperature" → daily max market
_HIGHEST_TEMP_RE = re.compile(
    r"\bhighest temperature\b|\bdaily max\b|\bdaily high\b|\bmax temp\b",
    re.I,
)

# Legacy generic temp (used only when bucket patterns don't match)
_TEMP_RE  = re.compile(
    r"(?:temperature|temp).*?(\d+(?:\.\d+)?)\s*(?:°|degrees?)?\s*([fc])\b"
    r"|(\d+(?:\.\d+)?)\s*(?:°|degrees?)?\s*([fc])\b"
    r"|(\d+(?:\.\d+)?)\s*degrees?\s+(?:fahrenheit|celsius)",
    re.I,
)
_ABOVE_RE = re.compile(
    r"\bexceed|\babove|\bover\b|\bsurpass|\bat least|\bor more|\breach|\bhit\b"
    r"|\bhigher than|\bwarmer than|\bgreater than",
    re.I,
)
_BELOW_RE = re.compile(
    r"\bbelow|\bunder\b|\bnot reach|\bno more than|\bat most|\bor less"
    r"|\bor lower|\bnot exceed|\bcolder than|\blower than",
    re.I,
)


def parse_market_condition(title: str) -> dict:
    """
    Attempt to extract event type and numeric threshold from a market title.

    Returns dict with keys:
      metric          -- one of RAIN / SNOW / TEMP_ABOVE / TEMP_BELOW / WIND_ABOVE /
                         TEMP_RANGE / TEMP_ABOVE_MAX / TEMP_BELOW_MAX /
                         CRYPTO_ABOVE / CRYPTO_BELOW  (or None)
      threshold       -- numeric value for single-threshold metrics
      threshold_low   -- lower bound for TEMP_RANGE
      threshold_high  -- upper bound for TEMP_RANGE
      threshold_unit  -- "F", "C", "mph", "kph", "USD"
      crypto_asset    -- (crypto only) "bitcoin", "ethereum", etc.
    """
    result = {
        "metric": None,
        "threshold": None,
        "threshold_low": None,
        "threshold_high": None,
        "threshold_unit": None,
        "crypto_asset": None,
    }

    # ── Crypto markets (check first — before weather) ─────────────────────────
    m = _CRYPTO_ABOVE_RE.search(title)
    if m:
        from .crypto_fetcher import SYMBOL_ALIASES
        asset = SYMBOL_ALIASES.get(m.group(1).lower())
        if asset:
            result["metric"] = CRYPTO_ABOVE
            result["threshold"] = float(m.group(2).replace(",", ""))
            result["threshold_unit"] = "USD"
            result["crypto_asset"] = asset
            return result

    m = _CRYPTO_BELOW_RE.search(title)
    if m:
        from .crypto_fetcher import SYMBOL_ALIASES
        asset = SYMBOL_ALIASES.get(m.group(1).lower())
        if asset:
            result["metric"] = CRYPTO_BELOW
            result["threshold"] = float(m.group(2).replace(",", ""))
            result["threshold_unit"] = "USD"
            result["crypto_asset"] = asset
            return result

    # Reverse pattern: "$100,000 Bitcoin"
    m = _CRYPTO_PRICE_FIRST_RE.search(title)
    if m:
        from .crypto_fetcher import SYMBOL_ALIASES
        asset = SYMBOL_ALIASES.get(m.group(2).lower())
        if asset:
            price_val = float(m.group(1).replace(",", ""))
            if _ABOVE_RE.search(title):
                result["metric"] = CRYPTO_ABOVE
            elif _BELOW_RE.search(title):
                result["metric"] = CRYPTO_BELOW
            else:
                result["metric"] = CRYPTO_ABOVE  # default to above
            result["threshold"] = price_val
            result["threshold_unit"] = "USD"
            result["crypto_asset"] = asset
            return result

    # ── Snow / Rain / Wind (not affected by bucket format) ─────────────────────
    if _SNOW_RE.search(title) and not _HIGHEST_TEMP_RE.search(title):
        result["metric"] = SNOW
        return result

    if _RAIN_RE.search(title) and not _HIGHEST_TEMP_RE.search(title):
        result["metric"] = RAIN
        return result

    m = _WIND_RE.search(title)
    if m and not _HIGHEST_TEMP_RE.search(title):
        result["metric"] = WIND_ABOVE
        result["threshold"] = float(m.group(1))
        result["threshold_unit"] = "kph" if "km" in m.group(0).lower() else "mph"
        return result

    # ── Temperature bucket patterns (check before legacy temp) ─────────────────
    # TEMP_RANGE: "between X-Y°F" or "between X and Y°F"
    m = _TEMP_RANGE_RE.search(title)
    if m:
        if m.group(1) is not None:           # dash format: "between 17-18°F"
            lo, hi, unit = m.group(1), m.group(2), m.group(3)
        else:                                 # "and" format: "between 17 and 18°F"
            lo, hi, unit = m.group(4), m.group(5), m.group(6)
        result["metric"] = TEMP_RANGE
        result["threshold_low"] = float(lo)
        result["threshold_high"] = float(hi)
        result["threshold_unit"] = unit.upper()
        return result

    # TEMP_ABOVE_MAX: "27°F or higher"
    m = _TEMP_ABOVE_MAX_RE.search(title)
    if m:
        result["metric"] = TEMP_ABOVE_MAX
        result["threshold"] = float(m.group(1))
        result["threshold_unit"] = m.group(2).upper()
        return result

    # TEMP_BELOW_MAX: "16°F or below"
    m = _TEMP_BELOW_MAX_RE.search(title)
    if m:
        result["metric"] = TEMP_BELOW_MAX
        result["threshold"] = float(m.group(1))
        result["threshold_unit"] = m.group(2).upper()
        return result

    # ── Legacy single-threshold temperature ────────────────────────────────────
    m = _TEMP_RE.search(title)
    if m:
        val = m.group(1) or m.group(3) or m.group(5)
        if m.group(2) or m.group(4):
            unit = (m.group(2) or m.group(4)).upper()
        else:
            unit = "C" if "celsius" in m.group(0).lower() else "F"
        result["threshold"] = float(val)
        result["threshold_unit"] = unit
        result["metric"] = TEMP_ABOVE if _ABOVE_RE.search(title) else TEMP_BELOW
        return result

    return result


def _parse_date_from_title(title: str) -> Optional[datetime]:
    """Extract a date from common patterns in market titles."""
    patterns = [
        r"\b(\w+ \d{1,2},?\s*\d{4})\b",           # "May 5, 2025"
        r"\b(\w+ \d{1,2}(?:st|nd|rd|th)?)\b",      # "May 5th"
        r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",          # "5/5/25"
        r"\b(\d{4}-\d{2}-\d{2})\b",                 # "2025-05-05"
    ]
    now = datetime.now(timezone.utc)
    for pat in patterns:
        m = re.search(pat, title)
        if m:
            raw = m.group(1)
            for fmt in ("%B %d, %Y", "%B %d %Y", "%B %d", "%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(raw.strip(), fmt)
                    if dt.year == 1900:
                        # Year not in pattern — guess: use current year, bump to next if in the past
                        dt = dt.replace(year=now.year)
                        if dt < now.replace(tzinfo=None) - timedelta(days=1):
                            dt = dt.replace(year=now.year + 1)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
    return None


# ── CLOB price fetcher ─────────────────────────────────────────────────────────

def _get_clob_midpoint(token_id: str) -> Optional[float]:
    """Return midpoint price for a YES token from CLOB (0-1 scale)."""
    data = _safe_get(f"{CLOB_API}/midpoint", params={"token_id": token_id})
    if not data:
        return None
    try:
        mid = data.get("mid")
        return float(mid) if mid is not None else None
    except (ValueError, TypeError):
        return None


def _get_clob_orderbook_price(token_id: str) -> Optional[float]:
    """Return midpoint price from CLOB orderbook for a YES token."""
    spread_data = _get_bid_ask_spread(token_id)
    return spread_data.get("mid")


def _get_bid_ask_spread(yes_token_id: str) -> dict:
    """
    Fetch full bid/ask data from CLOB orderbook.
    Returns {best_bid, best_ask, mid, spread} where spread = (ask-bid)/mid.
    All None on failure.
    """
    empty = {"best_bid": None, "best_ask": None, "mid": None, "spread": 0.0}
    data = _safe_get(f"{CLOB_API}/orderbook/{yes_token_id}")
    if not data:
        return empty
    try:
        bids = sorted(
            [float(b["price"]) for b in data.get("bids", []) if "price" in b],
            reverse=True,
        )
        asks = sorted(
            [float(a["price"]) for a in data.get("asks", []) if "price" in a],
        )
        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None

        if best_bid is not None and best_ask is not None:
            mid    = (best_bid + best_ask) / 2.0
            spread = (best_ask - best_bid) / mid if mid > 0 else 0.0
        elif best_ask is not None:
            mid, spread = best_ask, 0.0
        elif best_bid is not None:
            mid, spread = best_bid, 0.0
        else:
            return empty

        return {
            "best_bid": round(best_bid, 4) if best_bid else None,
            "best_ask": round(best_ask, 4) if best_ask else None,
            "mid":      round(mid, 4),
            "spread":   round(spread, 4),
        }
    except (KeyError, ValueError, TypeError):
        return empty


def _extract_yes_token(raw: dict) -> Optional[str]:
    """
    Extract the YES-side token ID from a Gamma API market record.
    Handles: list of strings, list of dicts, or JSON-encoded string.
    """
    import json

    clob_ids = raw.get("clobTokenIds")
    tokens   = raw.get("tokens")

    # Try clobTokenIds first — can be a JSON string or list
    if clob_ids:
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except (ValueError, TypeError):
                return None
        if isinstance(clob_ids, list) and clob_ids:
            val = clob_ids[0]
            if isinstance(val, str) and len(val) > 10:
                return val
            if isinstance(val, list) and val:
                return str(val[0])

    # Try tokens array — objects with tokenId/outcome fields
    if tokens:
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except (ValueError, TypeError):
                return None
        if isinstance(tokens, list):
            for t in tokens:
                if isinstance(t, dict):
                    outcome = (t.get("outcome") or "").lower()
                    if outcome in ("yes", ""):
                        tid = t.get("tokenId") or t.get("token_id")
                        if tid:
                            return str(tid)

    return None


def _extract_no_token(raw: dict) -> Optional[str]:
    """Extract the NO-side token ID (clobTokenIds[1]) from a Gamma API market record."""
    import json

    clob_ids = raw.get("clobTokenIds")
    tokens   = raw.get("tokens")

    if clob_ids:
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except (ValueError, TypeError):
                return None
        if isinstance(clob_ids, list) and len(clob_ids) > 1:
            val = clob_ids[1]
            if isinstance(val, str) and len(val) > 10:
                return val

    if tokens:
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except (ValueError, TypeError):
                return None
        if isinstance(tokens, list):
            for t in tokens:
                if isinstance(t, dict) and (t.get("outcome") or "").lower() == "no":
                    tid = t.get("tokenId") or t.get("token_id")
                    if tid:
                        return str(tid)

    return None


def _parse_gamma_price(raw: dict) -> Optional[float]:
    """
    Extract YES price directly from Gamma API fields.
    outcomePrices may be a list or a JSON-encoded string like '["0.75","0.25"]'.
    """
    import json

    # outcomePrices: ["0.75", "0.25"] — index 0 = YES
    op = raw.get("outcomePrices")
    if op:
        if isinstance(op, str):
            try:
                op = json.loads(op)
            except (ValueError, TypeError):
                pass
        if isinstance(op, list) and op:
            try:
                return float(op[0])
            except (ValueError, TypeError):
                pass

    # lastTradePrice: scalar
    ltp = raw.get("lastTradePrice")
    if ltp is not None:
        try:
            return float(ltp)
        except (ValueError, TypeError):
            pass

    return None


# ── Adversarial market detection ──────────────────────────────────────────────

_adv_log = logging.getLogger("adversarial")


def _setup_adversarial_logger() -> None:
    """Attach a file handler to the adversarial logger (idempotent)."""
    if _adv_log.handlers:
        return
    try:
        log_dir = Path(__file__).resolve().parent.parent / "data"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "adversarial.log")
        fh.setLevel(logging.WARNING)
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
        _adv_log.addHandler(fh)
        _adv_log.setLevel(logging.WARNING)
    except Exception:
        pass


class AdversarialDetector:
    """
    Detects potential market manipulation patterns across successive scans.
    Maintains a rolling per-market history and flags suspicious markets.

    Flags a market SUSPICIOUS when any condition is met:
      1. Price moved >10pp between consecutive scans
      2. Volume delta > 5× the rolling average of previous scan deltas
      3. Spread collapsed from >5% to <1% in a single scan interval

    Flagged markets block new entries for `cooldown_minutes`.
    This is a module-level singleton — persists state across all scan cycles.
    """

    _MAX_HISTORY = 8  # keep ~2h of scan history per market

    def __init__(self, config: dict) -> None:
        _setup_adversarial_logger()
        adv = config.get("adversarial", {})
        self.enabled               = adv.get("enabled", True)
        self.price_jump_threshold  = adv.get("price_jump_threshold", 0.10)
        self.vol_spike_mult        = adv.get("volume_spike_multiplier", 5.0)
        self.spread_hi             = adv.get("spread_collapse_high", 0.05)
        self.spread_lo             = adv.get("spread_collapse_low", 0.01)
        self.cooldown              = adv.get("cooldown_minutes", 30) * 60

        # {market_id: deque[(ts, price, volume, spread)]}
        self._history: Dict[str, collections.deque] = {}
        # {market_id: flagged_at_ts}
        self._flagged: Dict[str, float] = {}
        self._flag_count: int = 0

    def update_and_check(
        self, market_id: str, price: float, volume: float, spread: float
    ) -> bool:
        """
        Record a new market snapshot and return True if the market is suspicious.
        Call once per market per scan.
        """
        if not self.enabled:
            return False

        now = time.time()

        # Expire old cooldowns
        self._flagged = {
            mid: ts for mid, ts in self._flagged.items()
            if now - ts < self.cooldown
        }

        # Still in cooldown from a previous flag
        if market_id in self._flagged:
            return True

        if market_id not in self._history:
            self._history[market_id] = collections.deque(maxlen=self._MAX_HISTORY)

        hist = self._history[market_id]
        hist.append((now, price, volume, spread))

        if len(hist) < 2:
            return False  # need at least one prior snapshot to compare

        reason = self._detect(hist)
        if reason:
            self._flagged[market_id] = now
            self._flag_count += 1
            _adv_log.warning(
                "FLAG market=%s  %s  price=%.3f  vol=%.0f  spread=%.3f",
                market_id[:30], reason, price, volume, spread,
            )
            log.warning(
                "Adversarial flag on %s: %s — blocking for %dm",
                market_id[:30], reason, self.cooldown // 60,
            )
            try:
                from .notifier import notify_error
                notify_error(f"[ADVERSARIAL] {market_id[:30]}: {reason}")
            except Exception:
                pass
            return True

        return False

    def is_flagged(self, market_id: str) -> bool:
        now = time.time()
        return (
            market_id in self._flagged
            and now - self._flagged[market_id] < self.cooldown
        )

    def get_flag_count(self) -> int:
        return self._flag_count

    def _detect(self, hist: collections.deque) -> Optional[str]:
        points = list(hist)
        _, prev_price, prev_vol, prev_spread = points[-2]
        _, cur_price, cur_vol, cur_spread = points[-1]

        # Check 1: price jump > threshold between consecutive scans
        price_move = abs(cur_price - prev_price)
        if price_move > self.price_jump_threshold:
            return (
                f"price jump d{price_move:.3f} > {self.price_jump_threshold}"
                f" ({prev_price:.3f} -> {cur_price:.3f})"
            )

        # Check 2: volume delta spike (compare current delta to rolling average)
        if len(points) >= 3:
            prev_deltas = [
                max(0.0, points[i][2] - points[i - 1][2])
                for i in range(1, len(points) - 1)
            ]
            cur_delta = max(0.0, cur_vol - prev_vol)
            avg_delta = sum(prev_deltas) / len(prev_deltas) if prev_deltas else 0
            if avg_delta > 0 and cur_delta > self.vol_spike_mult * avg_delta:
                return (
                    f"volume spike: d{cur_delta:.0f} vs avg d{avg_delta:.0f}"
                    f" (x{cur_delta / avg_delta:.1f})"
                )

        # Check 3: spread collapse — was wide, now tight
        if prev_spread > self.spread_hi and cur_spread < self.spread_lo:
            return (
                f"spread collapse: {prev_spread:.3f} -> {cur_spread:.3f}"
                f" (threshold: >{self.spread_hi} -> <{self.spread_lo})"
            )

        return None


# Module-level singleton — must survive across scan calls
_ADVERSARIAL_DETECTOR: Optional[AdversarialDetector] = None


def get_adversarial_detector(config: dict) -> AdversarialDetector:
    global _ADVERSARIAL_DETECTOR
    if _ADVERSARIAL_DETECTOR is None:
        _ADVERSARIAL_DETECTOR = AdversarialDetector(config)
    return _ADVERSARIAL_DETECTOR


# ── Main scanner ───────────────────────────────────────────────────────────────

def _load_sports_blocklist(config_dir: Optional[Path] = None) -> List[re.Pattern]:
    """Load sports_blocklist.yaml and compile each phrase as a word-boundary pattern."""
    import yaml
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent.parent / "config"
    blocklist_path = config_dir / "sports_blocklist.yaml"
    if not blocklist_path.exists():
        return []
    try:
        with open(blocklist_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        phrases = data.get("blocked_phrases", [])
        patterns = []
        for phrase in phrases:
            phrase = str(phrase).strip()
            if not phrase:
                continue
            # Exact prefix match for "nhl:" style, word-boundary for others
            if phrase.endswith(":"):
                patterns.append(re.compile(re.escape(phrase), re.I))
            else:
                patterns.append(re.compile(r"\b" + re.escape(phrase) + r"\b", re.I))
        return patterns
    except Exception as exc:
        log.warning("Could not load sports blocklist: %s", exc)
        return []


class MarketScanner:
    def __init__(self, config: dict):
        cfg = config.get("markets", {})
        self.keywords: List[str] = cfg.get("weather_keywords", ["rain", "snow", "temperature"])
        # Compile whole-word patterns so "rain" doesn't match "Ukraine", etc.
        self._kw_patterns: List[re.Pattern] = [
            re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            for kw in self.keywords
        ]
        self.min_liquidity: float = cfg.get("min_liquidity_usdc", 5000)
        self.max_hours: int = cfg.get("max_hours_to_expiry", 168)
        self.min_hours: int = cfg.get("min_hours_to_expiry", 2)
        self.city_index = CityIndex(config.get("cities", []))
        self._detector = get_adversarial_detector(config)
        self._sports_blocklist: List[re.Pattern] = _load_sports_blocklist()

    def fetch_weather_markets(self) -> List[dict]:
        """
        Query Gamma API for active markets, filter for weather topics,
        enrich with CLOB prices, and return parsed market dicts.

        Uses TWO discovery strategies:
          1. Keyword matching on all active markets (broad)
          2. Tag-based fetch for "Weather" tagged markets (precise)
        Deduplicates by condition_id before parsing.
        """
        raw_markets = self._paginate_gamma()
        log.info("Gamma API returned %d total markets", len(raw_markets))

        # Strategy 1: keyword match
        matched = [m for m in raw_markets if self._is_weather_market(m)]
        log.info("%d weather-related markets found (keyword match)", len(matched))

        # Strategy 2: tag-based fetch — Gamma API supports tag_slug filter
        tag_markets = self._fetch_by_tag("weather")
        log.info("%d markets found via 'weather' tag", len(tag_markets))

        # Strategy 3: crypto tag-based fetch
        crypto_tag_markets = self._fetch_by_tag("crypto")
        log.info("%d markets found via 'crypto' tag", len(crypto_tag_markets))
        tag_markets = tag_markets + crypto_tag_markets

        # Deduplicate by condition_id
        seen_ids = set()
        combined = []
        for m in matched + tag_markets:
            cid = m.get("conditionId") or m.get("condition_id") or ""
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                combined.append(m)
            elif not cid:
                combined.append(m)
        log.info("%d unique weather markets after dedup", len(combined))

        results = []
        for raw in combined:
            parsed = self._parse_market(raw)
            if parsed:
                results.append(parsed)

        log.info("%d markets passed all filters", len(results))
        return results

    def _fetch_by_tag(self, tag_slug: str) -> List[dict]:
        """Fetch active markets with a specific tag from Gamma API."""
        markets = []
        offset = 0
        limit = 100
        max_pages = 10

        for _ in range(max_pages):
            data = _safe_get(f"{GAMMA_API}/markets", params={
                "active": "true",
                "closed": "false",
                "tag_slug": tag_slug,
                "limit": limit,
                "offset": offset,
            })
            if not data or not isinstance(data, list):
                break
            markets.extend(data)
            if len(data) < limit:
                break
            offset += limit

        return markets

    def _paginate_gamma(self) -> List[dict]:
        """Fetch all active markets from Gamma API with pagination."""
        markets = []
        offset = 0
        limit = 100
        max_pages = 50  # hard cap: 5,000 markets max; prevents infinite loops

        for page in range(max_pages):
            data = _safe_get(f"{GAMMA_API}/markets", params={
                "active": "true",
                "closed": "false",
                "limit": limit,
                "offset": offset,
            })
            if not data or not isinstance(data, list):
                break
            markets.extend(data)
            if len(data) < limit:
                break
            offset += limit
        else:
            log.warning("Gamma pagination hit max page limit (%d) — some markets may be missing", max_pages)

        return markets

    def _is_weather_market(self, market: dict) -> bool:
        title = (market.get("question") or market.get("title") or "").lower()
        # Apply sports blocklist first (before keyword check -- cheaper to skip early)
        for pat in self._sports_blocklist:
            if pat.search(title):
                return False
        # Weather keyword match
        if any(pat.search(title) for pat in self._kw_patterns):
            return True
        # Crypto keyword match
        if self._is_crypto_market(title):
            return True
        return False

    @staticmethod
    def _is_crypto_market(title: str) -> bool:
        """Check if a market title is about crypto prices."""
        crypto_keywords = [
            "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
            "xrp", "ripple", "dogecoin", "doge", "cardano", "ada",
        ]
        title_lower = title.lower()
        # Must mention a coin AND a price-like number
        has_coin = any(kw in title_lower for kw in crypto_keywords)
        has_price = bool(re.search(r"\$[\d,]+", title_lower))
        return has_coin and has_price

    def _parse_market(self, raw: dict) -> Optional[dict]:
        """
        Extract all fields needed for strategy evaluation.
        Returns None if the market fails liquidity or time filters.
        """
        title = raw.get("question") or raw.get("title") or ""
        condition_id = raw.get("conditionId") or raw.get("condition_id") or ""

        # Volume / liquidity check
        volume = float(raw.get("volume", 0) or 0)
        if volume < self.min_liquidity:
            log.debug("Skipping low-volume market: %s (%.0f USDC)", title[:60], volume)
            return None

        # Expiry window check
        end_date_str = raw.get("endDate") or raw.get("end_date_iso") or ""
        expiry_dt = None
        if end_date_str:
            try:
                expiry_dt = parse_utc_isoformat(end_date_str)
            except ValueError:
                pass

        now = datetime.now(timezone.utc)
        if expiry_dt:
            hours_left = (expiry_dt - now).total_seconds() / 3600
            if hours_left < self.min_hours:
                log.debug("Skipping near-expiry market: %s (%.1fh left)", title[:60], hours_left)
                return None
            if hours_left > self.max_hours:
                log.debug("Skipping far-future market: %s (%.1fh away)", title[:60], hours_left)
                return None

        # Get YES price + bid/ask spread from CLOB (for real-EV calculation)
        yes_price = _parse_gamma_price(raw)
        yes_token = _extract_yes_token(raw)
        no_token  = _extract_no_token(raw)

        # Fetch live orderbook data if we have a token ID
        spread_data = {"best_bid": None, "best_ask": None, "mid": None, "spread": 0.0}
        if yes_token:
            spread_data = _get_bid_ask_spread(yes_token)
            if yes_price is None and spread_data["mid"] is not None:
                yes_price = spread_data["mid"]
        if yes_price is None:
            yes_price = _get_clob_midpoint(yes_token) if yes_token else None

        if yes_price is None:
            log.debug("Could not determine price for: %s", title[:60])
            return None

        yes_price = max(0.01, min(0.99, yes_price))
        no_price  = 1.0 - yes_price

        best_ask = spread_data["best_ask"] or yes_price
        # Only use real bid data; a fabricated bid produces a fake spread that inflates EV
        best_bid = spread_data["best_bid"]
        spread   = spread_data["spread"] if best_bid is not None else 0.0

        # Parse city and condition
        city_match = self.city_index.match(title)
        city_name = city_match[0] if city_match else None
        city_coords = city_match[1] if city_match else None

        condition = parse_market_condition(title)
        target_dt = _parse_date_from_title(title) or expiry_dt

        # Bucket group ID — negRiskMarketID groups all buckets for the same city+date
        neg_risk_id = raw.get("negRiskMarketID") or raw.get("neg_risk_market_id")
        # Filter out null/zero IDs from non-neg-risk markets
        if neg_risk_id and set(neg_risk_id.replace("0x", "").replace("0", "")) == set():
            neg_risk_id = None

        # Adversarial check — updates in-memory history and returns True if suspicious
        adversarial = self._detector.update_and_check(
            market_id=condition_id,
            price=yes_price,
            volume=volume,
            spread=spread,
        )
        if adversarial:
            log.info("Adversarial flag on market %s — skipping new entries", condition_id[:30])

        # Crypto asset field (for crypto markets)
        crypto_asset = condition.get("crypto_asset")

        return {
            "condition_id":    condition_id,
            "yes_token_id":    yes_token,
            "no_token_id":     no_token,
            "title":           title,
            "yes_price":       round(yes_price, 4),
            "no_price":        round(no_price, 4),
            "best_ask":        round(best_ask, 4),
            "best_bid":        round(best_bid, 4) if best_bid is not None else None,
            "spread":          round(spread, 4),
            "volume_usdc":     volume,
            "expiry_dt":       expiry_dt.isoformat() if expiry_dt else None,
            "city":            city_name,
            "lat":             city_coords["lat"] if city_coords else None,
            "lon":             city_coords["lon"] if city_coords else None,
            "country":         city_coords["country"] if city_coords else None,
            "metric":          condition["metric"],
            "threshold":       condition["threshold"],
            "threshold_low":   condition["threshold_low"],
            "threshold_high":  condition["threshold_high"],
            "threshold_unit":  condition["threshold_unit"],
            "crypto_asset":    crypto_asset,
            "target_dt":       target_dt.isoformat() if target_dt else None,
            "bucket_group_id": neg_risk_id,
            "adversarial":     adversarial,
        }

    def filter_tradeable(self, markets: List[dict]) -> List[dict]:
        """Remove markets without a known city/asset or parseable metric."""
        result = []
        for m in markets:
            metric = m.get("metric")
            if not metric:
                continue
            # Crypto markets: need crypto_asset + threshold
            if metric in (CRYPTO_ABOVE, CRYPTO_BELOW):
                if m.get("crypto_asset") and m.get("threshold"):
                    result.append(m)
                continue
            # Weather markets: need city + lat
            if m.get("city") and m.get("lat") is not None:
                result.append(m)
        return result

    @staticmethod
    def compute_group_overround(bucket_group: List[dict]) -> float:
        """
        Compute the overround for a neg-risk bucket group.
        sum(YES mid-prices) - 1.0 == 0 for fair pricing,
        < 0 for underround (potential arbitrage), > 0 for overround (market fees).
        """
        mid_prices = []
        for m in bucket_group:
            best_bid = m.get("best_bid")
            best_ask = m.get("best_ask")
            if best_bid is not None and best_ask is not None:
                mid_prices.append((best_bid + best_ask) / 2.0)
            elif best_ask is not None:
                mid_prices.append(best_ask)
            else:
                mid_prices.append(m.get("yes_price", 0.0))
        return sum(mid_prices) - 1.0
