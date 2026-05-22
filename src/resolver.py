"""
Checks open paper/live trades against Polymarket for resolution.
Called from the main bot loop after each scan cycle.

Polymarket convention:
  resolutionPrice = 1.0  →  YES token paid out (YES wins)
  resolutionPrice = 0.0  →  NO  token paid out (YES loses)
  resolutionPrice ~ 0.5  →  void / split (refund both sides)
"""

import logging
import time

import requests

from .logger import get_open_positions, update_outcome

log = logging.getLogger(__name__)

_GAMMA = "https://gamma-api.polymarket.com"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "polymarket-weather-bot/1.0"})

_MAX_RETRIES = 3
_VOID_EPSILON = 0.01   # |res_price - 0.5| < this → VOID (avoids float == 0.5 trap)


def _fetch_resolution(condition_id: str):
    """
    Return (resolved, res_price) for a market with retry on transient errors.
    resolved=False means the market is still open or the fetch failed.
    """
    if not condition_id:
        return False, None

    for attempt in range(_MAX_RETRIES):
        try:
            r = _SESSION.get(
                f"{_GAMMA}/markets",
                params={"conditionIds": condition_id},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            break  # success
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status in (400, 401, 403, 404):
                log.debug("Resolver: HTTP %d for %s — not retrying", status, condition_id[:16])
                return False, None
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2.0 ** attempt)
            else:
                log.debug("Resolver: all retries exhausted for %s", condition_id[:16])
                return False, None
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2.0 ** attempt)
            else:
                log.debug("Resolver fetch failed for %s: %s", condition_id[:16], exc)
                return False, None
    else:
        return False, None

    if not data or not isinstance(data, list):
        return False, None

    market = data[0]
    if not (market.get("resolved") or market.get("closed")):
        return False, None

    raw = market.get("resolutionPrice")
    if raw is None:
        raw = market.get("resolution_price")
    if raw is None:
        return False, None

    try:
        return True, float(raw)
    except (ValueError, TypeError):
        log.warning("Could not parse resolutionPrice=%r for %s", raw, condition_id[:16])
        return False, None


def resolve_open_trades(paper: bool = True) -> int:
    """
    Checks every open trade against Polymarket.
    Marks WIN/LOSS/VOID via update_outcome() for settled markets.
    Returns the number of trades resolved.
    """
    open_trades = get_open_positions(paper=paper)
    if not open_trades:
        return 0

    resolved_count = 0
    for trade in open_trades:
        is_resolved, res_price = _fetch_resolution(trade["market_id"])
        if not is_resolved:
            continue

        side = trade["side"]
        # Use epsilon comparison for VOID to avoid float == 0.5 precision trap
        if abs(res_price - 0.5) < _VOID_EPSILON:
            outcome = "VOID"
        elif side == "YES":
            outcome = "WIN" if res_price > 0.5 else "LOSS"
        else:
            outcome = "WIN" if res_price < 0.5 else "LOSS"

        update_outcome(trade["id"], res_price, outcome)
        log.info(
            "Resolved trade #%d (%s %s) → %s  res_price=%.4f",
            trade["id"], side, trade["market_title"][:40], outcome, res_price,
        )
        resolved_count += 1

    if resolved_count:
        log.info("Resolution: %d/%d open trades settled", resolved_count, len(open_trades))

    return resolved_count
