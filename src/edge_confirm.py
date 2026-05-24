"""
Edge confirmation and early exit logic.

Two key features:
  1. Edge Confirmation: requires an opportunity to appear in 2+ consecutive scans
     before trading. Prevents acting on momentary mispricings / stale prices.

  2. Early Exit: checks open positions to see if the market moved significantly
     in our favor. If so, signals that we could sell the position early
     to lock in profit (reduces risk of reversal).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Edge confirmation ──────────────────────────────────────────────────────────

# Number of consecutive scans an opportunity must appear before we trade
_REQUIRED_CONFIRMATIONS = 2

# Maximum age of a "pending" confirmation (seconds). If we don't see the
# opportunity again within this window, it expires.
_CONFIRMATION_TTL = 3600  # 1 hour (4 scan cycles at 15min each)

# Minimum EV stability: the EV must not drop more than this between scans
_EV_STABILITY_THRESHOLD = 0.02  # 2% drift max


@dataclass
class PendingEdge:
    """Tracks an opportunity seen but not yet confirmed."""
    market_id: str
    side: str
    ev: float
    our_prob: float
    market_price: float
    first_seen: float       # timestamp
    last_seen: float        # timestamp
    confirmations: int = 1  # starts at 1 (first sighting)


class EdgeConfirmation:
    """
    Gate that prevents single-scan flukes from becoming trades.
    Only passes an opportunity through once it's been seen in N consecutive scans
    with stable EV.
    """

    def __init__(self, required: int = _REQUIRED_CONFIRMATIONS, ttl: float = _CONFIRMATION_TTL):
        self._required = required
        self._ttl = ttl
        # market_id+side -> PendingEdge
        self._pending: Dict[str, PendingEdge] = {}

    def check(self, market_id: str, side: str, ev: float,
              our_prob: float, market_price: float) -> bool:
        """
        Returns True if this opportunity is confirmed (seen enough times).
        Call once per scan for each opportunity that passes initial filters.
        """
        key = f"{market_id}:{side}"
        now = time.time()

        # Purge expired entries
        self._purge_stale(now)

        if key in self._pending:
            pending = self._pending[key]

            # Check EV stability — reject if edge drifted too much
            ev_drift = abs(ev - pending.ev)
            if ev_drift > _EV_STABILITY_THRESHOLD:
                log.debug(
                    "Edge unstable for %s: EV %.3f -> %.3f (drift %.3f > %.3f) — resetting",
                    market_id[:20], pending.ev, ev, ev_drift, _EV_STABILITY_THRESHOLD,
                )
                # Reset — the edge is shifting, wait for stability
                pending.ev = ev
                pending.our_prob = our_prob
                pending.market_price = market_price
                pending.confirmations = 1
                pending.last_seen = now
                return False

            # Edge confirmed — increment counter
            pending.confirmations += 1
            pending.last_seen = now
            pending.ev = ev
            pending.our_prob = our_prob
            pending.market_price = market_price

            if pending.confirmations >= self._required:
                # Confirmed! Remove from pending and allow trade
                del self._pending[key]
                log.info(
                    "Edge CONFIRMED after %d scans: %s %s EV=%.3f",
                    pending.confirmations, side, market_id[:30], ev,
                )
                return True

            log.debug(
                "Edge pending (%d/%d): %s %s EV=%.3f",
                pending.confirmations, self._required, side, market_id[:20], ev,
            )
            return False
        else:
            # First sighting — record and wait for confirmation
            self._pending[key] = PendingEdge(
                market_id=market_id,
                side=side,
                ev=ev,
                our_prob=our_prob,
                market_price=market_price,
                first_seen=now,
                last_seen=now,
                confirmations=1,
            )
            log.debug("Edge first seen: %s %s EV=%.3f — awaiting confirmation", side, market_id[:20], ev)
            return False

    def _purge_stale(self, now: float) -> None:
        """Remove entries not seen recently."""
        stale_keys = [
            k for k, v in self._pending.items()
            if (now - v.last_seen) > self._ttl
        ]
        for k in stale_keys:
            del self._pending[k]

    def get_pending_count(self) -> int:
        """How many opportunities are awaiting confirmation."""
        return len(self._pending)

    def get_pending_edges(self) -> List[PendingEdge]:
        """Return all pending edges for display/reporting."""
        return list(self._pending.values())


# ── Early exit detection ───────────────────────────────────────────────────────

@dataclass
class ExitSignal:
    """Signals that an open position should be considered for early exit."""
    trade_id: int
    market_title: str
    side: str
    entry_price: float
    current_price: float
    unrealized_pnl_pct: float
    reason: str


# Thresholds for early exit consideration
_PROFIT_TAKE_PCT = 0.60    # Take profit if unrealized gain > 60% of max possible
_STOP_LOSS_PCT = -0.50     # Cut loss if position lost > 50% of stake


def check_early_exits(
    open_positions: List[dict],
    current_prices: Dict[str, float],
) -> List[ExitSignal]:
    """
    Check open positions against current market prices.
    Returns list of ExitSignals for positions that should consider early exit.

    current_prices: {condition_id: current YES price}
    """
    signals = []

    for pos in open_positions:
        market_id = pos.get("market_id", "")
        if market_id not in current_prices:
            continue

        current_yes_price = current_prices[market_id]
        entry_price = pos["entry_price"]
        side = pos["side"]
        size_usdc = pos["size_usdc"]

        # Calculate unrealized PnL percentage
        if side == "YES":
            # We bought YES tokens at entry_price
            # Current value per unit = current_yes_price
            # Cost per unit = entry_price
            pnl_pct = (current_yes_price - entry_price) / entry_price
        else:
            # We bought NO tokens at entry_price (= 1 - yes_price at entry)
            # Current NO price = 1 - current_yes_price
            current_no_price = 1.0 - current_yes_price
            pnl_pct = (current_no_price - entry_price) / entry_price

        # Check profit-taking threshold
        if pnl_pct >= _PROFIT_TAKE_PCT:
            signals.append(ExitSignal(
                trade_id=pos["id"],
                market_title=pos.get("market_title", "")[:50],
                side=side,
                entry_price=entry_price,
                current_price=current_yes_price,
                unrealized_pnl_pct=pnl_pct,
                reason=f"Profit target hit: {pnl_pct:.0%} gain",
            ))
            log.info(
                "EARLY EXIT signal (profit): trade #%d %s %s  entry=%.3f  now=%.3f  pnl=%.0f%%",
                pos["id"], side, pos.get("market_title", "")[:30],
                entry_price, current_yes_price, pnl_pct * 100,
            )

        # Check stop-loss threshold
        elif pnl_pct <= _STOP_LOSS_PCT:
            signals.append(ExitSignal(
                trade_id=pos["id"],
                market_title=pos.get("market_title", "")[:50],
                side=side,
                entry_price=entry_price,
                current_price=current_yes_price,
                unrealized_pnl_pct=pnl_pct,
                reason=f"Stop loss hit: {pnl_pct:.0%} loss",
            ))
            log.info(
                "EARLY EXIT signal (stop): trade #%d %s %s  entry=%.3f  now=%.3f  pnl=%.0f%%",
                pos["id"], side, pos.get("market_title", "")[:30],
                entry_price, current_yes_price, pnl_pct * 100,
            )

    return signals


# ── Singleton ──────────────────────────────────────────────────────────────────

_edge_confirm_instance: Optional[EdgeConfirmation] = None


def get_edge_confirmation(config: Optional[dict] = None) -> EdgeConfirmation:
    """Get or create the EdgeConfirmation singleton."""
    global _edge_confirm_instance
    if _edge_confirm_instance is None:
        required = 2
        if config:
            required = config.get("trading", {}).get("edge_confirmations", 2)
        _edge_confirm_instance = EdgeConfirmation(required=required)
    return _edge_confirm_instance
