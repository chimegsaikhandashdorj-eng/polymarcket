"""Unit tests for edge confirmation and early exit logic."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import pytest
from src.edge_confirm import EdgeConfirmation, check_early_exits, ExitSignal


# ── Edge Confirmation Tests ────────────────────────────────────────────────────

class TestEdgeConfirmation:
    def test_first_sighting_not_confirmed(self):
        gate = EdgeConfirmation(required=2)
        result = gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        assert result is False

    def test_second_sighting_confirmed(self):
        gate = EdgeConfirmation(required=2)
        gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        result = gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        assert result is True

    def test_three_required_confirmations(self):
        gate = EdgeConfirmation(required=3)
        assert gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5) is False
        assert gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5) is False
        assert gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5) is True

    def test_different_markets_independent(self):
        gate = EdgeConfirmation(required=2)
        gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        gate.check("mkt2", "YES", ev=0.12, our_prob=0.6, market_price=0.4)
        # mkt2 is not yet confirmed
        assert gate.check("mkt2", "YES", ev=0.12, our_prob=0.6, market_price=0.4) is True
        # mkt1 should also be confirmed now
        assert gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5) is True

    def test_different_sides_independent(self):
        gate = EdgeConfirmation(required=2)
        gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        # Same market but NO side is a different entry
        gate.check("mkt1", "NO", ev=0.08, our_prob=0.3, market_price=0.5)
        # YES side should still need confirmation
        assert gate.get_pending_count() == 2

    def test_ev_drift_resets_counter(self):
        gate = EdgeConfirmation(required=2)
        gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        # EV drifted by more than threshold (0.02)
        result = gate.check("mkt1", "YES", ev=0.15, our_prob=0.7, market_price=0.5)
        assert result is False
        # Needs another confirmation after reset
        result = gate.check("mkt1", "YES", ev=0.15, our_prob=0.7, market_price=0.5)
        assert result is True

    def test_small_ev_drift_ok(self):
        gate = EdgeConfirmation(required=2)
        gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        # Small drift within threshold (< 0.02)
        result = gate.check("mkt1", "YES", ev=0.11, our_prob=0.7, market_price=0.5)
        assert result is True

    def test_ttl_expiry(self):
        gate = EdgeConfirmation(required=2, ttl=0.1)  # 100ms TTL
        gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        time.sleep(0.15)  # Wait for TTL to expire
        # Should be treated as first sighting again
        result = gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        assert result is False

    def test_pending_count(self):
        gate = EdgeConfirmation(required=3)
        gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        gate.check("mkt2", "NO", ev=0.08, our_prob=0.3, market_price=0.5)
        assert gate.get_pending_count() == 2

    def test_confirmed_removes_from_pending(self):
        gate = EdgeConfirmation(required=2)
        gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        assert gate.get_pending_count() == 1
        gate.check("mkt1", "YES", ev=0.10, our_prob=0.7, market_price=0.5)
        assert gate.get_pending_count() == 0


# ── Early Exit Tests ───────────────────────────────────────────────────────────

class TestEarlyExit:
    def test_no_positions_returns_empty(self):
        signals = check_early_exits([], {})
        assert signals == []

    def test_no_price_data_returns_empty(self):
        positions = [{"market_id": "mkt1", "side": "YES", "entry_price": 0.5, "size_usdc": 25, "id": 1}]
        signals = check_early_exits(positions, {})
        assert signals == []

    def test_profit_take_yes_side(self):
        positions = [{
            "id": 1,
            "market_id": "mkt1",
            "market_title": "Will it rain?",
            "side": "YES",
            "entry_price": 0.40,
            "size_usdc": 25,
        }]
        # Current price is 0.85 — our YES tokens worth 0.85, paid 0.40
        # PnL% = (0.85 - 0.40) / 0.40 = 112.5% > 60% threshold
        signals = check_early_exits(positions, {"mkt1": 0.85})
        assert len(signals) == 1
        assert signals[0].unrealized_pnl_pct > 0.60
        assert "Profit" in signals[0].reason

    def test_stop_loss_yes_side(self):
        positions = [{
            "id": 2,
            "market_id": "mkt2",
            "market_title": "Will it snow?",
            "side": "YES",
            "entry_price": 0.60,
            "size_usdc": 25,
        }]
        # Current price dropped to 0.25 — loss = (0.25-0.60)/0.60 = -58% > -50%
        signals = check_early_exits(positions, {"mkt2": 0.25})
        assert len(signals) == 1
        assert signals[0].unrealized_pnl_pct < -0.50
        assert "Stop" in signals[0].reason

    def test_no_signal_within_range(self):
        positions = [{
            "id": 3,
            "market_id": "mkt3",
            "market_title": "Will it be hot?",
            "side": "YES",
            "entry_price": 0.50,
            "size_usdc": 25,
        }]
        # Current price is 0.60 — PnL% = 20%, within normal range
        signals = check_early_exits(positions, {"mkt3": 0.60})
        assert signals == []

    def test_profit_take_no_side(self):
        positions = [{
            "id": 4,
            "market_id": "mkt4",
            "market_title": "Will temp exceed 90F?",
            "side": "NO",
            "entry_price": 0.35,  # We bought NO at 0.35
            "size_usdc": 25,
        }]
        # Current YES price is 0.10, so NO price = 0.90
        # PnL% = (0.90 - 0.35) / 0.35 = 157% > 60%
        signals = check_early_exits(positions, {"mkt4": 0.10})
        assert len(signals) == 1
        assert signals[0].unrealized_pnl_pct > 0.60

    def test_stop_loss_no_side(self):
        positions = [{
            "id": 5,
            "market_id": "mkt5",
            "market_title": "Will it rain tomorrow?",
            "side": "NO",
            "entry_price": 0.40,  # We bought NO at 0.40
            "size_usdc": 25,
        }]
        # Current YES price is 0.85, so NO price = 0.15
        # PnL% = (0.15 - 0.40) / 0.40 = -62.5% < -50%
        signals = check_early_exits(positions, {"mkt5": 0.85})
        assert len(signals) == 1
        assert signals[0].unrealized_pnl_pct < -0.50
