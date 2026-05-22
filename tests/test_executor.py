"""Unit tests for TradeExecutor — paper mode execution, slippage, order mode."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock
from src.executor import TradeExecutor, _LIMIT_SLIP, _PASSIVE_FILL_PROB
from src.risk_manager import TradeApproval, RiskVetoError
from src.strategy import Opportunity
from src.market_scanner import RAIN


def _cfg(paper=True, slip_base=0.005, slip_max=0.02):
    return {
        "trading": {
            "paper_mode": paper,
            "ev_threshold": 0.05,
            "kelly_fraction": 0.25,
            "max_position_usdc": 50,
            "max_open_positions": 5,
            "daily_loss_limit_usdc": 200,
            "min_confidence": 0.50,
        },
        "risk": {"stale_forecast_hours": 3, "correlation_block": True},
        "slippage": {"base": slip_base, "max_impact": slip_max},
    }


def _opp(**kwargs):
    defaults = dict(
        market_id="mkt1", market_title="Will it rain in NYC?", side="YES",
        our_prob=0.70, market_price=0.50, ev=0.40, confidence=0.80,
        score=0.32, city="new york", metric=RAIN, target_dt="2025-05-05",
        lat=40.71, lon=-74.01, volume_usdc=100_000.0,
    )
    defaults.update(kwargs)
    return Opportunity(**defaults)


def _approval(size=10.0):
    return TradeApproval(size_usdc=size, reason="OK")


# ── Constructor ───────────────────────────────────────────────────────────────

def test_constructor_reads_slippage_config():
    ex = TradeExecutor(_cfg(slip_base=0.007, slip_max=0.03))
    assert ex._slip_base == 0.007
    assert ex._slip_max_impact == 0.03


def test_constructor_paper_mode():
    ex = TradeExecutor(_cfg(paper=True))
    assert ex.paper is True


# ── Slippage estimation ───────────────────────────────────────────────────────

def test_estimate_slippage_zero_volume_uses_floor():
    ex = TradeExecutor(_cfg())
    impact = ex._estimate_slippage(size_usdc=10.0, volume_usdc=0.0)
    # depth = max(200, 0) = 200; impact = 0.005 * sqrt(10/200)
    import math
    expected = 0.005 * math.sqrt(10.0 / 200.0)
    assert abs(impact - expected) < 1e-9


def test_estimate_slippage_capped_at_max():
    ex = TradeExecutor(_cfg())
    impact = ex._estimate_slippage(size_usdc=1_000_000.0, volume_usdc=10.0)
    assert impact == ex._slip_max_impact


def test_estimate_slippage_large_deep_market_is_small():
    ex = TradeExecutor(_cfg())
    impact = ex._estimate_slippage(size_usdc=5.0, volume_usdc=10_000_000.0)
    assert impact < 0.001


# ── Order mode ────────────────────────────────────────────────────────────────

def test_order_mode_near_expiry_is_aggressive():
    ex = TradeExecutor(_cfg())
    opp = _opp()
    object.__setattr__(opp, "hours_to_expiry", 12.0)
    assert ex._choose_order_mode(opp) == "AGGRESSIVE"


def test_order_mode_high_ev_is_aggressive():
    ex = TradeExecutor(_cfg())
    opp = _opp(ev=0.15)
    object.__setattr__(opp, "hours_to_expiry", 72.0)
    assert ex._choose_order_mode(opp) == "AGGRESSIVE"


def test_order_mode_wide_spread_is_aggressive():
    ex = TradeExecutor(_cfg())
    opp = _opp()
    object.__setattr__(opp, "hours_to_expiry", 72.0)
    object.__setattr__(opp, "spread", 0.05)
    assert ex._choose_order_mode(opp) == "AGGRESSIVE"


def test_order_mode_default_normal_is_passive():
    ex = TradeExecutor(_cfg())
    opp = _opp(ev=0.06)
    # hours_to_expiry defaults to 72.0 via getattr; spread defaults to 0
    assert ex._choose_order_mode(opp) == "PASSIVE"


# ── execute() — paper mode ─────────────────────────────────────────────────────

@patch("src.executor.log_trade_prediction")
@patch("src.executor.log_trade", return_value=42)
def test_execute_paper_success(mock_log, mock_pred):
    ex = TradeExecutor(_cfg())
    with patch.object(ex.risk, "approve", return_value=_approval(10.0)):
        result = ex.execute(_opp())
    assert result == 42


@patch("src.executor.log_trade_prediction")
@patch("src.executor.log_trade", return_value=1)
def test_execute_paper_veto_returns_none(mock_log, mock_pred):
    ex = TradeExecutor(_cfg())
    with patch.object(ex.risk, "approve", side_effect=RiskVetoError("daily limit")):
        result = ex.execute(_opp())
    assert result is None


@patch("src.executor.log_trade_prediction")
@patch("src.executor.log_trade", return_value=1)
def test_execute_paper_thin_market_blocked(mock_log, mock_pred):
    """Slippage >= max_impact on thin market → None."""
    ex = TradeExecutor(_cfg(slip_max=0.001))  # very tight max
    with patch.object(ex.risk, "approve", return_value=_approval(50.0)):
        # volume_usdc=0 → depth=200 → impact=0.005*sqrt(50/200)≈0.0025 > 0.001
        result = ex.execute(_opp(volume_usdc=0.0))
    assert result is None


@patch("src.executor.log_trade_prediction")
@patch("src.executor.log_trade", return_value=7)
def test_execute_paper_passive_fill(mock_log, mock_pred):
    """Force passive fill by mocking random to always succeed."""
    ex = TradeExecutor(_cfg())
    with patch.object(ex.risk, "approve", return_value=_approval(10.0)):
        with patch("src.executor.random.random", return_value=0.0):  # 0 < 0.65 → fill
            result = ex.execute(_opp(ev=0.06))  # low EV → PASSIVE mode
    assert result == 7


@patch("src.executor.log_trade_prediction")
@patch("src.executor.log_trade", return_value=8)
def test_execute_paper_passive_miss_retries_aggressive(mock_log, mock_pred):
    """Force passive miss by mocking random to exceed fill prob."""
    ex = TradeExecutor(_cfg())
    with patch.object(ex.risk, "approve", return_value=_approval(10.0)):
        with patch("src.executor.random.random", return_value=0.99):  # > 0.65 → miss
            result = ex.execute(_opp(ev=0.06))
    assert result == 8


# ── refresh_bankroll ──────────────────────────────────────────────────────────

def test_refresh_bankroll_adds_pnl():
    ex = TradeExecutor(_cfg())
    with patch("src.executor.get_realized_pnl", return_value=200.0):
        b = ex.refresh_bankroll()
    assert b == pytest.approx(700.0)  # 500 initial + 200 pnl


def test_refresh_bankroll_floor_at_one():
    ex = TradeExecutor(_cfg())
    with patch("src.executor.get_realized_pnl", return_value=-10_000.0):
        b = ex.refresh_bankroll()
    assert b == 1.0


def test_get_bankroll_returns_current():
    ex = TradeExecutor(_cfg())
    ex._bankroll = 999.0
    assert ex.get_bankroll() == 999.0
