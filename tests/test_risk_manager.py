"""Unit tests for the risk manager — Kelly sizing and approval logic."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch
from src.risk_manager import RiskManager, RiskVetoError, TradeApproval
from src.strategy import Opportunity
from src.market_scanner import RAIN


def _make_opportunity(**kwargs):
    defaults = dict(
        market_id="mkt1",
        market_title="Will it rain in NYC?",
        side="YES",
        our_prob=0.70,
        market_price=0.50,
        ev=0.40,
        confidence=0.80,
        score=0.32,
        city="new york",
        metric=RAIN,
        target_dt="2025-05-05",
        lat=40.71,
        lon=-74.01,
    )
    defaults.update(kwargs)
    return Opportunity(**defaults)


@pytest.fixture
def config():
    return {
        "trading": {
            "paper_mode": True,
            "ev_threshold": 0.05,
            "kelly_fraction": 0.25,
            "max_position_usdc": 50,
            "max_open_positions": 5,
            "daily_loss_limit_usdc": 200,
            "min_confidence": 0.50,
        },
        "risk": {
            "stale_forecast_hours": 3,
            "correlation_block": True,
        },
    }


# ── Kelly sizing ───────────────────────────────────────────────────────────────

def test_kelly_positive_edge(config):
    rm = RiskManager(config)
    size = rm.kelly_size(prob=0.70, market_price=0.50, bankroll=1000)
    assert size > 0
    assert size <= 50  # capped at max_position


def test_kelly_no_edge(config):
    rm = RiskManager(config)
    # prob matches market price exactly → zero Kelly
    size = rm.kelly_size(prob=0.50, market_price=0.50, bankroll=1000)
    assert size == 0.0


def test_kelly_negative_edge(config):
    rm = RiskManager(config)
    size = rm.kelly_size(prob=0.20, market_price=0.50, bankroll=1000)
    assert size == 0.0


def test_kelly_respects_max_position(config):
    rm = RiskManager(config)
    # Very strong edge with large bankroll → should be capped
    size = rm.kelly_size(prob=0.95, market_price=0.10, bankroll=100_000)
    assert size <= 50.0


def test_kelly_scales_with_bankroll(config):
    rm = RiskManager(config)
    s1 = rm.kelly_size(0.70, 0.50, 100)
    s2 = rm.kelly_size(0.70, 0.50, 200)
    # s2 should be larger (double bankroll), unless capped
    assert s2 >= s1


# ── Approval logic ─────────────────────────────────────────────────────────────

@patch("src.risk_manager.get_daily_loss", return_value=0.0)
@patch("src.risk_manager.get_open_positions", return_value=[])
def test_approve_passes(mock_pos, mock_loss, config):
    rm = RiskManager(config)
    opp = _make_opportunity()
    approval = rm.approve(opp, bankroll=500, weather_age_seconds=100)
    assert isinstance(approval, TradeApproval)
    assert approval.size_usdc > 0


@patch("src.risk_manager.get_daily_loss", return_value=0.0)
@patch("src.risk_manager.get_open_positions", return_value=[])
def test_approve_rejects_low_confidence(mock_pos, mock_loss, config):
    rm = RiskManager(config)
    opp = _make_opportunity(confidence=0.30)
    with pytest.raises(RiskVetoError, match="Confidence"):
        rm.approve(opp, bankroll=500)


@patch("src.risk_manager.get_daily_loss", return_value=0.0)
@patch("src.risk_manager.get_open_positions", return_value=[])
def test_approve_rejects_stale_forecast(mock_pos, mock_loss, config):
    rm = RiskManager(config)
    opp = _make_opportunity()
    with pytest.raises(RiskVetoError, match="old"):
        rm.approve(opp, bankroll=500, weather_age_seconds=4 * 3600)  # 4h > 3h limit


@patch("src.risk_manager.get_daily_loss", return_value=250.0)  # over limit
@patch("src.risk_manager.get_open_positions", return_value=[])
def test_approve_rejects_daily_limit(mock_pos, mock_loss, config):
    rm = RiskManager(config)
    opp = _make_opportunity()
    with pytest.raises(RiskVetoError, match="Daily loss"):
        rm.approve(opp, bankroll=500)


@patch("src.risk_manager.get_daily_loss", return_value=0.0)
@patch("src.risk_manager.get_open_positions", return_value=[{} for _ in range(5)])
def test_approve_rejects_max_positions(mock_pos, mock_loss, config):
    rm = RiskManager(config)
    opp = _make_opportunity()
    with pytest.raises(RiskVetoError, match="max open positions"):
        rm.approve(opp, bankroll=500)


@patch("src.risk_manager.get_daily_loss", return_value=0.0)
@patch("src.risk_manager.get_open_positions", return_value=[{
    "city": "new york",
    "metric": RAIN,
    "expiry": "2025-05-05T00:00:00",
}])
def test_approve_rejects_correlated(mock_pos, mock_loss, config):
    rm = RiskManager(config)
    opp = _make_opportunity()
    with pytest.raises(RiskVetoError, match="Correlated"):
        rm.approve(opp, bankroll=500)


# ── Weekly / Monthly loss limits ───────────────────────────────────────────────

@patch("src.risk_manager.get_monthly_loss", return_value=0.0)
@patch("src.risk_manager.get_weekly_loss", return_value=600.0)  # over weekly limit
@patch("src.risk_manager.get_daily_loss", return_value=0.0)
@patch("src.risk_manager.get_open_positions", return_value=[])
def test_approve_rejects_weekly_limit(mock_pos, mock_daily, mock_weekly, mock_monthly, config):
    rm = RiskManager(config)
    rm.weekly_limit = 500.0
    opp = _make_opportunity()
    with pytest.raises(RiskVetoError, match="Weekly loss"):
        rm.approve(opp, bankroll=500)


@patch("src.risk_manager.get_monthly_loss", return_value=1200.0)  # over monthly limit
@patch("src.risk_manager.get_weekly_loss", return_value=0.0)
@patch("src.risk_manager.get_daily_loss", return_value=0.0)
@patch("src.risk_manager.get_open_positions", return_value=[])
def test_approve_rejects_monthly_limit(mock_pos, mock_daily, mock_weekly, mock_monthly, config):
    rm = RiskManager(config)
    rm.monthly_limit = 1000.0
    opp = _make_opportunity()
    with pytest.raises(RiskVetoError, match="Monthly loss"):
        rm.approve(opp, bankroll=500)


# ── Adversarial market detection ───────────────────────────────────────────────

@patch("src.risk_manager.get_monthly_loss", return_value=0.0)
@patch("src.risk_manager.get_weekly_loss", return_value=0.0)
@patch("src.risk_manager.get_daily_loss", return_value=0.0)
@patch("src.risk_manager.get_open_positions", return_value=[])
def test_approve_rejects_adversarial(mock_pos, mock_daily, mock_weekly, mock_monthly, config):
    rm = RiskManager(config)
    opp = _make_opportunity(adversarial=True)
    with pytest.raises(RiskVetoError, match="adversarial"):
        rm.approve(opp, bankroll=500)


@patch("src.risk_manager.get_monthly_loss", return_value=0.0)
@patch("src.risk_manager.get_weekly_loss", return_value=0.0)
@patch("src.risk_manager.get_daily_loss", return_value=0.0)
@patch("src.risk_manager.get_open_positions", return_value=[])
def test_approve_passes_non_adversarial(mock_pos, mock_daily, mock_weekly, mock_monthly, config):
    rm = RiskManager(config)
    opp = _make_opportunity(adversarial=False)
    approval = rm.approve(opp, bankroll=500)
    assert isinstance(approval, TradeApproval)
    assert approval.size_usdc > 0


# ── AdversarialDetector ────────────────────────────────────────────────────────

def test_adversarial_detector_no_flag_on_first_scan():
    from src.market_scanner import AdversarialDetector
    det = AdversarialDetector({"adversarial": {"enabled": True}})
    # First snapshot — no prior history to compare, should not flag
    flagged = det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.02)
    assert not flagged


def test_adversarial_detector_flags_price_jump():
    from src.market_scanner import AdversarialDetector
    det = AdversarialDetector({
        "adversarial": {"enabled": True, "price_jump_threshold": 0.10}
    })
    det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.02)
    # Price jumps 15pp — should be flagged
    flagged = det.update_and_check("mkt1", price=0.65, volume=10000, spread=0.02)
    assert flagged
    assert det.get_flag_count() == 1


def test_adversarial_detector_flags_spread_collapse():
    from src.market_scanner import AdversarialDetector
    det = AdversarialDetector({
        "adversarial": {
            "enabled": True,
            "price_jump_threshold": 0.10,
            "spread_collapse_high": 0.05,
            "spread_collapse_low": 0.01,
        }
    })
    det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.08)
    flagged = det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.005)
    assert flagged


def test_adversarial_detector_disabled():
    from src.market_scanner import AdversarialDetector
    det = AdversarialDetector({"adversarial": {"enabled": False}})
    det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.02)
    flagged = det.update_and_check("mkt1", price=0.95, volume=10000, spread=0.02)
    assert not flagged  # disabled — never flags


# ── Opportunity cost ───────────────────────────────────────────────────────────

def test_opportunity_ev_raw_field():
    """Opportunity.ev_raw defaults to 0.0 when not set."""
    opp = _make_opportunity()
    assert hasattr(opp, "ev_raw")
    assert opp.ev_raw == 0.0


def test_opportunity_adversarial_field():
    """Opportunity.adversarial defaults False."""
    opp = _make_opportunity()
    assert hasattr(opp, "adversarial")
    assert opp.adversarial is False
