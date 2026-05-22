"""Unit tests for dashboard.py — smoke tests with mocked console."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock
from src.strategy import Opportunity
from src.market_scanner import RAIN


def _opp(**kwargs):
    defaults = dict(
        market_id="mkt1", market_title="Will it rain in NYC?", side="YES",
        our_prob=0.70, market_price=0.50, ev=0.40, ev_raw=0.42,
        confidence=0.80, score=0.32, city="new york", metric=RAIN,
        target_dt="2025-05-05", lat=40.71, lon=-74.01,
    )
    defaults.update(kwargs)
    return Opportunity(**defaults)


# Patch rich console for all tests to avoid terminal output
@pytest.fixture(autouse=True)
def mock_console(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("src.dashboard.console", mock)
    return mock


# ── show_opportunities ────────────────────────────────────────────────────────

def test_show_opportunities_empty(mock_console):
    from src.dashboard import show_opportunities
    show_opportunities([])
    mock_console.print.assert_called_once()


def test_show_opportunities_single(mock_console):
    from src.dashboard import show_opportunities
    show_opportunities([_opp()])
    mock_console.print.assert_called_once()


def test_show_opportunities_adversarial_marker(mock_console):
    from src.dashboard import show_opportunities
    show_opportunities([_opp(adversarial=True)])
    mock_console.print.assert_called_once()


def test_show_opportunities_with_opp_cost(mock_console):
    """When ev != ev_raw, both are shown."""
    from src.dashboard import show_opportunities
    show_opportunities([_opp(ev=0.38, ev_raw=0.42)])
    mock_console.print.assert_called_once()


def test_show_opportunities_multiple(mock_console):
    from src.dashboard import show_opportunities
    opps = [_opp(market_id=f"mkt{i}") for i in range(5)]
    show_opportunities(opps)
    mock_console.print.assert_called_once()


# ── show_open_positions ───────────────────────────────────────────────────────

def test_show_open_positions_empty(mock_console):
    from src.dashboard import show_open_positions
    with patch("src.dashboard.get_open_positions", return_value=[]):
        show_open_positions(paper=True)
    mock_console.print.assert_called_once()


def test_show_open_positions_with_data(mock_console):
    from src.dashboard import show_open_positions
    fake_pos = [{
        "id": 1, "market_title": "Will it rain?", "side": "YES",
        "size_usdc": 10.0, "entry_price": 0.5, "ev": 0.40,
        "timestamp": "2026-05-05T10:00:00",
    }]
    with patch("src.dashboard.get_open_positions", return_value=fake_pos):
        show_open_positions(paper=True)
    mock_console.print.assert_called_once()


def test_show_open_positions_live_mode(mock_console):
    from src.dashboard import show_open_positions
    with patch("src.dashboard.get_open_positions", return_value=[]):
        show_open_positions(paper=False)
    mock_console.print.assert_called_once()


# ── show_pnl_summary ──────────────────────────────────────────────────────────

def test_show_pnl_summary_positive(mock_console):
    from src.dashboard import show_pnl_summary
    summary = {"total_trades": 5, "wins": 3, "losses": 2, "voids": 0,
               "win_rate": 0.6, "total_pnl_usdc": 50.0}
    with patch("src.dashboard.get_pnl_summary", return_value=summary):
        show_pnl_summary(paper=True)
    mock_console.print.assert_called_once()


def test_show_pnl_summary_negative_pnl(mock_console):
    from src.dashboard import show_pnl_summary
    summary = {"total_trades": 2, "wins": 0, "losses": 2, "voids": 0,
               "win_rate": 0.0, "total_pnl_usdc": -30.0}
    with patch("src.dashboard.get_pnl_summary", return_value=summary):
        show_pnl_summary(paper=False)
    mock_console.print.assert_called_once()


# ── show_scan_header ──────────────────────────────────────────────────────────

def test_show_scan_header_no_flags(mock_console):
    from src.dashboard import show_scan_header
    show_scan_header(7, 42, adversarial_flags=0)
    mock_console.print.assert_called_once()


def test_show_scan_header_with_flags(mock_console):
    from src.dashboard import show_scan_header
    show_scan_header(7, 42, adversarial_flags=3)
    mock_console.print.assert_called_once()


# ── show_loss_limits_panel ────────────────────────────────────────────────────

def test_show_loss_limits_panel_all_green(mock_console):
    from src.dashboard import show_loss_limits_panel
    config = {"loss_limits": {"daily_usdc": 200, "weekly_usdc": 500, "monthly_usdc": 1000}}
    with patch("src.dashboard.get_daily_loss", return_value=10.0), \
         patch("src.dashboard.get_weekly_loss", return_value=20.0), \
         patch("src.dashboard.get_monthly_loss", return_value=50.0):
        show_loss_limits_panel(paper=True, config=config)
    mock_console.print.assert_called_once()


def test_show_loss_limits_panel_halted(mock_console):
    from src.dashboard import show_loss_limits_panel
    config = {"loss_limits": {"daily_usdc": 200, "weekly_usdc": 500, "monthly_usdc": 1000}}
    with patch("src.dashboard.get_daily_loss", return_value=250.0), \
         patch("src.dashboard.get_weekly_loss", return_value=600.0), \
         patch("src.dashboard.get_monthly_loss", return_value=1100.0):
        show_loss_limits_panel(paper=True, config=config)
    mock_console.print.assert_called_once()


def test_show_loss_limits_panel_no_config(mock_console):
    from src.dashboard import show_loss_limits_panel
    with patch("src.dashboard.get_daily_loss", return_value=0.0), \
         patch("src.dashboard.get_weekly_loss", return_value=0.0), \
         patch("src.dashboard.get_monthly_loss", return_value=0.0):
        show_loss_limits_panel(paper=True, config=None)
    mock_console.print.assert_called_once()


# ── show_var_panel ────────────────────────────────────────────────────────────

def test_show_var_panel_empty(mock_console):
    from src.dashboard import show_var_panel
    show_var_panel({})
    mock_console.print.assert_not_called()


def test_show_var_panel_with_data(mock_console):
    from src.dashboard import show_var_panel
    var = {"total_exposure": 100.0, "expected_loss": 30.0,
           "var_95": 55.0, "var_99": 80.0, "pct_of_bankroll": 20.0}
    show_var_panel(var)
    mock_console.print.assert_called_once()


def test_show_var_panel_zero_exposure(mock_console):
    from src.dashboard import show_var_panel
    show_var_panel({"total_exposure": 0})
    mock_console.print.assert_not_called()


# ── show_backtest_results ─────────────────────────────────────────────────────

def test_show_backtest_results_success(mock_console):
    from src.dashboard import show_backtest_results
    results = {
        "total_trades": 30, "win_rate": 0.55, "total_pnl_usdc": 120.0,
        "roi_pct": 12.0, "sharpe_ratio": 1.5, "max_drawdown_pct": 8.0,
        "final_bankroll": 1120.0, "avg_ev": 0.07, "avg_confidence": 0.75,
    }
    show_backtest_results(results)
    mock_console.print.assert_called_once()


def test_show_backtest_results_error(mock_console):
    from src.dashboard import show_backtest_results
    show_backtest_results({"error": "No data"})
    mock_console.print.assert_called_once()


def test_show_backtest_results_empty(mock_console):
    from src.dashboard import show_backtest_results
    show_backtest_results({})
    mock_console.print.assert_called_once()


# ── show_model_status ─────────────────────────────────────────────────────────

def test_show_model_status_unfitted(mock_console):
    from src.dashboard import show_model_status
    mock_tracker = MagicMock()
    mock_tracker.get_calibration_info.return_value = {
        "method": "platt", "fitted": False, "meta": False
    }
    mock_tracker.detect_drift.return_value = None
    with patch("src.model_tracker.get_tracker", return_value=mock_tracker):
        show_model_status()
    mock_console.print.assert_called_once()


def test_show_model_status_fitted_platt(mock_console):
    from src.dashboard import show_model_status
    mock_tracker = MagicMock()
    mock_tracker.get_calibration_info.return_value = {
        "method": "platt", "fitted": True, "meta": True, "A": -1.2, "B": 0.5
    }
    mock_tracker.detect_drift.return_value = None
    with patch("src.model_tracker.get_tracker", return_value=mock_tracker):
        show_model_status()
    mock_console.print.assert_called_once()


def test_show_model_status_drift_warning(mock_console):
    from src.dashboard import show_model_status
    mock_tracker = MagicMock()
    mock_tracker.get_calibration_info.return_value = {
        "method": "isotonic", "fitted": True, "meta": True, "iso_pts": 5
    }
    mock_tracker.detect_drift.return_value = "Brier score increasing"
    with patch("src.model_tracker.get_tracker", return_value=mock_tracker):
        show_model_status()
    mock_console.print.assert_called_once()


def test_show_model_status_exception_silenced(mock_console):
    from src.dashboard import show_model_status
    with patch("src.model_tracker.get_tracker", side_effect=Exception("tracker error")):
        show_model_status()
    mock_console.print.assert_not_called()
