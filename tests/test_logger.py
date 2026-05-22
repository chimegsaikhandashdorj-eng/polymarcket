"""Unit tests for logger.py — SQLite CRUD and analytics."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

import src.logger as logger_mod


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temporary file for every test."""
    db = tmp_path / "test_trades.db"
    monkeypatch.setattr(logger_mod, "DB_PATH", db)
    logger_mod.init_db()
    yield db


def _trade(**kwargs):
    defaults = dict(
        market_id="mkt1", market_title="Will it rain?", side="YES",
        size_usdc=10.0, entry_price=0.50, our_prob=0.70,
        confidence=0.80, ev=0.40, paper=True,
        city="new york", metric="RAIN", expiry="2025-05-05",
    )
    defaults.update(kwargs)
    with patch("src.notifier.notify_trade"):
        return logger_mod.log_trade(**defaults)


# ── init_db ───────────────────────────────────────────────────────────────────

def test_init_db_idempotent():
    logger_mod.init_db()
    logger_mod.init_db()
    open_pos = logger_mod.get_open_positions(paper=True)
    assert isinstance(open_pos, list)


# ── log_trade ─────────────────────────────────────────────────────────────────

def test_log_trade_returns_integer_id():
    tid = _trade()
    assert isinstance(tid, int) and tid >= 1


def test_log_trade_appears_in_open_positions():
    _trade(market_id="mktA")
    pos = logger_mod.get_open_positions(paper=True)
    assert any(p["market_id"] == "mktA" for p in pos)


def test_log_trade_paper_false_not_in_paper_positions():
    _trade(paper=False, market_id="live_mkt")
    paper_pos = logger_mod.get_open_positions(paper=True)
    assert not any(p["market_id"] == "live_mkt" for p in paper_pos)
    live_pos = logger_mod.get_open_positions(paper=False)
    assert any(p["market_id"] == "live_mkt" for p in live_pos)


# ── update_outcome ────────────────────────────────────────────────────────────

def test_update_outcome_win_generates_positive_pnl():
    tid = _trade(size_usdc=10.0, entry_price=0.50)
    with patch("src.notifier.notify_outcome"):
        logger_mod.update_outcome(tid, 1.0, "WIN")
    summary = logger_mod.get_pnl_summary(paper=True)
    assert summary["wins"] == 1
    assert summary["total_pnl_usdc"] > 0


def test_update_outcome_loss_generates_negative_pnl():
    tid = _trade(size_usdc=10.0, entry_price=0.50)
    with patch("src.notifier.notify_outcome"):
        logger_mod.update_outcome(tid, 0.0, "LOSS")
    summary = logger_mod.get_pnl_summary(paper=True)
    assert summary["losses"] == 1
    assert summary["total_pnl_usdc"] < 0


def test_update_outcome_void_zero_pnl():
    tid = _trade(size_usdc=10.0, entry_price=0.50)
    with patch("src.notifier.notify_outcome"):
        logger_mod.update_outcome(tid, 0.5, "VOID")
    summary = logger_mod.get_pnl_summary(paper=True)
    assert summary["voids"] == 1
    assert summary["total_pnl_usdc"] == 0.0


def test_update_outcome_nonexistent_id_no_crash():
    with patch("src.notifier.notify_outcome"):
        logger_mod.update_outcome(99999, 1.0, "WIN")


def test_resolved_trade_no_longer_in_open_positions():
    tid = _trade()
    with patch("src.notifier.notify_outcome"):
        logger_mod.update_outcome(tid, 1.0, "WIN")
    pos = logger_mod.get_open_positions(paper=True)
    assert not any(p["id"] == tid for p in pos)


# ── get_pnl_summary ───────────────────────────────────────────────────────────

def test_pnl_summary_empty_db():
    s = logger_mod.get_pnl_summary(paper=True)
    assert s["total_trades"] == 0
    assert s["win_rate"] == 0.0
    assert s["total_pnl_usdc"] == 0.0


def test_pnl_summary_win_rate():
    t1 = _trade()
    t2 = _trade(market_id="mkt2")
    with patch("src.notifier.notify_outcome"):
        logger_mod.update_outcome(t1, 1.0, "WIN")
        logger_mod.update_outcome(t2, 0.0, "LOSS")
    s = logger_mod.get_pnl_summary(paper=True)
    assert s["wins"] == 1
    assert s["losses"] == 1
    assert abs(s["win_rate"] - 0.5) < 0.001


def test_pnl_summary_paper_none_includes_all():
    _trade(paper=True)
    _trade(paper=False, market_id="live")
    s = logger_mod.get_pnl_summary(paper=None)
    assert s["total_trades"] == 0  # both OPEN, not counted in resolved summary


# ── get_daily_loss / weekly / monthly ─────────────────────────────────────────

def test_get_daily_loss_no_trades():
    assert logger_mod.get_daily_loss(paper=True) == 0.0


def test_get_daily_loss_after_win_is_zero():
    tid = _trade()
    with patch("src.notifier.notify_outcome"):
        logger_mod.update_outcome(tid, 1.0, "WIN")
    assert logger_mod.get_daily_loss(paper=True) == 0.0


def test_get_daily_loss_after_loss_positive():
    tid = _trade(size_usdc=20.0)
    with patch("src.notifier.notify_outcome"):
        logger_mod.update_outcome(tid, 0.0, "LOSS")
    assert logger_mod.get_daily_loss(paper=True) == 20.0


def test_get_weekly_loss_no_trades():
    assert logger_mod.get_weekly_loss(paper=True) == 0.0


def test_get_monthly_loss_no_trades():
    assert logger_mod.get_monthly_loss(paper=True) == 0.0


def test_get_realized_pnl_empty():
    assert logger_mod.get_realized_pnl(paper=True) == 0.0


def test_get_realized_pnl_after_win():
    tid = _trade(size_usdc=10.0, entry_price=0.50)
    with patch("src.notifier.notify_outcome"):
        logger_mod.update_outcome(tid, 1.0, "WIN")
    pnl = logger_mod.get_realized_pnl(paper=True)
    assert pnl > 0


# ── weather cache ─────────────────────────────────────────────────────────────

def test_cache_weather_and_retrieve():
    logger_mod.cache_weather(40.7, -74.0, "2026-05-05",
                             0.3, 20.0, 60.0, 15.0, 0.85, ["tomorrow_io"])
    result = logger_mod.get_cached_weather(40.7, -74.0, "2026-05-05")
    assert result is not None
    assert abs(result["precip_prob"] - 0.3) < 0.001
    assert result["regime"] == "NORMAL"


def _insert_stale_weather(lat, lon, target_dt):
    """Insert a weather row with a historical fetched_at timestamp."""
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(str(logger_mod.DB_PATH))
    conn.execute(
        "INSERT INTO weather_cache "
        "(lat, lon, target_dt, fetched_at, precip_prob, temp_c, humidity, wind_kph, confidence, sources, regime) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (lat, lon, target_dt, "2000-01-01T00:00:00", 0.3, 20.0, 60.0, 15.0, 0.85, "open_meteo", "NORMAL"),
    )
    conn.commit()
    conn.close()


def test_cache_weather_expired_returns_none():
    _insert_stale_weather(40.7, -74.0, "2000-01-01")
    result = logger_mod.get_cached_weather(40.7, -74.0, "2000-01-01", ttl_seconds=1)
    assert result is None


def test_cache_weather_miss_returns_none():
    result = logger_mod.get_cached_weather(0.0, 0.0, "2099-01-01")
    assert result is None


def test_purge_stale_weather_cache():
    _insert_stale_weather(40.7, -74.0, "2000-01-01")
    deleted = logger_mod.purge_stale_weather_cache(max_age_hours=1)
    assert deleted >= 1


def test_purge_fresh_cache_not_deleted():
    logger_mod.cache_weather(40.7, -74.0, "2026-05-05",
                             0.3, 20.0, 60.0, 15.0, 0.85, ["open_meteo"])
    deleted = logger_mod.purge_stale_weather_cache(max_age_hours=24)
    assert deleted == 0


# ── get_pnl_volatility ────────────────────────────────────────────────────────

def test_pnl_volatility_no_trades():
    assert logger_mod.get_pnl_volatility(paper=True) == 0.0


def test_pnl_volatility_with_trades():
    for i in range(3):
        tid = _trade(market_id=f"mkt{i}", size_usdc=float(10 + i))
        with patch("src.notifier.notify_outcome"):
            logger_mod.update_outcome(tid, 1.0 if i % 2 == 0 else 0.0,
                                      "WIN" if i % 2 == 0 else "LOSS")
    vol = logger_mod.get_pnl_volatility(paper=True)
    assert vol >= 0.0


# ── setup_file_logging ────────────────────────────────────────────────────────

def test_setup_file_logging_creates_files(tmp_path):
    import logging
    log_dir = str(tmp_path / "testlogs")
    logger_mod.setup_file_logging(log_dir)
    assert (tmp_path / "testlogs" / "debug.log").exists()
    assert (tmp_path / "testlogs" / "error.log").exists()
    root = logging.getLogger()
    for h in root.handlers[:]:
        if isinstance(h, logging.FileHandler) and "testlogs" in h.baseFilename:
            h.close()
            root.removeHandler(h)
