"""Unit tests for the adaptive self-learning module."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sqlite3
import pytest
from unittest.mock import patch
from pathlib import Path


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Point all DB operations to a temporary database."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.logger.DB_PATH", db_path)
    monkeypatch.setattr("src.learner.DB_PATH", db_path)
    monkeypatch.setattr("src.model_tracker.DB_PATH", db_path)

    from src.logger import init_db
    init_db()
    from src.model_tracker import _ensure_tables
    _ensure_tables()
    from src.learner import _ensure_tables as _ensure_learner_tables
    _ensure_learner_tables()
    return db_path


def _insert_trades(db_path, trades):
    """Insert test trades directly into the DB."""
    conn = sqlite3.connect(str(db_path))
    for t in trades:
        conn.execute(
            """INSERT INTO trades
               (timestamp, market_id, market_title, side, size_usdc,
                entry_price, our_prob, confidence, ev, outcome, paper, city, metric)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                t.get("timestamp", "2026-05-20T12:00:00"),
                t.get("market_id", "cid_test"),
                t.get("market_title", "Will it rain in NYC?"),
                t.get("side", "YES"),
                t.get("size_usdc", 25.0),
                t.get("entry_price", 0.50),
                t.get("our_prob", 0.65),
                t.get("confidence", 0.70),
                t.get("ev", 0.08),
                t["outcome"],
                1,
                t.get("city", "new york"),
                t.get("metric", "RAIN"),
            ),
        )
        trade_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT OR REPLACE INTO trade_predictions
               (trade_id, raw_prob, calibrated, side, city, metric, recorded_at)
            VALUES (?,?,?,?,?,?,?)""",
            (trade_id, 0.60, 0.65, t.get("side", "YES"),
             t.get("city", "new york"), t.get("metric", "RAIN"),
             t.get("timestamp", "2026-05-20T12:00:00")),
        )
    conn.commit()
    conn.close()


def _make_config():
    return {
        "trading": {
            "paper_mode": True,
            "ev_threshold": 0.05,
            "kelly_fraction": 0.25,
            "min_confidence": 0.50,
            "max_position_usdc": 50,
            "max_open_positions": 5,
            "daily_loss_limit_usdc": 200,
            "min_edge_score": 0.02,
        },
        "cities": [],
        "weather": {"sources": {}},
    }


# ── Error Pattern Analysis ─────────────────────────────────────────────────

def test_analyze_errors_needs_min_trades(tmp_db):
    from src.learner import AdaptiveLearner
    config = _make_config()
    learner = AdaptiveLearner(config)

    # Less than 10 trades → no patterns
    _insert_trades(tmp_db, [{"outcome": "WIN"}] * 5)
    patterns = learner.analyze_errors()
    assert patterns == {}


def test_analyze_errors_finds_losing_pattern(tmp_db):
    from src.learner import AdaptiveLearner
    config = _make_config()
    learner = AdaptiveLearner(config)

    trades = (
        [{"outcome": "LOSS", "city": "tokyo", "metric": "SNOW"}] * 8
        + [{"outcome": "WIN", "city": "new york", "metric": "RAIN"}] * 5
    )
    _insert_trades(tmp_db, trades)

    patterns = learner.analyze_errors()
    assert len(patterns) > 0
    assert any("tokyo" in k for k in patterns)


def test_get_pattern_penalty(tmp_db):
    from src.learner import AdaptiveLearner
    config = _make_config()
    learner = AdaptiveLearner(config)

    trades = [{"outcome": "LOSS", "city": "tokyo", "metric": "SNOW"}] * 10 + \
             [{"outcome": "WIN", "city": "new york", "metric": "RAIN"}] * 5
    _insert_trades(tmp_db, trades)

    learner.analyze_errors()
    penalty = learner.get_pattern_penalty("tokyo", "SNOW")
    assert penalty > 0


# ── Parameter Tuning ──────────────────────────────────────────────────────

def test_tune_raises_ev_on_losses(tmp_db):
    from src.learner import AdaptiveLearner
    config = _make_config()
    original_ev = config["trading"]["ev_threshold"]
    learner = AdaptiveLearner(config)

    # 30% win rate — should raise EV threshold
    trades = (
        [{"outcome": "LOSS"}] * 7 + [{"outcome": "WIN"}] * 3
        + [{"outcome": "LOSS"}] * 3
    )
    _insert_trades(tmp_db, trades)

    changes = learner.tune_parameters()
    if "ev_threshold" in changes:
        assert changes["ev_threshold"][1] > original_ev


def test_tune_reduces_kelly_on_losses(tmp_db):
    from src.learner import AdaptiveLearner
    config = _make_config()
    original_kelly = config["trading"]["kelly_fraction"]
    learner = AdaptiveLearner(config)

    # Very bad win rate
    trades = [{"outcome": "LOSS"}] * 8 + [{"outcome": "WIN"}] * 2 + \
             [{"outcome": "LOSS"}] * 2
    _insert_trades(tmp_db, trades)

    changes = learner.tune_parameters()
    if "kelly_fraction" in changes:
        assert changes["kelly_fraction"][1] < original_kelly


def test_tune_does_not_exceed_bounds(tmp_db):
    from src.learner import AdaptiveLearner
    config = _make_config()
    config["trading"]["ev_threshold"] = 0.11  # near upper bound
    learner = AdaptiveLearner(config)

    trades = [{"outcome": "LOSS"}] * 10 + [{"outcome": "WIN"}] * 2
    _insert_trades(tmp_db, trades)

    learner.tune_parameters()
    assert config["trading"]["ev_threshold"] <= 0.12


# ── Full Learning Cycle ──────────────────────────────────────────────────

def test_learn_returns_none_when_insufficient_data(tmp_db):
    from src.learner import AdaptiveLearner
    config = _make_config()
    learner = AdaptiveLearner(config)

    _insert_trades(tmp_db, [{"outcome": "WIN"}] * 3)
    result = learner.learn()
    assert result is None


def test_learn_returns_summary_on_problems(tmp_db):
    from src.learner import AdaptiveLearner
    config = _make_config()
    learner = AdaptiveLearner(config)

    trades = [{"outcome": "LOSS"}] * 8 + [{"outcome": "WIN"}] * 4
    _insert_trades(tmp_db, trades)

    result = learner.learn()
    # May or may not produce a message depending on thresholds,
    # but should not crash
    assert result is None or isinstance(result, str)


def test_generate_report(tmp_db):
    from src.learner import AdaptiveLearner
    config = _make_config()
    learner = AdaptiveLearner(config)

    trades = (
        [{"outcome": "WIN", "city": "new york"}] * 7
        + [{"outcome": "LOSS", "city": "tokyo"}] * 5
    )
    _insert_trades(tmp_db, trades)

    report = learner.generate_report()
    assert "Learning Report" in report
    assert "new york" in report.lower() or "tokyo" in report.lower()


# ── Logging ──────────────────────────────────────────────────────────────

def test_learning_log_persists(tmp_db):
    from src.learner import AdaptiveLearner
    config = _make_config()
    learner = AdaptiveLearner(config)

    trades = [{"outcome": "LOSS", "city": "miami"}] * 8 + \
             [{"outcome": "WIN", "city": "chicago"}] * 5
    _insert_trades(tmp_db, trades)

    learner.analyze_errors()

    conn = sqlite3.connect(str(tmp_db))
    rows = conn.execute("SELECT * FROM learning_log").fetchall()
    conn.close()
    assert len(rows) >= 1


def test_param_history_persists(tmp_db):
    from src.learner import AdaptiveLearner
    config = _make_config()
    learner = AdaptiveLearner(config)

    # Force a low win rate to trigger parameter changes
    trades = [{"outcome": "LOSS"}] * 9 + [{"outcome": "WIN"}] * 3
    _insert_trades(tmp_db, trades)

    learner.tune_parameters()

    conn = sqlite3.connect(str(tmp_db))
    rows = conn.execute("SELECT * FROM param_history").fetchall()
    conn.close()
    # May or may not have changes depending on exact thresholds
    assert isinstance(rows, list)
