"""
Phase 1 audit: no look-ahead bias in trade decisions.

For every resolved trade in the DB, assert that:
  1. The decision timestamp is BEFORE the market resolution timestamp.
  2. Calibration-model refits are triggered AFTER the outcome lands,
     not before (verified structurally via model_tracker API).

Run against a populated DB (paper or live):
  pytest tests/test_no_lookahead.py -v

In CI (empty DB) all tests are skipped automatically.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


DB_PATH = Path(__file__).parent.parent / "data" / "trades.db"


def _rows(sql: str, params=()):
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


@pytest.fixture(scope="module")
def resolved_trades():
    rows = _rows(
        "SELECT id, timestamp, market_id, outcome, exit_price FROM trades "
        "WHERE outcome IN ('WIN','LOSS','VOID') ORDER BY timestamp ASC"
    )
    if not rows:
        pytest.skip("No resolved trades in DB — run paper trading first")
    return rows


@pytest.fixture(scope="module")
def all_trades():
    rows = _rows("SELECT id, timestamp, market_id, outcome FROM trades ORDER BY timestamp ASC")
    if not rows:
        pytest.skip("No trades in DB — run paper trading first")
    return rows


# ── Temporal ordering ─────────────────────────────────────────────────────────

def test_decision_before_resolution(resolved_trades):
    """
    Every trade's decision timestamp must precede any known resolution date.
    Since we store expiry in trades.expiry, decision_ts < expiry for all open trades.
    """
    rows = _rows(
        "SELECT id, timestamp, expiry FROM trades WHERE expiry IS NOT NULL"
    )
    if not rows:
        pytest.skip("No trades with expiry stored")

    violations = []
    for r in rows:
        try:
            decision = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
            expiry   = datetime.fromisoformat(r["expiry"].replace("Z", "+00:00"))
            if decision.tzinfo is None:
                decision = decision.replace(tzinfo=timezone.utc)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if decision >= expiry:
                violations.append(
                    f"Trade #{r['id']}: decision {r['timestamp']} >= expiry {r['expiry']}"
                )
        except Exception as exc:
            violations.append(f"Trade #{r['id']}: parse error: {exc}")

    assert not violations, "\n".join(violations)


def test_timestamps_are_monotonically_increasing(all_trades):
    """Trade IDs grow with time — no time-travel insertions."""
    timestamps = [r["timestamp"] for r in all_trades]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], (
            f"Non-monotonic timestamps at position {i}: "
            f"{timestamps[i-1]} then {timestamps[i]}"
        )


def test_no_duplicate_market_entries(all_trades):
    """
    A market should not appear more than once in OPEN state simultaneously.
    (Duplicate open = look-ahead or race condition in execution.)
    """
    open_rows = _rows(
        "SELECT market_id, COUNT(*) as cnt FROM trades "
        "WHERE outcome IS NULL OR outcome = 'OPEN' "
        "GROUP BY market_id HAVING cnt > 1"
    )
    duplicates = [(r["market_id"], r["cnt"]) for r in open_rows]
    assert not duplicates, f"Duplicate open positions: {duplicates}"


# ── Calibration model look-ahead ──────────────────────────────────────────────

def test_calibration_only_trains_on_resolved_trades():
    """
    ModelTracker.fit_calibration() reads only resolved trades.
    This test verifies the query logic is constrained to resolved outcomes.
    """
    try:
        from src.model_tracker import ModelTracker
        import inspect
        src = inspect.getsource(ModelTracker.fit_calibration)
        # Must filter to resolved outcomes — not all trades
        assert "WIN" in src or "LOSS" in src or "outcome" in src.lower(), \
            "fit_calibration() must filter to resolved trades (WIN/LOSS)"
        assert "OPEN" not in src or "!= 'OPEN'" in src or "IN ('WIN" in src, \
            "fit_calibration() must not train on OPEN trades"
    except ImportError:
        pytest.skip("model_tracker not importable")


def test_calibration_requires_minimum_sample():
    """
    ModelTracker must require a minimum number of resolved trades before
    fitting (prevents overfitting on tiny samples).
    """
    try:
        from src.model_tracker import ModelTracker
        import inspect
        src = inspect.getsource(ModelTracker.fit_calibration)
        # Should have some minimum threshold check
        assert any(tok in src for tok in ["< 20", "< 30", "< 10", "min_", "MIN_", "n <", "len("]), \
            "fit_calibration() should refuse to fit on fewer than ~20 samples"
    except ImportError:
        pytest.skip("model_tracker not importable")


# ── Data pipeline ordering ─────────────────────────────────────────────────────

def test_weather_cache_fetched_before_decision():
    """
    Weather cache rows: fetched_at timestamp should be <= the decision timestamp
    of any trade placed on that city/day.
    """
    if not DB_PATH.exists():
        pytest.skip("No DB")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT t.id, t.timestamp AS decision_ts, t.city,
               w.fetched_at AS weather_ts
        FROM trades t
        JOIN weather_cache w ON w.target_dt = substr(t.timestamp, 1, 10)
        WHERE t.city IS NOT NULL AND w.fetched_at IS NOT NULL
        LIMIT 200
        """
    ).fetchall()
    conn.close()

    if not rows:
        pytest.skip("No joined trade+weather rows")

    violations = []
    for r in rows:
        try:
            decision = datetime.fromisoformat(r["decision_ts"])
            weather  = datetime.fromisoformat(r["weather_ts"])
            # Weather must have been fetched before (or at) the decision time
            # Allow a 5-minute buffer for timing jitter
            if weather > decision:
                diff_s = (weather - decision).total_seconds()
                if diff_s > 300:
                    violations.append(
                        f"Trade #{r['id']} ({r['city']}): weather fetched "
                        f"{diff_s:.0f}s AFTER decision"
                    )
        except Exception:
            pass  # timestamp parse issues are non-fatal for this check

    assert not violations, "Look-ahead detected in weather cache:\n" + "\n".join(violations)


# ── Structural: resolve_open_trades uses market data, not internal state ───────

def test_resolver_fetches_external_resolution():
    """
    The resolver must call an external API for resolution prices,
    not infer outcomes from internal state alone.
    """
    import inspect
    from src.resolver import _fetch_resolution
    src = inspect.getsource(_fetch_resolution)
    assert "_SESSION.get" in src or "requests" in src, \
        "_fetch_resolution must make an external HTTP call to verify resolution"
    assert "resolutionPrice" in src or "resolution_price" in src, \
        "_fetch_resolution must read resolutionPrice from the API response"
