"""Unit tests for resolver.py — _fetch_resolution and resolve_open_trades."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock
import requests

from src.resolver import _fetch_resolution, resolve_open_trades


def _market_response(resolved=True, closed=False, res_price=1.0, condition_id="cond1"):
    return [{
        "conditionId": condition_id,
        "resolved": resolved,
        "closed": closed,
        "resolutionPrice": res_price,
    }]


# ── _fetch_resolution ─────────────────────────────────────────────────────────

def test_fetch_resolution_empty_condition_id():
    resolved, price = _fetch_resolution("")
    assert not resolved
    assert price is None


def test_fetch_resolution_none_condition_id():
    resolved, price = _fetch_resolution(None)
    assert not resolved
    assert price is None


def test_fetch_resolution_yes_wins():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _market_response(resolved=True, res_price=1.0)
    mock_resp.raise_for_status = MagicMock()
    with patch("src.resolver._SESSION") as mock_session:
        mock_session.get.return_value = mock_resp
        resolved, price = _fetch_resolution("cond1")
    assert resolved
    assert price == 1.0


def test_fetch_resolution_no_wins():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _market_response(resolved=True, res_price=0.0)
    mock_resp.raise_for_status = MagicMock()
    with patch("src.resolver._SESSION") as mock_session:
        mock_session.get.return_value = mock_resp
        resolved, price = _fetch_resolution("cond1")
    assert resolved
    assert price == 0.0


def test_fetch_resolution_void():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _market_response(resolved=True, res_price=0.5)
    mock_resp.raise_for_status = MagicMock()
    with patch("src.resolver._SESSION") as mock_session:
        mock_session.get.return_value = mock_resp
        resolved, price = _fetch_resolution("cond1")
    assert resolved
    assert price == pytest.approx(0.5)


def test_fetch_resolution_not_yet_resolved():
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"conditionId": "cond1", "resolved": False, "closed": False}]
    mock_resp.raise_for_status = MagicMock()
    with patch("src.resolver._SESSION") as mock_session:
        mock_session.get.return_value = mock_resp
        resolved, price = _fetch_resolution("cond1")
    assert not resolved


def test_fetch_resolution_no_resolution_price():
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"conditionId": "cond1", "resolved": True, "resolutionPrice": None}]
    mock_resp.raise_for_status = MagicMock()
    with patch("src.resolver._SESSION") as mock_session:
        mock_session.get.return_value = mock_resp
        resolved, price = _fetch_resolution("cond1")
    assert not resolved


def test_fetch_resolution_empty_response():
    mock_resp = MagicMock()
    mock_resp.json.return_value = []
    mock_resp.raise_for_status = MagicMock()
    with patch("src.resolver._SESSION") as mock_session:
        mock_session.get.return_value = mock_resp
        resolved, price = _fetch_resolution("cond1")
    assert not resolved


def test_fetch_resolution_http_404_not_retried():
    http_err = requests.HTTPError(response=MagicMock(status_code=404))
    with patch("src.resolver._SESSION") as mock_session:
        mock_session.get.side_effect = http_err
        resolved, price = _fetch_resolution("cond1")
    assert not resolved
    assert mock_session.get.call_count == 1  # no retry


def test_fetch_resolution_http_500_retries_exhausted():
    http_err = requests.HTTPError(response=MagicMock(status_code=500))
    with patch("src.resolver._SESSION") as mock_session, \
         patch("src.resolver.time.sleep"):
        mock_session.get.side_effect = http_err
        resolved, price = _fetch_resolution("cond1")
    assert not resolved
    assert mock_session.get.call_count == 3  # 3 retries


def test_fetch_resolution_network_error_retries():
    with patch("src.resolver._SESSION") as mock_session, \
         patch("src.resolver.time.sleep"):
        mock_session.get.side_effect = ConnectionError("timeout")
        resolved, price = _fetch_resolution("cond1")
    assert not resolved
    assert mock_session.get.call_count == 3


def test_fetch_resolution_bad_json_type():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"error": "bad"}  # not a list
    mock_resp.raise_for_status = MagicMock()
    with patch("src.resolver._SESSION") as mock_session:
        mock_session.get.return_value = mock_resp
        resolved, price = _fetch_resolution("cond1")
    assert not resolved


def test_fetch_resolution_closed_market():
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"resolved": False, "closed": True, "resolutionPrice": 1.0}]
    mock_resp.raise_for_status = MagicMock()
    with patch("src.resolver._SESSION") as mock_session:
        mock_session.get.return_value = mock_resp
        resolved, price = _fetch_resolution("cond1")
    assert resolved
    assert price == 1.0


# ── resolve_open_trades ───────────────────────────────────────────────────────

def test_resolve_open_trades_no_open_positions():
    with patch("src.resolver.get_open_positions", return_value=[]):
        count = resolve_open_trades(paper=True)
    assert count == 0


def test_resolve_open_trades_yes_win():
    positions = [{"id": 1, "market_id": "cond1", "side": "YES",
                  "market_title": "Will it rain?", "size_usdc": 10.0}]
    with patch("src.resolver.get_open_positions", return_value=positions), \
         patch("src.resolver._fetch_resolution", return_value=(True, 1.0)), \
         patch("src.resolver.update_outcome") as mock_update:
        count = resolve_open_trades(paper=True)
    assert count == 1
    mock_update.assert_called_once_with(1, 1.0, "WIN")


def test_resolve_open_trades_yes_loss():
    positions = [{"id": 2, "market_id": "cond2", "side": "YES",
                  "market_title": "Will it snow?", "size_usdc": 10.0}]
    with patch("src.resolver.get_open_positions", return_value=positions), \
         patch("src.resolver._fetch_resolution", return_value=(True, 0.0)), \
         patch("src.resolver.update_outcome") as mock_update:
        count = resolve_open_trades(paper=True)
    assert count == 1
    mock_update.assert_called_once_with(2, 0.0, "LOSS")


def test_resolve_open_trades_no_side_win():
    positions = [{"id": 3, "market_id": "cond3", "side": "NO",
                  "market_title": "Will it rain?", "size_usdc": 10.0}]
    with patch("src.resolver.get_open_positions", return_value=positions), \
         patch("src.resolver._fetch_resolution", return_value=(True, 0.0)), \
         patch("src.resolver.update_outcome") as mock_update:
        count = resolve_open_trades(paper=True)
    assert count == 1
    mock_update.assert_called_once_with(3, 0.0, "WIN")


def test_resolve_open_trades_void():
    positions = [{"id": 4, "market_id": "cond4", "side": "YES",
                  "market_title": "Will it rain?", "size_usdc": 10.0}]
    with patch("src.resolver.get_open_positions", return_value=positions), \
         patch("src.resolver._fetch_resolution", return_value=(True, 0.5)), \
         patch("src.resolver.update_outcome") as mock_update:
        count = resolve_open_trades(paper=True)
    assert count == 1
    mock_update.assert_called_once_with(4, 0.5, "VOID")


def test_resolve_open_trades_still_open():
    positions = [{"id": 5, "market_id": "cond5", "side": "YES",
                  "market_title": "Will it rain?", "size_usdc": 10.0}]
    with patch("src.resolver.get_open_positions", return_value=positions), \
         patch("src.resolver._fetch_resolution", return_value=(False, None)), \
         patch("src.resolver.update_outcome") as mock_update:
        count = resolve_open_trades(paper=True)
    assert count == 0
    mock_update.assert_not_called()


def test_resolve_open_trades_partial():
    positions = [
        {"id": 6, "market_id": "cond6", "side": "YES", "market_title": "Rain?", "size_usdc": 10.0},
        {"id": 7, "market_id": "cond7", "side": "NO",  "market_title": "Snow?", "size_usdc": 10.0},
    ]
    fetch_results = [(True, 1.0), (False, None)]
    with patch("src.resolver.get_open_positions", return_value=positions), \
         patch("src.resolver._fetch_resolution", side_effect=fetch_results), \
         patch("src.resolver.update_outcome") as mock_update:
        count = resolve_open_trades(paper=True)
    assert count == 1
    assert mock_update.call_count == 1
