"""Unit tests for the TelegramCommander class."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock

from src import telegram_cmd
from src.telegram_cmd import TelegramCommander


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config():
    return {
        "trading": {
            "paper_mode": True,
            "max_open_positions": 5,
            "kelly_fraction": 0.15,
            "max_position_usdc": 25,
            "ev_threshold": 0.08,
            "min_confidence": 0.65,
            "edge_confirmations": 2,
            "poll_interval_seconds": 900,
        },
        "risk": {
            "max_city_exposure_pct": 0.20,
            "max_portfolio_exposure_pct": 0.40,
            "drawdown_halt_pct": 0.20,
        },
        "loss_limits": {
            "daily_usdc": 75,
            "weekly_usdc": 200,
            "monthly_usdc": 400,
        },
        "markets": {
            "min_liquidity_usdc": 10000,
            "min_hours_to_expiry": 2,
            "max_hours_to_expiry": 168,
        },
        "cities": [
            {"name": "New York", "lat": 40.7128, "lon": -74.0060},
            {"name": "London", "lat": 51.5074, "lon": -0.1278},
        ],
        "crypto": {
            "enabled": True,
            "paper_only_until": "2026-06-05",
            "assets": ["bitcoin", "ethereum"],
            "max_position_usdc": 5,
            "daily_loss_limit_usdc": 15,
            "min_ev_threshold": 0.12,
            "min_confidence": 0.75,
            "max_volatility_daily": 0.12,
            "max_spread": 0.04,
            "edge_confirmations": 3,
        },
    }


@pytest.fixture(autouse=True)
def patched_send(monkeypatch):
    """Capture every _send and _send_with_keyboard call so we can assert on outgoing text."""
    sent: list = []

    def fake_send(text, *, silent=None):
        sent.append({"text": text, "silent": silent})
        return True

    def fake_send_kb(text, keyboard, *, silent=None):
        sent.append({"text": text, "silent": silent, "keyboard": keyboard})
        return True

    monkeypatch.setattr(telegram_cmd, "_send", fake_send)
    monkeypatch.setattr(telegram_cmd, "_send_with_keyboard", fake_send_kb)
    return sent


@pytest.fixture
def commander(base_config):
    return TelegramCommander(base_config)


# ── Initialization ────────────────────────────────────────────────────────────

class TestInit:
    def test_default_state(self, commander):
        assert commander.is_paused is False
        assert commander._paper is True
        assert commander._offset == 0
        assert commander._running is False

    def test_commands_registered(self, commander):
        expected = {
            "/help", "/start", "/status", "/pnl", "/positions", "/pos",
            "/balance", "/bal", "/scan", "/weather", "/risk", "/limits",
            "/var", "/history", "/crypto", "/regime", "/config",
            "/pending", "/learn", "/version", "/pause", "/resume",
        }
        assert expected.issubset(set(commander._commands.keys()))

    def test_aliases_share_handler(self, commander):
        assert commander._commands["/positions"] == commander._commands["/pos"]
        assert commander._commands["/balance"] == commander._commands["/bal"]
        assert commander._commands["/help"] == commander._commands["/start"]


# ── Pause / resume ────────────────────────────────────────────────────────────

class TestPauseResume:
    def test_pause(self, commander, patched_send):
        commander._cmd_pause([])
        assert commander.is_paused is True
        assert "paused" in patched_send[0]["text"].lower()

    def test_resume(self, commander, patched_send):
        commander._paused = True
        commander._cmd_resume([])
        assert commander.is_paused is False
        assert "resumed" in patched_send[0]["text"].lower()


# ── Risk / limits / config commands ──────────────────────────────────────────

class TestRiskCommands:
    def test_risk_shows_kelly(self, commander, patched_send):
        commander._cmd_risk([])
        text = patched_send[0]["text"]
        assert "Kelly" in text
        assert "15%" in text

    def test_limits_renders_bars(self, commander, patched_send):
        with patch("src.telegram_cmd.get_daily_loss", return_value=20.0), \
             patch("src.telegram_cmd.get_weekly_loss", return_value=50.0), \
             patch("src.telegram_cmd.get_monthly_loss", return_value=100.0):
            commander._cmd_limits([])
        text = patched_send[0]["text"]
        assert "Daily" in text and "Weekly" in text and "Monthly" in text
        # Emoji bars present (green/yellow/red squares + white squares)
        assert "\U0001f7e9" in text or "\U0001f7e8" in text or "\U0001f7e5" in text

    def test_config_view(self, commander, patched_send):
        commander._cmd_config([])
        text = patched_send[0]["text"]
        assert "Paper mode" in text
        assert "Poll interval" in text
        assert "Cities monitored: 2" in text

    def test_crypto_command(self, commander, patched_send):
        commander._cmd_crypto([])
        text = patched_send[0]["text"]
        assert "bitcoin" in text
        assert "$5" in text


# ── Progress bar helper ───────────────────────────────────────────────────────

class TestProgressBar:
    def test_empty(self):
        bar = TelegramCommander._progress_bar(0.0)
        assert bar == "[..........]"

    def test_full(self):
        bar = TelegramCommander._progress_bar(1.0)
        assert bar == "[##########]"

    def test_half(self):
        bar = TelegramCommander._progress_bar(0.5)
        assert bar.count("#") == 5
        assert bar.count(".") == 5

    def test_clamps_above_1(self):
        bar = TelegramCommander._progress_bar(2.0)
        assert bar == "[##########]"

    def test_clamps_below_0(self):
        bar = TelegramCommander._progress_bar(-0.5)
        assert bar == "[..........]"


# ── History command ───────────────────────────────────────────────────────────

class TestHistoryCommand:
    @patch("src.telegram_cmd.sqlite3")
    def test_history_empty(self, mock_sqlite, commander, patched_send):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_sqlite.connect.return_value = mock_conn
        commander._cmd_history([])
        assert any("No resolved trades" in s["text"] for s in patched_send)

    def test_history_clamps_n(self, commander, patched_send):
        # n out of range should clamp; just verify command doesn't crash
        with patch("src.telegram_cmd.sqlite3") as mock_sqlite:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_sqlite.connect.return_value = mock_conn
            commander._cmd_history(["9999"])
            commander._cmd_history(["abc"])  # non-numeric
        # Both calls should have produced sent messages without raising
        assert len(patched_send) == 2


# ── Help ──────────────────────────────────────────────────────────────────────

class TestHelp:
    def test_help_lists_commands(self, commander, patched_send):
        commander._cmd_help([])
        text = patched_send[0]["text"]
        # A representative sample of supported commands should appear
        for cmd in ["/status", "/pnl", "/positions",
                    "/crypto", "/pause", "/resume", "/config"]:
            assert cmd in text


# ── Dispatch / authorization ─────────────────────────────────────────────────

class TestDispatch:
    def test_unauthorized_chat_id_ignored(self, commander, patched_send, monkeypatch):
        monkeypatch.setattr(telegram_cmd, "_CHAT_IDS", ["999"])
        update = {
            "update_id": 1,
            "message": {"chat": {"id": "12345"}, "text": "/status"},
        }
        commander._dispatch_update(update)
        # Should NOT have sent anything since chat_id wasn't authorized
        assert patched_send == []

    def test_non_slash_message_ignored(self, commander, patched_send, monkeypatch):
        monkeypatch.setattr(telegram_cmd, "_CHAT_IDS", ["12345"])
        update = {
            "update_id": 2,
            "message": {"chat": {"id": "12345"}, "text": "hello bot"},
        }
        commander._dispatch_update(update)
        assert patched_send == []

    def test_unknown_command_replies(self, commander, patched_send, monkeypatch):
        monkeypatch.setattr(telegram_cmd, "_CHAT_IDS", ["12345"])
        # Bypass thread pool — invoke handler synchronously for the test
        with patch.object(commander._executor, "submit",
                          side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
            update = {
                "update_id": 3,
                "message": {"chat": {"id": "12345"}, "text": "/notarealcommand"},
            }
            commander._dispatch_update(update)
        assert any("Unknown command" in s["text"] for s in patched_send)


# ── Scan overlap protection ──────────────────────────────────────────────────

class TestScanOverlap:
    def test_scan_when_no_callback(self, base_config, patched_send):
        cmd = TelegramCommander(base_config, scan_callback=None)
        cmd._cmd_scan([])
        assert any("not available" in s["text"].lower() for s in patched_send)

    def test_overlapping_scan_rejected(self, base_config, patched_send):
        cmd = TelegramCommander(base_config, scan_callback=lambda: [])
        # Pre-acquire the inflight lock to simulate an in-progress scan
        cmd._scan_inflight.acquire()
        try:
            cmd._cmd_scan([])
        finally:
            cmd._scan_inflight.release()
        assert any("already in progress" in s["text"].lower() for s in patched_send)
