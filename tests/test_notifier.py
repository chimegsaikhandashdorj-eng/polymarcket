"""Unit tests for the Telegram notifier module."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock

from src import notifier


# ── Reset env between tests ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_env_and_state(monkeypatch):
    """Ensure each test starts with a clean configured state."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setenv("TELEGRAM_QUIET_START", "22")
    monkeypatch.setenv("TELEGRAM_QUIET_END", "7")
    notifier._reload_env()
    notifier._last_send_at = 0.0
    yield


# ── Configuration & gating ───────────────────────────────────────────────────

class TestConfiguration:
    def test_is_configured_true_when_both_set(self):
        assert notifier.is_configured() is True

    def test_is_configured_false_when_token_missing(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
        notifier._reload_env()
        assert notifier.is_configured() is False

    def test_is_configured_false_when_chat_missing(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
        notifier._reload_env()
        assert notifier.is_configured() is False

    def test_multi_chat_id_parsed(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "111, 222 , 333")
        notifier._reload_env()
        assert notifier._CHAT_IDS == ["111", "222", "333"]


# ── HTML escaping ─────────────────────────────────────────────────────────────

class TestHtmlEscape:
    def test_escape_amp(self):
        assert notifier.escape("Tom & Jerry") == "Tom &amp; Jerry"

    def test_escape_lt_gt(self):
        assert notifier.escape("<script>") == "&lt;script&gt;"

    def test_escape_none_safe(self):
        # escape should coerce non-strings to strings
        assert notifier.escape(42) == "42"


# ── Message chunking ─────────────────────────────────────────────────────────

class TestChunking:
    def test_short_message_single_chunk(self):
        chunks = notifier._chunk("hello")
        assert chunks == ["hello"]

    def test_long_message_split_at_newlines(self):
        # Build a message with lines that together exceed the limit
        line = "x" * 100
        text = "\n".join([line] * 60)  # 6000+ chars
        chunks = notifier._chunk(text, max_len=4096)
        assert len(chunks) >= 2
        # Each chunk is within the limit
        for c in chunks:
            assert len(c) <= 4096
        # Reassembling should yield original content (ignoring trim whitespace)
        assert "\n".join(chunks).count("x") == text.count("x")

    def test_hard_cut_when_no_newline(self):
        text = "a" * 5000
        chunks = notifier._chunk(text, max_len=4096)
        assert len(chunks) == 2
        assert len(chunks[0]) == 4096


# ── Quiet hours ───────────────────────────────────────────────────────────────

class TestQuietHours:
    @patch("src.notifier.datetime")
    def test_quiet_during_night_window(self, mock_dt, monkeypatch):
        monkeypatch.setenv("TELEGRAM_QUIET_START", "22")
        monkeypatch.setenv("TELEGRAM_QUIET_END", "7")
        notifier._reload_env()
        # 23:00 UTC should be quiet
        mock_dt.now.return_value.hour = 23
        assert notifier._is_quiet_hour() is True

    @patch("src.notifier.datetime")
    def test_not_quiet_during_day(self, mock_dt, monkeypatch):
        monkeypatch.setenv("TELEGRAM_QUIET_START", "22")
        monkeypatch.setenv("TELEGRAM_QUIET_END", "7")
        notifier._reload_env()
        mock_dt.now.return_value.hour = 12
        assert notifier._is_quiet_hour() is False

    @patch("src.notifier.datetime")
    def test_quiet_window_no_wrap(self, mock_dt, monkeypatch):
        # Same-day window: quiet from 1 to 6
        monkeypatch.setenv("TELEGRAM_QUIET_START", "1")
        monkeypatch.setenv("TELEGRAM_QUIET_END", "6")
        notifier._reload_env()
        mock_dt.now.return_value.hour = 3
        assert notifier._is_quiet_hour() is True
        mock_dt.now.return_value.hour = 9
        assert notifier._is_quiet_hour() is False

    @patch("src.notifier.datetime")
    def test_no_quiet_when_window_zero(self, mock_dt, monkeypatch):
        monkeypatch.setenv("TELEGRAM_QUIET_START", "0")
        monkeypatch.setenv("TELEGRAM_QUIET_END", "0")
        notifier._reload_env()
        mock_dt.now.return_value.hour = 3
        assert notifier._is_quiet_hour() is False


# ── Send logic with mocked requests.post ─────────────────────────────────────

class TestSend:
    @patch("src.notifier.requests.post")
    def test_send_skipped_when_not_configured(self, mock_post, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
        notifier._reload_env()
        assert notifier._send("hello") is False
        mock_post.assert_not_called()

    @patch("src.notifier.requests.post")
    def test_send_calls_telegram_api(self, mock_post):
        mock_post.return_value.status_code = 200
        result = notifier._send("hello", silent=True)
        assert result is True
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        assert payload["chat_id"] == "12345"
        assert payload["text"] == "hello"
        assert payload["parse_mode"] == "HTML"
        assert payload["disable_notification"] is True

    @patch("src.notifier.requests.post")
    def test_send_to_multiple_chats(self, mock_post, monkeypatch):
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "111,222")
        notifier._reload_env()
        mock_post.return_value.status_code = 200
        notifier._send("hello", silent=True)
        assert mock_post.call_count == 2

    @patch("src.notifier.requests.post")
    def test_send_returns_false_on_failure(self, mock_post):
        mock_post.return_value.status_code = 500
        # 5xx will retry 3 times then give up
        result = notifier._send("hello", silent=True)
        assert result is False
        assert mock_post.call_count == 3  # 3 retry attempts


# ── Retry & rate-limit logic ─────────────────────────────────────────────────

class TestRetry:
    @patch("src.notifier.time.sleep")
    @patch("src.notifier.requests.post")
    def test_429_honors_retry_after(self, mock_post, mock_sleep):
        # First call: 429 with retry_after; second: success
        resp_429 = MagicMock(status_code=429)
        resp_429.json.return_value = {"parameters": {"retry_after": 2}}
        resp_ok = MagicMock(status_code=200)
        mock_post.side_effect = [resp_429, resp_ok]

        result = notifier._post_with_retry({"chat_id": "x", "text": "y"})
        assert result is True
        assert mock_post.call_count == 2
        # Sleep should have been called with the retry_after value (capped at 30)
        mock_sleep.assert_any_call(2.0)

    @patch("src.notifier.time.sleep")
    @patch("src.notifier.requests.post")
    def test_5xx_retries_with_backoff(self, mock_post, mock_sleep):
        resp_500 = MagicMock(status_code=500, text="server err")
        resp_ok = MagicMock(status_code=200)
        mock_post.side_effect = [resp_500, resp_ok]

        result = notifier._post_with_retry({"chat_id": "x", "text": "y"})
        assert result is True
        assert mock_post.call_count == 2

    @patch("src.notifier.requests.post")
    def test_400_does_not_retry(self, mock_post):
        mock_post.return_value = MagicMock(status_code=400, text="bad request")
        result = notifier._post_with_retry({"chat_id": "x", "text": "y"})
        assert result is False
        # 4xx (non-429) should fail immediately
        assert mock_post.call_count == 1


# ── notify_* helpers escape title content ────────────────────────────────────

class TestNotifyHelpers:
    @patch("src.notifier._send")
    def test_notify_trade_escapes_title(self, mock_send):
        notifier.notify_trade("YES", "BTC <test> & other", 25.0, 0.55, 0.08, paper=True)
        text = mock_send.call_args[0][0]
        # HTML special chars should be escaped in the title portion
        assert "&lt;test&gt;" in text
        assert "&amp;" in text

    @patch("src.notifier._send")
    def test_notify_error_uses_silent_false(self, mock_send):
        notifier.notify_error("disk full")
        # notify_error should always alert (silent=False)
        assert mock_send.call_args.kwargs.get("silent") is False

    @patch("src.notifier._send")
    def test_notify_risk_halt_uses_silent_false(self, mock_send):
        notifier.notify_risk_halt("hit daily limit")
        assert mock_send.call_args.kwargs.get("silent") is False

    @patch("src.notifier._send")
    def test_notify_outcome_formats_win(self, mock_send):
        notifier.notify_outcome(42, "WIN", 12.34)
        text = mock_send.call_args[0][0]
        assert "WIN" in text
        assert "+$12.34" in text or "$12.34" in text

    @patch("src.notifier._send")
    def test_notify_daily_summary_loss(self, mock_send):
        notifier.notify_daily_summary(-20.0, 1, 3)
        text = mock_send.call_args[0][0]
        assert "DOWN" in text
        assert "1/3" in text
