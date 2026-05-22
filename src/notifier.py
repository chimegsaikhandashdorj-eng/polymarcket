"""
Telegram notification helpers used across the bot.

All notify_* functions are silent no-ops if TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
are not configured, so the rest of the bot can call them unconditionally.

Improvements over the original implementation:
  * HTML-escapes untrusted strings (market titles can contain <, >, &).
  * Splits messages above Telegram's 4096-character limit into chunks.
  * Token-bucket rate limiter — Telegram blocks chats that exceed ~30 msg/sec.
  * Exponential-backoff retry on transient failures (network / 5xx).
  * Quiet-hours support (22:00 - 07:00 UTC by default) sends silent messages.
  * Multi-recipient: TELEGRAM_CHAT_ID may be a comma-separated list.

Setup:
    1. Message @BotFather on Telegram -> /newbot -> copy the token
    2. Message @userinfobot to get your chat_id
    3. Add to .env:
         TELEGRAM_BOT_TOKEN=123456:ABCDEF...
         TELEGRAM_CHAT_ID=987654321              (or "111,222,333")
         TELEGRAM_QUIET_START=22                 (optional, UTC hour)
         TELEGRAM_QUIET_END=7                    (optional, UTC hour)
"""

import html
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

import requests

log = logging.getLogger(__name__)

# ── Configuration (read on import, refreshable via _reload_env) ──────────────

_TOKEN: str = ""
_CHAT_IDS: List[str] = []
_QUIET_START: int = 22
_QUIET_END: int = 7
_API_BASE = "https://api.telegram.org/bot"

# Telegram hard limits
_MAX_MSG_LEN = 4096          # characters
_MIN_INTERVAL = 0.05         # 20 msgs / sec ceiling for safety (Telegram = 30)

# Rate-limiter state
_send_lock = threading.Lock()
_last_send_at: float = 0.0


def _reload_env() -> None:
    """Re-read env vars (useful for tests that monkey-patch os.environ)."""
    global _TOKEN, _CHAT_IDS, _QUIET_START, _QUIET_END
    _TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    raw_chat = os.getenv("TELEGRAM_CHAT_ID", "")
    _CHAT_IDS = [c.strip() for c in raw_chat.split(",") if c.strip()]
    try:
        _QUIET_START = int(os.getenv("TELEGRAM_QUIET_START", "22"))
    except ValueError:
        _QUIET_START = 22
    try:
        _QUIET_END = int(os.getenv("TELEGRAM_QUIET_END", "7"))
    except ValueError:
        _QUIET_END = 7


_reload_env()


def is_configured() -> bool:
    """True iff the bot has both a token and at least one chat id."""
    return bool(_TOKEN and _CHAT_IDS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_quiet_hour() -> bool:
    """True during the configured silent-notification window (UTC)."""
    hour = datetime.now(timezone.utc).hour
    if _QUIET_START == _QUIET_END:
        return False
    if _QUIET_START < _QUIET_END:
        # e.g. 1 - 6: quiet at hour 1..5
        return _QUIET_START <= hour < _QUIET_END
    # wraps midnight, e.g. 22 - 7: quiet at hour >=22 or hour < 7
    return hour >= _QUIET_START or hour < _QUIET_END


def _chunk(text: str, max_len: int = _MAX_MSG_LEN) -> List[str]:
    """
    Split a long message into Telegram-safe chunks, breaking at newlines
    when possible so HTML tags stay balanced within chunks. Callers are
    responsible for keeping HTML tags well-formed inside any one line.
    """
    if len(text) <= max_len:
        return [text]
    chunks: List[str] = []
    remaining = text
    while len(remaining) > max_len:
        # Prefer a newline near the limit; fall back to a hard cut
        cut = remaining.rfind("\n", 0, max_len)
        if cut <= 0:
            cut = max_len
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


def _post_with_retry(payload: dict, max_attempts: int = 3) -> bool:
    """
    POST to Telegram with exponential backoff. Returns True on success.
    A 429 response is honored via the ``retry_after`` field; transient
    5xx and network errors retry up to ``max_attempts`` times.
    """
    url = f"{_API_BASE}{_TOKEN}/sendMessage"
    for attempt in range(max_attempts):
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                # Honor Telegram's flood-control retry_after if present
                try:
                    wait = float(r.json().get("parameters", {}).get("retry_after", 1))
                except Exception:
                    wait = 1.0
                log.warning("Telegram 429 — sleeping %.1fs", wait)
                time.sleep(min(wait, 30))
                continue
            if 500 <= r.status_code < 600 and attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
                continue
            log.debug("Telegram non-OK status %d: %s", r.status_code, r.text[:200])
            return False
        except requests.RequestException as exc:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
                continue
            log.debug("Telegram send failed after %d attempts: %s", max_attempts, exc)
            return False
    return False


def _send_one(chat_id: str, text: str, silent: bool) -> bool:
    """Send a single chunk to a single chat with rate limiting."""
    global _last_send_at
    with _send_lock:
        # Throttle to stay under the per-chat-per-second cap
        delta = time.time() - _last_send_at
        if delta < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - delta)
        _last_send_at = time.time()

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if silent:
        payload["disable_notification"] = True
    return _post_with_retry(payload)


def _send(text: str, *, silent: Optional[bool] = None) -> bool:
    """
    Send ``text`` (HTML-formatted) to every configured chat id.

    Returns True if at least one delivery succeeded. ``silent`` defaults to
    the configured quiet-hours behavior; pass an explicit bool to override.
    """
    if not is_configured():
        return False
    use_silent = _is_quiet_hour() if silent is None else silent
    success_any = False
    for chunk in _chunk(text):
        for chat_id in _CHAT_IDS:
            if _send_one(chat_id, chunk, use_silent):
                success_any = True
    return success_any


def escape(s: str) -> str:
    """HTML-escape a string so it can be safely interpolated into messages."""
    return html.escape(str(s), quote=False)


# ── Public notify_* helpers ──────────────────────────────────────────────────

def notify_trade(
    side: str,
    title: str,
    size_usdc: float,
    entry_price: float,
    ev: float,
    paper: bool,
) -> None:
    mode = "PAPER" if paper else "LIVE"
    _send(
        f"{mode} Trade Placed\n"
        f"<b>{escape(side)}</b> {escape(title[:70])}\n"
        f"Size: <b>${size_usdc:.2f}</b>  Price: {entry_price:.3f}  EV: {ev:+.3f}"
    )


def notify_outcome(trade_id: int, outcome: str, pnl_usdc: float) -> None:
    emoji = "[WIN]" if outcome == "WIN" else "[LOSS]" if outcome == "LOSS" else "[VOID]"
    sign = "+" if pnl_usdc >= 0 else "-"
    _send(
        f"{emoji} Trade #{trade_id} -> <b>{escape(outcome)}</b>  "
        f"PnL: <b>{sign}${abs(pnl_usdc):.2f}</b>"
    )


def notify_error(message: str) -> None:
    """High-priority error notification — never silenced by quiet hours."""
    _send(f"<b>Bot Error</b>\n{escape(message[:600])}", silent=False)


def notify_risk_halt(reason: str) -> None:
    """Critical halt notification — never silenced."""
    _send(
        f"<b>Trading Halted</b>\n{escape(reason[:600])}\n\n"
        f"Bot will resume when conditions improve.",
        silent=False,
    )


def notify_early_exit(trade_id: int, title: str, pnl_pct: float, reason: str) -> None:
    emoji = "[PROFIT]" if pnl_pct > 0 else "[STOP]"
    _send(
        f"{emoji} <b>Early Exit</b>\n"
        f"Trade #{trade_id}: {escape(title[:60])}\n"
        f"{escape(reason)}\n"
        f"Unrealized: <b>{pnl_pct:+.0%}</b>"
    )


def notify_edge_confirmed(side: str, title: str, ev: float, scans: int) -> None:
    _send(
        f"<b>Edge Confirmed</b> ({scans} scans)\n"
        f"<b>{escape(side)}</b> {escape(title[:60])}\n"
        f"EV: {ev:+.3f}"
    )


def notify_daily_summary(total_pnl: float, wins: int, losses: int) -> None:
    win_rate = wins / (wins + losses) * 100 if (wins + losses) else 0
    direction = "UP" if total_pnl >= 0 else "DOWN"
    _send(
        f"<b>Daily Summary [{direction}]</b>\n"
        f"PnL: <b>${total_pnl:+.2f}</b>  "
        f"W/L: {wins}/{losses}  ({win_rate:.0f}%)"
    )
