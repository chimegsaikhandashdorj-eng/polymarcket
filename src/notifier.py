"""
Optional Telegram notifications for trade events.

Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable.
If either is missing, all notify_* calls are silent no-ops.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Message @userinfobot to get your chat_id
  3. Add to .env:
       TELEGRAM_BOT_TOKEN=123456:ABCDEF...
       TELEGRAM_CHAT_ID=987654321
"""

import logging
import os

import requests

log = logging.getLogger(__name__)

_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")
_API   = "https://api.telegram.org/bot"


def _send(text: str) -> None:
    if not _TOKEN or not _CHAT:
        return
    try:
        requests.post(
            f"{_API}{_TOKEN}/sendMessage",
            json={"chat_id": _CHAT, "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as exc:
        log.debug("Telegram send failed: %s", exc)


def notify_trade(
    side: str,
    title: str,
    size_usdc: float,
    entry_price: float,
    ev: float,
    paper: bool,
) -> None:
    mode = "📄 PAPER" if paper else "💰 LIVE"
    _send(
        f"{mode} Trade Placed\n"
        f"<b>{side}</b> {title[:70]}\n"
        f"Size: <b>${size_usdc:.2f}</b>  Price: {entry_price:.3f}  EV: {ev:+.3f}"
    )


def notify_outcome(trade_id: int, outcome: str, pnl_usdc: float) -> None:
    emoji = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else "⚪"
    _send(f"{emoji} Trade #{trade_id} → <b>{outcome}</b>  PnL: <b>${pnl_usdc:+.2f}</b>")


def notify_error(message: str) -> None:
    _send(f"⚠️ <b>Bot Error</b>\n{message[:300]}")


def notify_risk_halt(reason: str) -> None:
    _send(f"🛑 <b>Trading Halted</b>\n{reason[:300]}\n\nBot will resume when conditions improve.")


def notify_early_exit(trade_id: int, title: str, pnl_pct: float, reason: str) -> None:
    emoji = "💰" if pnl_pct > 0 else "🚨"
    _send(
        f"{emoji} <b>Early Exit</b>\n"
        f"Trade #{trade_id}: {title[:60]}\n"
        f"{reason}\n"
        f"Unrealized: <b>{pnl_pct:+.0%}</b>"
    )


def notify_edge_confirmed(side: str, title: str, ev: float, scans: int) -> None:
    _send(
        f"✅ <b>Edge Confirmed</b> ({scans} scans)\n"
        f"<b>{side}</b> {title[:60]}\n"
        f"EV: {ev:+.3f}"
    )


def notify_daily_summary(total_pnl: float, wins: int, losses: int) -> None:
    emoji = "📈" if total_pnl >= 0 else "📉"
    win_rate = wins / (wins + losses) * 100 if (wins + losses) else 0
    _send(
        f"{emoji} <b>Daily Summary</b>\n"
        f"PnL: <b>${total_pnl:+.2f}</b>  "
        f"W/L: {wins}/{losses}  ({win_rate:.0f}%)"
    )
