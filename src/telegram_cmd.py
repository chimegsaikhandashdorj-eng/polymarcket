"""
Telegram command handler — interactive bot with inline keyboards.

Supported commands:
    /start, /help                Interactive menu with inline keyboard buttons
    /status                      Bot mode, uptime, last scan
    /pnl                         Realized PnL summary
    /positions   (/pos)          Open positions with close buttons
    /balance     (/bal)          Current bankroll
    /scan                        Trigger an immediate market scan
    /weather                     Latest ensemble forecasts
    /risk                        Risk parameters (Kelly, caps, etc.)
    /limits                      Daily/weekly/monthly loss limits + visual bars
    /var                         Portfolio Value-at-Risk
    /history   N                 Last N resolved trades (default 10)
    /crypto                      Crypto trading status & params
    /regime                      Crypto regime + Fear & Greed
    /market ASSET                Live crypto market data (CCXT + TA-Lib)
    /alert ASSET PRICE           Set price alert
    /alerts                      List active alerts
    /close TRADE_ID              Close an open position
    /macro                       Macro economic signals (OpenBB/VIX/DXY)
    /performance (/perf)         Detailed performance metrics
    /chart                       ASCII PnL equity curve
    /export                      Export trade history as CSV file
    /config                      Read-only view of effective config
    /pending                     Edges awaiting confirmation
    /learn                       Self-learning report
    /version                     Bot version / git commit
    /pause                       Pause new trades (scans continue)
    /resume                      Resume trading
    /schedule                    Show daily report schedule
    /morning                     Preview the morning report
    /evening                     Preview the evening report

Daily auto-notifications:
    - 09:00 UTC — Morning report (bankroll, positions, yesterday's PnL)
    - 21:00 UTC — Evening summary (today's PnL, performance metrics)
    - Monday 09:00 UTC — Weekly summary (7-day PnL, best/worst trades)
    - Every 15 min — Price alert checks
    All hours/intervals are configurable in config.yaml -> telegram.scheduler

Interactive features:
    - Inline keyboard buttons on /help for quick command access
    - Callback query handling for button presses
    - Close buttons on open positions
    - Navigation buttons on history pages
"""

import csv
import io
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import requests

from .logger import (
    DB_PATH,
    get_daily_loss,
    get_monthly_loss,
    get_open_positions,
    get_pnl_summary,
    get_realized_pnl,
    get_weekly_loss,
)
from .notifier import (
    _send, _send_with_keyboard, answer_callback, edit_message,
    send_document, escape, is_configured,
)

log = logging.getLogger(__name__)

# Authorized chat IDs (comma-separated env var). Empty list = bot disabled.
_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_IDS: List[str] = [
    c.strip() for c in os.getenv("TELEGRAM_CHAT_ID", "").split(",") if c.strip()
]
_API = f"https://api.telegram.org/bot{_TOKEN}"


def _bot_identity() -> Optional[str]:
    """Fetch bot username via getMe — used in startup message and logs."""
    if not _TOKEN:
        return None
    try:
        r = requests.get(f"{_API}/getMe", timeout=5)
        if r.status_code == 200 and r.json().get("ok"):
            return r.json().get("result", {}).get("username")
    except requests.RequestException:
        pass
    return None


class TelegramCommander:
    """
    Long-polling Telegram bot. Runs ``_poll_loop`` in a daemon thread and
    dispatches handlers via a thread pool so /scan and similar long-running
    commands don't block the loop.
    """

    def __init__(self, config: dict, scan_callback: Optional[Callable] = None):
        self.config = config
        self._scan_callback = scan_callback
        self._paper = config["trading"].get("paper_mode", True)
        self._paused = False
        self._started_at = datetime.now(timezone.utc)
        self._last_scan: Optional[datetime] = None
        self._offset = 0
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="tg-cmd")
        self._scan_inflight = threading.Lock()  # prevent overlapping /scan calls

        # Price alerts: {asset_lower: [(price, direction, chat_id), ...]}
        self._alerts: Dict[str, List[dict]] = {}
        self._alert_lock = threading.Lock()

        # Scheduler config (UTC hours). Defaults: 09:00 morning, 21:00 evening.
        tg_cfg = config.get("telegram", {}) or {}
        sched = tg_cfg.get("scheduler", {}) or {}
        self._morning_hour: int = int(sched.get("morning_hour", 9))
        self._evening_hour: int = int(sched.get("evening_hour", 21))
        self._alert_check_minutes: int = int(sched.get("alert_check_minutes", 15))
        self._weekly_summary_dow: int = int(sched.get("weekly_summary_dow", 0))  # 0=Monday
        # Track which reports we've already sent today so we don't double-fire
        self._last_morning_date: Optional[str] = None
        self._last_evening_date: Optional[str] = None
        self._last_weekly_date: Optional[str] = None
        self._last_alert_check: float = 0.0

        # Command -> handler mapping. Aliases share handler refs.
        self._commands: Dict[str, Callable[[List[str]], None]] = {
            "/start":       self._cmd_help,
            "/help":        self._cmd_help,
            "/status":      self._cmd_status,
            "/pnl":         self._cmd_pnl,
            "/positions":   self._cmd_positions,
            "/pos":         self._cmd_positions,
            "/balance":     self._cmd_balance,
            "/bal":         self._cmd_balance,
            "/scan":        self._cmd_scan,
            "/weather":     self._cmd_weather,
            "/risk":        self._cmd_risk,
            "/limits":      self._cmd_limits,
            "/var":         self._cmd_var,
            "/history":     self._cmd_history,
            "/crypto":      self._cmd_crypto,
            "/regime":      self._cmd_regime,
            "/market":      self._cmd_market,
            "/alert":       self._cmd_alert,
            "/alerts":      self._cmd_alerts,
            "/close":       self._cmd_close,
            "/macro":       self._cmd_macro,
            "/performance": self._cmd_performance,
            "/perf":        self._cmd_performance,
            "/chart":       self._cmd_chart,
            "/export":      self._cmd_export,
            "/config":      self._cmd_config,
            "/pending":     self._cmd_pending,
            "/learn":       self._cmd_learn,
            "/version":     self._cmd_version,
            "/pause":       self._cmd_pause,
            "/resume":      self._cmd_resume,
            "/schedule":    self._cmd_schedule,
            "/morning":     self._cmd_morning,
            "/evening":     self._cmd_evening,
        }

        # Callback query -> handler mapping
        self._callbacks: Dict[str, Callable] = {
            "menu":    self._cb_menu,
            "refresh": self._cb_refresh,
            "close":   self._cb_close_position,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_paused(self) -> bool:
        return self._paused

    def set_last_scan(self, dt: datetime) -> None:
        self._last_scan = dt

    def start(self) -> None:
        """Begin polling Telegram in a background daemon thread."""
        if not is_configured():
            log.warning("Telegram commander disabled — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
            return

        # Identity is best-effort: getMe can occasionally time out on slow networks.
        # Don't refuse to start over a transient network blip.
        identity = _bot_identity() or "bot"

        self._running = True
        t = threading.Thread(target=self._poll_loop, daemon=True, name="tg-cmd")
        t.start()

        # Start daily scheduler thread
        sched = threading.Thread(target=self._scheduler_loop, daemon=True, name="tg-sched")
        sched.start()

        log.info("Telegram commander started as @%s — polling for commands", identity)
        _send(
            f"🤖 <b>Polymarket Bot online</b> @{escape(identity)}\n"
            f"Mode: {'📝 PAPER' if self._paper else '🔴 LIVE'}\n"
            f"⏰ Daily reports: morning {self._morning_hour:02d}:00 / evening {self._evening_hour:02d}:00 UTC\n"
            f"Send /help for commands.",
            silent=False,
        )

    def stop(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)
        _send("🛑 Bot stopped.", silent=False)

    # ── Long-poll loop ────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Outer loop — catches all exceptions so a single bad update can't kill the thread."""
        backoff = 5
        while self._running:
            try:
                self._poll_once()
                backoff = 5  # reset on success
            except Exception as exc:                       # pragma: no cover
                log.warning("Telegram poll error: %s — retry in %ds", exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)  # exponential backoff capped at 60s

    def _poll_once(self) -> None:
        """One iteration of long polling. Uses Telegram's built-in long poll."""
        try:
            resp = requests.get(
                f"{_API}/getUpdates",
                params={"offset": self._offset, "timeout": 25},
                timeout=30,
            )
        except requests.RequestException as exc:
            log.debug("Telegram getUpdates network error: %s", exc)
            time.sleep(2)
            return

        if resp.status_code != 200:
            log.warning("Telegram getUpdates status %d: %s", resp.status_code, resp.text[:200])
            time.sleep(2)
            return

        try:
            data = resp.json()
        except ValueError:
            log.warning("Telegram getUpdates returned non-JSON")
            return
        if not data.get("ok"):
            log.warning("Telegram getUpdates not ok: %s", data.get("description"))
            return

        for update in data.get("result", []):
            self._offset = update["update_id"] + 1
            try:
                self._dispatch_update(update)
            except Exception:
                log.exception("Telegram dispatch failed for update %s", update.get("update_id"))

    # ── Scheduler ────────────────────────────────────────────────────────────

    def _scheduler_loop(self) -> None:
        """Background scheduler — sends morning/evening/weekly reports + price alerts."""
        # Wait briefly so the startup message fires first
        time.sleep(5)
        while self._running:
            try:
                self._scheduler_tick()
            except Exception:
                log.exception("Scheduler tick failed")
            # Check every 60 seconds; reports gated by date-tracking inside tick
            time.sleep(60)

    def _scheduler_tick(self) -> None:
        """One iteration of the scheduler — decide which auto-reports to send."""
        now = datetime.now(timezone.utc)
        today = now.date().isoformat()

        # Morning report (default 09:00 UTC)
        if (now.hour == self._morning_hour and
                self._last_morning_date != today):
            self._last_morning_date = today
            self._send_morning_report()

        # Evening report (default 21:00 UTC)
        if (now.hour == self._evening_hour and
                self._last_evening_date != today):
            self._last_evening_date = today
            self._send_evening_report()

        # Weekly summary on configured weekday (default Monday)
        if (now.weekday() == self._weekly_summary_dow and
                now.hour == self._morning_hour and
                self._last_weekly_date != today):
            self._last_weekly_date = today
            self._send_weekly_summary()

        # Price alerts: check every N minutes
        if time.time() - self._last_alert_check >= self._alert_check_minutes * 60:
            self._last_alert_check = time.time()
            self._check_price_alerts()

    def _send_morning_report(self) -> None:
        """🌅 Morning report — bankroll, positions, today's setup."""
        try:
            initial = float(os.getenv("BANKROLL_USDC", "500"))
            realized = get_realized_pnl(paper=self._paper)
            current = max(1.0, initial + realized)
            positions = get_open_positions(paper=self._paper)
            max_pos = self.config["trading"].get("max_open_positions", 5)

            daily_used = get_daily_loss(paper=self._paper)
            daily_cap = self.config.get("loss_limits", {}).get("daily_usdc", 75)
            paused = "⏸️ Paused" if self._paused else "✅ Active"

            lines = [
                "🌅 <b>Morning Report</b>",
                f"<i>{datetime.now(timezone.utc).strftime('%A, %Y-%m-%d')}</i>\n",
                f"💵 Bankroll: <b>${current:.2f}</b>  ({'+' if realized >= 0 else ''}{realized:.2f})",
                f"📋 Positions: {len(positions)}/{max_pos}",
                f"📅 Yesterday's PnL: ${daily_used:+.2f} / ${daily_cap:.0f} cap",
                f"🎯 State: {paused}  |  Mode: {'📝 PAPER' if self._paper else '🔴 LIVE'}",
            ]

            if positions:
                lines.append("\n<b>Open positions:</b>")
                for p in positions[:5]:
                    title = escape(p.get("market_title", "")[:40])
                    side = p.get("side", "?")
                    size = p.get("size_usdc", 0)
                    lines.append(f"  {'🟩' if side == 'YES' else '🟥'} {side} ${size:.0f} — {title}")

            _send("\n".join(lines), silent=False)
        except Exception as exc:
            log.exception("Morning report failed")
            _send(f"⚠️ Morning report failed: {escape(str(exc)[:200])}")

    def _send_evening_report(self) -> None:
        """🌙 Evening summary — today's PnL, performance metrics."""
        try:
            from .logger import get_pnl_summary
            s = get_pnl_summary(paper=self._paper)
            daily = get_daily_loss(paper=self._paper)
            weekly = get_weekly_loss(paper=self._paper)
            monthly = get_monthly_loss(paper=self._paper)
            positions = get_open_positions(paper=self._paper)

            wr = s["win_rate"] * 100
            arrow = "📈" if s["total_pnl_usdc"] >= 0 else "📉"

            lines = [
                "🌙 <b>Evening Summary</b>",
                f"<i>{datetime.now(timezone.utc).strftime('%A, %Y-%m-%d')}</i>\n",
                f"{arrow} Today: <b>${daily:+.2f}</b>",
                f"📆 Week-to-date: ${weekly:+.2f}",
                f"🗓 Month-to-date: ${monthly:+.2f}\n",
                f"🎯 Lifetime win rate: <b>{wr:.0f}%</b> ({s['wins']}W/{s['losses']}L)",
                f"💰 Total PnL: <b>${s['total_pnl_usdc']:+.2f}</b>",
                f"📋 Open positions: {len(positions)}",
            ]
            _send("\n".join(lines), silent=False)
        except Exception as exc:
            log.exception("Evening report failed")
            _send(f"⚠️ Evening report failed: {escape(str(exc)[:200])}")

    def _send_weekly_summary(self) -> None:
        """📊 Weekly summary — performance overview, top/worst trades."""
        try:
            from .logger import get_pnl_summary
            s = get_pnl_summary(paper=self._paper)
            weekly = get_weekly_loss(paper=self._paper)

            initial = float(os.getenv("BANKROLL_USDC", "500"))
            realized = get_realized_pnl(paper=self._paper)
            roi = ((realized) / initial * 100) if initial > 0 else 0

            try:
                conn = sqlite3.connect(str(DB_PATH))
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT pnl_usdc, market_title FROM trades "
                    "WHERE paper=? AND outcome IN ('WIN', 'LOSS') "
                    "AND timestamp > datetime('now', '-7 days') "
                    "ORDER BY pnl_usdc DESC",
                    (int(self._paper),),
                ).fetchall()
                conn.close()
            except sqlite3.Error:
                rows = []

            lines = [
                "📊 <b>Weekly Summary</b>",
                f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d')}</i>\n",
                f"📈 7-day PnL: <b>${weekly:+.2f}</b>",
                f"💰 Lifetime PnL: <b>${s['total_pnl_usdc']:+.2f}</b>",
                f"📊 ROI: <b>{roi:+.1f}%</b>",
                f"🎯 Win rate: <b>{s['win_rate']*100:.0f}%</b>",
                f"📋 Trades this week: {len(rows)}",
            ]
            if rows:
                top = rows[0]
                worst = rows[-1]
                lines.append(f"\n🏆 Best trade: ${top['pnl_usdc']:+.2f} — {escape(top['market_title'][:35])}")
                if worst != top:
                    lines.append(f"💀 Worst trade: ${worst['pnl_usdc']:+.2f} — {escape(worst['market_title'][:35])}")
            _send("\n".join(lines), silent=False)
        except Exception as exc:
            log.exception("Weekly summary failed")
            _send(f"⚠️ Weekly summary failed: {escape(str(exc)[:200])}")

    def _check_price_alerts(self) -> None:
        """Fire alerts whose price threshold has been crossed."""
        with self._alert_lock:
            if not self._alerts:
                return
            assets_to_check = list(self._alerts.keys())

        try:
            from .crypto_fetcher import CryptoEnsemble
            ensemble = CryptoEnsemble(self.config)
        except ImportError:
            return

        for asset in assets_to_check:
            try:
                data = ensemble.fetch(asset)
            except Exception as exc:
                log.debug("Alert price fetch failed for %s: %s", asset, exc)
                continue
            if not data:
                continue
            price = data.get("price", 0)
            if not price:
                continue

            with self._alert_lock:
                remaining = []
                for a in self._alerts.get(asset, []):
                    triggered = False
                    if a["direction"] == "above" and price >= a["price"]:
                        triggered = True
                    elif a["direction"] == "below" and price <= a["price"]:
                        triggered = True
                    if triggered:
                        dir_emoji = "⬆️" if a["direction"] == "above" else "⬇️"
                        _send(
                            f"🔔 <b>Price Alert!</b>\n"
                            f"{dir_emoji} <b>{escape(asset.upper())}</b> {a['direction']} "
                            f"${a['price']:,.2f}\n"
                            f"💲 Current: <b>${price:,.2f}</b>",
                            silent=False,
                        )
                    else:
                        remaining.append(a)
                if remaining:
                    self._alerts[asset] = remaining
                else:
                    self._alerts.pop(asset, None)

    def _dispatch_update(self, update: dict) -> None:
        """Parse one Telegram update and dispatch to the matching handler."""
        # ── Callback query (inline keyboard button press) ──
        callback = update.get("callback_query")
        if callback:
            self._dispatch_callback(callback)
            return

        msg = update.get("message") or update.get("edited_message") or {}
        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = (msg.get("text") or "").strip()

        # Authorization: ignore strangers silently (don't even reply)
        if chat_id not in _CHAT_IDS:
            if chat_id:
                log.warning("Ignoring Telegram message from unauthorized chat_id=%s", chat_id)
            return
        if not text.startswith("/"):
            return

        # Tokenize: /cmd@bot arg1 arg2 -> cmd="/cmd", args=["arg1", "arg2"]
        parts = text.split()
        cmd = parts[0].split("@")[0].lower()
        # ``text`` is str, so .split() yields a list[str]; annotate explicitly
        # because pyrefly otherwise infers list[LiteralString].
        args: List[str] = list(parts[1:])

        handler = self._commands.get(cmd)
        if not handler:
            _send(f"❓ Unknown command: {escape(cmd)}\nSend /help for available commands.")
            return

        # Dispatch in worker thread so slow commands don't block poll loop
        self._executor.submit(self._safe_invoke, handler, args, cmd)

    def _dispatch_callback(self, callback: dict) -> None:
        """Handle an inline keyboard button press."""
        chat_id = str(callback.get("message", {}).get("chat", {}).get("id", ""))
        if chat_id not in _CHAT_IDS:
            return

        cb_id = callback.get("id", "")
        data = callback.get("data", "")
        msg_id = callback.get("message", {}).get("message_id")

        # Parse: "action:param" or just "action"
        parts = data.split(":", 1)
        action = parts[0]
        param = parts[1] if len(parts) > 1 else ""

        handler = self._callbacks.get(action)
        if handler:
            answer_callback(cb_id)
            self._executor.submit(
                self._safe_invoke_cb, handler, param, chat_id, msg_id, action
            )
        else:
            answer_callback(cb_id, "❓ Unknown action")

    def _safe_invoke_cb(self, handler: Callable, param: str, chat_id: str, msg_id: int, action: str) -> None:
        try:
            handler(param, chat_id, msg_id)
        except Exception as exc:
            log.exception("Callback %s failed", action)
            _send(f"⚠️ Callback error: {escape(str(exc)[:200])}")

    def _safe_invoke(self, handler: Callable, args: List[str], cmd: str) -> None:
        """Wrap handler in try/except so an exception in one command can't crash the bot."""
        try:
            handler(args)
        except Exception as exc:
            log.exception("Telegram command %s failed", cmd)
            _send(f"Command error in {escape(cmd)}: {escape(str(exc)[:200])}")

    # ── Commands ──────────────────────────────────────────────────────────────

    def _cmd_help(self, args: List[str]) -> None:
        text = (
            "🤖 <b>Polymarket Trading Bot</b>\n\n"
            "Доорх товчлуурууд дээр дарж команд ажиллуулна уу, "
            "эсвэл шууд командыг бичнэ үү.\n\n"
            "📊 /status /pnl /positions /balance /history\n"
            "🌤 /weather /var /chart /performance\n"
            "📈 /crypto /regime /market /macro /alerts\n"
            "⏰ /schedule /morning /evening\n"
            "⚙️ /scan /pause /resume /config /export"
        )
        keyboard = [
            [
                {"text": "📊 Status", "callback_data": "menu:status"},
                {"text": "💰 PnL", "callback_data": "menu:pnl"},
                {"text": "📋 Positions", "callback_data": "menu:positions"},
            ],
            [
                {"text": "💵 Balance", "callback_data": "menu:balance"},
                {"text": "📈 Chart", "callback_data": "menu:chart"},
                {"text": "🏆 Performance", "callback_data": "menu:performance"},
            ],
            [
                {"text": "🔍 Scan Now", "callback_data": "menu:scan"},
                {"text": "📉 Limits", "callback_data": "menu:limits"},
                {"text": "⚠️ VaR", "callback_data": "menu:var"},
            ],
            [
                {"text": "🪙 Crypto", "callback_data": "menu:crypto"},
                {"text": "🌍 Macro", "callback_data": "menu:macro"},
                {"text": "🔔 Alerts", "callback_data": "menu:alerts"},
            ],
        ]
        _send_with_keyboard(text, keyboard)

    def _cmd_status(self, args: List[str]) -> None:
        mode = "📝 PAPER" if self._paper else "🔴 LIVE"
        paused = "⏸️ PAUSED" if self._paused else "✅ ACTIVE"
        uptime = datetime.now(timezone.utc) - self._started_at
        hours, rem = divmod(int(uptime.total_seconds()), 3600)
        mins = rem // 60
        last = self._last_scan.strftime("%H:%M UTC") if self._last_scan else "—"

        open_pos = get_open_positions(paper=self._paper)
        max_pos = self.config["trading"].get("max_open_positions", 5)

        keyboard = [
            [
                {"text": "🔄 Refresh", "callback_data": "refresh:status"},
                {"text": "🔍 Scan Now", "callback_data": "menu:scan"},
            ],
        ]
        _send_with_keyboard(
            f"📊 <b>Bot Status</b>\n\n"
            f"Mode: {mode}\n"
            f"State: {paused}\n"
            f"⏱ Uptime: {hours}h {mins}m\n"
            f"🔍 Last scan: {last}\n"
            f"📋 Positions: {len(open_pos)}/{max_pos}",
            keyboard,
        )

    def _cmd_pnl(self, args: List[str]) -> None:
        s = get_pnl_summary(paper=self._paper)
        wr = s["win_rate"] * 100
        pnl = s["total_pnl_usdc"]
        arrow = "📈" if pnl >= 0 else "📉"
        wr_bar = self._progress_bar(min(1.0, wr / 100), 10)
        _send(
            f"{arrow} <b>PnL Summary</b>\n\n"
            f"💰 Total PnL: <b>${pnl:+.2f}</b>\n"
            f"📊 Trades: {s['total_trades']}  "
            f"✅ {s['wins']} / ❌ {s['losses']} / ⬜ {s['voids']}\n"
            f"🎯 Win rate: {wr:.0f}% {wr_bar}"
        )

    def _cmd_positions(self, args: List[str]) -> None:
        positions = get_open_positions(paper=self._paper)
        if not positions:
            _send("📋 No open positions.")
            return

        current_prices = self._fetch_current_prices(positions)

        lines = [f"📋 <b>Open Positions ({len(positions)})</b>\n"]
        buttons = []
        for p in positions:
            title = escape(p["market_title"][:45])
            mid = current_prices.get(p["market_id"])
            pnl_emoji = ""
            pnl_str = ""
            if mid is not None:
                if p["side"] == "YES":
                    pnl_pct = (mid - p["entry_price"]) / max(0.01, p["entry_price"])
                else:
                    pnl_pct = (p["entry_price"] - mid) / max(0.01, 1.0 - p["entry_price"])
                pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                arrow = "+" if pnl_pct >= 0 else ""
                pnl_str = f"  {pnl_emoji} {arrow}{pnl_pct:.1%}"
            lines.append(
                f"\n{'🟩' if p['side'] == 'YES' else '🟥'} <b>{escape(p['side'])}</b> {title}\n"
                f"  💵 ${p['size_usdc']:.2f} @ {p['entry_price']:.3f}  "
                f"EV: {p['ev']:+.3f}{pnl_str}"
            )
            trade_id = p.get("id") or p.get("trade_id")
            if trade_id:
                buttons.append(
                    {"text": f"❌ Close #{trade_id}", "callback_data": f"close:{trade_id}"}
                )

        keyboard = [buttons[i:i + 2] for i in range(0, len(buttons), 2)] if buttons else []
        keyboard.append([{"text": "🔄 Refresh", "callback_data": "refresh:positions"}])

        _send_with_keyboard("\n".join(lines), keyboard)

    def _fetch_current_prices(self, positions: List[dict]) -> Dict[str, float]:
        """Best-effort lookup of current YES prices for open positions."""
        prices: Dict[str, float] = {}
        try:
            from .market_scanner import CLOB_API, _safe_get
        except ImportError:
            return prices
        seen = set()
        for p in positions:
            mid = p.get("market_id", "")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            data = _safe_get(f"{CLOB_API}/markets/{mid}")
            # CLOB returns an object; narrow before subscripting to keep the
            # type checker happy and reject malformed (e.g. list) responses.
            if not isinstance(data, dict):
                continue
            tokens = data.get("tokens") or []
            for tok in tokens:
                if isinstance(tok, dict) and tok.get("outcome") == "Yes":
                    try:
                        prices[mid] = float(tok.get("price", 0))
                    except (ValueError, TypeError):
                        pass
        return prices

    def _cmd_balance(self, args: List[str]) -> None:
        initial = float(os.getenv("BANKROLL_USDC", "500"))
        realized = get_realized_pnl(paper=self._paper)
        current = max(1.0, initial + realized)
        pnl_emoji = "📈" if realized >= 0 else "📉"
        roi = ((current - initial) / initial * 100) if initial > 0 else 0
        _send(
            f"💵 <b>Bankroll</b>\n\n"
            f"🏦 Initial: ${initial:.2f}\n"
            f"{pnl_emoji} Realized PnL: ${realized:+.2f}\n"
            f"💰 Current: <b>${current:.2f}</b>\n"
            f"📊 ROI: {roi:+.1f}%"
        )

    def _cmd_scan(self, args: List[str]) -> None:
        if not self._scan_callback:
            _send("⚠️ Scan not available (bot not fully started).")
            return
        if not self._scan_inflight.acquire(blocking=False):
            _send("⏳ A scan is already in progress — try again in a moment.")
            return
        try:
            _send("🔍 Scanning markets...")
            results = self._scan_callback() or []
            emoji = "✅" if results else "😴"
            _send(f"{emoji} Scan complete: <b>{len(results)}</b> opportunity(ies) found.")
        except Exception as exc:
            _send(f"❌ Scan failed: {escape(str(exc)[:200])}")
        finally:
            self._scan_inflight.release()

    def _cmd_weather(self, args: List[str]) -> None:
        if not DB_PATH.exists():
            _send("No weather data yet.")
            return
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT lat, lon, target_dt, precip_prob, temp_c, wind_kph, "
                "confidence, sources, regime "
                "FROM weather_cache ORDER BY fetched_at DESC LIMIT 8"
            ).fetchall()
            conn.close()
        except sqlite3.Error:
            _send("Could not read weather data.")
            return

        if not rows:
            _send("No weather data cached yet.")
            return

        cities = self.config.get("cities", [])
        city_lookup = {(c["lat"], c["lon"]): c["name"] for c in cities}

        lines = ["<b>Latest Weather</b>\n"]
        for r in rows:
            name = city_lookup.get((r["lat"], r["lon"]), f"{r['lat']:.2f},{r['lon']:.2f}")
            precip = (r["precip_prob"] or 0) * 100
            temp = r["temp_c"] or 0
            conf = r["confidence"] or 0
            regime = r["regime"] or "?"
            lines.append(
                f"<b>{escape(name)}</b>: {precip:.0f}% rain  "
                f"{temp:.1f}C  conf={conf:.2f}  [{escape(regime)}]"
            )
        _send("\n".join(lines))

    def _cmd_risk(self, args: List[str]) -> None:
        t = self.config.get("trading", {})
        r = self.config.get("risk", {})
        _send(
            "<b>Risk Parameters</b>\n"
            f"Kelly fraction: {t.get('kelly_fraction', 0):.0%}\n"
            f"Max position: ${t.get('max_position_usdc', 0):.0f}\n"
            f"Max open positions: {t.get('max_open_positions', 0)}\n"
            f"Min confidence: {t.get('min_confidence', 0):.0%}\n"
            f"EV threshold: {t.get('ev_threshold', 0):.1%}\n"
            f"Edge confirmations: {t.get('edge_confirmations', 1)}\n"
            f"Max city exposure: {r.get('max_city_exposure_pct', 0):.0%}\n"
            f"Max portfolio exposure: {r.get('max_portfolio_exposure_pct', 0):.0%}\n"
            f"Drawdown halt: {r.get('drawdown_halt_pct', 0):.0%}"
        )

    def _cmd_limits(self, args: List[str]) -> None:
        ll = self.config.get("loss_limits", {})
        daily_used = get_daily_loss(paper=self._paper)
        weekly_used = get_weekly_loss(paper=self._paper)
        monthly_used = get_monthly_loss(paper=self._paper)

        def _line(emoji: str, name: str, used: float, cap: float) -> str:
            cap = max(1.0, cap)
            pct = max(0.0, used) / cap * 100
            bar = self._emoji_bar(min(1.0, used / cap))
            warn = " ⚠️" if pct > 80 else ""
            return (
                f"{emoji} <b>{name}</b>: ${used:+.2f} / ${cap:.0f}  "
                f"({pct:.0f}%){warn}\n{bar}"
            )

        _send(
            "📉 <b>Loss Limits</b>\n\n"
            + _line("📅", "Daily", daily_used, ll.get("daily_usdc", 75))
            + "\n\n"
            + _line("📆", "Weekly", weekly_used, ll.get("weekly_usdc", 200))
            + "\n\n"
            + _line("🗓", "Monthly", monthly_used, ll.get("monthly_usdc", 400))
        )

    @staticmethod
    def _progress_bar(frac: float, width: int = 10) -> str:
        """Render an ASCII progress bar for proportional values [0, 1]."""
        filled = int(round(max(0.0, min(1.0, frac)) * width))
        return "[" + "#" * filled + "." * (width - filled) + "]"

    def _cmd_var(self, args: List[str]) -> None:
        try:
            from .risk_manager import RiskManager
        except ImportError:
            _send("Risk manager not available.")
            return
        try:
            rm = RiskManager(self.config)
            positions = get_open_positions(paper=self._paper)
            initial = float(os.getenv("BANKROLL_USDC", "500"))
            realized = get_realized_pnl(paper=self._paper)
            bankroll = max(1.0, initial + realized)
            var = rm.estimate_portfolio_var(positions, bankroll)
            _send(
                f"<b>Portfolio Value-at-Risk</b>\n"
                f"95% VaR: ${var.get('var_95_usdc', 0):.2f}\n"
                f"99% VaR: ${var.get('var_99_usdc', 0):.2f}\n"
                f"Expected loss: ${var.get('expected_loss_usdc', 0):.2f}\n"
                f"Max drawdown estimate: ${var.get('max_loss_usdc', 0):.2f}"
            )
        except Exception as exc:
            _send(f"VaR unavailable: {escape(str(exc)[:200])}")

    def _cmd_history(self, args: List[str]) -> None:
        try:
            n = int(args[0]) if args else 10
        except ValueError:
            n = 10
        n = max(1, min(50, n))  # clamp to [1, 50]

        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT timestamp, market_title, side, size_usdc, "
                "       outcome, pnl_usdc "
                "FROM trades "
                "WHERE paper=? AND outcome IN ('WIN', 'LOSS', 'VOID') "
                "ORDER BY id DESC LIMIT ?",
                (int(self._paper), n),
            ).fetchall()
            conn.close()
        except sqlite3.Error as exc:
            _send(f"DB error: {escape(str(exc))}")
            return

        if not rows:
            _send("No resolved trades yet.")
            return

        lines = [f"<b>Last {len(rows)} Resolved Trades</b>\n"]
        for r in rows:
            ts = (r["timestamp"] or "")[:10]
            outcome = r["outcome"] or "?"
            pnl = r["pnl_usdc"] or 0.0
            sign = "+" if pnl >= 0 else ""
            lines.append(
                f"<code>{ts}</code> {escape(outcome):5s} "
                f"{sign}${pnl:.2f}  {escape(r['market_title'][:38])}"
            )
        _send("\n".join(lines))

    def _cmd_crypto(self, args: List[str]) -> None:
        c = self.config.get("crypto", {})
        enabled = c.get("enabled", False)
        assets = c.get("assets", [])
        paper_until = c.get("paper_only_until", "—")
        _send(
            "<b>Crypto Trading</b>\n"
            f"Enabled: <b>{'YES' if enabled else 'NO'}</b>\n"
            f"Forced paper until: {escape(paper_until)}\n"
            f"Assets: {escape(', '.join(assets) or 'none')}\n"
            f"Max position: ${c.get('max_position_usdc', 0):.0f}\n"
            f"Daily loss limit: ${c.get('daily_loss_limit_usdc', 0):.0f}\n"
            f"Min EV: {c.get('min_ev_threshold', 0):.0%}\n"
            f"Min confidence: {c.get('min_confidence', 0):.0%}\n"
            f"Max daily volatility: {c.get('max_volatility_daily', 0):.0%}\n"
            f"Max spread: {c.get('max_spread', 0):.0%}\n"
            f"Edge confirmations: {c.get('edge_confirmations', 1)}"
        )

    def _cmd_regime(self, args: List[str]) -> None:
        try:
            from .crypto_signals import get_fear_greed_index
        except ImportError:
            _send("Crypto signals module not available.")
            return
        fg = get_fear_greed_index()
        if fg:
            fg_text = (
                f"Fear &amp; Greed: <b>{fg['value']}/100</b> "
                f"({escape(fg['classification'])})\n"
                f"Signal: {escape(fg['signal'])} (x{fg['signal_mult']:.2f})"
            )
        else:
            fg_text = "Fear &amp; Greed: unavailable"
        _send(f"<b>Crypto Regime</b>\n{fg_text}")

    def _cmd_config(self, args: List[str]) -> None:
        """Show a compact, sanitized view of the effective config."""
        t = self.config.get("trading", {})
        m = self.config.get("markets", {})
        _send(
            "<b>Effective Config (read-only)</b>\n"
            f"Paper mode: {t.get('paper_mode', True)}\n"
            f"Poll interval: {t.get('poll_interval_seconds', 900)}s\n"
            f"EV threshold: {t.get('ev_threshold', 0):.1%}\n"
            f"Kelly: {t.get('kelly_fraction', 0):.0%}\n"
            f"Max pos: ${t.get('max_position_usdc', 0):.0f}\n"
            f"Max open: {t.get('max_open_positions', 0)}\n"
            f"Min liquidity: ${m.get('min_liquidity_usdc', 0):.0f}\n"
            f"Hours-to-expiry: {m.get('min_hours_to_expiry', 0)}h to "
            f"{m.get('max_hours_to_expiry', 0)}h\n"
            f"Cities monitored: {len(self.config.get('cities', []))}"
        )

    def _cmd_pending(self, args: List[str]) -> None:
        try:
            from .edge_confirm import get_edge_confirmation
            gate = get_edge_confirmation(self.config)
            pending = gate.get_pending_edges()
            if not pending:
                _send("No edges awaiting confirmation.")
                return
            lines = [f"<b>Pending Edges ({len(pending)})</b>\n"]
            for p in sorted(pending, key=lambda x: -x.ev)[:10]:
                lines.append(
                    f"  {escape(p.side)} {escape(p.market_id[:16])}.. "
                    f"EV={p.ev:.3f}  scans={p.confirmations}"
                )
            _send("\n".join(lines))
        except Exception as exc:
            _send(f"Error: {escape(str(exc)[:200])}")

    def _cmd_learn(self, args: List[str]) -> None:
        try:
            from .learner import get_learner
            learner = get_learner(self.config)
            report = learner.generate_report()
            _send(f"<b>Learning Report</b>\n{escape(report)}")
        except Exception as exc:
            _send(f"Learning report failed: {escape(str(exc)[:200])}")

    def _cmd_version(self, args: List[str]) -> None:
        """Show bot version, git commit (best effort), and Python version."""
        import platform
        import subprocess
        commit = "unknown"
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).decode().strip()
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass
        _send(
            "<b>Bot Version</b>\n"
            f"Git: <code>{escape(commit)}</code>\n"
            f"Python: {escape(platform.python_version())}\n"
            f"Started: {self._started_at.strftime('%Y-%m-%d %H:%M UTC')}"
        )

    def _cmd_pause(self, args: List[str]) -> None:
        self._paused = True
        _send(
            "<b>Trading paused.</b>\n"
            "Scans continue but no new trades will be placed.\n"
            "Send /resume to restart."
        )

    def _cmd_resume(self, args: List[str]) -> None:
        self._paused = False
        _send("<b>Trading resumed.</b>")

    def _cmd_schedule(self, args: List[str]) -> None:
        now = datetime.now(timezone.utc)
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
        weekly_day = weekday_names[self._weekly_summary_dow]

        next_morning = "today" if now.hour < self._morning_hour else "tomorrow"
        next_evening = "today" if now.hour < self._evening_hour else "tomorrow"

        with self._alert_lock:
            alert_count = sum(len(a) for a in self._alerts.values())

        _send(
            "⏰ <b>Scheduled Reports</b>\n\n"
            f"🌅 Morning report: <b>{self._morning_hour:02d}:00 UTC</b> ({next_morning})\n"
            f"🌙 Evening summary: <b>{self._evening_hour:02d}:00 UTC</b> ({next_evening})\n"
            f"📊 Weekly summary: <b>{weekly_day} {self._morning_hour:02d}:00 UTC</b>\n"
            f"🔔 Alert check: every <b>{self._alert_check_minutes} min</b>\n\n"
            f"Current time: {now.strftime('%H:%M UTC')}\n"
            f"Active alerts: {alert_count}\n\n"
            f"Use /morning or /evening to preview a report."
        )

    def _cmd_morning(self, args: List[str]) -> None:
        """Manually trigger morning report (for testing)."""
        self._send_morning_report()

    def _cmd_evening(self, args: List[str]) -> None:
        """Manually trigger evening summary (for testing)."""
        self._send_evening_report()

    # ── Emoji bar helper ─────────────────────────────────────────────────────

    @staticmethod
    def _emoji_bar(frac: float, width: int = 10) -> str:
        frac = max(0.0, min(1.0, frac))
        filled = int(round(frac * width))
        if frac > 0.8:
            fill_char = "🟥"
        elif frac > 0.5:
            fill_char = "🟨"
        else:
            fill_char = "🟩"
        return fill_char * filled + "⬜" * (width - filled)

    # ── Callback handlers ────────────────────────────────────────────────────

    def _cb_menu(self, param: str, chat_id: str, msg_id: int) -> None:
        cmd = f"/{param}" if param else "/help"
        handler = self._commands.get(cmd)
        if handler:
            handler([])
        else:
            _send(f"❓ Unknown menu item: {escape(param)}")

    def _cb_refresh(self, param: str, chat_id: str, msg_id: int) -> None:
        cmd = f"/{param}" if param else "/status"
        handler = self._commands.get(cmd)
        if handler:
            handler([])

    def _cb_close_position(self, param: str, chat_id: str, msg_id: int) -> None:
        if not param:
            _send("⚠️ No trade ID specified.")
            return
        self._cmd_close([param])

    # ── New command handlers ─────────────────────────────────────────────────

    def _cmd_market(self, args: List[str]) -> None:
        if not args:
            _send("Usage: /market &lt;asset&gt;\nExample: /market bitcoin")
            return
        asset = args[0].lower()
        try:
            from .crypto_fetcher import CryptoEnsemble
            ensemble = CryptoEnsemble(self.config)
            data = ensemble.fetch(asset)
        except ImportError:
            _send("⚠️ Crypto fetcher not available.")
            return
        except Exception as exc:
            _send(f"❌ Failed to fetch {escape(asset)}: {escape(str(exc)[:200])}")
            return

        if not data:
            _send(f"❌ No data for <b>{escape(asset)}</b>.")
            return

        price = data.get("price", 0)
        vol = data.get("vol_24h", data.get("volatility_daily", 0))
        momentum = data.get("momentum", {})

        lines = [f"📈 <b>{escape(asset.upper())} Market Data</b>\n"]
        lines.append(f"💲 Price: <b>${price:,.2f}</b>")
        if vol:
            lines.append(f"📊 24h Volume/Volatility: {vol:.4f}")
        if momentum:
            rsi = momentum.get("rsi")
            trend = momentum.get("trend")
            strength = momentum.get("strength", 0)
            if rsi is not None:
                rsi_label = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
                lines.append(f"📉 RSI: {rsi:.1f} ({rsi_label})")
            if trend:
                lines.append(f"🔀 Trend: {escape(trend)} (strength: {strength:.2f})")
            sma_20 = momentum.get("sma_20")
            if sma_20:
                lines.append(f"📐 SMA-20: ${sma_20:,.2f}")

        _send("\n".join(lines))

    def _cmd_alert(self, args: List[str]) -> None:
        if len(args) < 2:
            _send("Usage: /alert &lt;asset&gt; &lt;price&gt;\nExample: /alert bitcoin 70000")
            return
        asset = args[0].lower()
        try:
            target_price = float(args[1])
        except ValueError:
            _send("⚠️ Invalid price. Use a number.")
            return

        try:
            from .crypto_fetcher import CryptoEnsemble
            ensemble = CryptoEnsemble(self.config)
            data = ensemble.fetch(asset)
            current = data.get("price", 0) if data else 0
        except Exception:
            current = 0

        direction = "above" if target_price > current else "below"

        with self._alert_lock:
            if asset not in self._alerts:
                self._alerts[asset] = []
            self._alerts[asset].append({
                "price": target_price,
                "direction": direction,
                "created": datetime.now(timezone.utc).isoformat(),
            })

        dir_emoji = "⬆️" if direction == "above" else "⬇️"
        _send(
            f"🔔 Alert set: <b>{escape(asset.upper())}</b> "
            f"{dir_emoji} ${target_price:,.2f}\n"
            f"Current price: ${current:,.2f}"
        )

    def _cmd_alerts(self, args: List[str]) -> None:
        with self._alert_lock:
            if not self._alerts:
                _send("🔔 No active price alerts.\nUse /alert &lt;asset&gt; &lt;price&gt; to set one.")
                return
            lines = ["🔔 <b>Active Alerts</b>\n"]
            total = 0
            for asset, alerts in self._alerts.items():
                for a in alerts:
                    dir_emoji = "⬆️" if a["direction"] == "above" else "⬇️"
                    lines.append(
                        f"  {dir_emoji} <b>{escape(asset.upper())}</b> "
                        f"@ ${a['price']:,.2f} ({a['direction']})"
                    )
                    total += 1
            lines[0] = f"🔔 <b>Active Alerts ({total})</b>\n"
            _send("\n".join(lines))

    def _cmd_close(self, args: List[str]) -> None:
        if not args:
            _send("Usage: /close &lt;trade_id&gt;\nExample: /close 42")
            return
        try:
            trade_id = int(args[0])
        except ValueError:
            _send("⚠️ Invalid trade ID. Use a number.")
            return

        positions = get_open_positions(paper=self._paper)
        match = None
        for p in positions:
            if (p.get("id") or p.get("trade_id")) == trade_id:
                match = p
                break
        if not match:
            _send(f"❌ No open position with trade ID #{trade_id}.")
            return

        if self._paper:
            try:
                conn = sqlite3.connect(str(DB_PATH))
                conn.execute(
                    "UPDATE trades SET outcome='VOID', pnl_usdc=0.0 WHERE id=? AND paper=1",
                    (trade_id,),
                )
                conn.commit()
                conn.close()
                title = escape(match.get("market_title", "")[:50])
                _send(f"✅ Closed paper trade #{trade_id}: {title}")
            except sqlite3.Error as exc:
                _send(f"❌ DB error: {escape(str(exc)[:200])}")
        else:
            _send(
                f"⚠️ Live trade #{trade_id} — manual close via Polymarket UI required.\n"
                f"Paper trades can be force-closed here."
            )

    def _cmd_macro(self, args: List[str]) -> None:
        try:
            from .crypto_signals import get_macro_signals
            macro = get_macro_signals(self.config)
        except ImportError:
            _send("⚠️ Macro signals module not available.")
            return
        except Exception as exc:
            _send(f"❌ Macro fetch failed: {escape(str(exc)[:200])}")
            return

        if not macro:
            _send("📊 No macro data available at the moment.")
            return

        vix = macro.get("vix")
        dxy = macro.get("dxy")
        bias = macro.get("bias", 0)
        bias_label = "bullish" if bias > 0 else "bearish" if bias < 0 else "neutral"
        bias_emoji = "🟢" if bias > 0 else "🔴" if bias < 0 else "⚪"

        lines = ["🌍 <b>Macro Signals</b>\n"]
        if vix is not None:
            vix_label = "high fear" if vix > 25 else "low fear" if vix < 15 else "moderate"
            lines.append(f"📊 VIX: <b>{vix:.1f}</b> ({vix_label})")
        if dxy is not None:
            lines.append(f"💵 DXY: <b>{dxy:.2f}</b>")
        lines.append(f"{bias_emoji} Macro bias: <b>{bias_label}</b> ({bias:+.2f})")

        _send("\n".join(lines))

    def _cmd_performance(self, args: List[str]) -> None:
        s = get_pnl_summary(paper=self._paper)
        total_trades = s["total_trades"]
        if total_trades == 0:
            _send("🏆 No trades yet — start trading to see performance metrics.")
            return

        wr = s["win_rate"] * 100
        avg_win = s["total_pnl_usdc"] / max(1, s["wins"]) if s["wins"] else 0
        avg_loss = s["total_pnl_usdc"] / max(1, s["losses"]) if s["losses"] else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        initial = float(os.getenv("BANKROLL_USDC", "500"))
        realized = get_realized_pnl(paper=self._paper)
        roi = ((realized) / initial * 100) if initial > 0 else 0

        daily = get_daily_loss(paper=self._paper)
        weekly = get_weekly_loss(paper=self._paper)

        lines = [
            "🏆 <b>Performance Report</b>\n",
            f"📊 Total trades: <b>{total_trades}</b>",
            f"✅ Wins: {s['wins']}  ❌ Losses: {s['losses']}  ⬜ Voids: {s['voids']}",
            f"🎯 Win rate: <b>{wr:.1f}%</b>  {self._progress_bar(wr / 100)}",
            f"💰 Total PnL: <b>${s['total_pnl_usdc']:+.2f}</b>",
            f"📈 ROI: <b>{roi:+.1f}%</b>",
            f"⚖️ Profit factor: {profit_factor:.2f}",
            f"📅 Today: ${daily:+.2f}  📆 This week: ${weekly:+.2f}",
        ]
        _send("\n".join(lines))

    def _cmd_chart(self, args: List[str]) -> None:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT pnl_usdc FROM trades "
                "WHERE paper=? AND outcome IN ('WIN', 'LOSS') "
                "ORDER BY id ASC",
                (int(self._paper),),
            ).fetchall()
            conn.close()
        except sqlite3.Error as exc:
            _send(f"❌ DB error: {escape(str(exc)[:200])}")
            return

        if not rows:
            _send("📈 No resolved trades yet — chart unavailable.")
            return

        cumulative = []
        running = 0.0
        for r in rows:
            running += r["pnl_usdc"] or 0.0
            cumulative.append(running)

        chart_width = min(40, len(cumulative))
        if len(cumulative) > chart_width:
            step = len(cumulative) / chart_width
            sampled = [cumulative[int(i * step)] for i in range(chart_width)]
        else:
            sampled = cumulative

        lo = min(sampled)
        hi = max(sampled)
        spread = hi - lo if hi != lo else 1.0
        chart_height = 8

        lines = ["📈 <b>Equity Curve</b>\n<pre>"]
        for row_idx in range(chart_height, -1, -1):
            threshold = lo + (row_idx / chart_height) * spread
            row_chars = []
            for val in sampled:
                if val >= threshold:
                    row_chars.append("█")
                else:
                    row_chars.append(" ")
            level = lo + (row_idx / chart_height) * spread
            lines.append(f"${level:>7.1f} |{''.join(row_chars)}")
        lines.append(f"         {'─' * len(sampled)}")
        lines.append(f"  trades: {len(cumulative)}  final: ${cumulative[-1]:+.2f}")
        lines.append("</pre>")

        _send("\n".join(lines))

    def _cmd_export(self, args: List[str]) -> None:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, timestamp, market_title, side, size_usdc, "
                "       entry_price, outcome, pnl_usdc, ev, paper "
                "FROM trades ORDER BY id DESC",
            ).fetchall()
            conn.close()
        except sqlite3.Error as exc:
            _send(f"❌ DB error: {escape(str(exc)[:200])}")
            return

        if not rows:
            _send("📄 No trades to export.")
            return

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "timestamp", "market_title", "side",
                         "size_usdc", "entry_price", "outcome", "pnl_usdc",
                         "ev", "paper"])
        for r in rows:
            writer.writerow([r[k] for k in ["id", "timestamp", "market_title",
                                             "side", "size_usdc", "entry_price",
                                             "outcome", "pnl_usdc", "ev", "paper"]])

        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix="trades_export_",
            delete=False, encoding="utf-8",
        )
        tmp.write(buf.getvalue())
        tmp.close()

        if send_document(tmp.name, caption="📄 Trade history export"):
            _send("✅ Trade history exported.")
        else:
            _send("⚠️ Could not send file. Check bot configuration.")

        try:
            os.unlink(tmp.name)
        except OSError:
            pass
