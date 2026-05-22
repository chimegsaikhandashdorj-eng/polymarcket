"""
Telegram command handler — poll-based bot for controlling the trading bot.

Supported commands:
  /status    — bot mode, uptime, last scan time
  /pnl       — realised PnL summary
  /positions — open positions list
  /scan      — trigger an immediate market scan
  /balance   — current bankroll
  /weather   — latest ensemble forecasts
  /pause     — pause trading (skip new entries)
  /resume    — resume trading
  /help      — list commands

Runs in a daemon thread alongside the main bot loop.
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import requests

from .logger import (
    get_open_positions, get_pnl_summary, get_realized_pnl, DB_PATH,
)

log = logging.getLogger(__name__)

_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")
_API   = f"https://api.telegram.org/bot{_TOKEN}"


def _send(text: str) -> None:
    if not _TOKEN or not _CHAT:
        return
    try:
        requests.post(
            f"{_API}/sendMessage",
            json={"chat_id": _CHAT, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as exc:
        log.debug("Telegram send failed: %s", exc)


class TelegramCommander:
    def __init__(self, config: dict, scan_callback: Optional[Callable] = None):
        self.config = config
        self._scan_callback = scan_callback
        self._paper = config["trading"].get("paper_mode", True)
        self._paused = False
        self._started_at = datetime.now(timezone.utc)
        self._last_scan: Optional[datetime] = None
        self._offset = 0
        self._running = False

        self._commands: Dict[str, Callable] = {
            "/start":     self._cmd_help,
            "/help":      self._cmd_help,
            "/status":    self._cmd_status,
            "/pnl":       self._cmd_pnl,
            "/positions": self._cmd_positions,
            "/pos":       self._cmd_positions,
            "/balance":   self._cmd_balance,
            "/bal":       self._cmd_balance,
            "/scan":      self._cmd_scan,
            "/weather":   self._cmd_weather,
            "/learn":     self._cmd_learn,
            "/pending":   self._cmd_pending,
            "/pause":     self._cmd_pause,
            "/resume":    self._cmd_resume,
        }

    @property
    def is_paused(self) -> bool:
        return self._paused

    def set_last_scan(self, dt: datetime) -> None:
        self._last_scan = dt

    def start(self) -> None:
        if not _TOKEN or not _CHAT:
            log.info("Telegram commander disabled — no token/chat configured")
            return
        self._running = True
        t = threading.Thread(target=self._poll_loop, daemon=True, name="tg-cmd")
        t.start()
        log.info("Telegram commander started — polling for commands")
        _send("🟢 <b>Bot started</b>\nSend /help for commands.")

    def stop(self) -> None:
        self._running = False
        _send("🔴 <b>Bot stopped</b>")

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._poll_updates()
            except Exception as exc:
                log.debug("Telegram poll error: %s", exc)
            time.sleep(2)

    def _poll_updates(self) -> None:
        try:
            resp = requests.get(
                f"{_API}/getUpdates",
                params={"offset": self._offset, "timeout": 10},
                timeout=15,
            )
            data = resp.json()
        except Exception:
            return

        if not data.get("ok"):
            return

        for update in data.get("result", []):
            self._offset = update["update_id"] + 1
            msg = update.get("message", {})
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = (msg.get("text") or "").strip()

            if chat_id != _CHAT:
                continue
            if not text.startswith("/"):
                continue

            cmd = text.split()[0].split("@")[0].lower()
            handler = self._commands.get(cmd)
            if handler:
                try:
                    handler()
                except Exception as exc:
                    _send(f"⚠️ Command error: {exc}")
            else:
                _send(f"Unknown command: {cmd}\nSend /help for available commands.")

    # ── Commands ──────────────────────────────────────────────────────────────

    def _cmd_help(self) -> None:
        _send(
            "🤖 <b>Polymarket Weather Bot</b>\n\n"
            "/status — Bot status & uptime\n"
            "/pnl — PnL summary\n"
            "/positions — Open positions\n"
            "/balance — Current bankroll\n"
            "/scan — Trigger immediate scan\n"
            "/weather — Latest forecasts\n"
            "/learn — Learning report & insights\n"
            "/pending — Edges awaiting confirmation\n"
            "/pause — Pause new trades\n"
            "/resume — Resume trading\n"
        )

    def _cmd_status(self) -> None:
        mode = "📄 PAPER" if self._paper else "💰 LIVE"
        paused = "⏸ PAUSED" if self._paused else "▶️ ACTIVE"
        uptime = datetime.now(timezone.utc) - self._started_at
        hours = int(uptime.total_seconds() // 3600)
        mins = int((uptime.total_seconds() % 3600) // 60)

        last = self._last_scan.strftime("%H:%M UTC") if self._last_scan else "—"

        open_pos = get_open_positions(paper=self._paper)
        max_pos = self.config["trading"].get("max_open_positions", 5)

        _send(
            f"📊 <b>Status</b>\n"
            f"Mode: {mode}  {paused}\n"
            f"Uptime: {hours}h {mins}m\n"
            f"Last scan: {last}\n"
            f"Positions: {len(open_pos)}/{max_pos}\n"
        )

    def _cmd_pnl(self) -> None:
        s = get_pnl_summary(paper=self._paper)
        wr = s["win_rate"] * 100
        emoji = "📈" if s["total_pnl_usdc"] >= 0 else "📉"
        _send(
            f"{emoji} <b>PnL Summary</b>\n"
            f"Total PnL: <b>${s['total_pnl_usdc']:+.2f}</b>\n"
            f"Trades: {s['total_trades']}  "
            f"W/L/V: {s['wins']}/{s['losses']}/{s['voids']}\n"
            f"Win rate: {wr:.0f}%"
        )

    def _cmd_positions(self) -> None:
        positions = get_open_positions(paper=self._paper)
        if not positions:
            _send("📭 No open positions.")
            return

        lines = ["📋 <b>Open Positions</b>\n"]
        for p in positions:
            lines.append(
                f"• <b>{p['side']}</b> {p['market_title'][:50]}\n"
                f"  ${p['size_usdc']:.2f} @ {p['entry_price']:.3f}  "
                f"EV: {p['ev']:+.3f}"
            )
        _send("\n".join(lines))

    def _cmd_balance(self) -> None:
        initial = float(os.getenv("BANKROLL_USDC", "500"))
        realized = get_realized_pnl(paper=self._paper)
        current = max(1.0, initial + realized)
        _send(
            f"💰 <b>Bankroll</b>\n"
            f"Initial: ${initial:.2f}\n"
            f"Realized PnL: ${realized:+.2f}\n"
            f"Current: <b>${current:.2f}</b>"
        )

    def _cmd_scan(self) -> None:
        if self._scan_callback:
            _send("🔍 Scanning markets...")
            try:
                results = self._scan_callback()
                n = len(results) if results else 0
                _send(f"✅ Scan complete: {n} opportunities found.")
            except Exception as exc:
                _send(f"⚠️ Scan failed: {exc}")
        else:
            _send("⚠️ Scan not available (bot not fully started).")

    def _cmd_weather(self) -> None:
        import sqlite3
        if not DB_PATH.exists():
            _send("No weather data yet.")
            return
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT lat, lon, target_dt, precip_prob, temp_c, wind_kph, "
                "confidence, sources, regime "
                "FROM weather_cache ORDER BY fetched_at DESC LIMIT 7"
            ).fetchall()
            conn.close()
        except Exception:
            _send("⚠️ Could not read weather data.")
            return

        if not rows:
            _send("No weather data cached yet.")
            return

        cities = self.config.get("cities", [])
        city_lookup = {(c["lat"], c["lon"]): c["name"] for c in cities}

        lines = ["🌤 <b>Latest Weather</b>\n"]
        for r in rows:
            name = city_lookup.get((r["lat"], r["lon"]), f"{r['lat']},{r['lon']}")
            precip = (r["precip_prob"] or 0) * 100
            temp = r["temp_c"] or 0
            conf = r["confidence"] or 0
            regime = r["regime"] or "?"
            lines.append(
                f"<b>{name}</b>: {precip:.0f}% rain  {temp:.1f}°C  "
                f"conf={conf:.2f}  [{regime}]"
            )
        _send("\n".join(lines))

    def _cmd_pending(self) -> None:
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
                    f"  {p.side} {p.market_id[:16]}..  "
                    f"EV={p.ev:.3f}  scans={p.confirmations}"
                )
            _send("\n".join(lines))
        except Exception as exc:
            _send(f"Error: {exc}")

    def _cmd_learn(self) -> None:
        try:
            from .learner import get_learner
            learner = get_learner(self.config)
            report = learner.generate_report()
            _send(f"🧠 {report}")
        except Exception as exc:
            _send(f"⚠️ Learning report failed: {exc}")

    def _cmd_pause(self) -> None:
        self._paused = True
        _send("⏸ <b>Trading paused.</b>\nScans continue but no new trades.\nSend /resume to restart.")

    def _cmd_resume(self) -> None:
        self._paused = False
        _send("▶️ <b>Trading resumed.</b>")
