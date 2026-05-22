"""
Telegram command handler — long-poll bot for controlling the trading bot.

Supported commands (all aliases shown in parentheses):
    /status                      Bot mode, uptime, last scan
    /pnl                         Realized PnL summary
    /positions   (/pos)          Open positions list with unrealized PnL
    /balance     (/bal)          Current bankroll
    /scan                        Trigger an immediate market scan (async)
    /weather                     Latest ensemble forecasts
    /risk                        Risk parameters (Kelly, caps, etc.)
    /limits                      Daily/weekly/monthly loss limits + usage
    /var                         Portfolio Value-at-Risk
    /history   N                 Last N resolved trades (default 10)
    /crypto                      Crypto trading status & params
    /regime                      Crypto regime + Fear & Greed
    /config                      Read-only view of effective config
    /pending                     Edges awaiting confirmation
    /learn                       Self-learning report
    /version                     Bot version / git commit
    /pause                       Pause new trades (scans continue)
    /resume                      Resume trading
    /help        (/start)        List commands

Concurrent commands are dispatched to a small thread pool so slow handlers
(e.g. /scan) cannot starve the poll loop. Unauthorized senders are silently
ignored. The bot only acts on messages from chat IDs listed in
``TELEGRAM_CHAT_ID`` (comma-separated supported).
"""

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
from .notifier import _send, escape, is_configured

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

        # Command -> handler mapping. Aliases share handler refs.
        self._commands: Dict[str, Callable[[List[str]], None]] = {
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
            "/risk":      self._cmd_risk,
            "/limits":    self._cmd_limits,
            "/var":       self._cmd_var,
            "/history":   self._cmd_history,
            "/crypto":    self._cmd_crypto,
            "/regime":    self._cmd_regime,
            "/config":    self._cmd_config,
            "/pending":   self._cmd_pending,
            "/learn":     self._cmd_learn,
            "/version":   self._cmd_version,
            "/pause":     self._cmd_pause,
            "/resume":    self._cmd_resume,
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
            log.info("Telegram commander disabled — no token/chat configured")
            return
        self._running = True
        t = threading.Thread(target=self._poll_loop, daemon=True, name="tg-cmd")
        t.start()

        identity = _bot_identity() or "bot"
        log.info("Telegram commander started as @%s — polling for commands", identity)
        _send(
            f"<b>Polymarket Bot online</b> @{escape(identity)}\n"
            f"Mode: {'PAPER' if self._paper else 'LIVE'}\n"
            f"Send /help for commands.",
            silent=False,
        )

    def stop(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)
        _send("Bot stopped.", silent=False)

    # ── Long-poll loop ────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Outer loop — catches all exceptions so a single bad update can't kill the thread."""
        while self._running:
            try:
                self._poll_once()
            except Exception as exc:                       # pragma: no cover
                log.debug("Telegram poll error: %s", exc)
                time.sleep(5)

    def _poll_once(self) -> None:
        """One iteration of long polling. Uses Telegram's built-in long poll."""
        try:
            resp = requests.get(
                f"{_API}/getUpdates",
                params={"offset": self._offset, "timeout": 25},
                timeout=30,
            )
        except requests.RequestException:
            time.sleep(2)
            return

        try:
            data = resp.json()
        except ValueError:
            return
        if not data.get("ok"):
            return

        for update in data.get("result", []):
            self._offset = update["update_id"] + 1
            self._dispatch_update(update)

    def _dispatch_update(self, update: dict) -> None:
        """Parse one Telegram update and dispatch to the matching handler."""
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
        args = parts[1:]

        handler = self._commands.get(cmd)
        if not handler:
            _send(f"Unknown command: {escape(cmd)}\nSend /help for available commands.")
            return

        # Dispatch in worker thread so slow commands don't block poll loop
        self._executor.submit(self._safe_invoke, handler, args, cmd)

    def _safe_invoke(self, handler: Callable, args: List[str], cmd: str) -> None:
        """Wrap handler in try/except so an exception in one command can't crash the bot."""
        try:
            handler(args)
        except Exception as exc:
            log.exception("Telegram command %s failed", cmd)
            _send(f"Command error in {escape(cmd)}: {escape(str(exc)[:200])}")

    # ── Commands ──────────────────────────────────────────────────────────────

    def _cmd_help(self, args: List[str]) -> None:
        _send(
            "<b>Polymarket Trading Bot</b>\n\n"
            "<b>Monitoring</b>\n"
            "/status — bot status &amp; uptime\n"
            "/pnl — realized PnL summary\n"
            "/positions — open positions (alias /pos)\n"
            "/balance — current bankroll (alias /bal)\n"
            "/history N — last N resolved trades\n"
            "/weather — latest forecasts\n"
            "/var — portfolio Value-at-Risk\n\n"
            "<b>Risk &amp; Limits</b>\n"
            "/risk — risk parameters\n"
            "/limits — daily/weekly/monthly limit usage\n"
            "/pending — edges awaiting confirmation\n\n"
            "<b>Crypto</b>\n"
            "/crypto — crypto trading status\n"
            "/regime — regime &amp; Fear &amp; Greed\n\n"
            "<b>Control</b>\n"
            "/scan — trigger immediate scan\n"
            "/pause — pause new trades\n"
            "/resume — resume trading\n"
            "/learn — self-learning report\n"
            "/config — show effective config\n"
            "/version — bot version"
        )

    def _cmd_status(self, args: List[str]) -> None:
        mode = "PAPER" if self._paper else "LIVE"
        paused = "PAUSED" if self._paused else "ACTIVE"
        uptime = datetime.now(timezone.utc) - self._started_at
        hours, rem = divmod(int(uptime.total_seconds()), 3600)
        mins = rem // 60
        last = self._last_scan.strftime("%H:%M UTC") if self._last_scan else "—"

        open_pos = get_open_positions(paper=self._paper)
        max_pos = self.config["trading"].get("max_open_positions", 5)
        _send(
            f"<b>Status</b>\n"
            f"Mode: <b>{mode}</b>  State: <b>{paused}</b>\n"
            f"Uptime: {hours}h {mins}m\n"
            f"Last scan: {last}\n"
            f"Positions: {len(open_pos)}/{max_pos}"
        )

    def _cmd_pnl(self, args: List[str]) -> None:
        s = get_pnl_summary(paper=self._paper)
        wr = s["win_rate"] * 100
        arrow = "UP" if s["total_pnl_usdc"] >= 0 else "DOWN"
        _send(
            f"<b>PnL Summary [{arrow}]</b>\n"
            f"Total PnL: <b>${s['total_pnl_usdc']:+.2f}</b>\n"
            f"Trades: {s['total_trades']}  "
            f"W/L/V: {s['wins']}/{s['losses']}/{s['voids']}\n"
            f"Win rate: {wr:.0f}%"
        )

    def _cmd_positions(self, args: List[str]) -> None:
        positions = get_open_positions(paper=self._paper)
        if not positions:
            _send("No open positions.")
            return

        # Try to fetch current prices for unrealized PnL — best-effort only
        current_prices = self._fetch_current_prices(positions)

        lines = [f"<b>Open Positions ({len(positions)})</b>\n"]
        for p in positions:
            title = escape(p["market_title"][:50])
            mid = current_prices.get(p["market_id"])
            if mid is not None:
                if p["side"] == "YES":
                    pnl_pct = (mid - p["entry_price"]) / max(0.01, p["entry_price"])
                else:
                    pnl_pct = (p["entry_price"] - mid) / max(0.01, 1.0 - p["entry_price"])
                arrow = "+" if pnl_pct >= 0 else ""
                pnl_str = f"  ({arrow}{pnl_pct:.1%})"
            else:
                pnl_str = ""
            lines.append(
                f"• <b>{escape(p['side'])}</b> {title}\n"
                f"  ${p['size_usdc']:.2f} @ {p['entry_price']:.3f}  "
                f"EV: {p['ev']:+.3f}{pnl_str}"
            )
        _send("\n".join(lines))

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
            if data and "tokens" in data:
                for tok in data["tokens"]:
                    if tok.get("outcome") == "Yes":
                        try:
                            prices[mid] = float(tok.get("price", 0))
                        except (ValueError, TypeError):
                            pass
        return prices

    def _cmd_balance(self, args: List[str]) -> None:
        initial = float(os.getenv("BANKROLL_USDC", "500"))
        realized = get_realized_pnl(paper=self._paper)
        current = max(1.0, initial + realized)
        _send(
            f"<b>Bankroll</b>\n"
            f"Initial: ${initial:.2f}\n"
            f"Realized PnL: ${realized:+.2f}\n"
            f"Current: <b>${current:.2f}</b>"
        )

    def _cmd_scan(self, args: List[str]) -> None:
        if not self._scan_callback:
            _send("Scan not available (bot not fully started).")
            return
        # Prevent overlapping scans
        if not self._scan_inflight.acquire(blocking=False):
            _send("A scan is already in progress — try again in a moment.")
            return
        try:
            _send("Scanning markets...")
            results = self._scan_callback() or []
            _send(f"Scan complete: {len(results)} opportunity(ies) found.")
        except Exception as exc:
            _send(f"Scan failed: {escape(str(exc)[:200])}")
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

        def _line(name: str, used: float, cap: float) -> str:
            cap = max(1.0, cap)
            pct = max(0.0, used) / cap * 100
            bar = self._progress_bar(min(1.0, used / cap)) if used > 0 else "[..........]"
            return (
                f"<b>{name}</b>: ${used:+.2f} / ${cap:.0f}  "
                f"({pct:.0f}%)\n{bar}"
            )

        _send(
            "<b>Loss Limits</b>\n\n"
            + _line("Daily", daily_used, ll.get("daily_usdc", 75))
            + "\n\n"
            + _line("Weekly", weekly_used, ll.get("weekly_usdc", 200))
            + "\n\n"
            + _line("Monthly", monthly_used, ll.get("monthly_usdc", 400))
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
