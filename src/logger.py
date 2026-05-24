"""
Trade logger and SQLite persistence layer.
All modules import from here — initialize early with init_db().
"""

import sqlite3
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ── stdlib logger setup ───────────────────────────────────────────────────────

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trades.db"

# Reuse the canonical UTC-safe ISO parser from the package root so cached
# timestamps and CLOB-returned timestamps are always normalized identically.
from . import parse_utc_isoformat  # noqa: E402


@contextmanager
def _connect():
    """Context manager that opens, yields, commits, and closes the DB connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables and indexes if they don't exist. Safe to call multiple times."""
    with _connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            market_id       TEXT NOT NULL,
            market_title    TEXT NOT NULL,
            side            TEXT NOT NULL,          -- YES or NO
            size_usdc       REAL NOT NULL,
            entry_price     REAL NOT NULL,           -- 0-1 implied prob
            exit_price      REAL,
            our_prob        REAL NOT NULL,           -- ensemble forecast prob
            confidence      REAL NOT NULL,           -- 0-1 confidence score
            ev              REAL NOT NULL,           -- expected value
            outcome         TEXT,                    -- WIN / LOSS / VOID / OPEN
            pnl_usdc        REAL,
            paper           INTEGER NOT NULL DEFAULT 1,  -- 1=paper, 0=live
            city            TEXT,
            metric          TEXT,                    -- RAIN / TEMP_ABOVE / etc.
            expiry          TEXT
        );

        CREATE TABLE IF NOT EXISTS weather_cache (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            lat             REAL NOT NULL,
            lon             REAL NOT NULL,
            target_dt       TEXT NOT NULL,
            fetched_at      TEXT NOT NULL,
            precip_prob     REAL,
            temp_c          REAL,
            humidity        REAL,
            wind_kph        REAL,
            confidence      REAL,
            sources         TEXT,                    -- comma-separated source names
            regime          TEXT DEFAULT 'NORMAL'   -- NORMAL / UNCERTAIN / EXTREME
        );

        CREATE TABLE IF NOT EXISTS backtest_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at          TEXT NOT NULL,
            market_id       TEXT NOT NULL,
            market_title    TEXT NOT NULL,
            side            TEXT NOT NULL,
            size_usdc       REAL NOT NULL,
            entry_price     REAL NOT NULL,
            our_prob        REAL NOT NULL,
            confidence      REAL NOT NULL,
            ev              REAL NOT NULL,
            outcome         TEXT,
            pnl_usdc        REAL
        );

        CREATE TABLE IF NOT EXISTS fill_tracking (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            trade_id        INTEGER,
            mode            TEXT NOT NULL,
            filled          INTEGER NOT NULL DEFAULT 1,
            price_saved     REAL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_trades_paper_outcome_ts
            ON trades(paper, outcome, timestamp);

        CREATE INDEX IF NOT EXISTS idx_trades_open
            ON trades(paper, outcome);

        CREATE INDEX IF NOT EXISTS idx_fill_tracking_mode
            ON fill_tracking(mode, timestamp);

        CREATE INDEX IF NOT EXISTS idx_weather_cache_lookup
            ON weather_cache(lat, lon, target_dt, fetched_at DESC);
        """)
        # Migrate: add regime column to existing weather_cache tables
        try:
            conn.execute("ALTER TABLE weather_cache ADD COLUMN regime TEXT DEFAULT 'NORMAL'")
        except Exception:
            pass  # column already exists
    log.info("Database initialized at %s", DB_PATH)


# ── Trade CRUD ─────────────────────────────────────────────────────────────────

def log_trade(
    market_id: str,
    market_title: str,
    side: str,
    size_usdc: float,
    entry_price: float,
    our_prob: float,
    confidence: float,
    ev: float,
    paper: bool = True,
    city: str = "",
    metric: str = "",
    expiry: str = "",
) -> int:
    """Insert a new trade record and return its row id."""
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO trades
              (timestamp, market_id, market_title, side, size_usdc,
               entry_price, our_prob, confidence, ev, outcome, paper,
               city, metric, expiry)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (now, market_id, market_title, side, size_usdc,
             entry_price, our_prob, confidence, ev, "OPEN",
             int(paper), city, metric, expiry),
        )
        # sqlite3 returns Optional[int]; after an INSERT it's always set,
        # but we guard explicitly so the return type stays a concrete int.
        if cur.lastrowid is None:
            raise RuntimeError("sqlite INSERT returned no row id")
        trade_id: int = cur.lastrowid
    log.info(
        "Trade logged id=%d  %s %s  size=%.2f  entry=%.3f  EV=%.3f  paper=%s",
        trade_id, side, market_title[:50], size_usdc, entry_price, ev, paper,
    )
    from .notifier import notify_trade
    notify_trade(side, market_title, size_usdc, entry_price, ev, paper)
    return trade_id


def log_trade_prediction(
    trade_id: int,
    raw_prob: float,
    calibrated: float,
    side: str,
    city: str = "",
    metric: str = "",
) -> None:
    """Record model prediction for self-learning feedback (call after log_trade)."""
    try:
        from .model_tracker import get_tracker
        get_tracker().record_trade(trade_id, raw_prob, calibrated, side, city, metric)
    except Exception as exc:
        log.debug("log_trade_prediction failed (non-fatal): %s", exc)


def update_outcome(trade_id: int, exit_price: float, outcome: str) -> None:
    """Mark a trade as WIN/LOSS/VOID with its resolved price and PnL."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT size_usdc, entry_price FROM trades WHERE id=?",
            (trade_id,),
        ).fetchone()
        if row is None:
            log.warning("update_outcome: trade id=%d not found", trade_id)
            return

        size_usdc, entry_price = row["size_usdc"], row["entry_price"]

        if outcome == "WIN":
            # 2% Polymarket fee on net profit (applied to both paper + live to simulate real returns)
            pnl = size_usdc * (1.0 / entry_price - 1.0) * 0.98
        elif outcome == "LOSS":
            pnl = -size_usdc
        else:
            pnl = 0.0

        conn.execute(
            "UPDATE trades SET exit_price=?, outcome=?, pnl_usdc=? WHERE id=?",
            (exit_price, outcome, pnl, trade_id),
        )
    log.info("Trade id=%d resolved → %s  pnl=%.2f", trade_id, outcome, pnl)
    from .notifier import notify_outcome
    notify_outcome(trade_id, outcome, pnl)


# ── Fill tracking ─────────────────────────────────────────────────────────────

def log_fill(trade_id: Optional[int], mode: str, filled: bool, price_saved: float = 0.0) -> None:
    """Record a fill attempt for learning passive vs aggressive fill rates."""
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO fill_tracking (timestamp, trade_id, mode, filled, price_saved) "
            "VALUES (?,?,?,?,?)",
            (now, trade_id, mode, int(filled), price_saved),
        )


# ── Weather cache ─────────────────────────────────────────────────────────────

def cache_weather(
    lat: float, lon: float, target_dt: str,
    precip_prob: float, temp_c: float, humidity: float,
    wind_kph: float, confidence: float, sources: List[str],
    regime: str = "NORMAL",
) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO weather_cache
              (lat, lon, target_dt, fetched_at, precip_prob, temp_c,
               humidity, wind_kph, confidence, sources, regime)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (lat, lon, target_dt, now, precip_prob, temp_c,
             humidity, wind_kph, confidence, ",".join(sources), regime),
        )


def get_cached_weather(
    lat: float, lon: float, target_dt: str, ttl_seconds: int = 3600
) -> Optional[dict]:
    """Return cached row if it exists and is fresh, else None."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM weather_cache
            WHERE lat=? AND lon=? AND target_dt=?
            ORDER BY fetched_at DESC LIMIT 1
            """,
            (lat, lon, target_dt),
        ).fetchone()
        if row is None:
            return None
        age = (
            datetime.now(timezone.utc)
            - parse_utc_isoformat(row["fetched_at"])
        ).total_seconds()
        if age > ttl_seconds:
            return None
        result = dict(row)
        # Ensure regime key is always present (NULL from old rows → NORMAL)
        if not result.get("regime"):
            result["regime"] = "NORMAL"
        return result


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_pnl_summary(paper: Optional[bool] = None) -> dict:
    """Return total PnL, win rate, and trade count."""
    with _connect() as conn:
        if paper is None:
            rows = conn.execute(
                "SELECT outcome, pnl_usdc FROM trades WHERE outcome != 'OPEN'"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT outcome, pnl_usdc FROM trades WHERE outcome != 'OPEN' AND paper=?",
                (int(paper),),
            ).fetchall()

    wins   = sum(1 for r in rows if r["outcome"] == "WIN")
    losses = sum(1 for r in rows if r["outcome"] == "LOSS")
    voids  = sum(1 for r in rows if r["outcome"] == "VOID")
    total  = wins + losses  # VOID trades are refunds, not wins or losses
    pnl    = sum(r["pnl_usdc"] or 0 for r in rows)
    return {
        "total_trades": len(rows),
        "wins":         wins,
        "losses":       losses,
        "voids":        voids,
        "win_rate":     wins / total if total else 0.0,
        "total_pnl_usdc": round(pnl, 2),
    }


def get_daily_loss(paper: bool = True) -> float:
    """Return today's net loss (positive = net loss, negative = net profit)."""
    today = datetime.now(timezone.utc).date().isoformat()  # UTC date matches stored timestamps
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT pnl_usdc FROM trades
            WHERE paper=? AND outcome IN ('WIN', 'LOSS')
              AND timestamp LIKE ?
            """,
            (int(paper), f"{today}%"),
        ).fetchall()
    net = sum(r["pnl_usdc"] or 0 for r in rows)
    return max(0.0, -net)  # positive number = net loss today


def get_open_positions(paper: bool = True) -> List[dict]:
    """Return all currently open (unresolved) trades."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE outcome='OPEN' AND paper=?",
            (int(paper),),
        ).fetchall()
    return [dict(r) for r in rows]


def get_realized_pnl(paper: bool = True) -> float:
    """Return total realized PnL from all resolved (WIN/LOSS) trades."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl_usdc), 0) as total FROM trades "
            "WHERE paper=? AND outcome IN ('WIN', 'LOSS')",
            (int(paper),),
        ).fetchone()
    return float(row["total"])


def get_weekly_loss(paper: bool = True) -> float:
    """Return this calendar week's net loss (positive = loss). Week starts Monday 00:00 UTC."""
    from datetime import timedelta
    today = datetime.now(timezone.utc)
    week_start = (today - timedelta(days=today.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    ).isoformat()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT pnl_usdc FROM trades WHERE paper=? AND outcome IN ('WIN','LOSS') "
            "AND timestamp >= ?",
            (int(paper), week_start),
        ).fetchall()
    net = sum(r["pnl_usdc"] or 0 for r in rows)
    return max(0.0, -net)


def get_monthly_loss(paper: bool = True) -> float:
    """Return this calendar month's net loss (positive = loss). Month starts on the 1st."""
    today = datetime.now(timezone.utc)
    month_start = today.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    ).isoformat()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT pnl_usdc FROM trades WHERE paper=? AND outcome IN ('WIN','LOSS') "
            "AND timestamp >= ?",
            (int(paper), month_start),
        ).fetchall()
    net = sum(r["pnl_usdc"] or 0 for r in rows)
    return max(0.0, -net)


def get_pnl_volatility(paper: bool = True, last_n: int = 30) -> float:
    """Standard deviation of recent trade PnL — used as proxy for portfolio volatility."""
    import statistics as _stats
    with _connect() as conn:
        rows = conn.execute(
            "SELECT pnl_usdc FROM trades WHERE paper=? AND outcome IN ('WIN','LOSS') "
            "ORDER BY timestamp DESC LIMIT ?",
            (int(paper), last_n),
        ).fetchall()
    pnls = [r["pnl_usdc"] for r in rows if r["pnl_usdc"] is not None]
    return _stats.stdev(pnls) if len(pnls) >= 2 else 0.0


def setup_file_logging(log_dir: str = "logs") -> None:
    """Add file handlers to root logger: DEBUG→debug.log, ERROR→error.log."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()

    dh = logging.FileHandler(log_path / "debug.log", encoding="utf-8")
    dh.setLevel(logging.DEBUG)
    dh.setFormatter(fmt)
    root.addHandler(dh)
    root.setLevel(logging.DEBUG)

    eh = logging.FileHandler(log_path / "error.log", encoding="utf-8")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    root.addHandler(eh)

    log.info("File logging initialised: %s/debug.log + %s/error.log", log_dir, log_dir)


def purge_stale_weather_cache(max_age_hours: int = 24) -> int:
    """Delete weather_cache rows older than max_age_hours. Returns row count deleted."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=max_age_hours)
    with _connect() as conn:
        result = conn.execute(
            "DELETE FROM weather_cache WHERE fetched_at < ?",
            (cutoff.isoformat(),),
        )
        deleted = result.rowcount
    if deleted:
        log.info("Purged %d stale weather cache rows (older than %dh)", deleted, max_age_hours)
    return deleted
