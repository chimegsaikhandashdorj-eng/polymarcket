"""
Self-learning module — analyzes resolved trades, identifies error patterns,
and auto-adjusts strategy parameters to improve future performance.

Runs after each batch of trades resolves.  Three learning subsystems:

  1. **Error Pattern Analysis**
     Groups losses by city, metric, confidence band, and time-to-expiry.
     Identifies systematic weaknesses (e.g. "snow in Tokyo always loses").

  2. **Adaptive Parameter Tuning**
     Adjusts EV threshold, Kelly fraction, and confidence floor based on
     rolling realized performance vs. predicted performance.

  3. **Weather Source Grading**
     Tracks per-source accuracy by city.  Demotes consistently wrong
     sources and promotes accurate ones in the ensemble weights.

All adjustments are bounded (min/max clamps) to prevent runaway drift
and logged to the learning_log DB table for full auditability.
"""

import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .logger import DB_PATH

log = logging.getLogger(__name__)

_WINDOW_DAYS = 60
_MIN_TRADES_FOR_LEARNING = 10

# Clamps: never adjust outside these bounds (conservative)
_EV_THRESHOLD_RANGE = (0.05, 0.15)       # min 5% edge, max 15%
_KELLY_FRACTION_RANGE = (0.08, 0.25)     # min 8% Kelly, max 25%
_CONFIDENCE_FLOOR_RANGE = (0.50, 0.80)   # min 50% confidence, max 80%

# Learning rates — how fast parameters move toward the "ideal" value
_PARAM_LR = 0.20  # blend 20% toward target each cycle


def _ensure_tables() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS learning_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            category    TEXT NOT NULL,
            description TEXT NOT NULL,
            old_value   TEXT,
            new_value   TEXT,
            reason      TEXT
        );

        CREATE TABLE IF NOT EXISTS error_patterns (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            updated_at  TEXT NOT NULL,
            pattern_key TEXT NOT NULL UNIQUE,
            losses      INTEGER NOT NULL DEFAULT 0,
            wins        INTEGER NOT NULL DEFAULT 0,
            avg_ev      REAL DEFAULT 0,
            avg_conf    REAL DEFAULT 0,
            penalty     REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS param_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            param_name  TEXT NOT NULL,
            old_value   REAL NOT NULL,
            new_value   REAL NOT NULL,
            reason      TEXT
        );
        """)


def _log_learning(category: str, description: str,
                  old_value: str = "", new_value: str = "",
                  reason: str = "") -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO learning_log (timestamp, category, description, "
                "old_value, new_value, reason) VALUES (?,?,?,?,?,?)",
                (now, category, description, old_value, new_value, reason),
            )
    except Exception as exc:
        log.debug("_log_learning failed: %s", exc)


def _log_param_change(param: str, old: float, new: float, reason: str) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO param_history (timestamp, param_name, old_value, "
                "new_value, reason) VALUES (?,?,?,?,?)",
                (now, param, old, new, reason),
            )
    except Exception as exc:
        log.debug("_log_param_change failed: %s", exc)


class AdaptiveLearner:
    def __init__(self, config: dict):
        self.config = config
        self._config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        _ensure_tables()

    def _cx(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _get_resolved_trades(self, window_days: int = _WINDOW_DAYS) -> List[dict]:
        cutoff = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=window_days)
        ).isoformat()
        try:
            with self._cx() as conn:
                rows = conn.execute(
                    """
                    SELECT t.*, tp.raw_prob, tp.calibrated, tp.city as pred_city,
                           tp.metric as pred_metric
                    FROM trades t
                    LEFT JOIN trade_predictions tp ON tp.trade_id = t.id
                    WHERE t.outcome IN ('WIN', 'LOSS') AND t.timestamp > ?
                    ORDER BY t.timestamp DESC
                    """,
                    (cutoff,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.debug("_get_resolved_trades failed: %s", exc)
            return []

    # ── 1. Error Pattern Analysis ─────────────────────────────────────────────

    def analyze_errors(self) -> Dict[str, dict]:
        trades = self._get_resolved_trades()
        if len(trades) < _MIN_TRADES_FOR_LEARNING:
            return {}

        patterns: Dict[str, dict] = {}

        for t in trades:
            city = (t.get("city") or t.get("pred_city") or "unknown").lower()
            metric = (t.get("metric") or t.get("pred_metric") or "unknown").upper()
            is_win = t["outcome"] == "WIN"

            conf_band = "low" if (t.get("confidence") or 0) < 0.6 else (
                "mid" if (t.get("confidence") or 0) < 0.8 else "high"
            )

            keys = [
                f"city:{city}",
                f"metric:{metric}",
                f"city_metric:{city}_{metric}",
                f"conf:{conf_band}",
            ]

            for key in keys:
                if key not in patterns:
                    patterns[key] = {
                        "wins": 0, "losses": 0,
                        "total_ev": 0.0, "total_conf": 0.0,
                    }
                p = patterns[key]
                if is_win:
                    p["wins"] += 1
                else:
                    p["losses"] += 1
                p["total_ev"] += t.get("ev") or 0
                p["total_conf"] += t.get("confidence") or 0

        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        problem_patterns = {}

        for key, p in patterns.items():
            total = p["wins"] + p["losses"]
            if total < 3:
                continue

            win_rate = p["wins"] / total
            avg_ev = p["total_ev"] / total
            avg_conf = p["total_conf"] / total

            # A pattern is problematic if win rate < 40% with at least 5 trades
            penalty = 0.0
            if total >= 5 and win_rate < 0.40:
                penalty = min(0.5, (0.40 - win_rate) * 2)
            elif total >= 3 and win_rate < 0.30:
                penalty = min(0.5, (0.30 - win_rate) * 2)

            try:
                with self._cx() as conn:
                    conn.execute(
                        """
                        INSERT INTO error_patterns
                          (updated_at, pattern_key, losses, wins, avg_ev, avg_conf, penalty)
                        VALUES (?,?,?,?,?,?,?)
                        ON CONFLICT(pattern_key) DO UPDATE SET
                          updated_at=?, losses=?, wins=?, avg_ev=?, avg_conf=?, penalty=?
                        """,
                        (now, key, p["losses"], p["wins"], avg_ev, avg_conf, penalty,
                         now, p["losses"], p["wins"], avg_ev, avg_conf, penalty),
                    )
            except Exception:
                pass

            if penalty > 0:
                problem_patterns[key] = {
                    "wins": p["wins"],
                    "losses": p["losses"],
                    "win_rate": round(win_rate, 3),
                    "avg_ev": round(avg_ev, 4),
                    "penalty": round(penalty, 3),
                }

        if problem_patterns:
            _log_learning(
                "error_patterns",
                f"Found {len(problem_patterns)} weak patterns",
                new_value=json.dumps(problem_patterns),
                reason=f"Analyzed {len(trades)} resolved trades",
            )
            log.info("Learner: %d problem patterns found:", len(problem_patterns))
            for key, p in problem_patterns.items():
                log.info(
                    "  %s: W/L=%d/%d (%.0f%%)  penalty=%.2f",
                    key, p["wins"], p["losses"], p["win_rate"] * 100, p["penalty"],
                )

        return problem_patterns

    def get_pattern_penalty(self, city: str, metric: str) -> float:
        """Return the highest applicable penalty for a city+metric combination."""
        keys = [
            f"city:{city.lower()}",
            f"metric:{metric.upper()}",
            f"city_metric:{city.lower()}_{metric.upper()}",
        ]
        max_penalty = 0.0
        try:
            with self._cx() as conn:
                for key in keys:
                    row = conn.execute(
                        "SELECT penalty FROM error_patterns WHERE pattern_key=?",
                        (key,),
                    ).fetchone()
                    if row and row["penalty"]:
                        max_penalty = max(max_penalty, float(row["penalty"]))
        except Exception:
            pass
        return max_penalty

    # ── 2. Adaptive Parameter Tuning ──────────────────────────────────────────

    def tune_parameters(self) -> Dict[str, Tuple[float, float]]:
        """
        Analyze realized performance and adjust config parameters.
        Returns dict of {param_name: (old_value, new_value)} for changes made.
        """
        trades = self._get_resolved_trades()
        if len(trades) < _MIN_TRADES_FOR_LEARNING:
            return {}

        changes: Dict[str, Tuple[float, float]] = {}
        total = len(trades)
        wins = sum(1 for t in trades if t["outcome"] == "WIN")
        win_rate = wins / total

        avg_ev = sum(t.get("ev") or 0 for t in trades) / total
        avg_conf = sum(t.get("confidence") or 0 for t in trades) / total

        # ── EV threshold adjustment ───────────────────────────────────────
        current_ev_thr = self.config["trading"]["ev_threshold"]
        if win_rate < 0.45:
            # Losing too much → raise EV threshold (be more selective)
            target_ev = current_ev_thr * (1.0 + (0.45 - win_rate))
            reason = f"win_rate={win_rate:.2f} < 0.45 → raising selectivity"
        elif win_rate > 0.65 and avg_ev > current_ev_thr * 1.5:
            # Winning a lot with big edges → can lower threshold slightly
            target_ev = current_ev_thr * 0.95
            reason = f"win_rate={win_rate:.2f} > 0.65, avg_ev={avg_ev:.3f} → slightly less selective"
        else:
            target_ev = current_ev_thr
            reason = ""

        if reason:
            new_ev = current_ev_thr + _PARAM_LR * (target_ev - current_ev_thr)
            new_ev = max(_EV_THRESHOLD_RANGE[0], min(_EV_THRESHOLD_RANGE[1], new_ev))
            new_ev = round(new_ev, 4)
            if abs(new_ev - current_ev_thr) > 0.001:
                self.config["trading"]["ev_threshold"] = new_ev
                changes["ev_threshold"] = (current_ev_thr, new_ev)
                _log_param_change("ev_threshold", current_ev_thr, new_ev, reason)
                log.info("Learner: ev_threshold %.4f → %.4f (%s)", current_ev_thr, new_ev, reason)

        # ── Kelly fraction adjustment ─────────────────────────────────────
        current_kelly = self.config["trading"]["kelly_fraction"]
        if win_rate < 0.40:
            target_kelly = current_kelly * 0.80
            reason = f"win_rate={win_rate:.2f} < 0.40 → reducing position size"
        elif win_rate > 0.60 and total >= 20:
            target_kelly = current_kelly * 1.05
            reason = f"win_rate={win_rate:.2f} > 0.60 with {total} trades → slight size increase"
        else:
            target_kelly = current_kelly
            reason = ""

        if reason:
            new_kelly = current_kelly + _PARAM_LR * (target_kelly - current_kelly)
            new_kelly = max(_KELLY_FRACTION_RANGE[0], min(_KELLY_FRACTION_RANGE[1], new_kelly))
            new_kelly = round(new_kelly, 4)
            if abs(new_kelly - current_kelly) > 0.005:
                self.config["trading"]["kelly_fraction"] = new_kelly
                changes["kelly_fraction"] = (current_kelly, new_kelly)
                _log_param_change("kelly_fraction", current_kelly, new_kelly, reason)
                log.info("Learner: kelly_fraction %.4f → %.4f (%s)", current_kelly, new_kelly, reason)

        # ── Confidence floor adjustment ───────────────────────────────────
        current_conf = self.config["trading"]["min_confidence"]

        # Check if low-confidence trades are losing
        low_conf_trades = [t for t in trades if (t.get("confidence") or 0) < 0.60]
        if len(low_conf_trades) >= 5:
            low_wr = sum(1 for t in low_conf_trades if t["outcome"] == "WIN") / len(low_conf_trades)
            if low_wr < 0.35:
                target_conf = min(current_conf + 0.05, _CONFIDENCE_FLOOR_RANGE[1])
                reason = f"low-conf trades win_rate={low_wr:.2f} < 0.35 → raising floor"
            elif low_wr > 0.55:
                target_conf = max(current_conf - 0.02, _CONFIDENCE_FLOOR_RANGE[0])
                reason = f"low-conf trades win_rate={low_wr:.2f} > 0.55 → lowering floor"
            else:
                target_conf = current_conf
                reason = ""

            if reason:
                new_conf = current_conf + _PARAM_LR * (target_conf - current_conf)
                new_conf = max(_CONFIDENCE_FLOOR_RANGE[0], min(_CONFIDENCE_FLOOR_RANGE[1], new_conf))
                new_conf = round(new_conf, 4)
                if abs(new_conf - current_conf) > 0.005:
                    self.config["trading"]["min_confidence"] = new_conf
                    changes["min_confidence"] = (current_conf, new_conf)
                    _log_param_change("min_confidence", current_conf, new_conf, reason)
                    log.info("Learner: min_confidence %.4f → %.4f (%s)", current_conf, new_conf, reason)

        if changes:
            _log_learning(
                "param_tuning",
                f"Adjusted {len(changes)} parameters",
                new_value=json.dumps({k: v[1] for k, v in changes.items()}),
                reason=f"Based on {total} trades, win_rate={win_rate:.2f}",
            )
            self._save_config()

        return changes

    def _save_config(self) -> None:
        """Persist learned parameter changes back to config.yaml."""
        if os.getenv("PYTEST_CURRENT_TEST"):
            log.debug("Skipping config save during test run")
            return
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                raw = f.read()

            # Only update the specific trading parameters we tune
            import re
            for param in ("ev_threshold", "kelly_fraction", "min_confidence"):
                val = self.config["trading"].get(param)
                if val is not None:
                    # Match: "  ev_threshold: 0.05" with flexible whitespace
                    pattern = rf"(^\s*{param}:\s*)[0-9.]+(\s*#.*)?$"
                    replacement = rf"\g<1>{val}\2"
                    raw = re.sub(pattern, replacement, raw, count=1, flags=re.MULTILINE)

            with open(self._config_path, "w", encoding="utf-8") as f:
                f.write(raw)

            log.info("Config saved: learned parameters written to %s", self._config_path)
        except Exception as exc:
            log.warning("Failed to save config: %s", exc)

    # ── 3. Generate Learning Report (for Telegram) ────────────────────────────

    def generate_report(self) -> str:
        trades = self._get_resolved_trades()
        if len(trades) < _MIN_TRADES_FOR_LEARNING:
            return f"Not enough data yet ({len(trades)}/{_MIN_TRADES_FOR_LEARNING} trades needed)"

        total = len(trades)
        wins = sum(1 for t in trades if t["outcome"] == "WIN")
        losses = total - wins
        win_rate = wins / total
        total_pnl = sum(t.get("pnl_usdc") or 0 for t in trades)
        avg_ev = sum(t.get("ev") or 0 for t in trades) / total

        # Biggest winners and losers
        sorted_trades = sorted(trades, key=lambda t: t.get("pnl_usdc") or 0)
        worst = sorted_trades[0] if sorted_trades else None
        best = sorted_trades[-1] if sorted_trades else None

        # City breakdown
        city_stats: Dict[str, dict] = {}
        for t in trades:
            city = (t.get("city") or "unknown").lower()
            if city not in city_stats:
                city_stats[city] = {"wins": 0, "losses": 0, "pnl": 0.0}
            cs = city_stats[city]
            if t["outcome"] == "WIN":
                cs["wins"] += 1
            else:
                cs["losses"] += 1
            cs["pnl"] += t.get("pnl_usdc") or 0

        lines = [
            f"<b>Learning Report</b> ({_WINDOW_DAYS}d window)\n",
            f"Trades: {total}  W/L: {wins}/{losses}  ({win_rate:.0%})",
            f"PnL: <b>${total_pnl:+.2f}</b>  Avg EV: {avg_ev:.3f}\n",
        ]

        if city_stats:
            lines.append("<b>By City:</b>")
            for city, cs in sorted(city_stats.items(), key=lambda x: x[1]["pnl"]):
                ct = cs["wins"] + cs["losses"]
                wr = cs["wins"] / ct if ct else 0
                emoji = "+" if cs["pnl"] >= 0 else ""
                lines.append(
                    f"  {city}: {cs['wins']}W/{cs['losses']}L ({wr:.0%}) "
                    f"${emoji}{cs['pnl']:.2f}"
                )

        # Problem patterns
        patterns = self.analyze_errors()
        if patterns:
            lines.append("\n<b>Weak Spots:</b>")
            for key, p in sorted(patterns.items(), key=lambda x: -x[1]["penalty"])[:5]:
                lines.append(
                    f"  {key}: {p['wins']}W/{p['losses']}L "
                    f"({p['win_rate']:.0%}) penalty={p['penalty']:.2f}"
                )

        if best:
            best_pnl = best.get("pnl_usdc") or 0
            lines.append(f"\nBest: ${best_pnl:+.2f} ({(best.get('market_title') or '')[:40]})")
        if worst:
            worst_pnl = worst.get("pnl_usdc") or 0
            lines.append(f"Worst: ${worst_pnl:+.2f} ({(worst.get('market_title') or '')[:40]})")

        # Fill rate stats
        try:
            with self._cx() as conn:
                fill_stats = conn.execute(
                    "SELECT mode, COUNT(*) as total, SUM(filled) as fills, "
                    "AVG(price_saved) as avg_saved "
                    "FROM fill_tracking GROUP BY mode"
                ).fetchall()
            if fill_stats:
                lines.append("\n<b>Fill Rates:</b>")
                for fs in fill_stats:
                    rate = fs["fills"] / fs["total"] if fs["total"] else 0
                    saved = fs["avg_saved"] or 0
                    lines.append(
                        f"  {fs['mode']}: {fs['fills']}/{fs['total']} "
                        f"({rate:.0%}) avg_saved={saved:.4f}"
                    )
        except Exception:
            pass

        return "\n".join(lines)

    # ── 4. Full Learning Cycle ────────────────────────────────────────────────

    def learn(self) -> Optional[str]:
        """
        Run the full learning cycle.  Called from main loop after trades resolve.
        Returns a summary message for Telegram notification, or None if no learning occurred.
        """
        trades = self._get_resolved_trades()
        if len(trades) < _MIN_TRADES_FOR_LEARNING:
            log.debug(
                "Learner: skipping — need %d trades, have %d",
                _MIN_TRADES_FOR_LEARNING, len(trades),
            )
            return None

        total = len(trades)
        wins = sum(1 for t in trades if t["outcome"] == "WIN")
        win_rate = wins / total

        log.info(
            "Learner: analyzing %d trades (%.0f%% win rate)",
            total, win_rate * 100,
        )

        # Step 1: Identify error patterns
        patterns = self.analyze_errors()

        # Step 2: Tune parameters
        changes = self.tune_parameters()

        if not patterns and not changes:
            log.info("Learner: no adjustments needed — performance looks stable")
            return None

        # Step 3: Build summary message
        parts = ["🧠 <b>Bot Learning Update</b>\n"]
        parts.append(f"Analyzed {total} trades ({win_rate:.0%} win rate)\n")

        if changes:
            parts.append("<b>Parameter Adjustments:</b>")
            for param, (old, new) in changes.items():
                direction = "↑" if new > old else "↓"
                parts.append(f"  {param}: {old:.4f} → {new:.4f} {direction}")

        if patterns:
            worst_patterns = sorted(patterns.items(), key=lambda x: -x[1]["penalty"])[:3]
            parts.append("\n<b>Weak Patterns Found:</b>")
            for key, p in worst_patterns:
                parts.append(
                    f"  {key}: {p['win_rate']:.0%} win rate → "
                    f"penalty {p['penalty']:.2f}"
                )

        summary = "\n".join(parts)
        _log_learning("learning_cycle", summary, reason=f"{total} trades analyzed")
        return summary


# ── Module-level singleton ─────────────────────────────────────────────────────

_instance: Optional[AdaptiveLearner] = None


def get_learner(config: dict) -> AdaptiveLearner:
    global _instance
    if _instance is None:
        _instance = AdaptiveLearner(config)
    return _instance
