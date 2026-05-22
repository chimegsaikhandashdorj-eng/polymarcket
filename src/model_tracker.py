"""
Self-learning model accuracy tracker — three subsystems:

  1. Calibration 2.0
     Auto-selects between:
       • Platt scaling (logistic, N < 50)  — compact, converges fast
       • Isotonic regression / PAV (N ≥ 50) — non-parametric, fits
         arbitrary bias shapes without over-smoothing

  2. Dynamic Weights
     Brier-score-based per-source reweighting; sources with lower
     MSE receive higher ensemble weight in WeatherEnsemble.

  3. Meta Model (Trade Filter)
     Logistic regression trained on (EV, spread, confidence, regime,
     hours_to_expiry, our_prob) to estimate P(trade wins).
     Trades below META_VETO_THRESHOLD are filtered before execution.
"""

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from .logger import DB_PATH

log = logging.getLogger(__name__)

_WINDOW_DAYS          = 60    # rolling window for all accuracy metrics
_MIN_CAL_SAMPLES      = 20    # minimum trades before any calibration is applied
_ISO_SWITCH_THRESHOLD = 50    # switch from Platt → isotonic at this N
_MIN_BRIER_SAMPLES    = 10    # minimum per-source samples for Brier reweighting
_META_MIN_SAMPLES     = 30    # minimum resolved trades before meta model activates
META_VETO_THRESHOLD   = 0.40  # meta P(win) below this → veto trade

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "tomorrow_io":    0.30,
    "open_meteo":     0.20,
    "nws":            0.15,
    "weatherapi":     0.15,
    "pirate_weather": 0.10,
    "met_norway":     0.10,
}


def _ensure_tables() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS trade_predictions (
            trade_id    INTEGER PRIMARY KEY,
            raw_prob    REAL NOT NULL,
            calibrated  REAL NOT NULL,
            side        TEXT NOT NULL,
            city        TEXT DEFAULT '',
            metric      TEXT DEFAULT '',
            recorded_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS source_precip_preds (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            loc_key     TEXT NOT NULL,
            target_dt   TEXT NOT NULL,
            source      TEXT NOT NULL,
            raw_precip  REAL NOT NULL,
            outcome     INTEGER,
            recorded_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_spp_unique
            ON source_precip_preds(loc_key, target_dt, source);
        CREATE INDEX IF NOT EXISTS idx_spp_lookup
            ON source_precip_preds(source, outcome, recorded_at);

        CREATE TABLE IF NOT EXISTS calibration_params (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            fitted_at   TEXT NOT NULL,
            cal_A       REAL NOT NULL DEFAULT 1.0,
            cal_B       REAL NOT NULL DEFAULT 0.0,
            n_samples   INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS calibration_isotonic (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            fitted_at   TEXT NOT NULL,
            n_samples   INTEGER NOT NULL,
            iso_xs      TEXT NOT NULL,
            iso_ys      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trade_meta_features (
            trade_id        INTEGER PRIMARY KEY,
            ev              REAL NOT NULL,
            spread          REAL NOT NULL,
            confidence      REAL NOT NULL,
            regime_code     REAL NOT NULL,
            hours_to_expiry REAL NOT NULL,
            our_prob        REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meta_model_params (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            fitted_at   TEXT NOT NULL,
            n_samples   INTEGER NOT NULL,
            weights     TEXT NOT NULL,
            bias        REAL NOT NULL
        );
        """)


class ModelTracker:
    """
    Tracks prediction accuracy for calibration and ensemble reweighting.
    Thread-safe: every public method opens and closes its own DB connection.
    """

    def __init__(self):
        _ensure_tables()

        # ── Calibration state ─────────────────────────────────────────────────
        self._cal_method: str   = "none"   # "platt", "isotonic", or "none"
        self._cal_A: float      = 1.0      # Platt sigmoid slope
        self._cal_B: float      = 0.0      # Platt sigmoid intercept
        self._cal_fitted: bool  = False
        self._iso_x: List[float] = []      # Isotonic breakpoint x values
        self._iso_y: List[float] = []      # Isotonic breakpoint y (calibrated) values

        # ── Meta model state ──────────────────────────────────────────────────
        self._meta_W: List[float] = [0.0] * 6
        self._meta_b: float       = 0.0
        self._meta_fitted: bool   = False

        # ── Dynamic weights cache ─────────────────────────────────────────────
        self._weights_cache: Optional[Dict[str, float]] = None
        self._weights_ts: float  = 0.0
        self._weights_ttl: float = 3600.0

        self._load_calibration()
        self._load_meta_model()

    # ── DB connection ─────────────────────────────────────────────────────────

    def _cx(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── Calibration: public API ───────────────────────────────────────────────

    def calibrate(self, raw_prob: float) -> float:
        """
        Apply calibration to a raw ensemble probability.
        Routes to isotonic or Platt based on which was last fitted.
        Returns raw_prob unchanged until enough trades resolve.
        """
        if self._cal_method == "isotonic" and self._iso_x:
            return self._isotonic_predict(raw_prob)
        if self._cal_method == "platt" and self._cal_fitted:
            return self._platt_predict(raw_prob)
        return raw_prob

    def fit_calibration(self) -> bool:
        """
        Auto-select and fit the best calibrator from resolved trades.
        N < 50  → Platt scaling (gradient descent on cross-entropy)
        N ≥ 50  → Isotonic regression (pool adjacent violators)
        Returns True if parameters were updated.
        """
        cutoff = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=_WINDOW_DAYS)
        ).isoformat()

        try:
            with self._cx() as conn:
                rows = conn.execute(
                    """
                    SELECT tp.raw_prob, tp.side, t.outcome
                    FROM trade_predictions tp
                    JOIN trades t ON t.id = tp.trade_id
                    WHERE t.outcome IN ('WIN', 'LOSS') AND tp.recorded_at > ?
                    """,
                    (cutoff,),
                ).fetchall()
        except Exception as exc:
            log.warning("Calibration fit: DB read failed: %s", exc)
            return False

        n = len(rows)
        if n < _MIN_CAL_SAMPLES:
            log.debug("Calibration: need ≥%d samples, have %d", _MIN_CAL_SAMPLES, n)
            return False

        # Convert to (raw_prob, yes_won) pairs
        pairs: List[Tuple[float, int]] = []
        for r in rows:
            yes_won = (r["outcome"] == "WIN") == (r["side"] == "YES")
            pairs.append((float(r["raw_prob"]), int(yes_won)))

        if n >= _ISO_SWITCH_THRESHOLD:
            xs, ys = self._fit_isotonic(pairs)
            self._iso_x = xs
            self._iso_y = ys
            self._cal_method = "isotonic"
            self._cal_fitted = True
            self._save_isotonic(n, xs, ys)
            log.info("Calibration 2.0: isotonic fitted (n=%d, %d breakpoints)", n, len(xs))
        else:
            A, B = self._fit_platt(pairs)
            self._cal_A, self._cal_B = A, B
            self._cal_method = "platt"
            self._cal_fitted = True
            self._save_calibration(n)
            log.info("Calibration 2.0: Platt fitted (n=%d) A=%.4f B=%.4f", n, A, B)

        return True

    # ── Isotonic regression (Pool Adjacent Violators) ─────────────────────────

    def _fit_isotonic(
        self, pairs: List[Tuple[float, int]]
    ) -> Tuple[List[float], List[float]]:
        """
        Isotonic regression via PAV algorithm.
        Finds the non-decreasing step function minimising sum-of-squares error.
        Returns (xs, ys) breakpoints suitable for linear interpolation.
        """
        # Sort by x; group identical x values together
        pts = sorted(pairs, key=lambda p: p[0])
        # Initial pools: (mean_x, mean_y, count)
        pools: List[Tuple[float, float, int]] = [
            (float(x), float(y), 1) for x, y in pts
        ]

        # Merge violating adjacent pools until monotone
        changed = True
        while changed:
            changed = False
            merged: List[Tuple[float, float, int]] = []
            i = 0
            while i < len(pools):
                if i + 1 < len(pools) and pools[i][1] > pools[i + 1][1]:
                    x0, y0, n0 = pools[i]
                    x1, y1, n1 = pools[i + 1]
                    new_x = (x0 * n0 + x1 * n1) / (n0 + n1)
                    new_y = (y0 * n0 + y1 * n1) / (n0 + n1)
                    merged.append((new_x, new_y, n0 + n1))
                    i += 2
                    changed = True
                else:
                    merged.append(pools[i])
                    i += 1
            pools = merged

        xs = [p[0] for p in pools]
        ys = [max(0.02, min(0.98, p[1])) for p in pools]
        return xs, ys

    def _isotonic_predict(self, x: float) -> float:
        """Linear interpolation on PAV breakpoints."""
        if x <= self._iso_x[0]:
            return self._iso_y[0]
        if x >= self._iso_x[-1]:
            return self._iso_y[-1]
        lo, hi = 0, len(self._iso_x) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._iso_x[mid] <= x:
                lo = mid
            else:
                hi = mid
        x0, x1 = self._iso_x[lo], self._iso_x[hi]
        y0, y1 = self._iso_y[lo], self._iso_y[hi]
        t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
        return max(0.02, min(0.98, y0 + t * (y1 - y0)))

    # ── Platt scaling ─────────────────────────────────────────────────────────

    def _fit_platt(self, pairs: List[Tuple[float, int]]) -> Tuple[float, float]:
        """Gradient descent on logistic cross-entropy. Returns (A, B)."""
        A, B = self._cal_A, self._cal_B
        lr = 0.1
        for step in range(800):
            dA = dB = 0.0
            for raw_p, outcome in pairs:
                try:
                    pred = 1.0 / (1.0 + math.exp(-(A * raw_p + B)))
                except OverflowError:
                    pred = 0.0 if (A * raw_p + B) < 0 else 1.0
                pred = max(1e-9, min(1.0 - 1e-9, pred))
                err = pred - outcome
                dA += err * raw_p
                dB += err
            n = len(pairs)
            A -= lr * dA / n
            B -= lr * dB / n
            if step % 200 == 199:
                lr *= 0.5
        return A, B

    def _platt_predict(self, raw_prob: float) -> float:
        try:
            val = self._cal_A * raw_prob + self._cal_B
            return max(0.02, min(0.98, 1.0 / (1.0 + math.exp(-val))))
        except (OverflowError, ValueError):
            return raw_prob

    # ── Calibration persistence ───────────────────────────────────────────────

    def _load_calibration(self) -> None:
        """Load the most recent calibration params from DB (both methods)."""
        try:
            with self._cx() as conn:
                # Isotonic
                iso_row = conn.execute(
                    "SELECT iso_xs, iso_ys, n_samples FROM calibration_isotonic "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if iso_row and iso_row["n_samples"] >= _ISO_SWITCH_THRESHOLD:
                    self._iso_x = json.loads(iso_row["iso_xs"])
                    self._iso_y = json.loads(iso_row["iso_ys"])
                    self._cal_method = "isotonic"
                    self._cal_fitted = True
                    log.info(
                        "Calibration loaded: isotonic (%d breakpoints, n=%d)",
                        len(self._iso_x), iso_row["n_samples"],
                    )
                    return

                # Platt fallback
                row = conn.execute(
                    "SELECT cal_A, cal_B, n_samples FROM calibration_params "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row and row["n_samples"] >= _MIN_CAL_SAMPLES:
                    self._cal_A = row["cal_A"]
                    self._cal_B = row["cal_B"]
                    self._cal_method = "platt"
                    self._cal_fitted = True
                    log.info("Calibration loaded: Platt A=%.4f B=%.4f", self._cal_A, self._cal_B)
        except Exception:
            pass  # first run

    def _save_calibration(self, n_samples: int) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        with self._cx() as conn:
            conn.execute(
                "INSERT INTO calibration_params (fitted_at, cal_A, cal_B, n_samples) "
                "VALUES (?,?,?,?)",
                (now, self._cal_A, self._cal_B, n_samples),
            )

    def _save_isotonic(self, n_samples: int, xs: List[float], ys: List[float]) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        with self._cx() as conn:
            conn.execute(
                "INSERT INTO calibration_isotonic (fitted_at, n_samples, iso_xs, iso_ys) "
                "VALUES (?,?,?,?)",
                (now, n_samples, json.dumps(xs), json.dumps(ys)),
            )

    # ── Trade recording ───────────────────────────────────────────────────────

    def record_trade(
        self,
        trade_id: int,
        raw_prob: float,
        calibrated: float,
        side: str,
        city: str = "",
        metric: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        try:
            with self._cx() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO trade_predictions
                      (trade_id, raw_prob, calibrated, side, city, metric, recorded_at)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (trade_id, raw_prob, calibrated, side, city, metric, now),
                )
        except Exception as exc:
            log.debug("ModelTracker.record_trade failed: %s", exc)

    def record_meta_features(
        self,
        trade_id: int,
        ev: float,
        spread: float,
        confidence: float,
        regime: str,
        hours_to_expiry: float,
        our_prob: float,
    ) -> None:
        """Store trade features for meta model training."""
        regime_code = 0.0 if regime == "NORMAL" else 0.5
        try:
            with self._cx() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO trade_meta_features
                      (trade_id, ev, spread, confidence, regime_code, hours_to_expiry, our_prob)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (trade_id, ev, spread, confidence, regime_code, hours_to_expiry, our_prob),
                )
        except Exception as exc:
            log.debug("ModelTracker.record_meta_features failed: %s", exc)

    # ── Meta model: fit + predict ─────────────────────────────────────────────

    def fit_meta_model(self) -> bool:
        """
        Fit logistic regression on resolved trade features to predict P(win).
        Features: [ev, spread, confidence, regime_code, log1p(hours)/5, our_prob]
        Returns True if model was updated.
        """
        try:
            with self._cx() as conn:
                rows = conn.execute(
                    """
                    SELECT f.ev, f.spread, f.confidence, f.regime_code,
                           f.hours_to_expiry, f.our_prob, t.outcome
                    FROM trade_meta_features f
                    JOIN trades t ON t.id = f.trade_id
                    WHERE t.outcome IN ('WIN', 'LOSS')
                    """,
                ).fetchall()
        except Exception as exc:
            log.warning("Meta model fit: DB read failed: %s", exc)
            return False

        if len(rows) < _META_MIN_SAMPLES:
            log.debug("Meta model: need ≥%d samples, have %d", _META_MIN_SAMPLES, len(rows))
            return False

        data: List[Tuple[List[float], float]] = []
        for r in rows:
            x = [
                float(r["ev"]),
                float(r["spread"]),
                float(r["confidence"]),
                float(r["regime_code"]),
                math.log1p(max(0, float(r["hours_to_expiry"]))) / 5.0,
                float(r["our_prob"]),
            ]
            y = 1.0 if r["outcome"] == "WIN" else 0.0
            data.append((x, y))

        n_feat = 6
        W = self._meta_W if len(self._meta_W) == n_feat else [0.0] * n_feat
        b = self._meta_b
        lr = 0.1

        for step in range(800):
            dW = [0.0] * n_feat
            db = 0.0
            for x, y in data:
                z = sum(W[i] * x[i] for i in range(n_feat)) + b
                try:
                    pred = 1.0 / (1.0 + math.exp(-z))
                except OverflowError:
                    pred = 0.0 if z < 0 else 1.0
                pred = max(1e-9, min(1.0 - 1e-9, pred))
                err = pred - y
                for i in range(n_feat):
                    dW[i] += err * x[i]
                db += err
            n = len(data)
            W = [W[i] - lr * dW[i] / n for i in range(n_feat)]
            b -= lr * db / n
            if step % 200 == 199:
                lr *= 0.5

        self._meta_W = W
        self._meta_b = b
        self._meta_fitted = True
        self._save_meta_model(len(data))
        log.info(
            "Meta model fitted (n=%d): EV_w=%.3f spread_w=%.3f conf_w=%.3f "
            "regime_w=%.3f hours_w=%.3f prob_w=%.3f  bias=%.3f",
            len(data), W[0], W[1], W[2], W[3], W[4], W[5], b,
        )
        return True

    def meta_predict(
        self,
        ev: float,
        spread: float,
        confidence: float,
        regime: str,
        hours_to_expiry: float,
        our_prob: float,
    ) -> float:
        """
        Return P(trade wins) from the meta model.
        Returns 0.5 (neutral) if the model has not yet been fitted.
        """
        if not self._meta_fitted or len(self._meta_W) != 6:
            return 0.5
        regime_code = 0.0 if regime == "NORMAL" else 0.5
        x = [
            ev,
            spread,
            confidence,
            regime_code,
            math.log1p(max(0, hours_to_expiry)) / 5.0,
            our_prob,
        ]
        z = sum(self._meta_W[i] * x[i] for i in range(6)) + self._meta_b
        try:
            return max(0.02, min(0.98, 1.0 / (1.0 + math.exp(-z))))
        except OverflowError:
            return 0.0 if z < 0 else 1.0

    def _save_meta_model(self, n_samples: int) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        with self._cx() as conn:
            conn.execute(
                "INSERT INTO meta_model_params (fitted_at, n_samples, weights, bias) "
                "VALUES (?,?,?,?)",
                (now, n_samples, json.dumps(self._meta_W), self._meta_b),
            )

    def _load_meta_model(self) -> None:
        try:
            with self._cx() as conn:
                row = conn.execute(
                    "SELECT weights, bias, n_samples FROM meta_model_params "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if row and row["n_samples"] >= _META_MIN_SAMPLES:
                self._meta_W = json.loads(row["weights"])
                self._meta_b = float(row["bias"])
                self._meta_fitted = True
                log.info("Meta model loaded (n=%d)", row["n_samples"])
        except Exception:
            pass

    # ── Dynamic weights (Brier score) ─────────────────────────────────────────

    def record_source_precip(
        self, lat: float, lon: float, target_dt: str, source: str, raw_precip: float
    ) -> None:
        loc_key = f"{lat:.4f},{lon:.4f}"
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        try:
            with self._cx() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO source_precip_preds
                      (loc_key, target_dt, source, raw_precip, recorded_at)
                    VALUES (?,?,?,?,?)
                    """,
                    (loc_key, target_dt, source, raw_precip, now),
                )
        except Exception as exc:
            log.debug("record_source_precip failed: %s", exc)

    def mark_rain_outcome(self, lat: float, lon: float, target_dt: str, did_rain: bool) -> None:
        loc_key = f"{lat:.4f},{lon:.4f}"
        try:
            with self._cx() as conn:
                conn.execute(
                    "UPDATE source_precip_preds SET outcome=? "
                    "WHERE loc_key=? AND target_dt=? AND outcome IS NULL",
                    (int(did_rain), loc_key, target_dt),
                )
        except Exception as exc:
            log.debug("mark_rain_outcome failed: %s", exc)

    def get_dynamic_weights(self) -> Dict[str, float]:
        """
        Per-source weights from Brier score (lower MSE → higher weight).
        Falls back to _DEFAULT_WEIGHTS when data is insufficient.
        """
        import time as _time
        now = _time.time()
        if self._weights_cache and (now - self._weights_ts) < self._weights_ttl:
            return self._weights_cache

        cutoff = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=_WINDOW_DAYS)
        ).isoformat()

        try:
            with self._cx() as conn:
                rows = conn.execute(
                    """
                    SELECT source,
                           AVG((raw_precip - outcome) * (raw_precip - outcome)) AS brier,
                           COUNT(*) AS n
                    FROM source_precip_preds
                    WHERE outcome IS NOT NULL AND recorded_at > ?
                    GROUP BY source
                    """,
                    (cutoff,),
                ).fetchall()
        except Exception as exc:
            log.debug("get_dynamic_weights DB read failed: %s — using defaults", exc)
            return _DEFAULT_WEIGHTS

        qualified = {r["source"]: r["brier"] for r in rows if r["n"] >= _MIN_BRIER_SAMPLES}
        if not qualified:
            return _DEFAULT_WEIGHTS

        raw = {src: 1.0 / max(0.001, b) for src, b in qualified.items()}
        mean_raw = sum(raw.values()) / len(raw)
        for src in _DEFAULT_WEIGHTS:
            if src not in raw:
                raw[src] = mean_raw * 0.5

        total = sum(raw.values())
        weights = {src: w / total for src, w in raw.items()}

        self._weights_cache = weights
        self._weights_ts = now
        log.info("Dynamic weights updated (%d sources qualified):", len(qualified))
        for src, w in sorted(weights.items(), key=lambda x: -x[1]):
            b = qualified.get(src, 0)
            log.info("  %-18s  w=%.3f  brier=%.4f", src, w, b)
        return weights

    # ── Drift detection & maintenance ─────────────────────────────────────────

    def detect_drift(self, window_days: int = 30) -> Optional[str]:
        """
        Compare mean calibrated probability vs actual win rate over the last
        window_days.  Returns a warning string if |drift| > 10%, else None.
        A positive drift means the model is over-confident (predicts higher
        probability than the actual win rate); negative means under-confident.
        """
        cutoff = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=window_days)
        ).isoformat()
        try:
            with self._cx() as conn:
                rows = conn.execute(
                    """
                    SELECT tp.calibrated, tp.side, t.outcome
                    FROM trade_predictions tp
                    JOIN trades t ON t.id = tp.trade_id
                    WHERE t.outcome IN ('WIN','LOSS') AND tp.recorded_at > ?
                    """,
                    (cutoff,),
                ).fetchall()
        except Exception as exc:
            log.debug("detect_drift DB read failed: %s", exc)
            return None

        if len(rows) < 10:
            return None

        # "our_prob" = probability that OUR SIDE wins.
        # calibrated stores P(YES wins); for NO trades, our_prob = 1 - calibrated.
        # We compare mean(our_prob) vs fraction of rows where we actually won.
        our_probs = [
            r["calibrated"] if r["side"] == "YES" else (1.0 - r["calibrated"])
            for r in rows
        ]
        mean_cal  = sum(our_probs) / len(our_probs)
        actual_wr = sum(1 for r in rows if r["outcome"] == "WIN") / len(rows)
        drift = mean_cal - actual_wr

        if abs(drift) > 0.10:
            direction = "over-confident" if drift > 0 else "under-confident"
            msg = (
                f"Calibration drift ({window_days}d window): "
                f"mean_prob={mean_cal:.3f}  actual_win_rate={actual_wr:.3f}  "
                f"drift={drift:+.3f}  [{direction}]  n={len(rows)}"
            )
            log.warning(msg)
            return msg
        return None

    def purge_stale_data(self, max_age_days: int = 90, keep_cal_rows: int = 10) -> int:
        """
        Prevent unbounded table growth:
        - source_precip_preds: delete rows older than max_age_days
        - calibration_params / calibration_isotonic: keep only the latest keep_cal_rows

        Returns total number of rows deleted.
        """
        cutoff = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=max_age_days)
        ).isoformat()
        total_deleted = 0
        try:
            with self._cx() as conn:
                # 1. Old source predictions
                cur = conn.execute(
                    "DELETE FROM source_precip_preds WHERE recorded_at < ?",
                    (cutoff,),
                )
                total_deleted += cur.rowcount

                # 2. Excess calibration_params history (keep latest N)
                cur = conn.execute(
                    "DELETE FROM calibration_params WHERE id NOT IN "
                    "(SELECT id FROM calibration_params ORDER BY id DESC LIMIT ?)",
                    (keep_cal_rows,),
                )
                total_deleted += cur.rowcount

                # 3. Excess calibration_isotonic history
                cur = conn.execute(
                    "DELETE FROM calibration_isotonic WHERE id NOT IN "
                    "(SELECT id FROM calibration_isotonic ORDER BY id DESC LIMIT ?)",
                    (keep_cal_rows,),
                )
                total_deleted += cur.rowcount

                # 4. Excess meta_model_params history
                cur = conn.execute(
                    "DELETE FROM meta_model_params WHERE id NOT IN "
                    "(SELECT id FROM meta_model_params ORDER BY id DESC LIMIT ?)",
                    (keep_cal_rows,),
                )
                total_deleted += cur.rowcount

            if total_deleted:
                log.info("purge_stale_data: deleted %d rows total", total_deleted)
            return total_deleted
        except Exception as exc:
            log.debug("purge_stale_data failed: %s", exc)
            return 0

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_brier_report(self) -> Dict[str, dict]:
        cutoff = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=_WINDOW_DAYS)
        ).isoformat()
        try:
            with self._cx() as conn:
                rows = conn.execute(
                    """
                    SELECT source,
                           AVG((raw_precip - outcome)*(raw_precip - outcome)) AS brier,
                           COUNT(*) AS n
                    FROM source_precip_preds
                    WHERE outcome IS NOT NULL AND recorded_at > ?
                    GROUP BY source
                    """,
                    (cutoff,),
                ).fetchall()
            return {r["source"]: {"brier": round(r["brier"], 4), "n": r["n"]} for r in rows}
        except Exception:
            return {}

    def get_calibration_info(self) -> dict:
        return {
            "method":   self._cal_method,
            "fitted":   self._cal_fitted,
            "A":        round(self._cal_A, 4),       # Platt
            "B":        round(self._cal_B, 4),       # Platt
            "iso_pts":  len(self._iso_x),            # Isotonic
            "meta":     self._meta_fitted,
        }


# ── Module-level singleton ─────────────────────────────────────────────────────

_instance: Optional[ModelTracker] = None


def get_tracker() -> ModelTracker:
    global _instance
    if _instance is None:
        _instance = ModelTracker()
    return _instance
