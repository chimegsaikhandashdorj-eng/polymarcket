"""
Simple web dashboard for the Polymarket Weather Bot.
Run:  python web.py
Open: http://localhost:5000
"""

import os
import sqlite3
from pathlib import Path
from datetime import date

from flask import Flask, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
DB_PATH = Path(__file__).parent / "data" / "trades.db"


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.route("/api/summary")
def api_summary():
    if not DB_PATH.exists():
        return jsonify({"error": "Database not found — run the bot first"})

    with _db() as conn:
        trades = conn.execute(
            "SELECT outcome, pnl_usdc, paper FROM trades WHERE outcome != 'OPEN'"
        ).fetchall()
        open_pos = conn.execute(
            "SELECT COUNT(*) as n FROM trades WHERE outcome='OPEN'"
        ).fetchone()["n"]
        today = date.today().isoformat()
        today_trades = conn.execute(
            "SELECT COUNT(*) as n FROM trades WHERE timestamp LIKE ?", (f"{today}%",)
        ).fetchone()["n"]

    total  = len(trades)
    wins   = sum(1 for t in trades if t["outcome"] == "WIN")
    pnl    = sum(t["pnl_usdc"] or 0 for t in trades)
    paper  = sum(1 for t in trades if t["paper"] == 1)
    live   = total - paper

    return jsonify({
        "total_trades": total,
        "open_positions": open_pos,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "total_pnl": round(pnl, 2),
        "paper_trades": paper,
        "live_trades": live,
        "today_trades": today_trades,
    })


@app.route("/api/trades")
def api_trades():
    if not DB_PATH.exists():
        return jsonify([])
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 50"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/open")
def api_open():
    if not DB_PATH.exists():
        return jsonify([])
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE outcome='OPEN' ORDER BY timestamp DESC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/weather")
def api_weather():
    if not DB_PATH.exists():
        return jsonify([])
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT lat, lon, target_dt, fetched_at, precip_prob, temp_c,
                   humidity, wind_kph, confidence, sources
            FROM weather_cache
            ORDER BY fetched_at DESC LIMIT 20
            """
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/pnl_chart")
def api_pnl_chart():
    """Daily cumulative PnL for charting."""
    if not DB_PATH.exists():
        return jsonify([])
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT DATE(timestamp) as day, SUM(pnl_usdc) as daily_pnl
            FROM trades
            WHERE outcome != 'OPEN'
            GROUP BY DATE(timestamp)
            ORDER BY day
            """
        ).fetchall()
    cumulative = 0
    result = []
    for r in rows:
        cumulative += r["daily_pnl"] or 0
        result.append({"day": r["day"], "pnl": round(cumulative, 2)})
    return jsonify(result)


# ── Main page ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Dashboard: http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
