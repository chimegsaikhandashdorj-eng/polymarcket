"""
Backtesting engine with two backends:

1. Custom engine (default) — Uses Open-Meteo historical archive for weather
   and Gamma API for past Polymarket markets.
2. Backtrader engine (optional) — Uses the Backtrader framework for crypto
   strategy backtesting with built-in analyzers (Sharpe, drawdown, etc.).

Run via:
  python main.py backtest --start 2024-01-01 --end 2024-12-31
  python main.py backtest --start 2024-01-01 --end 2024-12-31 --engine backtrader
"""

import json
import logging
import math
import random
import re
import sqlite3
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Union

import requests

from .strategy import ProbabilityEngine
from .market_scanner import (
    parse_market_condition,
    RAIN, SNOW, TEMP_ABOVE, TEMP_BELOW, WIND_ABOVE,
    TEMP_RANGE, TEMP_ABOVE_MAX, TEMP_BELOW_MAX,
)
from .logger import DB_PATH

try:
    import backtrader as bt
    import numpy as np
    _HAS_BACKTRADER = True
except ImportError:
    _HAS_BACKTRADER = False

log = logging.getLogger(__name__)

GAMMA_API    = "https://gamma-api.polymarket.com"
OPEN_METEO   = "https://archive-api.open-meteo.com/v1/archive"
_TIMEOUT     = 30
_SESSION     = requests.Session()
_SESSION.headers.update({"User-Agent": "polymarket-weather-bot/1.0"})


def _safe_get(url: str, params: Optional[dict] = None) -> Optional[Union[dict, list]]:
    try:
        r = _SESSION.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        log.warning("Backtest HTTP error %s: %s", url, exc)
        return None


# ── Historical weather ─────────────────────────────────────────────────────────

def fetch_historical_weather(
    lat: float, lon: float, date: str  # YYYY-MM-DD
) -> Optional[dict]:
    """
    Fetch actual observed weather for a given date via Open-Meteo archive.
    Returns normalized dict or None.
    """
    data = _safe_get(OPEN_METEO, params={
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,windspeed_10m_max",
        "timezone": "UTC",
    })
    # Open-Meteo returns an object {daily: {...}}; reject list/None responses.
    if not isinstance(data, dict):
        return None
    try:
        d = data["daily"]
        precip_mm = (d["precipitation_sum"][0] or 0)
        precip_prob = min(1.0, precip_mm / 5.0)  # rough: 5mm+ = high prob
        temp_max = d["temperature_2m_max"][0]
        temp_min = d["temperature_2m_min"][0]
        temp_avg = (temp_max + temp_min) / 2 if temp_max and temp_min else None
        wind_kph = d["windspeed_10m_max"][0]
        return {
            "precip_prob": precip_prob,
            "precip_mm": precip_mm,
            "temp_c": temp_avg,
            "temp_max_c": temp_max,
            "temp_min_c": temp_min,
            "wind_kph": wind_kph,
            "confidence": 0.85,  # historical data is highly reliable
            "sources_used": ["open_meteo_archive"],
        }
    except (KeyError, IndexError, TypeError) as exc:
        log.warning("Historical weather parse error: %s", exc)
        return None


def resolve_outcome(weather: dict, market: dict) -> Optional[str]:
    """
    Determine actual outcome (WIN/LOSS) given real weather and market.
    Returns 'WIN', 'LOSS', or None if unresolvable.
    """
    metric = market.get("metric")
    threshold = market.get("threshold")
    unit = market.get("threshold_unit", "C")
    side = market.get("_side", "YES")  # injected during simulation

    precip_mm = weather.get("precip_mm", 0)
    temp_max  = weather.get("temp_max_c")
    temp_min  = weather.get("temp_min_c")
    wind_kph  = weather.get("wind_kph")

    did_event_happen = None

    if metric == RAIN:
        did_event_happen = precip_mm > 1.0  # >1mm = rainy day

    elif metric == SNOW:
        did_event_happen = precip_mm > 1.0 and (weather.get("temp_c") or 5) < 2.0

    elif metric == TEMP_ABOVE and threshold is not None:
        thr_c = (threshold - 32) * 5 / 9 if (unit or "C").upper() == "F" else threshold
        did_event_happen = temp_max is not None and temp_max > thr_c

    elif metric == TEMP_BELOW and threshold is not None:
        thr_c = (threshold - 32) * 5 / 9 if (unit or "C").upper() == "F" else threshold
        did_event_happen = temp_min is not None and temp_min < thr_c

    elif metric == WIND_ABOVE and threshold is not None:
        thr_kph = threshold * 1.60934 if (unit or "kph").lower() == "mph" else threshold
        did_event_happen = wind_kph is not None and wind_kph > thr_kph

    elif metric == TEMP_ABOVE_MAX and threshold is not None:
        thr_c = (threshold - 32) * 5 / 9 if (unit or "C").upper() == "F" else threshold
        did_event_happen = temp_max is not None and temp_max >= thr_c

    elif metric == TEMP_BELOW_MAX and threshold is not None:
        thr_c = (threshold - 32) * 5 / 9 if (unit or "C").upper() == "F" else threshold
        did_event_happen = temp_max is not None and temp_max <= thr_c

    elif metric == TEMP_RANGE:
        lo = market.get("threshold_low")
        hi = market.get("threshold_high")
        if lo is not None and hi is not None:
            lo_c = (lo - 32) * 5 / 9 if (unit or "C").upper() == "F" else lo
            hi_c = (hi - 32) * 5 / 9 if (unit or "C").upper() == "F" else hi
            did_event_happen = temp_max is not None and lo_c <= temp_max <= hi_c

    if did_event_happen is None:
        return None

    if side == "YES":
        return "WIN" if did_event_happen else "LOSS"
    else:
        return "WIN" if not did_event_happen else "LOSS"


def _resolve_neg_risk(market: dict) -> Optional[str]:
    """
    Resolve a neg-risk market using the outcomePrices field from the Gamma API.
    outcomePrices = ["1","0"] means YES won; ["0","1"] means NO won.
    """
    import json as _json
    raw = market.get("_raw_outcome_prices")
    if raw is None:
        return None
    try:
        prices = _json.loads(raw) if isinstance(raw, str) else raw
        yes_won = float(prices[0]) == 1.0
    except (ValueError, TypeError, IndexError):
        return None
    side = market.get("_side", "YES")
    if side == "YES":
        return "WIN" if yes_won else "LOSS"
    else:
        return "WIN" if not yes_won else "LOSS"


# ── Historical market fetcher ─────────────────────────────────────────────────

def _build_keyword_patterns(keywords: List[str]) -> List[re.Pattern]:
    """Compile weather keywords as whole-word regex patterns to avoid substring false positives.

    E.g. 'rain' would otherwise match 'ukraine', 'rainbow', 'terrain'.
    """
    return [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in keywords]


def _fetch_quarter(
    w_start: str, w_end: str,
    kw_patterns: List[re.Pattern],
    seen_ids: set,
) -> List[dict]:
    """Fetch one 90-day window from the Gamma API, max 50 pages."""
    markets = []
    offset = 0
    limit = 100
    max_pages = 50

    for page in range(max_pages):
        data = _safe_get(f"{GAMMA_API}/markets", params={
            "closed": "true",
            "end_date_min": w_start,
            "end_date_max": w_end,
            "limit": limit,
            "offset": offset,
        })
        if not data or not isinstance(data, list) or not data:
            break

        matched = 0
        for m in data:
            end_dt = (m.get("endDate") or "")[:10]
            if not end_dt or not (w_start <= end_dt <= w_end):
                continue
            title = m.get("question") or m.get("title") or ""
            if not any(pat.search(title) for pat in kw_patterns):
                continue
            cid = m.get("conditionId") or m.get("id") or title
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            markets.append(m)
            matched += 1

        log.info("  %s->%s page %2d: %d returned, %d matched -> %d total",
                 w_start, w_end, page + 1, len(data), matched, len(markets))

        if len(data) < limit:
            break
        offset += limit

    return markets


def fetch_historical_markets(
    start_date: str, end_date: str, keywords: List[str]
) -> List[dict]:
    """
    Load resolved weather markets for backtesting.

    Strategy (in order):
      1. Load from data/discovery/raw_markets.jsonl if it exists.  This file
         is produced by scripts/discover_markets.py and contains the full
         3-year market history with all API fields intact.  Using it avoids
         pagination issues with the Gamma API's undocumented result caps.
      2. Fall back to quarterly-windowed Gamma API fetch if the cache is absent.

    For neg-risk markets (temperature bucket series), lastTradePrice is always
    1 after settlement — the synthetic_price field is computed instead as
    1/group_size so the backtester can evaluate these markets at a realistic
    pre-resolution baseline price.
    """
    cache_path = Path(__file__).resolve().parent.parent / "data" / "discovery" / "raw_markets.jsonl"

    if cache_path.exists():
        log.info("Using pre-discovered market cache: %s", cache_path)
        kw_patterns = _build_keyword_patterns(keywords)
        all_markets = []
        seen_ids: set = set()

        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue

                end_dt_str = (m.get("endDate") or "")[:10]
                if not end_dt_str or not (start_date <= end_dt_str <= end_date):
                    continue

                title = m.get("question") or m.get("title") or ""
                if not any(pat.search(title) for pat in kw_patterns):
                    continue

                cid = m.get("conditionId") or m.get("id") or title
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                all_markets.append(m)

        # Compute group sizes for neg-risk markets so we can assign synthetic prices
        all_markets = _assign_synthetic_prices(all_markets)
        log.info("Loaded %d markets from cache (range %s to %s)",
                 len(all_markets), start_date, end_date)
        return all_markets

    # Fallback: quarterly-windowed Gamma API fetch
    log.info("No market cache found — fetching from Gamma API")
    kw_patterns = _build_keyword_patterns(keywords)
    all_markets = []
    seen_ids = set()

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
    window   = timedelta(days=90)

    cursor = start_dt
    while cursor < end_dt:
        w_start = cursor.strftime("%Y-%m-%d")
        w_end   = min(cursor + window - timedelta(days=1), end_dt).strftime("%Y-%m-%d")
        log.info("Fetching quarter %s -> %s", w_start, w_end)
        batch = _fetch_quarter(w_start, w_end, kw_patterns, seen_ids)
        all_markets.extend(batch)
        cursor += window

    all_markets = _assign_synthetic_prices(all_markets)
    log.info("Historical markets total: %d (range %s to %s)",
             len(all_markets), start_date, end_date)
    return all_markets


def _assign_synthetic_prices(markets: List[dict]) -> List[dict]:
    """
    For neg-risk markets (temperature bucket series), lastTradePrice=1 after
    settlement tells us nothing about pre-resolution prices.  Assign a
    synthetic_price = 1/group_size as a uniform baseline — conservative and
    documented as a known limitation in the Phase 4 backtest report.

    Non-neg-risk markets keep their lastTradePrice untouched.
    """
    # Count buckets per negRiskMarketID group
    group_sizes: dict = {}
    for m in markets:
        neg_id = m.get("negRiskMarketID") or ""
        if neg_id and neg_id.strip("0x").strip("0"):
            group_sizes[neg_id] = group_sizes.get(neg_id, 0) + 1

    # Assign synthetic_price
    for m in markets:
        neg_id = m.get("negRiskMarketID") or ""
        ltp = m.get("lastTradePrice")
        try:
            ltp_f = float(ltp) if ltp is not None else None
        except (ValueError, TypeError):
            ltp_f = None

        is_neg_risk = bool(neg_id and neg_id.strip("0x").strip("0"))
        is_settled  = ltp_f in (0.0, 1.0)

        if is_neg_risk and is_settled:
            n = group_sizes.get(neg_id, 7)
            m["synthetic_price"] = round(1.0 / max(n, 1), 4)
        else:
            m["synthetic_price"] = None  # use lastTradePrice directly

    return markets


# ── SQLite persistence for backtest results ────────────────────────────────────

def _save_backtest_result(row: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO backtest_results
              (run_at, market_id, market_title, side, size_usdc,
               entry_price, our_prob, confidence, ev, outcome, pnl_usdc)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                row["market_id"], row["market_title"], row["side"],
                row["size_usdc"], row["entry_price"], row["our_prob"],
                row["confidence"], row["ev"], row["outcome"], row["pnl_usdc"],
            ),
        )


# ── Main backtester ────────────────────────────────────────────────────────────

class Backtester:
    def __init__(self, config: dict):
        self.config = config
        self.engine = ProbabilityEngine(config)
        self.keywords = config["markets"]["weather_keywords"]
        self.cities = {c["name"].lower(): c for c in config.get("cities", [])}
        self.ev_threshold = config["trading"]["ev_threshold"]
        self.kelly_fraction = config["trading"]["kelly_fraction"]
        self.max_position = config["trading"]["max_position_usdc"]
        self.min_confidence = config["trading"]["min_confidence"]
        self.min_liquidity = config.get("markets", {}).get(
            "backtest_min_liquidity_usdc",
            config.get("markets", {}).get("min_liquidity_usdc", 500),
        )

    def run(
        self,
        start_date: str,
        end_date: str,
        initial_bankroll: float = 500.0,
        simulate_fills: bool = True,
        fill_prob_aggressive: float = 0.95,
        fill_prob_passive: float = 0.65,
        slippage_mean: float = 0.003,
        slippage_std: float = 0.002,
        market_impact_threshold_usdc: float = 20.0,
        market_impact_per_usdc: float = 0.0002,
    ) -> dict:
        """
        Simulate strategy across historical weather markets with realistic execution.

        simulate_fills=True: models missed fills, slippage distribution, and
        market impact to produce more conservative backtest returns.
        """
        from .market_scanner import CityIndex
        city_index = CityIndex(self.config.get("cities", []))

        markets = fetch_historical_markets(start_date, end_date, self.keywords)
        if not markets:
            log.warning("No historical markets found for %s – %s", start_date, end_date)
            return {}

        bankroll = initial_bankroll
        results = []

        for raw in markets:
            title = raw.get("question") or raw.get("title") or ""
            condition_id = raw.get("conditionId") or raw.get("condition_id") or ""
            end_date_str = (raw.get("endDate") or "")[:10]

            if not end_date_str:
                continue

            # Parse city
            city_match = city_index.match(title)
            if not city_match:
                continue
            city_name, coords = city_match
            lat, lon = coords["lat"], coords["lon"]

            # Parse condition
            condition = parse_market_condition(title)
            if not condition["metric"]:
                continue

            # Get actual weather for the market resolution date
            weather = fetch_historical_weather(lat, lon, end_date_str)
            if not weather:
                continue

            # Volume / liquidity check
            if float(raw.get("volume") or raw.get("volumeNum") or 0) < self.min_liquidity:
                log.debug("Skipping illiquid backtest market: %s", title[:60])
                continue

            # Neg-risk temperature bucket markets: Gamma API stores only the settlement
            # price (lastTradePrice=1 for all buckets after resolution), not the actual
            # pre-resolution trading price.  Synthetic 1/n pricing produces grossly
            # inflated EV because it treats all buckets as equally likely.  Historical
            # CLOB price history is not retained by Polymarket for resolved markets.
            # These markets can only be validly backtested against real pre-resolution
            # prices.  Skip them here; paper-trading mode uses live CLOB prices.
            synthetic = raw.get("synthetic_price")
            if synthetic is not None:
                log.debug("Skipping neg-risk market (no real pre-resolution price): %s", title[:60])
                continue

            try:
                raw_price = float(raw.get("lastTradePrice") or 0.5)
            except (ValueError, TypeError):
                log.debug("Skipping market with unparseable price: %s", title[:60])
                continue
            # Skip markets where lastTradePrice is at or near settlement (0 or 1).
            # Values < 0.05 or > 0.95 indicate post-resolution prices — using them
            # produces unrealistic EVs because the market had already largely resolved.
            if raw_price < 0.05 or raw_price > 0.95:
                log.debug(
                    "Skipping near-settled non-neg-risk market (price=%.3f): %s",
                    raw_price, title[:60],
                )
                continue
            raw_price = max(0.01, min(0.99, raw_price))

            market = {
                "condition_id":  condition_id,
                "title":         title,
                "metric":        condition["metric"],
                "threshold":     condition["threshold"],
                "threshold_low": condition.get("threshold_low"),
                "threshold_high":condition.get("threshold_high"),
                "threshold_unit":condition["threshold_unit"],
                "yes_price":     raw_price,
                "no_price":      1.0 - raw_price,
                "volume_usdc":   float(raw.get("volume") or raw.get("volumeNum") or 0),
                "city":          city_name,
                "lat":           lat,
                "lon":           lon,
                "target_dt":     end_date_str,
                "_is_neg_risk":  synthetic is not None,
                "_raw_outcome_prices": raw.get("outcomePrices"),
            }

            opp = self.engine.evaluate(market, weather)
            if opp is None:
                continue

            # Kelly size — use probability of winning the bet, adjusted for side
            price = max(0.01, min(0.99, opp.market_price))
            b = (1.0 / price) - 1.0
            if b <= 0:
                continue
            prob_win = opp.our_prob if opp.side == "YES" else (1.0 - opp.our_prob)
            full_kelly = (prob_win * b - (1 - prob_win)) / b
            size = min(bankroll * full_kelly * self.kelly_fraction, self.max_position)
            if size < 1.0:
                continue

            # ── Realistic execution simulation ─────────────────────────────
            if simulate_fills:
                hours = getattr(opp, "hours_to_expiry", 72.0)
                is_aggressive = (hours < 24 or opp.ev > 0.12
                                 or getattr(opp, "spread", 0.0) > 0.03)
                fill_p = fill_prob_aggressive if is_aggressive else fill_prob_passive
                if random.random() > fill_p:
                    log.debug("Simulated missed fill: %s", market["title"][:50])
                    continue

                # Slippage: sample from half-normal (always positive)
                slip = abs(random.gauss(slippage_mean, slippage_std))
                price = min(0.99, price + slip)

                # Market impact: large orders move price further
                if size > market_impact_threshold_usdc:
                    impact = (size - market_impact_threshold_usdc) * market_impact_per_usdc
                    price = min(0.99, price + impact)

            # Resolve — neg-risk markets use outcomePrices, others use weather model
            market["_side"] = opp.side
            if market.get("_is_neg_risk") and market.get("_raw_outcome_prices"):
                outcome = _resolve_neg_risk(market)
            else:
                outcome = resolve_outcome(weather, market)
            if outcome is None:
                continue

            safe_price = max(0.01, min(0.99, price))
            if outcome == "WIN":
                pnl = size * (1.0 / safe_price - 1.0) * 0.98  # 2% Polymarket fee
            else:
                pnl = -size

            bankroll += pnl

            row = {
                "market_id": condition_id,
                "market_title": title,
                "side": opp.side,
                "size_usdc": round(size, 2),
                "entry_price": round(price, 4),
                "our_prob": opp.our_prob,
                "confidence": opp.confidence,
                "ev": opp.ev,
                "outcome": outcome,
                "pnl_usdc": round(pnl, 2),
            }
            results.append(row)
            _save_backtest_result(row)

            log.debug(
                "Backtest trade: %s %s  pnl=%.2f  bankroll=%.2f",
                opp.side, title[:50], pnl, bankroll,
            )

        return self._summarize(results, initial_bankroll, bankroll, start_date, end_date)

    def walk_forward(
        self,
        start_date: str,
        end_date: str,
        train_months: int = 3,
        test_months: int = 1,
        initial_bankroll: float = 500.0,
    ) -> List[dict]:
        """
        Walk-forward validation: slide a test window through the date range.
        Each test period is out-of-sample — the model has not "seen" these markets.
        Returns a list of per-period summary dicts for stability analysis.
        """
        from datetime import timedelta

        # Parse "YYYY-MM-DD" explicitly so positional args remain (year, month, day)
        # and the type checker doesn't widen the splat to include tzinfo.
        sy, sm, sd = (int(x) for x in start_date.split("-"))
        ey, em, ed = (int(x) for x in end_date.split("-"))
        start = datetime(sy, sm, sd)
        end = datetime(ey, em, ed)
        test_w = timedelta(days=test_months * 30)

        # First test period starts after training period
        cursor = start + timedelta(days=train_months * 30)
        periods: List[dict] = []

        while cursor < end:
            t_start = cursor
            t_end   = min(cursor + test_w, end)
            log.info("Walk-forward test window: %s → %s",
                     t_start.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d"))
            result = self.run(
                t_start.strftime("%Y-%m-%d"),
                t_end.strftime("%Y-%m-%d"),
                initial_bankroll,
            )
            if result and "error" not in result:
                result["test_window"] = (
                    f"{t_start.strftime('%Y-%m-%d')} → {t_end.strftime('%Y-%m-%d')}"
                )
                periods.append(result)
            cursor = t_end

        return periods

    def monte_carlo(
        self,
        results: List[dict],
        initial_bankroll: float = 500.0,
        n_sims: int = 1000,
    ) -> dict:
        """
        Bootstrap Monte Carlo on historical trade results.
        Samples (with replacement) from the trade list n_sims times.
        Returns percentile distribution of final bankrolls and ruin probability.
        """
        import random

        if not results:
            return {}

        finals = []
        for _ in range(n_sims):
            bankroll = initial_bankroll
            sample   = random.choices(results, k=len(results))
            for trade in sample:
                bankroll = max(0.0, bankroll + trade["pnl_usdc"])
            finals.append(bankroll)

        finals.sort()
        p = lambda pct: finals[int(n_sims * pct)]  # noqa: E731
        ruin = sum(1 for b in finals if b < initial_bankroll * 0.20) / n_sims

        return {
            "n_simulations":      n_sims,
            "p5_bankroll":        round(p(0.05), 2),
            "p25_bankroll":       round(p(0.25), 2),
            "median_bankroll":    round(p(0.50), 2),
            "p75_bankroll":       round(p(0.75), 2),
            "p95_bankroll":       round(p(0.95), 2),
            "ruin_probability":   round(ruin, 3),
            "expected_final":     round(sum(finals) / n_sims, 2),
        }

    def _summarize(
        self, results: List[dict], initial: float, final: float,
        start_date: str = "", end_date: str = "",
    ) -> dict:
        if not results:
            return {"error": "No trades in backtest period"}

        total = len(results)
        wins = sum(1 for r in results if r["outcome"] == "WIN")
        total_pnl = sum(r["pnl_usdc"] for r in results)

        # Sharpe ratio — annualised using actual trade frequency, not the
        # equity-market 252-day convention (prediction markets don't trade daily).
        # annualization_factor = sqrt(trades_per_year), where trades_per_year is
        # inferred from the backtest date range and the number of trades.
        pnls = [r["pnl_usdc"] for r in results]
        if len(pnls) > 1:
            avg_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            if start_date and end_date:
                try:
                    n_days = max(1, (
                        datetime.strptime(end_date, "%Y-%m-%d") -
                        datetime.strptime(start_date, "%Y-%m-%d")
                    ).days)
                    trades_per_year = total * 365 / n_days
                except ValueError:
                    trades_per_year = float(total)
            else:
                trades_per_year = float(total)
            sharpe = (avg_pnl / std_pnl) * math.sqrt(trades_per_year) if std_pnl > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        peak = initial
        max_dd = 0.0
        running = initial
        for r in results:
            running += r["pnl_usdc"]
            if running > peak:
                peak = running
            dd = (peak - running) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return {
            "period": f"{start_date} → {end_date}" if start_date else "unknown",
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(wins / total, 3),
            "total_pnl_usdc": round(total_pnl, 2),
            "roi_pct": round((final - initial) / initial * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "final_bankroll": round(final, 2),
            "avg_ev": round(sum(r["ev"] for r in results) / total, 4),
            "avg_confidence": round(
                sum(r["confidence"] for r in results) / total, 3
            ),
        }


# ── Backtrader Engine (optional alternative backend) ─────────────────────────

if _HAS_BACKTRADER:

    class CryptoOHLCVData(bt.feeds.PandasData):
        """Backtrader data feed from CCXT OHLCV via pandas DataFrame."""
        params = (
            ("datetime", None),
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("volume", "volume"),
            ("openinterest", -1),
        )

    class PolymarketCryptoStrategy(bt.Strategy):
        """
        Backtrader strategy that mirrors the bot's crypto signal pipeline:
        RSI + SMA crossover + regime filter + Kelly sizing.
        """
        params = (
            ("rsi_period", 14),
            ("sma_short", 12),
            ("sma_long", 26),
            ("rsi_oversold", 30),
            ("rsi_overbought", 70),
            ("kelly_fraction", 0.15),
            ("max_position_pct", 0.05),
        )

        def __init__(self):
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
            self.sma_short = bt.indicators.SMA(self.data.close, period=self.p.sma_short)
            self.sma_long = bt.indicators.SMA(self.data.close, period=self.p.sma_long)
            self.macd = bt.indicators.MACD(self.data.close)
            self.bbands = bt.indicators.BollingerBands(self.data.close, period=20)
            self.atr = bt.indicators.ATR(self.data, period=14)
            self.order = None
            self.trade_count = 0

        def next(self):
            if self.order:
                return

            if not self.position:
                # Entry conditions: RSI oversold + SMA bullish crossover
                if (self.rsi[0] < self.p.rsi_oversold
                        and self.sma_short[0] > self.sma_long[0]
                        and self.macd.macd[0] > self.macd.signal[0]):
                    size_pct = min(self.p.max_position_pct, self.p.kelly_fraction)
                    size = self.broker.getvalue() * size_pct / self.data.close[0]
                    if size >= 0.001:
                        self.order = self.buy(size=size)
                        self.trade_count += 1

                # Short entry: RSI overbought + SMA bearish crossover
                elif (self.rsi[0] > self.p.rsi_overbought
                      and self.sma_short[0] < self.sma_long[0]
                      and self.macd.macd[0] < self.macd.signal[0]):
                    size_pct = min(self.p.max_position_pct, self.p.kelly_fraction)
                    size = self.broker.getvalue() * size_pct / self.data.close[0]
                    if size >= 0.001:
                        self.order = self.sell(size=size)
                        self.trade_count += 1
            else:
                # Exit on RSI mean reversion
                if self.position.size > 0 and self.rsi[0] > 60:
                    self.order = self.close()
                elif self.position.size < 0 and self.rsi[0] < 40:
                    self.order = self.close()

        def notify_order(self, order):
            if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
                self.order = None


class BacktraderEngine:
    """
    Backtrader-based backtesting engine for crypto strategies.
    Fetches OHLCV data via CCXT and runs the PolymarketCryptoStrategy.
    """

    def __init__(self, config: dict):
        if not _HAS_BACKTRADER:
            raise ImportError("backtrader is not installed. Run: pip install backtrader")
        self.config = config

    def run(
        self,
        asset: str = "bitcoin",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        initial_bankroll: float = 500.0,
        timeframe: str = "1h",
    ) -> dict:
        """
        Run Backtrader backtest on crypto OHLCV data.
        Fetches data via CCXT (Binance default), runs strategy with analyzers.
        """
        import pandas as pd
        from .crypto_fetcher import SUPPORTED_ASSETS, SYMBOL_ALIASES, _HAS_CCXT

        asset_key = SYMBOL_ALIASES.get(asset.lower(), asset.lower())
        if asset_key not in SUPPORTED_ASSETS:
            return {"error": f"Unsupported asset: {asset}"}

        info = SUPPORTED_ASSETS[asset_key]

        # Fetch historical OHLCV via CCXT
        ohlcv_data = self._fetch_ohlcv(info, start_date, end_date, timeframe)
        if ohlcv_data is None or len(ohlcv_data) < 50:
            return {"error": "Insufficient OHLCV data for backtest"}

        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Set up Backtrader Cerebro
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_bankroll)
        cerebro.broker.setcommission(commission=0.001)

        # Add data feed
        data_feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data_feed)

        # Add strategy with config-driven params
        crypto_cfg = self.config.get("crypto", {})
        cerebro.addstrategy(
            PolymarketCryptoStrategy,
            kelly_fraction=self.config.get("trading", {}).get("kelly_fraction", 0.15),
            max_position_pct=crypto_cfg.get("max_position_usdc", 5) / initial_bankroll,
        )

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.05)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        # Run
        log.info("Backtrader: running %s %s → %s on %s", asset_key, start_date, end_date, timeframe)
        results = cerebro.run()
        strat = results[0]

        # Extract results
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        dd_analysis = strat.analyzers.drawdown.get_analysis()
        trade_analysis = strat.analyzers.trades.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()

        final_value = cerebro.broker.getvalue()
        total_trades = trade_analysis.get("total", {}).get("total", 0)
        won = trade_analysis.get("won", {}).get("total", 0)
        lost = trade_analysis.get("lost", {}).get("total", 0)

        return {
            "engine": "backtrader",
            "asset": asset_key,
            "period": f"{start_date} → {end_date}",
            "timeframe": timeframe,
            "initial_bankroll": initial_bankroll,
            "final_bankroll": round(final_value, 2),
            "total_pnl_usdc": round(final_value - initial_bankroll, 2),
            "roi_pct": round((final_value - initial_bankroll) / initial_bankroll * 100, 2),
            "total_trades": total_trades,
            "wins": won,
            "losses": lost,
            "win_rate": round(won / total_trades, 3) if total_trades > 0 else 0,
            "sharpe_ratio": round(sharpe_analysis.get("sharperatio", 0) or 0, 3),
            "max_drawdown_pct": round(dd_analysis.get("max", {}).get("drawdown", 0), 2),
            "total_return_pct": round((returns_analysis.get("rtot", 0) or 0) * 100, 2),
        }

    def _fetch_ohlcv(
        self, asset_info: dict, start_date: str, end_date: str, timeframe: str = "1h"
    ) -> Optional[list]:
        """Fetch OHLCV via CCXT, fallback to Binance direct API."""
        from .crypto_fetcher import _HAS_CCXT, _init_ccxt_exchange

        since_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

        if _HAS_CCXT:
            exchange_ids = self.config.get("crypto", {}).get("ccxt_exchanges", ["binance"])
            for eid in exchange_ids:
                ex = _init_ccxt_exchange(eid)
                if not ex:
                    continue
                try:
                    all_ohlcv = []
                    cursor = since_ms
                    while cursor < end_ms:
                        batch = ex.fetch_ohlcv(
                            asset_info.get("ccxt_symbol", f"{asset_info['symbol']}/USDT"),
                            timeframe=timeframe,
                            since=cursor,
                            limit=1000,
                        )
                        if not batch:
                            break
                        all_ohlcv.extend(batch)
                        cursor = batch[-1][0] + 1
                        if len(batch) < 1000:
                            break
                    if all_ohlcv:
                        log.info("Backtrader OHLCV: %d candles from %s", len(all_ohlcv), eid)
                        return [c for c in all_ohlcv if c[0] <= end_ms]
                except Exception as exc:
                    log.debug("CCXT OHLCV fetch failed (%s): %s", eid, exc)

        # Fallback: Binance direct API
        try:
            all_klines = []
            cursor = since_ms
            while cursor < end_ms:
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    "symbol": asset_info["binance"],
                    "interval": timeframe,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                }
                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                klines = r.json()
                if not klines:
                    break
                for k in klines:
                    all_klines.append([int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
                cursor = int(klines[-1][0]) + 1
                if len(klines) < 1000:
                    break
            log.info("Backtrader OHLCV: %d candles from Binance direct", len(all_klines))
            return all_klines if all_klines else None
        except Exception as exc:
            log.warning("Binance OHLCV fetch failed: %s", exc)
            return None