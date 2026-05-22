"""
Phase 1.1 — Full Historical Market Pull

Pulls every weather-related market from Polymarket Gamma API for
2023-01-01 → 2025-12-31 and saves raw JSON to data/discovery/raw_markets.jsonl.

Run:  python scripts/discover_markets.py
      python scripts/discover_markets.py --start 2023-01-01 --end 2025-12-31
      python scripts/discover_markets.py --append   # add to existing file instead of overwrite

Expected: 800-3,000 resolved weather-adjacent markets across 3 years.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "polymarket-weather-bot/1.0"})

# Broader keyword set than the live bot — cast a wide net for discovery
DISCOVERY_KEYWORDS = [
    "weather", "rain", "rainfall", "precip", "precipitation",
    "snow", "snowfall", "blizzard", "snowstorm",
    "temperature", "temp", "degrees", "fahrenheit", "celsius",
    "hot", "cold", "warm", "cool", "freeze", "frost", "freezing",
    "storm", "thunderstorm", "lightning",
    "hurricane", "tropical storm", "cyclone", "typhoon", "landfall",
    "tornado", "twister",
    "wind", "windspeed",
    "flood", "flooding", "flash flood",
    "drought", "heatwave", "heat wave", "heat index",
    "inches of rain", "inches of snow", "inches of snowfall",
    "record high", "record low", "all-time", "hottest", "coldest",
    "wettest", "driest",
]

# Compile as word-boundary patterns to avoid "rain" in "ukraine" etc.
_KW_PATTERNS = [
    re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
    for kw in DISCOVERY_KEYWORDS
]


def _is_weather_adjacent(title: str) -> bool:
    return any(pat.search(title) for pat in _KW_PATTERNS)


def _safe_get(url: str, params: dict) -> list | None:
    for attempt in range(3):
        try:
            r = _SESSION.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else None
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response else 0
            if code in (400, 401, 403, 404):
                print(f"  HTTP {code} — stopping pagination", flush=True)
                return None
            wait = 2.0 ** attempt
            print(f"  HTTP {code}, retry in {wait:.0f}s…", flush=True)
            time.sleep(wait)
        except requests.RequestException as exc:
            print(f"  Request error: {exc}", flush=True)
            if attempt < 2:
                time.sleep(2.0 ** attempt)
    return None


def pull_year(start: str, end: str, seen_ids: set) -> list:
    """Fetch all closed weather markets in the [start, end] window."""
    markets = []
    offset = 0
    limit = 100
    max_pages = 50  # 5,000 markets per window — more than enough

    print(f"\nPulling {start} -> {end}", flush=True)

    for page in range(max_pages):
        data = _safe_get(f"{GAMMA_API}/markets", params={
            "closed": "true",
            "end_date_min": start,
            "end_date_max": end,
            "limit": limit,
            "offset": offset,
        })

        if not data:
            print(f"  Page {page+1}: empty/error — done", flush=True)
            break

        in_window = 0
        matched = 0
        for m in data:
            end_dt = (m.get("endDate") or "")[:10]
            if not end_dt or not (start <= end_dt <= end):
                continue
            in_window += 1

            title = m.get("question") or m.get("title") or ""
            if not _is_weather_adjacent(title):
                continue

            cid = m.get("conditionId") or m.get("id") or title
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            markets.append(m)
            matched += 1

        print(
            f"  Page {page+1:2d}/{max_pages}: {len(data):3d} returned, "
            f"{in_window:3d} in window, {matched:3d} weather -> {len(markets):4d} total",
            flush=True,
        )

        if len(data) < limit:
            print(f"  Last page (returned {len(data)} < {limit})", flush=True)
            break

        offset += limit

    return markets


def main():
    parser = argparse.ArgumentParser(description="Pull Polymarket weather market history")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing file instead of overwriting")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "data" / "discovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_markets.jsonl"

    # Load already-seen IDs to avoid duplicates when appending
    seen_ids: set = set()
    if args.append and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    m = json.loads(line)
                    cid = m.get("conditionId") or m.get("id") or ""
                    if cid:
                        seen_ids.add(cid)
                except json.JSONDecodeError:
                    pass
        print(f"Append mode: {len(seen_ids)} markets already on disk", flush=True)

    # Split into quarterly windows — Gamma API has undocumented result caps
    # so smaller windows catch more markets
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    window_days = 90  # quarterly

    all_markets = []
    cursor = start_dt
    while cursor < end_dt:
        w_start = cursor.strftime("%Y-%m-%d")
        w_end = min(cursor + timedelta(days=window_days - 1), end_dt).strftime("%Y-%m-%d")
        batch = pull_year(w_start, w_end, seen_ids)
        all_markets.extend(batch)
        cursor += timedelta(days=window_days)

    print(f"\nTotal new markets pulled: {len(all_markets)}", flush=True)

    if not all_markets:
        print("No markets found — check API connectivity and date range.", flush=True)
        sys.exit(1)

    # Write JSONL
    mode = "a" if args.append else "w"
    written = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for m in all_markets:
            f.write(json.dumps(m) + "\n")
            written += 1

    print(f"\nSaved {written} markets to {out_path}", flush=True)
    print("Next step: python scripts/analyze_markets.py", flush=True)


if __name__ == "__main__":
    main()
