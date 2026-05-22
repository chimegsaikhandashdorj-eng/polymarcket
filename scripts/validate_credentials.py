"""
Phase 0 preflight: validate all API keys and connectivity.

Exit code 0  = all required keys valid.
Exit code 1  = one or more required keys missing or rejected.

Usage:
  python scripts/validate_credentials.py
"""

import os
import sys
import time
from pathlib import Path

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
SKIP = "\033[90mSKIP\033[0m"

results: list[tuple[str, str, str]] = []   # (name, status, detail)


def check(name: str, ok: bool, detail: str = "", required: bool = True) -> None:
    status = PASS if ok else (FAIL if required else WARN)
    results.append((name, status, detail))
    tag = "[REQUIRED]" if required else "[OPTIONAL]"
    icon = "v" if ok else "x"
    print(f"  {icon} {tag} {name:<35} {status}  {detail}")


# ── Environment variables ─────────────────────────────────────────────────────

print("\n=== Checking environment variables ===")

tomorrow = os.getenv("TOMORROW_API_KEY", "")
check("TOMORROW_API_KEY",    bool(tomorrow),     "key present" if tomorrow else "missing")

weatherapi = os.getenv("WEATHERAPI_KEY", "")
check("WEATHERAPI_KEY",     bool(weatherapi),   "key present" if weatherapi else "missing", required=False)

pirate = os.getenv("PIRATE_WEATHER_KEY", "")
check("PIRATE_WEATHER_KEY", bool(pirate),        "key present" if pirate else "missing",    required=False)

tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
tg_chat  = os.getenv("TELEGRAM_CHAT_ID",   "")
check("TELEGRAM_BOT_TOKEN", bool(tg_token),      "token present" if tg_token else "missing", required=False)
check("TELEGRAM_CHAT_ID",   bool(tg_chat),       "id present" if tg_chat else "missing",     required=False)

poly_pk = os.getenv("POLY_PRIVATE_KEY", "")
check("POLY_PRIVATE_KEY",   bool(poly_pk),       "key present" if poly_pk else "missing (live mode only)", required=False)

bankroll = os.getenv("BANKROLL_USDC", "")
check("BANKROLL_USDC",      bool(bankroll),      bankroll if bankroll else "will default to 500")

# ── API connectivity ──────────────────────────────────────────────────────────

print("\n=== Testing API connectivity ===")

try:
    import requests
    _S = requests.Session()
    _S.headers["User-Agent"] = "polymarket-bot-preflight/1.0"
    _S.request.__doc__  # just verify import

    # Tomorrow.io
    if tomorrow:
        try:
            r = _S.get(
                "https://api.tomorrow.io/v4/weather/realtime",
                params={"location": "40.71,-74.01", "apikey": tomorrow, "units": "metric"},
                timeout=10,
            )
            ok = r.status_code == 200
            check("Tomorrow.io API",      ok, f"HTTP {r.status_code}")
        except Exception as exc:
            check("Tomorrow.io API",      False, str(exc)[:60])
    else:
        check("Tomorrow.io API",          False, "no key — skipped", required=False)

    # Open-Meteo (no key needed)
    try:
        r = _S.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": 40.71, "longitude": -74.01, "hourly": "temperature_2m", "forecast_days": 1},
            timeout=10,
        )
        check("Open-Meteo API",           r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as exc:
        check("Open-Meteo API",           False, str(exc)[:60])

    # WeatherAPI
    if weatherapi:
        try:
            r = _S.get(
                "https://api.weatherapi.com/v1/current.json",
                params={"key": weatherapi, "q": "New York"},
                timeout=10,
            )
            check("WeatherAPI",           r.status_code == 200, f"HTTP {r.status_code}")
        except Exception as exc:
            check("WeatherAPI",           False, str(exc)[:60], required=False)
    else:
        check("WeatherAPI",               False, "no key — skipped", required=False)

    # Met.no (no key)
    try:
        r = _S.get(
            "https://api.met.no/weatherapi/locationforecast/2.0/compact",
            params={"lat": 40.71, "lon": -74.01},
            timeout=10,
        )
        check("Met Norway API",           r.status_code == 200, f"HTTP {r.status_code}", required=False)
    except Exception as exc:
        check("Met Norway API",           False, str(exc)[:60], required=False)

    # Polymarket Gamma API
    try:
        r = _S.get("https://gamma-api.polymarket.com/markets", params={"limit": 1}, timeout=10)
        check("Polymarket Gamma API",     r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as exc:
        check("Polymarket Gamma API",     False, str(exc)[:60])

    # Polymarket CLOB API
    try:
        r = _S.get("https://clob.polymarket.com/", timeout=10)
        check("Polymarket CLOB API",      r.status_code in (200, 404), f"HTTP {r.status_code}")
    except Exception as exc:
        check("Polymarket CLOB API",      False, str(exc)[:60])

    # Telegram (if configured)
    if tg_token and tg_chat:
        try:
            r = _S.get(
                f"https://api.telegram.org/bot{tg_token}/getMe",
                timeout=10,
            )
            data = r.json()
            ok = data.get("ok", False)
            check("Telegram Bot API",     ok, data.get("result", {}).get("username", "no username"))
        except Exception as exc:
            check("Telegram Bot API",     False, str(exc)[:60], required=False)
    else:
        check("Telegram Bot API",         False, "not configured", required=False)

except ImportError:
    print("  ERROR: requests not installed — run pip install requests")
    sys.exit(1)

# ── Database migration ────────────────────────────────────────────────────────

print("\n=== Testing database migration ===")
try:
    from src.logger import init_db, DB_PATH
    init_db()
    check("SQLite init_db()",             True, f"DB at {DB_PATH}")
except Exception as exc:
    check("SQLite init_db()",             False, str(exc)[:80])

# ── Summary ───────────────────────────────────────────────────────────────────

failures = [r for r in results if FAIL in r[1]]
warnings = [r for r in results if WARN in r[1]]

print(f"\n{'='*60}")
print(f"  Total checks : {len(results)}")
print(f"  Passed       : {len([r for r in results if PASS in r[1]])}")
print(f"  Warnings     : {len(warnings)}")
print(f"  FAILED       : {len(failures)}")

if failures:
    print("\nFailed checks:")
    for name, status, detail in failures:
        print(f"  x {name}: {detail}")
    print("\nPre-flight FAILED — resolve the above before proceeding.")
    sys.exit(1)

print("\nPre-flight PASSED.")
sys.exit(0)
