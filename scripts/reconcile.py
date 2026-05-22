"""
Phase 3 daily reconciliation: compare bot's internal state vs Polymarket.

For each OPEN trade in the DB, check Polymarket's actual state.
Report mismatches between internal bookkeeping and on-chain reality.

Usage:
  python scripts/reconcile.py [--paper] [--halt-on-mismatch]

Exit codes:
  0 = all positions match
  1 = mismatches found (inspect manually)
  2 = API error — could not complete reconciliation
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

try:
    import requests
except ImportError:
    print("ERROR: requests not installed")
    sys.exit(2)

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "polymarket-bot-reconcile/1.0"
_GAMMA = "https://gamma-api.polymarket.com"


def fetch_market_state(market_id: str) -> dict | None:
    """Fetch current market state from Gamma API."""
    try:
        r = _SESSION.get(
            f"{_GAMMA}/markets",
            params={"conditionIds": market_id},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data and isinstance(data, list):
            return data[0]
    except Exception as exc:
        print(f"  [WARN] Could not fetch market {market_id[:20]}: {exc}")
    return None


def reconcile(paper: bool = True, halt_on_mismatch: bool = False) -> int:
    from src.logger import get_open_positions

    positions = get_open_positions(paper=paper)
    mode = "PAPER" if paper else "LIVE"

    print(f"\n{'='*65}")
    print(f"Reconciliation  |  {mode}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*65}")

    if not positions:
        print("No open positions to reconcile.")
        return 0

    print(f"Checking {len(positions)} open position(s)...\n")

    mismatches = []
    api_errors  = 0

    for pos in positions:
        market_id = pos["market_id"]
        internal_status = "OPEN"
        print(f"  Trade #{pos['id']:>4}  {pos['market_title'][:45]:<45}", end="  ")

        state = fetch_market_state(market_id)
        time.sleep(0.2)  # rate limit

        if state is None:
            print("[WARN] API unavailable")
            api_errors += 1
            continue

        on_chain_resolved  = state.get("resolved") or state.get("closed")
        on_chain_res_price = state.get("resolutionPrice") or state.get("resolution_price")

        if on_chain_resolved and on_chain_res_price is not None:
            # Market settled on-chain but our DB still shows OPEN
            mismatch = {
                "trade_id": pos["id"],
                "market_id": market_id,
                "title": pos["market_title"],
                "internal": internal_status,
                "on_chain": f"RESOLVED @ {on_chain_res_price}",
            }
            mismatches.append(mismatch)
            print(f"[MISMATCH] on-chain RESOLVED={on_chain_res_price} but internal=OPEN")
        elif on_chain_resolved:
            print(f"[WARN]  on-chain closed but no resolution price")
        else:
            # Check for major price deviation (possible data staleness)
            yes_price = state.get("bestAsk") or state.get("lastTradePrice")
            our_entry = pos.get("entry_price")
            if yes_price and our_entry:
                try:
                    current = float(yes_price)
                    deviation = abs(current - float(our_entry))
                    if deviation > 0.30:
                        print(f"[WARN]  large price deviation: entry={our_entry:.3f} now={current:.3f}")
                    else:
                        print(f"[OK]    market still open (last={current:.3f})")
                except (ValueError, TypeError):
                    print("[OK]    market still open")
            else:
                print("[OK]    market still open")

    print(f"\n{'─'*65}")
    print(f"  Checked   : {len(positions)}")
    print(f"  OK        : {len(positions) - len(mismatches) - api_errors}")
    print(f"  Mismatches: {len(mismatches)}")
    print(f"  API errors: {api_errors}")

    if mismatches:
        print(f"\n[ACTION REQUIRED] {len(mismatches)} unresolved mismatch(es):")
        for m in mismatches:
            print(f"  Trade #{m['trade_id']}: {m['title'][:50]}")
            print(f"    Internal: {m['internal']}")
            print(f"    On-chain: {m['on_chain']}")
        print(
            "\nRun the resolver to sync: "
            "python -c \"from src.resolver import resolve_open_trades; resolve_open_trades()\""
        )
        if halt_on_mismatch:
            print("\n[HALT] --halt-on-mismatch set — stopping bot is advised.")
        return 1

    if api_errors > 0:
        print(f"\n[WARN] {api_errors} API call(s) failed. Run again to retry.")
        return 2

    print("\n[CLEAR] All open positions match Polymarket state.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconcile internal state vs Polymarket")
    parser.add_argument("--live",             action="store_true",  help="Reconcile live trades (default: paper)")
    parser.add_argument("--halt-on-mismatch", action="store_true",  help="Exit 1 on any mismatch")
    args = parser.parse_args()
    sys.exit(reconcile(paper=not args.live, halt_on_mismatch=args.halt_on_mismatch))
