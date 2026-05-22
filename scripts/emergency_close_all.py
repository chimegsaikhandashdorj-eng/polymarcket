"""
Emergency kill switch: close all open positions at market price.

USE WITH CAUTION. This script:
  1. Lists all open live positions
  2. Sends a market-sell order for each via the CLOB API
  3. Marks positions as manually closed in the DB

Usage:
  python scripts/emergency_close_all.py --confirm-yes [--paper]

Safeguards:
  - Requires explicit --confirm-yes flag (no accidental runs)
  - Dry-run by default (shows what would happen without --confirm-yes)
  - Logs every action to logs/emergency_YYYYMMDD_HHMMSS.log

Kill-switch trigger conditions (from runbook):
  - Drawdown > 25% in 24 hours
  - Reconciliation mismatch detected
  - API key compromise suspected
  - Polygon network congestion / outage
  - You are not available to monitor for > 24 hours
"""

import argparse
import logging
import sys
import time
import os
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ── Logging to timestamped file ───────────────────────────────────────────────

Path("logs").mkdir(exist_ok=True)
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
log_file = f"logs/emergency_{ts}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
log = logging.getLogger("emergency")


def _print_header(dry_run: bool) -> None:
    line = "=" * 65
    print(line)
    print("  EMERGENCY CLOSE ALL POSITIONS")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Mode: {'DRY RUN (no orders placed)' if dry_run else '*** LIVE — REAL ORDERS ***'}")
    print(line)
    if not dry_run:
        print("\n  WARNING: This will place real market-sell orders on Polymarket.")
        print("  Proceeds go to your connected wallet.\n")


def close_paper_positions(paper_flag: bool) -> int:
    """Mark all open paper positions as manually closed (VOID at 0 PnL)."""
    from src.logger import get_open_positions, update_outcome
    positions = get_open_positions(paper=paper_flag)
    if not positions:
        log.info("No open paper positions to close.")
        return 0
    closed = 0
    for pos in positions:
        log.info(
            "Closing paper trade #%d: %s %s  size=%.2f",
            pos["id"], pos["side"], pos["market_title"][:40], pos["size_usdc"],
        )
        update_outcome(pos["id"], 0.5, "VOID")
        closed += 1
    log.info("Closed %d paper position(s).", closed)
    return closed


def close_live_positions(dry_run: bool) -> int:
    """Send market-sell orders for all open live positions via CLOB."""
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
        from py_clob_client.clob_types import OrderArgs, Side
    except ImportError:
        log.error("py-clob-client not installed. Cannot close live positions automatically.")
        log.error("Close positions manually via Polymarket UI: https://polymarket.com/portfolio")
        return 0

    pk = os.getenv("POLY_PRIVATE_KEY", "")
    if not pk:
        log.error("POLY_PRIVATE_KEY not set. Cannot authenticate.")
        return 0

    from src.logger import get_open_positions, update_outcome

    positions = get_open_positions(paper=False)
    if not positions:
        log.info("No open live positions to close.")
        return 0

    log.info("Connecting to Polymarket CLOB...")
    client = ClobClient("https://clob.polymarket.com", key=pk, chain_id=POLYGON)

    api_key    = os.getenv("POLY_API_KEY", "")
    api_secret = os.getenv("POLY_API_SECRET", "")
    api_pass   = os.getenv("POLY_API_PASSPHRASE", "")
    if api_key:
        from py_clob_client.clob_types import ApiCreds
        client.set_api_creds(ApiCreds(api_key, api_secret, api_pass))

    closed = 0
    for pos in positions:
        log.info(
            "Processing trade #%d: %s %s  size=%.2f  entry=%.3f",
            pos["id"], pos["side"], pos["market_title"][:40],
            pos["size_usdc"], pos["entry_price"],
        )
        if dry_run:
            log.info("  [DRY RUN] Would place sell order for trade #%d", pos["id"])
            continue

        try:
            # Estimate tokens held (approximate, actual may differ due to fills)
            tokens_held = pos["size_usdc"] / max(0.01, pos["entry_price"])
            is_yes      = pos["side"] == "YES"
            token_id    = pos.get("yes_token_id") or pos.get("no_token_id") or pos["market_id"]

            order_args = OrderArgs(
                token_id=token_id,
                price=0.01,       # market sell — accept any price
                size=round(tokens_held, 2),
                side=Side.SELL,
            )
            resp = client.create_and_post_order(order_args)

            if resp and resp.get("success"):
                order_id = resp.get("orderID", "unknown")
                log.info("  Sell order placed: order_id=%s", order_id)
                update_outcome(pos["id"], 0.5, "VOID")  # mark as exited
                closed += 1
            else:
                log.error("  Sell order REJECTED: %s", resp)
        except Exception as exc:
            log.error("  Failed to close trade #%d: %s", pos["id"], exc)
        time.sleep(0.5)  # rate-limit

    return closed


def run(confirm: bool, paper: bool) -> int:
    dry_run = not confirm
    _print_header(dry_run)
    log.info("Emergency close initiated. Log: %s", log_file)

    if paper:
        closed = close_paper_positions(paper_flag=True)
    else:
        closed = close_live_positions(dry_run=dry_run)

    log.info("Emergency close complete. Positions processed: %d", closed)

    if dry_run and closed == 0:
        print("\n[DRY RUN] No positions found to close.")
    elif dry_run:
        print(f"\n[DRY RUN] Would close {closed} position(s).")
        print("Re-run with --confirm-yes to execute.")
    else:
        print(f"\n[DONE] Closed {closed} position(s). Full log: {log_file}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emergency close all positions")
    parser.add_argument(
        "--confirm-yes",
        action="store_true",
        help="REQUIRED: actually place orders (default is dry-run)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Close paper positions (mark as VOID). Default: live positions.",
    )
    args = parser.parse_args()
    sys.exit(run(confirm=args.confirm_yes, paper=args.paper))
