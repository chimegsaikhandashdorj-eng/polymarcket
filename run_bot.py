"""
Telegram bot standalone runner — keeps the bot online without the trading loop.

Usage:
    python run_bot.py

This starts ONLY the Telegram command handler + daily scheduler.
Use this when you want bot responsiveness without running scans/trades.
For the full trading bot, use `python main.py run` instead.
"""

import logging
import signal
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.logger import init_db, setup_file_logging
from src.telegram_cmd import TelegramCommander


def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    init_db()
    setup_file_logging()

    config = load_config()
    tg = TelegramCommander(config, scan_callback=None)
    tg.start()

    print("=" * 60)
    print(" Telegram bot is running.")
    print(" Send /help in Telegram to see commands.")
    print(" Press Ctrl+C to stop.")
    print("=" * 60)

    def _shutdown(sig, frame):
        print("\nStopping bot...")
        tg.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
