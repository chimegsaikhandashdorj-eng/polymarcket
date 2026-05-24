"""
Polymarket Weather Trading Bot — entry point.

Usage:
  python main.py run         # Start the live bot loop (paper or live per config)
  python main.py scan        # One-shot market scan (no trades placed)
  python main.py backtest    # Run historical simulation
  python main.py dashboard   # Show current positions and PnL
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import click
import yaml
from dotenv import load_dotenv

from src import parse_utc_isoformat  # canonical UTC-safe ISO parser

load_dotenv()

log = logging.getLogger(__name__)


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Parallel weather fetching ──────────────────────────────────────────────────

_weather_lock = threading.Lock()
_last_scan_max_ev = [0.0]  # mutable container to track max EV for dynamic interval

def _fetch_all_city_weather(weather_fetcher, cities: List[dict], target_dt: datetime) -> Dict[str, dict]:
    """
    Fetch ensemble weather for all cities in parallel.
    Returns {city_name_lower: weather_dict}.
    """
    city_weather: Dict[str, dict] = {}

    def _fetch(city: dict):
        weather = weather_fetcher.fetch(city["lat"], city["lon"], target_dt)
        return city["name"], weather

    with ThreadPoolExecutor(max_workers=min(len(cities), 6)) as pool:
        futures = {pool.submit(_fetch, city): city for city in cities}
        for future in as_completed(futures):
            try:
                city_name, weather = future.result()
                with _weather_lock:
                    city_weather[city_name.lower()] = {**weather, "fetched_at": time.time()}
                log.info(
                    "Weather %s: precip=%.0f%%  temp=%.1f°C  conf=%.2f  [%s]",
                    city_name,
                    (weather.get("precip_prob") or 0) * 100,
                    weather.get("temp_c") or 0,
                    weather.get("confidence") or 0,
                    ",".join(weather.get("sources_used") or []),
                )
            except RuntimeError as exc:
                city = futures[future]
                log.error("Weather fetch failed for %s: %s", city["name"], exc)

    return city_weather


# ── Shared scan logic ──────────────────────────────────────────────────────────

def _run_scan(config: dict, executor=None) -> list:
    """
    Fetch weather + markets, evaluate opportunities, optionally execute trades.
    Returns ranked list of Opportunity objects found.
    """
    from src.data_fetcher import WeatherEnsemble
    from src.market_scanner import MarketScanner
    from src.strategy import ProbabilityEngine
    from src.dashboard import show_scan_header, show_opportunities

    weather_fetcher = WeatherEnsemble(config)
    scanner = MarketScanner(config)
    engine = ProbabilityEngine(config)

    cities = config.get("cities", [])
    target_dt = datetime.now(timezone.utc)

    # 1. Fetch weather for all cities in parallel
    city_weather = _fetch_all_city_weather(weather_fetcher, cities, target_dt)

    if not city_weather:
        log.error("No weather data available — aborting scan")
        return []

    # 2. Scan Polymarket for weather markets
    raw_markets = scanner.fetch_weather_markets()
    tradeable = scanner.filter_tradeable(raw_markets)

    from src.market_scanner import get_adversarial_detector
    adv_count = get_adversarial_detector(config).get_flag_count()
    show_scan_header(len(cities), len(raw_markets), adversarial_flags=adv_count)

    # 2.5. Crypto data fetch (for crypto markets)
    from src.crypto_fetcher import CryptoEnsemble
    from src.market_scanner import CRYPTO_ABOVE, CRYPTO_BELOW
    from src.crypto_signals import (
        compute_composite_signal,
        check_crypto_correlation,
        detect_crypto_regime,
    )
    crypto_fetcher = CryptoEnsemble(config)

    # 3. Evaluate each market
    opportunities = []  # list of (Opportunity, age_seconds)

    for market in tradeable:
        metric = market.get("metric", "")

        # ── Crypto markets: fetch price data instead of weather ──
        if metric in (CRYPTO_ABOVE, CRYPTO_BELOW):
            crypto_cfg = config.get("crypto", {})
            if not crypto_cfg.get("enabled", False):
                continue

            # Force paper mode check for crypto
            paper_until = crypto_cfg.get("paper_only_until", "")
            if paper_until:
                try:
                    if parse_utc_isoformat(paper_until).date() > datetime.now(timezone.utc).date():
                        if not config["trading"].get("paper_mode", True):
                            log.debug("Crypto forced paper mode until %s", paper_until)
                            continue  # skip crypto in live mode during test period
                except (ValueError, TypeError):
                    pass

            crypto_asset = market.get("crypto_asset")
            threshold = market.get("threshold")
            if not crypto_asset or not threshold:
                continue

            # Check if asset is in allowed list
            allowed_assets = crypto_cfg.get("assets", [])
            if allowed_assets and crypto_asset not in allowed_assets:
                continue

            # Crypto-specific min hours to expiry
            min_hours = crypto_cfg.get("min_hours_to_expiry", 12)

            direction = "above" if metric == CRYPTO_ABOVE else "below"

            # Calculate hours to expiry
            hours_to_expiry = 72.0
            expiry_str = market.get("expiry_dt") or market.get("target_dt")
            if expiry_str:
                try:
                    exp_dt = parse_utc_isoformat(expiry_str)
                    hours_to_expiry = max(0.0, (exp_dt - target_dt).total_seconds() / 3600)
                except (ValueError, TypeError):
                    pass

            if hours_to_expiry < min_hours:
                log.debug("Crypto market too close to expiry: %.1fh < %dh", hours_to_expiry, min_hours)
                continue

            # Fetch full crypto data (price, hourly, volumes, technicals)
            crypto_data = crypto_fetcher.fetch(crypto_asset, hours_ahead=int(hours_to_expiry) + 24)
            if not crypto_data:
                continue

            # Check volatility cap before any computation
            daily_vol = crypto_data.get("volatility_daily", 0)
            max_vol = crypto_cfg.get("max_volatility_daily", 0.12)
            if daily_vol > max_vol:
                log.info(
                    "Crypto %s vol %.1f%% exceeds cap %.1f%% -- skipping",
                    crypto_asset, daily_vol * 100, max_vol * 100,
                )
                continue

            # Spread check: skip if bid-ask spread too wide
            spread = market.get("spread")
            if spread is None or spread == 0:
                try:
                    min_bid = float(market.get("min_bid") or market.get("best_bid") or 0)
                    max_ask = float(market.get("max_ask") or market.get("best_ask") or 0)
                    if min_bid > 0 and max_ask > 0:
                        mid = (min_bid + max_ask) / 2.0
                        spread = (max_ask - min_bid) / mid if mid > 0 else 0.0
                    else:
                        spread = 0.0
                except (ValueError, TypeError):
                    spread = 0.0
            max_spread = crypto_cfg.get("max_spread", 0.04)
            if spread > max_spread:
                log.debug("Crypto %s spread %.2f%% > max %.2f%% -- skipping",
                          crypto_asset, spread * 100, max_spread * 100)
                continue

            # Option Pricing Volatility Safeguard: Verify underpriced YES shares
            from src.crypto_signals import confirm_crypto_option_edge
            yes_mkt_price = market.get("best_ask") or market.get("yes_price") or 0.5
            option_check = confirm_crypto_option_edge(
                current_price=crypto_data.get("current_price", 0.0),
                threshold=threshold,
                direction=direction,
                hours_to_expiry=hours_to_expiry,
                daily_volatility=daily_vol,
                yes_market_price=yes_mkt_price,
                min_edge_threshold=crypto_cfg.get("min_ev_threshold", 0.12),
            )
            if not option_check["has_underpriced_edge"]:
                log.info(
                    "Crypto %s Option Model Safeguard: No underpriced option edge confirmed (Theo=%.3f, Mkt=%.3f, Edge=%.3f, IV=%.1f%%, HV=%.1f%%) -- skipping",
                    crypto_asset, option_check["theoretical_yes_price"], yes_mkt_price,
                    option_check["price_edge"], option_check["implied_vol"] * 100, option_check["historical_vol"] * 100
                )
                continue

            # Regime detection -- skip CRASH, warn on VOLATILE
            hourly_prices = crypto_data.get("hourly_prices", [])
            regime = detect_crypto_regime(hourly_prices)
            if regime == "CRASH":
                log.info("Crypto %s regime=CRASH -- skipping new entries", crypto_asset)
                continue
            if regime == "VOLATILE":
                log.info("Crypto %s regime=VOLATILE -- extra caution applied", crypto_asset)

            # Correlation guard -- block if correlated position already open
            from src.logger import get_open_positions as _get_crypto_pos
            open_positions = _get_crypto_pos(paper=config["trading"].get("paper_mode", True))
            # Filter to crypto-only positions (have crypto_asset field or CRYPTO_ metric)
            crypto_open = [
                p for p in open_positions
                if "crypto" in (p.get("market_title") or "").lower()
                or p.get("metric", "").startswith("CRYPTO_")
            ]
            corr_block = check_crypto_correlation(crypto_asset, direction, crypto_open)
            if corr_block:
                log.info("Crypto correlation guard: %s", corr_block)
                continue

            # Base probability from log-normal model
            prob_result = crypto_fetcher.get_probability(
                crypto_asset, threshold, direction, hours_to_expiry
            )
            if prob_result is None:
                continue
            prob, conf = prob_result

            # Composite signal: multi-timeframe + fear&greed + regime + volume
            volumes = crypto_data.get("volumes", [])
            composite = compute_composite_signal(
                prob_above=prob if direction == "above" else (1.0 - prob),
                direction=direction,
                hourly_prices=hourly_prices,
                volumes=volumes if volumes else None,
            )
            # Apply adjusted probability from composite signal
            adjusted_prob = composite["adjusted_prob"]
            if direction == "below":
                adjusted_prob = 1.0 - adjusted_prob
            # Blend: 70% composite, 30% raw (safety margin)
            prob = 0.7 * adjusted_prob + 0.3 * prob
            # Apply confidence multiplier
            conf *= composite["confidence_multiplier"]

            # Crypto-specific confidence floor
            crypto_min_conf = crypto_cfg.get("min_confidence", 0.75)
            if conf < crypto_min_conf:
                log.debug("Crypto confidence %.2f below minimum %.2f", conf, crypto_min_conf)
                continue

            # Build a "weather-like" dict for the strategy engine
            crypto_weather = {
                "crypto_prob": prob,
                "confidence": conf,
                "regime": regime,
                "sources_used": crypto_data.get("sources_used", []),
                "fetched_at": time.time(),
                "composite_signals": composite.get("signals", []),
            }
            opp = engine.evaluate(market, crypto_weather)
            if opp:
                if opp.ev < crypto_cfg.get("min_ev_threshold", 0.12):
                    log.debug("Crypto EV %.3f below crypto threshold", opp.ev)
                    continue
                opportunities.append((opp, 0))
                log.info(
                    "Crypto opportunity: %s %s $%.0f | prob=%.2f conf=%.2f regime=%s signals=%s",
                    crypto_asset, direction, threshold, prob, conf,
                    regime, ",".join(composite.get("signals", [])),
                )
            continue

        # ── Weather markets: use ensemble weather data ──
        city_key = (market.get("city") or "").lower()
        weather = city_weather.get(city_key)
        if not weather:
            continue
        age = time.time() - weather.get("fetched_at", time.time())
        opp = engine.evaluate(market, weather)
        if opp:
            opportunities.append((opp, age))

    # Rank by composite score; preserve age for execution
    ranked_opps = engine.rank([o for o, _ in opportunities])
    show_opportunities(ranked_opps)

    # 4. Execute ranked opportunities if executor provided (with edge confirmation)
    if executor:
        from src.edge_confirm import get_edge_confirmation
        edge_gate = get_edge_confirmation(config)

        age_map = {opp.market_id: age for opp, age in opportunities}
        for opp in ranked_opps:
            # Edge confirmation: only trade if seen in 2+ consecutive scans
            confirmed = edge_gate.check(
                market_id=opp.market_id,
                side=opp.side,
                ev=opp.ev,
                our_prob=opp.our_prob,
                market_price=opp.market_price,
            )
            if not confirmed:
                log.debug("Edge not yet confirmed for %s — waiting next scan", opp.market_title[:40])
                continue
            executor.execute(opp, weather_age_seconds=age_map.get(opp.market_id, 0))

        pending = edge_gate.get_pending_count()
        if pending > 0:
            log.info("%d opportunities awaiting edge confirmation", pending)

    return ranked_opps


# ── Early exit helper ──────────────────────────────────────────────────────────

def _check_early_exits(paper: bool, config: dict) -> None:
    """
    Check open positions for profit-taking or stop-loss opportunities.
    In paper mode: auto-resolves the position at current market price.
    In live mode: sends Telegram alert (manual exit required via CLOB sell).
    """
    from src.logger import get_open_positions
    from src.edge_confirm import check_early_exits
    from src.market_scanner import CLOB_API, _safe_get

    positions = get_open_positions(paper=paper)
    if not positions:
        return

    # Fetch current prices for all open position markets
    current_prices = {}
    for pos in positions:
        market_id = pos.get("market_id", "")
        if not market_id or market_id in current_prices:
            continue
        data = _safe_get(f"{CLOB_API}/markets/{market_id}")
        # CLOB market endpoint returns an object; narrow before subscripting.
        if not isinstance(data, dict):
            continue
        tokens = data.get("tokens") or []
        for token in tokens:
            if isinstance(token, dict) and token.get("outcome") == "Yes":
                try:
                    current_prices[market_id] = float(token.get("price", 0))
                except (ValueError, TypeError):
                    pass

    if not current_prices:
        return

    signals = check_early_exits(positions, current_prices)

    for signal in signals:
        if paper:
            # Paper mode: auto-exit (simulate selling at current price)
            from src.logger import update_outcome
            if signal.unrealized_pnl_pct > 0:
                update_outcome(signal.trade_id, signal.current_price, "WIN")
                log.info(
                    "EARLY EXIT (paper): trade #%d %s — profit-taking at %.0f%% gain",
                    signal.trade_id, signal.market_title, signal.unrealized_pnl_pct * 100,
                )
            else:
                update_outcome(signal.trade_id, signal.current_price, "LOSS")
                log.info(
                    "EARLY EXIT (paper): trade #%d %s — stop-loss at %.0f%% loss",
                    signal.trade_id, signal.market_title, signal.unrealized_pnl_pct * 100,
                )
        else:
            # Live mode: alert user via Telegram
            from src.notifier import _send as _tg_send
            emoji = "💰" if signal.unrealized_pnl_pct > 0 else "🚨"
            _tg_send(
                f"{emoji} <b>Early Exit Signal</b>\n"
                f"Trade #{signal.trade_id}: {signal.market_title}\n"
                f"{signal.reason}\n"
                f"Entry: {signal.entry_price:.3f} → Now: {signal.current_price:.3f}"
            )


# ── CLI commands ───────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Polymarket Weather Trading Bot"""


@cli.command()
def run():
    """Start the bot loop — scans markets and executes trades on a schedule."""
    config = load_config()

    from src.logger import init_db, get_pnl_summary, setup_file_logging
    from src.executor import TradeExecutor
    from src.resolver import resolve_open_trades
    from src.dashboard import (
        show_pnl_summary, show_open_positions, show_model_status,
        show_var_panel, show_loss_limits_panel,
    )
    from src.notifier import notify_error

    init_db()
    setup_file_logging()

    paper = config["trading"].get("paper_mode", True)
    interval = config["trading"].get("poll_interval_seconds", 900)
    executor = TradeExecutor(config)

    # Telegram command handler — lets user control bot via /status, /pnl, /scan, etc.
    from src.telegram_cmd import TelegramCommander
    tg_cmd = TelegramCommander(
        config,
        scan_callback=lambda: _run_scan(config, executor=executor),
    )
    tg_cmd.start()

    click.echo(f"Bot started — {'PAPER' if paper else 'LIVE'} mode")
    click.echo(f"Polling every {interval}s ({interval//60} min)")
    click.echo("Press Ctrl+C to stop.\n")

    _last_summary_date: list = [None]  # mutable container for nonlocal-style mutation

    def job():
        from src.logger import purge_stale_weather_cache
        from src.notifier import notify_daily_summary

        click.echo(f"\n{'='*60}")
        click.echo(f"Scan at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

        # Send daily summary + run daily maintenance at the start of each UTC day
        today = datetime.now(timezone.utc).date().isoformat()
        is_new_day = bool(_last_summary_date[0]) and _last_summary_date[0] != today
        if is_new_day:
            summary = get_pnl_summary(paper=paper)
            notify_daily_summary(
                total_pnl=summary.get("total_pnl_usdc", 0),
                wins=summary.get("wins", 0),
                losses=summary.get("losses", 0),
            )
            purge_stale_weather_cache(max_age_hours=24)
        _last_summary_date[0] = today

        # Resolve any settled trades first, then refresh bankroll + caches
        settled = resolve_open_trades(paper=paper)
        if settled:
            executor.refresh_bankroll()
            executor.risk.invalidate_caches()

        # Early exit check — detect profit-taking or stop-loss opportunities
        _check_early_exits(paper, config)

        # Refit models only when new outcomes arrive — gradient descent is expensive
        from src.model_tracker import get_tracker
        tracker = get_tracker()
        if settled:
            tracker.fit_calibration()
            tracker.fit_meta_model()

            # Self-learning: analyze errors and auto-tune parameters
            from src.learner import get_learner
            learner = get_learner(config)
            learning_msg = learner.learn()
            if learning_msg:
                from src.notifier import _send as _tg_send
                _tg_send(learning_msg)

        # Purge stale model data once per day (expensive DB DELETE, not every scan)
        if is_new_day:
            tracker.purge_stale_data()

        drift_warning = tracker.detect_drift()
        if drift_warning:
            notify_error(f"[DRIFT ALERT] {drift_warning}")

        if tg_cmd.is_paused:
            click.echo("Trading paused via Telegram — skipping new entries")
            opps = _run_scan(config, executor=None)
        else:
            opps = _run_scan(config, executor=executor)

        # Track maximum EV from this scan for dynamic interval
        _last_scan_max_ev[0] = max([opp.ev for opp in opps], default=0.0)

        tg_cmd.set_last_scan(datetime.now(timezone.utc))

        # Portfolio VaR snapshot
        from src.logger import get_open_positions as _get_open
        _positions = _get_open(paper=paper)
        _var = executor.risk.estimate_portfolio_var(_positions, executor.get_bankroll())
        show_var_panel(_var)

        show_loss_limits_panel(paper=paper, config=config)
        show_open_positions(paper=paper)
        show_pnl_summary(paper=paper)
        show_model_status()

    job()  # Run immediately on start

    # Dynamic interval: default 15min, 5min if any open position expires < 6h,
    # or 5min if a high-EV opportunity exists (High-activity mode)
    def _dynamic_interval() -> int:
        from src.logger import get_open_positions as _gop
        positions = _gop(paper=paper)
        if not positions:
            max_ev = _last_scan_max_ev[0]
            min_high_ev = max(0.15, config["trading"].get("ev_threshold", 0.08) * 1.5)
            if max_ev >= min_high_ev:
                log.info(
                    "High-activity mode triggered! High-EV opportunity found (EV = %.2f%% >= %.2f%%). Polling interval shortened to 5 min.",
                    max_ev * 100, min_high_ev * 100
                )
                return 300
            return interval
        for pos in positions:
            expiry_str = pos.get("expiry") or ""
            if expiry_str:
                try:
                    exp_dt = parse_utc_isoformat(expiry_str)
                    hours_left = (exp_dt - datetime.now(timezone.utc)).total_seconds() / 3600
                    if hours_left < 6:
                        log.info("Near-expiry open position found (<6h left). Polling interval shortened to 5 min.")
                        return max(300, interval // 3)  # 5 min minimum
                except (ValueError, TypeError):
                    pass
        # Check high-EV opportunity if open positions exist but none are near-expiry
        max_ev = _last_scan_max_ev[0]
        min_high_ev = max(0.15, config["trading"].get("ev_threshold", 0.08) * 1.5)
        if max_ev >= min_high_ev:
            log.info(
                "High-activity mode triggered! High-EV opportunity found (EV = %.2f%% >= %.2f%%). Polling interval shortened to 5 min.",
                max_ev * 100, min_high_ev * 100
            )
            return 300
        return interval

    _next_run = time.time() + interval

    try:
        while True:
            now = time.time()
            if now >= _next_run:
                job()
                _next_run = now + _dynamic_interval()
            time.sleep(10)
    except KeyboardInterrupt:
        tg_cmd.stop()
        click.echo("\nBot stopped.")
    except Exception as exc:
        tg_cmd.stop()
        notify_error(f"Bot crashed: {type(exc).__name__}: {exc}")
        log.exception("Bot crashed with unhandled exception")
        raise


@cli.command()
def scan():
    """One-shot market scan — shows opportunities without placing trades."""
    config = load_config()

    from src.logger import init_db
    init_db()

    _run_scan(config, executor=None)


@cli.command()
@click.option("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
@click.option("--end", default="2024-12-31", help="End date YYYY-MM-DD")
@click.option("--bankroll", default=500.0, type=float, help="Starting bankroll in USDC")
@click.option("--engine", default="custom", type=click.Choice(["custom", "backtrader"]),
              help="Backtesting engine: custom (weather markets) or backtrader (crypto OHLCV)")
@click.option("--asset", default="bitcoin", help="Crypto asset for backtrader engine")
def backtest(start: str, end: str, bankroll: float, engine: str, asset: str):
    """Run historical backtesting simulation."""
    config = load_config()

    from src.logger import init_db
    from src.dashboard import show_backtest_results

    init_db()

    if engine == "backtrader":
        from src.backtester import BacktraderEngine
        click.echo(f"Running Backtrader backtest: {asset} {start} -> {end}  bankroll={bankroll:.0f} USDC")
        bt_engine = BacktraderEngine(config)
        results = bt_engine.run(asset=asset, start_date=start, end_date=end, initial_bankroll=bankroll)
        show_backtest_results(results)
    else:
        from src.backtester import Backtester
        click.echo(f"Running backtest: {start} -> {end}  bankroll={bankroll:.0f} USDC")
        bt = Backtester(config)
        results = bt.run(start_date=start, end_date=end, initial_bankroll=bankroll)
        show_backtest_results(results)


@cli.command("walk-forward")
@click.option("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
@click.option("--end", default="2024-12-31", help="End date YYYY-MM-DD")
@click.option("--train-months", default=3, type=int, help="Training window months")
@click.option("--test-months", default=1, type=int, help="Test window months")
@click.option("--bankroll", default=500.0, type=float, help="Starting bankroll in USDC")
@click.option("--monte-carlo", is_flag=True, default=False, help="Run Monte Carlo after WF")
def walk_forward(start: str, end: str, train_months: int, test_months: int,
                 bankroll: float, monte_carlo: bool):
    """Walk-forward validation + optional Monte Carlo simulation."""
    config = load_config()
    from src.logger import init_db
    from src.backtester import Backtester
    from src.dashboard import show_backtest_results

    init_db()

    bt = Backtester(config)
    click.echo(f"Walk-forward: {start} -> {end}  train={train_months}m  test={test_months}m")
    periods = bt.walk_forward(start, end, train_months, test_months, bankroll)

    if not periods:
        click.echo("No out-of-sample periods produced.")
        return

    for i, p in enumerate(periods, 1):
        click.echo(f"\n--- Period {i}: {p.get('test_window', '')} ---")
        show_backtest_results(p)

    # Aggregate across all periods
    all_results: list = []
    for p in periods:
        all_results += [{"pnl_usdc": p["total_pnl_usdc"] / max(1, p["total_trades"])}] * p["total_trades"]

    agg_roi   = sum(p["roi_pct"] for p in periods) / len(periods)
    agg_sharpe = sum(p["sharpe_ratio"] for p in periods) / len(periods)
    agg_dd    = max(p["max_drawdown_pct"] for p in periods)
    click.echo(f"\n=== Walk-Forward Summary ({len(periods)} periods) ===")
    click.echo(f"  Avg ROI:       {agg_roi:+.2f}%")
    click.echo(f"  Avg Sharpe:    {agg_sharpe:.3f}")
    click.echo(f"  Worst Drawdown:{agg_dd:.1f}%")

    if monte_carlo and all_results:
        mc = bt.monte_carlo(all_results, bankroll)
        click.echo(f"\n=== Monte Carlo ({mc['n_simulations']} simulations) ===")
        click.echo(f"  Median bankroll: ${mc['median_bankroll']:.2f}")
        click.echo(f"  5th percentile:  ${mc['p5_bankroll']:.2f}")
        click.echo(f"  95th percentile: ${mc['p95_bankroll']:.2f}")
        click.echo(f"  Ruin probability (<20% of initial): {mc['ruin_probability']:.1%}")


@cli.command()
@click.option("--paper/--live", default=True, help="Show paper or live positions")
def dashboard(paper: bool):
    """Show current open positions and PnL summary."""
    from src.logger import init_db
    from src.dashboard import show_open_positions, show_pnl_summary

    init_db()

    show_open_positions(paper=paper)
    show_pnl_summary(paper=paper)


if __name__ == "__main__":
    cli()
