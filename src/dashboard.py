"""
Rich CLI dashboard — shows live markets, open positions, and PnL summary.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from .logger import (
    get_pnl_summary, get_open_positions,
    get_daily_loss, get_weekly_loss, get_monthly_loss,
)

log = logging.getLogger(__name__)
console = Console()


def _ev_color(ev: float) -> str:
    if ev >= 0.15:
        return "bold green"
    if ev >= 0.05:
        return "green"
    if ev >= 0:
        return "yellow"
    return "red"


def _conf_color(conf: float) -> str:
    if conf >= 0.80:
        return "green"
    if conf >= 0.60:
        return "yellow"
    return "red"


def show_opportunities(opportunities: list) -> None:
    """Print a ranked table of current trading opportunities."""
    if not opportunities:
        console.print(Panel("[yellow]No +EV opportunities found this scan.[/yellow]", title="Opportunities"))
        return

    table = Table(
        title="Trading Opportunities",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", width=4, justify="right")
    table.add_column("Market", min_width=35)
    table.add_column("Side", width=6, justify="center")
    table.add_column("Our P", width=7, justify="right")
    table.add_column("Mkt P", width=7, justify="right")
    table.add_column("EV", width=8, justify="right")
    table.add_column("Conf", width=6, justify="right")
    table.add_column("Volume", width=10, justify="right")

    for i, opp in enumerate(opportunities, 1):
        ev_raw = getattr(opp, "ev_raw", 0.0)
        has_opp_cost = ev_raw and abs(ev_raw - opp.ev) > 0.0001
        ev_str = (
            f"{opp.ev:+.3f} (raw {ev_raw:+.3f})" if has_opp_cost
            else f"{opp.ev:+.3f}"
        )
        adv_marker = " [red]⚠[/red]" if getattr(opp, "adversarial", False) else ""
        table.add_row(
            str(i),
            opp.market_title[:48] + adv_marker,
            f"[bold]{opp.side}[/bold]",
            f"{opp.our_prob:.3f}",
            f"{opp.market_price:.3f}",
            Text(ev_str, style=_ev_color(opp.ev)),
            Text(f"{opp.confidence:.2f}", style=_conf_color(opp.confidence)),
            "—",
        )

    console.print(table)


def show_open_positions(paper: bool = True) -> None:
    """Print a table of currently open positions."""
    positions = get_open_positions(paper=paper)
    mode = "[yellow]PAPER[/yellow]" if paper else "[red bold]LIVE[/red bold]"

    if not positions:
        console.print(Panel(f"No open positions ({mode})", title="Open Positions"))
        return

    table = Table(
        title=f"Open Positions ({mode})",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("ID", width=5)
    table.add_column("Market", min_width=35)
    table.add_column("Side", width=6, justify="center")
    table.add_column("Size $", width=8, justify="right")
    table.add_column("Entry", width=7, justify="right")
    table.add_column("EV", width=7, justify="right")
    table.add_column("Opened", width=20)

    for p in positions:
        table.add_row(
            str(p["id"]),
            p["market_title"][:50],
            p["side"],
            f"{p['size_usdc']:.2f}",
            f"{p['entry_price']:.3f}",
            Text(f"{p['ev']:+.3f}", style=_ev_color(p["ev"])),
            p["timestamp"][:19],
        )

    console.print(table)


def show_pnl_summary(paper: bool = True) -> None:
    """Print PnL summary panels for paper and/or live trades."""
    summary = get_pnl_summary(paper=paper)
    mode = "PAPER" if paper else "LIVE"

    win_rate = summary.get("win_rate", 0)
    total_pnl = summary.get("total_pnl_usdc", 0)
    pnl_color = "green" if total_pnl >= 0 else "red"
    pnl_str = f"[{pnl_color}]{total_pnl:+.2f} USDC[/{pnl_color}]"

    lines = [
        f"Mode:        {mode}",
        f"Total trades: {summary.get('total_trades', 0)}",
        f"Wins:         {summary.get('wins', 0)}",
        f"Losses:       {summary.get('losses', 0)}",
        f"Win rate:     {win_rate:.1%}",
        f"Total PnL:    {pnl_str}",
    ]

    console.print(Panel("\n".join(lines), title=f"PnL Summary ({mode})", border_style="cyan"))


def show_scan_header(
    city_count: int, market_count: int, adversarial_flags: int = 0
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    adv_line = (
        f"\n[red bold]Adversarial flags:  {adversarial_flags} markets blocked[/red bold]"
        if adversarial_flags > 0 else ""
    )
    console.print(
        Panel(
            f"[bold]Polymarket Weather Bot[/bold]\n"
            f"Scan time:  {now}\n"
            f"Cities:     {city_count}\n"
            f"Markets:    {market_count} weather markets found"
            f"{adv_line}",
            border_style="blue",
        )
    )


def show_loss_limits_panel(paper: bool = True, config: Optional[dict] = None) -> None:
    """Display daily/weekly/monthly loss consumption vs. configured limits."""
    config = config or {}
    ll = config.get("loss_limits", {})
    t  = config.get("trading", {})

    daily_limit   = ll.get("daily_usdc",   t.get("daily_loss_limit_usdc", 200))
    weekly_limit  = ll.get("weekly_usdc",  500)
    monthly_limit = ll.get("monthly_usdc", 1000)

    daily_loss   = get_daily_loss(paper=paper)
    weekly_loss  = get_weekly_loss(paper=paper)
    monthly_loss = get_monthly_loss(paper=paper)

    def _bar(loss: float, limit: float) -> str:
        pct = loss / limit if limit > 0 else 0.0
        color = (
            "bold red"  if pct >= 1.0  else
            "red"       if pct >= 0.75 else
            "yellow"    if pct >= 0.50 else
            "green"
        )
        label = " [HALTED]" if pct >= 1.0 else ""
        return f"[{color}]{loss:.2f}/{limit:.0f} USDC  ({pct:.0%}){label}[/{color}]"

    lines = [
        f"Daily:   {_bar(daily_loss,   daily_limit)}",
        f"Weekly:  {_bar(weekly_loss,  weekly_limit)}",
        f"Monthly: {_bar(monthly_loss, monthly_limit)}",
    ]
    console.print(Panel("\n".join(lines), title="Loss Limits", border_style="cyan"))


def show_model_status() -> None:
    """Print calibration method, meta model state, and drift warning if any."""
    try:
        from .model_tracker import get_tracker
        tracker = get_tracker()
        info = tracker.get_calibration_info()

        method  = info.get("method", "none")
        fitted  = info.get("fitted", False)
        m_color = "green" if fitted else "yellow"
        meta_ok = info.get("meta", False)

        lines = [
            f"Calibration:  [{m_color}]{method.upper()}[/{m_color}]"
            f"  {'✓ fitted' if fitted else '⏳ accumulating data'}",
        ]
        if method == "platt" and fitted:
            lines.append(f"  Platt A={info['A']:.4f}  B={info['B']:.4f}")
        elif method == "isotonic" and fitted:
            lines.append(f"  Isotonic breakpoints: {info['iso_pts']}")

        meta_color = "green" if meta_ok else "yellow"
        lines.append(
            f"Meta model:   [{meta_color}]"
            f"{'✓ active' if meta_ok else '⏳ accumulating data'}[/{meta_color}]"
        )

        drift = tracker.detect_drift()
        if drift:
            lines.append(f"[bold red]⚠ DRIFT: {drift}[/bold red]")

        console.print(Panel("\n".join(lines), title="Model Status", border_style="magenta"))
    except Exception as exc:
        log.debug("show_model_status failed: %s", exc)


def show_var_panel(var: dict) -> None:
    """Display portfolio VaR snapshot computed by RiskManager.estimate_portfolio_var."""
    if not var or var.get("total_exposure", 0) == 0:
        return

    exp     = var.get("total_exposure", 0)
    el      = var.get("expected_loss", 0)
    v95     = var.get("var_95", 0)
    v99     = var.get("var_99", 0)
    pct     = var.get("pct_of_bankroll", 0)
    v_color = "red" if v95 > exp * 2 else "yellow"

    lines = [
        f"Total exposure:   {exp:.2f} USDC  ({pct:.1f}% of bankroll)",
        f"Expected loss:    {el:.2f} USDC",
        f"[{v_color}]VaR 95%:          {v95:.2f} USDC[/{v_color}]",
        f"[{v_color}]VaR 99%:          {v99:.2f} USDC[/{v_color}]",
    ]
    console.print(Panel("\n".join(lines), title="Portfolio VaR", border_style="yellow"))


def show_backtest_results(results: dict) -> None:
    """Print backtest summary as a styled panel."""
    if not results or "error" in results:
        console.print(Panel(str(results.get("error", "No results")), title="Backtest", border_style="red"))
        return

    roi_color = "green" if results["roi_pct"] >= 0 else "red"
    lines = [
        f"Total trades:    {results['total_trades']}",
        f"Win rate:        {results['win_rate']:.1%}",
        f"Total PnL:       [{roi_color}]{results['total_pnl_usdc']:+.2f} USDC[/{roi_color}]",
        f"ROI:             [{roi_color}]{results['roi_pct']:+.2f}%[/{roi_color}]",
        f"Sharpe ratio:    {results['sharpe_ratio']:.3f}",
        f"Max drawdown:    {results['max_drawdown_pct']:.1f}%",
        f"Final bankroll:  {results['final_bankroll']:.2f} USDC",
        f"Avg EV:          {results['avg_ev']:+.4f}",
        f"Avg confidence:  {results['avg_confidence']:.2f}",
    ]
    console.print(Panel("\n".join(lines), title="Backtest Results", border_style="green"))
