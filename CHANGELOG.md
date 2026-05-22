# Changelog

All notable changes to this project are documented here.
Format: [version] — date — summary of changes.

---

## [2.2.0] — 2026-05-07  (Backtest Fixes)

### Bug Fixes

- **backtester.py** (`fetch_historical_markets`) — Keyword filter used plain substring
  matching, so `"ukraine"` matched keyword `"rain"`, `"rainbow"` matched `"rain"`, etc.
  Fixed by compiling each keyword as a word-boundary regex (`\brain\b`). False-positive
  rate dropped from ~19/20 to 0/11.

- **backtester.py** (`Backtester.run`) — Kelly formula always used `opp.our_prob` (the
  YES probability) regardless of side. For NO trades (`our_prob ≈ 0.02`) this produced
  negative Kelly, so every NO-side opportunity was silently discarded. Fixed by computing
  `prob_win = our_prob if side == "YES" else 1 - our_prob` before the Kelly calculation.

- **market_scanner.py** (`_CITY_ALIASES`) — Added `"la"` as an alias for Los Angeles.
  "la" appears as a substring in common words (`landfall`, `atlantic`, `relay`) causing
  false city matches on hurricane markets. Removed all ambiguous short aliases; kept only
  unambiguous multi-char aliases (`nyc`, `new york city`, `los angeles ca`, etc.).

### Config

- `config.yaml` — Added `markets.backtest_min_liquidity_usdc: 500`. Live trading keeps
  the 5000 USDC liquidity floor; backtesting uses 500 USDC to include lower-volume
  historical markets (most NYC snowfall bucket markets trade 1,700–5,800 USDC).

### Backtest Results (2024)

Confirmed working: `python main.py backtest --start 2024-01-01 --end 2024-12-31 --bankroll 1000`

```
Total trades:    5  (NYC snowfall bucket markets, Feb 2024 storm event)
Win rate:        100.0%
Total PnL:       +236.07 USDC
ROI:             +23.61%
Sharpe ratio:    276.9  (high due to zero variance — single correlated event)
Max drawdown:    0.0%
Final bankroll:  1236.07 USDC
Avg EV:          +0.9289
Avg confidence:  0.56
```

Data limitation: Polymarket had very few city-specific weather markets in 2024.
Summer/fall markets are hurricane-regional (no city match). Only 5 NYC snowfall
bucket markets from Feb 2024 matched the model. Walk-forward with this data
produces no meaningful out-of-sample periods.

---

## [2.1.0] — 2026-05-07  (Production Readiness — Phase 0 / Phase 1)

### Bug Fixes

- **resolver.py** — Fixed silent `NO` resolution bug: `resolutionPrice=0.0` was
  evaluated as falsy via `x or y`, silently returning `None` instead of `0.0`.
  All `NO`-wins were misclassified as unresolved. Fixed with explicit `if raw is None:`.

- **market_scanner.py** — Replaced Unicode `Δ`, `→`, `×` in log messages with
  ASCII equivalents to prevent `UnicodeEncodeError` on Windows cp1252 consoles.

- **risk_manager.py** (`_compute_peak_bankroll`, `_consecutive_losses`) — Explicit
  `conn.close()` in `finally` blocks eliminates `ResourceWarning: unclosed database`.

- **logger.py** (`_connect`) — Now a `@contextmanager` that always closes the
  connection, commits on success, rolls back on exception.

### New: Structured File Logging

- `setup_file_logging(log_dir)` in `logger.py` — `DEBUG→logs/debug.log`,
  `ERROR→logs/error.log`. Called automatically by `run` command.

### New: Test Suite Expansion

| File | Tests | Module coverage |
|---|---|---|
| `test_logger.py` | 28 | logger.py → 94% |
| `test_executor.py` | 14 | executor.py → 44% |
| `test_dashboard.py` | 23 | dashboard.py → 94% |
| `test_resolver.py` | 18 | resolver.py → 94% |
| `test_market_scanner.py` | 24 | market_scanner.py → 39% |
| `test_no_lookahead.py` | 7 | structural audit |

**162 passed, 4 skipped, 0 failures. Overall coverage: 53%.**
Critical-path modules (risk_manager, strategy, logger, dashboard, resolver) ≥72%.
Gap to 85%: `backtester.py` (0%) and `model_tracker.py` (30%) need mock-heavy suites.

### New: Production Scripts

| Script | Purpose |
|---|---|
| `scripts/validate_credentials.py` | Phase 0 preflight — API keys + connectivity |
| `scripts/drift_check.py` | Weekly drift vs backtest baseline (±2σ alert) |
| `scripts/reconcile.py` | Daily: internal DB state vs Polymarket on-chain |
| `scripts/emergency_close_all.py` | Kill switch — closes all positions (dry-run by default) |

### Git Tag

```
git tag v2.1.0-rc1 -m "Production readiness Phase 0 complete"
git push origin v2.1.0-rc1
```

---

## [2.0.0] — 2026-05-05

### Critical Safety (Phase 1)

#### 1. Adversarial Market Detection (`market_scanner.py`, `risk_manager.py`, `dashboard.py`)
- Added `AdversarialDetector` class — module-level singleton that persists rolling price/volume/spread history across all scan cycles
- Flags a market `SUSPICIOUS` when any of the following occur between consecutive scans:
  - Price moves > 10pp (configurable `price_jump_threshold`)
  - Volume delta > 5× rolling average of previous scan deltas (configurable `volume_spike_multiplier`)
  - Bid/ask spread collapses from > 5% to < 1% in a single scan interval
- Flagged markets block **new entries only** for 30 minutes (configurable `cooldown_minutes`); existing positions are NOT closed
- Adversarial events written to `data/adversarial.log` and sent as Telegram WARNING notifications
- `show_scan_header()` in dashboard now shows adversarial flag count in red when > 0
- New config section `adversarial:` in `config.yaml`

#### 2. Multi-Timeframe Loss Limits (`logger.py`, `risk_manager.py`, `dashboard.py`, `config.yaml`)
- Added `get_weekly_loss()` — calendar week net loss, resets Monday 00:00 UTC
- Added `get_monthly_loss()` — calendar month net loss, resets 1st of month 00:00 UTC
- `RiskManager.approve()` now enforces three independent limits:
  - **Daily** — 200 USDC soft pause (existing, now reads from `loss_limits.daily_usdc`)
  - **Weekly** — 500 USDC soft pause until Monday 00:00 UTC
  - **Monthly** — 1000 USDC hard stop (requires manual bot restart to override)
- New `show_loss_limits_panel()` dashboard panel shows current consumption with color-coded bars; shown after every scan in `run` mode
- All three limits configurable via `loss_limits:` section in `config.yaml`

### Important Risk Improvements (Phase 2)

#### 3. Opportunity Cost EV Adjustment (`strategy.py`, `dashboard.py`)
- New `opportunity_cost:` config section (default: disabled, `apy: 0.05`)
- When enabled, EV is reduced by `(days_to_expiry / 365) × apy` to account for capital locked in positions
- `Opportunity.ev` = opportunity-cost-adjusted EV (used for all decisions and ranking)
- `Opportunity.ev_raw` = pre-adjustment EV (shown alongside adjusted EV in dashboard opportunities table)
- No effect on existing behavior when `enabled: false`

#### 4. Square-Root Market Impact Slippage Model (`executor.py`, `config.yaml`)
- Replaces the flat `_SLIPPAGE = 0.003` assumption with a size-dependent model:
  - `impact = base_slippage × sqrt(order_size / market_depth)`
  - Paper mode: depth estimated as 10% of market volume (avoids extra API calls)
  - Live mode: fetches top-5 ask levels from CLOB orderbook for accurate depth
- Trade vetoed (returns `None`) if computed impact ≥ `max_impact` (default: 2%)
- New config section `slippage: {base: 0.005, max_impact: 0.02}` in `config.yaml`
- New `TradeExecutor._estimate_slippage()` and `_estimate_slippage_live()` methods

### New Fields on `Opportunity` Dataclass
| Field | Type | Description |
|---|---|---|
| `ev_raw` | `float` | EV before opportunity cost deduction |
| `adversarial` | `bool` | True if market flagged by adversarial detector |
| `volume_usdc` | `float` | Market volume (used for slippage depth estimate) |

### Tests Added (`tests/test_risk_manager.py`)
- `test_approve_rejects_weekly_limit` — weekly loss limit enforced
- `test_approve_rejects_monthly_limit` — monthly loss limit enforced
- `test_approve_rejects_adversarial` — adversarial flag blocks approval
- `test_approve_passes_non_adversarial` — non-adversarial markets pass
- `test_adversarial_detector_no_flag_on_first_scan` — no false positives on first data point
- `test_adversarial_detector_flags_price_jump` — 15pp jump triggers flag
- `test_adversarial_detector_flags_spread_collapse` — spread 8% → 0.5% triggers flag
- `test_adversarial_detector_disabled` — disabled detector never flags
- `test_opportunity_ev_raw_field` — ev_raw field present with default 0.0
- `test_opportunity_adversarial_field` — adversarial field defaults to False

**Total tests: 38/38 passing**

---

## [1.0.0] — Prior sessions

- Initial 5-phase implementation: weather ensemble, market scanner, probability engine, risk manager, executor, backtester, dashboard
- ITER 1: Real EV with spread + slippage, edge quality score, passive/aggressive execution
- ITER 2: Calibration 2.0 (Platt + Isotonic), meta model, per-source Brier scoring
- ITER 3: Regime detection (NORMAL/UNCERTAIN/EXTREME), dynamic Kelly, portfolio VaR
- ITER 4: Drift detection, stale data purge, auto-refit calibration on new outcomes
- ITER 5: Corrected Sharpe annualization, model status panel, VaR dashboard panel
- Full code audit: 8 bugs fixed across 6 files
