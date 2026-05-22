# Phase 1.5 Audit Summary

**Date**: 2026-05-07
**Total markets analyzed**: 785 (2023-2025)

---

## 1. 'Other' Sub-Classification

Total 'other' markets: 360

| Sub-category | Count | Action |
|---|---|---|
| nyc_alias (all use 'nyc') | 174 | Fix: add 'nyc' to `_MULTI_CITY_RE` |
| sports_fp (NHL/NBA/NFL games) | 60 | Fix: apply sports_blocklist.yaml |
| global_climate (NASA/NOAA) | 62 | Skip: not local weather |
| truly_unexplained (noise/FP) | 29 | Skip: no trading value |
| hot_fp / cold_fp / food_fp | 23 | Skip: keyword false positives |
| snow (real snow markets) | 3 | Review individually |
| rain (real rain markets) | 4 | Review individually |

---

## 2. New Cities to Add

**None qualify.** The ~64 'non-config city' markets are actually 62 global climate
anomaly markets (not local city weather) + ~2 one-off markets.

**Phase 2 city config**: NYC and London only (unchanged).

---

## 3. NYC Alias Fix (High Priority)

**All 174 NYC alias markets use 'nyc'** — a single regex fix adds them all.

Fix: Add `nyc` to `_MULTI_CITY_RE` in both `analyze_markets.py` and confirm
`market_scanner.py` `_CITY_ALIASES` includes 'nyc' -> NYC mapping.

**Resolution note**: NYC markets use **Central Park (KNYC) station**. lat=40.7794, lon=-73.9692.

---

## 4. Sports Blocklist

Generated `config/sports_blocklist.yaml`. Key blockers:
- League prefix: `nhl:`, `nba:`, `nfl:`, `wnba:`, `mlb:`, `mls:`
- Team names: Tampa Bay Lightning, Carolina Hurricanes, Miami Heat, Seattle Storm

Apply at scanner level (before parsers run).

---

## 5. Bucket Overround

Found **103 neg-risk bucket groups** in historical data.
Pre-resolution price data not available in Gamma API closed markets.
Overround computation deferred to real-time Phase 2 scanner.

Group sizes: mostly 6-7 buckets per group (full temperature range coverage).

---

## 6. Updated Trade Count Estimate

| Source | Markets | Estimated Tradeable (50% pass filters) |
|---|---|---|
| city_specific (current bot) | 345 | 172 |
| nyc_alias (after regex fix) | 174 | 87 |
| **Total** | **519** | **259** |

**Phase 4 backtest target: 259 trades** (was 5 before fixes).
This estimate assumes 50% pass the EV/liquidity/risk filters.

---

## 7. Phase 2 Scope Decision

NEW_CATEGORY count from truly unexplained: **29**

**PROCEED to Phase 2.** No new categories discovered that require scope revision.

Phase 2 priority order:
1. Apply sports blocklist to scanner
2. Fix NYC alias regex
3. Implement TEMP_RANGE / TEMP_ABOVE_MAX / TEMP_BELOW_MAX metric types
4. Re-run parser against all 785 markets; verify 80%+ coverage

---

## Phase 1.5 Acceptance

- [x] unexplained_29.md — written
- [x] non_config_cities.md — written
- [x] nyc_aliases.md — written
- [x] sports_blocklist_audit.md — written
- [x] bucket_overround.md — written (with data limitation noted)
- [x] config/sports_blocklist.yaml — generated
- [x] Phase 2 scope confirmed: proceed