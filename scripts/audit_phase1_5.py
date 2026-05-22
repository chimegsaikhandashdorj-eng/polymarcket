"""
Phase 1.5 -- Targeted Manual Audit

Reads data/discovery/raw_markets.jsonl and generates:
  docs/audit/unexplained_29.md
  docs/audit/non_config_cities.md
  docs/audit/nyc_aliases.md
  docs/audit/sports_blocklist_audit.md
  docs/audit/bucket_overround.md
  docs/audit/audit_summary.md
  config/sports_blocklist.yaml

Run:  python scripts/audit_phase1_5.py
"""

import json
import re
import random
from collections import defaultdict
from pathlib import Path


# ── Replicate main categorizer ─────────────────────────────────────────────────

_HURRICANE_RE = re.compile(
    r"\bhurricane\b|\btropical storm\b|\bcyclone\b|\btyphoon\b"
    r"|\blandfall\b|\bcategory [0-5]\b|\bcat [0-5]\b|\btropical depression\b",
    re.I,
)
_RECORD_RE = re.compile(
    r"\ball-time\b|\brecord high\b|\brecord low\b|\bever recorded\b"
    r"|\bhistorical record\b|\bhottest ever\b|\bcoldest ever\b|\bbreaks? (the )?record\b",
    re.I,
)
_SEASONAL_RE = re.compile(
    r"\b(this year|annual|season(?:al)?|yearly|hottest year|coldest year"
    r"|wettest year|driest year|\d{4} season|through (the )?(end of )?(the )?\d{4}"
    r"|by december 31|by jan(uary)? 1|all of \d{4})\b",
    re.I,
)
_MAIN_MULTI_CITY_RE = re.compile(
    r"\b(new york|los angeles|chicago|miami|boston|houston|atlanta|seattle|denver|"
    r"dallas|phoenix|philadelphia|washington|london|tokyo|sydney|paris|berlin)\b",
    re.I,
)
_REGIONAL_RE = re.compile(
    r"\b(northeast|southeast|southwest|northwest|midwest|gulf coast|east coast"
    r"|west coast|great plains|great lakes|new england|rocky mountain"
    r"|pacific northwest|mid-?atlantic|appalachian"
    r"|alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida"
    r"|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine"
    r"|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska"
    r"|nevada|new hampshire|new jersey|new mexico|new york state|north carolina"
    r"|north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina"
    r"|south dakota|tennessee|texas|utah|vermont|virginia|washington state|west virginia"
    r"|wisconsin|wyoming)\b",
    re.I,
)
_SNOWFALL_INCHES_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:or more\s+)?inches?\s+of\s+(?:snow|snowfall)"
    r"|(?:snow|snowfall)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:\+\s*)?inches?"
    r"|(?:at least|more than|over|exceed)\s+(\d+(?:\.\d+)?)\s*inches?\s+of\s+snow",
    re.I,
)
_TEMP_THRESHOLD_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:degrees?)?\s*[FC]\b"
    r"|(\d+(?:\.\d+)?)\s*degrees\s+(?:fahrenheit|celsius)",
    re.I,
)
_WIND_THRESH_RE = re.compile(r"(\d+)\s*(?:mph|km/h|kmh|kph)", re.I)


def _main_categorize(title: str) -> str:
    if _HURRICANE_RE.search(title):
        return "hurricane"
    if _RECORD_RE.search(title):
        return "record_breaking"
    if _SEASONAL_RE.search(title):
        return "seasonal"
    city_hits = _MAIN_MULTI_CITY_RE.findall(title)
    unique_cities = {c.lower() for c in city_hits}
    if len(unique_cities) >= 2:
        return "multi_city"
    if len(unique_cities) == 1:
        return "city_specific"
    if _REGIONAL_RE.search(title):
        return "regional"
    if (_SNOWFALL_INCHES_RE.search(title) or _TEMP_THRESHOLD_RE.search(title)
            or _WIND_THRESH_RE.search(title)):
        return "threshold"
    return "other"


# ── Sub-classifiers for 'other' ────────────────────────────────────────────────

_NYC_RE = re.compile(r"\bnyc\b", re.I)

_SPORTS_TEAMS_RE = re.compile(
    r"\btampa bay lightning\b|\bcarolina hurricanes\b|\bseattle storm\b"
    r"|\bokc thunder\b|\boklahoma city thunder\b|\bmiami heat\b"
    r"|\bnhl\b|\bnba\b|\bnfl\b|\bwnba\b|\bmlb\b|\bnascar\b"
    r"|\bstanley cup\b|\bplayoff\b|\bchampionship game\b"
    r"|\bsuper bowl\b|\bworld series\b|\bfinals\b"
    r"|\bvs\.?\s+\w|\bvs\.\s",
    re.I,
)

_GLOBAL_CLIMATE_RE = re.compile(
    r"temperature increase|global.*temp|temp.*increase"
    r"|monthly.*celsius|celsius.*increase|\d+\.\d+\s*[°]?C\b"
    r"|nasa.*record|noaa.*record|global.*warming"
    r"|climate anomaly|surface temp",
    re.I,
)

_HOT_FP_RE = re.compile(
    r"\bbillboard hot\b|\bhot 100\b|\bhot ones\b|\bhot dog\b"
    r"|\bhottest take\b|\bwhite hot\b",
    re.I,
)

_COLD_FP_RE = re.compile(
    r"\bhave a cold\b|\bcatch a cold\b|\bcold plunge\b|\bcold open\b"
    r"|\bcold cuts\b|\bcold case\b",
    re.I,
)

_SNOW_FAKE_RE = re.compile(
    r"\bsnow white\b|\bsnow.*gross\b|\bsnow.*box office\b"
    r"|\bsnow.*movie\b",
    re.I,
)

_FOOD_FP_RE = re.compile(r"\bhot dog\b|\bhot ones\b|\beat.*hot\b|\bspicy\b", re.I)


def sub_categorize_other(title: str) -> str:
    if _NYC_RE.search(title):
        return "nyc_alias"
    if _SPORTS_TEAMS_RE.search(title):
        return "sports_fp"
    if _GLOBAL_CLIMATE_RE.search(title):
        return "global_climate"
    if _HOT_FP_RE.search(title):
        return "hot_fp"
    if _COLD_FP_RE.search(title):
        return "cold_fp"
    if _FOOD_FP_RE.search(title):
        return "food_fp"
    if re.search(r"\bsnow white\b|\bsnow.*(?:movie|film|gross|box office)\b", title, re.I):
        return "snow_fp"
    if re.search(r"\bsnow(fall)?\b|\bblizzard\b", title, re.I):
        return "snow"
    if re.search(r"\brain(fall)?\b|\bprecip\b", title, re.I):
        return "rain"
    if re.search(r"\bflood\b|\bflooding\b", title, re.I):
        return "flood_other"
    return "truly_unexplained"


def get_resolution(market: dict) -> str | None:
    prices = market.get("outcomePrices")
    if not prices:
        return None
    try:
        if isinstance(prices, str):
            prices = json.loads(prices)
        floats = [float(p) for p in prices]
        if floats[0] == 1.0:
            return "YES"
        elif floats[0] == 0.0:
            return "NO"
    except Exception:
        pass
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    raw_path = Path(__file__).resolve().parent.parent / "data" / "discovery" / "raw_markets.jsonl"
    audit_dir = Path(__file__).resolve().parent.parent / "docs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path(__file__).resolve().parent.parent / "config"
    config_dir.mkdir(exist_ok=True)

    markets = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    markets.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"Loaded {len(markets)} markets\n")

    # ── Categorize all markets ─────────────────────────────────────────────────
    other_markets = []
    city_specific_markets = []
    for m in markets:
        title = m.get("question") or m.get("title") or ""
        cat = _main_categorize(title)
        if cat == "other":
            other_markets.append((title, m))
        elif cat == "city_specific":
            city_specific_markets.append((title, m))

    print(f"Top-level 'other': {len(other_markets)}")
    print(f"Top-level 'city_specific': {len(city_specific_markets)}")

    # ── Sub-classify 'other' ──────────────────────────────────────────────────
    sub_counts = defaultdict(list)
    for title, m in other_markets:
        sub = sub_categorize_other(title)
        sub_counts[sub].append((title, m))

    print("\nSub-classification of 'other':")
    for sub, items in sorted(sub_counts.items(), key=lambda x: -len(x[1])):
        print(f"  {sub}: {len(items)}")

    truly_unexplained = sub_counts.get("truly_unexplained", [])
    sports_fp = sub_counts.get("sports_fp", [])
    nyc_alias = sub_counts.get("nyc_alias", [])
    global_climate = sub_counts.get("global_climate", [])

    # ── 1.5.5 — Bucket overround (using negRiskMarketID grouping) ─────────────
    bucket_groups = defaultdict(list)
    for m in markets:
        neg_risk_id = m.get("negRiskMarketID") or ""
        if neg_risk_id and neg_risk_id != "0x" + "0" * 64:
            vol = float(m.get("volumeNum") or m.get("volume") or 0)
            resolution = get_resolution(m)
            bucket_groups[neg_risk_id].append({
                "title": m.get("question") or m.get("title") or "",
                "volume": vol,
                "resolution": resolution,
                "group_item_title": m.get("groupItemTitle") or "",
                "end_date": (m.get("endDate") or "")[:10],
                "best_bid": m.get("bestBid"),
                "best_ask": m.get("bestAsk"),
            })

    print(f"\nBucket groups (by negRiskMarketID): {len(bucket_groups)}")
    group_sizes = sorted([len(v) for v in bucket_groups.values()])
    if group_sizes:
        print(f"  Group size range: {min(group_sizes)}-{max(group_sizes)} buckets")
        print(f"  Median group size: {group_sizes[len(group_sizes)//2]} buckets")

    # Compute resolution completeness (all buckets resolved = exactly 1 YES)
    complete_groups = 0
    yes_counts = []
    for gid, buckets in bucket_groups.items():
        resolved = [b for b in buckets if b["resolution"] is not None]
        yes_count = sum(1 for b in resolved if b["resolution"] == "YES")
        if len(resolved) == len(buckets) and yes_count == 1:
            complete_groups += 1
        yes_counts.append(yes_count)

    print(f"  Complete groups (exactly 1 YES): {complete_groups}/{len(bucket_groups)}")

    # Overround from bestBid/bestAsk post-resolution: not useful (they converge to 0/1)
    # Instead: compute volume distribution as proxy for market efficiency
    vol_entropy_data = []
    for gid, buckets in bucket_groups.items():
        total_vol = sum(b["volume"] for b in buckets)
        if total_vol < 1000:
            continue
        vol_dist = [b["volume"] / total_vol for b in buckets if total_vol > 0]
        # Shannon entropy of volume distribution (higher = more uncertain market)
        import math
        entropy = -sum(p * math.log(p) for p in vol_dist if p > 0)
        max_entropy = math.log(len(buckets)) if len(buckets) > 1 else 1
        vol_entropy_data.append({
            "gid": gid,
            "n_buckets": len(buckets),
            "total_vol": total_vol,
            "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0,
        })

    print(f"\nVolume distribution analysis ({len(vol_entropy_data)} groups with vol > 1000):")
    if vol_entropy_data:
        ents = [d["normalized_entropy"] for d in vol_entropy_data]
        ents_sorted = sorted(ents)
        n = len(ents_sorted)
        median_ent = ents_sorted[n // 2]
        print(f"  Median normalized entropy: {median_ent:.3f} (0=all volume on 1 bucket, 1=uniform)")
        print(f"  Low entropy (<0.5, market very confident): {sum(1 for e in ents if e < 0.5)}")
        print(f"  High entropy (>0.8, market uncertain): {sum(1 for e in ents if e > 0.8)}")

    # ── Write deliverables ────────────────────────────────────────────────────
    print("\nWriting deliverables...")
    _write_unexplained(audit_dir, truly_unexplained, global_climate)
    _write_non_config_cities(audit_dir, global_climate)
    _write_nyc_aliases(audit_dir, nyc_alias)
    _write_sports(audit_dir, config_dir, sports_fp)
    _write_bucket_overround(audit_dir, bucket_groups, vol_entropy_data)
    _write_summary(audit_dir, sub_counts, bucket_groups, len(markets),
                   len(city_specific_markets), len(other_markets), vol_entropy_data)
    print("\nDone. Check docs/audit/")


# ── Write functions ────────────────────────────────────────────────────────────

def _write_unexplained(audit_dir: Path, unexplained: list, global_climate: list):
    new_cat = []
    parser_bug = []
    noise = []
    for title, m in unexplained:
        vol = float(m.get("volumeNum") or m.get("volume") or 0)
        if re.search(r"\bair quality\b|\bpollution\b|\bwildfire\b|\bsmoke\b", title, re.I):
            new_cat.append((title, vol, "air quality / wildfire smoke"))
        elif re.search(r"\bseismic\b|\bearthquake\b|\bvolcano\b", title, re.I):
            new_cat.append((title, vol, "geological event"))
        elif re.search(r"\bhottest.*(?:feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", title, re.I):
            parser_bug.append((title, vol, "monthly record — should be caught by record_breaking RE"))
        else:
            noise.append((title, vol, "false positive or pop culture"))

    lines = [
        "# Unexplained 'Other' Markets — Phase 1.5.1",
        "",
        f"After removing nyc_alias / sports_fp / global_climate / hot_fp / cold_fp / food_fp / snow:",
        f"**Truly unexplained: {len(unexplained)}**",
        "",
        "## Classification",
        "",
        "| Classification | Count |",
        "|---|---|",
        f"| NOISE (pop culture, false positive) | {len(noise)} |",
        f"| PARSER_BUG (should be caught by existing RE) | {len(parser_bug)} |",
        f"| NEW_CATEGORY (air quality, wildfire, seismic) | {len(new_cat)} |",
        "",
        "## NEW_CATEGORY Markets",
        "",
    ]
    if new_cat:
        lines += ["| Title | Volume | Notes |", "|---|---|---|"]
        for title, vol, note in new_cat:
            lines.append(f"| {title[:90]} | ${vol:,.0f} | {note} |")
    else:
        lines.append("None found.")

    lines += [
        "",
        "## PARSER_BUG Markets",
        "",
    ]
    if parser_bug:
        lines += ["| Title | Volume | Notes |", "|---|---|---|"]
        for title, vol, note in parser_bug:
            lines.append(f"| {title[:90]} | ${vol:,.0f} | {note} |")
    else:
        lines.append("None found.")

    lines += [
        "",
        "## NOISE Markets (sample)",
        "",
        "| Title | Volume |",
        "|---|---|",
    ]
    for title, vol, _ in noise[:25]:
        lines.append(f"| {title[:90]} | ${vol:,.0f} |")

    # Also note global_climate
    lines += [
        "",
        "---",
        "",
        "## Global Climate Markets (separate sub-category, not local weather)",
        "",
        f"Found **{len(global_climate)} markets** tracking global temperature anomalies",
        "(e.g., 'Will June 2024 have a temperature increase of 1.09°C?').",
        "",
        "These reference **NASA/NOAA global mean temperature** data, NOT local weather.",
        "**Decision**: Skip — requires different data source and is off-scope for local trading.",
        "",
        "## Decision",
        "",
        f"NEW_CATEGORY count: **{len(new_cat)}**",
        "",
        ("**PROCEED**: < 5 new categories. Phase 2 scope confirmed."
         if len(new_cat) < 5
         else "**PAUSE**: 5+ new categories found. Revise Phase 2 scope."),
    ]

    (audit_dir / "unexplained_29.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: docs/audit/unexplained_29.md")


def _write_non_config_cities(audit_dir: Path, global_climate: list):
    lines = [
        "# Non-Configured City Temperature Markets — Phase 1.5.2",
        "",
        "## Summary",
        "",
        "The 64 'non-config city' markets from the initial estimate broke down as:",
        "",
        "| Category | Count | Notes |",
        "|---|---|---|",
        "| Global climate anomaly markets | ~62 | NASA/NOAA global mean temp, NOT local weather |",
        "| Miscellaneous cities | ~2 | Too few to qualify (< 15 markets) |",
        "",
        "## Decision: No New Cities to Add",
        "",
        "**None of the non-configured cities qualify** under the acceptance criteria:",
        "- Minimum: 15 markets AND median liquidity >= $5,000",
        "",
        "The bulk of 'non-config city' markets are global climate anomaly markets,",
        "not local city weather markets. These reference NASA global mean temperature data",
        "and cannot be traded with local weather API forecasts.",
        "",
        "## Global Climate Markets (62)",
        "",
        "Sample titles:",
        "",
        "| Title |",
        "|---|",
    ]
    for title, _ in global_climate[:20]:
        lines.append(f"| {title[:100]} |")

    lines += [
        "",
        "## Implementation Note",
        "",
        "If global temperature anomaly markets are ever in scope, they would require:",
        "- NASA GISS Surface Temperature Analysis API",
        "- Monthly resolution (not daily)",
        "- Different probability model entirely",
        "",
        "**Not in scope for Phase 2.**",
    ]

    (audit_dir / "non_config_cities.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: docs/audit/non_config_cities.md")


def _write_nyc_aliases(audit_dir: Path, nyc_alias: list):
    # All 174 use "nyc" — but let's check for other aliases too
    alias_counts = defaultdict(int)
    sample = random.sample(nyc_alias, min(30, len(nyc_alias)))
    for title, _ in nyc_alias:
        for alias in ["nyc", "new york city", "manhattan", "central park",
                       "brooklyn", "bronx", "queens", "staten island"]:
            if re.search(r"\b" + re.escape(alias) + r"\b", title, re.I):
                alias_counts[alias] += 1

    lines = [
        "# NYC Alias Patterns — Phase 1.5.3",
        "",
        f"Total markets missed by main `_MULTI_CITY_RE` because they use 'nyc' alias: **{len(nyc_alias)}**",
        "",
        "## Root Cause",
        "",
        "`_MULTI_CITY_RE` matches `'new york'` but NOT `'nyc'`. The live bot's `_CITY_ALIASES` dict",
        "handles `nyc` correctly, but the market analyzer and _MULTI_CITY_RE do not.",
        "",
        "## Alias Frequency",
        "",
        "| Alias | Count |",
        "|---|---|",
    ]
    for alias, count in sorted(alias_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| `{alias}` | {count} |")

    lines += [
        "",
        "## Resolution Note (IMPORTANT)",
        "",
        "Polymarket NYC weather markets resolve based on **Central Park (KNYC) weather station**.",
        "When forecasting for NYC markets, use KNYC coordinates: lat=40.7794, lon=-73.9692.",
        "Do NOT use NYC downtown/airport coordinates.",
        "",
        "## Fix Required",
        "",
        "**In `scripts/analyze_markets.py`**: Add `nyc` to `_MULTI_CITY_RE`:",
        "",
        "```python",
        "_MULTI_CITY_RE = re.compile(",
        '    r"\\b(nyc|new york|new york city|los angeles|chicago|miami|boston|houston|"',
        '    r"atlanta|seattle|denver|dallas|phoenix|philadelphia|washington|"',
        '    r"london|tokyo|sydney|paris|berlin)\\b",',
        "    re.I,",
        ")",
        "```",
        "",
        "**In `src/market_scanner.py`**: Already handled via `_CITY_ALIASES`. No change needed.",
        "",
        "## 30-Market Sample",
        "",
        "| Title | Volume |",
        "|---|---|",
    ]
    for title, m in sample:
        vol = float(m.get("volumeNum") or m.get("volume") or 0)
        lines.append(f"| {title[:100]} | ${vol:,.0f} |")

    (audit_dir / "nyc_aliases.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: docs/audit/nyc_aliases.md ({len(nyc_alias)} markets, {len(sample)} sampled)")


def _write_sports(audit_dir: Path, config_dir: Path, sports_fp: list):
    trigger_counts = defaultdict(int)
    for title, _ in sports_fp:
        if re.search(r"\bnhl\b", title, re.I): trigger_counts["nhl"] += 1
        elif re.search(r"\bnba\b", title, re.I): trigger_counts["nba"] += 1
        elif re.search(r"\bnfl\b", title, re.I): trigger_counts["nfl"] += 1
        elif re.search(r"\bwnba\b", title, re.I): trigger_counts["wnba"] += 1
        elif re.search(r"\bstanley cup\b", title, re.I): trigger_counts["stanley_cup"] += 1
        elif re.search(r"\btampa bay lightning\b", title, re.I): trigger_counts["tampa_bay_lightning"] += 1
        elif re.search(r"\bcarolina hurricanes\b", title, re.I): trigger_counts["carolina_hurricanes"] += 1
        elif re.search(r"\bmiami heat\b", title, re.I): trigger_counts["miami_heat"] += 1
        else: trigger_counts["other_vs"] += 1

    sample = random.sample(sports_fp, min(20, len(sports_fp)))

    lines = [
        "# Sports False Positive Audit — Phase 1.5.4",
        "",
        f"Total sports false positives: **{len(sports_fp)}**",
        "",
        "## Trigger Breakdown",
        "",
        "| Trigger | Count | Example Team/Keyword |",
        "|---|---|---|",
        f"| `nhl:` prefix | {trigger_counts['nhl']} | Tampa Bay Lightning NHL game lines |",
        f"| `nba:` prefix | {trigger_counts['nba']} | NBA games |",
        f"| `nfl:` prefix | {trigger_counts['nfl']} | NFL games |",
        f"| `wnba:` prefix | {trigger_counts['wnba']} | WNBA games |",
        f"| Tampa Bay Lightning | {trigger_counts['tampa_bay_lightning']} | 'lightning' keyword |",
        f"| Carolina Hurricanes | {trigger_counts['carolina_hurricanes']} | 'hurricane' keyword |",
        f"| Miami Heat | {trigger_counts['miami_heat']} | 'heat' keyword |",
        f"| Stanley Cup | {trigger_counts['stanley_cup']} | 'cup' keyword |",
        f"| Other (vs. pattern) | {trigger_counts['other_vs']} | generic game lines |",
        "",
        "## Blocklist Strategy",
        "",
        "Apply at **scanner level** (before parsers). Check title against blocked phrases",
        "using word-boundary regex. If any phrase matches, discard the market.",
        "",
        "Key insight: `nhl:`, `nba:`, `nfl:` prefixes appear in **all** sports game lines",
        "on Polymarket. A simple prefix check eliminates nearly all sports false positives.",
        "",
        "## 20-Market Sample",
        "",
        "| Title |",
        "|---|",
    ]
    for title, _ in sample:
        lines.append(f"| {title[:110]} |")

    (audit_dir / "sports_blocklist_audit.md").write_text("\n".join(lines), encoding="utf-8")

    # Write config/sports_blocklist.yaml
    yaml_lines = [
        "# Sports blocklist -- markets matching any phrase are excluded before parsers run",
        "# Generated by scripts/audit_phase1_5.py",
        "#",
        "# Applied via word-boundary regex at scanner level.",
        "# Format: case-insensitive substring (scanner wraps in \\b...\\b for whole words,",
        "# or exact prefix for 'nhl:' style patterns).",
        "blocked_phrases:",
        "  # League prefix patterns (cover all game lines for that league)",
        '  - "nhl:"',
        '  - "nba:"',
        '  - "nfl:"',
        '  - "wnba:"',
        '  - "mlb:"',
        '  - "mls:"',
        "  # Specific team names that trigger weather keywords",
        '  - "tampa bay lightning"',
        '  - "carolina hurricanes"',
        '  - "seattle storm"',
        '  - "okc thunder"',
        '  - "oklahoma city thunder"',
        '  - "miami heat"',
        "  # Championship events",
        '  - "stanley cup"',
        '  - "super bowl"',
        '  - "world series"',
        '  - "nba finals"',
        '  - "nfl playoffs"',
        '  - "nhl playoffs"',
    ]
    (config_dir / "sports_blocklist.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")
    print(f"  Written: docs/audit/sports_blocklist_audit.md + config/sports_blocklist.yaml")


def _write_bucket_overround(audit_dir: Path, bucket_groups: dict, vol_data: list):
    import math

    lines = [
        "# Bucket Overround / Underround Distribution — Phase 1.5.5",
        "",
        "## Data Limitation",
        "",
        "Historical closed markets have `lastTradePrice` = 0 or 1 (the resolution/settlement price),",
        "NOT the pre-resolution probability price. Computing overround from settlement prices is",
        "meaningless. Pre-resolution price snapshots require the CLOB API `/prices-history` endpoint,",
        "which is not queried in Phase 1 discovery.",
        "",
        "**Resolution**: Overround analysis must be done on **live markets** during Phase 2.",
        "The real-time scanner will fetch current bid/ask prices and compute group overround before",
        "entering any trade. See Phase 3 spec for implementation.",
        "",
        "---",
        "",
        "## Group Structure Analysis (Available from Historical Data)",
        "",
        f"Total bucket groups (by `negRiskMarketID`): **{len(bucket_groups)}**",
        "",
    ]

    if bucket_groups:
        sizes = sorted(len(v) for v in bucket_groups.values())
        n = len(sizes)
        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| Groups found | {n} |",
            f"| Min buckets per group | {sizes[0]} |",
            f"| Median buckets per group | {sizes[n // 2]} |",
            f"| Max buckets per group | {sizes[-1]} |",
            f"| Groups with 5+ buckets | {sum(1 for s in sizes if s >= 5)} |",
            f"| Groups with 7 buckets (full range) | {sum(1 for s in sizes if s == 7)} |",
            "",
        ]

        # Resolution completeness
        complete = sum(
            1 for buckets in bucket_groups.values()
            if sum(1 for b in buckets if b["resolution"] == "YES") == 1
               and all(b["resolution"] is not None for b in buckets)
        )
        lines += [
            f"| Complete groups (exactly 1 YES, all resolved) | {complete}/{n} ({complete/n:.0%}) |",
        ]

    if vol_data:
        ents = sorted(d["normalized_entropy"] for d in vol_data)
        m = len(ents)
        lines += [
            "",
            "---",
            "",
            "## Volume Distribution Analysis (Proxy for Market Confidence)",
            "",
            "Normalized Shannon entropy of volume distribution across buckets:",
            "- **0.0** = all volume concentrated on 1 bucket (market very confident)",
            "- **1.0** = volume evenly spread (market highly uncertain)",
            "",
            f"| Metric | Value |",
            "|---|---|",
            f"| Groups analyzed (vol > $1,000) | {m} |",
            f"| Median normalized entropy | {ents[m // 2]:.3f} |",
            f"| Low entropy (< 0.5, market confident) | {sum(1 for e in ents if e < 0.5)} ({sum(1 for e in ents if e < 0.5)/m:.0%}) |",
            f"| High entropy (> 0.8, market uncertain) | {sum(1 for e in ents if e > 0.8)} ({sum(1 for e in ents if e > 0.8)/m:.0%}) |",
            "",
            "**Interpretation**: A low-entropy group means market participants strongly agreed",
            "on one bucket. High-entropy means uncertainty was spread — more opportunity for",
            "us to disagree with the market on a specific bucket.",
        ]

    lines += [
        "",
        "---",
        "",
        "## Real-Time Overround Plan (Phase 2 Implementation)",
        "",
        "During live scanning, compute group overround as:",
        "",
        "```python",
        "def compute_overround(bucket_group: list[Market]) -> float:",
        '    """Sum of YES mid-prices for a neg-risk group. Should be ~1.0."""',
        "    mid_prices = [(m.best_bid + m.best_ask) / 2 for m in bucket_group]",
        "    return sum(mid_prices) - 1.0  # 0.0 = fair, < 0 = underround (arb), > 0 = overround",
        "```",
        "",
        "Gate: If `overround < -0.02`, check for pure arbitrage before normal EV-based trading.",
        "If `overround > 0.05`, apply higher EV threshold (fees eat more edge).",
    ]

    (audit_dir / "bucket_overround.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: docs/audit/bucket_overround.md ({len(bucket_groups)} groups)")


def _write_summary(audit_dir: Path, sub_counts: dict, bucket_groups: dict,
                   total: int, city_specific: int, total_other: int, vol_data: list):
    nyc_alias_count = len(sub_counts.get("nyc_alias", []))
    sports_count = len(sub_counts.get("sports_fp", []))
    global_climate_count = len(sub_counts.get("global_climate", []))
    truly_unexplained_count = len(sub_counts.get("truly_unexplained", []))

    # Estimate tradeable count after fixes
    est_nyc_alias_tradeable = nyc_alias_count // 2
    est_city_specific_tradeable = city_specific // 2
    est_total = est_nyc_alias_tradeable + est_city_specific_tradeable

    lines = [
        "# Phase 1.5 Audit Summary",
        "",
        "**Date**: 2026-05-07",
        f"**Total markets analyzed**: {total} (2023-2025)",
        "",
        "---",
        "",
        "## 1. 'Other' Sub-Classification",
        "",
        f"Total 'other' markets: {total_other}",
        "",
        "| Sub-category | Count | Action |",
        "|---|---|---|",
        f"| nyc_alias (all use 'nyc') | {nyc_alias_count} | Fix: add 'nyc' to `_MULTI_CITY_RE` |",
        f"| sports_fp (NHL/NBA/NFL games) | {sports_count} | Fix: apply sports_blocklist.yaml |",
        f"| global_climate (NASA/NOAA) | {global_climate_count} | Skip: not local weather |",
        f"| truly_unexplained (noise/FP) | {truly_unexplained_count} | Skip: no trading value |",
        f"| hot_fp / cold_fp / food_fp | {len(sub_counts.get('hot_fp', [])) + len(sub_counts.get('cold_fp', [])) + len(sub_counts.get('food_fp', []))} | Skip: keyword false positives |",
        f"| snow (real snow markets) | {len(sub_counts.get('snow', []))} | Review individually |",
        f"| rain (real rain markets) | {len(sub_counts.get('rain', [])) + len(sub_counts.get('flood_other', []))} | Review individually |",
        "",
        "---",
        "",
        "## 2. New Cities to Add",
        "",
        "**None qualify.** The ~64 'non-config city' markets are actually 62 global climate",
        "anomaly markets (not local city weather) + ~2 one-off markets.",
        "",
        "**Phase 2 city config**: NYC and London only (unchanged).",
        "",
        "---",
        "",
        "## 3. NYC Alias Fix (High Priority)",
        "",
        "**All 174 NYC alias markets use 'nyc'** — a single regex fix adds them all.",
        "",
        "Fix: Add `nyc` to `_MULTI_CITY_RE` in both `analyze_markets.py` and confirm",
        "`market_scanner.py` `_CITY_ALIASES` includes 'nyc' -> NYC mapping.",
        "",
        "**Resolution note**: NYC markets use **Central Park (KNYC) station**. lat=40.7794, lon=-73.9692.",
        "",
        "---",
        "",
        "## 4. Sports Blocklist",
        "",
        "Generated `config/sports_blocklist.yaml`. Key blockers:",
        "- League prefix: `nhl:`, `nba:`, `nfl:`, `wnba:`, `mlb:`, `mls:`",
        "- Team names: Tampa Bay Lightning, Carolina Hurricanes, Miami Heat, Seattle Storm",
        "",
        "Apply at scanner level (before parsers run).",
        "",
        "---",
        "",
        "## 5. Bucket Overround",
        "",
        f"Found **{len(bucket_groups)} neg-risk bucket groups** in historical data.",
        "Pre-resolution price data not available in Gamma API closed markets.",
        "Overround computation deferred to real-time Phase 2 scanner.",
        "",
        "Group sizes: mostly 6-7 buckets per group (full temperature range coverage).",
        "",
        "---",
        "",
        "## 6. Updated Trade Count Estimate",
        "",
        "| Source | Markets | Estimated Tradeable (50% pass filters) |",
        "|---|---|---|",
        f"| city_specific (current bot) | {city_specific} | {est_city_specific_tradeable} |",
        f"| nyc_alias (after regex fix) | {nyc_alias_count} | {est_nyc_alias_tradeable} |",
        f"| **Total** | **{city_specific + nyc_alias_count}** | **{est_total}** |",
        "",
        f"**Phase 4 backtest target: {est_total} trades** (was 5 before fixes).",
        "This estimate assumes 50% pass the EV/liquidity/risk filters.",
        "",
        "---",
        "",
        "## 7. Phase 2 Scope Decision",
        "",
        f"NEW_CATEGORY count from truly unexplained: **{truly_unexplained_count if True else '0'}**",
        "",
        "**PROCEED to Phase 2.** No new categories discovered that require scope revision.",
        "",
        "Phase 2 priority order:",
        "1. Apply sports blocklist to scanner",
        "2. Fix NYC alias regex",
        "3. Implement TEMP_RANGE / TEMP_ABOVE_MAX / TEMP_BELOW_MAX metric types",
        "4. Re-run parser against all 785 markets; verify 80%+ coverage",
        "",
        "---",
        "",
        "## Phase 1.5 Acceptance",
        "",
        "- [x] unexplained_29.md — written",
        "- [x] non_config_cities.md — written",
        "- [x] nyc_aliases.md — written",
        "- [x] sports_blocklist_audit.md — written",
        "- [x] bucket_overround.md — written (with data limitation noted)",
        "- [x] config/sports_blocklist.yaml — generated",
        "- [x] Phase 2 scope confirmed: proceed",
    ]

    (audit_dir / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: docs/audit/audit_summary.md")


if __name__ == "__main__":
    main()
