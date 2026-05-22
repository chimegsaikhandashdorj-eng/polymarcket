"""
Phase 1.3 -- Distribution Analysis & Market Categorization

Reads data/discovery/raw_markets.jsonl, applies heuristic category labels,
and generates:
  - Console summary table (counts, volumes, parsability)
  - docs/market_scope_decision.md  (Phase 1.4 deliverable)

Run:  python scripts/analyze_markets.py
      python scripts/analyze_markets.py --sample 200   # print random sample for manual review
      python scripts/analyze_markets.py --csv           # also dump data/discovery/categorized.csv
"""

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


# ── Categorization heuristics ──────────────────────────────────────────────────

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
_MULTI_CITY_RE = re.compile(
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


def categorize(title: str) -> str:
    """Assign a single best-fit category label to a market title."""
    if _HURRICANE_RE.search(title):
        return "hurricane"
    if _RECORD_RE.search(title):
        return "record_breaking"
    if _SEASONAL_RE.search(title):
        return "seasonal"
    city_hits = _MULTI_CITY_RE.findall(title)
    unique_cities = {c.lower() for c in city_hits}
    if len(unique_cities) >= 2:
        return "multi_city"
    if len(unique_cities) == 1:
        return "city_specific"
    if _REGIONAL_RE.search(title):
        return "regional"
    if _SNOWFALL_INCHES_RE.search(title) or _TEMP_THRESHOLD_RE.search(title) or _WIND_THRESH_RE.search(title):
        return "threshold"
    return "other"


def current_bot_handles(title: str, category: str) -> bool:
    """Rough check: can the current bot (v2.2) parse AND model this market?"""
    if category != "city_specific":
        return False
    known = {"new york", "nyc", "los angeles", "chicago", "miami", "london", "tokyo", "sydney"}
    title_l = title.lower()
    if not any(c in title_l for c in known):
        return False
    metric_re = re.compile(
        r"\brain\b|\bprecip|\bsnow\b|\btemperature\b|\bdegrees?\b|\bwind\b|\bblizzard\b",
        re.I,
    )
    return bool(metric_re.search(title))


# ── Acceptance scoring ─────────────────────────────────────────────────────────

ACCEPTANCE_THRESHOLDS = {
    "min_markets_per_year": 30,
    "min_median_volume_usdc": 5_000,
    "min_resolve_rate": 0.95,
}


def score_category(stats: dict, total_years: float) -> dict:
    issues = []
    annualized = stats["count"] / max(total_years, 1)
    if annualized < ACCEPTANCE_THRESHOLDS["min_markets_per_year"]:
        issues.append(f"only {annualized:.0f} markets/year (need {ACCEPTANCE_THRESHOLDS['min_markets_per_year']})")
    if stats["median_vol"] < ACCEPTANCE_THRESHOLDS["min_median_volume_usdc"]:
        issues.append(f"median volume ${stats['median_vol']:.0f} (need ${ACCEPTANCE_THRESHOLDS['min_median_volume_usdc']:.0f})")
    resolve_rate = stats["resolved"] / max(stats["count"], 1)
    if resolve_rate < ACCEPTANCE_THRESHOLDS["min_resolve_rate"]:
        issues.append(f"only {resolve_rate:.0%} have resolution data (need {ACCEPTANCE_THRESHOLDS['min_resolve_rate']:.0%})")
    return {"passes": len(issues) == 0, "issues": issues, "annualized": annualized}


# ── Main ───────────────────────────────────────────────────────────────────────

def load_markets(path: Path) -> list:
    markets = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    markets.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return markets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Print N random markets for manual review (0 = skip)")
    parser.add_argument("--csv", action="store_true",
                        help="Also write data/discovery/categorized.csv")
    args = parser.parse_args()

    raw_path = Path(__file__).resolve().parent.parent / "data" / "discovery" / "raw_markets.jsonl"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run discover_markets.py first.")
        return

    markets = load_markets(raw_path)
    print(f"Loaded {len(markets)} markets from {raw_path}\n")
    if not markets:
        print("File is empty.")
        return

    # Determine date range span
    end_dates = [(m.get("endDate") or "")[:10] for m in markets]
    end_dates = [d for d in end_dates if d]
    if end_dates:
        min_date = min(end_dates)
        max_date = max(end_dates)
        from datetime import datetime
        span_days = (datetime.strptime(max_date, "%Y-%m-%d") - datetime.strptime(min_date, "%Y-%m-%d")).days
        total_years = max(span_days / 365.25, 1.0)
    else:
        min_date = max_date = "unknown"
        total_years = 1.0

    print(f"Date range: {min_date} to {max_date}  ({total_years:.1f} years)\n")

    # Categorize
    stats = defaultdict(lambda: {"count": 0, "total_vol": 0, "volumes": [], "resolved": 0,
                                  "parseable_now": 0, "median_vol": 0})
    categorized = []
    for m in markets:
        title = m.get("question") or m.get("title") or ""
        cat = categorize(title)
        vol = float(m.get("volume") or 0)
        resolution = m.get("resolutionPrice")
        parseable = current_bot_handles(title, cat)

        stats[cat]["count"] += 1
        stats[cat]["total_vol"] += vol
        stats[cat]["volumes"].append(vol)
        if resolution is not None:
            stats[cat]["resolved"] += 1
        if parseable:
            stats[cat]["parseable_now"] += 1

        categorized.append({
            "title": title,
            "category": cat,
            "volume_usdc": vol,
            "resolved": resolution is not None,
            "resolution_price": resolution,
            "parseable_now": parseable,
            "end_date": (m.get("endDate") or "")[:10],
            "condition_id": m.get("conditionId") or m.get("id") or "",
        })

    # Compute medians
    for cat in stats:
        vols = sorted(stats[cat]["volumes"])
        stats[cat]["median_vol"] = vols[len(vols) // 2] if vols else 0

    # Print summary table
    categories = sorted(stats.keys(), key=lambda c: stats[c]["count"], reverse=True)
    COL = 20
    print(f"{'Category':<{COL}} {'Count':>6} {'Mkt/yr':>7} {'TotalVol':>11} {'MedianVol':>10} {'Resolved':>9} {'Parseable':>10}")
    print("-" * 80)
    for cat in categories:
        s = stats[cat]
        annualized = s["count"] / total_years
        resolve_rate = s["resolved"] / max(s["count"], 1)
        parse_rate = s["parseable_now"] / max(s["count"], 1)
        print(f"  {cat:<{COL-2}} {s['count']:>6} {annualized:>7.0f} "
              f"${s['total_vol']:>10,.0f} ${s['median_vol']:>9,.0f} "
              f"{resolve_rate:>8.0%} {parse_rate:>9.0%}")
    total_count = len(markets)
    total_parseable = sum(s["parseable_now"] for s in stats.values())
    print("-" * 80)
    print(f"  {'TOTAL':<{COL-2}} {total_count:>6}")
    print(f"\n  Currently parseable by bot: {total_parseable}/{total_count} ({total_parseable/max(total_count,1):.0%})\n")

    # Acceptance scoring
    print("\n=== ACCEPTANCE SCORING ===\n")
    passed = []
    for cat in categories:
        result = score_category(stats[cat], total_years)
        icon = "PASS" if result["passes"] else "FAIL"
        annualized_str = f"{result['annualized']:.0f}/yr"
        if result["passes"]:
            print(f"  [{icon}] {cat:<{COL-2}} ({annualized_str})")
            passed.append(cat)
        else:
            print(f"  [{icon}] {cat:<{COL-2}} ({annualized_str}) -- {'; '.join(result['issues'])}")

    print(f"\n  Candidates passing all criteria: {passed or ['none']}\n")

    # Sample for manual review
    if args.sample > 0:
        sample_n = min(args.sample, len(categorized))
        sample = random.sample(categorized, sample_n)
        print(f"\n=== RANDOM SAMPLE ({sample_n} markets for manual review) ===\n")
        for i, m in enumerate(sample, 1):
            print(f"  {i:3d}. [{m['category']:<18}] vol=${m['volume_usdc']:>8,.0f}  "
                  f"resolved={'Y' if m['resolved'] else 'N'}  "
                  f"parseable={'Y' if m['parseable_now'] else 'N'}")
            print(f"       {m['title'][:120]}")

    # CSV export
    if args.csv:
        csv_path = raw_path.parent / "categorized.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "condition_id", "end_date", "category", "volume_usdc",
                "resolved", "resolution_price", "parseable_now", "title",
            ])
            writer.writeheader()
            writer.writerows(categorized)
        print(f"\nCSV written to {csv_path}")

    # Write decision document
    _write_decision_doc(stats, categories, passed, total_years, min_date, max_date, total_count)


def _write_decision_doc(stats, categories, passed, total_years, min_date, max_date, total_count):
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    out = docs_dir / "market_scope_decision.md"

    lines = [
        "# Market Scope Decision",
        "",
        f"Date range analyzed: {min_date} to {max_date} ({total_years:.1f} years)",
        f"Total weather-adjacent markets found: {total_count}",
        "",
        "---",
        "",
        "## Category Distribution",
        "",
        "| Category | Count | Mkt/yr | Total Volume | Median Vol | Resolved | Parseable Now |",
        "|---|---|---|---|---|---|---|",
    ]
    for cat in categories:
        s = stats[cat]
        annualized = s["count"] / total_years
        resolve_rate = s["resolved"] / max(s["count"], 1)
        parse_rate = s["parseable_now"] / max(s["count"], 1)
        lines.append(
            f"| {cat} | {s['count']} | {annualized:.0f} "
            f"| ${s['total_vol']:,.0f} | ${s['median_vol']:,.0f} "
            f"| {resolve_rate:.0%} | {parse_rate:.0%} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Acceptance Criteria",
        "",
        "| Criterion | Threshold |",
        "|---|---|",
        "| Annualized market count | >= 30 markets/year |",
        "| Median liquidity | >= 5,000 USDC |",
        "| Resolvability | >= 95% have resolution data |",
        "",
        "## Candidate Scoring",
        "",
    ]
    for cat in categories:
        result = score_category(stats[cat], total_years)
        icon = "PASS" if result["passes"] else "FAIL"
        lines.append(f"- **[{icon}] {cat}** -- {result['annualized']:.0f} markets/year")
        for issue in result["issues"]:
            lines.append(f"  - {issue}")

    lines += [
        "",
        "---",
        "",
        "## Selected Categories for Implementation",
        "",
        "Categories that passed all acceptance criteria:",
        "",
    ]
    if passed:
        for cat in passed:
            lines.append(f"- **{cat}**")
    else:
        lines += [
            "**None passed all criteria automatically.**",
            "",
            "Review the category distribution and sample markets "
            "(`python scripts/analyze_markets.py --sample 200 --csv`) "
            "to decide manually which categories are worth implementing despite marginal stats.",
        ]

    lines += [
        "",
        "---",
        "",
        "## Known Unknowns",
        "",
        "- Markets in the `other` category need manual review to find sub-patterns.",
        "- Volume data from Gamma API is total lifetime volume, not daily liquidity.",
        "- Resolution data availability: check categorized.csv for gaps in resolutionPrice.",
        "",
        "---",
        "",
        "*Generated by `scripts/analyze_markets.py` -- re-run after any data refresh.*",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDecision document written to {out}")


if __name__ == "__main__":
    main()
