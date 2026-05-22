"""
Phase 2.6 -- Parser Validation Against Historical Markets

Tests the updated parse_market_condition() against all 785 raw markets.
Reports coverage, precision per metric, and bucket completeness.

Run:  python scripts/validate_parser.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from market_scanner import (
    parse_market_condition,
    RAIN, SNOW, TEMP_ABOVE, TEMP_BELOW, WIND_ABOVE,
    TEMP_RANGE, TEMP_ABOVE_MAX, TEMP_BELOW_MAX,
)

ALL_METRICS = [RAIN, SNOW, TEMP_ABOVE, TEMP_BELOW, WIND_ABOVE,
               TEMP_RANGE, TEMP_ABOVE_MAX, TEMP_BELOW_MAX]

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "discovery" / "raw_markets.jsonl"


def load_markets():
    markets = []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    markets.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return markets


def main():
    markets = load_markets()
    print(f"Loaded {len(markets)} markets\n")

    metric_counts = defaultdict(int)
    bucket_groups = defaultdict(set)  # neg_risk_id -> set of metric types found
    no_metric_titles = []

    for m in markets:
        title = m.get("question") or m.get("title") or ""
        cond = parse_market_condition(title)
        metric = cond["metric"]
        neg_risk_id = m.get("negRiskMarketID") or ""

        if metric is None:
            metric_counts["NONE"] += 1
            no_metric_titles.append(title)
        else:
            metric_counts[metric] += 1

        if neg_risk_id and metric in (TEMP_RANGE, TEMP_ABOVE_MAX, TEMP_BELOW_MAX):
            bucket_groups[neg_risk_id].add(metric)

    total = len(markets)
    parseable = total - metric_counts["NONE"]

    print("=" * 65)
    print("METRIC DISTRIBUTION")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Count':>5}  {'%':>6}")
    print("-" * 40)
    for metric in ALL_METRICS + ["NONE"]:
        count = metric_counts.get(metric, 0)
        pct = count / total * 100
        bar = "NEW" if metric in (TEMP_RANGE, TEMP_ABOVE_MAX, TEMP_BELOW_MAX) else "   "
        print(f"  {metric:<22} {count:>5}  {pct:>5.1f}%  {bar}")
    print("-" * 40)
    print(f"  {'PARSEABLE':<22} {parseable:>5}  {parseable/total*100:>5.1f}%")
    print(f"  {'NONE (unparsed)':<22} {metric_counts['NONE']:>5}  {metric_counts['NONE']/total*100:>5.1f}%")

    print(f"\n  Coverage: {parseable}/{total} = {parseable/total*100:.1f}%")
    print(f"  Target:   >= 80% (Phase 2 acceptance gate)")
    print(f"  Status:   {'PASS' if parseable/total >= 0.80 else 'FAIL'}")

    # Bucket group completeness
    print("\n" + "=" * 65)
    print("BUCKET GROUP ANALYSIS")
    print("=" * 65)
    print(f"  Groups with TEMP_RANGE/ABOVE_MAX/BELOW_MAX: {len(bucket_groups)}")

    complete = sum(
        1 for types in bucket_groups.values()
        if TEMP_RANGE in types and TEMP_ABOVE_MAX in types and TEMP_BELOW_MAX in types
    )
    partial = len(bucket_groups) - complete
    print(f"  Complete groups (range + top + bottom): {complete}/{len(bucket_groups)}")
    print(f"  Partial groups: {partial}")

    # Sample new metrics
    print("\n" + "=" * 65)
    print("SAMPLE: New metric types parsed")
    print("=" * 65)
    shown = {TEMP_RANGE: 0, TEMP_ABOVE_MAX: 0, TEMP_BELOW_MAX: 0}
    for m in markets:
        title = m.get("question") or m.get("title") or ""
        cond = parse_market_condition(title)
        metric = cond["metric"]
        if metric in shown and shown[metric] < 3:
            shown[metric] += 1
            if metric == TEMP_RANGE:
                print(f"  [{metric}] lo={cond['threshold_low']} hi={cond['threshold_high']} {cond['threshold_unit']}")
            else:
                print(f"  [{metric}] thr={cond['threshold']} {cond['threshold_unit']}")
            print(f"    {title[:100]}")

    # Sample unparsed
    print("\n" + "=" * 65)
    print(f"SAMPLE: First 15 unparsed markets")
    print("=" * 65)
    for title in no_metric_titles[:15]:
        print(f"  {title[:110]}")


if __name__ == "__main__":
    main()
