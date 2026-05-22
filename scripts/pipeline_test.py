"""Quick end-to-end pipeline smoke test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import json
import logging

from src.market_scanner import parse_market_condition
from src.data_fetcher import WeatherEnsemble
from src.strategy import ProbabilityEngine

logging.basicConfig(level=logging.WARNING)
cfg = yaml.safe_load(open("config.yaml"))

cache = Path("data/discovery/raw_markets.jsonl")
sample = None
for line in cache.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    m = json.loads(line)
    neg_id = m.get("negRiskMarketID", "")
    if not (neg_id and neg_id.strip("0x").strip("0")):
        continue
    q = m.get("question") or m.get("title") or ""
    if "highest temperature" in q.lower() and "New York" in q:
        sample = m
        break

if sample:
    title = sample.get("question") or sample.get("title")
    print(f"Test market: {title[:80]}")
    cond = parse_market_condition(title)
    print(f"  metric={cond['metric']} lo={cond.get('threshold_low')} hi={cond.get('threshold_high')} unit={cond['threshold_unit']}")

    from datetime import datetime, timezone, timedelta
    ensemble = WeatherEnsemble(cfg)
    target = datetime.now(timezone.utc) + timedelta(hours=24)
    weather = ensemble.fetch(40.7128, -74.0060, target)
    print(f"  Live NYC: temp={weather.get('temp_c')}C  temp_max={weather.get('temp_max_c')}C  conf={weather.get('confidence')}")

    market = {
        "condition_id": sample.get("conditionId", ""),
        "title": title,
        "metric": cond["metric"],
        "threshold": cond["threshold"],
        "threshold_low": cond.get("threshold_low"),
        "threshold_high": cond.get("threshold_high"),
        "threshold_unit": cond["threshold_unit"],
        "yes_price": 0.15,
        "no_price": 0.85,
        "volume_usdc": 10000,
        "city": "New York",
        "lat": 40.7128,
        "lon": -74.0060,
        "target_dt": sample.get("endDate", ""),
    }
    engine = ProbabilityEngine(cfg)
    opp = engine.evaluate(market, weather)
    if opp:
        print(f"  -> Opportunity: side={opp.side}  our_prob={opp.our_prob:.3f}  EV={opp.ev:.3f}  conf={opp.confidence:.2f}")
    else:
        print("  -> No opportunity (filtered by strategy — expected for past market with far expiry)")
else:
    print("No sample market found in corpus")
