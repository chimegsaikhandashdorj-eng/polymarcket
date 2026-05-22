import sys
sys.path.insert(0, ".")
from src.market_scanner import parse_market_condition
from src.data_fetcher import WeatherEnsemble
import yaml

titles = [
    "Will the highest temperature in New York City be 82 F or below on June 19?",
    "Will the highest temperature in NYC be 16F or below on January 22?",
    "Will the highest temperature in London be between 17-18F on May 8?",
    "Will the highest temperature in London be 25F or higher on May 8?",
]
for t in titles:
    c = parse_market_condition(t)
    m = c["metric"]
    thr = c.get("threshold")
    lo = c.get("threshold_low")
    hi = c.get("threshold_high")
    unit = c["threshold_unit"]
    print(f"metric={m} thr={thr} lo={lo} hi={hi} unit={unit}")
    print(f"  {t[:70]}")
    print()

cfg = yaml.safe_load(open("config.yaml"))
ens = WeatherEnsemble(cfg)
methods = [m for m in dir(ens) if not m.startswith("_")]
print("WeatherEnsemble methods:", methods)
