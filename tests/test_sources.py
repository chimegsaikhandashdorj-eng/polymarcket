import logging, yaml, sqlite3
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone
from src.data_fetcher import WeatherEnsemble
from src.logger import init_db

init_db()
with sqlite3.connect(Path("data/trades.db")) as conn:
    conn.execute("DELETE FROM weather_cache")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

e = WeatherEnsemble(config)
cities = [
    ("New York", 40.7128, -74.0060),
    ("London",   51.5074, -0.1278),
    ("Tokyo",    35.6762, 139.6503),
]
for name, lat, lon in cities:
    r = e.fetch(lat, lon, datetime.now(timezone.utc))
    src = r.get("sources_used", ["cached"])
    n   = r.get("source_count", 0)
    print(f"{name}: {n}/6 sources | precip={r['precip_prob']:.0%}  temp={r['temp_c']:.1f}C  conf={r['confidence']:.2f}")
    print(f"  => {src}")
