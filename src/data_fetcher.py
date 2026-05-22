"""
Ensemble weather fetcher — 6 sources, 4 different underlying models.

Sources and their underlying weather models:
  1. Tomorrow.io   — proprietary ML model        (requires API key)
  2. Open-Meteo    — GFS + ERA5 + ICON models    (free, no key)
  3. NWS           — NOAA GFS                    (free, US only)
  4. WeatherAPI    — proprietary                  (free 1M/month, requires key)
  5. Pirate Weather — NOAA HRRR + NBM            (free 10K/month, requires key)
  6. MET Norway    — ECMWF                       (free, no key, global)

Using sources with DIFFERENT underlying models increases ensemble accuracy.
Call WeatherEnsemble.fetch(lat, lon, target_dt) for a normalized forecast.
"""

import logging
import os
import statistics
import time
from datetime import datetime, timezone
from typing import List, Optional

import requests

from .logger import cache_weather, get_cached_weather

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "polymarket-weather-bot/1.0"})

_TIMEOUT = 10
_MAX_RETRIES = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_get(url: str, params: dict = None, headers: dict = None) -> Optional[dict]:
    """HTTP GET with exponential-backoff retry. Non-retryable 4xx errors fail immediately."""
    for attempt in range(_MAX_RETRIES):
        try:
            r = _SESSION.get(url, params=params, headers=headers, timeout=_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status in (400, 401, 403, 404):
                log.warning("HTTP %d for %s — not retrying", status, url.split("?")[0])
                return None
            if attempt < _MAX_RETRIES - 1:
                wait = 2.0 ** attempt
                log.warning("HTTP %d for %s (attempt %d/%d) — retry in %.0fs",
                            status, url.split("?")[0], attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                log.warning("HTTP %d for %s — all retries exhausted", status, url.split("?")[0])
                return None
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = 2.0 ** attempt
                log.warning("Request error %s (attempt %d/%d) — retry in %.0fs",
                            url.split("?")[0], attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                log.warning("Request failed %s — all retries exhausted: %s",
                            url.split("?")[0], type(exc).__name__)
                return None
    return None


def _dt_to_iso_hour(dt: datetime) -> str:
    return dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00:00Z")


def _closest_idx(times: list, target_ts: float, fmt: str = "%Y-%m-%dT%H:%M") -> int:
    """Return index of the time string in `times` closest to target_ts."""
    def _ts(s):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return float("inf")
    return min(range(len(times)), key=lambda i: abs(_ts(times[i]) - target_ts))


# ── Source 1: Tomorrow.io (proprietary ML) ────────────────────────────────────

def _fetch_tomorrow(lat: float, lon: float, target_dt: datetime) -> Optional[dict]:
    api_key = os.getenv("TOMORROW_API_KEY", "")
    if not api_key:
        log.warning("TOMORROW_API_KEY not set — skipping Tomorrow.io")
        return None

    data = _safe_get(
        "https://api.tomorrow.io/v4/weather/forecast",
        params={
            "location": f"{lat},{lon}",
            "apikey": api_key,
            "timesteps": "1h",
            "units": "metric",
            "fields": "precipitationProbability,temperature,humidity,windSpeed",
        },
    )
    if not data:
        return None

    try:
        hourly = data["timelines"]["hourly"]
        target_ts = target_dt.replace(tzinfo=timezone.utc).timestamp()
        best = min(
            hourly,
            key=lambda h: abs(
                datetime.fromisoformat(h["time"].replace("Z", "+00:00")).timestamp()
                - target_ts
            ),
        )
        v = best["values"]
        return {
            "precip_prob": v.get("precipitationProbability", 0) / 100.0,
            "temp_c": v.get("temperature"),
            "humidity": v.get("humidity"),
            "wind_kph": v.get("windSpeed"),
            "source": "tomorrow_io",
        }
    except (KeyError, ValueError, TypeError) as exc:
        log.warning("Tomorrow.io parse error: %s", exc)
        return None


# ── Source 2: Open-Meteo (GFS + ERA5 + ICON) ─────────────────────────────────

def _fetch_open_meteo(lat: float, lon: float, target_dt: datetime) -> Optional[dict]:
    data = _safe_get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation_probability,temperature_2m,relative_humidity_2m,wind_speed_10m",
            "daily": "temperature_2m_max",
            "wind_speed_unit": "kmh",
            "timezone": "UTC",
            "forecast_days": 7,
        },
    )
    if not data:
        return None

    try:
        times = data["hourly"]["time"]
        target_ts = target_dt.replace(tzinfo=timezone.utc).timestamp()
        idx = _closest_idx(times, target_ts, fmt="%Y-%m-%dT%H:%M")

        # Daily max: find which day the target falls on
        target_date_str = target_dt.strftime("%Y-%m-%d")
        temp_max_c = None
        daily = data.get("daily") or {}
        daily_times = daily.get("time") or []
        daily_max = daily.get("temperature_2m_max") or []
        if target_date_str in daily_times:
            day_idx = daily_times.index(target_date_str)
            if day_idx < len(daily_max):
                temp_max_c = daily_max[day_idx]

        return {
            "precip_prob": (data["hourly"]["precipitation_probability"][idx] or 0) / 100.0,
            "temp_c": data["hourly"]["temperature_2m"][idx],
            "temp_max_c": temp_max_c,
            "humidity": data["hourly"]["relative_humidity_2m"][idx],
            "wind_kph": data["hourly"]["wind_speed_10m"][idx],
            "source": "open_meteo",
        }
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        log.warning("Open-Meteo parse error: %s", exc)
        return None


# ── Source 3: NWS (NOAA GFS, US only) ────────────────────────────────────────

def _fetch_nws(lat: float, lon: float, target_dt: datetime) -> Optional[dict]:
    points = _safe_get(f"https://api.weather.gov/points/{lat},{lon}")
    if not points:
        return None

    try:
        forecast_url = points["properties"]["forecastHourly"]
    except (KeyError, TypeError):
        log.debug("NWS: non-US location, skipping")
        return None

    data = _safe_get(forecast_url)
    if not data:
        return None

    try:
        periods = data["properties"]["periods"]
        target_ts = target_dt.replace(tzinfo=timezone.utc).timestamp()
        best = min(
            periods,
            key=lambda p: abs(datetime.fromisoformat(p["startTime"]).timestamp() - target_ts),
        )
        prob_val = best.get("probabilityOfPrecipitation", {}).get("value") or 0
        precip_prob = float(prob_val) / 100.0

        temp_f = best.get("temperature")
        temp_c = (temp_f - 32) * 5 / 9 if temp_f is not None else None

        wind_str = best.get("windSpeed", "")
        wind_mph = float(wind_str.split()[0]) if wind_str and wind_str[0].isdigit() else None
        wind_kph = wind_mph * 1.60934 if wind_mph else None

        return {
            "precip_prob": precip_prob,
            "temp_c": temp_c,
            "humidity": None,
            "wind_kph": wind_kph,
            "source": "nws",
        }
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        log.warning("NWS parse error: %s", exc)
        return None


# ── Source 4: WeatherAPI.com (proprietary, free 1M/month) ────────────────────

def _fetch_weatherapi(lat: float, lon: float, target_dt: datetime) -> Optional[dict]:
    api_key = os.getenv("WEATHERAPI_KEY", "")
    if not api_key:
        log.debug("WEATHERAPI_KEY not set — skipping WeatherAPI")
        return None

    # WeatherAPI forecast: up to 14 days
    days_ahead = max(1, min(14, int((target_dt - datetime.now(timezone.utc)).total_seconds() / 86400) + 1))
    data = _safe_get(
        "https://api.weatherapi.com/v1/forecast.json",
        params={
            "key": api_key,
            "q": f"{lat},{lon}",
            "days": days_ahead,
            "aqi": "no",
            "alerts": "no",
        },
    )
    if not data:
        return None

    try:
        target_ts = target_dt.replace(tzinfo=timezone.utc).timestamp()
        best_hour = None
        best_diff = float("inf")

        for day in data.get("forecast", {}).get("forecastday", []):
            for hour in day.get("hour", []):
                # WeatherAPI "time" is LOCAL — use "time_epoch" (UTC Unix timestamp) instead
                epoch = hour.get("time_epoch")
                if epoch is not None:
                    diff = abs(int(epoch) - target_ts)
                else:
                    # Fallback: parse local time (inaccurate for non-UTC zones)
                    try:
                        t = datetime.strptime(hour["time"], "%Y-%m-%d %H:%M")
                        diff = abs(t.replace(tzinfo=timezone.utc).timestamp() - target_ts)
                    except ValueError:
                        continue
                if diff < best_diff:
                    best_diff = diff
                    best_hour = hour

        if not best_hour:
            return None

        # Use the higher of rain/snow probability as total precip probability
        rain_pct = best_hour.get("chance_of_rain") or 0
        snow_pct = best_hour.get("chance_of_snow") or 0
        precip_pct = max(rain_pct, snow_pct)
        return {
            "precip_prob": precip_pct / 100.0,
            "temp_c": best_hour.get("temp_c"),
            "humidity": best_hour.get("humidity"),
            "wind_kph": best_hour.get("wind_kph"),
            "source": "weatherapi",
        }
    except (KeyError, ValueError, TypeError) as exc:
        log.warning("WeatherAPI parse error: %s", exc)
        return None


# ── Source 5: Pirate Weather (NOAA HRRR + NBM, free 10K/month) ───────────────

def _fetch_pirate_weather(lat: float, lon: float, target_dt: datetime) -> Optional[dict]:
    api_key = os.getenv("PIRATE_WEATHER_KEY", "")
    if not api_key:
        log.debug("PIRATE_WEATHER_KEY not set — skipping Pirate Weather")
        return None

    target_ts = int(target_dt.replace(tzinfo=timezone.utc).timestamp())
    data = _safe_get(
        f"https://api.pirateweather.net/forecast/{api_key}/{lat},{lon},{target_ts}",
        params={"units": "si", "exclude": "daily,alerts,flags"},
    )
    if not data:
        return None

    try:
        # Time-machine call returns currently object for the requested time
        current = data.get("currently", {})
        # Hourly block also available
        hourly = data.get("hourly", {}).get("data", [])
        if hourly:
            best = min(
                hourly,
                key=lambda h: abs(h.get("time", 0) - target_ts),
            )
        else:
            best = current

        precip_prob = best.get("precipProbability") or 0
        temp_c = best.get("temperature")           # SI = Celsius
        humidity = (best.get("humidity") or 0) * 100  # 0-1 → 0-100
        wind_kph = (best.get("windSpeed") or 0) * 3.6  # m/s → kph

        return {
            "precip_prob": float(precip_prob),
            "temp_c": float(temp_c) if temp_c is not None else None,
            "humidity": float(humidity),
            "wind_kph": float(wind_kph),
            "source": "pirate_weather",
        }
    except (KeyError, ValueError, TypeError) as exc:
        log.warning("Pirate Weather parse error: %s", exc)
        return None


# ── Source 6: MET Norway / Yr.no (ECMWF, free, no key) ───────────────────────

def _fetch_met_norway(lat: float, lon: float, target_dt: datetime) -> Optional[dict]:
    """
    MET Norway Locationforecast 2.0 — uses ECMWF model, global coverage.
    No API key required but needs a meaningful User-Agent (required by their ToS).
    """
    data = _safe_get(
        "https://api.met.no/weatherapi/locationforecast/2.0/compact",
        params={"lat": round(lat, 4), "lon": round(lon, 4)},
        headers={"User-Agent": "polymarket-weather-bot/1.0 github.com/user/polymarcket"},
    )
    if not data:
        return None

    try:
        timeseries = data["properties"]["timeseries"]
        target_ts = target_dt.replace(tzinfo=timezone.utc).timestamp()

        best = min(
            timeseries,
            key=lambda t: abs(
                datetime.fromisoformat(t["time"].replace("Z", "+00:00")).timestamp()
                - target_ts
            ),
        )

        instant = best["data"]["instant"]["details"]
        next1h  = best["data"].get("next_1_hours", {}).get("details", {})
        next6h  = best["data"].get("next_6_hours", {}).get("details", {})

        # Prefer explicit probability field; fall back to expected-mm heuristic
        if "probability_of_precipitation" in next1h:
            precip_prob = float(next1h["probability_of_precipitation"]) / 100.0
        elif "probability_of_precipitation" in next6h:
            precip_prob = float(next6h["probability_of_precipitation"]) / 100.0
        else:
            # Expected mm is a proxy, not a probability; use a conservative sigmoid-like mapping.
            # 0mm → ~0%, 1mm → ~33%, 2mm → ~55%, 5mm → ~83%
            precip_mm = float(next1h.get("precipitation_amount") or next6h.get("precipitation_amount") or 0)
            precip_prob = precip_mm / (precip_mm + 2.0) if precip_mm > 0 else 0.0

        return {
            "precip_prob": precip_prob,
            "temp_c": instant.get("air_temperature"),
            "humidity": instant.get("relative_humidity"),
            "wind_kph": (instant.get("wind_speed") or 0) * 3.6,  # m/s → kph
            "source": "met_norway",
        }
    except (KeyError, ValueError, TypeError) as exc:
        log.warning("MET Norway parse error: %s", exc)
        return None


# ── Ensemble aggregation ──────────────────────────────────────────────────────

# Source weights — reflect accuracy and model independence
# (lower weight = less trusted or correlated with others)
_WEIGHTS = {
    "tomorrow_io":    0.30,
    "open_meteo":     0.20,
    "nws":            0.15,
    "weatherapi":     0.15,
    "pirate_weather": 0.10,
    "met_norway":     0.10,
}


def _detect_regime(precip_probs: List[float]) -> str:
    """
    Classify model-agreement regime from cross-source precipitation spread.

    NORMAL   (std < 0.10) — sources agree; safe to trade at full size
    UNCERTAIN(std < 0.22) — moderate disagreement; reduce size
    EXTREME  (std ≥ 0.22) — wild disagreement; skip trading entirely
    """
    if len(precip_probs) < 2:
        return "UNCERTAIN"
    std = statistics.stdev(precip_probs)
    if std >= 0.22:
        return "EXTREME"
    if std >= 0.10:
        return "UNCERTAIN"
    return "NORMAL"


def _aggregate(forecasts: List[dict], weights_override: Optional[dict] = None) -> dict:
    """
    Weighted average across available sources with regime classification.
    Uses dynamic weights from ModelTracker if provided, else static defaults.
    """
    if not forecasts:
        raise ValueError("No weather sources returned data")

    weights = weights_override or _WEIGHTS

    def wavg(field: str) -> Optional[float]:
        vals = [
            (f[field], weights.get(f["source"], 0.10))
            for f in forecasts
            if f.get(field) is not None
        ]
        if not vals:
            return None
        total_w = sum(w for _, w in vals)
        return sum(v * w for v, w in vals) / total_w if total_w > 0 else None

    precip_probs = [f["precip_prob"] for f in forecasts if f.get("precip_prob") is not None]
    temp_vals    = [f["temp_c"]      for f in forecasts if f.get("temp_c")      is not None]
    wind_vals    = [f["wind_kph"]    for f in forecasts if f.get("wind_kph")    is not None]

    conf_components = []
    if len(precip_probs) >= 2:
        std = statistics.stdev(precip_probs)
        conf_components.append(max(0.0, 1.0 - std / 0.5))
    if len(temp_vals) >= 2:
        std_t = statistics.stdev(temp_vals)
        conf_components.append(max(0.0, 1.0 - std_t / 5.0))
    if len(wind_vals) >= 2:
        std_w = statistics.stdev(wind_vals)
        conf_components.append(max(0.0, 1.0 - std_w / 20.0))

    if conf_components:
        base_conf = sum(conf_components) / len(conf_components)
        source_bonus = min(0.10, (len(forecasts) - 2) * 0.02) if len(forecasts) >= 2 else 0
        confidence = min(1.0, base_conf + source_bonus)
    else:
        confidence = 0.30

    regime = _detect_regime(precip_probs)

    return {
        "precip_prob": wavg("precip_prob"),
        "temp_c":      wavg("temp_c"),
        "temp_max_c":  wavg("temp_max_c"),   # daily maximum temperature (for bucket markets)
        "humidity":    wavg("humidity"),
        "wind_kph":    wavg("wind_kph"),
        "confidence":  round(confidence, 3),
        "regime":      regime,
        "sources_used": [f["source"] for f in forecasts],
        "source_count": len(forecasts),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

class WeatherEnsemble:
    """
    Fetches weather from up to 6 sources and returns a consensus forecast.
    Sources that lack an API key or fail are silently skipped.
    Results are cached in SQLite for `cache_ttl_seconds`.
    """

    # Ordered fetch list: (config_key, fetch_fn)
    _SOURCES = [
        ("tomorrow_io",    _fetch_tomorrow),
        ("open_meteo",     _fetch_open_meteo),
        ("nws",            _fetch_nws),
        ("weatherapi",     _fetch_weatherapi),
        ("pirate_weather", _fetch_pirate_weather),
        ("met_norway",     _fetch_met_norway),
    ]

    def __init__(self, config: dict):
        self.cfg = config.get("weather", {})
        self.ttl = self.cfg.get("cache_ttl_seconds", 3600)

    def fetch(self, lat: float, lon: float, target_dt: datetime) -> dict:
        """
        Return ensemble forecast for (lat, lon) at target_dt.
        Raises RuntimeError only if every source fails.
        """
        target_key = _dt_to_iso_hour(target_dt)

        cached = get_cached_weather(lat, lon, target_key, self.ttl)
        if cached:
            log.debug("Weather cache hit (%.4f, %.4f) @ %s", lat, lon, target_key)
            return {
                "precip_prob":  cached.get("precip_prob"),
                "temp_c":       cached.get("temp_c"),
                "humidity":     cached.get("humidity"),
                "wind_kph":     cached.get("wind_kph"),
                "confidence":   cached.get("confidence"),
                "regime":       cached.get("regime", "NORMAL"),
                "sources_used": (cached.get("sources") or "").split(",") if cached.get("sources") else [],
            }

        log.info("Fetching ensemble (%.4f, %.4f) @ %s", lat, lon, target_key)

        source_cfg = self.cfg.get("sources", {})
        forecasts = []

        for key, fn in self._SOURCES:
            enabled = source_cfg.get(key, {}).get("enabled", True)
            if not enabled:
                continue
            result = fn(lat, lon, target_dt)
            if result:
                forecasts.append(result)
                log.debug("  ✓ %s: precip=%.0f%%  temp=%.1f°C",
                          key, (result.get("precip_prob") or 0) * 100,
                          result.get("temp_c") or 0)
            else:
                log.debug("  ✗ %s: unavailable", key)

        if not forecasts:
            raise RuntimeError(
                f"All 6 weather sources failed for ({lat}, {lon}) @ {target_key}"
            )

        # Get dynamic weights from self-learning model tracker
        weights = _WEIGHTS
        try:
            from .model_tracker import get_tracker
            tracker = get_tracker()
            dyn_weights = tracker.get_dynamic_weights()
            if dyn_weights:
                weights = dyn_weights
            # Record per-source precip predictions for future Brier scoring
            for f in forecasts:
                if f.get("precip_prob") is not None:
                    tracker.record_source_precip(
                        lat, lon, target_key, f["source"], f["precip_prob"]
                    )
        except Exception as exc:
            log.debug("ModelTracker integration failed (non-fatal): %s", exc)

        ensemble = _aggregate(forecasts, weights_override=weights)
        log.info(
            "Ensemble (%d/%d sources): precip=%.0f%%  temp=%.1f°C  conf=%.2f  regime=%s  [%s]",
            len(forecasts), len(self._SOURCES),
            (ensemble["precip_prob"] or 0) * 100,
            ensemble["temp_c"] or 0,
            ensemble["confidence"],
            ensemble["regime"],
            ", ".join(ensemble["sources_used"]),
        )

        cache_weather(
            lat=lat, lon=lon, target_dt=target_key,
            precip_prob=ensemble["precip_prob"] or 0,
            temp_c=ensemble["temp_c"] or 0,
            humidity=ensemble["humidity"] or 0,
            wind_kph=ensemble["wind_kph"] or 0,
            confidence=ensemble["confidence"],
            sources=ensemble["sources_used"],
            regime=ensemble.get("regime", "NORMAL"),
        )

        return ensemble
