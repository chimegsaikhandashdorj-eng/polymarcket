"""Unit tests for market_scanner.py — parsing, CityIndex, AdversarialDetector."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from src.market_scanner import (
    parse_market_condition, CityIndex,
    AdversarialDetector, get_adversarial_detector,
    RAIN, SNOW, TEMP_ABOVE, TEMP_BELOW, WIND_ABOVE,
)


# ── parse_market_condition ─────────────────────────────────────────────────────

def test_parse_rain_title():
    r = parse_market_condition("Will it rain in New York on May 5?")
    assert r["metric"] == RAIN


def test_parse_precipitation_title():
    r = parse_market_condition("Will there be precipitation in Miami today?")
    assert r["metric"] == RAIN


def test_parse_rainfall_title():
    r = parse_market_condition("Rainfall exceeds 10mm in London on Friday")
    assert r["metric"] == RAIN


def test_parse_snow_title():
    r = parse_market_condition("Will it snow in Chicago this weekend?")
    assert r["metric"] == SNOW


def test_parse_blizzard_title():
    r = parse_market_condition("Blizzard conditions in NYC on Jan 15?")
    assert r["metric"] == SNOW


def test_parse_snowfall_title():
    r = parse_market_condition("Snowfall exceeds 5 inches in Boston?")
    assert r["metric"] == SNOW


def test_parse_temp_above_fahrenheit():
    r = parse_market_condition("Will temperature exceed 90°F in Phoenix?")
    assert r["metric"] == TEMP_ABOVE
    assert r["threshold"] == pytest.approx(90.0)
    assert r["threshold_unit"] == "F"


def test_parse_temp_below_celsius():
    r = parse_market_condition("Will temp be below 5°C in London?")
    assert r["metric"] == TEMP_BELOW
    assert r["threshold"] == pytest.approx(5.0)
    assert r["threshold_unit"] == "C"


def test_parse_temp_above_celsius():
    r = parse_market_condition("Will it be above 30 degrees C in Tokyo?")
    assert r["metric"] == TEMP_ABOVE
    assert r["threshold"] == pytest.approx(30.0)


def test_parse_wind_mph():
    r = parse_market_condition("Will winds exceed 50 mph during the storm?")
    assert r["metric"] == WIND_ABOVE
    assert r["threshold"] == pytest.approx(50.0)
    assert r["threshold_unit"] == "mph"


def test_parse_wind_kph():
    r = parse_market_condition("Will wind speed reach 80 km/h in Sydney?")
    assert r["metric"] == WIND_ABOVE
    assert r["threshold"] == pytest.approx(80.0)
    assert r["threshold_unit"] == "kph"


def test_parse_unknown_metric():
    r = parse_market_condition("Will the stock market rally this week?")
    assert r["metric"] is None
    assert r["threshold"] is None


def test_parse_empty_title():
    r = parse_market_condition("")
    assert r["metric"] is None


def test_parse_snow_takes_priority_over_rain():
    """A title mentioning both snow and rain should be classified as SNOW (snow check first)."""
    r = parse_market_condition("Will snow or rain fall in Denver?")
    assert r["metric"] == SNOW


# ── CityIndex ─────────────────────────────────────────────────────────────────

@pytest.fixture
def city_index():
    return CityIndex([
        {"name": "New York", "lat": 40.71, "lon": -74.01, "country": "US"},
        {"name": "Los Angeles", "lat": 34.05, "lon": -118.24, "country": "US"},
        {"name": "London", "lat": 51.51, "lon": -0.13, "country": "GB"},
    ])


def test_city_index_exact_match(city_index):
    result = city_index.match("Will it rain in New York on May 5?")
    assert result is not None
    name, loc = result
    assert name == "new york"
    assert loc["country"] == "US"


def test_city_index_case_insensitive(city_index):
    result = city_index.match("RAIN IN LONDON TOMORROW")
    assert result is not None
    name, loc = result
    assert name == "london"


def test_city_index_no_match(city_index):
    result = city_index.match("Will it rain in Paris?")
    assert result is None


def test_city_index_longer_name_wins(city_index):
    """'Los Angeles' should match before a substring like 'Angeles'."""
    result = city_index.match("Temperature in Los Angeles exceeds 100F")
    assert result is not None
    name, _ = result
    assert "los angeles" in name


def test_city_index_empty(city_index):
    result = city_index.match("")
    assert result is None


def test_city_index_no_cities():
    ci = CityIndex([])
    assert ci.match("Any city?") is None


# ── AdversarialDetector ────────────────────────────────────────────────────────

def _det(enabled=True, price_jump=0.10, vol_mult=5.0, spread_hi=0.05, spread_lo=0.01, cooldown=30):
    return AdversarialDetector({
        "adversarial": {
            "enabled": enabled,
            "price_jump_threshold": price_jump,
            "volume_spike_multiplier": vol_mult,
            "spread_collapse_high": spread_hi,
            "spread_collapse_low": spread_lo,
            "cooldown_minutes": cooldown,
        }
    })


def test_adversarial_no_flag_first_scan():
    det = _det()
    assert not det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.02)


def test_adversarial_no_flag_normal_movement():
    det = _det()
    det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.02)
    flagged = det.update_and_check("mkt1", price=0.52, volume=10500, spread=0.02)
    assert not flagged


def test_adversarial_flags_price_jump():
    det = _det(price_jump=0.10)
    det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.02)
    flagged = det.update_and_check("mkt1", price=0.65, volume=10000, spread=0.02)
    assert flagged


def test_adversarial_flags_spread_collapse():
    det = _det(spread_hi=0.05, spread_lo=0.01)
    det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.08)
    flagged = det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.005)
    assert flagged


def test_adversarial_flags_volume_spike():
    det = _det(vol_mult=3.0)
    # Need >= 3 history points for volume spike detection.
    # Previous deltas: 100, 100 → avg = 100. Current delta must be > 3×100 = 300.
    det.update_and_check("mkt1", price=0.50, volume=100,  spread=0.02)
    det.update_and_check("mkt1", price=0.50, volume=200,  spread=0.02)
    det.update_and_check("mkt1", price=0.50, volume=300,  spread=0.02)
    # delta = 700 - 300 = 400; avg of [100, 100] = 100; 400 > 3×100 → True
    flagged = det.update_and_check("mkt1", price=0.50, volume=700, spread=0.02)
    assert flagged


def test_adversarial_disabled_never_flags():
    det = _det(enabled=False)
    det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.02)
    flagged = det.update_and_check("mkt1", price=0.99, volume=10000, spread=0.0001)
    assert not flagged


def test_adversarial_is_flagged_returns_false_when_not_flagged():
    det = _det()
    assert not det.is_flagged("unknown_mkt")


def test_adversarial_flag_count_increments():
    det = _det()
    assert det.get_flag_count() == 0
    det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.02)
    det.update_and_check("mkt1", price=0.65, volume=10000, spread=0.02)
    assert det.get_flag_count() == 1


def test_adversarial_different_markets_independent():
    det = _det()
    det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.02)
    det.update_and_check("mkt2", price=0.30, volume=5000,  spread=0.03)
    # Flagging mkt1 should not affect mkt2's flag status
    det.update_and_check("mkt1", price=0.70, volume=10000, spread=0.02)
    assert det.is_flagged("mkt1")
    assert not det.is_flagged("mkt2")


def test_adversarial_flag_persists_during_active_cooldown():
    """is_flagged() returns True while the cooldown window is still open."""
    det = _det(cooldown=60)  # 60-minute cooldown — won't expire during test
    det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.02)
    det.update_and_check("mkt1", price=0.65, volume=10000, spread=0.02)  # 0.15 > 0.10
    assert det.is_flagged("mkt1")  # still within 60-min window


def test_adversarial_zero_cooldown_expires_immediately():
    """cooldown_minutes=0 means is_flagged() is False straight after flagging."""
    det = _det(cooldown=0)
    det.update_and_check("mkt1", price=0.50, volume=10000, spread=0.02)
    det.update_and_check("mkt1", price=0.65, volume=10000, spread=0.02)  # flags
    # cooldown = 0s → now - ts >= 0, not < 0 → is_flagged() returns False
    assert not det.is_flagged("mkt1")


# ── get_adversarial_detector (singleton) ─────────────────────────────────────

def test_singleton_returns_same_instance():
    import src.market_scanner as ms
    ms._ADVERSARIAL_DETECTOR = None  # reset
    cfg = {"adversarial": {"enabled": True}}
    d1 = get_adversarial_detector(cfg)
    d2 = get_adversarial_detector(cfg)
    assert d1 is d2
    ms._ADVERSARIAL_DETECTOR = None  # clean up
