"""Unit tests for the probability engine and EV calculator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from src.strategy import (
    calculate_ev,
    weather_to_probability,
    ProbabilityEngine,
    _prob_above,
    _prob_below,
)
from src.market_scanner import RAIN, SNOW, TEMP_ABOVE, TEMP_BELOW, WIND_ABOVE


# ── EV calculation ─────────────────────────────────────────────────────────────

def test_ev_positive_yes():
    # We think 70% chance, market says 50%: positive EV on YES
    ev_yes, ev_no = calculate_ev(0.70, 0.50)
    assert ev_yes > 0
    assert ev_no < 0


def test_ev_positive_no():
    # We think 20% chance, market says 50%: positive EV on NO
    ev_yes, ev_no = calculate_ev(0.20, 0.50)
    assert ev_no > 0
    assert ev_yes < 0


def test_ev_formula():
    # With 2% Polymarket fee on net profit:
    # EV_yes = 0.6 * (1/0.5 - 1) * 0.98 - 0.4 = 0.6 * 1.0 * 0.98 - 0.4 = 0.188
    ev_yes, ev_no = calculate_ev(0.60, 0.50)
    assert abs(ev_yes - 0.188) < 0.001


def test_ev_zero_edge():
    # At fair odds (50/50) with 2% fee, EV is slightly negative (house edge)
    ev_yes, ev_no = calculate_ev(0.50, 0.50)
    assert ev_yes == pytest.approx(-0.01, abs=0.001)
    assert ev_no  == pytest.approx(-0.01, abs=0.001)


# ── Probability conversion ─────────────────────────────────────────────────────

def test_rain_probability_direct():
    weather = {"precip_prob": 0.70, "temp_c": 15, "confidence": 0.8}
    market = {"metric": RAIN}
    prob = weather_to_probability(weather, market)
    assert prob == pytest.approx(0.70)


def test_snow_too_warm():
    weather = {"precip_prob": 0.80, "temp_c": 10.0, "confidence": 0.8}
    market = {"metric": SNOW}
    prob = weather_to_probability(weather, market)
    assert prob < 0.10  # warm temp suppresses snow probability


def test_snow_cold_enough():
    weather = {"precip_prob": 0.60, "temp_c": -5.0, "confidence": 0.8}
    market = {"metric": SNOW}
    prob = weather_to_probability(weather, market)
    assert prob > 0.40  # cold + precip = reasonable snow chance


def test_temp_above_threshold():
    # 20°C, threshold 15°C → should be high probability
    weather = {"precip_prob": 0, "temp_c": 20.0, "confidence": 0.8}
    market = {"metric": TEMP_ABOVE, "threshold": 15.0, "threshold_unit": "C"}
    prob = weather_to_probability(weather, market)
    assert prob > 0.70


def test_temp_above_below_threshold():
    # 10°C, threshold 20°C → low probability of exceeding
    weather = {"precip_prob": 0, "temp_c": 10.0, "confidence": 0.8}
    market = {"metric": TEMP_ABOVE, "threshold": 20.0, "threshold_unit": "C"}
    prob = weather_to_probability(weather, market)
    assert prob < 0.20


def test_temp_fahrenheit_conversion():
    # 68°F = 20°C. Threshold 60°F = 15.6°C. temp > threshold → prob > 0.5
    weather = {"precip_prob": 0, "temp_c": 20.0, "confidence": 0.8}
    market = {"metric": TEMP_ABOVE, "threshold": 60.0, "threshold_unit": "F"}
    prob = weather_to_probability(weather, market)
    assert prob > 0.50


def test_wind_above_threshold():
    weather = {"precip_prob": 0, "temp_c": 15, "wind_kph": 80, "confidence": 0.8}
    market = {"metric": WIND_ABOVE, "threshold": 50.0, "threshold_unit": "kph"}
    prob = weather_to_probability(weather, market)
    assert prob > 0.70


def test_missing_precip_returns_none():
    weather = {"precip_prob": None, "temp_c": 15, "confidence": 0.8}
    market = {"metric": RAIN}
    assert weather_to_probability(weather, market) is None


def test_unknown_metric_returns_none():
    weather = {"precip_prob": 0.5, "temp_c": 15, "confidence": 0.8}
    market = {"metric": "EARTHQUAKE"}
    assert weather_to_probability(weather, market) is None


# ── ProbabilityEngine ──────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return {
        "trading": {"ev_threshold": 0.05, "min_confidence": 0.50},
        "markets": {},
    }


def test_engine_finds_opportunity(config):
    engine = ProbabilityEngine(config)
    market = {
        "condition_id": "abc123",
        "title": "Will it rain in New York on May 5?",
        "metric": RAIN,
        "threshold": None,
        "threshold_unit": None,
        "yes_price": 0.40,
        "no_price": 0.60,
        "volume_usdc": 10000,
        "city": "new york",
        "lat": 40.71,
        "lon": -74.01,
        "target_dt": "2025-05-05",
    }
    weather = {"precip_prob": 0.75, "temp_c": 15, "confidence": 0.80}
    opp = engine.evaluate(market, weather)
    assert opp is not None
    assert opp.side == "YES"
    assert opp.ev > 0.05


def test_engine_rejects_low_confidence(config):
    engine = ProbabilityEngine(config)
    market = {
        "condition_id": "abc123",
        "title": "Rain in NYC",
        "metric": RAIN,
        "threshold": None,
        "threshold_unit": None,
        "yes_price": 0.40,
        "no_price": 0.60,
        "volume_usdc": 10000,
        "city": "new york",
        "lat": 40.71,
        "lon": -74.01,
        "target_dt": "2025-05-05",
    }
    weather = {"precip_prob": 0.75, "temp_c": 15, "confidence": 0.30}
    opp = engine.evaluate(market, weather)
    assert opp is None  # low confidence should veto


def test_engine_rejects_no_edge(config):
    engine = ProbabilityEngine(config)
    market = {
        "condition_id": "abc123",
        "title": "Rain in NYC",
        "metric": RAIN,
        "threshold": None,
        "threshold_unit": None,
        "yes_price": 0.75,
        "no_price": 0.25,
        "volume_usdc": 10000,
        "city": "new york",
        "lat": 40.71,
        "lon": -74.01,
        "target_dt": "2025-05-05",
    }
    # Our prob matches market price → no edge
    weather = {"precip_prob": 0.75, "temp_c": 15, "confidence": 0.80}
    opp = engine.evaluate(market, weather)
    assert opp is None


def test_engine_picks_no_side(config):
    engine = ProbabilityEngine(config)
    market = {
        "condition_id": "abc123",
        "title": "Rain in NYC",
        "metric": RAIN,
        "threshold": None,
        "threshold_unit": None,
        "yes_price": 0.70,
        "no_price": 0.30,
        "volume_usdc": 10000,
        "city": "new york",
        "lat": 40.71,
        "lon": -74.01,
        "target_dt": "2025-05-05",
    }
    # Our prob is only 30% — market is pricing YES at 70%: NO has edge
    weather = {"precip_prob": 0.30, "temp_c": 15, "confidence": 0.80}
    opp = engine.evaluate(market, weather)
    assert opp is not None
    assert opp.side == "NO"
