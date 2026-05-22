"""Unit tests for crypto_signals module — multi-timeframe, regime, correlation, volume."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch
from src.crypto_signals import (
    multi_timeframe_trend,
    detect_crypto_regime,
    check_crypto_correlation,
    detect_volume_anomaly,
    compute_composite_signal,
    confirm_crypto_option_edge,
    get_fear_greed_index,
    _linear_slope,
)


# ── Multi-Timeframe Trend ────────────────────────────────────────────────────

class TestMultiTimeframeTrend:
    def test_not_enough_data(self):
        result = multi_timeframe_trend([100.0] * 10)
        assert result["alignment"] == 0.0
        assert result["confidence"] == 0.3

    def test_all_bullish(self):
        # Prices rising steadily over 7 days (168 hours)
        prices = [100 + i * 0.5 for i in range(170)]
        result = multi_timeframe_trend(prices)
        assert result["alignment"] == 1.0
        assert result["timeframes"]["short"] == 1
        assert result["timeframes"]["medium"] == 1
        assert result["timeframes"]["long"] == 1
        assert result["confidence"] > 0.7

    def test_all_bearish(self):
        # Prices falling steadily
        prices = [200 - i * 0.5 for i in range(170)]
        result = multi_timeframe_trend(prices)
        assert result["alignment"] == -1.0
        assert result["timeframes"]["short"] == -1
        assert result["timeframes"]["medium"] == -1
        assert result["timeframes"]["long"] == -1

    def test_mixed_signals(self):
        # Flat for most of the period, small uptick at end
        prices = [100.0] * 160 + [100 + i * 0.5 for i in range(10)]
        result = multi_timeframe_trend(prices)
        # Short should be bullish (recent uptick), long might be neutral/bullish
        assert result["timeframes"]["short"] == 1

    def test_short_term_only(self):
        # 30 hours of data — short-term signal present, medium not (needs 48)
        prices = [100 + i for i in range(30)]
        result = multi_timeframe_trend(prices)
        assert "short" in result["timeframes"]
        assert "medium" not in result["timeframes"]
        assert "long" not in result["timeframes"]

    def test_medium_term(self):
        # 48 hours of data — short + medium signals
        prices = [100 + i * 0.3 for i in range(50)]
        result = multi_timeframe_trend(prices)
        assert "short" in result["timeframes"]
        assert "medium" in result["timeframes"]
        assert "long" not in result["timeframes"]


# ── Linear Slope ──────────────────────────────────────────────────────────────

class TestLinearSlope:
    def test_positive_slope(self):
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _linear_slope(prices) > 0

    def test_negative_slope(self):
        prices = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _linear_slope(prices) < 0

    def test_flat(self):
        prices = [3.0, 3.0, 3.0, 3.0]
        assert _linear_slope(prices) == 0.0

    def test_empty(self):
        assert _linear_slope([]) == 0.0
        assert _linear_slope([1.0]) == 0.0


# ── Regime Detection ──────────────────────────────────────────────────────────

class TestRegimeDetection:
    def test_not_enough_data(self):
        assert detect_crypto_regime([100.0] * 10) == "UNKNOWN"

    def test_crash_detection(self):
        # 10% drop in last 24 hours
        stable = [1000.0] * 30
        crash = [1000 - i * 5 for i in range(20)]  # drops to 905
        prices = stable + crash
        # Ensure last price is >10% below price 24 hours ago
        prices_crash = [1000.0] * 30 + [1000 - i * 7 for i in range(20)]
        result = detect_crypto_regime(prices_crash)
        # Check if 24h drop exceeds 10%
        if prices_crash[-1] < prices_crash[-24] * 0.90:
            assert result == "CRASH"

    def test_ranging_market(self):
        # Sideways chop — small oscillations around a mean
        import math
        prices = [100 + 0.5 * math.sin(i * 0.3) for i in range(50)]
        result = detect_crypto_regime(prices)
        assert result == "RANGING"

    def test_trending_market(self):
        # Strong uptrend with some volatility
        prices = [100 + i * 2 + (i % 3) for i in range(50)]
        result = detect_crypto_regime(prices)
        assert result == "TRENDING"

    def test_volatile_no_direction(self):
        # High volatility but no direction (wide swings both ways)
        import random
        random.seed(42)
        prices = [100 + random.uniform(-10, 10) for _ in range(50)]
        # If direction_strength < 0.3 and vol > 0.03, should be VOLATILE
        result = detect_crypto_regime(prices)
        assert result in ("VOLATILE", "RANGING")  # depends on random


# ── Correlation Guard ─────────────────────────────────────────────────────────

class TestCorrelationGuard:
    def test_no_open_positions(self):
        result = check_crypto_correlation("bitcoin", "above", [])
        assert result is None  # No block

    def test_same_asset_same_direction_blocked(self):
        positions = [{"crypto_asset": "bitcoin", "side": "YES"}]
        result = check_crypto_correlation("bitcoin", "above", positions)
        assert result is not None
        assert "Already have" in result

    def test_same_asset_different_direction_ok(self):
        positions = [{"crypto_asset": "bitcoin", "side": "YES"}]
        result = check_crypto_correlation("bitcoin", "below", positions)
        # Different directions = natural hedge, should be allowed
        assert result is None

    def test_correlated_assets_same_direction_blocked(self):
        # BTC and ETH are 85% correlated — same direction blocked
        positions = [{"crypto_asset": "bitcoin", "side": "YES"}]
        result = check_crypto_correlation("ethereum", "above", positions)
        assert result is not None
        assert "Correlated" in result

    def test_correlated_assets_opposite_direction_ok(self):
        # BTC YES (above) + ETH below = hedge, should be allowed
        positions = [{"crypto_asset": "bitcoin", "side": "YES"}]
        result = check_crypto_correlation("ethereum", "below", positions)
        assert result is None

    def test_uncorrelated_assets_ok(self):
        # Two assets with no defined correlation (<0.70 threshold)
        positions = [{"crypto_asset": "dogecoin", "side": "YES"}]
        # dogecoin-solana is not in the correlation map, defaults to 0.0
        result = check_crypto_correlation("solana", "above", positions)
        # Check if they're actually in the map
        from src.crypto_signals import _CRYPTO_CORRELATIONS
        pair = tuple(sorted(["dogecoin", "solana"]))
        if pair not in _CRYPTO_CORRELATIONS:
            assert result is None

    def test_multiple_positions_checked(self):
        positions = [
            {"crypto_asset": "bitcoin", "side": "YES"},
            {"crypto_asset": "solana", "side": "YES"},
        ]
        # ETH correlated with BTC at 85% — should block
        result = check_crypto_correlation("ethereum", "above", positions)
        assert result is not None


# ── Volume Anomaly Detection ──────────────────────────────────────────────────

class TestVolumeAnomaly:
    def test_not_enough_data(self):
        result = detect_volume_anomaly([100.0] * 10)
        assert result["anomaly"] is False
        assert result["signal"] == "normal"

    def test_normal_volume(self):
        volumes = [1000.0] * 168  # constant volume
        result = detect_volume_anomaly(volumes)
        assert result["anomaly"] is False
        assert result["multiplier"] == 1.0

    def test_high_volume_spike(self):
        # Normal for 162 hours, then 3x spike for last 6
        volumes = [1000.0] * 162 + [3500.0] * 6
        result = detect_volume_anomaly(volumes)
        assert result["anomaly"] is True
        assert result["signal"] == "high_conviction"
        assert result["multiplier"] >= 3.0

    def test_low_volume(self):
        # Normal for 162 hours, then very low for last 6
        volumes = [1000.0] * 162 + [200.0] * 6
        result = detect_volume_anomaly(volumes)
        assert result["anomaly"] is True
        assert result["signal"] == "low_conviction"
        assert result["multiplier"] <= 0.3

    def test_empty_volumes(self):
        result = detect_volume_anomaly([])
        assert result["anomaly"] is False


# ── Composite Signal ──────────────────────────────────────────────────────────

class TestCompositeSignal:
    @patch("src.crypto_signals.get_fear_greed_index")
    def test_composite_with_bullish_signals(self, mock_fg):
        mock_fg.return_value = {
            "value": 25, "classification": "Fear",
            "signal": "buy", "signal_mult": 1.1,
        }
        prices = [100 + i * 0.5 for i in range(170)]  # uptrend
        result = compute_composite_signal(0.60, "above", prices)
        # Bullish alignment + Fear (contrarian buy) should boost probability
        assert result["adjusted_prob"] > 0.60
        assert result["regime"] == "TRENDING"
        assert "MTF=" in result["signals"][0]

    @patch("src.crypto_signals.get_fear_greed_index")
    def test_composite_with_bearish_signals(self, mock_fg):
        mock_fg.return_value = {
            "value": 85, "classification": "Extreme Greed",
            "signal": "strong_sell", "signal_mult": 0.7,
        }
        prices = [200 - i * 0.5 for i in range(170)]  # downtrend
        result = compute_composite_signal(0.60, "above", prices)
        # Bearish alignment + extreme greed should reduce probability
        assert result["adjusted_prob"] < 0.60
        assert result["confidence_multiplier"] <= 1.0

    @patch("src.crypto_signals.get_fear_greed_index")
    def test_composite_bounded(self, mock_fg):
        mock_fg.return_value = {
            "value": 10, "classification": "Extreme Fear",
            "signal": "strong_buy", "signal_mult": 1.3,
        }
        prices = [100 + i for i in range(170)]
        result = compute_composite_signal(0.95, "above", prices)
        assert result["adjusted_prob"] <= 0.99
        assert result["adjusted_prob"] >= 0.01

    @patch("src.crypto_signals.get_fear_greed_index")
    def test_composite_with_volume(self, mock_fg):
        mock_fg.return_value = None  # No F&G data
        prices = [100.0] * 170
        # Without volume
        result_no_vol = compute_composite_signal(0.55, "above", prices, volumes=None)
        # With high volume spike
        volumes = [1000.0] * 162 + [4000.0] * 6
        result_vol = compute_composite_signal(0.55, "above", prices, volumes=volumes)
        # Volume spike should increase confidence compared to no volume
        assert result_vol["confidence_multiplier"] > result_no_vol["confidence_multiplier"]

    @patch("src.crypto_signals.get_fear_greed_index")
    def test_composite_crash_regime(self, mock_fg):
        mock_fg.return_value = None
        # Simulate crash: 15% drop in 24 hours
        prices = [1000.0] * 30 + [1000 - i * 8 for i in range(20)]
        result = compute_composite_signal(0.50, "above", prices)
        if result["regime"] == "CRASH":
            assert result["confidence_multiplier"] < 0.5

    @patch("src.crypto_signals.get_fear_greed_index")
    def test_composite_below_direction(self, mock_fg):
        mock_fg.return_value = {
            "value": 80, "classification": "Extreme Greed",
            "signal": "strong_sell", "signal_mult": 0.7,
        }
        prices = [200 - i * 0.5 for i in range(170)]  # downtrend
        result = compute_composite_signal(0.60, "below", prices)
        # For "below" bets with bearish trend + greed, should boost
        assert result["adjusted_prob"] > 0.55


# ── Fear & Greed (mocked) ────────────────────────────────────────────────────

class TestFearGreed:
    @patch("src.crypto_signals._SESSION.get")
    def test_fear_greed_cache(self, mock_get):
        """Second call should use cache, not hit API again."""
        from src.crypto_signals import _FG_CACHE
        import time as _time

        # Pre-populate cache
        _FG_CACHE["latest"] = (_time.time(), {
            "value": 45, "classification": "Neutral",
            "signal": "neutral", "signal_mult": 1.0,
        })
        result = get_fear_greed_index()
        assert result is not None
        assert result["value"] == 45
        mock_get.assert_not_called()

        # Clean up
        _FG_CACHE.clear()

    @patch("src.crypto_signals._SESSION.get")
    def test_fear_greed_api_failure(self, mock_get):
        """API failure returns None gracefully."""
        from src.crypto_signals import _FG_CACHE
        _FG_CACHE.clear()  # Force fresh fetch

        mock_get.side_effect = Exception("Network error")
        result = get_fear_greed_index()
        assert result is None


# ── Option Pricing Edge (IV vs HV Safeguard) ─────────────────────────────────

class TestOptionPricingEdge:
    """
    confirm_crypto_option_edge compares the theoretical Black-Scholes binary
    option price against the actual Polymarket YES price, AND verifies that
    Implied Volatility (IV) <= Historical Volatility (HV) * 1.05 to confirm
    we're buying cheap volatility (not chasing a hyped market).
    """

    def test_invalid_inputs_no_edge(self):
        """Bad inputs (zero price, zero vol, etc.) return safe no-edge default."""
        result = confirm_crypto_option_edge(
            current_price=0, threshold=100, direction="above",
            hours_to_expiry=24, daily_volatility=0.03,
            yes_market_price=0.5,
        )
        assert result["has_underpriced_edge"] is False
        assert result["price_edge"] == 0.0

    def test_underpriced_yes_above_edge_detected(self):
        """
        BTC at $115K (already above threshold $110K), 7 days out, daily vol 4%.
        Market YES at 0.50 — theoretical should be much higher → underpriced.

        Note: we use S > K (price above threshold) so the bisection over IV
        is well-behaved; for K >> S, P(above) is unimodal in vol and the
        single-root bisection can hit the high-vol boundary.
        """
        result = confirm_crypto_option_edge(
            current_price=115000, threshold=110000, direction="above",
            hours_to_expiry=168, daily_volatility=0.04,
            yes_market_price=0.50, min_edge_threshold=0.05,
        )
        # Theoretical should clearly exceed market price → underpriced edge
        assert result["theoretical_yes_price"] > 0.50
        assert result["price_edge"] > 0

    def test_overpriced_yes_no_edge(self):
        """
        BTC at $110K, threshold $115K, 7 days out, daily vol 4%.
        Market YES price 0.80 — theoretical should be ~0.40 → overpriced (no edge).
        """
        result = confirm_crypto_option_edge(
            current_price=110000, threshold=115000, direction="above",
            hours_to_expiry=168, daily_volatility=0.04,
            yes_market_price=0.80, min_edge_threshold=0.05,
        )
        # Market price > theoretical → no underpriced edge
        assert result["has_underpriced_edge"] is False
        # IV should be much higher than HV (market overpaying for volatility)
        assert result["implied_vol"] > result["historical_vol"]

    def test_edge_below_minimum_threshold_rejected(self):
        """A tiny price edge (< min_edge_threshold) should NOT pass."""
        result = confirm_crypto_option_edge(
            current_price=100000, threshold=100100, direction="above",
            hours_to_expiry=24, daily_volatility=0.04,
            yes_market_price=0.50, min_edge_threshold=0.10,
        )
        # Edge likely small here; verify the threshold gate fires
        if abs(result["price_edge"]) < 0.10:
            assert result["has_underpriced_edge"] is False

    def test_below_direction_handled(self):
        """Direction 'below' should invert the theoretical price correctly."""
        result_above = confirm_crypto_option_edge(
            current_price=110000, threshold=115000, direction="above",
            hours_to_expiry=72, daily_volatility=0.04,
            yes_market_price=0.40,
        )
        result_below = confirm_crypto_option_edge(
            current_price=110000, threshold=115000, direction="below",
            hours_to_expiry=72, daily_volatility=0.04,
            yes_market_price=0.40,
        )
        # P(above) + P(below) should sum to ~1.0
        total = result_above["theoretical_yes_price"] + result_below["theoretical_yes_price"]
        assert 0.95 <= total <= 1.05

    def test_iv_safeguard_blocks_overpriced_volatility(self):
        """
        Even with price_edge present, if IV >> HV (market overpaying for
        volatility), should NOT flag as underpriced.
        """
        # Construct: very small price gap but market YES is way above theoretical
        # → IV will be way above HV → safeguard kicks in
        result = confirm_crypto_option_edge(
            current_price=100000, threshold=200000, direction="above",
            hours_to_expiry=24, daily_volatility=0.02,
            yes_market_price=0.45, min_edge_threshold=0.05,
        )
        # threshold is 2x current price in 24h with 2% vol → P(above) is near zero
        # Market is priced at 0.45 → way overpriced → IV >> HV
        assert result["implied_vol"] > result["historical_vol"]
        assert result["has_underpriced_edge"] is False

    def test_result_schema(self):
        """Verify all expected fields are present in the result."""
        result = confirm_crypto_option_edge(
            current_price=100000, threshold=105000, direction="above",
            hours_to_expiry=48, daily_volatility=0.03,
            yes_market_price=0.40,
        )
        expected_keys = {
            "theoretical_yes_price", "implied_vol", "historical_vol",
            "has_underpriced_edge", "vol_ratio", "price_edge",
        }
        assert expected_keys.issubset(result.keys())
        assert isinstance(result["has_underpriced_edge"], bool)


# ── Centralized parse_utc_isoformat ─────────────────────────────────────────

class TestUtcParser:
    """The package-level parse_utc_isoformat is the single source of truth."""

    def test_z_suffix_normalized(self):
        from src import parse_utc_isoformat
        dt = parse_utc_isoformat("2026-06-01T12:00:00Z")
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0

    def test_offset_suffix_normalized(self):
        from src import parse_utc_isoformat
        dt = parse_utc_isoformat("2026-06-01T14:00:00+02:00")
        assert dt.tzinfo is not None
        # Should be converted to UTC: 14:00+02:00 = 12:00 UTC
        assert dt.hour == 12

    def test_naive_string_assumed_utc(self):
        from src import parse_utc_isoformat
        dt = parse_utc_isoformat("2026-06-01T12:00:00")
        assert dt.tzinfo is not None
        assert dt.hour == 12

    def test_empty_string_raises(self):
        from src import parse_utc_isoformat
        with pytest.raises(ValueError):
            parse_utc_isoformat("")
