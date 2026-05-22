"""Unit tests for crypto price prediction module."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from src.crypto_fetcher import (
    estimate_prob_above,
    _compute_volatility,
    _compute_momentum,
    _compute_rsi,
    CryptoEnsemble,
    SUPPORTED_ASSETS,
    SYMBOL_ALIASES,
)
from src.market_scanner import parse_market_condition, CRYPTO_ABOVE, CRYPTO_BELOW


# ── Market title parsing ───────────────────────────────────────────────────────

class TestCryptoParsing:
    def test_btc_above(self):
        r = parse_market_condition("Will Bitcoin be above $120,000 on June 1?")
        assert r["metric"] == CRYPTO_ABOVE
        assert r["crypto_asset"] == "bitcoin"
        assert r["threshold"] == 120000

    def test_eth_exceed(self):
        r = parse_market_condition("Will ETH exceed $5,000 by end of May?")
        assert r["metric"] == CRYPTO_ABOVE
        assert r["crypto_asset"] == "ethereum"
        assert r["threshold"] == 5000

    def test_sol_below(self):
        r = parse_market_condition("Will Solana drop below $100 this week?")
        assert r["metric"] == CRYPTO_BELOW
        assert r["crypto_asset"] == "solana"
        assert r["threshold"] == 100

    def test_price_first_pattern(self):
        r = parse_market_condition("$150,000 Bitcoin by July?")
        assert r["metric"] == CRYPTO_ABOVE
        assert r["crypto_asset"] == "bitcoin"
        assert r["threshold"] == 150000

    def test_doge_above(self):
        r = parse_market_condition("Will DOGE hit $1?")
        assert r["metric"] == CRYPTO_ABOVE
        assert r["crypto_asset"] == "dogecoin"
        assert r["threshold"] == 1

    def test_non_crypto_not_matched(self):
        r = parse_market_condition("Will it rain in NYC tomorrow?")
        assert r["metric"] != CRYPTO_ABOVE
        assert r["metric"] != CRYPTO_BELOW
        assert r["crypto_asset"] is None

    def test_xrp_below(self):
        r = parse_market_condition("Will XRP be under $2 by Friday?")
        assert r["metric"] == CRYPTO_BELOW
        assert r["crypto_asset"] == "xrp"
        assert r["threshold"] == 2


# ── Probability estimation ─────────────────────────────────────────────────────

class TestCryptoProb:
    def test_above_when_price_far_above_threshold(self):
        # BTC at 110K, threshold 80K, should be very likely above
        p = estimate_prob_above(110000, 80000, 48, 0.04, {"trend": 0, "strength": 0, "rsi": 50})
        assert p > 0.90

    def test_above_when_price_far_below_threshold(self):
        # BTC at 50K, threshold 100K, should be very unlikely
        p = estimate_prob_above(50000, 100000, 48, 0.04, {"trend": 0, "strength": 0, "rsi": 50})
        assert p < 0.10

    def test_above_at_threshold(self):
        # Price == threshold, should be ~50%
        p = estimate_prob_above(100000, 100000, 48, 0.04, {"trend": 0, "strength": 0, "rsi": 50})
        assert 0.35 < p < 0.65

    def test_momentum_bias_bullish(self):
        p_neutral = estimate_prob_above(100000, 110000, 48, 0.04,
                                        {"trend": 0, "strength": 0, "rsi": 50})
        p_bull = estimate_prob_above(100000, 110000, 48, 0.04,
                                     {"trend": 1, "strength": 0.8, "rsi": 65})
        # Bullish momentum should increase probability
        assert p_bull > p_neutral

    def test_momentum_bias_bearish(self):
        p_neutral = estimate_prob_above(100000, 110000, 48, 0.04,
                                        {"trend": 0, "strength": 0, "rsi": 50})
        p_bear = estimate_prob_above(100000, 110000, 48, 0.04,
                                     {"trend": -1, "strength": 0.8, "rsi": 35})
        # Bearish momentum should decrease probability
        assert p_bear < p_neutral

    def test_higher_vol_wider_distribution(self):
        # High vol -> more uncertain -> probabilities closer to 0.5
        p_low_vol = estimate_prob_above(100000, 120000, 48, 0.02,
                                        {"trend": 0, "strength": 0, "rsi": 50})
        p_high_vol = estimate_prob_above(100000, 120000, 48, 0.08,
                                         {"trend": 0, "strength": 0, "rsi": 50})
        # High vol means higher chance of reaching distant threshold
        assert p_high_vol > p_low_vol

    def test_more_time_higher_prob(self):
        # More time -> more volatility accumulation -> more chance of reaching threshold
        p_short = estimate_prob_above(100000, 120000, 12, 0.04,
                                      {"trend": 0, "strength": 0, "rsi": 50})
        p_long = estimate_prob_above(100000, 120000, 168, 0.04,
                                     {"trend": 0, "strength": 0, "rsi": 50})
        assert p_long > p_short

    def test_prob_bounded(self):
        # Should always be between 0.01 and 0.99
        p = estimate_prob_above(1, 1000000, 1, 0.01, {"trend": 0, "strength": 0, "rsi": 50})
        assert 0.01 <= p <= 0.99
        p2 = estimate_prob_above(1000000, 1, 1, 0.01, {"trend": 0, "strength": 0, "rsi": 50})
        assert 0.01 <= p2 <= 0.99


# ── Technical indicators ───────────────────────────────────────────────────────

class TestTechnicals:
    def test_volatility_zero_for_constant_prices(self):
        prices = [100.0] * 24
        assert _compute_volatility(prices) == 0.0

    def test_volatility_positive_for_varying_prices(self):
        # Simulate some price movement
        prices = [100 + i * 0.5 for i in range(48)]
        vol = _compute_volatility(prices)
        assert vol > 0

    def test_volatility_short_list(self):
        assert _compute_volatility([100.0]) == 0.0
        assert _compute_volatility([]) == 0.0

    def test_momentum_bullish(self):
        # Prices trending up
        prices = [100 + i for i in range(30)]
        m = _compute_momentum(prices)
        assert m["trend"] == 1
        assert m["strength"] > 0

    def test_momentum_bearish(self):
        # Prices trending down
        prices = [100 - i for i in range(30)]
        m = _compute_momentum(prices)
        assert m["trend"] == -1

    def test_rsi_overbought(self):
        # All gains -> RSI near 100
        prices = [100 + i * 2 for i in range(20)]
        rsi = _compute_rsi(prices)
        assert rsi > 70

    def test_rsi_oversold(self):
        # All losses -> RSI near 0
        prices = [100 - i * 2 for i in range(20)]
        rsi = _compute_rsi(prices)
        assert rsi < 30

    def test_rsi_neutral(self):
        # Alternating gains and losses -> RSI near 50
        prices = [100 + (1 if i % 2 == 0 else -1) for i in range(20)]
        rsi = _compute_rsi(prices)
        assert 40 < rsi < 60


# ── Symbol aliases ─────────────────────────────────────────────────────────────

class TestAliases:
    def test_btc_alias(self):
        assert SYMBOL_ALIASES["btc"] == "bitcoin"
        assert SYMBOL_ALIASES["bitcoin"] == "bitcoin"

    def test_eth_alias(self):
        assert SYMBOL_ALIASES["eth"] == "ethereum"
        assert SYMBOL_ALIASES["ether"] == "ethereum"

    def test_supported_assets(self):
        assert "bitcoin" in SUPPORTED_ASSETS
        assert "ethereum" in SUPPORTED_ASSETS
        assert "solana" in SUPPORTED_ASSETS
        assert SUPPORTED_ASSETS["bitcoin"]["symbol"] == "BTC"
