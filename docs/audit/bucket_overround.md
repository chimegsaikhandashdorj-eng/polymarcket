# Bucket Overround / Underround Distribution — Phase 1.5.5

## Data Limitation

Historical closed markets have `lastTradePrice` = 0 or 1 (the resolution/settlement price),
NOT the pre-resolution probability price. Computing overround from settlement prices is
meaningless. Pre-resolution price snapshots require the CLOB API `/prices-history` endpoint,
which is not queried in Phase 1 discovery.

**Resolution**: Overround analysis must be done on **live markets** during Phase 2.
The real-time scanner will fetch current bid/ask prices and compute group overround before
entering any trade. See Phase 3 spec for implementation.

---

## Group Structure Analysis (Available from Historical Data)

Total bucket groups (by `negRiskMarketID`): **103**

| Metric | Value |
|---|---|
| Groups found | 103 |
| Min buckets per group | 1 |
| Median buckets per group | 7 |
| Max buckets per group | 7 |
| Groups with 5+ buckets | 88 |
| Groups with 7 buckets (full range) | 72 |

| Complete groups (exactly 1 YES, all resolved) | 93/103 (90%) |

---

## Volume Distribution Analysis (Proxy for Market Confidence)

Normalized Shannon entropy of volume distribution across buckets:
- **0.0** = all volume concentrated on 1 bucket (market very confident)
- **1.0** = volume evenly spread (market highly uncertain)

| Metric | Value |
|---|---|
| Groups analyzed (vol > $1,000) | 103 |
| Median normalized entropy | 0.917 |
| Low entropy (< 0.5, market confident) | 11 (11%) |
| High entropy (> 0.8, market uncertain) | 82 (80%) |

**Interpretation**: A low-entropy group means market participants strongly agreed
on one bucket. High-entropy means uncertainty was spread — more opportunity for
us to disagree with the market on a specific bucket.

---

## Real-Time Overround Plan (Phase 2 Implementation)

During live scanning, compute group overround as:

```python
def compute_overround(bucket_group: list[Market]) -> float:
    """Sum of YES mid-prices for a neg-risk group. Should be ~1.0."""
    mid_prices = [(m.best_bid + m.best_ask) / 2 for m in bucket_group]
    return sum(mid_prices) - 1.0  # 0.0 = fair, < 0 = underround (arb), > 0 = overround
```

Gate: If `overround < -0.02`, check for pure arbitrage before normal EV-based trading.
If `overround > 0.05`, apply higher EV threshold (fees eat more edge).