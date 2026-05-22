# NYC Alias Patterns — Phase 1.5.3

Total markets missed by main `_MULTI_CITY_RE` because they use 'nyc' alias: **174**

## Root Cause

`_MULTI_CITY_RE` matches `'new york'` but NOT `'nyc'`. The live bot's `_CITY_ALIASES` dict
handles `nyc` correctly, but the market analyzer and _MULTI_CITY_RE do not.

## Alias Frequency

| Alias | Count |
|---|---|
| `nyc` | 174 |

## Resolution Note (IMPORTANT)

Polymarket NYC weather markets resolve based on **Central Park (KNYC) weather station**.
When forecasting for NYC markets, use KNYC coordinates: lat=40.7794, lon=-73.9692.
Do NOT use NYC downtown/airport coordinates.

## Fix Required

**In `scripts/analyze_markets.py`**: Add `nyc` to `_MULTI_CITY_RE`:

```python
_MULTI_CITY_RE = re.compile(
    r"\b(nyc|new york|new york city|los angeles|chicago|miami|boston|houston|"
    r"atlanta|seattle|denver|dallas|phoenix|philadelphia|washington|"
    r"london|tokyo|sydney|paris|berlin)\b",
    re.I,
)
```

**In `src/market_scanner.py`**: Already handled via `_CITY_ALIASES`. No change needed.

## 30-Market Sample

| Title | Volume |
|---|---|
| Will the highest temperature in NYC be 25°F or below on January 23? | $18,965 |
| Will the highest temperature in NYC be between 53-54°F on April 1? | $9,643 |
| Will the highest temperature in NYC be 83°F or higher on March 29? | $9,797 |
| Will the highest temperature in NYC be between 32-33°F on January 24? | $940 |
| Will the highest temperature in NYC be between 49-50°F on March 30? | $6,161 |
| Will the highest temperature in NYC be 52°F or higher on January 29? | $7,966 |
| Will the highest temperature in NYC be 53°F or higher on March 23? | $11,693 |
| Will the highest temperature in NYC be between 40-41°F on January 30? | $6,250 |
| Will the highest temperature in NYC be between 33-34°F on February 2? | $28,595 |
| Will the highest temperature in NYC be between 49-50°F on March 27? | $3,417 |
| Will the highest temperature in NYC be 55°F or below on March 28? | $10,231 |
| Will the highest temperature in NYC be between 54-55°F on March 26? | $7,767 |
| Will the highest temperature in NYC be between 53-54°F on March 27? | $8,006 |
| Will the highest temperature in NYC be 55°F or higher on March 30? | $4,588 |
| Will the highest temperature in NYC be 42°F or below on February 3? | $8,171 |
| Will the highest temperature in NYC be between 34-35°F on January 30? | $4,228 |
| Will the highest temperature in NYC be between 62-63°F on March 22? | $15,491 |
| Will the highest temperature in NYC be between 36-37°F on January 24? | $108 |
| Will the highest temperature in NYC be between 42-43°F on January 31? | $6,097 |
| Will the highest temperature in NYC be 44°F or higher on January 30? | $8,462 |
| Will the highest temperature in NYC be between 55-56°F on March 24? | $13,999 |
| Will the highest temperature in NYC be between 48-49°F on March 26? | $4,806 |
| Will NYC have between 3 and 4 inches of precipitation in December? | $72,966 |
| Will the highest temperature in NYC be 72°F or higher on March 31? | $11,484 |
| Will the highest temperature in NYC be between 46-47°F on January 31? | $15,698 |
| Will the highest temperature in NYC be between 33-34°F on January 25? | $3,807 |
| Will the highest temperature in NYC be between 55-56°F on March 25? | $12,444 |
| Will the highest temperature in NYC be 26°F or below on January 25? | $5,005 |
| Will the highest temperature in NYC be between 75-76°F on March 29? | $1,696 |
| Will the highest temperature in NYC be between 30-31°F on January 24? | $731 |