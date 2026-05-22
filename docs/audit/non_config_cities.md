# Non-Configured City Temperature Markets — Phase 1.5.2

## Summary

The 64 'non-config city' markets from the initial estimate broke down as:

| Category | Count | Notes |
|---|---|---|
| Global climate anomaly markets | ~62 | NASA/NOAA global mean temp, NOT local weather |
| Miscellaneous cities | ~2 | Too few to qualify (< 15 markets) |

## Decision: No New Cities to Add

**None of the non-configured cities qualify** under the acceptance criteria:
- Minimum: 15 markets AND median liquidity >= $5,000

The bulk of 'non-config city' markets are global climate anomaly markets,
not local city weather markets. These reference NASA global mean temperature data
and cannot be traded with local weather API forecasts.

## Global Climate Markets (62)

Sample titles:

| Title |
|---|
| Will June 2024 have a temperature increase of less than 1.09°C? |
| Will June 2024 have a temperature increase of between 1.09°C and 1.15°C? |
| Will June 2024 have a temperature increase of between 1.16°C and 1.22°C? |
| Will June 2024 have a temperature increase of between 1.23°C and 1.29°C? |
| Will June 2024 have a temperature increase of greater than 1.29°C? |
| August temperature increase by less than 1.15°C? |
| August temperature increase by between 1.15-1.19°C? |
| August temperature increase by between 1.20-1.24°C? |
| August temperature increase by between 1.25-1.29°C? |
| August temperature increase greater than 1.29°C? |
| Will the October 2024 temperature increase be less than 1.17°C? |
| Will the October 2024 temperature increase be between 1.17-1.22°C? |
| Will the October 2024 temperature increase be between 1.23-1.28°C? |
| Will the October 2024 temperature increase be between 1.29-1.34°C? |
| Will the October 2024 temperature increase be between 1.35-1.40°C? |
| Will the October 2024 temperature increase be greater than 1.40°C? |
| Will the November 2024 temperature increase be less than 1.20°C? |
| Will the November 2024 temperature increase be between 1.20-1.24°C? |
| Will the November 2024 temperature increase be between 1.25-1.29°C? |
| Will the November 2024 temperature increase be between 1.30-1.34°C? |

## Implementation Note

If global temperature anomaly markets are ever in scope, they would require:
- NASA GISS Surface Temperature Analysis API
- Monthly resolution (not daily)
- Different probability model entirely

**Not in scope for Phase 2.**