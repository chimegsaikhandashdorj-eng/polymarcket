# Sports False Positive Audit — Phase 1.5.4

Total sports false positives: **60**

## Trigger Breakdown

| Trigger | Count | Example Team/Keyword |
|---|---|---|
| `nhl:` prefix | 6 | Tampa Bay Lightning NHL game lines |
| `nba:` prefix | 0 | NBA games |
| `nfl:` prefix | 0 | NFL games |
| `wnba:` prefix | 2 | WNBA games |
| Tampa Bay Lightning | 1 | 'lightning' keyword |
| Carolina Hurricanes | 0 | 'hurricane' keyword |
| Miami Heat | 0 | 'heat' keyword |
| Stanley Cup | 2 | 'cup' keyword |
| Other (vs. pattern) | 49 | generic game lines |

## Blocklist Strategy

Apply at **scanner level** (before parsers). Check title against blocked phrases
using word-boundary regex. If any phrase matches, discard the market.

Key insight: `nhl:`, `nba:`, `nfl:` prefixes appear in **all** sports game lines
on Polymarket. A simple prefix check eliminates nearly all sports false positives.

## 20-Market Sample

| Title |
|---|
| Lightning vs. Bruins |
| Lightning vs. Golden Knights |
| Rangers vs. Lightning |
| Tampa Bay Lightning win the 2024 Stanley Cup? |
| Aces vs. Storm |
| Canadiens vs. Lightning |
| Lightning vs. Panthers |
| Valkyries vs. Storm |
| DePaul Blue Demons vs. St. John's Red Storm |
| Lightning vs. Ducks |
| Bruins vs. Lightning |
| Blackhawks vs. Lightning |
| NHL: Carolina Hurricanes vs. Tampa Bay Lightning 2023-03-28 |
| Will the Tampa Bay Lightning win the 2025 President’s Trophy? |
| Storm vs. Aces |
| Islanders vs. Lightning |
| Lightning vs. Bruins |
| Storm vs Valkyries |
| Lightning vs. Stars |
| Lightning vs. Maple Leafs |