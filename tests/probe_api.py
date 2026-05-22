import requests

BASE = "https://gamma-api.polymarket.com/markets"
tests = [
    {"closed": "true", "limit": 3, "startDate": "2026-04-01"},
    {"closed": "true", "limit": 3, "end_date_min": "2026-04-01"},
    {"closed": "true", "limit": 3, "endDateMin": "2026-04-01"},
    {"closed": "true", "limit": 3, "tag_slug": "weather"},
    {"closed": "true", "limit": 3, "offset": 45000},   # skip to near end
]
for p in tests:
    try:
        r = requests.get(BASE, params=p, timeout=20)
        data = r.json()
        if isinstance(data, list) and data:
            first_date = (data[0].get("endDate") or "?")[:10]
            last_date  = (data[-1].get("endDate") or "?")[:10]
            print(f"OK  {p} => {len(data)} markets  first={first_date}  last={last_date}")
        else:
            print(f"EMPTY  {p}")
    except Exception as e:
        print(f"ERR  {p} => {e}")
