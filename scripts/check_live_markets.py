"""Check live Polymarket weather markets."""
import requests
import json

def main():
    # Try events API with weather tag
    r = requests.get("https://gamma-api.polymarket.com/events", params={
        "active": "true",
        "closed": "false",
        "limit": 50,
        "tag": "weather",
    }, timeout=15)
    data = r.json()
    if isinstance(data, list) and data:
        print(f"Events with tag=weather: {len(data)}")
        for e in data[:10]:
            print(f"  Slug: {e.get('slug','')[:80]}")
            print(f"  Title: {e.get('title','')[:80]}")
            print()
    else:
        print("No events with tag=weather found, trying slug search...")

    # Try searching by known weather slug pattern
    for slug_query in ["highest-temperature", "rain-in", "will-it-rain", "temperature-in"]:
        r2 = requests.get("https://gamma-api.polymarket.com/events", params={
            "active": "true",
            "closed": "false",
            "limit": 20,
            "slug": slug_query,
        }, timeout=15)
        d2 = r2.json()
        if isinstance(d2, list) and d2:
            print(f"Slug query '{slug_query}': {len(d2)} events")
            for e in d2[:5]:
                print(f"  {e.get('title','')[:80]}")

    # Check current active markets for "highest temperature" style
    r3 = requests.get("https://gamma-api.polymarket.com/markets", params={
        "active": "true",
        "closed": "false",
        "limit": 100,
        "offset": 0,
        "tag_slug": "weather",
    }, timeout=15)
    d3 = r3.json()
    if isinstance(d3, list):
        weather_markets = [m for m in d3 if any(
            kw in (m.get("question") or m.get("title") or "").lower()
            for kw in ["temperature", "rain", "snow", "hurricane"]
        )]
        print(f"\nMarkets with tag_slug=weather: {len(d3)} total, {len(weather_markets)} weather-related")
        for m in weather_markets[:10]:
            title = m.get("question") or m.get("title") or ""
            end = (m.get("endDate",""))[:10]
            vol = float(m.get("volume", 0) or 0)
            print(f"  {end}  vol={vol:8.0f}  {title[:80]}")

if __name__ == "__main__":
    main()
