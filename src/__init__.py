"""
Polymarket trading bot package.

Exposes shared utilities (timezone-safe ISO parsing, etc.) that are used
across multiple submodules. Centralizing here avoids code duplication and
ensures one canonical implementation per helper.
"""

from datetime import datetime, timezone


def parse_utc_isoformat(dt_str: str) -> datetime:
    """
    Safely convert an ISO format datetime string to a UTC-aware ``datetime``.

    Polymarket and external APIs return timestamps in inconsistent formats:
    some include a trailing ``Z`` (Zulu), some include explicit ``+00:00``
    offsets, and some legacy fields are naive (no offset at all). Comparing
    naive and aware datetimes raises ``TypeError`` in Python, so every
    timestamp the bot consumes must be normalized.

    Behavior:
      - ``"2026-06-01T12:00:00Z"``          -> aware UTC datetime
      - ``"2026-06-01T12:00:00+00:00"``     -> aware UTC datetime
      - ``"2026-06-01T12:00:00"``           -> assumed UTC, returned as aware
      - ``"2026-06-01T14:00:00+02:00"``     -> converted to UTC

    Raises:
        ValueError: if ``dt_str`` is empty or cannot be parsed.
    """
    if not dt_str:
        raise ValueError("Empty datetime string")
    normalized = dt_str.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        # Treat naive timestamps as UTC (Polymarket's documented default).
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # Normalize to UTC so downstream subtraction is unambiguous.
        dt = dt.astimezone(timezone.utc)
    return dt


__all__ = ["parse_utc_isoformat"]
