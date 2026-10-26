"""Live flood overlay — active NWS flood watches, warnings and advisories.

Single upstream: ``api.weather.gov/alerts/active``, already wrapped by
:mod:`app.services.weather_alerts` for the hurricane view. This module narrows
that feed to flood products, splits the polygon-bearing alerts from the
zone-coded ones, and rolls up affected states.

Why only alerts in v1: the fast-moving flood products (Flood Advisory, Flash
Flood Warning, Flood Warning) carry real polygons — a live sample returned
19 of 20 with geometry — so they drop straight into the exposure engine.
NOAA's National Water Model inundation service publishes modelled water extent
at higher precision, but it is flagged EXPERIMENTAL, covers ~30% of the US
population, excludes coastal processes, and serves Esri JSON that needs ring
conversion. It is planned as an additive second layer, not a replacement.

Alerts are *warning areas*, not observed water. They are frequently drawn to
county or zone boundaries, so exposed TIV derived from them is an upper bound
on the affected area — surfaced in the UI rather than silently implied.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .weather_alerts import AlertFeedUnavailable, WeatherAlert, fetch_active_alerts

# Flood products only. Deliberately excludes Hydrologic Outlook (too weak to
# map as an event) and Tsunami (a different peril with a different response).
FLOOD_EVENTS: frozenset[str] = frozenset(
    {
        "Flood Warning", "Flood Watch", "Flood Advisory", "Flood Statement",
        "Flash Flood Warning", "Flash Flood Watch", "Flash Flood Statement",
        "Coastal Flood Warning", "Coastal Flood Watch", "Coastal Flood Advisory",
        "Lakeshore Flood Warning", "Lakeshore Flood Watch", "Lakeshore Flood Advisory",
    }
)

# NWS CAP severity vocabulary, ordered. Used for the "major flooding only"
# filter — the NWS `severity` field is the only severity signal that arrives
# with the geometry, so it is what the map ramp and the filter both key on.
SEVERITY_ORDER: tuple[str, ...] = ("Unknown", "Minor", "Moderate", "Severe", "Extreme")
SEVERITY_RANK: dict[str, int] = {s: i for i, s in enumerate(SEVERITY_ORDER)}

CACHE_TTL_S = 60.0
_CACHE_MAX_ENTRIES = 32
# NWS asks for considerate polling and both the map layer and the side panel
# hit this endpoint. 60s is well inside the rate alerts actually change.
# Cached BEFORE the severity floor is applied, so switching floors filters an
# already-held list instead of re-pulling the ~1.3MB nationwide feed each time.
_CACHE: dict[tuple, tuple[float, list[WeatherAlert]]] = {}


@dataclass(slots=True, frozen=True)
class FloodBundle:
    alerts: list[WeatherAlert]      # polygon-bearing only — safe to map/intersect
    zone_only_count: int            # matched the filter but carried no geometry
    affected_states: list[tuple[str, int]]
    notes: list[str]


def _cache_get(key: tuple) -> list[WeatherAlert] | None:
    hit = _CACHE.get(key)
    if hit is None:
        return None
    stamped, value = hit
    if time.monotonic() - stamped > CACHE_TTL_S:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_put(key: tuple, value: list[WeatherAlert]) -> None:
    if len(_CACHE) >= _CACHE_MAX_ENTRIES:
        # Evict the oldest by timestamp; the key space here is small (a handful
        # of bbox/state combinations) so a full scan is cheaper than an LRU.
        oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
        _CACHE.pop(oldest, None)
    _CACHE[key] = (time.monotonic(), value)


def _states_from_area(area_desc: str) -> set[str]:
    """State codes out of an NWS areaDesc like ``"Houston, TN; Stewart, TN"``."""
    out: set[str] = set()
    for part in area_desc.split(";"):
        _, _, tail = part.rpartition(",")
        code = tail.strip().upper()
        if len(code) == 2 and code.isalpha():
            out.add(code)
    return out


def fetch_flood_alerts(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    states: list[str] | None = None,
    min_severity: str = "Unknown",
    events: frozenset[str] | None = None,
) -> FloodBundle:
    """Active flood alerts, split into mappable and zone-only.

    Args:
        bbox: Optional ``(west, south, east, north)`` in lon/lat. Applied
            client-side against alert geometry by the underlying client.
        states: Optional state codes narrowing the upstream request.
        min_severity: Drop alerts below this NWS CAP severity. One of
            ``Unknown | Minor | Moderate | Severe | Extreme``.
        events: Override the flood event set (defaults to FLOOD_EVENTS).

    Returns:
        FloodBundle whose ``alerts`` contain only polygon-bearing alerts.
        Zone-coded alerts are counted, not returned — they have no geometry to
        map or intersect, and silently dropping them would understate the
        picture without explanation.
    """
    floor = SEVERITY_RANK.get(min_severity, 0)
    # The floor is NOT part of the key — it is a pure filter over the cached
    # list, so moving the severity chip re-filters instead of re-fetching.
    key = (bbox, tuple(states) if states else None, events or FLOOD_EVENTS)

    degraded = False
    raw = _cache_get(key)
    if raw is None:
        try:
            raw = fetch_active_alerts(
                bbox=bbox, states=states, event_filter=events or FLOOD_EVENTS
            )
        except AlertFeedUnavailable:
            # Deliberately NOT cached. Caching an outage would hold a false
            # all-clear for the full TTL, and "no alerts" is the one answer that
            # must never be guessed during a flood.
            degraded = True
            raw = []
        else:
            _cache_put(key, raw)

    kept = [a for a in raw if SEVERITY_RANK.get(a.severity, 0) >= floor]
    mappable = [a for a in kept if a.geometry is not None]
    zone_only = len(kept) - len(mappable)

    counts: dict[str, int] = {}
    for a in mappable:
        for st in _states_from_area(a.areas_affected):
            counts[st] = counts.get(st, 0) + 1
    affected = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    notes: list[str] = []
    if zone_only:
        notes.append(
            f"{zone_only} flood alert(s) are zone-coded and carry no polygon, so they "
            f"are not mapped or included in exposed TIV. Coastal, Lakeshore and most "
            f"Watch products are issued this way."
        )
    if degraded:
        notes.append(
            "The NWS alert feed could not be reached, so no flood alerts could be "
            "loaded. This is NOT a statement that there is no flooding — treat the "
            "map as empty, not clear, and retry."
        )
    elif not mappable:
        notes.append("No active flood alerts with mappable geometry right now.")

    return FloodBundle(
        alerts=mappable,
        zone_only_count=zone_only,
        affected_states=affected,
        notes=notes,
    )


__all__ = [
    "FLOOD_EVENTS",
    "SEVERITY_ORDER",
    "SEVERITY_RANK",
    "FloodBundle",
    "fetch_flood_alerts",
]
