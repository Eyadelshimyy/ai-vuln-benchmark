"""NWS active watches + warnings, filtered to what matters during a storm.

Single source: ``api.weather.gov/alerts/active`` (free, no auth, GeoJSON).
Useful event types for a hurricane-impact view:

  Hurricane Warning / Watch
  Tropical Storm Warning / Watch
  Storm Surge Warning / Watch
  Tornado Warning / Watch
  Flash Flood Warning / Emergency
  Coastal Flood Warning

NHC's coastal Tropical Cyclone Watch/Warning areas show up in this same
feed (issued by NHC and propagated to NWS) — no separate fetch needed.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

from ..brand import USER_AGENT as NWS_USER_AGENT


class AlertFeedUnavailable(RuntimeError):
    """The NWS alerts feed could not be reached or parsed.

    Distinct from "the feed returned nothing": callers must be able to tell an
    outage from a quiet day before telling a user there is no active weather.
    """


NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"
FETCH_TIMEOUT_S = 30

# Event types we render. Anything else (red flag, winter storm, ...) is
# ignored to keep the map readable during a hurricane.
RELEVANT_EVENTS: frozenset[str] = frozenset(
    {
        "Hurricane Warning", "Hurricane Watch",
        "Tropical Storm Warning", "Tropical Storm Watch",
        "Storm Surge Warning", "Storm Surge Watch",
        "Tornado Warning", "Tornado Watch",
        "Flash Flood Warning", "Flash Flood Emergency", "Flood Warning", "Flood Watch",
        "Coastal Flood Warning", "Coastal Flood Watch", "Coastal Flood Advisory",
        "Extreme Wind Warning",
        "High Wind Warning", "High Wind Watch",
        "Special Marine Warning",
        "Tropical Cyclone Statement",
    }
)

# Severity ordering used by the frontend for stacking (most severe = top).
SEVERITY_RANK = {"Extreme": 4, "Severe": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}


@dataclass(slots=True, frozen=True)
class WeatherAlert:
    alert_id: str
    event: str
    headline: str
    severity: str
    urgency: str
    certainty: str
    sent_at: str
    expires_at: str
    areas_affected: str   # human description (multiple counties / zones)
    geometry: dict | None  # GeoJSON geometry (Polygon / MultiPolygon) or None


def _bbox_intersects(geometry: dict | None, bbox: tuple[float, float, float, float] | None) -> bool:
    """Cheap overlap test: any geometry vertex inside bbox = include.
    Skipped (returns True) when no bbox supplied."""
    if bbox is None or geometry is None:
        return True
    west, south, east, north = bbox

    def _walk(coords):
        if not coords:
            return False
        # Polygon: list of rings; Ring: list of [lon, lat]; MultiPolygon: list of Polygons
        if isinstance(coords[0], (int, float)):
            lon, lat = coords[0], coords[1]
            return west <= lon <= east and south <= lat <= north
        return any(_walk(c) for c in coords)

    return _walk(geometry.get("coordinates"))


def fetch_active_alerts(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    states: list[str] | None = None,
    event_filter: frozenset[str] | None = RELEVANT_EVENTS,
) -> list[WeatherAlert]:
    """Pull active alerts. Either a bbox (lon/lat) OR a list of state codes
    (``["FL","GA","SC"]``) narrows the request. When both are given, the
    state-area filter is sent to NWS and the bbox is applied client-side."""
    params: dict[str, str] = {"status": "actual"}
    if states:
        params["area"] = ",".join(s.upper() for s in states)
    url = NWS_ALERTS_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        # Raise rather than return []. An empty list is indistinguishable from a
        # genuinely quiet feed, and a caller that caches it then reports "no
        # active alerts" — a confident all-clear during a live event. Callers
        # that want soft-fail must say so explicitly.
        raise AlertFeedUnavailable(str(exc)) from exc

    out: list[WeatherAlert] = []
    for f in data.get("features", []) or []:
        p = f.get("properties") or {}
        event = p.get("event") or ""
        if event_filter is not None and event not in event_filter:
            continue
        geom = f.get("geometry")
        if not _bbox_intersects(geom, bbox):
            continue
        out.append(
            WeatherAlert(
                alert_id=p.get("id") or f.get("id") or "",
                event=event,
                headline=p.get("headline") or "",
                severity=p.get("severity") or "Unknown",
                urgency=p.get("urgency") or "Unknown",
                certainty=p.get("certainty") or "Unknown",
                sent_at=p.get("sent") or "",
                expires_at=p.get("expires") or "",
                areas_affected=p.get("areaDesc") or "",
                geometry=geom,
            )
        )
    # Sort most-severe first so the map z-order works naturally.
    out.sort(key=lambda a: -SEVERITY_RANK.get(a.severity, 0))
    return out
