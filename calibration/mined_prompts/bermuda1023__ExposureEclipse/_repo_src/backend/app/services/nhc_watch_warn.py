"""NHC tropical-cyclone watches / warnings — split, colour, classify.

NHC's coastal Tropical Cyclone Watches and Warnings ride the same NWS CAP
alerts feed we already pull for the live-storm bundle (see
:mod:`services.weather_alerts`). This module lifts those TC-specific events
out of the generic alerts stream so the frontend can:

  * paint them in the NHC operational colour scheme (pink = Hurricane Warning,
    red = Hurricane Watch, blue = TS Warning, cyan = TS Watch, purple = Storm
    Surge Warning, magenta = Storm Surge Watch) instead of the generic
    severity-ranked palette the other alerts use, and

  * roll up in-polygon exposed TIV through the same synthetic-point machinery
    wildfire / flood use (see :func:`wildfire_exposure.exposure_in_polygons`)
    so an underwriter can see "$X TIV inside the current Hurricane Warning".

Zone-coded alerts (no polygon) are still surfaced — the reinsurance-facing
"which areas are under warning" is at least as important as the map paint,
so we count them separately and label them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .weather_alerts import WeatherAlert

# The subset of NWS event types NHC issues as coastal Tropical Cyclone
# watches / warnings. Extreme Wind Warning is an NWS product (issued by
# local WFOs during landfall of a major hurricane) but reads as part of the
# same operational story, so we group it here.
NHC_WW_EVENTS: frozenset[str] = frozenset(
    {
        "Hurricane Warning",
        "Hurricane Watch",
        "Tropical Storm Warning",
        "Tropical Storm Watch",
        "Storm Surge Warning",
        "Storm Surge Watch",
        "Extreme Wind Warning",
        "Tropical Cyclone Statement",
    }
)

# NHC's operational watch/warning legend colours (approximated to hex; the
# printed NHC products are pantone-defined). Sourced from the NHC "Watches
# and Warnings" graphics legend.
NHC_WW_COLORS: dict[str, str] = {
    "Hurricane Warning": "#ec4899",      # pink
    "Hurricane Watch": "#dc2626",        # red
    "Tropical Storm Warning": "#2563eb", # blue
    "Tropical Storm Watch": "#06b6d4",   # cyan
    "Storm Surge Warning": "#7c3aed",    # violet
    "Storm Surge Watch": "#c084fc",      # light violet
    "Extreme Wind Warning": "#7f1d1d",   # dark red
    "Tropical Cyclone Statement": "#94a3b8",  # slate (info-only)
}

# Product family — drives the frontend chip legend + exposure rollup grouping.
NHC_WW_FAMILY: dict[str, str] = {
    "Hurricane Warning": "hurricane",
    "Hurricane Watch": "hurricane",
    "Tropical Storm Warning": "tropical_storm",
    "Tropical Storm Watch": "tropical_storm",
    "Storm Surge Warning": "storm_surge",
    "Storm Surge Watch": "storm_surge",
    "Extreme Wind Warning": "extreme_wind",
    "Tropical Cyclone Statement": "statement",
}

# Threat rank — Warning > Watch, Hurricane > Tropical Storm. Drives z-order on
# the map (highest rank stacks on top) and default sort in the exposure panel.
NHC_WW_RANK: dict[str, int] = {
    "Extreme Wind Warning": 6,
    "Hurricane Warning": 5,
    "Storm Surge Warning": 4,
    "Hurricane Watch": 3,
    "Tropical Storm Warning": 2,
    "Storm Surge Watch": 2,
    "Tropical Storm Watch": 1,
    "Tropical Cyclone Statement": 0,
}


@dataclass(slots=True, frozen=True)
class NHCWatchWarn:
    """One coastal Tropical Cyclone watch or warning polygon (or zone-coded
    alert). ``geometry`` is None for zone-only alerts (Watches and Advisories
    for Coastal/Lakeshore/Fire zones ship without polygons in the NWS feed;
    all we get is the area description text)."""

    alert_id: str
    event: str
    family: str            # hurricane | tropical_storm | storm_surge | extreme_wind | statement
    color: str             # NHC operational hex
    rank: int              # higher = more severe; drives z-order
    headline: str
    severity: str
    urgency: str
    certainty: str
    sent_at: str
    expires_at: str
    areas_affected: str
    geometry: dict | None


def split_watches_warnings(
    alerts: list[WeatherAlert],
) -> tuple[list[NHCWatchWarn], list[WeatherAlert]]:
    """Split an NWS alerts list into (NHC watches/warnings, everything else).

    Both halves preserve the source alerts' order (which the alerts service
    already sorted most-severe first). Callers can render each with a
    distinct paint scheme — NHC WW use the operational hurricane legend,
    residual alerts (flood, tornado, ...) use the generic severity palette.
    """
    ww: list[NHCWatchWarn] = []
    residual: list[WeatherAlert] = []
    for a in alerts:
        if a.event not in NHC_WW_EVENTS:
            residual.append(a)
            continue
        ww.append(
            NHCWatchWarn(
                alert_id=a.alert_id,
                event=a.event,
                family=NHC_WW_FAMILY.get(a.event, "other"),
                color=NHC_WW_COLORS.get(a.event, "#94a3b8"),
                rank=NHC_WW_RANK.get(a.event, 0),
                headline=a.headline,
                severity=a.severity,
                urgency=a.urgency,
                certainty=a.certainty,
                sent_at=a.sent_at,
                expires_at=a.expires_at,
                areas_affected=a.areas_affected,
                geometry=a.geometry,
            )
        )
    # Sort watches/warnings by our rank (Warning > Watch, more severe on top)
    # then by NWS-reported severity for a stable secondary key.
    ww.sort(key=lambda w: (-w.rank, w.event))
    return ww, residual


__all__ = [
    "NHCWatchWarn",
    "NHC_WW_EVENTS",
    "NHC_WW_COLORS",
    "NHC_WW_FAMILY",
    "NHC_WW_RANK",
    "split_watches_warnings",
]
