"""Ensemble-derived risk aggregates for one live storm.

Two aggregates over the a-deck's ensemble members (GEFS + ECMWF-ENS + AI):

**Strike probability by coastal county** — for each county centroid, the
fraction of ensemble members whose track passes within a threshold distance
(default 60 nm, roughly the operational R64 envelope of a mature hurricane).
This is the pre-loss underwriting signal an insurer actually acts on: not
"where is the mean track" but "how many independent models put damaging
wind at this county". A single deterministic model can be wildly wrong;
20+ agreeing members is an operational commitment.

**Intensity spread** — per lead-time min / mean / max wind kt across
ensemble members. Tight spread at T+72 = models agree on strength;
wide spread = one of the ensembles is landing a Cat 4 while another
lands a TS, and the pricing spread on any exposed portfolio is enormous.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .atcf_adecks import ENSEMBLE_FAMILIES, ModelTrack
from .hurricane_impact import CountyMeta, county_centroids

# Default landfall-strike threshold — 60 nautical miles is close to the R64
# envelope of a mature hurricane (Cat 3+), so a county within 60 nm of a
# forecast track will most likely see damaging wind. Configurable per call.
DEFAULT_STRIKE_THRESHOLD_NM = 60.0

# Only consider ensemble members that reach at least this lead time when
# computing strike probability — a member that dropped out at T+24 cannot
# be counted as evidence for or against a T+72 landfall.
MIN_LEAD_HOURS = 24

# Standard operational lead-time buckets the frontend renders as columns.
INTENSITY_LEAD_BUCKETS: tuple[int, ...] = (0, 12, 24, 36, 48, 72, 96, 120)


# ─────────────────────────── data classes ───────────────────────────


@dataclass(slots=True, frozen=True)
class CountyStrikeProb:
    """P(damaging wind) for one county from the ensemble.

    ``member_count`` is the number of ensemble members whose track passes
    within the threshold. ``max_intensity_kt`` is the peak wind of any
    passing member near the county — a rough upper-bound on what could hit.
    """

    geoid: str
    geography_id: str
    name: str
    state_usps: str
    centroid_lat: float
    centroid_lon: float
    strike_probability: float   # 0..1
    member_count: int           # members whose track passes within threshold
    ensemble_total: int         # total ensemble members considered
    max_intensity_kt: int       # peak wind kt of any passing member near the county


@dataclass(slots=True, frozen=True)
class IntensityStat:
    hours_out: int
    member_count: int
    min_kt: int
    mean_kt: float
    max_kt: int
    std_kt: float


@dataclass(slots=True)
class EnsembleRisk:
    """Aggregated risk views for one storm from one init cycle."""

    init_cycle: str | None
    ensemble_total: int
    threshold_nm: float
    strike_by_county: list[CountyStrikeProb] = field(default_factory=list)
    intensity_by_lead: list[IntensityStat] = field(default_factory=list)


# ─────────────────────────── math ───────────────────────────


_EARTH_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_KM * math.asin(math.sqrt(a))


def _point_to_segment_km(
    plat: float, plon: float,
    alat: float, alon: float,
    blat: float, blon: float,
) -> float:
    """Great-circle distance from point P to segment A→B (km), approximated on
    a small local tangent plane. Good enough for track-vs-county spacing at
    tropical latitudes where segments are ≤ ~150 nm.

    Falls back to the closer endpoint if the segment length is degenerate."""
    seg_km = _haversine_km(alat, alon, blat, blon)
    if seg_km < 1e-3:
        return _haversine_km(plat, plon, alat, alon)
    # Project P onto A→B in a local equirectangular approximation. Convert to
    # a metric plane centered on A.
    cos_lat = max(math.cos(math.radians(alat)), 0.05)
    ax = 0.0
    ay = 0.0
    bx = (blon - alon) * cos_lat * 111.0    # ~km per degree lon at this lat
    by = (blat - alat) * 111.0
    px = (plon - alon) * cos_lat * 111.0
    py = (plat - alat) * 111.0
    dx = bx - ax
    dy = by - ay
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-3:
        return _haversine_km(plat, plon, alat, alon)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def _passes_within(
    track: ModelTrack, county: CountyMeta, threshold_km: float,
) -> tuple[bool, int]:
    """Does ``track`` come within ``threshold_km`` of the county centroid at
    any point? Returns (yes/no, peak_wind_kt_of_close_fixes).

    A minimum lead threshold (MIN_LEAD_HOURS) filters out members that only
    have very-near-term fixes — a member that stops at T+12 doesn't tell us
    anything about a T+96 landfall.
    """
    if len(track.fixes) < 2:
        return False, 0
    if track.fixes[-1].hours_out < MIN_LEAD_HOURS:
        return False, 0
    peak = 0
    hit = False
    # Walk segments (fix i, fix i+1) so we don't miss a track that skips
    # a county between two 12-hour-apart fixes. Leads under 24 h are where
    # every member still sits on the storm. Counting them paints a blob on
    # the current position, west of a forecast that has already turned.
    for i in range(len(track.fixes) - 1):
        a = track.fixes[i]
        b = track.fixes[i + 1]
        if a.hours_out < MIN_LEAD_HOURS:
            continue
        d_km = _point_to_segment_km(
            county.centroid_lat, county.centroid_lon,
            a.lat, a.lon, b.lat, b.lon,
        )
        if d_km <= threshold_km:
            hit = True
            peak = max(peak, a.wind_kt, b.wind_kt)
    return hit, peak


# ─────────────────────────── public ───────────────────────────


def compute_ensemble_risk(
    tracks: list[ModelTrack],
    *,
    threshold_nm: float = DEFAULT_STRIKE_THRESHOLD_NM,
    include_families: frozenset[str] | None = None,
    coastal_states: frozenset[str] | None = None,
) -> EnsembleRisk:
    """Fold a-deck tracks into strike-probability + intensity aggregates.

    Args:
        tracks: All model tracks for the storm (from fetch_model_tracks).
        threshold_nm: Distance within which a track is considered a "strike"
            for a given county centroid. Default 60 nm ≈ operational R64
            envelope of a mature hurricane.
        include_families: Which model families to treat as ensemble members
            (default: GEFS members + ECMWF-ENS members + AI models). Passing
            a subset lets a caller compute AI-only strike probabilities, say.
        coastal_states: If set, restrict the county sweep to these USPS
            codes (e.g. Atlantic + Gulf hurricane-prone states). Cuts the
            county × member walk substantially — the full 3,000-county sweep
            is wasteful when the storm is 300 nm from Cuba.

    Returns:
        EnsembleRisk containing per-county strike probability + per-lead
        intensity stats. Empty aggregates when not enough ensemble members.
    """
    families = include_families or ENSEMBLE_FAMILIES
    members = [t for t in tracks if t.family in families]
    threshold_km = threshold_nm * 1.852

    if not members:
        return EnsembleRisk(
            init_cycle=None,
            ensemble_total=0,
            threshold_nm=threshold_nm,
        )

    # The first member can be a single newer AI track. The cycle that
    # owns the vote is the one most members actually come from.
    init_cycle = Counter(m.init_cycle for m in members).most_common(1)[0][0]
    counties = county_centroids()

    strikes: list[CountyStrikeProb] = []
    for geoid, county in counties.items():
        if coastal_states is not None and county.state_usps not in coastal_states:
            continue
        hits = 0
        peak_of_hits = 0
        for m in members:
            passed, peak = _passes_within(m, county, threshold_km)
            if passed:
                hits += 1
                peak_of_hits = max(peak_of_hits, peak)
        if hits == 0:
            continue
        strikes.append(
            CountyStrikeProb(
                geoid=geoid,
                geography_id=county.geography_id,
                name=county.name,
                state_usps=county.state_usps,
                centroid_lat=county.centroid_lat,
                centroid_lon=county.centroid_lon,
                strike_probability=round(hits / len(members), 3),
                member_count=hits,
                ensemble_total=len(members),
                max_intensity_kt=peak_of_hits,
            )
        )
    strikes.sort(key=lambda s: -s.strike_probability)

    # Intensity spread by lead time — grouped across all ensemble members.
    by_lead: dict[int, list[int]] = defaultdict(list)
    for m in members:
        for fix in m.fixes:
            if fix.wind_kt <= 0:
                continue
            if fix.hours_out in INTENSITY_LEAD_BUCKETS:
                by_lead[fix.hours_out].append(fix.wind_kt)

    intensity: list[IntensityStat] = []
    for h in INTENSITY_LEAD_BUCKETS:
        vals = by_lead.get(h, [])
        if len(vals) < 3:
            continue
        mean = sum(vals) / len(vals)
        # Population std is fine here — the ensemble IS the population we're
        # summarising, not a sample from a larger one.
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        intensity.append(
            IntensityStat(
                hours_out=h,
                member_count=len(vals),
                min_kt=min(vals),
                mean_kt=round(mean, 1),
                max_kt=max(vals),
                std_kt=round(math.sqrt(variance), 1),
            )
        )

    return EnsembleRisk(
        init_cycle=init_cycle,
        ensemble_total=len(members),
        threshold_nm=threshold_nm,
        strike_by_county=strikes,
        intensity_by_lead=intensity,
    )


# Hurricane-prone Atlantic + Gulf coastal states — used as the default
# county-sweep restriction so the aggregator doesn't pointlessly walk
# 3,000 US counties for a storm 300 nm from the Bahamas.
ATLANTIC_COASTAL_STATES: frozenset[str] = frozenset({
    "FL", "GA", "SC", "NC", "VA", "MD", "DE", "NJ", "NY", "CT",
    "RI", "MA", "NH", "ME", "AL", "MS", "LA", "TX", "PR", "VI",
})


__all__ = [
    "CountyStrikeProb",
    "IntensityStat",
    "EnsembleRisk",
    "DEFAULT_STRIKE_THRESHOLD_NM",
    "ATLANTIC_COASTAL_STATES",
    "compute_ensemble_risk",
]
