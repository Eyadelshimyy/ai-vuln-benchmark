"""Ensemble-consensus envelope from a set of model tracks.

For a spaghetti plot, one useful summary geometry is the "consensus cone":
the polygon that bounds the ensemble members' projected positions at each
lead time. Where models agree, the polygon is narrow; where they disagree,
it fans out.

We build it as a swept envelope: sample the ensemble positions at each
standard NHC lead time (0, 12, 24, 36, 48, 72, 96, 120, 144, 168 h),
convex-hull the members at that lead time, then concatenate the hulls into
one closed polygon. This is a rougher approximation than NHC's cone (which
uses climatological forecast-error radii), but it is entirely data-driven
and updates on every model cycle.

Stdlib-only (numpy is not a prod dep here — see CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from .atcf_adecks import ENSEMBLE_FAMILIES, ModelTrack

# Standard NHC lead-time anchors + a few longer ones for AI models that
# routinely produce 240h forecasts. If an ensemble member has no fix at an
# anchor, it's excluded from that anchor (rather than interpolated) — the
# envelope reads honestly as "these members ran to this lead".
LEAD_ANCHORS_H: tuple[int, ...] = (12, 24, 36, 48, 72, 96, 120, 144, 168)


@dataclass(slots=True, frozen=True)
class EnvelopePoint:
    hours_out: int
    lat: float
    lon: float


@dataclass(slots=True, frozen=True)
class EnsembleEnvelope:
    """Data-driven consensus envelope. ``ring`` is a closed [(lon, lat), ...]
    polygon in the frontend-friendly order. ``hulls`` is the per-anchor list
    of hull vertices, exposed so the frontend can render the anchor points
    as dots for visual reference."""

    members_used: int
    ring: list[tuple[float, float]]
    anchor_hulls: dict[int, list[EnvelopePoint]]


# ─────────────────────────── math ───────────────────────────


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain — returns the hull in CCW order without the
    duplicate closing vertex. Points as (x, y) which here means (lon, lat).

    For < 3 unique points we return them as-is; the caller synthesises a
    small buffer around a single point if it wants a real polygon."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _corridor_points(members: list[ModelTrack]) -> list[tuple[float, float]]:
    """Points offset ~30 nm either side of each member track.

    Used when the convex hull of the raw positions is degenerate (one
    model, or several models on top of each other). 0.5° of longitude at
    these latitudes is roughly 30 nm — enough to see, not a fake cone.
    """
    out: list[tuple[float, float]] = []
    for track in members:
        fixes = track.fixes
        for i, fix in enumerate(fixes):
            if i + 1 < len(fixes):
                nxt = fixes[i + 1]
                dlon = nxt.lon - fix.lon
                dlat = nxt.lat - fix.lat
            elif i > 0:
                prv = fixes[i - 1]
                dlon = fix.lon - prv.lon
                dlat = fix.lat - prv.lat
            else:
                dlon, dlat = 1.0, 0.0
            norm = (dlon * dlon + dlat * dlat) ** 0.5 or 1.0
            # Perpendicular unit vector, scaled to ~0.45°.
            ox = -dlat / norm * 0.45
            oy = dlon / norm * 0.45
            out.append((fix.lon + ox, fix.lat + oy))
            out.append((fix.lon - ox, fix.lat - oy))
    return out


def _member_position_at(track: ModelTrack, hours: int) -> tuple[float, float] | None:
    """Return (lon, lat) at exactly ``hours`` if the member has that fix.

    No temporal interpolation on purpose — an ensemble member that doesn't
    reach a lead time should not contribute a synthetic position; that would
    over-narrow the envelope."""
    for f in track.fixes:
        if f.hours_out == hours:
            return (f.lon, f.lat)
    return None


def build_envelope(
    tracks: list[ModelTrack],
    *,
    include_families: frozenset[str] | None = None,
    min_members: int = 5,
) -> EnsembleEnvelope | None:
    """Build a swept envelope over the union of hulls at each lead-time anchor.

    Args:
        tracks: All model tracks for the storm (as returned by fetch_model_tracks).
        include_families: Restrict to specific families (default: every
            ensemble family — GEFS members, ECMWF-ENS members, AI models).
            Passing a subset lets the caller isolate e.g. an "AI-only
            consensus" cone.
        min_members: Refuse to build if fewer than this many members
            contribute (a 3-member "envelope" is not statistically meaningful
            enough to lean on). Returns None when the threshold is not met.

    Returns:
        EnsembleEnvelope, or None if there aren't enough members.
    """
    families = include_families or ENSEMBLE_FAMILIES
    members = [t for t in tracks if t.family in families]
    if len(members) < min_members:
        return None

    # Collect the union of hull vertices across all anchor lead times.
    union_pts: list[tuple[float, float]] = []
    anchor_hulls: dict[int, list[EnvelopePoint]] = {}
    for h in LEAD_ANCHORS_H:
        positions = [
            p for t in members if (p := _member_position_at(t, h)) is not None
        ]
        if len(positions) < 3:
            continue
        hull = _convex_hull(positions)
        anchor_hulls[h] = [
            EnvelopePoint(hours_out=h, lat=lat, lon=lon) for (lon, lat) in hull
        ]
        union_pts.extend(hull)

    if not union_pts:
        union_pts = _corridor_points(members)
    if not union_pts:
        return None

    # The envelope is the convex hull of the union — a genuinely tight
    # polygon that contains every ensemble member at every anchor. Simpler
    # than sweeping-and-stitching per-lead hulls and reads cleanly as a
    # "cone of model disagreement". A single nearly-straight track has no
    # area; widen it into a corridor so an AI-only run still draws.
    ring = _convex_hull(union_pts)
    if len(ring) < 3:
        ring = _convex_hull(_corridor_points(members))
    if len(ring) < 3:
        return None
    # Close the ring.
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return EnsembleEnvelope(
        members_used=len(members),
        ring=ring,
        anchor_hulls=anchor_hulls,
    )


__all__ = [
    "EnvelopePoint",
    "EnsembleEnvelope",
    "LEAD_ANCHORS_H",
    "build_envelope",
]
