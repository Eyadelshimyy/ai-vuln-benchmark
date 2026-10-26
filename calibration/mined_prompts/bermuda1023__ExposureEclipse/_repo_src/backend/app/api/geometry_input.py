"""Shared wire types for "exposed TIV inside these polygons" endpoints.

Both the wildfire and flood overlays let a user select shapes on the map and
post them back for an exposure rollup, so they take byte-identical input. The
validation here is a DoS guard, not a formality — the rollup derives a bbox
from these numbers and walks a grid between the corners — so it lives in one
place rather than being copied per router, where one copy could be hardened
and the other left behind.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from ..models.common import CamelModel

# Caller-supplied geometry drives a grid walk and a point-in-polygon sweep, so
# the polygon count and total vertex count are capped here; the service applies
# a further (candidates × vertices) work budget that these alone can't express.
MAX_POLYGONS = 50
MAX_VERTICES = 100_000
# NHC watches and warnings are one polygon per coastal zone, not one shape per
# storm. A live event is ~150 polygons and can be several hundred when warnings
# and watches stack. The shared 50-cap is for a user picking fire or flood
# shapes; it rejects a full warning set before any TIV is computed. The work
# budget still prices the ray-cast.
MAX_WW_POLYGONS = 1_000


def rings_of(geom: dict) -> list:
    """Flatten a GeoJSON Polygon/MultiPolygon to its list of rings, validating
    structure. Raises ValueError on anything we would not want to walk."""
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Polygon":
        rings = coords
    elif gtype == "MultiPolygon":
        if not isinstance(coords, list):
            raise ValueError("coordinates must be a list")
        rings = [r for poly in coords for r in (poly if isinstance(poly, list) else [])]
    else:
        raise ValueError("geometry type must be Polygon or MultiPolygon")
    if not isinstance(rings, list) or not rings:
        raise ValueError("geometry has no rings")
    return rings


class PolygonIn(CamelModel):
    id: str = Field(max_length=200)
    name: str | None = Field(default=None, max_length=200)
    geometry: dict  # GeoJSON Polygon / MultiPolygon

    @field_validator("geometry")
    @classmethod
    def _validate_geometry(cls, v: dict) -> dict:
        # The rollup derives a bbox from these numbers and walks a 0.5° grid
        # between the corners, so unbounded coordinates are a DoS, not a typo.
        for ring in rings_of(v):
            if not isinstance(ring, list) or len(ring) < 4:
                raise ValueError("each ring needs at least 4 positions")
            for pos in ring:
                if not isinstance(pos, list) or len(pos) < 2:
                    raise ValueError("ring positions must be [lon, lat]")
                lon, lat = pos[0], pos[1]
                if isinstance(lon, bool) or isinstance(lat, bool):
                    raise ValueError("ring positions must be numbers")
                if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
                    raise ValueError("ring positions must be numbers")
                if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                    raise ValueError("coordinates must be within WGS84 bounds")
        return v


def _reject_oversized(polygons: list[PolygonIn]) -> list[PolygonIn]:
    total = sum(len(r) for p in polygons for r in rings_of(p.geometry))
    if total > MAX_VERTICES:
        raise ValueError(f"at most {MAX_VERTICES} vertices per request")
    return polygons


class ExposureRequest(CamelModel):
    polygons: list[PolygonIn] = Field(max_length=MAX_POLYGONS)

    @field_validator("polygons")
    @classmethod
    def _validate_budget(cls, v: list[PolygonIn]) -> list[PolygonIn]:
        return _reject_oversized(v)


class WatchWarnExposureRequest(CamelModel):
    """Same polygon shape as ExposureRequest, with room for a full NHC set."""

    polygons: list[PolygonIn] = Field(max_length=MAX_WW_POLYGONS)

    @field_validator("polygons")
    @classmethod
    def _validate_budget(cls, v: list[PolygonIn]) -> list[PolygonIn]:
        return _reject_oversized(v)


class ClientExposureOut(CamelModel):
    client: str
    tiv: float
    location_count: int


class PolygonExposureOut(CamelModel):
    id: str
    name: str | None
    total_tiv: float
    location_count: int
    by_client: list[ClientExposureOut]


def exposure_out(pid: str, name: str | None, rollup: tuple) -> PolygonExposureOut:
    total, count, by_client = rollup
    return PolygonExposureOut(
        id=pid,
        name=name,
        total_tiv=round(total, 2),
        location_count=count,
        by_client=sorted(
            [ClientExposureOut(client=c, tiv=round(t, 2), location_count=n)
             for c, (t, n) in by_client.items()],
            key=lambda x: -x.tiv,
        ),
    )


__all__ = [
    "MAX_POLYGONS",
    "MAX_WW_POLYGONS",
    "MAX_VERTICES",
    "rings_of",
    "PolygonIn",
    "ExposureRequest",
    "WatchWarnExposureRequest",
    "ClientExposureOut",
    "PolygonExposureOut",
    "exposure_out",
]
