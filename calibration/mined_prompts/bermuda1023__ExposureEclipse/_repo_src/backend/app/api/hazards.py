"""Hazard-overlay endpoint — smooth lat/lon grids of hail / tornado /
wildfire risk. Sampled at ~0.4° over CONUS and clipped to US land so the
choropleth doesn't paint the ocean. Each cell carries raw + normalised
value; the legend metadata travels with the response so the UI can render
a colour bar with the publishing-agency attribution."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models.common import CamelModel
from ..services.hazard_overlay import HazardType, build_grid

router = APIRouter(prefix="/hazards", tags=["hazards"])


class HazardGridPointOut(CamelModel):
    lat: float
    lon: float
    raw: float
    normalised: float


class HazardLegendOut(CamelModel):
    title: str
    unit: str
    source: str
    source_url: str
    raw_min: float
    raw_max: float
    palette: list[str]
    stops: list[float]
    note: str | None = None


class HazardResponse(CamelModel):
    hazard: str
    grid: list[HazardGridPointOut]
    step_deg: float
    legend: HazardLegendOut


_VALID: tuple[HazardType, ...] = ("tornado", "hail", "wildfire")


@router.get("/{hazard}", response_model=HazardResponse)
def get_hazard(hazard: str) -> HazardResponse:
    if hazard not in _VALID:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "message": f"Unknown hazard '{hazard}'. Valid: {_VALID}.",
            },
        )
    grid, legend, step = build_grid(hazard)  # type: ignore[arg-type]
    return HazardResponse(
        hazard=hazard,
        grid=[
            HazardGridPointOut(
                lat=p.lat, lon=p.lon, raw=p.raw, normalised=p.normalised,
            )
            for p in grid
        ],
        step_deg=step,
        legend=HazardLegendOut(
            title=legend.title,
            unit=legend.unit,
            source=legend.source,
            source_url=legend.source_url,
            raw_min=legend.raw_min,
            raw_max=legend.raw_max,
            palette=legend.palette,
            stops=legend.stops,
            note=legend.note,
        ),
    )


__all__ = ["router"]
