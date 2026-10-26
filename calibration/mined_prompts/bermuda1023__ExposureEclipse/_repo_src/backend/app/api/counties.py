"""County reference data — population, household counts, avg insured home
cost. Mock data only; in production sourced from US Census + Marshall & Swift.

Used by the right-rail detail panel and the hurricane-impact roll-up to give
the underwriter a sense-check baseline (e.g. "60% of impacted county housing
is on the coast — expect concentration").
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..services.county_reference import get_reference_by_geography_id
from ..services.industry_book import county_industry_tiv

router = APIRouter(prefix="/counties", tags=["counties"])


@router.get("/{geography_id}/reference")
def county_reference(geography_id: str) -> dict:
    """Return mock reference stats for a county. ``geography_id`` may be the
    canonical ``US-FL-12086`` or a raw 5-digit GEOID (``12086``)."""
    ref = get_reference_by_geography_id(geography_id)
    if ref is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "DATASET_NOT_FOUND",
                "message": f"County '{geography_id}' was not found in the reference index.",
                "details": {"geographyId": geography_id},
            },
        )
    book = county_industry_tiv(geography_id)
    return {
        "geoid": ref.geoid,
        "state": ref.state_usps,
        "population": ref.population,
        "households": ref.households,
        "avgReplacementCost": ref.avg_replacement_cost,
        "avgInsuredValue": ref.avg_insured_value,
        "coastalExposurePct": ref.coastal_exposure_pct,
        "source": ref.source,
        "currency": "USD",
        # Same dollars as the Industry client. Null when the county is not
        # on the supplied windstorm sheet.
        "industryResidentialTiv": None if book is None else book[0],
        "industryCommercialTiv": None if book is None else book[1],
        "industryAutoTiv": None if book is None else book[2],
        "industryTiv": None if book is None else book[3],
    }
