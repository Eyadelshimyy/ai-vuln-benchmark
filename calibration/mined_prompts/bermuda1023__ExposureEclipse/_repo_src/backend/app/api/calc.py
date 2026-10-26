"""Calculation endpoints. v1: layer-loss scenarios for reinsurance XOL stacks.

Live consumer is the (future) frontend "what-if" panel where an underwriter
sees deterministic payouts at a series of damage ratios for the currently
selected programme. Endpoint stays generic — caller supplies layers +
scenarios; we don't pull from Programme records yet because programme-level
layer terms aren't in the mock data model.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import Field

from ..models.common import CamelModel
from ..services.layer_calc import (
    DEFAULT_SWEEP_DAMAGE_RATIOS,
    LayerTerms,
    run_scenario,
    run_sweep,
)

router = APIRouter(prefix="/calc", tags=["calc"])


# ─────────────────────────── wire types ───────────────────────────


class LayerTermsIn(CamelModel):
    deductible: float = Field(ge=0)
    limit: float = Field(gt=0)
    share: float = Field(default=1.0, ge=0.0, le=1.0)
    name: str | None = None


class ScenarioIn(CamelModel):
    """Either ``gross_loss`` OR (``tiv`` + ``damage_ratio``)."""

    gross_loss: float | None = Field(default=None, ge=0)
    tiv: float | None = Field(default=None, ge=0)
    damage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    label: str | None = None


class LayerOutcomeOut(CamelModel):
    name: str | None
    deductible: float
    limit: float
    share: float
    loss_to_layer: float
    ceded_loss: float
    exhausted: bool


class ScenarioResultOut(CamelModel):
    label: str | None
    tiv: float | None
    damage_ratio: float | None
    ground_up_loss: float
    layers: list[LayerOutcomeOut]
    total_ceded: float
    cedent_net_loss: float


class LayerCalcRequest(CamelModel):
    layers: list[LayerTermsIn]
    scenarios: list[ScenarioIn] = []
    # When `scenarios` is empty and `sweep_tiv` is set, we run the default
    # damage-ratio sweep instead — that's the typical "give me a payout curve"
    # case. Both can be combined: explicit scenarios plus the sweep appended.
    sweep_tiv: float | None = Field(default=None, ge=0)


class LayerCalcResponse(CamelModel):
    scenarios: list[ScenarioResultOut]


# ─────────────────────────── endpoint ───────────────────────────


def _to_terms(layers_in: list[LayerTermsIn]) -> list[LayerTerms]:
    try:
        return [
            LayerTerms(
                deductible=l.deductible,
                limit=l.limit,
                share=l.share,
                name=l.name,
            )
            for l in layers_in
        ]
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "message": str(exc),
            },
        ) from exc


def _to_out(result) -> ScenarioResultOut:
    return ScenarioResultOut(
        label=result.label,
        tiv=result.tiv,
        damage_ratio=result.damage_ratio,
        ground_up_loss=result.ground_up_loss,
        layers=[
            LayerOutcomeOut(
                name=o.name,
                deductible=o.deductible,
                limit=o.limit,
                share=o.share,
                loss_to_layer=o.loss_to_layer,
                ceded_loss=o.ceded_loss,
                exhausted=o.exhausted,
            )
            for o in result.layers
        ],
        total_ceded=result.total_ceded,
        cedent_net_loss=result.cedent_net_loss,
    )


@router.post("/layers", response_model=LayerCalcResponse)
def calc_layers(payload: LayerCalcRequest) -> LayerCalcResponse:
    """Run deterministic loss scenarios through a stack of XOL layers."""
    if not payload.layers:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": "layers must be non-empty."},
        )
    if not payload.scenarios and payload.sweep_tiv is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "message": "Provide at least one scenario or set sweepTiv to run the default damage-ratio sweep.",
            },
        )
    terms = _to_terms(payload.layers)

    results = []
    for s in payload.scenarios:
        try:
            results.append(
                run_scenario(
                    terms,
                    gross_loss=s.gross_loss,
                    tiv=s.tiv,
                    damage_ratio=s.damage_ratio,
                    label=s.label,
                )
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "VALIDATION_ERROR", "message": str(exc)},
            ) from exc

    if payload.sweep_tiv is not None:
        results.extend(run_sweep(terms, payload.sweep_tiv, DEFAULT_SWEEP_DAMAGE_RATIOS))

    return LayerCalcResponse(scenarios=[_to_out(r) for r in results])


__all__ = ["router"]
