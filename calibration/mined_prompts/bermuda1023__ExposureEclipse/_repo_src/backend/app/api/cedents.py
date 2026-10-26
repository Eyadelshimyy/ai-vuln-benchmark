"""Cedent tree endpoints — primary navigation API.

  GET /api/cedents              → full tree (Cedent → ProgrammeChain → Programme)
  GET /api/cedents/{id}         → one cedent
  GET /api/chains/{id}          → one chain
  GET /api/programmes/{id}      → one programme
  GET /api/programmes/{id}/status → ERT status of the programme's underlying EDM

Replaces the flat `/api/datasets` surface from the pre-cedent design. The chain
is the unit of YoY comparison (latest year auto-pairs with prior).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..models.cedent import Cedent, CedentTreeResponse, Programme, ProgrammeChain
from ..models.dataset import DatasetStatusResponse
from ..models.enums import ErrorCode
from ..providers import ExposureDataProvider, get_provider

router = APIRouter(tags=["cedents"])


@router.get("/cedents", response_model=CedentTreeResponse)
def list_cedents(
    provider: ExposureDataProvider = Depends(get_provider),
) -> CedentTreeResponse:
    return CedentTreeResponse(cedents=provider.list_cedents())


@router.get("/cedents/{cedent_id}", response_model=Cedent)
def get_cedent(
    cedent_id: str,
    provider: ExposureDataProvider = Depends(get_provider),
) -> Cedent:
    cedent = provider.get_cedent(cedent_id)
    if cedent is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": ErrorCode.DATASET_NOT_FOUND.value,
                "message": f"Cedent '{cedent_id}' was not found.",
                "details": {"cedentId": cedent_id},
            },
        )
    return cedent


@router.get("/chains/{chain_id}", response_model=ProgrammeChain)
def get_chain(
    chain_id: str,
    provider: ExposureDataProvider = Depends(get_provider),
) -> ProgrammeChain:
    chain = provider.get_chain(chain_id)
    if chain is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": ErrorCode.DATASET_NOT_FOUND.value,
                "message": f"Programme chain '{chain_id}' was not found.",
                "details": {"chainId": chain_id},
            },
        )
    return chain


@router.get("/programmes/{programme_id}", response_model=Programme)
def get_programme(
    programme_id: str,
    provider: ExposureDataProvider = Depends(get_provider),
) -> Programme:
    programme = provider.get_programme(programme_id)
    if programme is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": ErrorCode.DATASET_NOT_FOUND.value,
                "message": f"Programme '{programme_id}' was not found.",
                "details": {"programmeId": programme_id},
            },
        )
    return programme


@router.get("/programmes/{programme_id}/status", response_model=DatasetStatusResponse)
def programme_status(
    programme_id: str,
    provider: ExposureDataProvider = Depends(get_provider),
) -> DatasetStatusResponse:
    programme = provider.get_programme(programme_id)
    if programme is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": ErrorCode.DATASET_NOT_FOUND.value,
                "message": f"Programme '{programme_id}' was not found.",
                "details": {"programmeId": programme_id},
            },
        )
    # status is keyed off the dataset_id (the legacy id used by exposure_facts).
    return provider.get_dataset_status(programme.dataset_id)


__all__ = ["router"]
