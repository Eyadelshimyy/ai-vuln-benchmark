"""Dataset-group endpoints (API_SPEC.md §Dataset Group APIs).

Group creation enforces three things at the router boundary, per CLAUDE.md:

1. **Currency consistency** — mixed-currency members blow up with
   ``409 CURRENCY_MISMATCH`` (rule 5: never silently mix currencies).
2. **SUM_DISTINCT_SEGMENTS gate** — only allowed when the user explicitly
   confirms the members are distinct exposure segments (rule 3 — never
   silently sum across peril EDMs).
3. **Multi-peril warning** — when ``MAX_ACROSS_PERILS_AT_VIEW_GRAIN`` is
   applied to a multi-peril group, attach ``WARN_DATASET_GROUP_MAX_ACROSS_PERILS``
   so the UI surfaces the assumption (CONTRACTS.md §4 / §10).

All of the actual persistence happens in the provider — the router only
validates and shapes the response.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..models.dataset import (
    DatasetGroupCreate,
    DatasetGroupCreated,
    DatasetGroupListResponse,
)
from ..models.enums import CombinationMethod, ErrorCode, WarningCode
from ..models.warnings import Warning, make_warning
from ..providers import ExposureDataProvider, get_provider

router = APIRouter(prefix="/dataset-groups", tags=["dataset-groups"])


@router.post("", status_code=201, response_model=DatasetGroupCreated)
def create_group(
    payload: DatasetGroupCreate,
    provider: ExposureDataProvider = Depends(get_provider),
) -> DatasetGroupCreated:
    """Create a dataset group (API_SPEC.md `POST /api/dataset-groups`)."""
    # ─── Validate members exist and currencies match ───
    member_currencies: set[str] = set()
    missing_ids: list[str] = []
    for member in payload.members:
        prog = provider.get_programme_by_dataset_id(member.dataset_id)
        if prog is None:
            missing_ids.append(member.dataset_id)
            continue
        member_currencies.add(prog.edm.currency)

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail={
                "code": ErrorCode.DATASET_NOT_FOUND.value,
                "message": "One or more member datasets were not found.",
                "details": {"missingDatasetIds": missing_ids},
            },
        )

    if len(member_currencies) > 1:
        raise HTTPException(
            status_code=409,
            detail={
                "code": ErrorCode.CURRENCY_MISMATCH.value,
                "message": (
                    "Selected datasets use different currencies. "
                    "Provide a conversion assumption or compare them separately."
                ),
                "details": {"memberCurrencies": sorted(member_currencies)},
            },
        )

    # ─── SUM_DISTINCT_SEGMENTS gate ───
    if (
        payload.combination_method == CombinationMethod.SUM_DISTINCT_SEGMENTS
        and not payload.distinct_segments_confirmed
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": (
                    "SUM_DISTINCT_SEGMENTS requires distinctSegmentsConfirmed=true. "
                    "Mark the member EDMs as distinct exposure segments to continue."
                ),
                "details": {"field": "distinctSegmentsConfirmed", "expected": True},
            },
        )

    # ─── Persist via provider ───
    group = provider.create_dataset_group(payload)

    # ─── Attach warnings for the combination method ───
    warnings: list[Warning] = []
    # `payload.members[*].peril` may arrive as plain `str` (model uses
    # `use_enum_values=True`); coerce to str for the warning context.
    distinct_perils = {str(m.peril.value if hasattr(m.peril, "value") else m.peril)
                       for m in payload.members}
    is_multi_peril = len(distinct_perils) > 1

    if (
        is_multi_peril
        and payload.combination_method == CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN
    ):
        warnings.append(
            make_warning(
                WarningCode.WARN_DATASET_GROUP_MAX_ACROSS_PERILS,
                context={
                    "datasetGroupId": group.dataset_group_id,
                    "perils": sorted(distinct_perils),
                },
            )
        )
    elif payload.combination_method == CombinationMethod.SUM_DISTINCT_SEGMENTS:
        warnings.append(
            make_warning(
                WarningCode.WARN_DATASET_GROUP_SUMMED,
                context={"datasetGroupId": group.dataset_group_id},
            )
        )

    return DatasetGroupCreated(
        dataset_group_id=group.dataset_group_id,
        warnings=warnings,
    )


@router.get("", response_model=DatasetGroupListResponse)
def list_groups(
    provider: ExposureDataProvider = Depends(get_provider),
) -> DatasetGroupListResponse:
    """List saved dataset groups (API_SPEC.md `GET /api/dataset-groups`)."""
    return DatasetGroupListResponse(dataset_groups=provider.list_dataset_groups())


__all__ = ["router"]
