"""Resolve map/detail/pivot selection targets into a flat fact view.

Extracted from the exposures router so export/hurricanes can share it without
importing HTTP-layer modules, and so exposures.py stays under the 1k-line bar.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from ..models.cedent import ProgrammeChain
from ..models.enums import CombinationMethod, ErrorCode, WarningCode
from ..models.exposure import ExposureFactNormalized
from ..models.warnings import Warning, make_warning
from ..providers.base import ExposureDataProvider
from ..providers.fact_load import FactBatch


class ResolvedView:
    """Flattened result of resolving any view target into the same shape."""

    __slots__ = (
        "facts",
        "currency",
        "combination_method",
        "base_dataset_id",
        "comparison_dataset_id",
        "warnings",
    )

    def __init__(
        self,
        facts: list[ExposureFactNormalized],
        currency: str,
        combination_method: CombinationMethod | None,
        base_dataset_id: str | None,
        comparison_dataset_id: str | None,
        warnings: list[Warning] | None = None,
    ) -> None:
        self.facts = facts
        self.currency = currency
        self.combination_method = combination_method
        self.base_dataset_id = base_dataset_id
        self.comparison_dataset_id = comparison_dataset_id
        self.warnings = warnings or []


# Back-compat alias used by export_excel / older imports
_ResolvedView = ResolvedView


def require_at_most_one_target(payload: Any) -> None:
    """AT MOST one of programmeId / chainId / chainIds / cedentId / datasetId / groupId."""
    chain_ids = list(getattr(payload, "chain_ids", []) or [])
    targets = {
        "programmeId": getattr(payload, "programme_id", None),
        "chainId": getattr(payload, "chain_id", None),
        "chainIds": chain_ids if chain_ids else None,
        "cedentId": getattr(payload, "cedent_id", None),
        "datasetId": getattr(payload, "dataset_id", None),
        "datasetGroupId": getattr(payload, "dataset_group_id", None),
    }
    set_count = sum(1 for v in targets.values() if v)
    if set_count > 1:
        raise HTTPException(
            status_code=422,
            detail={
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": (
                    "Provide AT MOST one of 'programmeId', 'chainId', 'chainIds', "
                    "'cedentId', 'datasetId', or 'datasetGroupId'. Omit them all "
                    "for the in-force portfolio view."
                ),
                "details": targets,
            },
        )


# Alias matching previous exposures name
_require_exactly_one_target = require_at_most_one_target


def _not_found(code: ErrorCode, message: str, details: dict) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": code.value, "message": message, "details": details},
    )


def _currency_for_scope(currencies: set[str]) -> str:
    """Return the sole scope currency or block unsafe cross-currency aggregation."""
    if len(currencies) > 1:
        raise HTTPException(
            status_code=409,
            detail={
                "code": ErrorCode.CURRENCY_MISMATCH.value,
                "message": (
                    "Selected programmes use different currencies and cannot be "
                    "aggregated without an explicit conversion model."
                ),
                "details": {"currencies": sorted(currencies)},
            },
        )
    return next(iter(currencies)) if currencies else "USD"


def _load_batch_warnings(batch: FactBatch, provider: ExposureDataProvider) -> list[Warning]:
    """Surface load failures and hybrid SQL→mock fallbacks."""
    warnings: list[Warning] = []
    if batch.errors:
        warnings.append(
            make_warning(
                WarningCode.WARN_EDM_LOAD_FAILED,
                context={
                    "datasetIds": batch.failed_dataset_ids,
                    "errors": [
                        {"datasetId": e.dataset_id, "message": e.message} for e in batch.errors
                    ],
                },
            )
        )
    fallbacks = getattr(provider, "last_fallback_reasons", None)
    if isinstance(fallbacks, dict) and fallbacks:
        # Only mention datasets that appear in this batch
        relevant = {k: v for k, v in fallbacks.items() if k in batch.by_id}
        if relevant:
            warnings.append(
                make_warning(
                    WarningCode.WARN_EDM_MOCK_FALLBACK,
                    context={
                        "datasetIds": list(relevant.keys()),
                        "reasons": relevant,
                    },
                )
            )
    return warnings


def facts_for_dataset_ids(
    provider: ExposureDataProvider,
    dataset_ids: list[str],
) -> tuple[list[ExposureFactNormalized], list[Warning]]:
    """Parallel multi-EDM load; returns (facts, load/fallback warnings)."""
    if not dataset_ids:
        return [], []
    if len(dataset_ids) == 1:
        ds = dataset_ids[0]
        try:
            facts = provider.get_facts_for_dataset(ds)
            batch = FactBatch(by_id={ds: facts}, errors=[])
        except Exception as exc:
            from ..providers.fact_load import LoadError

            batch = FactBatch(
                by_id={ds: []},
                errors=[LoadError(dataset_id=ds, message=f"{type(exc).__name__}: {exc}")],
            )
        return batch.flatten(dataset_ids), _load_batch_warnings(batch, provider)

    batch = provider.get_facts_for_datasets(dataset_ids)
    return batch.flatten(dataset_ids), _load_batch_warnings(batch, provider)


# Back-compat name
_facts_for_dataset_ids = lambda provider, ids: facts_for_dataset_ids(provider, ids)[0]  # noqa: E731


def resolve_comparison_dataset_id(
    provider: ExposureDataProvider,
    payload: Any,
) -> str | None:
    cmp_prog_id = getattr(payload, "comparison_programme_id", None)
    if cmp_prog_id:
        prog = provider.get_programme(cmp_prog_id)
        if prog is None:
            raise _not_found(
                ErrorCode.PRIOR_DB_NOT_FOUND,
                "Prior programme was not found.",
                {"comparisonProgrammeId": cmp_prog_id},
            )
        return prog.dataset_id
    return getattr(payload, "comparison_dataset_id", None)


_resolve_comparison_dataset_id = resolve_comparison_dataset_id


def resolve_view(
    provider: ExposureDataProvider,
    payload: Any,
) -> ResolvedView:
    """Resolve programme/chain/cedent/portfolio selection into facts + metadata."""
    chain_ids = list(getattr(payload, "chain_ids", []) or [])
    any_target = any(
        [
            getattr(payload, "programme_id", None),
            getattr(payload, "chain_id", None),
            chain_ids,
            getattr(payload, "cedent_id", None),
            getattr(payload, "dataset_id", None),
            getattr(payload, "dataset_group_id", None),
        ]
    )
    if not any_target:
        dataset_ids = provider.portfolio_dataset_ids(in_force_only=True)
        currencies: set[str] = set()
        for cedent in provider.list_cedents():
            for chain in cedent.chains:
                for prog in chain.programmes:
                    if prog.dataset_id in dataset_ids:
                        currencies.add(prog.edm.currency)
        facts, load_warnings = facts_for_dataset_ids(provider, dataset_ids)
        warnings: list[Warning] = list(load_warnings)
        currency = _currency_for_scope(currencies)
        method = CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN
        if len(dataset_ids) > 1:
            warnings.append(make_warning(WarningCode.WARN_DATASET_GROUP_MAX_ACROSS_PERILS))
        return ResolvedView(
            facts=facts,
            currency=currency,
            combination_method=method,
            base_dataset_id=None,
            comparison_dataset_id=None,
            warnings=warnings,
        )

    if getattr(payload, "programme_id", None):
        prog = provider.get_programme(payload.programme_id)
        if prog is None:
            raise _not_found(
                ErrorCode.DATASET_NOT_FOUND,
                f"Programme '{payload.programme_id}' was not found.",
                {"programmeId": payload.programme_id},
            )
        facts, load_warnings = facts_for_dataset_ids(provider, [prog.dataset_id])
        return ResolvedView(
            facts=facts,
            currency=prog.edm.currency,
            combination_method=CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN,
            base_dataset_id=prog.dataset_id,
            comparison_dataset_id=resolve_comparison_dataset_id(provider, payload),
            warnings=load_warnings,
        )

    if getattr(payload, "chain_id", None):
        chain = provider.get_chain(payload.chain_id)
        if chain is None:
            raise _not_found(
                ErrorCode.DATASET_NOT_FOUND,
                f"Programme chain '{payload.chain_id}' was not found.",
                {"chainId": payload.chain_id},
            )
        progs = sorted(chain.programmes, key=lambda p: p.treaty_year, reverse=True)
        if not progs:
            raise _not_found(
                ErrorCode.DATASET_NOT_FOUND,
                f"Chain '{chain.chain_id}' has no programmes.",
                {"chainId": chain.chain_id},
            )
        current = progs[0]
        comparison_id = resolve_comparison_dataset_id(provider, payload)
        if comparison_id is None and len(progs) > 1:
            comparison_id = progs[1].dataset_id
        facts, load_warnings = facts_for_dataset_ids(provider, [current.dataset_id])
        return ResolvedView(
            facts=facts,
            currency=current.edm.currency,
            combination_method=CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN,
            base_dataset_id=current.dataset_id,
            comparison_dataset_id=comparison_id,
            warnings=load_warnings,
        )

    if chain_ids:
        chains: list[ProgrammeChain] = []
        for cid in chain_ids:
            ch = provider.get_chain(cid)
            if ch is None:
                raise _not_found(
                    ErrorCode.DATASET_NOT_FOUND,
                    f"Programme chain '{cid}' was not found.",
                    {"chainId": cid},
                )
            chains.append(ch)
        dataset_ids = []
        currencies = set()
        for ch in chains:
            progs = sorted(ch.programmes, key=lambda p: p.treaty_year, reverse=True)
            if not progs:
                continue
            current = progs[0]
            dataset_ids.append(current.dataset_id)
            currencies.add(current.edm.currency)
        facts, load_warnings = facts_for_dataset_ids(provider, dataset_ids)
        warnings = list(load_warnings)
        currency = _currency_for_scope(currencies)
        method = CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN
        if len(dataset_ids) > 1:
            warnings.append(make_warning(WarningCode.WARN_DATASET_GROUP_MAX_ACROSS_PERILS))
        return ResolvedView(
            facts=facts,
            currency=currency,
            combination_method=method,
            base_dataset_id=None,
            comparison_dataset_id=None,
            warnings=warnings,
        )

    if getattr(payload, "cedent_id", None):
        cedent = provider.get_cedent(payload.cedent_id)
        if cedent is None:
            raise _not_found(
                ErrorCode.DATASET_NOT_FOUND,
                f"Cedent '{payload.cedent_id}' was not found.",
                {"cedentId": payload.cedent_id},
            )
        dataset_ids = []
        currencies = set()
        for chain in cedent.chains:
            progs = sorted(chain.programmes, key=lambda p: p.treaty_year, reverse=True)
            if not progs:
                continue
            current = progs[0]
            dataset_ids.append(current.dataset_id)
            currencies.add(current.edm.currency)
        facts, load_warnings = facts_for_dataset_ids(provider, dataset_ids)
        warnings = list(load_warnings)
        currency = _currency_for_scope(currencies)
        method = CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN
        if len(dataset_ids) > 1:
            warnings.append(make_warning(WarningCode.WARN_DATASET_GROUP_MAX_ACROSS_PERILS))
        return ResolvedView(
            facts=facts,
            currency=currency,
            combination_method=method,
            base_dataset_id=None,
            comparison_dataset_id=None,
            warnings=warnings,
        )

    if getattr(payload, "dataset_id", None):
        prog = provider.get_programme_by_dataset_id(payload.dataset_id)
        if prog is None:
            raise _not_found(
                ErrorCode.DATASET_NOT_FOUND,
                f"Dataset '{payload.dataset_id}' was not found.",
                {"datasetId": payload.dataset_id},
            )
        facts, load_warnings = facts_for_dataset_ids(provider, [prog.dataset_id])
        return ResolvedView(
            facts=facts,
            currency=prog.edm.currency,
            combination_method=CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN,
            base_dataset_id=prog.dataset_id,
            comparison_dataset_id=resolve_comparison_dataset_id(provider, payload),
            warnings=load_warnings,
        )

    assert getattr(payload, "dataset_group_id", None) is not None
    group = provider.get_dataset_group(payload.dataset_group_id)
    if group is None:
        raise _not_found(
            ErrorCode.DATASET_GROUP_NOT_FOUND,
            f"Dataset group '{payload.dataset_group_id}' was not found.",
            {"datasetGroupId": payload.dataset_group_id},
        )
    member_ids = [m.dataset_id for m in group.members]
    facts, load_warnings = facts_for_dataset_ids(provider, member_ids)
    base = group.members[0].dataset_id if group.members else None
    return ResolvedView(
        facts=facts,
        currency=group.currency,
        combination_method=group.combination_method,
        base_dataset_id=base,
        comparison_dataset_id=resolve_comparison_dataset_id(provider, payload),
        warnings=load_warnings,
    )


# Back-compat for callers expecting private names
_resolve_view = resolve_view
