"""Explicit item-order helper for judge-result IRT projection."""

from __future__ import annotations

from .llm_judge import JudgeFormatError, LLMJudgeResult


def _exact_result_mapping(value: object, name: str) -> dict[str, object]:
    """Admit one inert built-in result mapping before any mapping protocol runs."""
    if type(value) is not dict:
        raise JudgeFormatError(f"{name} must be a plain dict keyed by criterion id")
    if any(type(key) is not str for key in value):
        raise JudgeFormatError(f"{name} keys must be strings")
    return value


def _ordered_criterion_ids(
    result: LLMJudgeResult,
    criterion_order: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Validate one explicit inert criterion order without caller callbacks."""
    if type(criterion_order) not in (list, tuple):
        raise JudgeFormatError("criterion_order must be a list or tuple of strings")
    if any(type(criterion_id) is not str for criterion_id in criterion_order):
        raise JudgeFormatError("criterion_order must contain only strings")
    criterion_ids = tuple(criterion_order)
    if len(criterion_ids) < 2:
        raise JudgeFormatError(
            "IRT output requires multiple criterion items; a scalar judge result is invalid"
        )
    if len(set(criterion_ids)) != len(criterion_ids):
        raise JudgeFormatError("criterion_order must contain unique criterion ids")
    criterion_scores = _exact_result_mapping(
        result.criterion_scores,
        "criterion_scores",
    )
    if set(criterion_ids) != set(criterion_scores):
        raise JudgeFormatError(
            "criterion_order must contain exactly the rubric criterion ids"
        )
    return criterion_ids


def project_row_in_order(
    result: LLMJudgeResult,
    *,
    item_type: str,
    n_categories: int | None,
    criterion_order: list[str] | tuple[str, ...],
) -> tuple[int, ...]:
    """Project one result with the canonical binning core, then apply item order."""
    if type(result) is not LLMJudgeResult:
        raise TypeError("result must be an LLMJudgeResult")
    if type(item_type) is not str or item_type not in {"dichotomous", "polytomous"}:
        raise JudgeFormatError("item_type must be dichotomous or polytomous")

    criterion_ids = _ordered_criterion_ids(result, criterion_order)
    if result.criterion_categories is not None:
        criterion_categories = _exact_result_mapping(
            result.criterion_categories,
            "criterion_categories",
        )
        if set(criterion_categories) != set(criterion_ids):
            raise JudgeFormatError(
                "criterion categories must contain exactly the rubric criterion ids"
            )

    # `LLMJudgeResult.to_irt_row` is the single package-owned category/binning
    # authority. It projects in sorted criterion-id order; this adapter only
    # permutes that already-validated row into the caller's authoritative order.
    canonical_ids = tuple(sorted(criterion_ids))
    canonical_row = result.to_irt_row(
        item_type=item_type,
        n_categories=n_categories,
    )
    canonical_position = {
        criterion_id: position for position, criterion_id in enumerate(canonical_ids)
    }
    return tuple(
        canonical_row[canonical_position[criterion_id]] for criterion_id in criterion_ids
    )


__all__ = ["project_row_in_order"]
