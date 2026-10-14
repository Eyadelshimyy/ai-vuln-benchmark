"""Validated construct-level measurement contract for LLM-as-a-Judge handoffs.

Automated judges and downstream agents frequently mis-specify the measurement
side of an IRT handoff: they submit a single scalar score where a multi-item
construct is required, treat one-based rating labels as zero-based categories,
or ship rubrics too short to identify a unidimensional factor. This module
owns the package-level contract for that boundary so every caller receives
the same validated answer before any model fitting starts.

Contract summary
----------------

1. One judge criterion is one polytomous or dichotomous item. A scalar judge
   result is not an IRT observation.
2. Categories are always zero based: valid categories are ``0 .. k - 1`` for
   ``k`` ordered options, matching the internal representation used by every
   fast-mlsirm estimator (the recursive Lord-Wingersky style formulas index
   categories from zero even when a rubric displays ``1, 2, 3, ...``).
3. A unidimensional judge construct (one facet) must carry at least
   :data:`MIN_JUDGE_CONSTRUCT_ITEMS` criteria by default, with
   :data:`RECOMMENDED_JUDGE_CONSTRUCT_ITEMS` as the recommended target and at
   most :data:`MAX_JUDGE_CONSTRUCT_ITEMS` before the rubric should be split
   into separate facets.

The item-count policy is deliberately stricter than the bare identification
floor. Factor models are only identified from three salient variables per
common factor (Thurstone, 1947; Fabrigar et al., 1999), applied measurement
guidance asks for three indicators minimum with five or more preferred
(Hair et al., 2019), and facet scales in validated inventories commonly carry
about eight items per facet (Costa & McCrae, 1992). The default floor of five
items keeps judge-derived constructs inside that evidence-backed range while
:data:`ABSOLUTE_JUDGE_CONSTRUCT_FLOOR` preserves the hard mathematical
boundary for explicitly acknowledged short forms.

References (APA 7th ed.)
    Costa, P. T., Jr., & McCrae, R. R. (1992). *Revised NEO Personality
        Inventory (NEO-PI-R) and NEO Five-Factor Inventory (NEO-FFI)
        professional manual*. Psychological Assessment Resources.
    Fabrigar, L. R., Wegener, D. T., MacCallum, R. C., & Strahan, E. J.
        (1999). Evaluating the use of exploratory factor analysis in
        psychological research. *Psychological Methods, 4*(3), 272-299.
        https://doi.org/10.1037/1082-989X.4.3.272
    Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2019).
        *Multivariate data analysis* (8th ed.). Cengage Learning.
    Thurstone, L. L. (1947). *Multiple-factor analysis: A development and
        expansion of The Vectors of Mind*. University of Chicago Press.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ._judge_projection_order import project_row_in_order
from .llm_judge import (
    JudgeFormatError,
    LLMJudgeResult,
    MAX_JUDGE_CATEGORIES,
)

MIN_JUDGE_CONSTRUCT_ITEMS = 5
"""Default policy floor: fewest criteria admitted per unidimensional facet."""

RECOMMENDED_JUDGE_CONSTRUCT_ITEMS = 7
"""Recommended criterion count per facet before splitting or pooling."""

MAX_JUDGE_CONSTRUCT_ITEMS = 11
"""Largest admitted facet size; larger rubrics must be declared as facets."""

ABSOLUTE_JUDGE_CONSTRUCT_FLOOR = 3
"""Identification-only floor admitted solely through ``allow_short_form``."""

ZERO_BASED_CATEGORY_CODING = "zero_based_0_to_k_minus_1"
"""Stable machine-readable name of the category coding this package accepts."""

JUDGE_MEASUREMENT_GUIDANCE = (
    "LLM-as-a-Judge to fast-mlsirm measurement contract: "
    f"(1) one rubric criterion is one item; design {MIN_JUDGE_CONSTRUCT_ITEMS}"
    f"-{RECOMMENDED_JUDGE_CONSTRUCT_ITEMS} criteria per unidimensional facet "
    f"(recommended {RECOMMENDED_JUDGE_CONSTRUCT_ITEMS}, hard maximum "
    f"{MAX_JUDGE_CONSTRUCT_ITEMS}); "
    "(2) report ordered ratings as zero-based integer categories in "
    "0..k-1 so displayed option 1 becomes category 0; "
    "(3) collect one LLMJudgeResult per response and project them through "
    "project_judge_results_to_matrix before calling any fit_* function."
)


def _policy_integer(value: object, name: str) -> int:
    """Normalize one inert built-in policy integer without caller callbacks."""
    value_type = type(value)
    if value_type is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    return value


def _boolean_control(value: object, name: str) -> bool:
    """Normalize one trusted Boolean control without caller truthiness."""
    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is np.bool_:
        return bool(value)
    raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True)
class JudgeConstructPolicy:
    """Validated item-count bounds for one judge-derived facet."""

    min_items: int = MIN_JUDGE_CONSTRUCT_ITEMS
    recommended_items: int = RECOMMENDED_JUDGE_CONSTRUCT_ITEMS
    max_items: int = MAX_JUDGE_CONSTRUCT_ITEMS

    def __post_init__(self) -> None:
        """Reject malformed policies before any construct admission runs."""
        min_items = _policy_integer(self.min_items, "min_items")
        recommended_items = _policy_integer(
            self.recommended_items, "recommended_items"
        )
        max_items = _policy_integer(self.max_items, "max_items")
        if not ABSOLUTE_JUDGE_CONSTRUCT_FLOOR <= min_items <= max_items:
            raise ValueError(
                "min_items must be between "
                f"{ABSOLUTE_JUDGE_CONSTRUCT_FLOOR} and max_items"
            )
        if not min_items <= recommended_items <= max_items:
            raise ValueError(
                "recommended_items must be between min_items and max_items"
            )
        if max_items > MAX_JUDGE_CONSTRUCT_ITEMS:
            raise ValueError(
                f"max_items cannot exceed {MAX_JUDGE_CONSTRUCT_ITEMS}"
            )
        object.__setattr__(self, "min_items", min_items)
        object.__setattr__(self, "recommended_items", recommended_items)
        object.__setattr__(self, "max_items", max_items)

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-safe copy of the policy bounds."""
        return {
            "min_items": self.min_items,
            "recommended_items": self.recommended_items,
            "max_items": self.max_items,
        }


DEFAULT_JUDGE_CONSTRUCT_POLICY = JudgeConstructPolicy()


@dataclass(frozen=True)
class JudgeConstructSpec:
    """Validated design contract for projecting judge criteria onto items."""

    criterion_ids: tuple[str, ...]
    item_type: str
    n_categories: int | None
    category_coding: str
    meets_policy: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)
    policy: JudgeConstructPolicy | None = None

    def __post_init__(self) -> None:
        """Freeze identity ordering and re-validate admitted content."""
        if (
            type(self.criterion_ids) is not tuple
            or not self.criterion_ids
            or any(
                type(criterion_id) is not str or not criterion_id.strip()
                for criterion_id in self.criterion_ids
            )
        ):
            raise TypeError("criterion_ids must be a non-empty tuple of strings")
        if len(set(self.criterion_ids)) != len(self.criterion_ids):
            raise ValueError("criterion_ids must be unique")
        if type(self.item_type) is not str or self.item_type not in {
            "dichotomous",
            "polytomous",
        }:
            raise ValueError("item_type must be dichotomous or polytomous")
        if self.item_type == "dichotomous":
            if type(self.n_categories) is not int or self.n_categories != 2:
                raise ValueError("dichotomous specs require n_categories == 2")
        elif (
            type(self.n_categories) is not int
            or not 2 <= self.n_categories <= MAX_JUDGE_CATEGORIES
        ):
            raise ValueError(
                f"polytomous specs require n_categories in 2..{MAX_JUDGE_CATEGORIES}"
            )
        if (
            type(self.category_coding) is not str
            or self.category_coding != ZERO_BASED_CATEGORY_CODING
        ):
            raise ValueError(
                "category_coding must be "
                f"{ZERO_BASED_CATEGORY_CODING!r}"
            )
        if type(self.meets_policy) is not bool:
            raise TypeError("meets_policy must be a built-in bool")
        if (
            type(self.warnings) is not tuple
            or any(type(warning) is not str or not warning.strip() for warning in self.warnings)
        ):
            raise TypeError("warnings must be a tuple of non-empty strings")
        if self.meets_policy is not True and not self.warnings:
            raise ValueError("short-form specs require explicit warnings")
        count = len(self.criterion_ids)
        if not ABSOLUTE_JUDGE_CONSTRUCT_FLOOR <= count <= MAX_JUDGE_CONSTRUCT_ITEMS:
            raise ValueError(
                "criterion_ids must contain "
                f"{ABSOLUTE_JUDGE_CONSTRUCT_FLOOR}..{MAX_JUDGE_CONSTRUCT_ITEMS} items"
            )
        if self.policy is not None:
            if type(self.policy) is not JudgeConstructPolicy:
                raise TypeError("policy must be a JudgeConstructPolicy")
            resolved_policy = replace(self.policy)
            object.__setattr__(self, "policy", resolved_policy)
            within_policy = (
                resolved_policy.min_items <= count <= resolved_policy.max_items
            )
            if self.meets_policy is not within_policy:
                raise ValueError("meets_policy does not match policy bounds")
            if count > resolved_policy.max_items:
                raise ValueError("criterion count exceeds policy max_items")

    @property
    def n_items(self) -> int:
        """Return the admitted criterion count."""
        return len(self.criterion_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe handoff contract consumed by agent runtimes."""
        return {
            "criterion_ids": list(self.criterion_ids),
            "n_items": self.n_items,
            "item_type": self.item_type,
            "n_categories": self.n_categories,
            "category_coding": self.category_coding,
            "meets_policy": self.meets_policy,
            "warnings": list(self.warnings),
            "policy": None if self.policy is None else self.policy.to_dict(),
        }


def validate_judge_construct(
    criterion_ids: Sequence[object] | Iterable[object],
    *,
    item_type: str = "polytomous",
    n_categories: int | None = None,
    policy: JudgeConstructPolicy | None = None,
    allow_short_form: bool = False,
) -> JudgeConstructSpec:
    """Validate one judge-derived facet against the measurement policy.

    Parameters
    ----------
    criterion_ids:
        Ordered rubric criterion identifiers; each becomes exactly one item.
    item_type:
        Either ``"dichotomous"`` (two categories) or ``"polytomous"``
        (``n_categories >= 2``).
    n_categories:
        Ordered option count for polytomous output; required there and
        forbidden for dichotomous output.
    policy:
        Optional custom :class:`JudgeConstructPolicy`; defaults to the
        package policy of 5 minimum / 7 recommended / 11 maximum items.
    allow_short_form:
        Explicit escape hatch admitting 3-4 item facets below the policy
        floor. Every admitted short form carries a precision warning;
        fewer than three criteria are rejected regardless because a common
        factor is then unidentified (Thurstone, 1947).

    Returns
    -------
    JudgeConstructSpec
        Frozen, JSON-serializable design contract for the IRT handoff.

    Raises
    ------
    JudgeFormatError
        When the facet violates the policy without an acknowledged short
        form, falls under the identification floor, exceeds the maximum
        facet size, or carries invalid identifiers or category semantics.
    """
    resolved_policy = DEFAULT_JUDGE_CONSTRUCT_POLICY if policy is None else policy
    if type(resolved_policy) is not JudgeConstructPolicy:
        raise TypeError("policy must be a JudgeConstructPolicy")
    resolved_policy = replace(resolved_policy)
    if type(item_type) is not str or item_type not in {"dichotomous", "polytomous"}:
        raise JudgeFormatError("item_type must be dichotomous or polytomous")
    if item_type == "dichotomous":
        if n_categories is not None:
            raise JudgeFormatError(
                "dichotomous constructs do not accept n_categories"
            )
        resolved_categories: int | None = 2
    else:
        if (
            type(n_categories) is not int
            or not 2 <= n_categories <= MAX_JUDGE_CATEGORIES
        ):
            raise JudgeFormatError(
                f"polytomous constructs require n_categories in 2..{MAX_JUDGE_CATEGORIES}"
            )
        resolved_categories = n_categories
    resolved_allow_short_form = _boolean_control(
        allow_short_form, "allow_short_form"
    )
    ids = tuple(criterion_ids)
    if any(type(criterion_id) is not str or not criterion_id.strip() for criterion_id in ids):
        raise JudgeFormatError("criterion_ids must contain non-empty strings")
    if len(set(ids)) != len(ids):
        raise JudgeFormatError("criterion_ids must be unique")
    count = len(ids)
    if count < ABSOLUTE_JUDGE_CONSTRUCT_FLOOR:
        raise JudgeFormatError(
            "a judge construct needs at least "
            f"{ABSOLUTE_JUDGE_CONSTRUCT_FLOOR} criteria for factor "
            f"identification; received {count}"
        )
    warnings: tuple[str, ...] = ()
    meets_policy = resolved_policy.min_items <= count <= resolved_policy.max_items
    if not meets_policy:
        if count > resolved_policy.max_items:
            raise JudgeFormatError(
                f"facet exceeds the {resolved_policy.max_items}-item maximum; "
                "split the rubric into multiple facets"
            )
        if not resolved_allow_short_form:
            raise JudgeFormatError(
                f"facet has {count} criteria but policy requires at least "
                f"{resolved_policy.min_items}; pass allow_short_form=True to "
                "admit an identified short form with reduced precision"
            )
        warnings = (
            "short form admitted below the "
            f"{resolved_policy.min_items}-item policy floor; person "
            "precision and reliability are reduced",
        )
    elif count < resolved_policy.recommended_items:
        warnings = (
            f"{count} items is below the recommended "
            f"{resolved_policy.recommended_items}; add criteria when possible",
        )
    return JudgeConstructSpec(
        criterion_ids=ids,
        item_type=item_type,
        n_categories=resolved_categories,
        category_coding=ZERO_BASED_CATEGORY_CODING,
        meets_policy=meets_policy,
        warnings=warnings,
        policy=resolved_policy,
    )


def project_judge_results_to_matrix(
    results: Iterable[LLMJudgeResult],
    spec: JudgeConstructSpec,
) -> np.ndarray:
    """Project validated judge results into a persons x items matrix.

    Every row is projected by a package-owned explicit-order helper using the
    zero-based ``0..k-1`` category semantics described by
    ``spec.category_coding``. Each result must contain exactly the criterion
    identities declared by ``spec``; the returned matrix columns follow
    ``spec.criterion_ids`` order exactly, regardless of mapping insertion,
    lexical key order, or changes to the shared default row helper. Every row
    in one panel must also use the same category-generation semantics: either
    all rows provide explicit criterion categories, or every row derives
    categories from criterion scores.

    Parameters
    ----------
    results:
        One :class:`LLMJudgeResult` per scored response, in person order.
    spec:
        Validated design returned by :func:`validate_judge_construct`.

    Returns
    -------
    numpy.ndarray
        Integer matrix of shape ``(len(results), spec.n_items)`` with columns
        ordered exactly as ``spec.criterion_ids`` and categories in ``0..k-1``
        (or ``{0, 1}`` when dichotomous), ready for ``fit_grm`` / ``fit_gpcm`` /
        ``fit_irt_experiment``.
    """
    if type(spec) is not JudgeConstructSpec:
        raise TypeError("spec must be a JudgeConstructSpec")
    spec = replace(spec)
    rows: list[tuple[int, ...]] = []
    expected_ids = set(spec.criterion_ids)
    panel_uses_explicit_categories: bool | None = None
    for result in results:
        if type(result) is not LLMJudgeResult:
            raise TypeError("results must contain LLMJudgeResult values")
        if type(result.criterion_scores) is not dict:
            raise JudgeFormatError(
                "criterion_scores must be a plain dict keyed by criterion id; "
                "rebuild each result with dict(result.criterion_scores) before "
                "projecting"
            )
        if set(result.criterion_scores) != expected_ids:
            raise JudgeFormatError(
                "result criterion set does not match the validated construct spec"
            )
        uses_explicit_categories = result.criterion_categories is not None
        if panel_uses_explicit_categories is None:
            panel_uses_explicit_categories = uses_explicit_categories
        elif uses_explicit_categories is not panel_uses_explicit_categories:
            raise JudgeFormatError(
                "judge panel must use one category-generation mode: either all "
                "rows provide explicit criterion categories or all rows derive "
                "categories from criterion scores"
            )
        row = project_row_in_order(
            result,
            item_type=spec.item_type,
            n_categories=(
                spec.n_categories if spec.item_type == "polytomous" else None
            ),
            criterion_order=spec.criterion_ids,
        )
        rows.append(tuple(row))
    if not rows:
        raise JudgeFormatError("at least one judged response is required")
    matrix = np.asarray(rows, dtype=np.int64)
    upper = 1 if spec.item_type == "dichotomous" else int(spec.n_categories or 0) - 1
    if matrix.size and (matrix.min(initial=0) < 0 or matrix.max() > upper):
        raise JudgeFormatError(
            f"projected categories must stay within 0..{upper}"
        )
    return matrix


def describe_measurement_contract() -> dict[str, Any]:
    """Return the agent-facing measurement contract as structured data."""
    return {
        "guidance": JUDGE_MEASUREMENT_GUIDANCE,
        "category_coding": ZERO_BASED_CATEGORY_CODING,
        "policy": DEFAULT_JUDGE_CONSTRUCT_POLICY.to_dict(),
        "absolute_floor": ABSOLUTE_JUDGE_CONSTRUCT_FLOOR,
        "max_categories": MAX_JUDGE_CATEGORIES,
        "references": [
            "Thurstone (1947) Multiple-factor analysis",
            "Fabrigar et al. (1999) Psychological Methods, 4(3), 272-299",
            "Hair et al. (2019) Multivariate data analysis (8th ed.)",
            "Costa & McCrae (1992) NEO-PI-R professional manual",
        ],
    }


__all__ = [
    "ABSOLUTE_JUDGE_CONSTRUCT_FLOOR",
    "DEFAULT_JUDGE_CONSTRUCT_POLICY",
    "JUDGE_MEASUREMENT_GUIDANCE",
    "JudgeConstructPolicy",
    "JudgeConstructSpec",
    "MAX_JUDGE_CONSTRUCT_ITEMS",
    "MIN_JUDGE_CONSTRUCT_ITEMS",
    "RECOMMENDED_JUDGE_CONSTRUCT_ITEMS",
    "ZERO_BASED_CATEGORY_CODING",
    "describe_measurement_contract",
    "project_judge_results_to_matrix",
    "validate_judge_construct",
]
