"""Paired controls for measuring polytomous LLM-judge sensitivity.

The controls are diagnostic evidence only.  They keep the same injected judge
for every variant, preserve failed calls in the denominator, and never repair
or infer a missing IRT item.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .llm_judge import (
    MAX_JUDGE_CATEGORIES,
    MAX_JUDGE_CRITERIA,
    MAX_JUDGE_TEXT_CHARACTERS,
    ContextualOrchestratorJudge,
    JudgeCriterion,
    LLMJudgeResult,
    _category,
    _category_count,
    _criteria,
)

CALIBRATION_VARIANTS = frozenset(
    {"baseline", "option_only", "shuffled_options", "replaced_distractor"}
)
CONTAMINATION_STATUSES = frozenset({"unknown", "held_out", "known_overlap"})
MAX_CALIBRATION_CASES = 256
_MAX_METADATA_ENTRIES = 32
_MAX_METADATA_TEXT = 256
_MAX_ERROR_TEXT = 512
_MAX_EVIDENCE_ENTRIES = 64
_MAX_EVIDENCE_DEPTH = 6
_CALIBRATION_STATUSES = frozenset({"passed", "judge_failed", "irt_failed"})
_CALIBRATION_ERROR_CODES = frozenset(
    {"judge_call_failed", "irt_projection_failed"}
)
_CALIBRATION_ERROR_TYPES = frozenset(
    {
        "ConnectionError",
        "JudgeFormatError",
        "OSError",
        "RuntimeError",
        "TimeoutError",
        "TypeError",
        "UnknownError",
        "ValueError",
    }
)
_ALLOWED_EVIDENCE_KEYS = frozenset(
    {
        "call_count",
        "call_status",
        "category_count",
        "category_method",
        "completed_call_count",
        "criterion_id",
        "error_type",
        "failed_call_count",
        "failure_code",
        "meets_threshold",
        "parse_status",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "records",
        "semantic_status",
        "threshold_index",
        "trace_step_count",
        "usage",
    }
)


def _safe_error_type(exc: Exception) -> str:
    """Return an allowlisted exception class without serializing exception text."""
    name = type(exc).__name__
    return name if name in _CALIBRATION_ERROR_TYPES else "UnknownError"


def _text(value: Any, name: str, *, maximum: int = MAX_JUDGE_TEXT_CHARACTERS) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return normalized


def _metadata(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be an object")
    if len(value) > _MAX_METADATA_ENTRIES:
        raise ValueError(f"metadata must contain at most {_MAX_METADATA_ENTRIES} entries")
    normalized: dict[str, str] = {}
    for key, member in value.items():
        normalized_key = _text(key, "metadata key", maximum=_MAX_METADATA_TEXT)
        if normalized_key in normalized:
            raise ValueError("metadata keys must remain unique after trimming")
        normalized[normalized_key] = _text(
            member, "metadata value", maximum=_MAX_METADATA_TEXT
        )
    return MappingProxyType(normalized)


def _bounded_evidence(value: Any, *, depth: int = 0) -> dict[str, Any] | None:
    """Keep structured failure evidence while excluding source/model text."""
    if not isinstance(value, Mapping) or depth > _MAX_EVIDENCE_DEPTH:
        return None
    result: dict[str, Any] = {}
    for key, member in list(value.items())[:_MAX_EVIDENCE_ENTRIES]:
        if type(key) is not str or key not in _ALLOWED_EVIDENCE_KEYS:
            continue
        if isinstance(member, Mapping):
            nested = _bounded_evidence(member, depth=depth + 1)
            if nested is not None:
                result[key] = nested
        elif isinstance(member, (list, tuple)):
            nested_items: list[Any] = []
            for item in list(member)[:_MAX_EVIDENCE_ENTRIES]:
                if isinstance(item, Mapping):
                    nested = _bounded_evidence(item, depth=depth + 1)
                    if nested is not None:
                        nested_items.append(nested)
                elif type(item) in (bool, int, float) or item is None:
                    nested_items.append(item)
                elif type(item) is str:
                    nested_items.append(item[:_MAX_ERROR_TEXT])
            result[key] = nested_items
        elif type(member) in (bool, int, float) or member is None:
            result[key] = member
        elif type(member) is str:
            result[key] = member[:_MAX_ERROR_TEXT]
    return result


@dataclass(frozen=True)
class JudgeCalibrationCase:
    """One paired judge input variant.

    ``task`` is deliberately caller-rendered.  For an option-only case it
    must omit the question; this prevents the library from pretending that an
    arbitrary task has a universally safe question/options decomposition.
    """

    case_id: str
    variant: str
    task: str
    answer: str
    reference_answer: str | None = None
    contamination_status: str = "unknown"
    gold_categories: Mapping[str, int] | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    option_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id", maximum=256))
        variant = _text(self.variant, "variant", maximum=64)
        if variant not in CALIBRATION_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(CALIBRATION_VARIANTS)}")
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "task", _text(self.task, "task"))
        object.__setattr__(self, "answer", _text(self.answer, "answer"))
        if self.reference_answer is not None:
            object.__setattr__(
                self,
                "reference_answer",
                _text(self.reference_answer, "reference_answer"),
            )
        status = _text(self.contamination_status, "contamination_status", maximum=64)
        if status not in CONTAMINATION_STATUSES:
            raise ValueError(
                f"contamination_status must be one of {sorted(CONTAMINATION_STATUSES)}"
            )
        object.__setattr__(self, "contamination_status", status)
        if self.option_count is not None and (
            type(self.option_count) is not int
            or not 2 <= self.option_count <= MAX_JUDGE_CATEGORIES
        ):
            raise ValueError(
                f"option_count must be an integer in 2..{MAX_JUDGE_CATEGORIES}"
            )
        if self.gold_categories is not None:
            if not isinstance(self.gold_categories, Mapping) or not self.gold_categories:
                raise ValueError("gold_categories must be a non-empty object")
            gold: dict[str, int] = {}
            for key, value in self.gold_categories.items():
                if type(key) is not str or not key.strip() or type(value) is not int:
                    raise ValueError("gold_categories must map strings to built-in integers")
                normalized_key = key.strip()
                if normalized_key in gold:
                    raise ValueError(
                        "gold_categories keys must remain unique after trimming"
                    )
                gold[normalized_key] = value
            object.__setattr__(self, "gold_categories", MappingProxyType(gold))
        normalized_metadata = _metadata(self.metadata)
        metadata_option_count = normalized_metadata.get("option_count")
        if metadata_option_count is not None:
            if not metadata_option_count.isascii() or not metadata_option_count.isdigit():
                raise ValueError("metadata option_count must be a canonical integer")
            parsed_option_count = int(metadata_option_count)
            if not 2 <= parsed_option_count <= MAX_JUDGE_CATEGORIES:
                raise ValueError(
                    f"metadata option_count must be in 2..{MAX_JUDGE_CATEGORIES}"
                )
            if str(parsed_option_count) != metadata_option_count:
                raise ValueError("metadata option_count must be canonical")
            if self.option_count is None:
                object.__setattr__(self, "option_count", parsed_option_count)
            elif self.option_count != parsed_option_count:
                raise ValueError("option_count must agree with metadata option_count")
        object.__setattr__(self, "metadata", normalized_metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return bounded metadata without copying task, answer, or source text."""
        return {
            "case_id": self.case_id,
            "variant": self.variant,
            "option_count": self.option_count,
            "contamination_status": self.contamination_status,
            "metadata": dict(self.metadata),
            "gold_categories_provided": self.gold_categories is not None,
        }


def _render_options(question: str | None, options: tuple[str, ...]) -> str:
    parts: list[str] = []
    if question is not None:
        parts.append(f"Question:\n{question}")
    parts.append(
        "Options:\n"
        + "\n".join(f"{index}. {option}" for index, option in enumerate(options, start=1))
    )
    return "\n\n".join(parts)


def _case_metadata(
    options: tuple[str, ...],
    order: tuple[int, ...],
    correct_option_index: int,
    *,
    question_present: bool,
    replacement_index: int | None = None,
) -> dict[str, str]:
    result = {
        "option_count": str(len(options)),
        "option_order": ",".join(str(index) for index in order),
        "correct_option_index": str(correct_option_index),
        "question_present": str(question_present).lower(),
    }
    if replacement_index is not None:
        result["replacement_index"] = str(replacement_index)
    return result


def build_multiple_choice_calibration_cases(
    *,
    case_id: str,
    question: str,
    options: Iterable[str],
    answer: str,
    correct_option_index: int,
    replacement_distractor: str,
    reference_answer: str | None = None,
    contamination_status: str = "unknown",
    shuffle_seed: int = 0,
    gold_categories: Mapping[str, int] | None = None,
) -> tuple[JudgeCalibrationCase, ...]:
    """Build baseline and three paired MCQ sensitivity controls.

    The returned variants preserve the correct option while changing only the
    requested presentation factor.  ``option_only`` intentionally removes the
    question from the rendered task; contamination status remains caller-owned
    metadata and is never inferred from the prompt.
    """
    question = _text(question, "question")
    answer = _text(answer, "answer")
    if reference_answer is not None:
        reference_answer = _text(reference_answer, "reference_answer")
    if isinstance(options, (str, bytes)):
        raise TypeError("options must be an iterable of distinct strings")
    raw_options = tuple(options)
    if not 2 <= len(raw_options) <= MAX_JUDGE_CATEGORIES:
        raise ValueError(
            f"options must contain 2..{MAX_JUDGE_CATEGORIES} distinct values"
        )
    normalized_options = tuple(
        _text(option, f"options[{index}", maximum=2_000)
        for index, option in enumerate(raw_options)
    )
    if len(set(normalized_options)) != len(normalized_options):
        raise ValueError("options must be distinct")
    if type(correct_option_index) is not int or not 0 <= correct_option_index < len(normalized_options):
        raise ValueError("correct_option_index must identify one option")
    replacement_distractor = _text(
        replacement_distractor, "replacement_distractor", maximum=2_000
    )
    if replacement_distractor in normalized_options:
        raise ValueError("replacement_distractor must be new")
    if type(shuffle_seed) is not int:
        raise ValueError("shuffle_seed must be an integer")

    original_order = tuple(range(len(normalized_options)))
    shuffled_order = list(original_order)
    random.Random(shuffle_seed).shuffle(shuffled_order)
    if tuple(shuffled_order) == original_order:
        shuffled_order = shuffled_order[1:] + shuffled_order[:1]
    shuffled_options = tuple(normalized_options[index] for index in shuffled_order)
    shuffled_correct_index = shuffled_options.index(normalized_options[correct_option_index])

    replacement_index = next(
        index for index in original_order if index != correct_option_index
    )
    replaced_options = list(normalized_options)
    replaced_options[replacement_index] = replacement_distractor

    common = {
        "case_id": case_id,
        "answer": answer,
        "reference_answer": reference_answer,
        "contamination_status": contamination_status,
        "gold_categories": gold_categories,
        "option_count": len(normalized_options),
    }
    return (
        JudgeCalibrationCase(
            **common,
            variant="baseline",
            task=_render_options(question, normalized_options),
            metadata=_case_metadata(
                normalized_options,
                original_order,
                correct_option_index,
                question_present=True,
            ),
        ),
        JudgeCalibrationCase(
            **common,
            variant="option_only",
            task=_render_options(None, normalized_options),
            metadata=_case_metadata(
                normalized_options,
                original_order,
                correct_option_index,
                question_present=False,
            ),
        ),
        JudgeCalibrationCase(
            **common,
            variant="shuffled_options",
            task=_render_options(question, shuffled_options),
            metadata=_case_metadata(
                shuffled_options,
                tuple(shuffled_order),
                shuffled_correct_index,
                question_present=True,
            ),
        ),
        JudgeCalibrationCase(
            **common,
            variant="replaced_distractor",
            task=_render_options(question, tuple(replaced_options)),
            metadata=_case_metadata(
                tuple(replaced_options),
                original_order,
                correct_option_index,
                question_present=True,
                replacement_index=replacement_index,
            ),
        ),
    )


@dataclass(frozen=True)
class JudgeCalibrationOutcome:
    """One completed or failed paired calibration comparison."""

    case: JudgeCalibrationCase
    status: str
    result: LLMJudgeResult | None = None
    irt_row: tuple[int, ...] | None = None
    gold_exact: bool | None = None
    error_type: str | None = None
    error: str | None = None
    evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in _CALIBRATION_STATUSES:
            raise ValueError(f"status must be one of {sorted(_CALIBRATION_STATUSES)}")
        if self.status == "passed" and (self.result is None or self.irt_row is None):
            raise ValueError("passed outcomes require result and irt_row")
        if self.status != "passed" and self.irt_row is not None:
            raise ValueError("failed outcomes cannot contain an irt_row")
        if self.result is not None and not isinstance(self.result, LLMJudgeResult):
            raise TypeError("result must be an LLMJudgeResult")
        if self.irt_row is not None and (
            type(self.irt_row) is not tuple
            or any(type(category) is not int for category in self.irt_row)
        ):
            raise TypeError("irt_row must be a tuple of built-in integers")
        if self.gold_exact is not None and self.status != "passed":
            raise ValueError("gold_exact is only valid for passed outcomes")
        if self.error_type is not None:
            error_type = _text(self.error_type, "error_type", maximum=128)
            if error_type not in _CALIBRATION_ERROR_TYPES:
                raise ValueError("error_type must be an allowlisted calibration error type")
            object.__setattr__(self, "error_type", error_type)
        if self.error is not None:
            error = _text(self.error, "error", maximum=_MAX_ERROR_TEXT)
            if error not in _CALIBRATION_ERROR_CODES:
                raise ValueError("error must be a stable calibration error code")
            object.__setattr__(self, "error", error)
        if self.evidence is not None:
            sanitized = _bounded_evidence(self.evidence)
            if sanitized is None:
                raise ValueError("evidence must be a bounded object")
            object.__setattr__(self, "evidence", sanitized)

    def to_dict(self) -> dict[str, Any]:
        result = None
        if self.result is not None:
            result = {
                "score": self.result.score,
                "accepted": self.result.accepted,
                "criterion_scores": dict(self.result.criterion_scores),
                "criterion_categories": (
                    dict(self.result.criterion_categories)
                    if self.result.criterion_categories is not None
                    else None
                ),
                "category_count": self.result.category_count,
                "category_method": self.result.category_method,
                "trace_step_count": self.result.trace_step_count,
                "usage": dict(self.result.usage),
            }
        return {
            **self.case.to_dict(),
            "status": self.status,
            "result": result,
            "irt_row": list(self.irt_row) if self.irt_row is not None else None,
            "gold_exact": self.gold_exact,
            "error_type": self.error_type,
            "error": self.error,
            "evidence": dict(self.evidence) if self.evidence is not None else None,
        }


@dataclass(frozen=True)
class JudgeCalibrationReport:
    """Bounded paired evidence; score deltas are not causal bias estimates."""

    category_count: int
    criterion_ids: tuple[str, ...]
    outcomes: tuple[JudgeCalibrationOutcome, ...]

    def __post_init__(self) -> None:
        """Reject malformed direct construction before report methods run."""
        category_count = _category_count(self.category_count)
        object.__setattr__(self, "category_count", category_count)
        if (
            type(self.criterion_ids) is not tuple
            or not 2 <= len(self.criterion_ids) <= MAX_JUDGE_CRITERIA
        ):
            raise ValueError(
                f"criterion_ids must contain 2..{MAX_JUDGE_CRITERIA} values"
            )
        if any(type(criterion_id) is not str or not criterion_id.strip() for criterion_id in self.criterion_ids):
            raise ValueError("criterion_ids must contain non-empty strings")
        if len(set(self.criterion_ids)) != len(self.criterion_ids):
            raise ValueError("criterion_ids must be unique")
        if type(self.outcomes) is not tuple or not self.outcomes:
            raise ValueError("outcomes must contain at least one value")
        if any(not isinstance(outcome, JudgeCalibrationOutcome) for outcome in self.outcomes):
            raise TypeError("outcomes must contain JudgeCalibrationOutcome values")
        _validate_case_groups(tuple(outcome.case for outcome in self.outcomes))
        for outcome in self.outcomes:
            if outcome.status != "passed":
                continue
            if outcome.irt_row is None or len(outcome.irt_row) != len(self.criterion_ids):
                raise ValueError("passed outcomes must contain one IRT item per criterion")
            if outcome.result is None or outcome.result.category_count != category_count:
                raise ValueError("passed outcomes must use the report category_count")

    def status_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(outcome.status for outcome in self.outcomes).items()))

    def category_occupancy(self) -> dict[str, dict[str, int]]:
        """Count each observed category without hiding ceiling/floor saturation."""
        occupancy = {
            criterion_id: {
                str(category): 0 for category in range(self.category_count)
            }
            for criterion_id in self.criterion_ids
        }
        for outcome in self.outcomes:
            if outcome.status != "passed" or outcome.result is None:
                continue
            categories = outcome.result.criterion_categories
            if categories is None:
                continue
            for criterion_id in self.criterion_ids:
                category = categories.get(criterion_id)
                if type(category) is int and 0 <= category < self.category_count:
                    occupancy[criterion_id][str(category)] += 1
        return occupancy

    def option_count_summary(self) -> tuple[dict[str, Any], ...]:
        """Return descriptive K/variant strata without claiming a causal effect."""
        grouped: dict[tuple[int, str], list[JudgeCalibrationOutcome]] = {}
        for outcome in self.outcomes:
            option_count = outcome.case.option_count
            if option_count is None:
                continue
            grouped.setdefault((option_count, outcome.case.variant), []).append(outcome)

        summaries: list[dict[str, Any]] = []
        for (option_count, variant), outcomes in sorted(grouped.items()):
            passed = [
                outcome
                for outcome in outcomes
                if outcome.status == "passed" and outcome.result is not None
            ]
            occupancy = {
                criterion_id: {str(category): 0 for category in range(self.category_count)}
                for criterion_id in self.criterion_ids
            }
            scores: list[float] = []
            gold_scored = [outcome for outcome in outcomes if outcome.gold_exact is not None]
            for outcome in passed:
                if outcome.result is None:
                    raise RuntimeError("Passed outcome is missing a valid result")
                scores.append(outcome.result.score)
                categories = outcome.result.criterion_categories or {}
                for criterion_id in self.criterion_ids:
                    category = categories.get(criterion_id)
                    if type(category) is int and 0 <= category < self.category_count:
                        occupancy[criterion_id][str(category)] += 1
            summaries.append(
                {
                    "option_count": option_count,
                    "variant": variant,
                    "outcome_count": len(outcomes),
                    "status_counts": dict(
                        sorted(Counter(outcome.status for outcome in outcomes).items())
                    ),
                    "passed_count": len(passed),
                    "mean_score": sum(scores) / len(scores) if scores else None,
                    "category_occupancy": occupancy,
                    "gold_scored_count": len(gold_scored),
                    "gold_exact_agreement": (
                        sum(outcome.gold_exact is True for outcome in gold_scored)
                        / len(gold_scored)
                        if gold_scored
                        else None
                    ),
                }
            )
        return tuple(summaries)

    def paired_effects(self) -> tuple[dict[str, Any], ...]:
        grouped: dict[str, list[JudgeCalibrationOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.case.case_id, []).append(outcome)
        effects: list[dict[str, Any]] = []
        for case_id, outcomes in grouped.items():
            baseline = next(outcome for outcome in outcomes if outcome.case.variant == "baseline")
            for control in outcomes:
                if control.case.variant == "baseline":
                    continue
                effect: dict[str, Any] = {
                    "case_id": case_id,
                    "variant": control.case.variant,
                    "option_count": control.case.option_count,
                    "contamination_status": control.case.contamination_status,
                    "baseline_status": baseline.status,
                    "control_status": control.status,
                }
                if baseline.status == control.status == "passed":
                    if baseline.result is None or control.result is None:
                        raise RuntimeError("Passed baseline or control is missing a valid result")
                    effect.update(
                        {
                            "baseline_score": baseline.result.score,
                            "control_score": control.result.score,
                            "score_delta": control.result.score - baseline.result.score,
                            "baseline_categories": dict(baseline.result.criterion_categories or {}),
                            "control_categories": dict(control.result.criterion_categories or {}),
                        }
                    )
                effects.append(effect)
        return tuple(effects)

    def to_dict(self) -> dict[str, Any]:
        gold_scored = [outcome for outcome in self.outcomes if outcome.gold_exact is not None]
        return {
            "category_count": self.category_count,
            "criterion_ids": list(self.criterion_ids),
            "case_count": len({outcome.case.case_id for outcome in self.outcomes}),
            "outcome_count": len(self.outcomes),
            "status_counts": self.status_counts(),
            "category_occupancy": self.category_occupancy(),
            "option_count_summary": list(self.option_count_summary()),
            "option_count_unstratified_count": sum(
                outcome.case.option_count is None for outcome in self.outcomes
            ),
            "gold_scored_count": len(gold_scored),
            "gold_exact_agreement": (
                sum(outcome.gold_exact is True for outcome in gold_scored) / len(gold_scored)
                if gold_scored
                else None
            ),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "paired_effects": list(self.paired_effects()),
        }


def _validate_case_groups(cases: tuple[JudgeCalibrationCase, ...]) -> None:
    grouped: dict[str, list[JudgeCalibrationCase]] = {}
    for case in cases:
        grouped.setdefault(case.case_id, []).append(case)
    for case_id, members in grouped.items():
        variants = {case.variant for case in members}
        if "baseline" not in variants or len(variants) < 2:
            raise ValueError(f"case {case_id!r} requires baseline and at least one control")
        if len(variants) != len(members):
            raise ValueError(f"case {case_id!r} contains duplicate variants")
        statuses = {case.contamination_status for case in members}
        if len(statuses) != 1:
            raise ValueError(f"case {case_id!r} must use one contamination_status")


def _calibration_concurrency(
    judge: Any, case_count: int, category_method: str | None
) -> int:
    """Reuse the gateway limit for one-call methods without multiplying nested calls."""
    if category_method not in {"direct", "cumulative_threshold"}:
        # The implicit/default polytomous method is binary_threshold, which
        # already uses this same capability for its per-boundary calls.
        return 1
    try:
        configured = judge._binary_threshold_concurrency(case_count)
    except Exception:  # noqa: BLE001 - optional capability discovery is fail-closed
        return 1
    if type(configured) is not int or configured < 1:
        return 1
    return min(configured, case_count)


def evaluate_paired_calibration(
    judge: ContextualOrchestratorJudge,
    cases: Iterable[JudgeCalibrationCase],
    *,
    criteria: Iterable[JudgeCriterion | Mapping[str, Any]],
    category_count: int,
    category_method: str | None = None,
) -> JudgeCalibrationReport:
    """Run paired controls through one existing contextual-orchestrator judge.

    Provider errors, parse errors, semantic threshold failures, and invalid IRT
    projections remain explicit outcomes.  No case is retried, repaired, or
    removed from the report.
    """
    if not callable(getattr(judge, "judge", None)):
        raise TypeError("judge must provide judge(...)")
    normalized_cases = tuple(cases)
    if not 1 <= len(normalized_cases) <= MAX_CALIBRATION_CASES:
        raise ValueError(f"cases must contain 1..{MAX_CALIBRATION_CASES} values")
    if any(not isinstance(case, JudgeCalibrationCase) for case in normalized_cases):
        raise TypeError("cases must contain JudgeCalibrationCase values")
    if len({(case.case_id, case.variant) for case in normalized_cases}) != len(normalized_cases):
        raise ValueError("case_id and variant pairs must be unique")
    _validate_case_groups(normalized_cases)
    category_count = _category_count(category_count)
    normalized_criteria = _criteria(criteria)
    if len(normalized_criteria) < 2:
        raise ValueError("calibration requires multiple criterion items for IRT")
    criterion_ids = tuple(sorted(criterion.criterion_id for criterion in normalized_criteria))

    gold_rows: dict[tuple[str, str], tuple[int, ...] | None] = {}
    for case in normalized_cases:
        case_key = (case.case_id, case.variant)
        if case.gold_categories is None:
            gold_rows[case_key] = None
            continue
        if set(case.gold_categories) != set(criterion_ids):
            raise ValueError("gold_categories must contain exactly the rubric criterion ids")
        gold_rows[case_key] = tuple(
            _category(
                case.gold_categories[criterion_id],
                f"gold_categories.{criterion_id}",
                category_count,
            )
            for criterion_id in criterion_ids
        )

    def evaluate_case(case: JudgeCalibrationCase) -> JudgeCalibrationOutcome:
        try:
            result = judge.judge(
                task=case.task,
                answer=case.answer,
                criteria=normalized_criteria,
                reference_answer=case.reference_answer,
                category_count=category_count,
                category_method=category_method,
            )
            if not isinstance(result, LLMJudgeResult):
                raise TypeError("judge.judge(...) must return an LLMJudgeResult")
        except Exception as exc:  # noqa: BLE001 - preserve provider/parse failures in denominator
            return JudgeCalibrationOutcome(
                case=case,
                status="judge_failed",
                error_type=_safe_error_type(exc),
                error="judge_call_failed",
                evidence=getattr(exc, "evidence", None),
            )
        try:
            irt_row = result.to_irt_row(
                item_type="polytomous",
                n_categories=category_count,
            )
        except Exception as exc:  # noqa: BLE001 - invalid projection is calibration evidence
            return JudgeCalibrationOutcome(
                case=case,
                status="irt_failed",
                result=result,
                error_type=_safe_error_type(exc),
                error="irt_projection_failed",
            )
        gold_row = gold_rows[(case.case_id, case.variant)]
        return JudgeCalibrationOutcome(
            case=case,
            status="passed",
            result=result,
            irt_row=irt_row,
            gold_exact=None if gold_row is None else irt_row == gold_row,
        )

    max_workers = _calibration_concurrency(judge, len(normalized_cases), category_method)
    if max_workers == 1:
        outcomes = [evaluate_case(case) for case in normalized_cases]
    else:
        # executor.map preserves the caller's case order even when provider
        # completions finish out of order; failures remain one-for-one.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(evaluate_case, normalized_cases))
    return JudgeCalibrationReport(category_count, criterion_ids, tuple(outcomes))


__all__ = [
    "CALIBRATION_VARIANTS",
    "CONTAMINATION_STATUSES",
    "MAX_CALIBRATION_CASES",
    "JudgeCalibrationCase",
    "JudgeCalibrationOutcome",
    "JudgeCalibrationReport",
    "build_multiple_choice_calibration_cases",
    "evaluate_paired_calibration",
]
