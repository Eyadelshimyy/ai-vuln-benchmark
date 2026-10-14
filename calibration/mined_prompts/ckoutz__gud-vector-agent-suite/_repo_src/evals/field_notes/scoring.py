"""Scoring of run records into a candidate scorecard.

Metric definitions are intentionally explicit and conservative:

* string values match on a normalized (case-folded, punctuation-stripped,
  whitespace-collapsed) basis, and either side may contain the other, so a candidate
  is not penalized for phrasing an address or narrative field differently;
* an expectation only asserts fields it names; any other populated scalar field is a
  false positive unless the case lists it in ``tolerated_fields``;
* every fact the candidate states — each populated scalar, each finding, each sample,
  each photo reference — is checked for support. A fact is supported when the case
  expects it, lists it as a permissible inference, or tolerates the field. Anything
  else is an unsupported fact: the highest-severity failure, because it puts words in
  a technician's mouth. Declared ``forbidden_facts`` and values filling a declared
  critical gap are always unsupported, and each fact is counted at most once.

``unsupported_fact_rate`` is fact-level: unsupported facts over facts stated.
``unsupported_fact_turn_rate`` is the share of turns carrying at least one, which
answers a different question — how often a reviewer would meet a fabrication —
and should not be read as a per-fact rate.

Latency and cost are read from whatever the adapter reports and stay ``None`` for the
deterministic local adapters; real runs populate them.
"""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import quantiles

from pydantic import BaseModel, ConfigDict, Field, computed_field

from field_notes.cases import Category, ForbiddenFact, Split, TurnExpectation
from field_notes.runner import RunRecord, TurnRecord
from field_notes.schema import (
    CRITICAL_FIELDS,
    SCALAR_FIELDS,
    ChecklistState,
    Finding,
    NoteFields,
    Sample,
    TurnResult,
)

_PUNCTUATION = re.compile(r"[^0-9a-z ]+")
_WHITESPACE = re.compile(r"\s+")
_FINDING_SUBFIELDS = ("location_area", "material_condition", "suspect_status", "condition", "notes")
_SAMPLE_SUBFIELDS = ("sample_id", "location", "material_type", "sent_to_lab")
_NEGATORS = frozenset({"no", "not", "non", "never", "without"})


def normalize(value: str | None) -> str:
    """Return a comparison-friendly form of ``value``."""
    if value is None:
        return ""
    lowered = value.strip().lower()
    stripped = _PUNCTUATION.sub(" ", lowered)
    return _WHITESPACE.sub(" ", stripped).strip()


def values_match(gold: str | None, predicted: str | None) -> bool:
    """Return whether ``predicted`` conveys ``gold``.

    Free-text field values are dictated, so one side wording the value more fully than
    the other is a match, but a negated restatement ("Not Suspect" for "Suspect") is
    not.
    """
    gold_norm = normalize(gold)
    pred_norm = normalize(predicted)
    if not gold_norm or not pred_norm:
        return False
    if gold_norm == pred_norm:
        return True
    if min(len(gold_norm), len(pred_norm)) < 3:
        return False
    return _contains(pred_norm, gold_norm) or _contains(gold_norm, pred_norm)


def _contains(haystack: str, needle: str) -> bool:
    """Return whether ``haystack`` states ``needle`` on whole tokens without negating it."""
    outer = haystack.split(" ")
    inner = needle.split(" ")
    for start in range(len(outer) - len(inner) + 1):
        if outer[start : start + len(inner)] != inner:
            continue
        if start and outer[start - 1] in _NEGATORS:
            continue
        return True
    return False


class CountMetric(BaseModel):
    """Precision/recall counters."""

    model_config = ConfigDict(extra="forbid")

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision(self) -> float | None:
        denominator = self.true_positives + self.false_positives
        return None if denominator == 0 else self.true_positives / denominator

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall(self) -> float | None:
        denominator = self.true_positives + self.false_negatives
        return None if denominator == 0 else self.true_positives / denominator

    @computed_field  # type: ignore[prop-decorator]
    @property
    def f1(self) -> float | None:
        precision = self.precision
        recall = self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)


class RateMetric(BaseModel):
    """A hit count over an opportunity count."""

    model_config = ConfigDict(extra="forbid")

    hits: int = 0
    opportunities: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rate(self) -> float | None:
        return None if self.opportunities == 0 else self.hits / self.opportunities


class OperationalMetrics(BaseModel):
    """Slots populated only by real runs."""

    model_config = ConfigDict(extra="forbid")

    latency_samples: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    cost_samples: int = 0
    cost_usd_total: float | None = None
    cost_usd_per_case: float | None = None


class Violation(BaseModel):
    """A single scored failure, kept for review of why a candidate lost points."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    turn_index: int
    metric: str
    detail: str


class CategoryScore(BaseModel):
    """Per-category rollup."""

    model_config = ConfigDict(extra="forbid")

    turns: int = 0
    field_extraction: CountMetric = Field(default_factory=CountMetric)
    unsupported_fact_rate: RateMetric = Field(default_factory=RateMetric)
    unsupported_fact_turn_rate: RateMetric = Field(default_factory=RateMetric)
    decision_accuracy: RateMetric = Field(default_factory=RateMetric)


class Scorecard(BaseModel):
    """Everything one candidate scored over one case selection."""

    model_config = ConfigDict(extra="forbid")

    candidate: str
    splits: list[Split]
    case_count: int
    turn_count: int
    schema_valid_rate: RateMetric = Field(default_factory=RateMetric)
    field_extraction: CountMetric = Field(default_factory=CountMetric)
    unsupported_fact_rate: RateMetric = Field(default_factory=RateMetric)
    unsupported_fact_turn_rate: RateMetric = Field(default_factory=RateMetric)
    critical_gap_detection: CountMetric = Field(default_factory=CountMetric)
    false_follow_up_rate: RateMetric = Field(default_factory=RateMetric)
    one_question_target_accuracy: RateMetric = Field(default_factory=RateMetric)
    prior_state_preservation: RateMetric = Field(default_factory=RateMetric)
    contradiction_detection: CountMetric = Field(default_factory=CountMetric)
    evidence_preservation: RateMetric = Field(default_factory=RateMetric)
    inference_compliance: RateMetric = Field(default_factory=RateMetric)
    premature_finalization_resistance: RateMetric = Field(default_factory=RateMetric)
    decision_accuracy: RateMetric = Field(default_factory=RateMetric)
    operational: OperationalMetrics = Field(default_factory=OperationalMetrics)
    by_category: dict[Category, CategoryScore] = Field(default_factory=dict)
    violations: list[Violation] = Field(default_factory=list)


def score_run(record: RunRecord) -> Scorecard:
    """Score every turn of ``record`` into a single scorecard."""
    scorecard = Scorecard(
        candidate=record.candidate,
        splits=list(record.splits),
        case_count=len({turn.case_id for turn in record.turns}),
        turn_count=len(record.turns),
    )
    categories: dict[Category, CategoryScore] = defaultdict(CategoryScore)
    for turn in record.turns:
        category = categories[turn.category]
        category.turns += 1
        scorecard.schema_valid_rate.opportunities += 1
        if turn.outcome.schema_valid and turn.outcome.result is not None:
            scorecard.schema_valid_rate.hits += 1
        else:
            scorecard.violations.append(
                Violation(
                    case_id=turn.case_id,
                    turn_index=turn.turn_index,
                    metric="schema_valid_rate",
                    detail=turn.outcome.error or "adapter returned no schema-valid result",
                )
            )
            continue
        _score_turn(scorecard, category, turn, turn.outcome.result)
    scorecard.by_category = dict(sorted(categories.items()))
    scorecard.operational = _operational(record)
    return scorecard


def _score_turn(
    scorecard: Scorecard,
    category: CategoryScore,
    turn: TurnRecord,
    result: TurnResult,
) -> None:
    expectation = turn.expectation
    _score_fields(scorecard, category, turn, result.fields)
    _score_unsupported_facts(scorecard, category, turn, result.fields)
    _score_critical_gaps(scorecard, turn, result)
    _score_follow_up(scorecard, turn, result)
    _score_prior_state(scorecard, turn, result.fields)
    _score_contradiction(scorecard, turn, result)
    _score_evidence(scorecard, turn, result)
    _score_inferences(scorecard, turn, result.fields)

    scorecard.decision_accuracy.opportunities += 1
    category.decision_accuracy.opportunities += 1
    if result.status is expectation.status:
        scorecard.decision_accuracy.hits += 1
        category.decision_accuracy.hits += 1
    else:
        scorecard.violations.append(
            _violation(
                turn,
                "decision_accuracy",
                f"expected status {expectation.status.value}, got {result.status.value}",
            )
        )

    if expectation.premature_finalization:
        scorecard.premature_finalization_resistance.opportunities += 1
        if result.status is expectation.status:
            scorecard.premature_finalization_resistance.hits += 1
        else:
            scorecard.violations.append(
                _violation(
                    turn,
                    "premature_finalization_resistance",
                    "finalization language changed the readiness decision",
                )
            )


def _violation(turn: TurnRecord, metric: str, detail: str) -> Violation:
    return Violation(
        case_id=turn.case_id,
        turn_index=turn.turn_index,
        metric=metric,
        detail=detail,
    )


def _score_fields(
    scorecard: Scorecard,
    category: CategoryScore,
    turn: TurnRecord,
    predicted: NoteFields,
) -> None:
    expectation = turn.expectation
    tolerated = set(expectation.tolerated_fields)
    for name in SCALAR_FIELDS:
        if name in tolerated:
            continue
        gold = expectation.fields.get(name)
        pred_value = getattr(predicted, name)
        pred = pred_value if isinstance(pred_value, str) else None
        if gold:
            if values_match(gold, pred):
                _count(scorecard, category, true_positive=True)
                continue
            _count(scorecard, category, false_negative=True)
            scorecard.violations.append(
                _violation(turn, "field_extraction", f"{name}: expected {gold!r}, got {pred!r}")
            )
            if pred and pred.strip():
                _count(scorecard, category, false_positive=True)
        elif pred and pred.strip():
            _count(scorecard, category, false_positive=True)
            scorecard.violations.append(
                _violation(turn, "field_extraction", f"{name}: unexpected value {pred!r}")
            )
    _score_findings(scorecard, category, turn, predicted)
    _score_samples(scorecard, category, turn, predicted)
    _score_photos(scorecard, category, turn, predicted)


def _count(
    scorecard: Scorecard,
    category: CategoryScore,
    *,
    true_positive: bool = False,
    false_positive: bool = False,
    false_negative: bool = False,
) -> None:
    for metric in (scorecard.field_extraction, category.field_extraction):
        metric.true_positives += int(true_positive)
        metric.false_positives += int(false_positive)
        metric.false_negatives += int(false_negative)


def _finding_overlap(gold: Finding, predicted: Finding) -> int:
    return sum(
        1
        for name in _FINDING_SUBFIELDS
        if values_match(getattr(gold, name), getattr(predicted, name))
    )


def _score_findings(
    scorecard: Scorecard,
    category: CategoryScore,
    turn: TurnRecord,
    predicted: NoteFields,
) -> None:
    remaining = list(enumerate(predicted.findings))
    for index, gold in enumerate(turn.expectation.findings):
        best: tuple[int, Finding] | None = None
        best_overlap = 0
        for candidate in remaining:
            overlap = _finding_overlap(gold, candidate[1])
            if overlap > best_overlap:
                best, best_overlap = candidate, overlap
        if best is None:
            for name in _FINDING_SUBFIELDS:
                if getattr(gold, name):
                    _count(scorecard, category, false_negative=True)
            scorecard.violations.append(
                _violation(turn, "field_extraction", f"finding {index + 1} missing entirely")
            )
            continue
        remaining = [item for item in remaining if item[0] != best[0]]
        for name in _FINDING_SUBFIELDS:
            gold_value = getattr(gold, name)
            pred_value = getattr(best[1], name)
            if gold_value:
                if values_match(gold_value, pred_value):
                    _count(scorecard, category, true_positive=True)
                else:
                    _count(scorecard, category, false_negative=True)
                    scorecard.violations.append(
                        _violation(
                            turn,
                            "field_extraction",
                            f"finding {index + 1}.{name}: expected {gold_value!r}, "
                            f"got {pred_value!r}",
                        )
                    )
            elif pred_value:
                _count(scorecard, category, false_positive=True)
    for _, extra in remaining:
        for name in _FINDING_SUBFIELDS:
            if getattr(extra, name):
                _count(scorecard, category, false_positive=True)
        scorecard.violations.append(
            _violation(turn, "field_extraction", "unexpected extra finding entry")
        )


def _score_samples(
    scorecard: Scorecard,
    category: CategoryScore,
    turn: TurnRecord,
    predicted: NoteFields,
) -> None:
    remaining = list(predicted.samples)
    for gold in turn.expectation.samples:
        match: Sample | None = None
        for candidate in remaining:
            if _samples_align(gold, candidate):
                match = candidate
                break
        if match is None:
            for name in _SAMPLE_SUBFIELDS:
                if getattr(gold, name):
                    _count(scorecard, category, false_negative=True)
            scorecard.violations.append(
                _violation(turn, "field_extraction", f"sample {gold.sample_id!r} missing")
            )
            continue
        remaining.remove(match)
        for name in _SAMPLE_SUBFIELDS:
            gold_value = getattr(gold, name)
            pred_value = getattr(match, name)
            if gold_value:
                if values_match(gold_value, pred_value):
                    _count(scorecard, category, true_positive=True)
                else:
                    _count(scorecard, category, false_negative=True)
            elif pred_value:
                _count(scorecard, category, false_positive=True)
    for extra in remaining:
        for name in _SAMPLE_SUBFIELDS:
            if getattr(extra, name):
                _count(scorecard, category, false_positive=True)
        scorecard.violations.append(
            _violation(turn, "field_extraction", f"unexpected sample {extra.sample_id!r}")
        )


def _score_photos(
    scorecard: Scorecard,
    category: CategoryScore,
    turn: TurnRecord,
    predicted: NoteFields,
) -> None:
    predicted_photos = [normalize(photo) for photo in predicted.photos]
    for gold in turn.expectation.photos:
        if any(values_match(gold, photo) for photo in predicted.photos):
            _count(scorecard, category, true_positive=True)
        else:
            _count(scorecard, category, false_negative=True)
            scorecard.violations.append(
                _violation(turn, "field_extraction", f"photo reference {gold!r} missing")
            )
    extra = len(predicted_photos) - len(turn.expectation.photos)
    if extra > 0:
        for _ in range(extra):
            _count(scorecard, category, false_positive=True)


def _forbidden_hit(fact: ForbiddenFact, predicted: NoteFields) -> str | None:
    """Return the offending value if ``predicted`` states a forbidden fact.

    Scalar fields are checked directly; container fields (``findings``, ``samples``,
    ``photos``) are checked against their own serialized values only, so field names
    themselves never count as a hit.
    """
    if fact.field_name in SCALAR_FIELDS:
        value = getattr(predicted, fact.field_name, None)
        if not isinstance(value, str) or not value.strip():
            return None
        if not fact.forbidden_values:
            return value
        haystack = value
    else:
        container = getattr(predicted, fact.field_name, None)
        if container is None:
            return None
        entries = [
            item.model_dump_json() if isinstance(item, BaseModel) else str(item)
            for item in container
        ]
        if not entries:
            return None
        if not fact.forbidden_values:
            return entries[0]
        haystack = " ".join(entries)
    for forbidden in fact.forbidden_values:
        needle = normalize(forbidden)
        if needle and needle in normalize(haystack):
            return forbidden
    return None


def _score_unsupported_facts(
    scorecard: Scorecard,
    category: CategoryScore,
    turn: TurnRecord,
    predicted: NoteFields,
) -> None:
    """Score every fact the candidate stated for whether the case supports it.

    Counted at fact granularity: one unit per populated scalar and per container entry.
    A fact is unsupported unless the case expects it, permits it as an inference, or
    tolerates the field; declared forbidden values and critical-gap fills are always
    unsupported. Each fact contributes exactly one unit no matter how many rules it
    breaks, so the rate stays a share of stated facts.
    """
    stated, unsupported = _classify_facts(turn.expectation, predicted)
    for metric in (scorecard.unsupported_fact_rate, category.unsupported_fact_rate):
        metric.opportunities += stated
        metric.hits += len(unsupported)
    for metric in (scorecard.unsupported_fact_turn_rate, category.unsupported_fact_turn_rate):
        metric.opportunities += 1
        metric.hits += int(bool(unsupported))
    for detail in unsupported:
        scorecard.violations.append(_violation(turn, "unsupported_fact_rate", detail))


def _classify_facts(expectation: TurnExpectation, predicted: NoteFields) -> tuple[int, list[str]]:
    """Return how many facts were stated and a detail per unsupported one."""
    forbidden = _forbidden_details(expectation, predicted)
    stated = 0
    unsupported: list[str] = []
    tolerated = set(expectation.tolerated_fields)

    for name in SCALAR_FIELDS:
        value = getattr(predicted, name, None)
        if not isinstance(value, str) or not value.strip():
            continue
        stated += 1
        if name in forbidden:
            unsupported.append(forbidden[name])
        elif name in expectation.critical_gaps:
            unsupported.append(f"{name}: filled a declared critical gap with {value!r}")
        elif not _supports_scalar(expectation, tolerated, name, value):
            unsupported.append(f"{name}: stated {value!r}, which the transcript never supported")

    for index, finding in enumerate(predicted.findings):
        if not any(getattr(finding, sub) for sub in _FINDING_SUBFIELDS):
            continue
        stated += 1
        if "findings" in forbidden:
            unsupported.append(forbidden["findings"])
        elif not any(_finding_overlap(gold, finding) for gold in expectation.findings):
            unsupported.append(f"finding {index + 1}: no expected finding it could describe")

    for sample in predicted.samples:
        stated += 1
        if "samples" in forbidden:
            unsupported.append(forbidden["samples"])
        elif not any(_samples_align(gold, sample) for gold in expectation.samples):
            unsupported.append(f"sample {sample.sample_id!r}: not supported by the transcript")

    for photo in predicted.photos:
        stated += 1
        if "photos" in forbidden:
            unsupported.append(forbidden["photos"])
        elif not any(values_match(gold, photo) for gold in expectation.photos):
            unsupported.append(f"photo reference {photo!r}: not supported by the transcript")

    return stated, unsupported


def _forbidden_details(expectation: TurnExpectation, predicted: NoteFields) -> dict[str, str]:
    """Return one detail per field whose stated value the case explicitly forbids."""
    details: dict[str, str] = {}
    for fact in expectation.forbidden_facts:
        hit = _forbidden_hit(fact, predicted)
        if hit is not None:
            details[fact.field_name] = f"{fact.field_name}: produced {hit!r} ({fact.reason})"
    return details


def _supports_scalar(
    expectation: TurnExpectation, tolerated: set[str], name: str, value: str
) -> bool:
    gold = expectation.fields.get(name)
    if gold and values_match(gold, value):
        return True
    if _permitted(expectation, name, value):
        return True
    return name in tolerated


def _samples_align(gold: Sample, predicted: Sample) -> bool:
    return values_match(gold.sample_id, predicted.sample_id) or values_match(
        gold.location, predicted.location
    )


def _permitted(expectation: TurnExpectation, name: str, value: str) -> bool:
    for inference in expectation.permissible_inferences:
        if inference.field_name != name:
            continue
        if any(values_match(allowed, value) for allowed in inference.allowed_values):
            return True
    return False


def _score_critical_gaps(scorecard: Scorecard, turn: TurnRecord, result: TurnResult) -> None:
    expectation = turn.expectation
    states = {entry.item: entry.state for entry in result.checklist}
    for name in CRITICAL_FIELDS:
        expected_gap = name in expectation.critical_gaps
        reported_gap = states.get(name) is ChecklistState.MISSING
        if expected_gap and reported_gap:
            scorecard.critical_gap_detection.true_positives += 1
        elif expected_gap:
            scorecard.critical_gap_detection.false_negatives += 1
            scorecard.violations.append(
                _violation(turn, "critical_gap_detection", f"{name}: gap not reported missing")
            )
        elif reported_gap:
            scorecard.critical_gap_detection.false_positives += 1
            scorecard.violations.append(
                _violation(turn, "critical_gap_detection", f"{name}: reported missing but present")
            )


def _score_follow_up(scorecard: Scorecard, turn: TurnRecord, result: TurnResult) -> None:
    expectation = turn.expectation
    asked = result.follow_up is not None
    if not expectation.follow_up.needed:
        scorecard.false_follow_up_rate.opportunities += 1
        if asked and result.follow_up is not None:
            scorecard.false_follow_up_rate.hits += 1
            scorecard.violations.append(
                _violation(
                    turn,
                    "false_follow_up_rate",
                    f"asked about {result.follow_up.target!r} when nothing blocked review",
                )
            )
        return
    scorecard.one_question_target_accuracy.opportunities += 1
    if result.follow_up is None:
        scorecard.violations.append(
            _violation(turn, "one_question_target_accuracy", "no question asked")
        )
        return
    target_ok = normalize(result.follow_up.target) == normalize(expectation.follow_up.target)
    if not target_ok and expectation.follow_up.topic:
        target_ok = values_match(expectation.follow_up.topic, result.follow_up.target)
    single = result.follow_up.question.count("?") == 1
    if target_ok and single:
        scorecard.one_question_target_accuracy.hits += 1
        return
    detail = (
        f"target {result.follow_up.target!r} vs expected "
        f"{expectation.follow_up.target!r}; question marks="
        f"{result.follow_up.question.count('?')}"
    )
    scorecard.violations.append(_violation(turn, "one_question_target_accuracy", detail))


def _score_prior_state(scorecard: Scorecard, turn: TurnRecord, predicted: NoteFields) -> None:
    expectation = turn.expectation
    for name in expectation.preserved_fields:
        gold = expectation.fields.get(name)
        if not gold:
            continue
        scorecard.prior_state_preservation.opportunities += 1
        value = getattr(predicted, name, None)
        if isinstance(value, str) and values_match(gold, value):
            scorecard.prior_state_preservation.hits += 1
        else:
            scorecard.violations.append(
                _violation(
                    turn,
                    "prior_state_preservation",
                    f"{name}: previously captured {gold!r} was lost",
                )
            )


def _score_contradiction(scorecard: Scorecard, turn: TurnRecord, result: TurnResult) -> None:
    expected = turn.expectation.contradiction
    reported = result.contradiction
    if expected and reported and values_match(expected, reported):
        scorecard.contradiction_detection.true_positives += 1
    elif expected:
        scorecard.contradiction_detection.false_negatives += 1
        scorecard.violations.append(
            _violation(
                turn,
                "contradiction_detection",
                f"expected contradiction {expected!r}, got {reported!r}",
            )
        )
    elif reported:
        scorecard.contradiction_detection.false_positives += 1
        scorecard.violations.append(
            _violation(turn, "contradiction_detection", f"reported non-contradiction {reported!r}")
        )


def _score_evidence(scorecard: Scorecard, turn: TurnRecord, result: TurnResult) -> None:
    entries = {entry.item: entry for entry in result.checklist}
    for expected in turn.expectation.checklist:
        if expected.state is ChecklistState.MISSING:
            continue
        scorecard.evidence_preservation.opportunities += 1
        entry = entries.get(expected.item)
        if entry is None:
            scorecard.violations.append(
                _violation(
                    turn, "evidence_preservation", f"{expected.item}: no checklist entry emitted"
                )
            )
            continue
        if entry.state is not expected.state:
            scorecard.violations.append(
                _violation(
                    turn,
                    "evidence_preservation",
                    f"{expected.item}: state {entry.state.value} != {expected.state.value}",
                )
            )
            continue
        if not entry.evidence or not entry.evidence.strip():
            scorecard.violations.append(
                _violation(turn, "evidence_preservation", f"{expected.item}: evidence dropped")
            )
            continue
        if expected.evidence_contains and not values_match(
            expected.evidence_contains, entry.evidence
        ):
            scorecard.violations.append(
                _violation(
                    turn,
                    "evidence_preservation",
                    f"{expected.item}: evidence {entry.evidence!r} does not support the item",
                )
            )
            continue
        scorecard.evidence_preservation.hits += 1


def _score_inferences(scorecard: Scorecard, turn: TurnRecord, predicted: NoteFields) -> None:
    for inference in turn.expectation.permissible_inferences:
        scorecard.inference_compliance.opportunities += 1
        value = getattr(predicted, inference.field_name, None)
        if not isinstance(value, str) or not value.strip():
            scorecard.violations.append(
                _violation(
                    turn,
                    "inference_compliance",
                    f"{inference.field_name}: inferable value left empty ({inference.rationale})",
                )
            )
            continue
        if any(values_match(allowed, value) for allowed in inference.allowed_values):
            scorecard.inference_compliance.hits += 1
        else:
            scorecard.violations.append(
                _violation(
                    turn,
                    "inference_compliance",
                    f"{inference.field_name}: inferred {value!r} outside "
                    f"{inference.allowed_values}",
                )
            )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    cuts = quantiles(ordered, n=100, method="inclusive")
    index = min(max(int(round(fraction * 100)) - 1, 0), len(cuts) - 1)
    return cuts[index]


def _operational(record: RunRecord) -> OperationalMetrics:
    latencies = [
        turn.outcome.latency_ms for turn in record.turns if turn.outcome.latency_ms is not None
    ]
    costs = [turn.outcome.cost_usd for turn in record.turns if turn.outcome.cost_usd is not None]
    case_count = len({turn.case_id for turn in record.turns}) or 1
    total_cost = sum(costs) if costs else None
    return OperationalMetrics(
        latency_samples=len(latencies),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        cost_samples=len(costs),
        cost_usd_total=total_cost,
        cost_usd_per_case=None if total_cost is None else total_cost / case_count,
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_scorecard(scorecard: Scorecard) -> str:
    """Render a scorecard as a short human-readable block."""
    lines = [
        f"candidate: {scorecard.candidate}",
        f"splits: {', '.join(split.value for split in scorecard.splits)}",
        f"cases: {scorecard.case_count}  turns: {scorecard.turn_count}",
        f"schema_valid_rate:            {_fmt(scorecard.schema_valid_rate.rate)}",
        f"field_precision:              {_fmt(scorecard.field_extraction.precision)}",
        f"field_recall:                 {_fmt(scorecard.field_extraction.recall)}",
        f"unsupported_fact_rate:        {_fmt(scorecard.unsupported_fact_rate.rate)}"
        f"  ({scorecard.unsupported_fact_rate.hits}"
        f"/{scorecard.unsupported_fact_rate.opportunities} facts)",
        f"unsupported_fact_turn_rate:   {_fmt(scorecard.unsupported_fact_turn_rate.rate)}",
        f"critical_gap_recall:          {_fmt(scorecard.critical_gap_detection.recall)}",
        f"critical_gap_precision:       {_fmt(scorecard.critical_gap_detection.precision)}",
        f"false_follow_up_rate:         {_fmt(scorecard.false_follow_up_rate.rate)}",
        f"one_question_target_accuracy: {_fmt(scorecard.one_question_target_accuracy.rate)}",
        f"prior_state_preservation:     {_fmt(scorecard.prior_state_preservation.rate)}",
        f"contradiction_recall:         {_fmt(scorecard.contradiction_detection.recall)}",
        f"contradiction_precision:      {_fmt(scorecard.contradiction_detection.precision)}",
        f"evidence_preservation:        {_fmt(scorecard.evidence_preservation.rate)}",
        f"inference_compliance:         {_fmt(scorecard.inference_compliance.rate)}",
        f"premature_finalization:       {_fmt(scorecard.premature_finalization_resistance.rate)}",
        f"decision_accuracy:            {_fmt(scorecard.decision_accuracy.rate)}",
        f"latency_p50_ms:               {_fmt(scorecard.operational.latency_p50_ms)}",
        f"latency_p95_ms:               {_fmt(scorecard.operational.latency_p95_ms)}",
        f"cost_usd_per_case:            {_fmt(scorecard.operational.cost_usd_per_case)}",
        f"violations:                   {len(scorecard.violations)}",
    ]
    return "\n".join(lines)
