"""Strict redacted parsing of untrusted provider-generated assessment items."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, TypeAlias

from .generation import GenerationRequest, MAX_RAW_RESPONSE_CHARACTERS
from .models import (
    EvidenceMode,
    ResponseFormat,
    SCHEMA_VERSION,
    _IDENTIFIER_PATTERN,
    _sha256_hex,
)

MAX_CANDIDATE_TEXT_CHARACTERS = 8_192
MAX_CANDIDATE_COLLECTION_VALUES = 32
MAX_OPTIONS = MAX_CANDIDATE_COLLECTION_VALUES
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 50_000
_MAX_OPTION_ID_LENGTH = 128

_PROVENANCE_FIELDS = frozenset(
    {
        "blueprint_id",
        "blueprint_handle",
        "blueprint_fingerprint",
        "rubric_id",
        "rubric_version",
        "rubric_fingerprint",
    }
)
_REQUIRED_FIELDS = frozenset(
    {
        *_PROVENANCE_FIELDS,
        "item_id",
        "stem",
        "stimulus",
        "response_format",
        "options",
        "answer_key",
        "scoring_guide",
        "rubric_alignment",
        "source_attributions",
        "safety_notes",
    }
)
_CANDIDATE_PROOF_SEAL = object()


class CandidateValidationError(ValueError):
    """Redacted structured failure raised for untrusted candidate content."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Store stable machine-readable failure metadata without rejected values."""
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


class _DuplicateJsonKey(Exception):
    """Internal signal used to reject duplicate JSON object member names."""


class _NonFiniteJsonNumber(Exception):
    """Internal signal used to reject non-standard non-finite JSON constants."""


def _error(code: str, path: str, message: str) -> CandidateValidationError:
    """Construct one redacted candidate-validation error."""
    return CandidateValidationError(code, path, message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate member names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> None:
    """Reject NaN and infinity tokens accepted by Python's permissive decoder."""
    raise _NonFiniteJsonNumber


def _validate_raw_json_depth(content: str) -> None:
    """Reject string JSON whose nesting depth exceeds the maximum budget."""
    depth = 0
    in_string = False
    escaped = False
    for char in content:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise _error(
                    "json_too_deep",
                    "$",
                    f"JSON nesting exceeds the maximum depth of {MAX_JSON_DEPTH}",
                )
        elif char in "]}":
            depth -= 1


def _validate_json_depth(value: Any) -> None:
    """Reject provider JSON whose container nesting or node count exceeds a budget."""
    stack: list[tuple[Any, int]] = [(value, 1)]
    node_count = 0
    while stack:
        current, depth = stack.pop()
        node_count += 1
        if node_count > MAX_JSON_NODES:
            raise _error(
                "json_node_budget",
                "$",
                f"JSON node count exceeds the maximum of {MAX_JSON_NODES}",
            )
        if depth > MAX_JSON_DEPTH:
            raise _error(
                "json_too_deep",
                "$",
                f"JSON nesting exceeds the maximum depth of {MAX_JSON_DEPTH}",
            )
        if isinstance(current, dict):
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)


def _object(value: Any, path: str) -> dict[str, Any]:
    """Require a JSON object at ``path``."""
    if not isinstance(value, dict):
        raise _error("invalid_type", path, "value must be an object")
    return value


def _array(
    value: Any,
    path: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_CANDIDATE_COLLECTION_VALUES,
) -> list[Any]:
    """Require a bounded JSON array at ``path``."""
    if not isinstance(value, list):
        raise _error("invalid_type", path, "value must be an array")
    if len(value) < minimum:
        raise _error("collection_too_small", path, "array has too few values")
    if len(value) > maximum:
        raise _error("collection_too_large", path, "array has too many values")
    return value


def _exact_fields(
    value: dict[str, Any],
    expected: Iterable[str],
    path: str,
) -> None:
    """Reject missing and unknown object fields without disclosing values."""
    expected_set = set(expected)
    missing = sorted(expected_set.difference(value))
    if missing:
        raise _error(
            "missing_field",
            f"{path}.{missing[0]}",
            "required field is missing",
        )
    unknown = sorted(set(value).difference(expected_set))
    if unknown:
        raise _error(
            "unknown_field",
            f"{path}.{unknown[0]}",
            "field is not allowed",
        )


def _string(
    value: Any,
    path: str,
    *,
    maximum: int = MAX_CANDIDATE_TEXT_CHARACTERS,
) -> str:
    """Normalize bounded non-empty candidate text."""
    if not isinstance(value, str):
        raise _error("invalid_type", path, "value must be a string")
    normalized = value.strip()
    if not normalized:
        raise _error("invalid_text", path, "text must not be empty")
    if len(normalized) > maximum:
        raise _error("text_too_large", path, "text exceeds the allowed size")
    return normalized


def _identifier(value: Any, path: str) -> str:
    """Normalize a candidate identifier using the repository naming contract."""
    normalized = _string(value, path, maximum=_MAX_OPTION_ID_LENGTH)
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise _error(
            "invalid_identifier",
            path,
            "identifier must use two-or-more-token lower snake_case",
        )
    return normalized


def _fingerprint(value: Any, path: str) -> str:
    """Normalize a lower hexadecimal SHA-256 provenance fingerprint."""
    normalized = _string(value, path, maximum=64)
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise _error(
            "invalid_fingerprint",
            path,
            "fingerprint must be 64 lower hexadecimal characters",
        )
    return normalized


def _integer(value: Any, path: str) -> int:
    """Require a JSON integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error("invalid_type", path, "value must be an integer")
    return value


def _boolean(value: Any, path: str) -> bool:
    """Require a JSON boolean without numeric coercion."""
    if not isinstance(value, bool):
        raise _error("invalid_type", path, "value must be a boolean")
    return value


def _string_array(
    value: Any,
    path: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_CANDIDATE_COLLECTION_VALUES,
    unique: bool = False,
    identifier_values: bool = False,
) -> tuple[str, ...]:
    """Normalize a bounded candidate string array."""
    raw = _array(value, path, minimum=minimum, maximum=maximum)
    validator = _identifier if identifier_values else _string
    normalized = tuple(
        validator(item, f"{path}[{index}]") for index, item in enumerate(raw)
    )
    if unique and len(set(normalized)) != len(normalized):
        raise _error("duplicate_value", path, "array values must be unique")
    return normalized


@dataclass(frozen=True)
class GeneratedOption:
    """One normalized option in a selected or comparative response item."""

    option_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible option representation."""
        return {"option_id": self.option_id, "text": self.text}


@dataclass(frozen=True)
class ConstructedAnswerKey:
    """Reference response, accepted variants, and rationale for free text."""

    reference_response: str
    accepted_variants: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return the generation-contract answer-key shape."""
        return {
            "reference_response": self.reference_response,
            "accepted_variants": list(self.accepted_variants),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SelectedAnswerKey:
    """One or more correct option identifiers with an audit rationale."""

    option_ids: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return the generation-contract answer-key shape."""
        return {"option_ids": list(self.option_ids), "rationale": self.rationale}


@dataclass(frozen=True)
class BinaryAnswerKey:
    """Boolean target value with an audit rationale."""

    value: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return the generation-contract answer-key shape."""
        return {"value": self.value, "rationale": self.rationale}


@dataclass(frozen=True)
class OrdinalAnswerKey:
    """Rubric score target with an audit rationale."""

    score: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return the generation-contract answer-key shape."""
        return {"score": self.score, "rationale": self.rationale}


@dataclass(frozen=True)
class PairwiseAnswerKey:
    """Ordered pairwise outcome, optional winner, and audit rationale."""

    outcome: str
    preferred_option_id: str | None
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return the generation-contract answer-key shape."""
        return {
            "outcome": self.outcome,
            "preferred_option_id": self.preferred_option_id,
            "rationale": self.rationale,
        }


GeneratedAnswerKey: TypeAlias = (
    ConstructedAnswerKey
    | SelectedAnswerKey
    | BinaryAnswerKey
    | OrdinalAnswerKey
    | PairwiseAnswerKey
)


@dataclass(frozen=True)
class ScoreGuideEntry:
    """Observable scoring evidence and rationale for one exact rubric score."""

    score: int
    evidence: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible scoring-guide representation."""
        return {
            "score": self.score,
            "evidence": self.evidence,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class RubricAlignmentEntry:
    """Candidate-authored observable indicators aligned to one rubric score."""

    score: int
    observable_indicators: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible rubric-alignment representation."""
        return {
            "score": self.score,
            "observable_indicators": list(self.observable_indicators),
        }


@dataclass(frozen=True)
class SourceAttribution:
    """One exact evidence span attributed to a supplied source document."""

    source_id: str
    evidence_span: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible source-attribution representation."""
        return {
            "source_id": self.source_id,
            "evidence_span": self.evidence_span,
        }


@dataclass(frozen=True)
class _CandidateValidationProof:
    """Private parser-issued seal over exact normalized candidate content."""

    seal: object = field(repr=False, compare=False)
    candidate_fingerprint: str

    def __post_init__(self) -> None:
        """Reject proof objects not issued inside this module."""
        if self.seal is not _CANDIDATE_PROOF_SEAL:
            raise ValueError("candidate validation proof is invalid")


@dataclass(frozen=True)
class GeneratedItemCandidate:
    """Immutable candidate that passed structural and source-grounding validation."""

    request_id: str
    request_fingerprint: str
    contract_id: str
    contract_handle: str
    contract_fingerprint: str
    blueprint_id: str
    blueprint_handle: str
    blueprint_fingerprint: str
    rubric_id: str
    rubric_version: str
    rubric_fingerprint: str
    item_id: str
    stem: str
    stimulus: tuple[str, ...]
    response_format: ResponseFormat
    options: tuple[GeneratedOption, ...]
    answer_key: GeneratedAnswerKey
    scoring_guide: tuple[ScoreGuideEntry, ...]
    rubric_alignment: tuple[RubricAlignmentEntry, ...]
    source_attributions: tuple[SourceAttribution, ...]
    safety_notes: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION
    _validation_proof: _CandidateValidationProof | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Reject direct construction outside the strict parser boundary."""
        self._verify_validation_proof()

    @classmethod
    def _from_validated(cls, **values: Any) -> GeneratedItemCandidate:
        """Construct one candidate only after every parser gate has succeeded."""
        expected_fields = set(cls.__dataclass_fields__).difference(
            {"_validation_proof"}
        )
        if set(values) != expected_fields:
            raise RuntimeError("validated candidate factory received invalid fields")
        candidate = object.__new__(cls)
        for field_name in expected_fields:
            object.__setattr__(candidate, field_name, values[field_name])
        proof = _CandidateValidationProof(
            seal=_CANDIDATE_PROOF_SEAL,
            candidate_fingerprint=_sha256_hex(candidate._content_dict()),
        )
        object.__setattr__(candidate, "_validation_proof", proof)
        candidate._verify_validation_proof()
        return candidate

    def _content_dict(self) -> dict[str, Any]:
        """Return normalized candidate content without its derived fingerprint."""
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "request_fingerprint": self.request_fingerprint,
            "contract_id": self.contract_id,
            "contract_handle": self.contract_handle,
            "contract_fingerprint": self.contract_fingerprint,
            "blueprint_id": self.blueprint_id,
            "blueprint_handle": self.blueprint_handle,
            "blueprint_fingerprint": self.blueprint_fingerprint,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "rubric_fingerprint": self.rubric_fingerprint,
            "item_id": self.item_id,
            "stem": self.stem,
            "stimulus": list(self.stimulus),
            "response_format": self.response_format.value,
            "options": [option.to_dict() for option in self.options],
            "answer_key": self.answer_key.to_dict(),
            "scoring_guide": [entry.to_dict() for entry in self.scoring_guide],
            "rubric_alignment": [entry.to_dict() for entry in self.rubric_alignment],
            "source_attributions": [
                attribution.to_dict() for attribution in self.source_attributions
            ],
            "safety_notes": list(self.safety_notes),
        }

    def _computed_candidate_fingerprint(self) -> str:
        """Return the digest of the candidate's current normalized content."""
        return _sha256_hex(self._content_dict())

    def _verify_validation_proof(self) -> None:
        """Fail closed unless parser proof matches every current candidate field."""
        proof = self._validation_proof
        if not isinstance(proof, _CandidateValidationProof):
            raise ValueError(
                "candidate provenance requires parse_generated_item_candidate"
            )
        if proof.seal is not _CANDIDATE_PROOF_SEAL:
            raise ValueError("candidate provenance proof is invalid")
        if proof.candidate_fingerprint != self._computed_candidate_fingerprint():
            raise ValueError("candidate provenance does not match validated content")

    @property
    def candidate_fingerprint(self) -> str:
        """Return SHA-256 over canonical normalized candidate content."""
        self._verify_validation_proof()
        proof = self._validation_proof
        if proof is None:  # pragma: no cover - guarded above for type narrowing
            raise RuntimeError("candidate validation proof is unavailable")
        return proof.candidate_fingerprint

    def to_dict(self) -> dict[str, Any]:
        """Return normalized content plus its deterministic fingerprint."""
        self._verify_validation_proof()
        return {
            **self._content_dict(),
            "candidate_fingerprint": self.candidate_fingerprint,
        }


def _expected_provenance(request: GenerationRequest) -> dict[str, str]:
    """Return immutable provider-echo constants from the exact request contract."""
    contract = request.contract
    rubric = _object(contract.get("rubric"), "$.generation_contract.rubric")
    blueprint = _object(
        contract.get("blueprint"),
        "$.generation_contract.blueprint",
    )
    return {
        "blueprint_id": request.blueprint.blueprint_id,
        "blueprint_handle": _string(
            blueprint.get("blueprint_handle"),
            "$.generation_contract.blueprint.blueprint_handle",
            maximum=_MAX_OPTION_ID_LENGTH,
        ),
        "blueprint_fingerprint": request.blueprint.blueprint_fingerprint,
        "rubric_id": request.blueprint.rubric_id,
        "rubric_version": request.blueprint.rubric_version,
        "rubric_fingerprint": _fingerprint(
            rubric.get("fingerprint"),
            "$.generation_contract.rubric.fingerprint",
        ),
    }


def _parse_provenance(
    decoded: dict[str, Any],
    request: GenerationRequest,
) -> dict[str, str]:
    """Validate every provider-echoed provenance field without disclosing mismatches."""
    expected = _expected_provenance(request)
    result: dict[str, str] = {}
    for field_name, expected_value in expected.items():
        path = f"$.{field_name}"
        if field_name.endswith("fingerprint"):
            actual = _fingerprint(decoded[field_name], path)
        elif field_name == "rubric_version":
            actual = _string(decoded[field_name], path, maximum=64)
        else:
            actual = _identifier(decoded[field_name], path)
        if actual != expected_value:
            raise _error(
                "provenance_mismatch",
                path,
                "value does not match the immutable generation contract",
            )
        result[field_name] = actual
    return result


def _parse_options(value: Any) -> tuple[GeneratedOption, ...]:
    """Parse bounded unique option objects."""
    raw = _array(value, "$.options", maximum=MAX_OPTIONS)
    options: list[GeneratedOption] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        path = f"$.options[{index}]"
        obj = _object(item, path)
        _exact_fields(obj, {"option_id", "text"}, path)
        option_id = _identifier(obj["option_id"], f"{path}.option_id")
        if option_id in seen:
            raise _error(
                "duplicate_option_id",
                "$.options",
                "option identifiers must be unique",
            )
        seen.add(option_id)
        options.append(
            GeneratedOption(
                option_id=option_id,
                text=_string(obj["text"], f"{path}.text"),
            )
        )
    return tuple(options)


def _parse_scoring_guide(
    value: Any,
    expected_scores: tuple[int, ...],
) -> tuple[ScoreGuideEntry, ...]:
    """Parse each scoring-guide entry exactly once in declared rubric order."""
    raw = _array(
        value,
        "$.scoring_guide",
        minimum=len(expected_scores),
        maximum=len(expected_scores),
    )
    entries: list[ScoreGuideEntry] = []
    seen: set[int] = set()
    for index, item in enumerate(raw):
        path = f"$.scoring_guide[{index}]"
        obj = _object(item, path)
        _exact_fields(obj, {"score", "evidence", "rationale"}, path)
        score = _integer(obj["score"], f"{path}.score")
        if score in seen:
            raise _error(
                "duplicate_score",
                "$.scoring_guide",
                "each rubric score must occur once",
            )
        seen.add(score)
        entries.append(
            ScoreGuideEntry(
                score=score,
                evidence=_string(obj["evidence"], f"{path}.evidence"),
                rationale=_string(obj["rationale"], f"{path}.rationale"),
            )
        )
    observed = tuple(entry.score for entry in entries)
    if set(observed) != set(expected_scores):
        raise _error(
            "score_coverage",
            "$.scoring_guide",
            "entries must cover every rubric score exactly once",
        )
    if observed != expected_scores:
        raise _error(
            "score_order",
            "$.scoring_guide",
            "entries must follow ascending rubric-score order",
        )
    return tuple(entries)


def _parse_rubric_alignment(
    value: Any,
    expected_scores: tuple[int, ...],
) -> tuple[RubricAlignmentEntry, ...]:
    """Parse each alignment entry exactly once in declared rubric order."""
    raw = _array(
        value,
        "$.rubric_alignment",
        minimum=len(expected_scores),
        maximum=len(expected_scores),
    )
    entries: list[RubricAlignmentEntry] = []
    seen: set[int] = set()
    for index, item in enumerate(raw):
        path = f"$.rubric_alignment[{index}]"
        obj = _object(item, path)
        _exact_fields(obj, {"score", "observable_indicators"}, path)
        score = _integer(obj["score"], f"{path}.score")
        if score in seen:
            raise _error(
                "duplicate_score",
                "$.rubric_alignment",
                "each rubric score must occur once",
            )
        seen.add(score)
        entries.append(
            RubricAlignmentEntry(
                score=score,
                observable_indicators=_string_array(
                    obj["observable_indicators"],
                    f"{path}.observable_indicators",
                    minimum=1,
                    unique=True,
                ),
            )
        )
    observed = tuple(entry.score for entry in entries)
    if set(observed) != set(expected_scores):
        raise _error(
            "score_coverage",
            "$.rubric_alignment",
            "entries must cover every rubric score exactly once",
        )
    if observed != expected_scores:
        raise _error(
            "score_order",
            "$.rubric_alignment",
            "entries must follow ascending rubric-score order",
        )
    return tuple(entries)


def _parse_attributions(
    value: Any,
    request: GenerationRequest,
) -> tuple[SourceAttribution, ...]:
    """Parse unique source attributions and verify every verbatim evidence span."""
    raw = _array(value, "$.source_attributions")
    mode = request.blueprint.evidence_mode
    if mode is EvidenceMode.CLOSED_BOOK:
        if raw:
            raise _error(
                "closed_book_attribution",
                "$.source_attributions",
                "closed-book candidates cannot contain source attributions",
            )
        return ()
    if not raw:
        raise _error(
            "source_attribution_required",
            "$.source_attributions",
            "source-backed candidates require at least one attribution",
        )

    sources = {source.source_id: source for source in request.sources}
    result: list[SourceAttribution] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw):
        path = f"$.source_attributions[{index}]"
        obj = _object(item, path)
        _exact_fields(obj, {"source_id", "evidence_span"}, path)
        source_id = _identifier(obj["source_id"], f"{path}.source_id")
        if source_id not in sources:
            raise _error(
                "unknown_source",
                f"{path}.source_id",
                "source identifier is not present in the request",
            )
        evidence_span = _string(obj["evidence_span"], f"{path}.evidence_span")
        pair = (source_id, evidence_span)
        if pair in seen:
            raise _error(
                "duplicate_attribution",
                "$.source_attributions",
                "source attribution pairs must be unique",
            )
        seen.add(pair)
        if evidence_span not in sources[source_id].content:
            raise _error(
                "evidence_span_not_found",
                f"{path}.evidence_span",
                "evidence span does not occur verbatim in the source",
            )
        result.append(SourceAttribution(source_id, evidence_span))

    distinct_sources = {entry.source_id for entry in result}
    if mode is EvidenceMode.SINGLE_SOURCE and len(distinct_sources) != 1:
        raise _error(
            "source_cardinality",
            "$.source_attributions",
            "single-source candidates must attribute exactly one source",
        )
    if mode is EvidenceMode.MULTI_SOURCE and len(distinct_sources) < 2:
        raise _error(
            "source_cardinality",
            "$.source_attributions",
            "multi-source candidates must attribute at least two distinct sources",
        )
    return tuple(result)


def _parse_answer_key(
    response_format: ResponseFormat,
    options: tuple[GeneratedOption, ...],
    value: Any,
    scoring_levels: tuple[int, ...],
) -> GeneratedAnswerKey:
    """Parse the typed bounded answer-key object declared by the contract."""
    answer = _object(value, "$.answer_key")
    option_ids = {option.option_id for option in options}

    if response_format is ResponseFormat.CONSTRUCTED_RESPONSE:
        if options:
            raise _error(
                "options_not_allowed",
                "$.options",
                "constructed responses cannot contain options",
            )
        _exact_fields(
            answer,
            {"reference_response", "accepted_variants", "rationale"},
            "$.answer_key",
        )
        return ConstructedAnswerKey(
            reference_response=_string(
                answer["reference_response"], "$.answer_key.reference_response"
            ),
            accepted_variants=_string_array(
                answer["accepted_variants"],
                "$.answer_key.accepted_variants",
                unique=True,
            ),
            rationale=_string(answer["rationale"], "$.answer_key.rationale"),
        )

    if response_format is ResponseFormat.SELECTED_RESPONSE:
        if len(options) < 2:
            raise _error(
                "option_count",
                "$.options",
                "selected responses require at least two options",
            )
        _exact_fields(answer, {"option_ids", "rationale"}, "$.answer_key")
        correct_ids = _string_array(
            answer["option_ids"],
            "$.answer_key.option_ids",
            minimum=1,
            maximum=MAX_OPTIONS,
            unique=True,
            identifier_values=True,
        )
        if not set(correct_ids).issubset(option_ids):
            raise _error(
                "invalid_answer_key",
                "$.answer_key.option_ids",
                "answer key may reference only supplied options",
            )
        return SelectedAnswerKey(
            option_ids=correct_ids,
            rationale=_string(answer["rationale"], "$.answer_key.rationale"),
        )

    if response_format is ResponseFormat.BINARY_JUDGMENT:
        if options:
            raise _error(
                "options_not_allowed",
                "$.options",
                "binary judgments cannot contain options",
            )
        _exact_fields(answer, {"value", "rationale"}, "$.answer_key")
        return BinaryAnswerKey(
            value=_boolean(answer["value"], "$.answer_key.value"),
            rationale=_string(answer["rationale"], "$.answer_key.rationale"),
        )

    if response_format is ResponseFormat.ORDINAL_RATING:
        if options:
            raise _error(
                "options_not_allowed",
                "$.options",
                "ordinal ratings cannot contain options",
            )
        _exact_fields(answer, {"score", "rationale"}, "$.answer_key")
        score = _integer(answer["score"], "$.answer_key.score")
        if score not in scoring_levels:
            raise _error(
                "invalid_answer_key",
                "$.answer_key.score",
                "answer key score must be an allowed rubric score",
            )
        return OrdinalAnswerKey(
            score=score,
            rationale=_string(answer["rationale"], "$.answer_key.rationale"),
        )

    if len(options) != 2:
        raise _error(
            "option_count",
            "$.options",
            "pairwise comparison requires exactly two options",
        )
    _exact_fields(
        answer,
        {"outcome", "preferred_option_id", "rationale"},
        "$.answer_key",
    )
    outcome = _string(answer["outcome"], "$.answer_key.outcome", maximum=32)
    if outcome not in {"left_option", "right_option", "tie"}:
        raise _error(
            "invalid_answer_key",
            "$.answer_key.outcome",
            "outcome must be left_option, right_option, or tie",
        )
    raw_preferred = answer["preferred_option_id"]
    preferred = (
        None
        if raw_preferred is None
        else _identifier(raw_preferred, "$.answer_key.preferred_option_id")
    )
    if outcome == "tie":
        if preferred is not None:
            raise _error(
                "invalid_answer_key",
                "$.answer_key.preferred_option_id",
                "tie outcome requires a null preferred option",
            )
    else:
        expected = (
            options[0].option_id if outcome == "left_option" else options[1].option_id
        )
        if preferred != expected:
            raise _error(
                "invalid_answer_key",
                "$.answer_key.preferred_option_id",
                "preferred option must match the ordered pairwise outcome",
            )
        if preferred not in option_ids:
            raise _error(
                "invalid_answer_key",
                "$.answer_key.preferred_option_id",
                "preferred option must identify one supplied option",
            )
    return PairwiseAnswerKey(
        outcome=outcome,
        preferred_option_id=preferred,
        rationale=_string(answer["rationale"], "$.answer_key.rationale"),
    )


def parse_generated_item_candidate(
    raw_json: str,
    request: GenerationRequest,
) -> GeneratedItemCandidate:
    """Parse untrusted provider JSON into one strict source-grounded candidate."""
    if not isinstance(request, GenerationRequest):
        raise TypeError("request must be a GenerationRequest")
    if type(raw_json) is not str:
        raise TypeError("raw_json must be a string")
    if len(raw_json) > MAX_RAW_RESPONSE_CHARACTERS:
        raise _error(
            "raw_json_too_large",
            "$",
            "provider output exceeds the allowed size",
        )
    _validate_raw_json_depth(raw_json)
    try:
        decoded = json.loads(
            raw_json,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except _DuplicateJsonKey:
        raise _error(
            "duplicate_json_key",
            "$",
            "JSON object member names must be unique",
        ) from None
    except _NonFiniteJsonNumber:
        raise _error(
            "nonfinite_json_number",
            "$",
            "JSON numbers must be finite",
        ) from None
    except (json.JSONDecodeError, RecursionError):
        raise _error("invalid_json", "$", "provider output is not valid JSON") from None

    _validate_json_depth(decoded)
    if not isinstance(decoded, dict):
        raise _error("top_level_type", "$", "top-level JSON value must be an object")
    _exact_fields(decoded, _REQUIRED_FIELDS, "$")
    provenance = _parse_provenance(decoded, request)

    item_id = _identifier(decoded["item_id"], "$.item_id")
    stem = _string(decoded["stem"], "$.stem")
    stimulus = _string_array(decoded["stimulus"], "$.stimulus")
    raw_format = decoded["response_format"]
    if not isinstance(raw_format, str):
        raise _error(
            "invalid_type",
            "$.response_format",
            "response format must be a string",
        )
    try:
        response_format = ResponseFormat(raw_format)
    except ValueError:
        raise _error(
            "invalid_response_format",
            "$.response_format",
            "response format is not supported",
        ) from None
    if response_format is not request.blueprint.response_format:
        raise _error(
            "response_format_mismatch",
            "$.response_format",
            "response format does not match the generation contract",
        )

    options = _parse_options(decoded["options"])
    scoring_guide = _parse_scoring_guide(
        decoded["scoring_guide"],
        request.blueprint.scoring_levels,
    )
    rubric_alignment = _parse_rubric_alignment(
        decoded["rubric_alignment"],
        request.blueprint.scoring_levels,
    )
    source_attributions = _parse_attributions(
        decoded["source_attributions"],
        request,
    )
    answer_key = _parse_answer_key(
        response_format,
        options,
        decoded["answer_key"],
        request.blueprint.scoring_levels,
    )
    safety_notes = _string_array(
        decoded["safety_notes"],
        "$.safety_notes",
        unique=True,
    )

    return GeneratedItemCandidate._from_validated(
        request_id=request.request_id,
        request_fingerprint=request.request_fingerprint,
        contract_id=request.contract_id,
        contract_handle=request.contract_handle,
        contract_fingerprint=request.contract_fingerprint,
        blueprint_id=provenance["blueprint_id"],
        blueprint_handle=provenance["blueprint_handle"],
        blueprint_fingerprint=provenance["blueprint_fingerprint"],
        rubric_id=provenance["rubric_id"],
        rubric_version=provenance["rubric_version"],
        rubric_fingerprint=provenance["rubric_fingerprint"],
        item_id=item_id,
        stem=stem,
        stimulus=stimulus,
        response_format=response_format,
        options=options,
        answer_key=answer_key,
        scoring_guide=scoring_guide,
        rubric_alignment=rubric_alignment,
        source_attributions=source_attributions,
        safety_notes=safety_notes,
        schema_version=SCHEMA_VERSION,
    )
