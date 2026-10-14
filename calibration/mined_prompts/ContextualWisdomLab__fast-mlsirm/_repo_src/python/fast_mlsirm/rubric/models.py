"""Versioned immutable schemas for rubric-centered item authoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import operator
import re
from typing import Any, Iterable, TypeVar

import numpy as np

SCHEMA_VERSION = "1.0"
MAX_TEXT_LENGTH = 8_192
MAX_COLLECTION_VALUES = 32
MAX_LEVELS = 16
MAX_SCORE = 31
MAX_ITEMS_PER_CELL = 100
MAX_REPLICATE_INDEX = 99
MAX_U64 = (1 << 64) - 1

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_LOCALE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)

EnumValue = TypeVar("EnumValue", bound=Enum)
_TRUSTED_NUMPY_INTEGER_TYPES = tuple(
    np.dtype(code).type for code in np.typecodes["AllInteger"]
)


class ResponseFormat(str, Enum):
    """Supported response representations for generated assessment items."""

    CONSTRUCTED_RESPONSE = "constructed_response"
    SELECTED_RESPONSE = "selected_response"
    BINARY_JUDGMENT = "binary_judgment"
    ORDINAL_RATING = "ordinal_rating"
    PAIRWISE_COMPARISON = "pairwise_comparison"


class DifficultyBand(str, Enum):
    """Coarse item-difficulty targets used by a blueprint plan."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvidenceMode(str, Enum):
    """Evidence conditions under which an item should elicit performance."""

    CLOSED_BOOK = "closed_book"
    SINGLE_SOURCE = "single_source"
    MULTI_SOURCE = "multi_source"
    ADVERSARIAL_CONTEXT = "adversarial_context"
    UNANSWERABLE = "unanswerable"


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_LENGTH) -> str:
    """Normalize bounded non-empty text or raise a field-specific error."""
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    return normalized


def _identifier(value: Any, name: str) -> str:
    """Normalize a two-or-more-token lower snake-case identifier."""
    normalized = _text(value, name, maximum=128)
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must use two-or-more-token lower snake_case")
    return normalized


def _bounded_values(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> tuple[Any, ...]:
    """Materialize at most ``maximum`` caller values without unbounded copying."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a collection")
    try:
        iterator = iter(values)
    except MemoryError:
        raise
    except Exception:
        raise ValueError(f"{name} must be a collection") from None
    materialized: list[Any] = []
    for index in range(maximum + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        except MemoryError:
            raise
        except Exception:
            raise ValueError(f"{name} iteration failed") from None
        if index >= maximum:
            raise ValueError(f"{name} must contain at most {maximum} values")
        materialized.append(value)
    if len(materialized) < minimum:
        plural = "s" if minimum != 1 else ""
        raise ValueError(f"{name} must contain at least {minimum} value{plural}")
    return tuple(materialized)


def _text_tuple(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
    maximum: int = MAX_COLLECTION_VALUES,
) -> tuple[str, ...]:
    """Normalize a bounded collection of unique text values."""
    raw = _bounded_values(values, name, minimum=minimum, maximum=maximum)
    normalized = tuple(
        _text(value, f"{name}[{index}]") for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _identifier_tuple(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
    maximum: int = MAX_COLLECTION_VALUES,
) -> tuple[str, ...]:
    """Normalize a bounded collection of unique public identifiers."""
    raw = _bounded_values(values, name, minimum=minimum, maximum=maximum)
    normalized = tuple(
        _identifier(value, f"{name}[{index}]") for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _enum_value(value: Any, enum_type: type[EnumValue], name: str) -> EnumValue:
    """Normalize an enum instance or its exact built-in string value."""
    if isinstance(value, enum_type):
        return value
    choices = [member.value for member in enum_type]
    if type(value) is not str:
        raise ValueError(f"{name} must be one of {choices}")
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be one of {choices}") from exc


def _enum_tuple(
    values: Iterable[Any],
    enum_type: type[EnumValue],
    name: str,
) -> tuple[EnumValue, ...]:
    """Normalize a non-empty unique collection of enum values."""
    raw = _bounded_values(values, name, minimum=1, maximum=len(enum_type))
    try:
        normalized = tuple(_enum_value(value, enum_type, name) for value in raw)
    except ValueError as exc:
        choices = [member.value for member in enum_type]
        raise ValueError(f"{name} must contain only {choices}") from exc
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _integer(value: Any, name: str) -> int:
    """Normalize only exact package-trusted integer scalar identities."""
    value_type = type(value)
    if value_type is bool:
        raise ValueError(f"{name} must be an integer")
    if value_type is int:
        return value
    if not any(value_type is trusted_type for trusted_type in _TRUSTED_NUMPY_INTEGER_TYPES):
        raise ValueError(f"{name} must be an integer")
    return operator.index(value)


def _unsigned_integer(value: Any, name: str) -> int:
    """Normalize an unsigned 64-bit integer."""
    normalized = _integer(value, name)
    if not 0 <= normalized <= MAX_U64:
        raise ValueError(f"{name} must fit an unsigned 64-bit integer")
    return normalized


def _schema_version(value: Any) -> str:
    """Accept only the schema version implemented by this package slice."""
    normalized = _text(value, "schema_version", maximum=16)
    if normalized != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be '{SCHEMA_VERSION}'")
    return normalized


def _semantic_version(value: Any, name: str = "rubric_version") -> str:
    """Normalize a canonical numeric semantic version without ambiguous zeros."""
    normalized = _text(value, name, maximum=64)
    if _SEMANTIC_VERSION_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a canonical semantic version (major.minor.patch)")
    return normalized


def _canonical_json(payload: Any) -> str:
    """Serialize JSON-compatible content with a stable UTF-8 representation."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_hex(payload: Any) -> str:
    """Return a SHA-256 hex digest of canonical JSON content."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RubricLevel:
    """One ordered score level and its directly observable evidence."""

    score: int
    label: str
    descriptor: str
    observable_indicators: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize and validate the immutable score-level fields."""
        score = _integer(self.score, "score")
        if not 0 <= score <= MAX_SCORE:
            raise ValueError(f"score must be between 0 and {MAX_SCORE}")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "label", _text(self.label, "label", maximum=256))
        object.__setattr__(self, "descriptor", _text(self.descriptor, "descriptor"))
        object.__setattr__(
            self,
            "observable_indicators",
            _text_tuple(
                self.observable_indicators,
                "observable_indicators",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible score-level representation."""
        return {
            "score": self.score,
            "label": self.label,
            "descriptor": self.descriptor,
            "observable_indicators": list(self.observable_indicators),
        }


@dataclass(frozen=True)
class RubricSpecification:
    """Versioned construct, evidence, task-family, and score-model specification."""

    rubric_id: str
    construct_id: str
    construct_definition: str
    response_format: ResponseFormat
    levels: tuple[RubricLevel, ...]
    task_families: tuple[str, ...]
    evidence_requirements: tuple[str, ...]
    prohibited_patterns: tuple[str, ...] = ()
    locale: str = "en"
    rubric_version: str = "1.0.0"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize the rubric and enforce its measurement-schema invariants."""
        object.__setattr__(self, "rubric_id", _identifier(self.rubric_id, "rubric_id"))
        object.__setattr__(
            self,
            "construct_id",
            _identifier(self.construct_id, "construct_id"),
        )
        object.__setattr__(
            self,
            "construct_definition",
            _text(self.construct_definition, "construct_definition"),
        )
        object.__setattr__(
            self,
            "response_format",
            _enum_value(self.response_format, ResponseFormat, "response_format"),
        )
        raw_levels = _bounded_values(
            self.levels,
            "levels",
            minimum=2,
            maximum=MAX_LEVELS,
        )
        for index, level in enumerate(raw_levels):
            if not isinstance(level, RubricLevel):
                raise ValueError(f"levels[{index}] must be a RubricLevel")
        levels = tuple(raw_levels)
        if tuple(level.score for level in levels) != tuple(range(len(levels))):
            raise ValueError(
                "level scores must be contiguous integers beginning at zero"
            )
        labels = tuple(level.label for level in levels)
        if len(set(labels)) != len(labels):
            raise ValueError("level labels must be unique")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(
            self,
            "task_families",
            _identifier_tuple(self.task_families, "task_families", minimum=1),
        )
        object.__setattr__(
            self,
            "evidence_requirements",
            _text_tuple(
                self.evidence_requirements,
                "evidence_requirements",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "prohibited_patterns",
            _text_tuple(
                self.prohibited_patterns,
                "prohibited_patterns",
                minimum=0,
            ),
        )
        locale = _text(self.locale, "locale", maximum=64)
        if _LOCALE_PATTERN.fullmatch(locale) is None:
            raise ValueError("locale must be a BCP 47-style tag")
        object.__setattr__(self, "locale", locale)
        object.__setattr__(
            self,
            "rubric_version",
            _semantic_version(self.rubric_version),
        )
        object.__setattr__(
            self,
            "schema_version",
            _schema_version(self.schema_version),
        )

    @property
    def fingerprint(self) -> str:
        """Return the canonical SHA-256 identity of this rubric specification."""
        return _sha256_hex(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible rubric representation without derived ids."""
        return {
            "schema_version": self.schema_version,
            "rubric_version": self.rubric_version,
            "rubric_id": self.rubric_id,
            "construct_id": self.construct_id,
            "construct_definition": self.construct_definition,
            "response_format": self.response_format.value,
            "levels": [level.to_dict() for level in self.levels],
            "task_families": list(self.task_families),
            "evidence_requirements": list(self.evidence_requirements),
            "prohibited_patterns": list(self.prohibited_patterns),
            "locale": self.locale,
        }


@dataclass(frozen=True)
class BlueprintPlan:
    """Bounded task/evidence matrix used to compile deterministic blueprints."""

    difficulty_bands: tuple[DifficultyBand, ...] = tuple(DifficultyBand)
    evidence_modes: tuple[EvidenceMode, ...] = (EvidenceMode.SINGLE_SOURCE,)
    items_per_cell: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        """Normalize matrix dimensions and enforce caller-controlled work bounds."""
        object.__setattr__(
            self,
            "difficulty_bands",
            _enum_tuple(
                self.difficulty_bands,
                DifficultyBand,
                "difficulty_bands",
            ),
        )
        object.__setattr__(
            self,
            "evidence_modes",
            _enum_tuple(self.evidence_modes, EvidenceMode, "evidence_modes"),
        )
        items_per_cell = _integer(self.items_per_cell, "items_per_cell")
        if not 1 <= items_per_cell <= MAX_ITEMS_PER_CELL:
            raise ValueError(
                f"items_per_cell must be between 1 and {MAX_ITEMS_PER_CELL}"
            )
        object.__setattr__(self, "items_per_cell", items_per_cell)
        object.__setattr__(self, "seed", _unsigned_integer(self.seed, "seed"))


@dataclass(frozen=True)
class ItemBlueprint:
    """One immutable task/evidence design cell compiled from an exact rubric."""

    blueprint_id: str
    rubric_id: str
    rubric_fingerprint: str
    task_family: str
    difficulty_band: DifficultyBand
    evidence_mode: EvidenceMode
    replicate_index: int
    generation_seed: int
    response_format: ResponseFormat
    scoring_levels: tuple[int, ...]
    evidence_requirements: tuple[str, ...]
    prohibited_patterns: tuple[str, ...] = ()
    rubric_version: str = "1.0.0"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize the compiled blueprint and prevent direct invariant bypass."""
        object.__setattr__(
            self,
            "blueprint_id",
            _identifier(self.blueprint_id, "blueprint_id"),
        )
        object.__setattr__(
            self,
            "rubric_id",
            _identifier(self.rubric_id, "rubric_id"),
        )
        fingerprint = _text(
            self.rubric_fingerprint,
            "rubric_fingerprint",
            maximum=64,
        )
        if _FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
            raise ValueError(
                "rubric_fingerprint must be a 64-character lower hexadecimal digest"
            )
        object.__setattr__(self, "rubric_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "task_family",
            _identifier(self.task_family, "task_family"),
        )
        object.__setattr__(
            self,
            "difficulty_band",
            _enum_value(self.difficulty_band, DifficultyBand, "difficulty_band"),
        )
        object.__setattr__(
            self,
            "evidence_mode",
            _enum_value(self.evidence_mode, EvidenceMode, "evidence_mode"),
        )
        replicate_index = _integer(self.replicate_index, "replicate_index")
        if not 0 <= replicate_index <= MAX_REPLICATE_INDEX:
            raise ValueError(
                f"replicate_index must be between 0 and {MAX_REPLICATE_INDEX}"
            )
        object.__setattr__(self, "replicate_index", replicate_index)
        object.__setattr__(
            self,
            "generation_seed",
            _unsigned_integer(self.generation_seed, "generation_seed"),
        )
        object.__setattr__(
            self,
            "response_format",
            _enum_value(self.response_format, ResponseFormat, "response_format"),
        )
        raw_levels = _bounded_values(
            self.scoring_levels,
            "scoring_levels",
            minimum=2,
            maximum=MAX_LEVELS,
        )
        scoring_levels = tuple(
            _integer(value, f"scoring_levels[{index}]")
            for index, value in enumerate(raw_levels)
        )
        if scoring_levels != tuple(range(len(scoring_levels))):
            raise ValueError(
                "scoring_levels must be contiguous integers beginning at zero"
            )
        object.__setattr__(self, "scoring_levels", scoring_levels)
        object.__setattr__(
            self,
            "evidence_requirements",
            _text_tuple(
                self.evidence_requirements,
                "evidence_requirements",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "prohibited_patterns",
            _text_tuple(
                self.prohibited_patterns,
                "prohibited_patterns",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "rubric_version",
            _semantic_version(self.rubric_version),
        )
        object.__setattr__(
            self,
            "schema_version",
            _schema_version(self.schema_version),
        )

    def _fingerprint_payload(self) -> dict[str, Any]:
        """Return stable blueprint content excluding display and derived ids."""
        return {
            "schema_version": self.schema_version,
            "rubric_version": self.rubric_version,
            "rubric_id": self.rubric_id,
            "rubric_fingerprint": self.rubric_fingerprint,
            "task_family": self.task_family,
            "difficulty_band": self.difficulty_band.value,
            "evidence_mode": self.evidence_mode.value,
            "replicate_index": self.replicate_index,
            "generation_seed": self.generation_seed,
            "response_format": self.response_format.value,
            "scoring_levels": list(self.scoring_levels),
            "evidence_requirements": list(self.evidence_requirements),
            "prohibited_patterns": list(self.prohibited_patterns),
        }

    @property
    def blueprint_fingerprint(self) -> str:
        """Return the full SHA-256 identity of the normalized blueprint content."""
        return _sha256_hex(self._fingerprint_payload())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible compiled blueprint representation."""
        return {
            **self._fingerprint_payload(),
            "blueprint_id": self.blueprint_id,
            "blueprint_fingerprint": self.blueprint_fingerprint,
        }