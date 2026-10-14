"""Provider-neutral contracts for preregistered external-validity evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re

SCHEMA_VERSION = "1.0"
MAX_TEXT_LENGTH = 4_096
MAX_COLLECTION_VALUES = 64

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EvidenceClass(str, Enum):
    """Distinct validity-evidence regimes that must not be collapsed."""

    TECHNICAL = "technical"
    CONSTRUCT = "construct"
    TRANSPORTABILITY = "transportability"
    FAIRNESS = "fairness"
    DECISION_UTILITY = "decision_utility"


class EvidenceStatus(str, Enum):
    """Explicit execution/verdict states for one validation evidence artifact."""

    PASSED = "passed"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"
    NOT_EXECUTED = "not_executed"
    NOT_APPLICABLE = "not_applicable"


def _text(value: object, name: str, *, maximum: int = MAX_TEXT_LENGTH) -> str:
    """Normalize bounded exact built-in text without caller callback dispatch."""
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    return normalized


def _identifier(value: object, name: str) -> str:
    """Normalize one lower-snake-case public identifier."""
    normalized = _text(value, name, maximum=128)
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must use two-or-more-token lower snake_case")
    return normalized


def _fingerprint(value: object, name: str) -> str:
    """Validate a lowercase SHA-256 hexadecimal fingerprint."""
    normalized = _text(value, name, maximum=64)
    if _FINGERPRINT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return normalized


def _optional_fingerprint(value: object | None, name: str) -> str | None:
    """Validate an optional lowercase SHA-256 fingerprint."""
    return None if value is None else _fingerprint(value, name)


def _enum_value(value: object, enum_type: type[Enum], name: str) -> Enum:
    """Admit an exact enum member or exact built-in serialized value."""
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise ValueError(f"{name} must be a supported {enum_type.__name__} value")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = [member.value for member in enum_type]
        raise ValueError(f"{name} must be one of {choices}") from exc


def _text_tuple(
    values: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_COLLECTION_VALUES,
) -> tuple[str, ...]:
    """Normalize a bounded list/tuple of unique provider-neutral text values."""
    if type(values) not in {tuple, list}:
        raise ValueError(f"{name} must be a list or tuple")
    if not minimum <= len(values) <= maximum:
        raise ValueError(
            f"{name} must contain between {minimum} and {maximum} values"
        )
    normalized = tuple(
        _text(value, f"{name}[{index}]", maximum=512)
        for index, value in enumerate(values)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _timestamp(value: object, name: str) -> tuple[str, datetime]:
    """Normalize one timezone-aware ISO-8601 timestamp to UTC ``Z`` form."""
    text = _text(value, name, maximum=64)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        raise ValueError(
            f"{name} must be an ISO-8601 timestamp with timezone"
        ) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    utc = parsed.astimezone(timezone.utc)
    normalized = utc.isoformat(timespec="seconds").replace("+00:00", "Z")
    return normalized, utc


def _evidence_tuple(values: object) -> tuple[ValidationEvidence, ...]:
    """Normalize a bounded exact list/tuple of package-owned evidence records."""
    if type(values) not in {tuple, list}:
        raise ValueError("evidence must be a list or tuple")
    if len(values) > MAX_COLLECTION_VALUES:
        raise ValueError(
            f"evidence must contain at most {MAX_COLLECTION_VALUES} values"
        )
    normalized: list[ValidationEvidence] = []
    for index, value in enumerate(values):
        if type(value) is not ValidationEvidence:
            raise ValueError(f"evidence[{index}] must be a ValidationEvidence")
        if type(value.evidence_id) is not str:
            raise ValueError(f"evidence[{index}].evidence_id must be a string")
        if type(value.evidence_class) is not EvidenceClass:
            raise ValueError(
                f"evidence[{index}].evidence_class must be an EvidenceClass"
            )
        if type(value.status) is not EvidenceStatus:
            raise ValueError(f"evidence[{index}].status must be an EvidenceStatus")
        if type(value.available_time) is not str:
            raise ValueError(f"evidence[{index}].available_time must be a string")
        if value.artifact_sha256 is not None and type(value.artifact_sha256) is not str:
            raise ValueError(f"evidence[{index}].artifact_sha256 must be a string")
        if value.limitation is not None and type(value.limitation) is not str:
            raise ValueError(f"evidence[{index}].limitation must be a string")
        normalized.append(
            ValidationEvidence(
                evidence_id=value.evidence_id,
                evidence_class=value.evidence_class,
                status=value.status,
                available_time=value.available_time,
                artifact_sha256=value.artifact_sha256,
                limitation=value.limitation,
            )
        )
    if len({item.evidence_id for item in normalized}) != len(normalized):
        raise ValueError("evidence_id values must be unique")
    return tuple(sorted(normalized, key=lambda item: item.evidence_id))


def _canonical_json(payload: object) -> str:
    """Serialize source-free manifest content deterministically."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(payload: object) -> str:
    """Return SHA-256 over canonical UTF-8 JSON."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ValidationEvidence:
    """Source-free reference to one preregistered validation evidence artifact."""

    evidence_id: str
    evidence_class: EvidenceClass
    status: EvidenceStatus
    available_time: str
    artifact_sha256: str | None = None
    limitation: str | None = None

    def __post_init__(self) -> None:
        """Normalize and validate one immutable evidence record."""
        if type(self) is not ValidationEvidence:
            raise ValueError("ValidationEvidence must be an exact package record")
        object.__setattr__(
            self,
            "evidence_id",
            _identifier(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(
            self,
            "evidence_class",
            _enum_value(self.evidence_class, EvidenceClass, "evidence_class"),
        )
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, EvidenceStatus, "status"),
        )
        available_time, _ = _timestamp(self.available_time, "available_time")
        object.__setattr__(self, "available_time", available_time)
        object.__setattr__(
            self,
            "artifact_sha256",
            _optional_fingerprint(self.artifact_sha256, "artifact_sha256"),
        )
        if self.limitation is not None:
            object.__setattr__(
                self,
                "limitation",
                _text(self.limitation, "limitation"),
            )

    def to_manifest(self) -> dict[str, object]:
        """Return the JSON-compatible evidence projection after replay validation."""
        validated = ValidationEvidence(
            evidence_id=self.evidence_id,
            evidence_class=self.evidence_class,
            status=self.status,
            available_time=self.available_time,
            artifact_sha256=self.artifact_sha256,
            limitation=self.limitation,
        )
        return {
            "artifact_sha256": validated.artifact_sha256,
            "available_time": validated.available_time,
            "evidence_class": validated.evidence_class.value,
            "evidence_id": validated.evidence_id,
            "limitation": validated.limitation,
            "status": validated.status.value,
        }


@dataclass(frozen=True, slots=True)
class ExternalValidationProfile:
    """Immutable preregistered profile separating validity evidence regimes."""

    validation_profile_id: str
    construct: str
    score_interpretation: str
    population: str
    setting: str
    decision_use: str
    assessment_fingerprint: str
    rubric_fingerprint: str
    item_bank_fingerprint: str
    model_fingerprint: str
    development_dataset_ids: tuple[str, ...]
    internal_validation_dataset_ids: tuple[str, ...]
    external_validation_dataset_ids: tuple[str, ...]
    sites: tuple[str, ...]
    languages: tuple[str, ...]
    preregistration_reference: str
    preregistered_at: str
    analysis_cutoff: str
    data_license: str
    purpose_classification: str
    evidence: tuple[ValidationEvidence, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize profile fields and reject temporal or cohort leakage."""
        if type(self) is not ExternalValidationProfile:
            raise ValueError(
                "ExternalValidationProfile must be an exact package record"
            )
        object.__setattr__(
            self,
            "validation_profile_id",
            _identifier(self.validation_profile_id, "validation_profile_id"),
        )
        for field_name in (
            "construct",
            "score_interpretation",
            "population",
            "setting",
            "decision_use",
            "preregistration_reference",
            "data_license",
            "purpose_classification",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name),
            )
        for field_name in (
            "assessment_fingerprint",
            "rubric_fingerprint",
            "item_bank_fingerprint",
            "model_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _fingerprint(getattr(self, field_name), field_name),
            )
        for field_name in (
            "development_dataset_ids",
            "internal_validation_dataset_ids",
            "external_validation_dataset_ids",
            "sites",
        ):
            object.__setattr__(
                self,
                field_name,
                _text_tuple(getattr(self, field_name), field_name),
            )

        dataset_groups = (
            ("development_dataset_ids", self.development_dataset_ids),
            (
                "internal_validation_dataset_ids",
                self.internal_validation_dataset_ids,
            ),
            (
                "external_validation_dataset_ids",
                self.external_validation_dataset_ids,
            ),
        )
        seen_dataset_ids: dict[str, str] = {}
        for group_name, dataset_ids in dataset_groups:
            for dataset_id in dataset_ids:
                previous_group = seen_dataset_ids.get(dataset_id)
                if previous_group is not None:
                    raise ValueError(
                        f"dataset id must not occur in both {previous_group} and "
                        f"{group_name}"
                    )
                seen_dataset_ids[dataset_id] = group_name

        languages = self.languages
        if type(languages) not in {tuple, list}:
            raise ValueError("languages must be a list or tuple")
        if len(languages) > MAX_COLLECTION_VALUES:
            raise ValueError(
                f"languages must contain at most {MAX_COLLECTION_VALUES} values"
            )
        normalized_languages = tuple(
            _text(value, f"languages[{index}]", maximum=32)
            for index, value in enumerate(languages)
        )
        if len(set(normalized_languages)) != len(normalized_languages):
            raise ValueError("languages must not contain duplicates")
        object.__setattr__(self, "languages", normalized_languages)

        schema_version = _text(self.schema_version, "schema_version", maximum=16)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be '{SCHEMA_VERSION}'")
        object.__setattr__(self, "schema_version", schema_version)

        preregistered_at, preregistered_dt = _timestamp(
            self.preregistered_at,
            "preregistered_at",
        )
        analysis_cutoff, cutoff_dt = _timestamp(
            self.analysis_cutoff,
            "analysis_cutoff",
        )
        if preregistered_dt > cutoff_dt:
            raise ValueError("preregistered_at must not exceed analysis_cutoff")
        object.__setattr__(self, "preregistered_at", preregistered_at)
        object.__setattr__(self, "analysis_cutoff", analysis_cutoff)

        evidence = _evidence_tuple(self.evidence)
        for row in evidence:
            _, available_dt = _timestamp(row.available_time, "available_time")
            if available_dt > cutoff_dt:
                raise ValueError("available_time must not exceed analysis_cutoff")
        object.__setattr__(self, "evidence", evidence)

    def _manifest_without_fingerprint(self) -> dict[str, object]:
        """Return canonical profile content after replaying all admission checks."""
        validated = ExternalValidationProfile(
            validation_profile_id=self.validation_profile_id,
            construct=self.construct,
            score_interpretation=self.score_interpretation,
            population=self.population,
            setting=self.setting,
            decision_use=self.decision_use,
            assessment_fingerprint=self.assessment_fingerprint,
            rubric_fingerprint=self.rubric_fingerprint,
            item_bank_fingerprint=self.item_bank_fingerprint,
            model_fingerprint=self.model_fingerprint,
            development_dataset_ids=self.development_dataset_ids,
            internal_validation_dataset_ids=self.internal_validation_dataset_ids,
            external_validation_dataset_ids=self.external_validation_dataset_ids,
            sites=self.sites,
            languages=self.languages,
            preregistration_reference=self.preregistration_reference,
            preregistered_at=self.preregistered_at,
            analysis_cutoff=self.analysis_cutoff,
            data_license=self.data_license,
            purpose_classification=self.purpose_classification,
            evidence=self.evidence,
            schema_version=self.schema_version,
        )
        return {
            "analysis_cutoff": validated.analysis_cutoff,
            "assessment_fingerprint": validated.assessment_fingerprint,
            "construct": validated.construct,
            "data_license": validated.data_license,
            "decision_use": validated.decision_use,
            "development_dataset_ids": list(validated.development_dataset_ids),
            "evidence": [row.to_manifest() for row in validated.evidence],
            "external_validation_dataset_ids": list(
                validated.external_validation_dataset_ids
            ),
            "internal_validation_dataset_ids": list(
                validated.internal_validation_dataset_ids
            ),
            "item_bank_fingerprint": validated.item_bank_fingerprint,
            "languages": list(validated.languages),
            "model_fingerprint": validated.model_fingerprint,
            "population": validated.population,
            "preregistered_at": validated.preregistered_at,
            "preregistration_reference": validated.preregistration_reference,
            "purpose_classification": validated.purpose_classification,
            "rubric_fingerprint": validated.rubric_fingerprint,
            "schema_version": validated.schema_version,
            "score_interpretation": validated.score_interpretation,
            "setting": validated.setting,
            "sites": list(validated.sites),
            "validation_profile_id": validated.validation_profile_id,
        }

    @property
    def profile_fingerprint(self) -> str:
        """Return the immutable SHA-256 identity of the normalized profile."""
        return _sha256(self._manifest_without_fingerprint())

    def to_manifest(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible source-free profile."""
        payload = self._manifest_without_fingerprint()
        payload["profile_fingerprint"] = _sha256(payload)
        return payload


__all__ = [
    "EvidenceClass",
    "EvidenceStatus",
    "ExternalValidationProfile",
    "ValidationEvidence",
]
