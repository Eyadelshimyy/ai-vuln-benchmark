"""Source-free preregistered external-validation evidence contracts.

The module records provenance, chronology, explicit evidence classes, and
fail-closed evidence states. It performs no validity, transportability,
fairness, calibration, utility, or other statistical arithmetic; future
production numerical analyses remain Rust-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
from typing import Any

MAX_VALIDATION_EVIDENCE = 256
MAX_VALIDATION_LIMITATIONS = 64
_MAX_IDENTIFIER_LENGTH = 128
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class ValidationEvidenceClass(str, Enum):
    """Distinct evidence families that support a declared score interpretation."""

    TECHNICAL = "technical"
    CONSTRUCT = "construct"
    TRANSPORTABILITY = "transportability"
    FAIRNESS = "fairness"
    DECISION_UTILITY = "decision_utility"


class ValidationEvidenceStatus(str, Enum):
    """Explicit non-aggregated state of one validation evidence artifact."""

    PASSED = "passed"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"
    NOT_EXECUTED = "not_executed"
    NOT_APPLICABLE = "not_applicable"


def _identifier(value: Any, name: str) -> str:
    """Return one exact bounded opaque identifier without caller coercion."""
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    if not value or len(value) > _MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"{name} must contain 1..{_MAX_IDENTIFIER_LENGTH} characters")
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be an opaque identifier")
    return value


def _fingerprint(value: Any, name: str) -> str:
    """Return one exact lower-hexadecimal SHA-256 artifact fingerprint."""
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be 64 lower hexadecimal characters")
    return value


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> Enum:
    """Normalize an exact enum member or exact serialized enum value."""
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact supported value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a supported value") from exc


def _utc_datetime(value: Any, name: str) -> datetime:
    """Normalize one exact callback-free fixed-offset timestamp to UTC."""
    if type(value) is not datetime:
        raise ValueError(f"{name} must be an exact datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be offset-aware")
    if type(value.tzinfo) is not timezone:
        raise ValueError(f"{name} must use datetime.timezone")
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    """Serialize one validated UTC timestamp with an explicit Z suffix."""
    return value.isoformat().replace("+00:00", "Z")


def _bounded_identifiers(value: Any, name: str) -> tuple[str, ...]:
    """Normalize one bounded exact list/tuple of unique identifiers."""
    if type(value) not in {list, tuple}:
        raise ValueError(f"{name} must be an exact list or tuple")
    if len(value) > MAX_VALIDATION_LIMITATIONS:
        raise ValueError(f"{name} exceeds maximum {MAX_VALIDATION_LIMITATIONS}")
    normalized = tuple(_identifier(item, name) for item in value)
    return tuple(sorted(set(normalized)))


def _canonical_sha256(payload: dict[str, Any]) -> str:
    """Hash canonical source-free JSON content with SHA-256."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ValidationEvidenceReference:
    """Bounded source-free identity and state of one validation evidence artifact."""

    evidence_id: str
    artifact_fingerprint: str
    evidence_class: ValidationEvidenceClass
    status: ValidationEvidenceStatus
    available_time: datetime
    limitation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Seal evidence identity, chronology, class, status, and limitations."""
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        object.__setattr__(
            self,
            "artifact_fingerprint",
            _fingerprint(self.artifact_fingerprint, "artifact_fingerprint"),
        )
        object.__setattr__(
            self,
            "evidence_class",
            _enum_value(self.evidence_class, ValidationEvidenceClass, "evidence_class"),
        )
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, ValidationEvidenceStatus, "status"),
        )
        object.__setattr__(
            self,
            "available_time",
            _utc_datetime(self.available_time, "available_time"),
        )
        object.__setattr__(
            self,
            "limitation_ids",
            _bounded_identifiers(self.limitation_ids, "limitation_ids"),
        )

    def _validated_content(self) -> dict[str, Any]:
        """Replay evidence invariants before granting public serialization authority."""
        if type(self) is not ValidationEvidenceReference:
            raise ValueError("evidence reference must be exact ValidationEvidenceReference")
        evidence_id = _identifier(self.evidence_id, "evidence_id")
        artifact_fingerprint = _fingerprint(
            self.artifact_fingerprint,
            "artifact_fingerprint",
        )
        evidence_class = _enum_value(
            self.evidence_class,
            ValidationEvidenceClass,
            "evidence_class",
        )
        status = _enum_value(
            self.status,
            ValidationEvidenceStatus,
            "status",
        )
        available_time = _utc_datetime(self.available_time, "available_time")
        limitation_ids = _bounded_identifiers(self.limitation_ids, "limitation_ids")
        return {
            "evidence_id": evidence_id,
            "artifact_fingerprint": artifact_fingerprint,
            "evidence_class": evidence_class.value,
            "status": status.value,
            "available_time": _timestamp_text(available_time),
            "limitation_ids": list(limitation_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible evidence metadata."""
        return self._validated_content()


@dataclass(frozen=True, slots=True)
class ValidationProfile:
    """Preregistered domain-neutral profile for external validation evidence."""

    validation_profile_id: str
    protocol_fingerprint: str
    assessment_fingerprint: str
    rubric_fingerprint: str
    item_bank_fingerprint: str
    model_fingerprint: str
    intended_construct: str
    score_interpretation: str
    population: str
    setting: str
    decision_use: str
    analysis_cutoff: datetime
    evidence_references: tuple[ValidationEvidenceReference, ...]

    def __post_init__(self) -> None:
        """Seal preregistration identity and reject unavailable or ambiguous evidence."""
        for name in (
            "validation_profile_id",
            "intended_construct",
            "score_interpretation",
            "population",
            "setting",
            "decision_use",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "protocol_fingerprint",
            "assessment_fingerprint",
            "rubric_fingerprint",
            "item_bank_fingerprint",
            "model_fingerprint",
        ):
            object.__setattr__(self, name, _fingerprint(getattr(self, name), name))
        cutoff = _utc_datetime(self.analysis_cutoff, "analysis_cutoff")
        object.__setattr__(self, "analysis_cutoff", cutoff)

        raw = self.evidence_references
        if type(raw) not in {list, tuple}:
            raise ValueError("evidence_references must be an exact list or tuple")
        if len(raw) > MAX_VALIDATION_EVIDENCE:
            raise ValueError(
                f"evidence_references exceeds maximum {MAX_VALIDATION_EVIDENCE}"
            )

        normalized: list[ValidationEvidenceReference] = []
        seen_ids: dict[str, str] = {}
        for reference in raw:
            if type(reference) is not ValidationEvidenceReference:
                raise ValueError(
                    "evidence_references must contain exact ValidationEvidenceReference values"
                )
            content = reference._validated_content()
            evidence_id = content["evidence_id"]
            if evidence_id in seen_ids:
                raise ValueError("evidence_id must be unique within a validation profile")
            seen_ids[evidence_id] = content["artifact_fingerprint"]
            if reference.available_time > cutoff:
                raise ValueError("available_time must not exceed analysis_cutoff")
            normalized.append(reference)

        object.__setattr__(
            self,
            "evidence_references",
            tuple(sorted(normalized, key=lambda reference: reference.evidence_id)),
        )

    def _validated_content(self) -> dict[str, Any]:
        """Replay profile and nested-evidence invariants before public identity use."""
        if type(self) is not ValidationProfile:
            raise ValueError("validation profile must be exact ValidationProfile")

        scalar_values = {
            name: _identifier(getattr(self, name), name)
            for name in (
                "validation_profile_id",
                "intended_construct",
                "score_interpretation",
                "population",
                "setting",
                "decision_use",
            )
        }
        fingerprint_values = {
            name: _fingerprint(getattr(self, name), name)
            for name in (
                "protocol_fingerprint",
                "assessment_fingerprint",
                "rubric_fingerprint",
                "item_bank_fingerprint",
                "model_fingerprint",
            )
        }
        cutoff = _utc_datetime(self.analysis_cutoff, "analysis_cutoff")

        raw = self.evidence_references
        if type(raw) not in {list, tuple}:
            raise ValueError("evidence_references must be an exact list or tuple")
        if len(raw) > MAX_VALIDATION_EVIDENCE:
            raise ValueError(
                f"evidence_references exceeds maximum {MAX_VALIDATION_EVIDENCE}"
            )

        evidence_payloads: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for reference in raw:
            if type(reference) is not ValidationEvidenceReference:
                raise ValueError(
                    "evidence_references must contain exact ValidationEvidenceReference values"
                )
            payload = reference._validated_content()
            evidence_id = payload["evidence_id"]
            if evidence_id in seen_ids:
                raise ValueError("evidence_id must be unique within a validation profile")
            seen_ids.add(evidence_id)
            available_time = _utc_datetime(reference.available_time, "available_time")
            if available_time > cutoff:
                raise ValueError("available_time must not exceed analysis_cutoff")
            evidence_payloads.append(payload)

        evidence_payloads.sort(key=lambda payload: payload["evidence_id"])
        return {
            "validation_profile_id": scalar_values["validation_profile_id"],
            "protocol_fingerprint": fingerprint_values["protocol_fingerprint"],
            "assessment_fingerprint": fingerprint_values["assessment_fingerprint"],
            "rubric_fingerprint": fingerprint_values["rubric_fingerprint"],
            "item_bank_fingerprint": fingerprint_values["item_bank_fingerprint"],
            "model_fingerprint": fingerprint_values["model_fingerprint"],
            "intended_construct": scalar_values["intended_construct"],
            "score_interpretation": scalar_values["score_interpretation"],
            "population": scalar_values["population"],
            "setting": scalar_values["setting"],
            "decision_use": scalar_values["decision_use"],
            "analysis_cutoff": _timestamp_text(cutoff),
            "evidence_references": evidence_payloads,
        }

    def _content_dict(self) -> dict[str, Any]:
        """Return replay-validated source-free content without the derived digest."""
        return self._validated_content()

    @property
    def profile_fingerprint(self) -> str:
        """Return the deterministic SHA-256 identity of canonical profile content."""
        return _canonical_sha256(self._validated_content())

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON-compatible profile content and its fingerprint."""
        content = self._validated_content()
        return {
            **content,
            "profile_fingerprint": _canonical_sha256(content),
        }


__all__ = [
    "MAX_VALIDATION_EVIDENCE",
    "ValidationEvidenceClass",
    "ValidationEvidenceReference",
    "ValidationEvidenceStatus",
    "ValidationProfile",
]
