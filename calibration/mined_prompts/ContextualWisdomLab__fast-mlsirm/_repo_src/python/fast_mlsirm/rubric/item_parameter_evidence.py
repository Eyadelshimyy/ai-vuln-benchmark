"""Immutable provisional/calibrated parameter provenance for governed item banks.

This module records *how a parameter artifact is allowed to be interpreted*; it
performs no calibration, prediction, shrinkage, linking, scoring, or other
psychometric arithmetic. Production numerical work remains Rust-owned.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any

from .models import (
    SCHEMA_VERSION,
    _FINGERPRINT_PATTERN,
    _identifier,
    _schema_version,
    _semantic_version,
    _sha256_hex,
)

_CREATION_TOKEN = object()
_INSTANCE_FIELDS = frozenset(
    {
        "item_id",
        "item_version",
        "response_model_id",
        "parameter_artifact_fingerprint",
        "status",
        "provisional_method",
        "provisional_basis_fingerprint",
        "calibration_evidence_fingerprint",
        "schema_version",
        "_evidence_fingerprint",
    }
)


class ItemParameterStatus(str, Enum):
    """Governed interpretation status of one item-parameter artifact."""

    PROVISIONAL = "provisional"
    CALIBRATED = "calibrated"


class ProvisionalParameterMethod(str, Enum):
    """Declared non-definitive strategies allowed before empirical calibration."""

    RASCH_COMMON_DISCRIMINATION = "rasch_common_discrimination"
    TEMPLATE_PRIOR = "template_prior"
    LLTM_PREDICTED = "lltm_predicted"
    CONSTRAINED_PRIOR = "constrained_prior"


def _fingerprint(value: object, name: str) -> str:
    """Validate one exact complete lowercase SHA-256 identity."""

    if type(value) is not str or _FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a complete lowercase SHA-256 fingerprint")
    return value


def _optional_fingerprint(value: object | None, name: str) -> str | None:
    """Validate one optional exact complete SHA-256 identity."""

    if value is None:
        return None
    return _fingerprint(value, name)


def _status(value: object) -> ItemParameterStatus:
    """Normalize one exact package status without caller conversion callbacks."""

    if type(value) is ItemParameterStatus:
        return value
    if type(value) is str:
        try:
            return ItemParameterStatus(value)
        except ValueError:
            pass
    raise ValueError("status must be 'provisional' or 'calibrated'")


def _provisional_method(
    value: object | None,
) -> ProvisionalParameterMethod | None:
    """Normalize one optional exact provisional-method identity."""

    if value is None:
        return None
    if type(value) is ProvisionalParameterMethod:
        return value
    if type(value) is str:
        try:
            return ProvisionalParameterMethod(value)
        except ValueError:
            pass
    choices = ", ".join(method.value for method in ProvisionalParameterMethod)
    raise ValueError(f"provisional_method must be one of: {choices}")


@dataclass(frozen=True)
class ItemParameterEvidence:
    """Source-text-free provenance for one governed item-parameter artifact.

    ``provisional`` evidence explicitly identifies the bounded cold-start method
    and the exact basis artifact from which the provisional parameters were
    derived. ``calibrated`` evidence instead identifies completed calibration
    evidence and cannot carry provisional provenance. The record contains no raw
    responses, prompts, provider output, or numerical fitting implementation.
    """

    item_id: str
    item_version: str
    response_model_id: str
    parameter_artifact_fingerprint: str
    status: ItemParameterStatus
    provisional_method: ProvisionalParameterMethod | None
    provisional_basis_fingerprint: str | None
    calibration_evidence_fingerprint: str | None
    schema_version: str = SCHEMA_VERSION
    _creation_token: InitVar[object | None] = None
    _evidence_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self, _creation_token: object | None) -> None:
        """Seal one record behind the package-owned factory and its invariants."""

        if _creation_token is not _CREATION_TOKEN:
            raise TypeError("use build_item_parameter_evidence to create parameter evidence")
        if self.status is ItemParameterStatus.PROVISIONAL:
            if self.provisional_method is None:
                raise ValueError(
                    "provisional parameter evidence requires a provisional method"
                )
            if self.provisional_basis_fingerprint is None:
                raise ValueError(
                    "provisional parameter evidence requires a provisional basis"
                )
            if self.provisional_basis_fingerprint == self.parameter_artifact_fingerprint:
                raise ValueError(
                    "provisional basis must not be the parameter artifact itself"
                )
            if self.calibration_evidence_fingerprint is not None:
                raise ValueError(
                    "provisional parameter evidence must not carry calibration evidence"
                )
        else:
            if (
                self.provisional_method is not None
                or self.provisional_basis_fingerprint is not None
            ):
                raise ValueError(
                    "calibrated parameter evidence must not carry provisional provenance"
                )
            if self.calibration_evidence_fingerprint is None:
                raise ValueError(
                    "calibrated parameter evidence requires calibration evidence"
                )
            if self.calibration_evidence_fingerprint == self.parameter_artifact_fingerprint:
                raise ValueError(
                    "calibration evidence must not be the parameter artifact itself"
                )
        object.__setattr__(
            self,
            "_evidence_fingerprint",
            _sha256_hex(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical provenance content without its derived fingerprint."""

        return {
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "item_version": self.item_version,
            "response_model_id": self.response_model_id,
            "parameter_artifact_fingerprint": self.parameter_artifact_fingerprint,
            "status": self.status.value,
            "provisional_method": (
                None if self.provisional_method is None else self.provisional_method.value
            ),
            "provisional_basis_fingerprint": self.provisional_basis_fingerprint,
            "calibration_evidence_fingerprint": self.calibration_evidence_fingerprint,
        }

    @property
    def evidence_fingerprint(self) -> str:
        """Return the creation-time SHA-256 identity of this provenance record."""

        _verify_item_parameter_evidence(self)
        return self._evidence_fingerprint

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation after invariant replay."""

        _verify_item_parameter_evidence(self)
        return {
            **self._content_dict(),
            "evidence_fingerprint": self._evidence_fingerprint,
        }


def build_item_parameter_evidence(
    *,
    item_id: str,
    item_version: str,
    response_model_id: str,
    parameter_artifact_fingerprint: str,
    status: ItemParameterStatus | str,
    provisional_method: ProvisionalParameterMethod | str | None = None,
    provisional_basis_fingerprint: str | None = None,
    calibration_evidence_fingerprint: str | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> ItemParameterEvidence:
    """Build one callback-safe provisional or calibrated parameter record."""

    normalized = ItemParameterEvidence(
        item_id=_identifier(item_id, "item_id"),
        item_version=_semantic_version(item_version, "item_version"),
        response_model_id=_identifier(response_model_id, "response_model_id"),
        parameter_artifact_fingerprint=_fingerprint(
            parameter_artifact_fingerprint,
            "parameter_artifact_fingerprint",
        ),
        status=_status(status),
        provisional_method=_provisional_method(provisional_method),
        provisional_basis_fingerprint=_optional_fingerprint(
            provisional_basis_fingerprint,
            "provisional_basis_fingerprint",
        ),
        calibration_evidence_fingerprint=_optional_fingerprint(
            calibration_evidence_fingerprint,
            "calibration_evidence_fingerprint",
        ),
        schema_version=_schema_version(schema_version),
        _creation_token=_CREATION_TOKEN,
    )
    return normalized


def _verify_item_parameter_evidence(
    evidence: object,
) -> ItemParameterEvidence:
    """Replay one exact package record before granting provenance authority."""

    if type(evidence) is not ItemParameterEvidence:
        raise ValueError("parameter evidence must be an exact ItemParameterEvidence")
    state = vars(evidence)
    if frozenset(state) != _INSTANCE_FIELDS:
        raise ValueError("parameter evidence no longer matches its creation-time identity")
    string_fields = (
        "item_id",
        "item_version",
        "response_model_id",
        "parameter_artifact_fingerprint",
        "schema_version",
        "_evidence_fingerprint",
    )
    if any(type(state[name]) is not str for name in string_fields):
        raise ValueError("parameter evidence no longer matches its creation-time identity")
    if type(state["status"]) is not ItemParameterStatus:
        raise ValueError("parameter evidence no longer matches its creation-time identity")
    method = state["provisional_method"]
    if method is not None and type(method) is not ProvisionalParameterMethod:
        raise ValueError("parameter evidence no longer matches its creation-time identity")
    for name in (
        "provisional_basis_fingerprint",
        "calibration_evidence_fingerprint",
    ):
        value = state[name]
        if value is not None and type(value) is not str:
            raise ValueError("parameter evidence no longer matches its creation-time identity")

    try:
        replay = build_item_parameter_evidence(
            item_id=state["item_id"],
            item_version=state["item_version"],
            response_model_id=state["response_model_id"],
            parameter_artifact_fingerprint=state["parameter_artifact_fingerprint"],
            status=state["status"],
            provisional_method=method,
            provisional_basis_fingerprint=state["provisional_basis_fingerprint"],
            calibration_evidence_fingerprint=state[
                "calibration_evidence_fingerprint"
            ],
            schema_version=state["schema_version"],
        )
    except (TypeError, ValueError):
        raise ValueError(
            "parameter evidence no longer matches its creation-time identity"
        ) from None
    if replay._evidence_fingerprint != state["_evidence_fingerprint"]:
        raise ValueError("parameter evidence no longer matches its creation-time identity")
    return evidence


__all__ = [
    "ItemParameterEvidence",
    "ItemParameterStatus",
    "ProvisionalParameterMethod",
    "build_item_parameter_evidence",
]
