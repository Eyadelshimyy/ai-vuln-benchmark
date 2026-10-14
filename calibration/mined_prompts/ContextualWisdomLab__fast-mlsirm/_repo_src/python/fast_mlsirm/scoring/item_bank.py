"""Immutable logical contracts for governed item-bank lifecycle state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any

from ._contract_safety import (
    artifact_digest,
    descriptive_identifier,
    enum_value,
    freeze_metadata,
    semantic_version,
    sorted_fingerprints,
)
from ._validation import CanonicalContract, assessment_error, fingerprint, thaw_json_value

_ENTRY_TOKEN = object()
_RELEASE_TOKEN = object()


class ItemLifecycleState(str, Enum):
    """Governed maturity state for one immutable item version."""

    DRAFT = "draft"
    AUDITED = "audited"
    SCREENED = "screened"
    PILOTING = "piloting"
    CALIBRATED = "calibrated"
    APPROVED = "approved"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


def _optional_digest(value: Any, name: str) -> str | None:
    """Validate one optional SHA-256 provenance identity."""
    return None if value is None else fingerprint(value, name)


def _evidence_digests(values: Any, name: str) -> tuple[str, ...]:
    """Canonicalize one bounded evidence-fingerprint collection."""
    return sorted_fingerprints(values, name, minimum=0, maximum=64)


@dataclass(frozen=True)
class ItemBankEntry(CanonicalContract):
    """Exact immutable item version plus cumulative lifecycle evidence."""

    entry_id: str
    item_id: str
    item_version: str
    rubric_fingerprint: str
    blueprint_fingerprint: str
    generation_contract_fingerprint: str
    item_content_fingerprint: str
    lifecycle_state: ItemLifecycleState
    audit_evidence_fingerprints: tuple[str, ...]
    screening_result_fingerprints: tuple[str, ...]
    pilot_assignment_fingerprints: tuple[str, ...]
    calibration_evidence_fingerprints: tuple[str, ...]
    approval_decision_fingerprint: str | None
    retirement_decision_fingerprint: str | None
    predecessor_entry_fingerprint: str | None
    metadata: Mapping[str, Any]
    _token: InitVar[object | None] = None

    def __post_init__(self, _token: object | None) -> None:
        """Validate factory sealing, provenance, and claimed maturity evidence."""
        if _token is not _ENTRY_TOKEN:
            raise assessment_error(
                "unverified_item_bank_entry",
                "$",
                "use build_item_bank_entry",
            )
        object.__setattr__(
            self,
            "entry_id",
            descriptive_identifier(self.entry_id, "entry_id"),
        )
        object.__setattr__(
            self,
            "item_id",
            descriptive_identifier(self.item_id, "item_id"),
        )
        object.__setattr__(
            self,
            "item_version",
            semantic_version(self.item_version, "item_version"),
        )
        for name in (
            "rubric_fingerprint",
            "blueprint_fingerprint",
            "generation_contract_fingerprint",
            "item_content_fingerprint",
        ):
            object.__setattr__(self, name, fingerprint(getattr(self, name), name))
        object.__setattr__(
            self,
            "lifecycle_state",
            enum_value(self.lifecycle_state, ItemLifecycleState, "lifecycle_state"),
        )
        for name in (
            "audit_evidence_fingerprints",
            "screening_result_fingerprints",
            "pilot_assignment_fingerprints",
            "calibration_evidence_fingerprints",
        ):
            object.__setattr__(
                self,
                name,
                _evidence_digests(getattr(self, name), name),
            )
        for name in (
            "approval_decision_fingerprint",
            "retirement_decision_fingerprint",
            "predecessor_entry_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _optional_digest(getattr(self, name), name),
            )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        self._validate_evidence_for_state()

    def _validate_evidence_for_state(self) -> None:
        """Require cumulative governance evidence for every claimed maturity state."""
        order = list(ItemLifecycleState)
        rank = order.index(self.lifecycle_state)
        requirements = (
            (
                ItemLifecycleState.AUDITED,
                self.audit_evidence_fingerprints,
                "audit_evidence_required",
                "$.audit_evidence_fingerprints",
            ),
            (
                ItemLifecycleState.SCREENED,
                self.screening_result_fingerprints,
                "screening_evidence_required",
                "$.screening_result_fingerprints",
            ),
            (
                ItemLifecycleState.PILOTING,
                self.pilot_assignment_fingerprints,
                "pilot_evidence_required",
                "$.pilot_assignment_fingerprints",
            ),
            (
                ItemLifecycleState.CALIBRATED,
                self.calibration_evidence_fingerprints,
                "calibration_evidence_required",
                "$.calibration_evidence_fingerprints",
            ),
            (
                ItemLifecycleState.APPROVED,
                self.approval_decision_fingerprint,
                "approval_decision_required",
                "$.approval_decision_fingerprint",
            ),
        )
        for state, evidence, code, path in requirements:
            if rank >= order.index(state) and not evidence:
                raise assessment_error(
                    code,
                    path,
                    "lifecycle state requires cumulative governance evidence",
                )
        if (
            self.lifecycle_state is ItemLifecycleState.RETIRED
            and self.retirement_decision_fingerprint is None
        ):
            raise assessment_error(
                "retirement_decision_required",
                "$.retirement_decision_fingerprint",
                "retired state requires retirement provenance",
            )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical entry content excluding derived identities."""
        return {
            "entry_id": self.entry_id,
            "item_id": self.item_id,
            "item_version": self.item_version,
            "rubric_fingerprint": self.rubric_fingerprint,
            "blueprint_fingerprint": self.blueprint_fingerprint,
            "generation_contract_fingerprint": self.generation_contract_fingerprint,
            "item_content_fingerprint": self.item_content_fingerprint,
            "lifecycle_state": self.lifecycle_state.value,
            "audit_evidence_fingerprints": list(self.audit_evidence_fingerprints),
            "screening_result_fingerprints": list(self.screening_result_fingerprints),
            "pilot_assignment_fingerprints": list(self.pilot_assignment_fingerprints),
            "calibration_evidence_fingerprints": list(
                self.calibration_evidence_fingerprints
            ),
            "approval_decision_fingerprint": self.approval_decision_fingerprint,
            "retirement_decision_fingerprint": self.retirement_decision_fingerprint,
            "predecessor_entry_fingerprint": self.predecessor_entry_fingerprint,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def entry_fingerprint(self) -> str:
        """Return the exact immutable item-bank entry digest."""
        return artifact_digest(self)

    @property
    def entry_handle(self) -> str:
        """Return a compact public handle derived from the full digest."""
        return f"item_bank_entry_{self.entry_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical content plus deterministic item-entry identities."""
        return {
            **self._content_dict(),
            "entry_handle": self.entry_handle,
            "entry_fingerprint": self.entry_fingerprint,
        }


@dataclass(frozen=True)
class ItemBankRelease(CanonicalContract):
    """Immutable release manifest over exact item-entry versions."""

    release_id: str
    release_version: str
    entry_fingerprints: tuple[str, ...]
    predecessor_release_fingerprint: str | None
    cross_version_comparable: bool
    linking_evidence_fingerprints: tuple[str, ...]
    metadata: Mapping[str, Any]
    _token: InitVar[object | None] = None

    def __post_init__(self, _token: object | None) -> None:
        """Validate factory sealing and cross-version comparability provenance."""
        if _token is not _RELEASE_TOKEN:
            raise assessment_error(
                "unverified_item_bank_release",
                "$",
                "use build_item_bank_release",
            )
        object.__setattr__(
            self,
            "release_id",
            descriptive_identifier(self.release_id, "release_id"),
        )
        object.__setattr__(
            self,
            "release_version",
            semantic_version(self.release_version, "release_version"),
        )
        object.__setattr__(
            self,
            "entry_fingerprints",
            sorted_fingerprints(
                self.entry_fingerprints,
                "entry_fingerprints",
                minimum=1,
                maximum=1024,
            ),
        )
        object.__setattr__(
            self,
            "predecessor_release_fingerprint",
            _optional_digest(
                self.predecessor_release_fingerprint,
                "predecessor_release_fingerprint",
            ),
        )
        if type(self.cross_version_comparable) is not bool:
            raise assessment_error(
                "invalid_cross_version_comparable",
                "$.cross_version_comparable",
                "cross_version_comparable must be boolean",
            )
        object.__setattr__(
            self,
            "linking_evidence_fingerprints",
            _evidence_digests(
                self.linking_evidence_fingerprints,
                "linking_evidence_fingerprints",
            ),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        if self.cross_version_comparable and self.predecessor_release_fingerprint is None:
            raise assessment_error(
                "predecessor_release_required",
                "$.predecessor_release_fingerprint",
                "comparability requires predecessor",
            )
        if self.cross_version_comparable and not self.linking_evidence_fingerprints:
            raise assessment_error(
                "linking_evidence_required",
                "$.linking_evidence_fingerprints",
                "comparability requires linking evidence",
            )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical release content excluding derived identities."""
        return {
            "release_id": self.release_id,
            "release_version": self.release_version,
            "entry_fingerprints": list(self.entry_fingerprints),
            "predecessor_release_fingerprint": self.predecessor_release_fingerprint,
            "cross_version_comparable": self.cross_version_comparable,
            "linking_evidence_fingerprints": list(self.linking_evidence_fingerprints),
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def release_fingerprint(self) -> str:
        """Return the exact immutable item-bank release digest."""
        return artifact_digest(self)

    @property
    def release_handle(self) -> str:
        """Return a compact public handle derived from the full digest."""
        return f"item_bank_release_{self.release_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical content plus deterministic release identities."""
        return {
            **self._content_dict(),
            "release_handle": self.release_handle,
            "release_fingerprint": self.release_fingerprint,
        }


def build_item_bank_entry(**values: Any) -> ItemBankEntry:
    """Build one validated immutable item-bank entry."""
    normalized = dict(values)
    normalized.setdefault("approval_decision_fingerprint", None)
    normalized.setdefault("retirement_decision_fingerprint", None)
    normalized.setdefault("predecessor_entry_fingerprint", None)
    return ItemBankEntry(**normalized, _token=_ENTRY_TOKEN)


def build_item_bank_release(**values: Any) -> ItemBankRelease:
    """Build one validated immutable item-bank release."""
    return ItemBankRelease(**values, _token=_RELEASE_TOKEN)
