"""Replay-integrity tests for external-validation manifests."""

from __future__ import annotations

import pytest

from fast_mlsirm.external_validation import (
    EvidenceClass,
    EvidenceStatus,
    ExternalValidationProfile,
    ValidationEvidence,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


class _HostileTuple(tuple[object, ...]):
    """Tuple subclass whose iteration callback must never run during replay."""

    callbacks = 0

    def __iter__(self):  # type: ignore[override]
        """Fail if replay iterates a caller-controlled tuple subclass."""
        type(self).callbacks += 1
        raise AssertionError("caller tuple callback executed")


class _HostileStatus:
    """Object whose enum-value callback must never run during replay."""

    callbacks = 0

    @property
    def value(self) -> str:
        """Fail if replay trusts a rebound evidence status."""
        type(self).callbacks += 1
        raise AssertionError("caller enum callback executed")


def _profile() -> ExternalValidationProfile:
    """Build one valid source-free external-validation profile."""
    evidence = ValidationEvidence(
        evidence_id="technical_conformance",
        evidence_class=EvidenceClass.TECHNICAL,
        status=EvidenceStatus.PASSED,
        available_time="2026-07-01T00:00:00Z",
        artifact_sha256=_SHA_D,
        limitation="aggregate-only evidence",
    )
    return ExternalValidationProfile(
        validation_profile_id="external_validity_v1",
        construct="reasoning quality",
        score_interpretation="higher scores indicate stronger reasoning",
        population="adult knowledge workers",
        setting="independent held-out evaluation",
        decision_use="research validation only",
        assessment_fingerprint=_SHA_A,
        rubric_fingerprint=_SHA_B,
        item_bank_fingerprint=_SHA_C,
        model_fingerprint=_SHA_D,
        development_dataset_ids=("development_set_v1",),
        internal_validation_dataset_ids=("internal_validation_v1",),
        external_validation_dataset_ids=("external_validation_v1",),
        sites=("site_alpha",),
        languages=("en",),
        preregistration_reference="protocol_2026_001",
        preregistered_at="2026-06-01T00:00:00Z",
        analysis_cutoff="2026-08-01T00:00:00Z",
        data_license="synthetic_and_open",
        purpose_classification="research_validation",
        evidence=(evidence,),
    )


def test_manifest_revalidates_rebound_profile_container_before_iteration() -> None:
    """Reject a post-construction container rebind before caller iteration."""
    profile = _profile()
    object.__setattr__(profile, "languages", _HostileTuple(("en",)))
    _HostileTuple.callbacks = 0

    with pytest.raises(ValueError, match="languages must be a list or tuple"):
        profile.to_manifest()

    assert _HostileTuple.callbacks == 0


def test_manifest_revalidates_rebound_evidence_status_before_value_access() -> None:
    """Reject post-construction evidence mutation before enum callbacks."""
    profile = _profile()
    object.__setattr__(profile.evidence[0], "status", _HostileStatus())
    _HostileStatus.callbacks = 0

    with pytest.raises(ValueError, match=r"evidence\[0\]\.status must be an EvidenceStatus"):
        profile.to_manifest()

    assert _HostileStatus.callbacks == 0
