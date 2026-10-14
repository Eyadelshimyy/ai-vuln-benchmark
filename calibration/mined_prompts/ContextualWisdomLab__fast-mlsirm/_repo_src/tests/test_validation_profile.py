"""Public contract tests for preregistered external-validation profiles."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo

import pytest

from fast_mlsirm.validation_profile import (
    MAX_VALIDATION_EVIDENCE,
    ValidationEvidenceClass,
    ValidationEvidenceReference,
    ValidationEvidenceStatus,
    ValidationProfile,
)


FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64
FINGERPRINT_C = "c" * 64
FINGERPRINT_D = "d" * 64
FINGERPRINT_E = "e" * 64
FINGERPRINT_F = "f" * 64


def _evidence(
    evidence_id: str,
    *,
    evidence_class: ValidationEvidenceClass = ValidationEvidenceClass.TECHNICAL,
    status: ValidationEvidenceStatus = ValidationEvidenceStatus.PASSED,
    available_time: datetime | None = None,
    artifact_fingerprint: str = FINGERPRINT_A,
    limitation_ids: tuple[str, ...] = (),
) -> ValidationEvidenceReference:
    """Build one exact public evidence fixture."""
    return ValidationEvidenceReference(
        evidence_id=evidence_id,
        artifact_fingerprint=artifact_fingerprint,
        evidence_class=evidence_class,
        status=status,
        available_time=available_time
        or datetime(2026, 8, 20, 12, 30, tzinfo=timezone.utc),
        limitation_ids=limitation_ids,
    )


def _profile(*evidence: ValidationEvidenceReference) -> ValidationProfile:
    """Build one exact preregistered profile fixture."""
    return ValidationProfile(
        validation_profile_id="enterprise_validation_profile",
        protocol_fingerprint=FINGERPRINT_A,
        assessment_fingerprint=FINGERPRINT_B,
        rubric_fingerprint=FINGERPRINT_C,
        item_bank_fingerprint=FINGERPRINT_D,
        model_fingerprint=FINGERPRINT_E,
        intended_construct="decision_quality",
        score_interpretation="relative_decision_quality",
        population="held_out_enterprise_systems",
        setting="external_validation_sites",
        decision_use="human_review_prioritization",
        analysis_cutoff=datetime(2026, 8, 21, tzinfo=timezone.utc),
        evidence_references=evidence,
    )


def test_profile_preserves_distinct_evidence_classes_and_failure_states() -> None:
    """Serialization never collapses explicit evidence classes or statuses."""
    profile = _profile(
        _evidence("technical_evidence"),
        _evidence(
            "construct_evidence",
            evidence_class=ValidationEvidenceClass.CONSTRUCT,
            status=ValidationEvidenceStatus.INDETERMINATE,
            artifact_fingerprint=FINGERPRINT_B,
        ),
        _evidence(
            "transport_evidence",
            evidence_class=ValidationEvidenceClass.TRANSPORTABILITY,
            status=ValidationEvidenceStatus.FAILED,
            artifact_fingerprint=FINGERPRINT_C,
        ),
        _evidence(
            "fairness_evidence",
            evidence_class=ValidationEvidenceClass.FAIRNESS,
            status=ValidationEvidenceStatus.NOT_EXECUTED,
            artifact_fingerprint=FINGERPRINT_D,
        ),
        _evidence(
            "utility_evidence",
            evidence_class=ValidationEvidenceClass.DECISION_UTILITY,
            status=ValidationEvidenceStatus.NOT_APPLICABLE,
            artifact_fingerprint=FINGERPRINT_E,
            limitation_ids=("no_operational_utility_claim",),
        ),
    )

    payload = profile.to_dict()
    rows = {row["evidence_id"]: row for row in payload["evidence_references"]}

    assert rows["technical_evidence"]["evidence_class"] == "technical"
    assert rows["construct_evidence"]["status"] == "indeterminate"
    assert rows["transport_evidence"]["status"] == "failed"
    assert rows["fairness_evidence"]["status"] == "not_executed"
    assert rows["utility_evidence"]["status"] == "not_applicable"
    assert "aggregate_score" not in payload


def test_profile_fingerprint_is_stable_across_input_order_and_timezones() -> None:
    """Canonical ordering and UTC normalization define deterministic identity."""
    eastern = timezone(timedelta(hours=9))
    first = _evidence(
        "evidence_b",
        available_time=datetime(2026, 8, 20, 21, 30, tzinfo=eastern),
        artifact_fingerprint=FINGERPRINT_B,
    )
    second = _evidence("evidence_a", artifact_fingerprint=FINGERPRINT_C)

    left = _profile(first, second)
    right = _profile(second, first)

    assert left.profile_fingerprint == right.profile_fingerprint
    assert left.to_dict() == right.to_dict()
    assert left.to_dict()["evidence_references"][1]["available_time"] == (
        "2026-08-20T12:30:00Z"
    )


def test_future_available_evidence_fails_before_profile_claim() -> None:
    """Evidence unavailable at analysis cutoff cannot enter the profile."""
    future = _evidence(
        "future_evidence",
        available_time=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError, match="available_time must not exceed analysis_cutoff"):
        _profile(future)


def test_duplicate_evidence_identity_is_rejected() -> None:
    """One evidence id cannot ambiguously name multiple artifacts."""
    left = _evidence("duplicate_evidence", artifact_fingerprint=FINGERPRINT_A)
    right = _evidence("duplicate_evidence", artifact_fingerprint=FINGERPRINT_F)

    with pytest.raises(ValueError, match="evidence_id must be unique"):
        _profile(left, right)


def test_callback_bearing_evidence_collection_fails_before_iteration() -> None:
    """Caller collection protocols are not executed by profile admission."""
    callbacks = 0

    class HostileList(list):
        def __iter__(self):
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller iteration must not execute")

    values = HostileList([_evidence("technical_evidence")])

    with pytest.raises(ValueError, match="evidence_references must be an exact list or tuple"):
        ValidationProfile(
            validation_profile_id="enterprise_validation_profile",
            protocol_fingerprint=FINGERPRINT_A,
            assessment_fingerprint=FINGERPRINT_B,
            rubric_fingerprint=FINGERPRINT_C,
            item_bank_fingerprint=FINGERPRINT_D,
            model_fingerprint=FINGERPRINT_E,
            intended_construct="decision_quality",
            score_interpretation="relative_decision_quality",
            population="held_out_enterprise_systems",
            setting="external_validation_sites",
            decision_use="human_review_prioritization",
            analysis_cutoff=datetime(2026, 8, 21, tzinfo=timezone.utc),
            evidence_references=values,
        )

    assert callbacks == 0


def test_callback_bearing_identifier_fails_without_string_conversion() -> None:
    """Caller string subclasses are rejected without executing conversion hooks."""
    callbacks = 0

    class HostileString(str):
        def __str__(self):
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller string conversion must not execute")

    with pytest.raises(ValueError, match="evidence_id must be an exact string"):
        _evidence(HostileString("hostile_evidence"))

    assert callbacks == 0


def test_timestamps_must_be_exact_offset_aware_datetimes() -> None:
    """Naive or caller-subclass timestamps cannot define validation chronology."""
    with pytest.raises(ValueError, match="available_time must be offset-aware"):
        _evidence("naive_evidence", available_time=datetime(2026, 8, 20))

    class HostileDatetime(datetime):
        pass

    with pytest.raises(ValueError, match="available_time must be an exact datetime"):
        _evidence(
            "subclass_time",
            available_time=HostileDatetime(2026, 8, 20, tzinfo=timezone.utc),
        )


def test_callback_bearing_timezone_fails_before_offset_protocol() -> None:
    """A caller tzinfo provider cannot execute while validation chronology is sealed."""
    callbacks = 0

    class HostileTimezone(tzinfo):
        def utcoffset(self, dt):
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller timezone offset must not execute")

        def dst(self, dt):
            return timedelta(0)

        def tzname(self, dt):
            return "hostile"

    value = datetime(2026, 8, 20, tzinfo=HostileTimezone())

    with pytest.raises(ValueError, match="available_time must use datetime.timezone"):
        _evidence("hostile_timezone", available_time=value)

    assert callbacks == 0


def test_evidence_collection_limit_is_checked_before_member_validation() -> None:
    """The bounded profile rejects an oversized exact carrier before item work."""
    values = [object()] * (MAX_VALIDATION_EVIDENCE + 1)

    with pytest.raises(ValueError, match="evidence_references exceeds maximum"):
        ValidationProfile(
            validation_profile_id="enterprise_validation_profile",
            protocol_fingerprint=FINGERPRINT_A,
            assessment_fingerprint=FINGERPRINT_B,
            rubric_fingerprint=FINGERPRINT_C,
            item_bank_fingerprint=FINGERPRINT_D,
            model_fingerprint=FINGERPRINT_E,
            intended_construct="decision_quality",
            score_interpretation="relative_decision_quality",
            population="held_out_enterprise_systems",
            setting="external_validation_sites",
            decision_use="human_review_prioritization",
            analysis_cutoff=datetime(2026, 8, 21, tzinfo=timezone.utc),
            evidence_references=values,
        )


def test_limitation_identifiers_are_bounded_and_canonicalized() -> None:
    """Limitations remain explicit, unique, and deterministically ordered."""
    reference = _evidence(
        "limited_evidence",
        limitation_ids=("underpowered_subgroup", "criterion_reliability", "criterion_reliability"),
    )

    assert reference.limitation_ids == (
        "criterion_reliability",
        "underpowered_subgroup",
    )


def test_profile_public_identity_replays_post_construction_invariants() -> None:
    """Post-construction profile mutation cannot gain fingerprint/serialization authority."""
    profile = _profile(_evidence("technical_evidence"))
    original_payload = profile.to_dict()
    original_fingerprint = profile.profile_fingerprint

    object.__setattr__(profile, "validation_profile_id", "bad profile id")

    with pytest.raises(ValueError, match="validation_profile_id must be an opaque identifier"):
        _ = profile.profile_fingerprint
    with pytest.raises(ValueError, match="validation_profile_id must be an opaque identifier"):
        profile.to_dict()

    valid = _profile(_evidence("technical_evidence"))
    assert valid.profile_fingerprint == original_fingerprint
    assert valid.to_dict() == original_payload


def test_profile_replay_rejects_callback_bearing_mutated_collection_before_iteration() -> None:
    """Serialization replay rejects substituted collection subclasses without callbacks."""
    callbacks = 0

    class HostileTuple(tuple):
        def __iter__(self):
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller iteration must not execute")

    profile = _profile(_evidence("technical_evidence"))
    object.__setattr__(
        profile,
        "evidence_references",
        HostileTuple(profile.evidence_references),
    )

    with pytest.raises(ValueError, match="evidence_references must be an exact list or tuple"):
        profile.to_dict()
    assert callbacks == 0


def test_profile_replay_rejects_mutated_nested_evidence_chronology() -> None:
    """Nested evidence is revalidated before it contributes to profile identity."""
    reference = _evidence("technical_evidence")
    profile = _profile(reference)
    object.__setattr__(
        reference,
        "available_time",
        datetime(2026, 8, 22, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError, match="available_time must not exceed analysis_cutoff"):
        _ = profile.profile_fingerprint
    with pytest.raises(ValueError, match="available_time must not exceed analysis_cutoff"):
        profile.to_dict()
