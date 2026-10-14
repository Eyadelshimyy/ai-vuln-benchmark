"""Contract tests for preregistered external-validity evidence profiles."""

from __future__ import annotations

from dataclasses import replace

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


class _HostileText(str):
    """String subclass whose callbacks must not run during admission."""

    callbacks = 0

    def strip(self, *args: object, **kwargs: object) -> str:
        """Fail if package validation dispatches caller-owned text behavior."""
        type(self).callbacks += 1
        raise AssertionError("caller text callback executed")

    def __hash__(self) -> int:
        """Fail if package validation hashes caller-owned text."""
        type(self).callbacks += 1
        raise AssertionError("caller text hash executed")

    def __lt__(self, other: object) -> bool:
        """Fail if package validation compares caller-owned text."""
        type(self).callbacks += 1
        raise AssertionError("caller text comparison executed")


def _evidence(
    evidence_id: str,
    evidence_class: EvidenceClass,
    *,
    status: EvidenceStatus = EvidenceStatus.PASSED,
    available_time: str = "2026-07-01T00:00:00Z",
) -> ValidationEvidence:
    """Build one source-free evidence reference for a profile fixture."""
    return ValidationEvidence(
        evidence_id=evidence_id,
        evidence_class=evidence_class,
        status=status,
        available_time=available_time,
        artifact_sha256=_SHA_D,
        limitation="aggregate-only evidence; no raw participant content",
    )


def _profile(*evidence: ValidationEvidence) -> ExternalValidationProfile:
    """Build a bounded provider-neutral validation profile fixture."""
    return ExternalValidationProfile(
        validation_profile_id="external_validity_v1",
        construct="reasoning quality",
        score_interpretation="higher scores indicate stronger rubric-aligned reasoning",
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
        sites=("site_alpha", "site_beta"),
        languages=("en", "ko"),
        preregistration_reference="protocol_2026_001",
        preregistered_at="2026-06-01T00:00:00Z",
        analysis_cutoff="2026-08-01T00:00:00Z",
        data_license="synthetic_and_open",
        purpose_classification="research_validation",
        evidence=evidence,
    )


def test_manifest_is_deterministic_and_keeps_evidence_classes_distinct() -> None:
    """Evidence order cannot change identity and classes remain separately visible."""
    technical = _evidence("technical_conformance", EvidenceClass.TECHNICAL)
    transport = _evidence(
        "transportability_holdout",
        EvidenceClass.TRANSPORTABILITY,
        status=EvidenceStatus.INDETERMINATE,
    )

    first = _profile(technical, transport)
    second = _profile(transport, technical)

    assert first.profile_fingerprint == second.profile_fingerprint
    manifest = first.to_manifest()
    assert manifest["profile_fingerprint"] == first.profile_fingerprint
    assert [row["evidence_class"] for row in manifest["evidence"]] == [
        "technical",
        "transportability",
    ]
    assert [row["status"] for row in manifest["evidence"]] == [
        "passed",
        "indeterminate",
    ]


def test_serialized_enums_optional_fields_and_offset_time_normalize() -> None:
    """Accept JSON-style enums while preserving explicit optional-field absence."""
    evidence = ValidationEvidence(
        evidence_id="technical_conformance",
        evidence_class="technical",
        status="passed",
        available_time="2026-07-01T09:00:00+09:00",
        artifact_sha256=None,
        limitation=None,
    )

    assert evidence.evidence_class is EvidenceClass.TECHNICAL
    assert evidence.status is EvidenceStatus.PASSED
    assert evidence.available_time == "2026-07-01T00:00:00Z"
    assert evidence.to_manifest()["artifact_sha256"] is None


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"evidence_id": 7}, "evidence_id must be a string"),
        ({"evidence_id": ""}, "evidence_id must not be empty"),
        ({"evidence_id": "single"}, "lower snake_case"),
        ({"evidence_class": object()}, "supported EvidenceClass"),
        ({"evidence_class": "unknown"}, "evidence_class must be one of"),
        ({"status": object()}, "supported EvidenceStatus"),
        ({"status": "unknown"}, "status must be one of"),
        ({"available_time": "not-a-time"}, "ISO-8601 timestamp"),
        ({"available_time": "2026-07-01T00:00:00"}, "include a timezone"),
        ({"artifact_sha256": "bad"}, "lowercase SHA-256"),
        ({"artifact_sha256": object()}, "artifact_sha256 must be a string"),
        ({"limitation": ""}, "limitation must not be empty"),
        ({"limitation": "x" * 4_097}, "at most 4096"),
    ],
)
def test_evidence_rejects_invalid_public_inputs(
    changes: dict[str, object],
    match: str,
) -> None:
    """Reject malformed evidence controls at the public constructor boundary."""
    kwargs: dict[str, object] = {
        "evidence_id": "technical_conformance",
        "evidence_class": EvidenceClass.TECHNICAL,
        "status": EvidenceStatus.PASSED,
        "available_time": "2026-07-01T00:00:00Z",
        "artifact_sha256": _SHA_D,
        "limitation": "bounded source-free evidence",
    }
    kwargs.update(changes)

    with pytest.raises(ValueError, match=match):
        ValidationEvidence(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("validation_profile_id", "single", "lower snake_case"),
        ("construct", "", "construct must not be empty"),
        ("construct", object(), "construct must be a string"),
        ("construct", "x" * 4_097, "at most 4096"),
        ("assessment_fingerprint", "bad", "lowercase SHA-256"),
        ("development_dataset_ids", "dataset", "must be a list or tuple"),
        (
            "development_dataset_ids",
            tuple(f"dataset-{index}" for index in range(65)),
            "between 0 and 64 values",
        ),
        ("development_dataset_ids", ("dup", "dup"), "must not contain duplicates"),
        ("development_dataset_ids", ("",), "must not be empty"),
        ("languages", "en", "languages must be a list or tuple"),
        (
            "languages",
            tuple(f"lang-{index}" for index in range(65)),
            "at most 64 values",
        ),
        ("languages", ("en", "en"), "languages must not contain duplicates"),
        ("schema_version", "2.0", "schema_version must be '1.0'"),
    ],
)
def test_profile_rejects_invalid_public_inputs(
    field_name: str,
    value: object,
    match: str,
) -> None:
    """Cover bounded profile admission and schema-version failure branches."""
    profile = _profile(_evidence("technical_conformance", EvidenceClass.TECHNICAL))

    with pytest.raises(ValueError, match=match):
        replace(profile, **{field_name: value})


def test_evidence_collection_rejects_wrong_shape_size_and_record_type() -> None:
    """Bound evidence containers before any package-record field access."""
    profile = _profile(_evidence("technical_conformance", EvidenceClass.TECHNICAL))

    with pytest.raises(ValueError, match="evidence must be a list or tuple"):
        replace(profile, evidence=object())

    many = tuple(
        _evidence(f"evidence_{index}", EvidenceClass.TECHNICAL)
        for index in range(65)
    )
    with pytest.raises(ValueError, match="evidence must contain at most 64 values"):
        replace(profile, evidence=many)

    with pytest.raises(ValueError, match=r"evidence\[0\] must be a ValidationEvidence"):
        replace(profile, evidence=(object(),))


def test_mutated_evidence_fields_fail_before_reuse() -> None:
    """Revalidate every stored evidence field before reconstructing trusted records."""
    profile = _profile(_evidence("technical_conformance", EvidenceClass.TECHNICAL))
    cases = [
        ("evidence_class", object(), "evidence_class must be an EvidenceClass"),
        ("status", object(), "status must be an EvidenceStatus"),
        ("available_time", object(), "available_time must be a string"),
        ("artifact_sha256", object(), "artifact_sha256 must be a string"),
        ("limitation", object(), "limitation must be a string"),
    ]

    for field_name, value, match in cases:
        evidence = _evidence("technical_conformance", EvidenceClass.TECHNICAL)
        object.__setattr__(evidence, field_name, value)
        with pytest.raises(ValueError, match=match):
            replace(profile, evidence=(evidence,))


def test_future_evidence_cannot_cross_the_analysis_cutoff() -> None:
    """Evidence unavailable at the preregistered cutoff fails closed."""
    future = _evidence(
        "future_outcome",
        EvidenceClass.DECISION_UTILITY,
        available_time="2026-08-02T00:00:00Z",
    )

    with pytest.raises(
        ValueError,
        match="available_time must not exceed analysis_cutoff",
    ):
        _profile(future)


def test_preregistration_must_precede_the_analysis_cutoff() -> None:
    """Reject a protocol timestamp after the declared analysis cutoff."""
    profile = _profile(_evidence("technical_conformance", EvidenceClass.TECHNICAL))

    with pytest.raises(
        ValueError,
        match="preregistered_at must not exceed analysis_cutoff",
    ):
        replace(profile, preregistered_at="2026-08-02T00:00:00Z")


def test_duplicate_evidence_ids_are_rejected() -> None:
    """One evidence identity cannot silently represent multiple evidence classes."""
    first = _evidence("shared_evidence", EvidenceClass.TECHNICAL)
    second = _evidence("shared_evidence", EvidenceClass.FAIRNESS)

    with pytest.raises(ValueError, match="evidence_id values must be unique"):
        _profile(first, second)


def test_mutated_evidence_id_fails_before_hash_or_sort_callbacks() -> None:
    """Revalidate package records before hashing or sorting their identifiers."""
    evidence = _evidence("technical_conformance", EvidenceClass.TECHNICAL)
    hostile = _HostileText("technical_conformance")
    object.__setattr__(evidence, "evidence_id", hostile)
    _HostileText.callbacks = 0

    with pytest.raises(
        ValueError,
        match=r"evidence\[0\]\.evidence_id must be a string",
    ):
        _profile(evidence)

    assert _HostileText.callbacks == 0


def test_provider_neutral_dataset_and_site_identifiers_are_preserved() -> None:
    """Preserve provider-neutral dataset and site identity syntax."""
    technical = _evidence("technical_conformance", EvidenceClass.TECHNICAL)
    profile = replace(
        _profile(technical),
        development_dataset_ids=("doi:10.1234/example.dataset",),
        external_validation_dataset_ids=("urn:dataset:external:2026-01",),
        sites=("site/eu-west/01",),
    )

    assert profile.development_dataset_ids == ("doi:10.1234/example.dataset",)
    assert profile.external_validation_dataset_ids == (
        "urn:dataset:external:2026-01",
    )
    assert profile.sites == ("site/eu-west/01",)


def test_dataset_cohorts_must_be_disjoint() -> None:
    """Prevent external evidence from reusing a development cohort identity."""
    with pytest.raises(
        ValueError,
        match="dataset id must not occur in both development_dataset_ids and "
        "external_validation_dataset_ids",
    ):
        replace(
            _profile(_evidence("technical_conformance", EvidenceClass.TECHNICAL)),
            external_validation_dataset_ids=("development_set_v1",),
        )


def test_status_vocabulary_preserves_failure_and_nonexecution_states() -> None:
    """The reusable contract must never collapse non-success states into pass."""
    assert {member.value for member in EvidenceStatus} == {
        "passed",
        "failed",
        "indeterminate",
        "not_executed",
        "not_applicable",
    }
    assert {member.value for member in EvidenceClass} == {
        "technical",
        "construct",
        "transportability",
        "fairness",
        "decision_utility",
    }


def test_text_subclasses_fail_before_text_callbacks() -> None:
    """Profile identity admission rejects hostile text without invoking callbacks."""
    _HostileText.callbacks = 0
    technical = _evidence("technical_conformance", EvidenceClass.TECHNICAL)

    with pytest.raises(ValueError, match="validation_profile_id must be a string"):
        replace(
            _profile(technical),
            validation_profile_id=_HostileText("profile_v1"),
        )

    assert _HostileText.callbacks == 0


def test_evidence_subclasses_fail_before_field_callbacks() -> None:
    """Reject evidence subclasses before reading caller-controlled fields."""

    class HostileEvidence(ValidationEvidence):
        """Subclass fixture that raises if validation reads a field."""

        def __getattribute__(self, name: str) -> object:
            if name == "evidence_id":
                raise AssertionError("evidence field callback executed")
            return super().__getattribute__(name)

    with pytest.raises(ValueError, match="exact package record"):
        HostileEvidence(
            evidence_id="technical_conformance",
            evidence_class=EvidenceClass.TECHNICAL,
            status=EvidenceStatus.PASSED,
            available_time="2026-07-01T00:00:00Z",
        )


def test_profile_subclasses_fail_before_field_callbacks() -> None:
    """Reject profile subclasses before reading caller-controlled fields."""
    base = _profile(_evidence("technical_conformance", EvidenceClass.TECHNICAL))

    class HostileProfile(ExternalValidationProfile):
        """Subclass fixture that raises if validation reads a field."""

        def __getattribute__(self, name: str) -> object:
            if name == "construct":
                raise AssertionError("profile field callback executed")
            return super().__getattribute__(name)

    from dataclasses import fields

    values = {
        field.name: getattr(base, field.name)
        for field in fields(ExternalValidationProfile)
    }
    with pytest.raises(ValueError, match="exact package record"):
        HostileProfile(**values)
