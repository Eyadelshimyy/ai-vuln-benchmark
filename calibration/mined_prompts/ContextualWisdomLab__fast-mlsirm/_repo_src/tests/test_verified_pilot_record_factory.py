"""Factory-sealing contracts for public pilot-admission records."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric import (
    PilotCandidateRecord,
    audit_generated_item_candidate,
    build_pilot_candidate_record,
)
from fast_mlsirm.rubric.verified_pilot import _from_verified_core

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_candidate_audit.py"))
)
_candidate = _FIXTURES["_candidate"]
_pilot_kwargs = _FIXTURES["_pilot_kwargs"]
_PUBLIC_FIELDS = (
    "pilot_study_id",
    "query_testlet_id",
    "generator_family_id",
    "judge_policy_id",
    "occasion_id",
    "item_id",
    "candidate_fingerprint",
    "audit_report_fingerprint",
    "audit_policy_id",
    "audit_policy_version",
    "blueprint_id",
    "rubric_id",
    "rubric_version",
    "lifecycle_state",
    "schema_version",
)


def test_public_pilot_record_requires_replay_verified_factory():
    """Valid-looking public values cannot directly mint an admission record."""
    candidate = _candidate()
    report = audit_generated_item_candidate(candidate)
    verified = build_pilot_candidate_record(candidate, report, **_pilot_kwargs())
    public_values = {name: getattr(verified, name) for name in _PUBLIC_FIELDS}

    assert verified.pilot_record_fingerprint == verified.to_dict()[
        "pilot_record_fingerprint"
    ]
    with pytest.raises(ValueError, match="build_pilot_candidate_record"):
        PilotCandidateRecord(**public_values)


def test_verified_core_wrapper_rejects_wrong_runtime_type():
    """The private adapter fails closed before dereferencing caller objects."""
    with pytest.raises(TypeError, match="core PilotCandidateRecord"):
        _from_verified_core(object())
