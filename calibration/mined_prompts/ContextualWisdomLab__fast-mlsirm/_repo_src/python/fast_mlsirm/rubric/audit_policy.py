"""Public fail-closed audit policy and verified pilot-admission entrypoints."""

from __future__ import annotations

from .audit import (
    CandidateAuditReport,
    PilotAdmissionError,
    audit_generated_item_candidate as _audit_generated_item_candidate,
    build_pilot_candidate_record as _build_pilot_candidate_record,
)
from .candidates import GeneratedItemCandidate
from .verified_pilot import PilotCandidateRecord
from .verified_pilot import _from_verified_core

AUDIT_POLICY_ID = "generated_item_audit"
AUDIT_POLICY_VERSION = "1.0.0"


def audit_generated_item_candidate(
    candidate: GeneratedItemCandidate,
) -> CandidateAuditReport:
    """Run the exact audit policy implemented by this package version."""
    return _audit_generated_item_candidate(
        candidate,
        audit_policy_id=AUDIT_POLICY_ID,
        audit_policy_version=AUDIT_POLICY_VERSION,
    )


def build_pilot_candidate_record(
    candidate: GeneratedItemCandidate,
    audit_report: CandidateAuditReport,
    *,
    pilot_study_id: str,
    query_testlet_id: str,
    generator_family_id: str,
    judge_policy_id: str,
    occasion_id: str,
) -> PilotCandidateRecord:
    """Admit a candidate only after replaying and matching the current policy."""
    if not isinstance(candidate, GeneratedItemCandidate):
        raise TypeError("candidate must be a GeneratedItemCandidate")
    if not isinstance(audit_report, CandidateAuditReport):
        raise TypeError("audit_report must be a CandidateAuditReport")

    candidate_fingerprint = candidate.candidate_fingerprint
    if audit_report.candidate_fingerprint != candidate_fingerprint:
        raise PilotAdmissionError(
            "candidate_report_mismatch",
            "$.candidate_fingerprint",
            "audit report does not bind the exact candidate",
        )
    if (
        audit_report.audit_policy_id != AUDIT_POLICY_ID
        or audit_report.audit_policy_version != AUDIT_POLICY_VERSION
    ):
        raise PilotAdmissionError(
            "unsupported_audit_policy",
            "$.audit_report.audit_policy_version",
            "audit report policy is not implemented by this package version",
        )

    expected_report = audit_generated_item_candidate(candidate)
    if audit_report.audit_report_fingerprint != expected_report.audit_report_fingerprint:
        raise PilotAdmissionError(
            "audit_report_unverified",
            "$.audit_report.audit_report_fingerprint",
            "audit report does not match a replay of the current policy",
        )

    core_record = _build_pilot_candidate_record(
        candidate,
        audit_report,
        pilot_study_id=pilot_study_id,
        query_testlet_id=query_testlet_id,
        generator_family_id=generator_family_id,
        judge_policy_id=judge_policy_id,
        occasion_id=occasion_id,
    )
    return _from_verified_core(core_record)
