"""HackerOne report drafts, validation, dry-run, and gated submission."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4

from app.adapters.hackerone.client import HackerOneApiClient
from app.adapters.hackerone.errors import (
    HackerOneError,
    HackerOneIdentityVerificationError,
    HackerOneReconciliationAmbiguousError,
    HackerOneSubmissionInProgressError,
    HackerOneSubmissionUnknownError,
)
from app.adapters.hackerone.evaluator import HackerOneScopeDecision, HackerOneScopeEvaluator
from app.adapters.hackerone.freshness import require_fresh_program
from app.adapters.hackerone.hashes import (
    evidence_hash,
    payload_hash,
    report_content_hash,
)
from app.adapters.hackerone.models import (
    HackerOneProgram,
    ScopeSnapshot,
    parse_hackerone_id,
)
from app.adapters.hackerone.remote_state import HackerOneRemoteState, map_remote_state
from app.adapters.hackerone.weakness import map_program_weakness
from app.domain.findings import FindingStatus, SecurityFinding
from app.security_testing.operator_auth import OperatorSession
from app.security_testing.secrets import configured_secret_values, detect_secrets, redact_text

_NOT_DRAFTABLE = frozenset(
    {
        FindingStatus.POTENTIAL,
        FindingStatus.CORROBORATED,
        FindingStatus.REPRODUCED,
        FindingStatus.REJECTED,
    }
)
_SEVERITIES = ("critical", "high", "medium", "low", "informational")
_H1_SEVERITY = {
    "critical": "critical",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "informational": "none",
    "none": "none",
}
_APPROVAL_TTL = timedelta(hours=24)


class ReportHumanReviewState(StrEnum):
    UNREVIEWED = "unreviewed"
    READY_FOR_REVIEW = "ready_for_review"
    HUMAN_APPROVED = "human_approved"
    REJECTED = "rejected"


class ReportSubmissionState(StrEnum):
    LOCAL_DRAFT = "local_draft"
    READY_FOR_REVIEW = "ready_for_review"
    HUMAN_APPROVED = "human_approved"
    SUBMISSION_ATTEMPTED = "submission_attempted"
    SUBMISSION_IN_PROGRESS = "submission_in_progress"
    SUBMITTED = "submitted"
    SUBMISSION_FAILED = "submission_failed"
    SUBMISSION_OUTCOME_UNKNOWN = "submission_outcome_unknown"
    IDENTITY_VERIFICATION_REQUIRED = "identity_verification_required"


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str
    blocking: bool = True


@dataclass(frozen=True)
class ReportValidationReport:
    ok: bool
    issues: tuple[ValidationIssue, ...]

    def blocking(self) -> tuple[ValidationIssue, ...]:
        return tuple(item for item in self.issues if item.blocking)


@dataclass(frozen=True)
class ApprovalRecord:
    approved_by: str
    approved_at: datetime
    expires_at: datetime
    report_content_hash: str
    evidence_hash: str
    scope_snapshot_hash: str
    payload_hash: str
    source: str = "operator_token"

    def expired(self, now: datetime | None = None) -> bool:
        clock = now or datetime.now(UTC)
        expires = self.expires_at
        if clock.tzinfo is None:
            clock = clock.replace(tzinfo=UTC)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        return clock >= expires


@dataclass
class HackerOneReportDraft:
    program_handle: str
    title: str
    vulnerability_information: str
    impact: str = ""
    severity: str | None = None
    weakness_id: int | None = None
    weakness_candidates: tuple[int, ...] = ()
    structured_scope_id: int | None = None
    evidence_references: tuple[str, ...] = ()
    finding_id: str | None = None
    project_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    human_review_state: ReportHumanReviewState = ReportHumanReviewState.UNREVIEWED
    submission_state: ReportSubmissionState = ReportSubmissionState.LOCAL_DRAFT
    remote_state: HackerOneRemoteState = HackerOneRemoteState.NOT_FETCHED
    hackerone_report_id: str | None = None
    finding_verification: str | None = None
    error: str | None = None
    last_payload: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    operator: str | None = None
    reproduction: str | None = None
    target: str | None = None
    eligible_for_submission: bool = False
    eligible_for_bounty: bool = False
    report_content_hash: str = ""
    evidence_hash: str = ""
    scope_snapshot_hash: str = ""
    payload_hash: str = ""
    approval: ApprovalRecord | None = None
    scope_snapshot: ScopeSnapshot | None = None
    weakness_reason: str = ""
    remote_state_raw: str = ""
    submission_result: dict[str, Any] | None = None
    claim_token: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "program_handle": self.program_handle,
            "project_id": self.project_id,
            "title": self.title,
            "vulnerability_information": self.vulnerability_information,
            "impact": self.impact,
            "severity": self.severity,
            "weakness_id": self.weakness_id,
            "weakness_candidates": list(self.weakness_candidates),
            "structured_scope_id": self.structured_scope_id,
            "finding_id": self.finding_id,
            "human_review_state": self.human_review_state.value,
            "submission_state": self.submission_state.value,
            "remote_state": self.remote_state.value,
            "finding_verification": self.finding_verification,
            "hackerone_report_id": self.hackerone_report_id,
            "eligible_for_submission": self.eligible_for_submission,
            "eligible_for_bounty": self.eligible_for_bounty,
            "target": self.target,
            "reproduction": self.reproduction,
            "evidence_references": list(self.evidence_references),
            "report_content_hash": self.report_content_hash,
            "evidence_hash": self.evidence_hash,
            "scope_snapshot_hash": self.scope_snapshot_hash,
            "payload_hash": self.payload_hash,
            "approval_timestamp": (
                self.approval.approved_at.isoformat() if self.approval else None
            ),
            "approved_by": self.approval.approved_by if self.approval else None,
            "scope_snapshot": self.scope_snapshot.as_dict() if self.scope_snapshot else None,
            "error": self.error,
            "remote_state_raw": self.remote_state_raw,
            "submission_result": self.submission_result,
        }


@dataclass
class SubmissionRecord:
    finding_id: str
    program: str
    submission_state: ReportSubmissionState
    hackerone_report_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str] = field(default_factory=dict)
    remote_state: HackerOneRemoteState = HackerOneRemoteState.NOT_FETCHED
    error: str | None = None
    result: dict[str, Any] | None = None
    remote_state_raw: str = ""


class ReportQualityValidator:
    def validate(
        self,
        draft: HackerOneReportDraft,
        *,
        program: HackerOneProgram,
        scope: HackerOneScopeDecision | None,
        finding: SecurityFinding | None = None,
    ) -> ReportValidationReport:
        issues: list[ValidationIssue] = []
        if not draft.title.strip():
            issues.append(ValidationIssue("title", "Title is required"))
        if not draft.vulnerability_information.strip():
            issues.append(
                ValidationIssue(
                    "vulnerability_information", "Vulnerability information is required"
                )
            )
        elif "## Steps to reproduce" not in draft.vulnerability_information:
            issues.append(
                ValidationIssue(
                    "vulnerability_information",
                    "Vulnerability information must include steps to reproduce",
                )
            )
        if not draft.impact.strip():
            issues.append(ValidationIssue("impact", "Impact is required"))
        if not draft.target:
            issues.append(ValidationIssue("target", "Target is required"))
        if scope is None or not scope.in_scope:
            issues.append(
                ValidationIssue("target", "Target is not in the current HackerOne structured scope")
            )
        if scope is not None and not scope.eligible_for_submission:
            issues.append(ValidationIssue("target", "Target is not eligible for submission"))
        if draft.structured_scope_id is None:
            issues.append(ValidationIssue("structured_scope_id", "structured_scope_id is required"))
        elif scope is not None:
            current = parse_hackerone_id(scope.structured_scope_id)
            if current != draft.structured_scope_id:
                issues.append(
                    ValidationIssue(
                        "structured_scope_id",
                        "structured_scope_id does not match the current scope match",
                    )
                )
        if program.requires_severity:
            if not draft.severity:
                issues.append(ValidationIssue("severity", "Severity is required for this program"))
            elif draft.severity not in _SEVERITIES:
                issues.append(
                    ValidationIssue("severity", "Severity is not a recognized HackerOne rating")
                )
        if draft.weakness_id is None:
            issues.append(
                ValidationIssue(
                    "weakness_id",
                    "Weakness must be a synchronized numeric HackerOne id; "
                    "a human must select one when the mapping is missing or ambiguous",
                    blocking=True,
                )
            )
        else:
            known = {item.id for item in program.weaknesses}
            if known and draft.weakness_id not in known:
                issues.append(
                    ValidationIssue(
                        "weakness_id",
                        "weakness_id is not in the current program weakness list",
                    )
                )
        if not draft.evidence_references:
            issues.append(ValidationIssue("evidence", "Evidence must be linked"))
        if finding is not None:
            if finding.status not in {FindingStatus.VERIFIED, FindingStatus.HUMAN_ACCEPTED}:
                issues.append(
                    ValidationIssue(
                        "finding",
                        "Only verified findings can become submission-ready reports",
                    )
                )
            if not finding.evidence.verifying_items():
                issues.append(ValidationIssue("evidence", "Finding lacks verifying evidence"))
            if not (finding.reproduction or draft.reproduction):
                issues.append(
                    ValidationIssue(
                        "reproduction", "Reproduction information is required", blocking=True
                    )
                )
        leaked = detect_secrets(
            draft.title,
            draft.vulnerability_information,
            draft.impact,
            draft.reproduction or "",
            " ".join(draft.evidence_references),
            configured=configured_secret_values(),
        )
        if leaked:
            issues.append(
                ValidationIssue("secrets", f"Possible secret leakage: {', '.join(leaked)}")
            )
        if "unsupported claim" in draft.vulnerability_information.lower():
            issues.append(ValidationIssue("claims", "Unsupported claims must be removed"))
        return ReportValidationReport(
            ok=not any(item.blocking for item in issues), issues=tuple(issues)
        )


class HackerOneReportWorkflow:
    def __init__(
        self,
        client: HackerOneApiClient,
        *,
        evaluator: HackerOneScopeEvaluator | None = None,
    ) -> None:
        self.client = client
        self.evaluator = evaluator or HackerOneScopeEvaluator()
        self.validator = ReportQualityValidator()
        self.drafts: dict[str, HackerOneReportDraft] = {}
        self.submissions: dict[tuple[str, str], SubmissionRecord] = {}
        self._claim_lock = threading.Lock()

    def draft_from_finding(
        self,
        finding: SecurityFinding,
        program: HackerOneProgram,
        *,
        severity: str | None = None,
        weakness_id: int | str | None = None,
        operator: OperatorSession | str | None = None,
        project_id: str | None = None,
    ) -> HackerOneReportDraft:
        _assert_verified_finding(finding)
        target = finding.target or finding.endpoint or finding.asset or ""
        scope = (
            self.evaluator.evaluate(
                program, target, vulnerability_class=finding.vulnerability_class
            )
            if target
            else HackerOneScopeDecision(False, "Finding has no target")
        )
        mapped = map_program_weakness(
            finding.vulnerability_class, program, requested_id=weakness_id
        )
        if weakness_id is not None and mapped.selected_id is None:
            raise HackerOneError(mapped.reason, code="weakness_invalid")
        selected = mapped.selected_id
        if mapped.requires_human_selection and selected is None:
            selected = None
        structured_id = parse_hackerone_id(scope.structured_scope_id)
        vuln_info = compose_vulnerability_information(finding, target=target)
        snapshot = scope.snapshot(program)
        evidence_refs = tuple(str(item.id) for item in finding.evidence.verifying_items())
        draft = HackerOneReportDraft(
            program_handle=program.handle,
            title=finding.report_title or finding.title,
            vulnerability_information=vuln_info,
            impact=finding.impact or "",
            severity=_normalize_severity(severity),
            weakness_id=selected,
            weakness_candidates=tuple(item.id for item in mapped.matched),
            structured_scope_id=structured_id,
            evidence_references=evidence_refs,
            finding_id=str(finding.id),
            project_id=project_id,
            finding_verification=finding.status.value,
            reproduction=finding.reproduction,
            target=target,
            eligible_for_submission=scope.eligible_for_submission,
            eligible_for_bounty=scope.eligible_for_bounty,
            operator=_operator_identity(operator),
            scope_snapshot=snapshot,
            weakness_reason=mapped.reason,
        )
        self._refresh_hashes(draft, program=program, scope=scope, finding=finding)
        self.drafts[draft.id] = draft
        return draft

    def mark_ready(
        self,
        draft_id: str,
        program: HackerOneProgram,
        *,
        finding: SecurityFinding | None = None,
    ) -> HackerOneReportDraft:
        draft = self._get(draft_id)
        self._revalidate(draft, program, finding=finding, stage="ready_for_review")
        updated = replace(
            draft,
            human_review_state=ReportHumanReviewState.READY_FOR_REVIEW,
            submission_state=ReportSubmissionState.READY_FOR_REVIEW,
        )
        self.drafts[draft_id] = updated
        return updated

    def human_approve(
        self,
        draft_id: str,
        program: HackerOneProgram,
        *,
        session: OperatorSession,
        finding: SecurityFinding | None = None,
    ) -> HackerOneReportDraft:
        session.assert_human()
        draft = self._get(draft_id)
        if draft.submission_state not in {
            ReportSubmissionState.READY_FOR_REVIEW,
            ReportSubmissionState.LOCAL_DRAFT,
            ReportSubmissionState.SUBMISSION_FAILED,
            ReportSubmissionState.IDENTITY_VERIFICATION_REQUIRED,
        }:
            raise HackerOneError("Draft is not awaiting human approval", code="invalid_state")
        scope = self._revalidate(draft, program, finding=finding, stage="human_approved")
        self._refresh_hashes(draft, program=program, scope=scope, finding=finding)
        approval = ApprovalRecord(
            approved_by=session.identity,
            approved_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + _APPROVAL_TTL,
            report_content_hash=draft.report_content_hash,
            evidence_hash=draft.evidence_hash,
            scope_snapshot_hash=draft.scope_snapshot_hash,
            payload_hash=payload_hash(self.build_payload(draft)),
            source=session.source,
        )
        updated = replace(
            draft,
            human_review_state=ReportHumanReviewState.HUMAN_APPROVED,
            submission_state=ReportSubmissionState.HUMAN_APPROVED,
            operator=session.identity,
            approval=approval,
        )
        self.drafts[draft_id] = updated
        return updated

    def apply_edits(
        self,
        draft_id: str,
        program: HackerOneProgram,
        *,
        title: str | None = None,
        vulnerability_information: str | None = None,
        impact: str | None = None,
        severity: str | None = None,
        weakness_id: int | str | None = None,
        finding: SecurityFinding | None = None,
    ) -> HackerOneReportDraft:
        draft = self._get(draft_id)
        updates: dict[str, Any] = {}
        if title is not None:
            updates["title"] = title
        if vulnerability_information is not None:
            updates["vulnerability_information"] = vulnerability_information
        if impact is not None:
            updates["impact"] = impact
        if severity is not None:
            updates["severity"] = _normalize_severity(severity)
        if weakness_id is not None:
            mapped = map_program_weakness(
                finding.vulnerability_class if finding else None,
                program,
                requested_id=weakness_id,
            )
            if mapped.selected_id is None:
                raise HackerOneError(mapped.reason, code="weakness_invalid")
            updates["weakness_id"] = mapped.selected_id
        updated = replace(draft, **updates)
        scope = self.evaluator.evaluate(
            program,
            updated.target or "",
            vulnerability_class=finding.vulnerability_class if finding else None,
        )
        self._refresh_hashes(updated, program=program, scope=scope, finding=finding)
        if draft.approval is not None and not self._approval_matches(updated, draft.approval):
            updated = replace(
                updated,
                human_review_state=ReportHumanReviewState.READY_FOR_REVIEW,
                submission_state=ReportSubmissionState.READY_FOR_REVIEW,
                approval=None,
                error="Approval invalidated because the report, evidence, or scope changed",
            )
        self.drafts[draft_id] = updated
        return updated

    def build_payload(self, draft: HackerOneReportDraft) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            "team_handle": draft.program_handle,
            "title": draft.title,
            "vulnerability_information": draft.vulnerability_information,
            "impact": draft.impact,
        }
        if draft.severity:
            attributes["severity_rating"] = _H1_SEVERITY.get(draft.severity, draft.severity)
        if draft.weakness_id is not None:
            attributes["weakness_id"] = int(draft.weakness_id)
        if draft.structured_scope_id is not None:
            attributes["structured_scope_id"] = int(draft.structured_scope_id)
        return {"data": {"type": "report", "attributes": attributes}}

    def redacted_payload(self, draft: HackerOneReportDraft) -> dict[str, Any]:
        return _redact_payload(self.build_payload(draft))

    def dry_run(
        self,
        draft_id: str,
        program: HackerOneProgram,
        *,
        finding: SecurityFinding | None = None,
    ) -> dict[str, Any]:
        draft = self._get(draft_id)
        scope = self.evaluator.evaluate(
            program,
            draft.target or "",
            vulnerability_class=finding.vulnerability_class if finding else None,
        )
        report = self.validator.validate(draft, program=program, scope=scope, finding=finding)
        payload = self.redacted_payload(draft)
        before = list(self.client.calls)
        return {
            "would_submit": False,
            "payload": payload,
            "validation": {
                "ok": report.ok,
                "issues": [issue.__dict__ for issue in report.issues],
            },
            "scope": {
                "in_scope": scope.in_scope,
                "allowed": scope.allowed,
                "reason": scope.reason,
                "structured_scope_id": scope.structured_scope_id,
                "eligible_for_submission": scope.eligible_for_submission,
                "eligible_for_bounty": scope.eligible_for_bounty,
                "asset_identifier": scope.asset_identifier,
            },
            "hashes": {
                "report_content_hash": draft.report_content_hash,
                "evidence_hash": draft.evidence_hash,
                "scope_snapshot_hash": draft.scope_snapshot_hash,
                "payload_hash": payload_hash(self.build_payload(draft)),
            },
            "api_calls_during_dry_run": list(self.client.calls[len(before) :]),
        }

    def get_draft(self, draft_id: str) -> HackerOneReportDraft:
        return self._get(draft_id)

    def claim_submission(self, draft_id: str, program: HackerOneProgram) -> HackerOneReportDraft:
        """Atomically claim a local submission. Second caller gets in-progress."""
        with self._claim_lock:
            draft = self._get(draft_id)
            if draft.submission_state is ReportSubmissionState.SUBMISSION_ATTEMPTED:
                raise HackerOneSubmissionInProgressError()
            if draft.submission_state is ReportSubmissionState.SUBMISSION_IN_PROGRESS:
                raise HackerOneSubmissionInProgressError()
            if draft.submission_state is ReportSubmissionState.SUBMISSION_OUTCOME_UNKNOWN:
                raise HackerOneSubmissionUnknownError(
                    "A previous submission attempt has an unknown outcome. "
                    "Reconcile the remote report before retrying."
                )
            if (
                draft.hackerone_report_id
                or draft.submission_state is ReportSubmissionState.SUBMITTED
            ):
                raise HackerOneError("Draft already has a HackerOne report id", code="duplicate")
            if draft.submission_state is not ReportSubmissionState.HUMAN_APPROVED:
                raise HackerOneError(
                    "Real submission requires HUMAN_APPROVED. Use dry-run until a human approves.",
                    code="not_approved",
                )
            if draft.finding_id:
                existing = self.submissions.get((draft.finding_id, program.handle))
                if existing is not None:
                    if existing.submission_state in {
                        ReportSubmissionState.SUBMISSION_ATTEMPTED,
                        ReportSubmissionState.SUBMISSION_IN_PROGRESS,
                    }:
                        raise HackerOneSubmissionInProgressError()
                    if (
                        existing.submission_state
                        is ReportSubmissionState.SUBMISSION_OUTCOME_UNKNOWN
                    ):
                        raise HackerOneSubmissionUnknownError(
                            "A previous submission attempt has an unknown outcome. Reconcile first."
                        )
                    if (
                        existing.hackerone_report_id
                        or existing.submission_state is ReportSubmissionState.SUBMITTED
                    ):
                        raise HackerOneError(
                            "Finding already submitted as HackerOne report "
                            f"{existing.hackerone_report_id}",
                            code="duplicate",
                        )
                self.submissions[(draft.finding_id, program.handle)] = SubmissionRecord(
                    finding_id=draft.finding_id,
                    program=program.handle,
                    submission_state=ReportSubmissionState.SUBMISSION_ATTEMPTED,
                    metadata={"draft_id": draft_id, "claim_token": uuid4().hex},
                )
            claimed = replace(
                draft,
                submission_state=ReportSubmissionState.SUBMISSION_ATTEMPTED,
                claim_token=uuid4().hex,
                last_payload=self.redacted_payload(draft),
            )
            self.drafts[draft_id] = claimed
            return claimed

    def submit(
        self,
        draft_id: str,
        program: HackerOneProgram,
        *,
        finding: SecurityFinding | None = None,
        session: OperatorSession,
        already_claimed: bool = False,
    ) -> HackerOneReportDraft:
        session.assert_human()
        require_fresh_program(program)
        draft = self._get(draft_id)
        if draft.approval is None:
            raise HackerOneError("Human approval record is missing", code="not_approved")
        if draft.approval.expired():
            self._invalidate_approval(draft, "Approval expired")
            raise HackerOneError("Human approval has expired", code="approval_expired")
        scope = self._revalidate(draft, program, finding=finding, stage="submit")
        self._refresh_hashes(draft, program=program, scope=scope, finding=finding)
        if not self._approval_matches(draft, draft.approval):
            self._invalidate_approval(draft, "Approved hashes no longer match the current draft")
            raise HackerOneError(
                "Human approval is bound to a previous report/scope/evidence version",
                code="approval_stale",
            )
        current_payload = self.build_payload(draft)
        if payload_hash(current_payload) != draft.approval.payload_hash:
            self._invalidate_approval(draft, "Approved payload hash no longer matches")
            raise HackerOneError(
                "Approved payload does not match the current report payload",
                code="approval_stale",
            )
        attempted = draft if already_claimed else self.claim_submission(draft_id, program)
        try:
            body = self.client.post("hackers/reports", json_body=current_payload)
        except HackerOneIdentityVerificationError as exc:
            failed = replace(
                attempted,
                submission_state=ReportSubmissionState.IDENTITY_VERIFICATION_REQUIRED,
                error=str(exc),
                submission_result=_submission_result(
                    self.client, error_code=exc.code, report_id=None
                ),
            )
            self._store_draft(failed, program)
            return failed
        except HackerOneError as exc:
            if exc.code in {"timeout", "network"}:
                unknown = replace(
                    attempted,
                    submission_state=ReportSubmissionState.SUBMISSION_OUTCOME_UNKNOWN,
                    error=str(exc),
                    submission_result=_submission_result(
                        self.client, error_code=exc.code, report_id=None
                    ),
                )
                self._store_draft(unknown, program)
                return unknown
            failed = replace(
                attempted,
                submission_state=ReportSubmissionState.SUBMISSION_FAILED,
                error=str(exc),
                submission_result=_submission_result(
                    self.client, error_code=exc.code, report_id=None
                ),
            )
            self._store_draft(failed, program)
            return failed
        report_id = str((body.get("data") or {}).get("id") or "")
        mapped, raw = map_remote_state(
            ((body.get("data") or {}).get("attributes") or {}).get("state")
        )
        submitted = replace(
            attempted,
            submission_state=ReportSubmissionState.SUBMITTED,
            remote_state=mapped
            if mapped is not HackerOneRemoteState.NOT_FETCHED
            else HackerOneRemoteState.NEW,
            remote_state_raw=raw,
            hackerone_report_id=report_id,
            error=None,
            submission_result=_submission_result(self.client, error_code=None, report_id=report_id),
        )
        self._store_draft(submitted, program)
        return submitted

    def fetch_remote_report(self, report_id: str) -> dict[str, Any]:
        return self.client.get(f"hackers/reports/{report_id}")

    def reconcile(
        self,
        draft_id: str,
        *,
        remote: dict[str, Any] | None = None,
        human_confirmed_id: str | None = None,
        program: HackerOneProgram | None = None,
    ) -> HackerOneReportDraft:
        draft = self._get(draft_id)
        if human_confirmed_id:
            body = remote or self.fetch_remote_report(human_confirmed_id)
            return self._apply_remote(draft, body, require_id=human_confirmed_id)
        if draft.hackerone_report_id:
            body = remote or self.fetch_remote_report(draft.hackerone_report_id)
            return self._apply_remote(draft, body, require_id=draft.hackerone_report_id)
        if draft.submission_state is not ReportSubmissionState.SUBMISSION_OUTCOME_UNKNOWN:
            raise HackerOneError("No HackerOne report id to reconcile", code="not_found")
        if remote is not None:
            return self._apply_remote(draft, remote)
        matches = self._search_remote_matches(draft, program)
        if len(matches) == 1:
            return self._apply_remote(draft, matches[0])
        if len(matches) > 1:
            raise HackerOneReconciliationAmbiguousError(
                "Multiple remote HackerOne reports match; a human must confirm the report id"
            )
        return draft

    def _search_remote_matches(
        self, draft: HackerOneReportDraft, program: HackerOneProgram | None
    ) -> list[dict[str, Any]]:
        rows = self.client.list_hacker_reports()
        window_start = draft.created_at.timestamp() - 3600
        window_end = datetime.now(UTC).timestamp() + 3600
        matches: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_attrs = row.get("attributes")
            attrs: dict[str, Any] = raw_attrs if isinstance(raw_attrs, dict) else {}
            team = str(attrs.get("team_handle") or attrs.get("handle") or "")
            title = str(attrs.get("title") or "")
            submitted_at = str(attrs.get("submitted_at") or attrs.get("created_at") or "")
            if program and team and team != program.handle and team != draft.program_handle:
                continue
            if title != draft.title:
                continue
            # Title is necessary but never sufficient. Require time window and/or payload hash.
            time_ok = False
            if submitted_at:
                try:
                    parsed = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
                    time_ok = window_start <= parsed.timestamp() <= window_end
                except ValueError:
                    time_ok = False
            payload_ok = False
            remote_hash = str(attrs.get("vulnerability_information") or "")
            if draft.payload_hash and draft.payload_hash in str(attrs):
                payload_ok = True
            if draft.vulnerability_information and remote_hash == draft.vulnerability_information:
                payload_ok = True
            if time_ok and payload_ok:
                matches.append(row)
        return matches

    def _apply_remote(
        self,
        draft: HackerOneReportDraft,
        body: dict[str, Any],
        *,
        require_id: str | None = None,
    ) -> HackerOneReportDraft:
        data = body.get("data") if isinstance(body.get("data"), dict) else body
        remote_id = str((data or {}).get("id") or draft.hackerone_report_id or "")
        if require_id and remote_id and remote_id != require_id:
            raise HackerOneError("Remote report id does not match", code="mismatch")
        attrs = (data or {}).get("attributes") if isinstance(data, dict) else {}
        mapped, raw = map_remote_state(
            (attrs or {}).get("state") if isinstance(attrs, dict) else ""
        )
        updated = replace(
            draft,
            hackerone_report_id=remote_id or draft.hackerone_report_id,
            remote_state=mapped,
            remote_state_raw=raw,
            submission_state=(
                ReportSubmissionState.SUBMITTED if remote_id else draft.submission_state
            ),
            error=None,
            submission_result=_submission_result(
                self.client, error_code=None, report_id=remote_id or None
            ),
        )
        self.drafts[draft.id] = updated
        if updated.finding_id:
            key = (updated.finding_id, updated.program_handle)
            existing = self.submissions.get(key)
            self.submissions[key] = SubmissionRecord(
                finding_id=updated.finding_id,
                program=updated.program_handle,
                hackerone_report_id=updated.hackerone_report_id,
                submission_state=updated.submission_state,
                remote_state=updated.remote_state,
                remote_state_raw=raw,
                metadata={"draft_id": draft.id},
                error=existing.error if existing else None,
                result=updated.submission_result,
            )
        return updated

    def _store_draft(self, draft: HackerOneReportDraft, program: HackerOneProgram) -> None:
        self.drafts[draft.id] = draft
        if not draft.finding_id:
            return
        self.submissions[(draft.finding_id, program.handle)] = SubmissionRecord(
            finding_id=draft.finding_id,
            program=program.handle,
            hackerone_report_id=draft.hackerone_report_id,
            submission_state=draft.submission_state,
            remote_state=draft.remote_state,
            remote_state_raw=draft.remote_state_raw,
            metadata={"draft_id": draft.id, "claim_token": draft.claim_token or ""},
            error=draft.error,
            result=draft.submission_result,
        )

    def _revalidate(
        self,
        draft: HackerOneReportDraft,
        program: HackerOneProgram,
        *,
        finding: SecurityFinding | None,
        stage: str,
    ) -> HackerOneScopeDecision:
        if finding is not None:
            _assert_verified_finding(finding)
            if draft.finding_id and str(finding.id) != draft.finding_id:
                raise HackerOneError(
                    "Finding does not belong to this report draft", code="finding_mismatch"
                )
            draft.finding_verification = finding.status.value
            draft.evidence_references = tuple(
                str(item.id) for item in finding.evidence.verifying_items()
            )
            if finding.reproduction:
                draft.reproduction = finding.reproduction
        scope = self.evaluator.evaluate(
            program,
            draft.target or "",
            vulnerability_class=finding.vulnerability_class if finding else None,
        )
        current_snapshot = scope.snapshot(program)
        if draft.scope_snapshot is not None and stage == "submit":
            if current_snapshot.snapshot_hash != draft.scope_snapshot.snapshot_hash:
                self._invalidate_approval(draft, "Structured scope snapshot changed")
                raise HackerOneError(
                    "Current structured scope differs from the approved scope snapshot",
                    code="scope_changed",
                )
        report = self.validator.validate(draft, program=program, scope=scope, finding=finding)
        if not report.ok:
            raise HackerOneError(
                f"Draft failed {stage} validation: "
                + "; ".join(item.message for item in report.blocking()),
                code="validation_failed",
            )
        return scope

    def _refresh_hashes(
        self,
        draft: HackerOneReportDraft,
        *,
        program: HackerOneProgram,
        scope: HackerOneScopeDecision,
        finding: SecurityFinding | None = None,
    ) -> None:
        snapshot = scope.snapshot(program)
        draft.scope_snapshot = snapshot
        draft.report_content_hash = report_content_hash(
            title=draft.title,
            vulnerability_information=draft.vulnerability_information,
            impact=draft.impact,
            severity=draft.severity,
            weakness_id=draft.weakness_id,
            structured_scope_id=draft.structured_scope_id,
            target=draft.target,
            program_handle=draft.program_handle,
            finding_verification=draft.finding_verification,
            reproduction=draft.reproduction,
        )
        extra: dict[str, Any] = {
            "summaries": list(draft.evidence_references),
            "reproduction": draft.reproduction or "",
            "finding_verification": draft.finding_verification or "",
            "finding_id": draft.finding_id or "",
        }
        if finding is not None:
            extra["finding_status"] = finding.status.value
            extra["finding_evidence"] = [
                {
                    "id": str(item.id),
                    "kind": item.kind.value,
                    "summary": item.summary,
                    "provenance": getattr(item.provenance, "value", str(item.provenance)),
                }
                for item in finding.evidence.items
            ]
            extra["finding_reproduction"] = finding.reproduction or ""
            extra["vulnerability_class"] = finding.vulnerability_class or ""
        draft.evidence_hash = evidence_hash(draft.evidence_references, extra)
        draft.scope_snapshot_hash = snapshot.snapshot_hash
        draft.payload_hash = payload_hash(self.build_payload(draft))
        draft.eligible_for_submission = scope.eligible_for_submission
        draft.eligible_for_bounty = scope.eligible_for_bounty

    def _approval_matches(self, draft: HackerOneReportDraft, approval: ApprovalRecord) -> bool:
        return (
            approval.report_content_hash == draft.report_content_hash
            and approval.evidence_hash == draft.evidence_hash
            and approval.scope_snapshot_hash == draft.scope_snapshot_hash
            and not approval.expired()
        )

    def _invalidate_approval(
        self, draft: HackerOneReportDraft, reason: str
    ) -> HackerOneReportDraft:
        updated = replace(
            draft,
            human_review_state=ReportHumanReviewState.READY_FOR_REVIEW,
            submission_state=ReportSubmissionState.READY_FOR_REVIEW,
            approval=None,
            error=reason,
        )
        self.drafts[draft.id] = updated
        return updated

    def _get(self, draft_id: str) -> HackerOneReportDraft:
        draft = self.drafts.get(draft_id)
        if draft is None:
            raise HackerOneError("Unknown report draft", code="not_found")
        return draft


def compose_vulnerability_information(
    finding: SecurityFinding,
    *,
    target: str = "",
) -> str:
    summary = (
        finding.report_description
        or finding.description
        or finding.observed_behavior
        or finding.title
    )
    asset = target or finding.target or finding.asset or finding.endpoint or "unknown"
    steps = finding.reproduction or "Not provided"
    observed = finding.observed_behavior or finding.description or "See reproduction"
    expected = finding.expected_behavior or "The operation should be authorized only for the owner"
    evidence_lines = []
    for item in finding.evidence.verifying_items():
        evidence_lines.append(f"- {item.summary} ({item.kind.value}, id={item.id})")
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "- none"
    return (
        f"## Summary\n{summary}\n\n"
        f"## Affected asset\n{asset}\n\n"
        f"## Steps to reproduce\n{steps}\n\n"
        f"## Observed result\n{observed}\n\n"
        f"## Expected result\n{expected}\n\n"
        f"## Supporting evidence/references\n{evidence_block}"
    )


def _assert_verified_finding(finding: SecurityFinding) -> None:
    if finding.status in _NOT_DRAFTABLE or finding.status not in {
        FindingStatus.VERIFIED,
        FindingStatus.HUMAN_ACCEPTED,
    }:
        raise HackerOneError(
            f"A finding in status {finding.status.value!r} cannot become a "
            "submission-ready HackerOne draft. Only independently verified "
            "persisted findings are accepted.",
            code="finding_not_verified",
        )
    if not finding.evidence or not finding.evidence.verifying_items():
        raise HackerOneError("Verified evidence is required", code="evidence_required")


def _normalize_severity(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    lowered = value.strip().lower()
    if lowered not in _SEVERITIES and lowered != "none":
        raise HackerOneError(
            "Unknown severity; BugForge will not invent one", code="severity_invalid"
        )
    if lowered == "none":
        return "informational"
    return lowered


def _operator_identity(operator: OperatorSession | str | None) -> str | None:
    if isinstance(operator, OperatorSession):
        return operator.identity
    if isinstance(operator, str) and operator.strip():
        return operator.strip()
    return None


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    import json

    text = redact_text(json.dumps(payload, default=str))
    parsed: object = json.loads(text)
    if isinstance(parsed, dict):
        return parsed
    return payload


def _submission_result(
    client: HackerOneApiClient,
    *,
    error_code: str | None,
    report_id: str | None,
) -> dict[str, Any]:
    return {
        "http_status": client.last_status,
        "remote_report_id": report_id,
        "error_code": error_code,
        "timestamp": datetime.now(UTC).isoformat(),
        "response": client.last_sanitized_body,
    }
