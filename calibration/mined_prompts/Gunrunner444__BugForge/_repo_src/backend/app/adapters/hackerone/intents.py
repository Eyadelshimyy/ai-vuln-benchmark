"""Local and remote HackerOne report intents plus attachment review.

A local BugForge intent is not a HackerOne report intent. Remote create /
patch / submit happen only after human approval, through the API client.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.adapters.hackerone.errors import HackerOneError
from app.adapters.hackerone.hashes import sha256_bytes
from app.adapters.hackerone.models import HackerOneProgram, attributes, node_id
from app.adapters.hackerone.remote_state import HackerOneRemoteState, map_remote_state
from app.adapters.hackerone.reports import HackerOneReportDraft, HackerOneReportWorkflow
from app.core.config import get_settings
from app.domain.findings import SecurityFinding
from app.security_testing.operator_auth import OperatorSession
from app.security_testing.secrets import configured_secret_values, detect_secrets

_TEXT_TYPES = {
    "text/plain",
    "text/html",
    "text/csv",
    "text/xml",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-www-form-urlencoded",
}
_MAX_NAME = 180
_GENERIC_SECRET_MARKERS = (
    b"-----BEGIN ",
    b"AKIA",
    b"ghp_",
    b"github_pat_",
    b"xoxb-",
    b"xoxp-",
    b"hackerone_api_token",
    b"BUGFORGE_OPERATOR_TOKEN",
)


class IntentLocalState(StrEnum):
    LOCAL_DRAFT = "local_draft"
    READY_FOR_REVIEW = "ready_for_review"
    HUMAN_APPROVED = "human_approved"
    REMOTE_INTENT_CREATED = "remote_intent_created"
    REMOTE_READY_TO_SUBMIT = "remote_ready_to_submit"
    SUBMITTED = "submitted"
    FAILED = "failed"


class AttachmentReviewState(StrEnum):
    RECEIVED = "received"
    REVIEWED = "reviewed"
    UPLOAD_AUTHORIZED = "upload_authorized"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    REJECTED = "rejected"
    UPLOAD_FAILED = "upload_failed"


@dataclass
class ReportIntentRecord:
    id: str
    project_id: str
    finding_id: str | None
    draft_id: str
    program_handle: str
    title: str = ""
    vulnerability_information: str = ""
    impact: str = ""
    severity: str | None = None
    local_status: IntentLocalState = IntentLocalState.LOCAL_DRAFT
    human_review_state: IntentLocalState = IntentLocalState.LOCAL_DRAFT
    remote_intent_id: str | None = None
    remote_state: HackerOneRemoteState = HackerOneRemoteState.NOT_FETCHED
    remote_state_raw: str = ""
    error: str | None = None
    last_payload: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "finding_id": self.finding_id,
            "draft_id": self.draft_id,
            "program_handle": self.program_handle,
            "title": self.title,
            "vulnerability_information": self.vulnerability_information,
            "impact": self.impact,
            "severity": self.severity,
            "local_status": self.local_status.value,
            "human_review_state": self.human_review_state.value,
            "remote_intent_id": self.remote_intent_id,
            "remote_state": self.remote_state.value,
            "remote_state_raw": self.remote_state_raw,
            "error": self.error,
            "assistant_output_is_evidence": False,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class AttachmentRecord:
    id: str
    intent_id: str
    project_id: str
    finding_id: str
    draft_id: str
    filename: str
    stored_path: str
    sha256: str
    size: int
    detected_content_type: str
    review_state: AttachmentReviewState = AttachmentReviewState.RECEIVED
    remote_attachment_id: str | None = None
    upload_state: AttachmentReviewState = AttachmentReviewState.RECEIVED
    error: str | None = None
    reviewed_by: str | None = None
    authorized_by: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "intent_id": self.intent_id,
            "project_id": self.project_id,
            "finding_id": self.finding_id,
            "draft_id": self.draft_id,
            "filename": self.filename,
            "sha256": self.sha256,
            "size": self.size,
            "detected_content_type": self.detected_content_type,
            "review_state": self.review_state.value,
            "reviewed": self.review_state
            in {
                AttachmentReviewState.REVIEWED,
                AttachmentReviewState.UPLOAD_AUTHORIZED,
                AttachmentReviewState.UPLOADING,
                AttachmentReviewState.UPLOADED,
            },
            "authorized_for_upload": self.review_state
            in {
                AttachmentReviewState.UPLOAD_AUTHORIZED,
                AttachmentReviewState.UPLOADING,
                AttachmentReviewState.UPLOADED,
            },
            "uploaded": self.review_state is AttachmentReviewState.UPLOADED,
            "remote_attachment_id": self.remote_attachment_id,
            "upload_state": self.upload_state.value,
            "error": self.error,
            "reviewed_by": self.reviewed_by,
            "authorized_by": self.authorized_by,
        }


class ReportIntentWorkflow:
    """Local report-intent records bound to a persisted finding. Not verification."""

    def __init__(
        self,
        reports: HackerOneReportWorkflow,
        *,
        evidence_root: Path | None = None,
    ) -> None:
        self.reports = reports
        self.intents: dict[str, ReportIntentRecord] = {}
        self.attachments: dict[str, list[AttachmentRecord]] = {}
        self._evidence_root = evidence_root

    def create_from_draft(self, draft: HackerOneReportDraft, *, project_id: str) -> dict[str, Any]:
        if draft.project_id and draft.project_id != project_id:
            raise HackerOneError("Report intent does not belong to this project", code="forbidden")
        intent = ReportIntentRecord(
            id=uuid4().hex,
            project_id=project_id,
            finding_id=draft.finding_id,
            draft_id=draft.id,
            program_handle=draft.program_handle,
            title=draft.title,
            vulnerability_information=draft.vulnerability_information,
            impact=draft.impact,
            severity=draft.severity,
            local_status=IntentLocalState.LOCAL_DRAFT,
            human_review_state=IntentLocalState.LOCAL_DRAFT,
        )
        self.intents[intent.id] = intent
        return intent.snapshot()

    def get(self, intent_id: str) -> dict[str, Any]:
        return self._record(intent_id).snapshot()

    def record(self, intent_id: str) -> ReportIntentRecord:
        return self._record(intent_id)

    def mark_ready(self, intent_id: str, *, project_id: str) -> dict[str, Any]:
        intent = self._owned(intent_id, project_id)
        updated = replace(
            intent,
            local_status=IntentLocalState.READY_FOR_REVIEW,
            human_review_state=IntentLocalState.READY_FOR_REVIEW,
            updated_at=datetime.now(UTC),
        )
        self.intents[intent_id] = updated
        return updated.snapshot()

    def human_approve(
        self, intent_id: str, *, session: OperatorSession, project_id: str
    ) -> dict[str, Any]:
        session.assert_human()
        intent = self._owned(intent_id, project_id)
        if intent.local_status not in {
            IntentLocalState.READY_FOR_REVIEW,
            IntentLocalState.LOCAL_DRAFT,
            IntentLocalState.FAILED,
        }:
            raise HackerOneError("Intent is not awaiting human approval", code="invalid_state")
        updated = replace(
            intent,
            local_status=IntentLocalState.HUMAN_APPROVED,
            human_review_state=IntentLocalState.HUMAN_APPROVED,
            updated_at=datetime.now(UTC),
            error=None,
        )
        self.intents[intent_id] = updated
        return updated.snapshot()

    def patch(self, intent_id: str, payload: dict[str, Any], *, project_id: str) -> dict[str, Any]:
        intent = self._owned(intent_id, project_id)
        if "payload" in payload:
            raise HackerOneError(
                "Raw HackerOne intent payloads cannot bypass BugForge finding evidence",
                code="raw_payload_forbidden",
            )
        updates: dict[str, Any] = {}
        for key in ("title", "impact", "severity"):
            if key in payload and payload[key] is not None:
                updates[key] = payload[key]
        if intent.human_review_state is IntentLocalState.HUMAN_APPROVED and updates:
            updates["local_status"] = IntentLocalState.READY_FOR_REVIEW
            updates["human_review_state"] = IntentLocalState.READY_FOR_REVIEW
            updates["error"] = "Approval invalidated because the intent changed"
        updated = replace(intent, **updates, updated_at=datetime.now(UTC))
        self.intents[intent_id] = updated
        return updated.snapshot()

    def create_remote(
        self,
        intent_id: str,
        program: HackerOneProgram,
        *,
        session: OperatorSession,
        project_id: str,
    ) -> dict[str, Any]:
        session.assert_human()
        intent = self._owned(intent_id, project_id)
        if intent.local_status is not IntentLocalState.HUMAN_APPROVED:
            raise HackerOneError(
                "Remote HackerOne intents require HUMAN_APPROVED", code="not_approved"
            )
        draft = self.reports.get_draft(intent.draft_id)
        payload = _intent_payload(draft, program)
        try:
            body = self.reports.client.create_report_intent(payload)
        except HackerOneError as exc:
            failed = replace(
                intent,
                local_status=IntentLocalState.FAILED,
                error=str(exc),
                updated_at=datetime.now(UTC),
            )
            self.intents[intent_id] = failed
            return failed.snapshot()
        data = body.get("data") if isinstance(body.get("data"), dict) else body
        remote_id = node_id(data) or None
        attrs = attributes(data)
        mapped, raw = map_remote_state(attrs.get("state") or attrs.get("substate"))
        updated = replace(
            intent,
            local_status=IntentLocalState.REMOTE_INTENT_CREATED,
            human_review_state=IntentLocalState.HUMAN_APPROVED,
            remote_intent_id=remote_id,
            remote_state=mapped,
            remote_state_raw=raw,
            last_payload=_redact_mapping(payload),
            error=None,
            updated_at=datetime.now(UTC),
        )
        self.intents[intent_id] = updated
        return updated.snapshot()

    def refresh_remote(self, intent_id: str, *, project_id: str) -> dict[str, Any]:
        intent = self._owned(intent_id, project_id)
        if not intent.remote_intent_id:
            raise HackerOneError("No remote HackerOne intent id", code="not_found")
        body = self.reports.client.get_report_intent(intent.remote_intent_id)
        data = body.get("data") if isinstance(body.get("data"), dict) else body
        attrs = attributes(data)
        mapped, raw = map_remote_state(attrs.get("state") or attrs.get("substate"))
        status = intent.local_status
        if mapped is not HackerOneRemoteState.UNKNOWN:
            if raw in {"ready-to-submit", "ready_to_submit"}:
                status = IntentLocalState.REMOTE_READY_TO_SUBMIT
        updated = replace(
            intent,
            remote_state=mapped,
            remote_state_raw=raw or intent.remote_state_raw,
            local_status=status,
            updated_at=datetime.now(UTC),
        )
        self.intents[intent_id] = updated
        return updated.snapshot()

    def submit(
        self,
        intent_id: str,
        program: HackerOneProgram,
        *,
        session: OperatorSession,
        finding: SecurityFinding | None,
        project_id: str,
    ) -> HackerOneReportDraft:
        intent = self._owned(intent_id, project_id)
        if intent.local_status not in {
            IntentLocalState.HUMAN_APPROVED,
            IntentLocalState.REMOTE_INTENT_CREATED,
            IntentLocalState.REMOTE_READY_TO_SUBMIT,
        }:
            raise HackerOneError("Intent submission requires human approval", code="not_approved")
        if intent.remote_intent_id:
            try:
                self.reports.client.submit_report_intent(intent.remote_intent_id)
                updated = replace(
                    intent,
                    local_status=IntentLocalState.SUBMITTED,
                    updated_at=datetime.now(UTC),
                    error=None,
                )
                self.intents[intent_id] = updated
            except HackerOneError as exc:
                failed = replace(intent, local_status=IntentLocalState.FAILED, error=str(exc))
                self.intents[intent_id] = failed
                raise
        draft = self.reports.submit(intent.draft_id, program, finding=finding, session=session)
        if draft.submission_state.value == "submitted":
            self.intents[intent_id] = replace(
                self._record(intent_id),
                local_status=IntentLocalState.SUBMITTED,
                updated_at=datetime.now(UTC),
            )
        return draft

    def list_attachments(
        self, intent_id: str, *, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        if project_id is not None:
            self._owned(intent_id, project_id)
        else:
            self._record(intent_id)
        return [item.snapshot() for item in self.attachments.get(intent_id, [])]

    def add_attachment(
        self,
        intent_id: str,
        *,
        filename: str,
        content: bytes,
        session: OperatorSession,
        project_id: str,
        reviewed: bool | None = None,
    ) -> dict[str, Any]:
        del reviewed  # client-supplied reviewed=true is never a review
        session.assert_human()
        intent = self._owned(intent_id, project_id)
        safe_name = _safe_filename(filename)
        max_bytes = get_settings().hackerone_attachment_max_bytes
        if len(content) > max_bytes:
            raise HackerOneError("Attachment exceeds size limit", code="too_large")
        if not content:
            raise HackerOneError("Attachment is empty", code="empty_attachment")
        digest = sha256_bytes(content)
        detected = sniff_content_type(content, safe_name)
        leaked = scan_attachment(content, safe_name, detected)
        if leaked:
            raise HackerOneError(
                f"Attachment rejected: possible secret leakage ({', '.join(leaked)})",
                code="secret_detected",
            )
        stored = self._store_bytes(project_id, intent.id, digest, safe_name, content)
        record = AttachmentRecord(
            id=uuid4().hex,
            intent_id=intent.id,
            project_id=project_id,
            finding_id=intent.finding_id or "",
            draft_id=intent.draft_id,
            filename=safe_name,
            stored_path=str(stored),
            sha256=digest,
            size=len(content),
            detected_content_type=detected,
        )
        self.attachments.setdefault(intent_id, []).append(record)
        return record.snapshot()

    def review_attachment(
        self,
        intent_id: str,
        attachment_id: str,
        *,
        session: OperatorSession,
        project_id: str,
        reject: bool = False,
    ) -> dict[str, Any]:
        session.assert_human()
        item = self._attachment(intent_id, attachment_id, project_id)
        if item.review_state is not AttachmentReviewState.RECEIVED:
            raise HackerOneError("Attachment is not awaiting review", code="invalid_state")
        item.review_state = (
            AttachmentReviewState.REJECTED if reject else AttachmentReviewState.REVIEWED
        )
        item.upload_state = item.review_state
        item.reviewed_by = session.identity
        return item.snapshot()

    def authorize_upload(
        self, intent_id: str, attachment_id: str, *, session: OperatorSession, project_id: str
    ) -> dict[str, Any]:
        session.assert_human()
        item = self._attachment(intent_id, attachment_id, project_id)
        if item.review_state is not AttachmentReviewState.REVIEWED:
            raise HackerOneError("Attachment has not been reviewed", code="not_reviewed")
        item.review_state = AttachmentReviewState.UPLOAD_AUTHORIZED
        item.upload_state = AttachmentReviewState.UPLOAD_AUTHORIZED
        item.authorized_by = session.identity
        return item.snapshot()

    def upload_attachment(
        self,
        intent_id: str,
        attachment_id: str,
        *,
        session: OperatorSession,
        project_id: str,
    ) -> dict[str, Any]:
        session.assert_human()
        intent = self._owned(intent_id, project_id)
        item = self._attachment(intent_id, attachment_id, project_id)
        if item.review_state is not AttachmentReviewState.UPLOAD_AUTHORIZED:
            raise HackerOneError(
                "Attachment upload requires explicit human authorization",
                code="not_authorized",
            )
        if not intent.remote_intent_id:
            raise HackerOneError(
                "Create the remote HackerOne intent before uploading attachments",
                code="no_remote_intent",
            )
        item.review_state = AttachmentReviewState.UPLOADING
        item.upload_state = AttachmentReviewState.UPLOADING
        try:
            body = self.reports.client.create_report_intent_attachment(
                intent.remote_intent_id,
                {
                    "data": {
                        "type": "attachment",
                        "attributes": {
                            "file_name": item.filename,
                            "content_type": item.detected_content_type,
                            "file_size": item.size,
                            "digest": item.sha256,
                        },
                    }
                },
            )
        except HackerOneError as exc:
            item.review_state = AttachmentReviewState.UPLOAD_FAILED
            item.upload_state = AttachmentReviewState.UPLOAD_FAILED
            item.error = str(exc)
            return item.snapshot()
        data = body.get("data") if isinstance(body.get("data"), dict) else body
        item.remote_attachment_id = node_id(data) or None
        item.review_state = AttachmentReviewState.UPLOADED
        item.upload_state = AttachmentReviewState.UPLOADED
        item.error = None
        return item.snapshot()

    def delete_attachment(self, intent_id: str, attachment_id: str, *, project_id: str) -> None:
        intent = self._owned(intent_id, project_id)
        items = self.attachments.get(intent_id, [])
        remaining: list[AttachmentRecord] = []
        for item in items:
            if item.id != attachment_id:
                remaining.append(item)
                continue
            if item.remote_attachment_id and intent.remote_intent_id:
                self.reports.client.delete_report_intent_attachment(
                    intent.remote_intent_id, item.remote_attachment_id
                )
        self.attachments[intent_id] = remaining

    def attachment(self, intent_id: str, attachment_id: str, project_id: str) -> AttachmentRecord:
        return self._attachment(intent_id, attachment_id, project_id)

    def _store_bytes(
        self, project_id: str, intent_id: str, digest: str, filename: str, content: bytes
    ) -> Path:
        root = (
            self._evidence_root
            or Path(get_settings().bugforge_evidence_dir or "/tmp/bugforge-evidence")
        ).resolve()
        safe_project = _safe_filename(project_id)
        target_dir = (root / safe_project / "hackerone-attachments" / intent_id).resolve()
        if root not in target_dir.parents:
            raise HackerOneError("Attachment path escaped evidence directory", code="unsafe_path")
        target_dir.mkdir(parents=True, exist_ok=True)
        path = (target_dir / f"{digest[:16]}-{filename}").resolve()
        if root not in path.parents:
            raise HackerOneError("Attachment path escaped evidence directory", code="unsafe_path")
        path.write_bytes(content)
        return path

    def _owned(self, intent_id: str, project_id: str) -> ReportIntentRecord:
        intent = self._record(intent_id)
        if intent.project_id != project_id:
            raise HackerOneError("Report intent does not belong to this project", code="forbidden")
        return intent

    def _record(self, intent_id: str) -> ReportIntentRecord:
        intent = self.intents.get(intent_id)
        if intent is None:
            raise HackerOneError("Unknown report intent", code="not_found")
        return intent

    def _attachment(self, intent_id: str, attachment_id: str, project_id: str) -> AttachmentRecord:
        self._owned(intent_id, project_id)
        for item in self.attachments.get(intent_id, []):
            if item.id == attachment_id:
                if item.project_id != project_id:
                    raise HackerOneError(
                        "Attachment does not belong to this project", code="forbidden"
                    )
                return item
        raise HackerOneError("Unknown attachment", code="not_found")


def sniff_content_type(data: bytes, filename: str) -> str:
    if data.startswith(b"%PDF"):
        return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    suffix = Path(filename).suffix.lower()
    if _looks_text(data):
        if suffix == ".json":
            return "application/json"
        if suffix in {".xml", ".html", ".htm", ".csv", ".txt", ".log", ".md"}:
            return {
                ".xml": "application/xml",
                ".html": "text/html",
                ".htm": "text/html",
                ".csv": "text/csv",
                ".txt": "text/plain",
                ".log": "text/plain",
                ".md": "text/plain",
            }[suffix]
        return "text/plain"
    return "application/octet-stream"


def scan_attachment(data: bytes, filename: str, content_type: str) -> tuple[str, ...]:
    hits: list[str] = []
    lowered = data.lower()
    for marker in _GENERIC_SECRET_MARKERS:
        if marker.lower() in lowered:
            hits.append(marker.decode("ascii", errors="replace").strip())
    if content_type.startswith("text/") or content_type in _TEXT_TYPES:
        text = data.decode("utf-8", errors="replace")
        hits.extend(detect_secrets(filename, text, configured=configured_secret_values()))
    return tuple(dict.fromkeys(hits))


def _looks_text(data: bytes) -> bool:
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    printable = sum(32 <= byte < 127 or byte in {9, 10, 13} for byte in sample)
    return printable / max(len(sample), 1) > 0.85


def _safe_filename(name: str) -> str:
    base = name.replace("\\", "/").split("/")[-1]
    cleaned = "".join(ch for ch in base if ch.isalnum() or ch in {".", "-", "_"})
    if not cleaned or cleaned in {".", ".."}:
        raise HackerOneError("Unsafe attachment filename", code="unsafe_path")
    return cleaned[:_MAX_NAME]


def _intent_payload(draft: HackerOneReportDraft, program: HackerOneProgram) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "team_handle": program.handle,
        "title": draft.title,
        "vulnerability_information": draft.vulnerability_information,
        "impact": draft.impact,
    }
    if draft.severity:
        attributes["severity_rating"] = draft.severity
    if draft.weakness_id is not None:
        attributes["weakness_id"] = int(draft.weakness_id)
    if draft.structured_scope_id is not None:
        attributes["structured_scope_id"] = int(draft.structured_scope_id)
    return {"data": {"type": "report-intent", "attributes": attributes}}


def _redact_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    import json

    from app.security_testing.secrets import redact_text

    parsed: object = json.loads(redact_text(json.dumps(payload, default=str)))
    return parsed if isinstance(parsed, dict) else payload
