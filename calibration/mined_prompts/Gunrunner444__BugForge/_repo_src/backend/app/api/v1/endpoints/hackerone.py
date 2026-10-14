"""HackerOne program + report workflow API. Credentials never leave the adapter."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.hackerone.credentials import HackerOneCredentials
from app.adapters.hackerone.errors import HackerOneError
from app.adapters.hackerone.models import HackerOneProgram
from app.adapters.hackerone.provider import HackerOneProvider
from app.database import get_db
from app.domain.findings import SecurityFinding
from app.models.project import Project
from app.repositories.hackerone_repo import HackerOneRepository
from app.repositories.security_finding_repo import SecurityFindingRepository
from app.schemas.hackerone import (
    ActiveTestingRequest,
    AttachmentAuthorizeRequest,
    AttachmentCreateRequest,
    AttachmentReviewRequest,
    CreateDraftRequest,
    DraftActionRequest,
    DraftEditRequest,
    HackerOneSyncRequest,
    IntentCreateRequest,
    IntentPatchRequest,
    IntentSubmitRequest,
    OpenScopeAckRequest,
    ReconcileRequest,
)
from app.security_testing.errors import RestrictedActivityError
from app.security_testing.operator_auth import OperatorSession, require_operator

router = APIRouter(prefix="/hackerone", tags=["HackerOne"])

_PROVIDER: HackerOneProvider | None = None


def get_hackerone_provider() -> HackerOneProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = HackerOneProvider(credentials=HackerOneCredentials.from_env())
    return _PROVIDER


def reset_hackerone_provider(provider: HackerOneProvider | None = None) -> None:
    global _PROVIDER
    _PROVIDER = provider


@router.get("/status")
async def connection_status(db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    provider = get_hackerone_provider()
    repo = HackerOneRepository(db)
    await repo.hydrate(provider)
    status_payload = provider.connection_status()
    return {
        key: value
        for key, value in status_payload.items()
        if "token" not in key or key == "token_present"
    }


@router.post("/programs/sync")
async def sync_program(
    payload: HackerOneSyncRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    if not provider.credentials.configured:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="HackerOne API credentials are not configured (HACKERONE_API_USERNAME / HACKERONE_API_TOKEN)",
        )
    try:
        provider.lookup_program(payload.handle)
        program = provider.sync_scope(payload.handle)
        await repo.replace_program(program)
        await repo.audit(
            "program_imported" if program.scope_version <= 1 else "scope_synced",
            program_handle=program.handle,
            operator_identity=session.identity,
            details={"scope_count": program.scope_count, "weakness_count": len(program.weaknesses)},
        )
        if program.weaknesses:
            await repo.audit(
                "weaknesses_synced",
                program_handle=program.handle,
                operator_identity=session.identity,
                details={"weakness_count": len(program.weaknesses)},
            )
        await db.commit()
    except HackerOneError as exc:
        await db.rollback()
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    return _program_view(program)


@router.get("/programs/{handle}")
async def get_program(handle: str, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    provider, repo = await _bound(db)
    program = provider.programs.get(handle) or await repo.get_program(handle)
    if program is None:
        raise HTTPException(status_code=404, detail="Program has not been imported")
    return _program_view(program)


@router.post("/programs/{handle}/open-scope")
async def ack_open_scope(
    handle: str,
    payload: OpenScopeAckRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    try:
        program = provider.acknowledge_open_scope(handle, session=session, policy=payload.policy)
        await repo.replace_program(program)
        await repo.audit(
            "open_scope_acknowledged",
            program_handle=handle,
            operator_identity=session.identity,
        )
        await db.commit()
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HackerOneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _program_view(program)


@router.post("/programs/{handle}/active-testing")
async def approve_active_testing(
    handle: str,
    payload: ActiveTestingRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    del payload
    provider, repo = await _bound(db)
    try:
        program = provider.approve_active_testing(handle, session=session)
        await repo.replace_program(program)
        await repo.audit(
            "active_testing_approved",
            program_handle=handle,
            operator_identity=session.identity,
        )
        await db.commit()
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HackerOneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _program_view(program)


@router.post("/reports/drafts")
async def create_draft(
    payload: CreateDraftRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    finding = await _load_finding(db, payload.finding_id, payload.project_id)
    try:
        draft = provider.draft_from_finding(
            finding,
            payload.program_handle,
            severity=payload.severity,
            weakness_id=payload.weakness_id,
            operator=session,
            project_id=str(payload.project_id),
        )
        await repo.save_draft(draft)
        await repo.audit(
            "draft_created",
            project_id=str(payload.project_id),
            finding_id=str(finding.id),
            draft_id=draft.id,
            program_handle=payload.program_handle,
            operator_identity=session.identity,
        )
        await repo.audit(
            "finding_selected",
            project_id=str(payload.project_id),
            finding_id=str(finding.id),
            draft_id=draft.id,
            program_handle=payload.program_handle,
            operator_identity=session.identity,
        )
        await db.commit()
    except HackerOneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return draft.snapshot()


@router.post("/reports/{draft_id}/review")
async def mark_review(
    draft_id: str,
    payload: DraftActionRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    draft = provider.reports.drafts.get(draft_id)
    program = _program_for(
        provider, payload.program_handle, draft.program_handle if draft else None
    )
    finding = await _finding_for_draft(db, draft)
    try:
        updated = provider.reports.mark_ready(draft_id, program, finding=finding)
        await repo.save_draft(updated)
        await repo.audit(
            "draft_reviewed",
            project_id=updated.project_id or "",
            finding_id=updated.finding_id or "",
            draft_id=draft_id,
            program_handle=updated.program_handle,
            operator_identity=session.identity,
        )
        await db.commit()
    except HackerOneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return updated.snapshot()


@router.patch("/reports/{draft_id}")
async def edit_draft(
    draft_id: str,
    payload: DraftEditRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    draft = provider.reports.drafts.get(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    program = _program_for(provider, None, draft.program_handle)
    finding = await _finding_for_draft(db, draft)
    try:
        updated = provider.reports.apply_edits(
            draft_id,
            program,
            title=payload.title,
            vulnerability_information=payload.vulnerability_information,
            impact=payload.impact,
            severity=payload.severity,
            weakness_id=payload.weakness_id,
            finding=finding,
        )
        await repo.save_draft(updated)
        if updated.approval is None and draft.approval is not None:
            await repo.record_approval_event(
                draft_id=draft_id,
                action="approval_invalidated",
                approved_by=session.identity,
                reason="Draft edited after approval",
            )
            await repo.audit(
                "approval_invalidated",
                project_id=updated.project_id or "",
                finding_id=updated.finding_id or "",
                draft_id=draft_id,
                program_handle=updated.program_handle,
                operator_identity=session.identity,
            )
        await db.commit()
    except HackerOneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return updated.snapshot()


@router.post("/reports/{draft_id}/dry-run")
async def dry_run(
    draft_id: str,
    payload: DraftActionRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    draft = provider.reports.drafts.get(draft_id)
    program = _program_for(
        provider, payload.program_handle, draft.program_handle if draft else None
    )
    if draft is None:
        raise HTTPException(status_code=404, detail="Draft or program not found")
    finding = await _finding_for_draft(db, draft)
    result = provider.reports.dry_run(draft_id, program, finding=finding)
    await repo.audit(
        "dry_run_generated",
        project_id=draft.project_id or "",
        finding_id=draft.finding_id or "",
        draft_id=draft_id,
        program_handle=draft.program_handle,
        operator_identity=session.identity,
    )
    await db.commit()
    return result


@router.post("/reports/{draft_id}/approve")
async def approve(
    draft_id: str,
    payload: DraftActionRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    draft = provider.reports.drafts.get(draft_id)
    program = _program_for(
        provider, payload.program_handle, draft.program_handle if draft else None
    )
    finding = await _finding_for_draft(db, draft)
    try:
        updated = provider.reports.human_approve(
            draft_id, program, session=session, finding=finding
        )
        await repo.save_draft(updated)
        if updated.approval:
            await repo.record_approval_event(
                draft_id=draft_id,
                action="approval_granted",
                approved_by=session.identity,
                report_content_hash=updated.approval.report_content_hash,
                evidence_hash=updated.approval.evidence_hash,
                scope_snapshot_hash=updated.approval.scope_snapshot_hash,
                payload_hash=updated.approval.payload_hash,
            )
        await repo.audit(
            "approval_granted",
            project_id=updated.project_id or "",
            finding_id=updated.finding_id or "",
            draft_id=draft_id,
            program_handle=updated.program_handle,
            operator_identity=session.identity,
        )
        await db.commit()
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HackerOneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return updated.snapshot()


@router.post("/reports/{draft_id}/submit")
async def submit(
    draft_id: str,
    payload: DraftActionRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    draft = provider.reports.drafts.get(draft_id)
    program = _program_for(
        provider, payload.program_handle, draft.program_handle if draft else None
    )
    if draft is None:
        raise HTTPException(status_code=404, detail="Draft or program not found")
    if draft.human_review_state.value != "human_approved":
        raise HTTPException(
            status_code=403,
            detail="Submit remains disabled until HUMAN_APPROVED and validation pass",
        )
    finding = await _finding_for_draft(db, draft)
    try:
        await repo.audit(
            "submission_attempted",
            project_id=draft.project_id or "",
            finding_id=draft.finding_id or "",
            draft_id=draft_id,
            program_handle=draft.program_handle,
            operator_identity=session.identity,
        )
        claimed, reason = await repo.claim_submission(
            draft_id=draft_id,
            finding_id=draft.finding_id,
            program_handle=draft.program_handle,
        )
        await db.commit()
        await repo.hydrate(provider)
        if not claimed:
            code = (
                409 if reason in {"submission_in_progress", "submission_outcome_unknown"} else 400
            )
            detail = {
                "submission_in_progress": "SUBMISSION_IN_PROGRESS",
                "submission_outcome_unknown": "SUBMISSION_OUTCOME_UNKNOWN",
                "duplicate": "Finding already submitted to this program",
            }.get(reason, reason)
            raise HTTPException(status_code=code, detail=detail)
        result = provider.reports.submit(
            draft_id, program, finding=finding, session=session, already_claimed=True
        )
        await repo.save_draft(result)
        if result.finding_id:
            existing = provider.reports.submissions.get((result.finding_id, result.program_handle))
            if existing:
                await repo.save_submission(existing)
        action = {
            "submitted": "submission_accepted",
            "submission_failed": "submission_failed",
            "submission_outcome_unknown": "submission_outcome_unknown",
            "identity_verification_required": "submission_failed",
        }.get(result.submission_state.value, "submission_attempted")
        await repo.audit(
            action,
            project_id=result.project_id or "",
            finding_id=result.finding_id or "",
            draft_id=draft_id,
            program_handle=result.program_handle,
            operator_identity=session.identity,
            details={"hackerone_report_id": result.hackerone_report_id},
        )
        await db.commit()
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HackerOneError as exc:
        raise HTTPException(status_code=exc.status_code or 400, detail=str(exc)) from exc
    return result.snapshot()


@router.post("/reports/{draft_id}/reconcile")
async def reconcile(
    draft_id: str,
    payload: ReconcileRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    del payload
    provider, repo = await _bound(db)
    try:
        updated = provider.reports.reconcile(draft_id)
        await repo.save_draft(updated)
        if updated.finding_id:
            existing = provider.reports.submissions.get(
                (updated.finding_id, updated.program_handle)
            )
            if existing:
                await repo.save_submission(existing)
        await repo.audit(
            "remote_report_reconciled",
            project_id=updated.project_id or "",
            finding_id=updated.finding_id or "",
            draft_id=draft_id,
            program_handle=updated.program_handle,
            operator_identity=session.identity,
            details={"remote_state": updated.remote_state.value},
        )
        await db.commit()
    except HackerOneError as exc:
        raise HTTPException(status_code=exc.status_code or 400, detail=str(exc)) from exc
    return updated.snapshot()


@router.post("/report-intents")
async def create_intent(
    payload: IntentCreateRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    finding = await _load_finding(db, payload.finding_id, payload.project_id)
    try:
        draft = provider.draft_from_finding(
            finding,
            payload.program_handle,
            severity=payload.severity,
            weakness_id=payload.weakness_id,
            operator=session,
            project_id=str(payload.project_id),
        )
        intent = provider.intents.create_from_draft(draft, project_id=str(payload.project_id))
        await repo.save_draft(draft)
        await repo.save_intent(provider.intents.record(str(intent["id"])))
        await repo.audit(
            "draft_created",
            project_id=str(payload.project_id),
            finding_id=str(finding.id),
            draft_id=draft.id,
            program_handle=payload.program_handle,
            operator_identity=session.identity,
            details={"intent_id": intent["id"]},
        )
        await db.commit()
    except HackerOneError as exc:
        raise HTTPException(status_code=exc.status_code or 400, detail=str(exc)) from exc
    return intent


@router.get("/report-intents/{intent_id}")
async def get_intent(
    intent_id: str,
    project_id: UUID,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    del session
    provider, _repo = await _bound(db)
    try:
        intent = provider.intents.get(intent_id)
        if intent.get("project_id") != str(project_id):
            raise HTTPException(
                status_code=403, detail="Report intent does not belong to this project"
            )
        return intent
    except HackerOneError as exc:
        raise HTTPException(status_code=exc.status_code or 400, detail=str(exc)) from exc


@router.post("/report-intents/{intent_id}/review")
async def review_intent(
    intent_id: str,
    payload: IntentSubmitRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    del session
    provider, repo = await _bound(db)
    try:
        updated = provider.intents.mark_ready(intent_id, project_id=str(payload.project_id))
        await repo.save_intent(provider.intents.record(intent_id))
        await db.commit()
        return updated
    except HackerOneError as exc:
        raise HTTPException(status_code=exc.status_code or 400, detail=str(exc)) from exc


@router.post("/report-intents/{intent_id}/approve")
async def approve_intent(
    intent_id: str,
    payload: IntentSubmitRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    try:
        updated = provider.intents.human_approve(
            intent_id, session=session, project_id=str(payload.project_id)
        )
        await repo.save_intent(provider.intents.record(intent_id))
        await db.commit()
        return updated
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HackerOneError as exc:
        raise HTTPException(status_code=exc.status_code or 400, detail=str(exc)) from exc


@router.post("/report-intents/{intent_id}/remote")
async def create_remote_intent(
    intent_id: str,
    payload: IntentSubmitRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    try:
        intent = provider.intents.record(intent_id)
        program = _program_for(provider, payload.program_handle, intent.program_handle)
        updated = provider.intents.create_remote(
            intent_id, program, session=session, project_id=str(payload.project_id)
        )
        await repo.save_intent(provider.intents.record(intent_id))
        await db.commit()
        return updated
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HackerOneError as exc:
        raise HTTPException(status_code=exc.status_code or 400, detail=str(exc)) from exc


@router.patch("/report-intents/{intent_id}")
async def patch_intent(
    intent_id: str,
    payload: IntentPatchRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    del session
    provider, repo = await _bound(db)
    try:
        updated = provider.intents.patch(
            intent_id,
            {"title": payload.title, "impact": payload.impact, "severity": payload.severity},
            project_id=str(payload.project_id),
        )
        await repo.save_intent(provider.intents.record(intent_id))
        await db.commit()
        return updated
    except HackerOneError as exc:
        raise HTTPException(status_code=exc.status_code or 400, detail=str(exc)) from exc


@router.post("/report-intents/{intent_id}/submit")
async def submit_intent(
    intent_id: str,
    payload: IntentSubmitRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    try:
        intent = provider.intents.record(intent_id)
        if intent.project_id != str(payload.project_id):
            raise HTTPException(
                status_code=403, detail="Report intent does not belong to this project"
            )
        draft = provider.reports.drafts.get(intent.draft_id)
        program = _program_for(provider, payload.program_handle, intent.program_handle)
        finding = await _finding_for_draft(db, draft)
        result = provider.intents.submit(
            intent_id,
            program,
            session=session,
            finding=finding,
            project_id=str(payload.project_id),
        )
        await repo.save_draft(result)
        await repo.save_intent(provider.intents.record(intent_id))
        await db.commit()
        return result.snapshot()
    except HackerOneError as exc:
        raise HTTPException(status_code=exc.status_code or 400, detail=str(exc)) from exc
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/report-intents/{intent_id}/attachments")
async def list_attachments(
    intent_id: str,
    project_id: UUID,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    del session
    provider, _repo = await _bound(db)
    return {"items": provider.intents.list_attachments(intent_id, project_id=str(project_id))}


@router.post("/report-intents/{intent_id}/attachments")
async def add_attachment(
    intent_id: str,
    payload: AttachmentCreateRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    import base64

    provider, repo = await _bound(db)
    try:
        content = base64.b64decode(payload.content_base64.encode("ascii"), validate=False)
        record = provider.intents.add_attachment(
            intent_id,
            filename=payload.filename,
            content=content,
            session=session,
            project_id=str(payload.project_id),
        )
        await repo.save_attachment(
            provider.intents.attachment(intent_id, record["id"], str(payload.project_id))
        )
        await db.commit()
        return record
    except HackerOneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/report-intents/{intent_id}/attachments/{attachment_id}/review")
async def review_attachment(
    intent_id: str,
    attachment_id: str,
    payload: AttachmentReviewRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    try:
        record = provider.intents.review_attachment(
            intent_id,
            attachment_id,
            session=session,
            project_id=str(payload.project_id),
            reject=payload.reject,
        )
        await repo.save_attachment(
            provider.intents.attachment(intent_id, attachment_id, str(payload.project_id))
        )
        await db.commit()
        return record
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HackerOneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/report-intents/{intent_id}/attachments/{attachment_id}/authorize")
async def authorize_attachment(
    intent_id: str,
    attachment_id: str,
    payload: AttachmentAuthorizeRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    try:
        record = provider.intents.authorize_upload(
            intent_id, attachment_id, session=session, project_id=str(payload.project_id)
        )
        await repo.save_attachment(
            provider.intents.attachment(intent_id, attachment_id, str(payload.project_id))
        )
        await db.commit()
        return record
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HackerOneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/report-intents/{intent_id}/attachments/{attachment_id}/upload")
async def upload_attachment(
    intent_id: str,
    attachment_id: str,
    payload: AttachmentAuthorizeRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    provider, repo = await _bound(db)
    try:
        record = provider.intents.upload_attachment(
            intent_id, attachment_id, session=session, project_id=str(payload.project_id)
        )
        await repo.save_attachment(
            provider.intents.attachment(intent_id, attachment_id, str(payload.project_id))
        )
        await db.commit()
        return record
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HackerOneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/report-intents/{intent_id}/attachments/{attachment_id}")
async def delete_attachment(
    intent_id: str,
    attachment_id: str,
    project_id: UUID,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    del session
    provider, repo = await _bound(db)
    provider.intents.delete_attachment(intent_id, attachment_id, project_id=str(project_id))
    await db.commit()
    return {"deleted": True}


def _program_view(program: object) -> dict[str, object]:
    from app.adapters.hackerone.models import HackerOneProgram

    assert isinstance(program, HackerOneProgram)
    snap = program.snapshot()
    snap["structured_scopes"] = [
        {
            "id": item.id,
            "asset_type": item.asset_type.value,
            "asset_identifier": item.asset_identifier,
            "instruction": item.instruction,
            "eligible_for_bounty": item.eligible_for_bounty,
            "eligible_for_submission": item.eligible_for_submission,
            "reference": item.reference,
        }
        for item in program.structured_scopes
    ]
    snap["exclusions"] = [
        {
            "id": item.id,
            "category": item.category,
            "details": item.details,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
        }
        for item in program.exclusions
    ]
    snap["weaknesses"] = [
        {
            "id": item.id,
            "name": item.name,
            "description": item.description,
            "external_id": item.external_id,
        }
        for item in program.weaknesses
    ]
    snap["instructions"] = program.instructions
    return snap


async def _bound(db: AsyncSession) -> tuple[HackerOneProvider, HackerOneRepository]:
    provider = get_hackerone_provider()
    repo = HackerOneRepository(db)
    await repo.hydrate(provider)
    return provider, repo


def _program_for(
    provider: HackerOneProvider, requested: str | None, fallback: str | None
) -> HackerOneProgram:
    handle = requested or fallback or ""
    program = provider.programs.get(handle)
    if program is None:
        raise HTTPException(status_code=404, detail="Draft or program not found")
    return program


async def _load_finding(db: AsyncSession, finding_id: UUID, project_id: UUID) -> SecurityFinding:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    row = await SecurityFindingRepository(db).get(finding_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    if row.project_id is None or row.project_id != project_id:
        raise HTTPException(
            status_code=403,
            detail="Finding does not belong to the requested BugForge project",
        )
    finding = await SecurityFindingRepository(db).get_domain(finding_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    return finding


async def _finding_for_draft(db: AsyncSession, draft: object) -> SecurityFinding | None:
    from app.adapters.hackerone.reports import HackerOneReportDraft

    if not isinstance(draft, HackerOneReportDraft) or not draft.finding_id:
        return None
    try:
        finding_id = UUID(draft.finding_id)
    except ValueError:
        return None
    if draft.project_id:
        return await _load_finding(db, finding_id, UUID(draft.project_id))
    return await SecurityFindingRepository(db).get_domain(finding_id)
