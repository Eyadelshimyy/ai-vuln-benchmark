"""Bounty-campaign API (Phase 51/52).

Exposes the Phase 49 orchestrator, the Phase 50 bounty engines, and the Phase 52
local stateful execution over the existing FastAPI + operator-auth conventions.
Every route requires an authenticated local operator. The AI/Cursor cannot grant
an approval, change scope, raise a budget, mark a finding verified, or submit
anything through these routes.

Expensive campaign work (running the orchestrator, synthesizing and running
harnesses, building the report) never blocks the event loop: it runs on a worker
thread, either through the background-job endpoints or via ``run_in_threadpool``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.discovery.bounty.campaign import BountyManifest, ManifestError
from app.discovery.bounty.service import (
    APPROVABLE_CAPABILITIES,
    BountyCampaign,
    CampaignControlError,
    CampaignError,
    CampaignPersistenceError,
    CampaignSpec,
    UnknownCampaignError,
    get_bounty_service,
)
from app.security_testing.approvals import is_ai_operator
from app.security_testing.operator_auth import OperatorSession, require_operator

router = APIRouter(prefix="/bounty", tags=["Bounty Campaign"])


class CreateCampaignRequest(BaseModel):
    manifest: dict[str, object]
    repo_root: str
    target: str = ""
    contract: str = ""
    function: str = ""
    source_file: str = ""
    files: list[str] = []
    language: str = "solidity"
    max_rounds: int = 16
    max_engines: int = 16
    address: str = ""
    chain_id: str = ""


class ExecuteRequest(BaseModel):
    capability: str = ""
    reason: str = ""


class SuggestRequest(BaseModel):
    capability: str
    engine: str = ""
    reason: str = ""


class VfcsFeedbackRequest(BaseModel):
    signals: list[dict[str, object]] = []


class ApprovalRequest(BaseModel):
    capability: str


class ControlRequest(BaseModel):
    reason: str = "operator"


class JobRequest(BaseModel):
    max_steps: int = 32


class StatefulRequest(BaseModel):
    rounds: int = 3


def _owned(campaign_id: str, operator: OperatorSession) -> BountyCampaign:
    try:
        campaign = get_bounty_service().get(campaign_id)
    except UnknownCampaignError as exc:
        raise HTTPException(status_code=404, detail="Unknown campaign") from exc
    except CampaignPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    owner = campaign.operator_identity
    if owner and owner != operator.identity:
        raise HTTPException(status_code=403, detail="campaign_operator_mismatch")
    return campaign


def _control_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


@router.post("/campaigns")
async def create_campaign(
    payload: CreateCampaignRequest,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    try:
        manifest = BountyManifest.from_dict(payload.manifest)
    except ManifestError as exc:
        raise HTTPException(status_code=400, detail=f"invalid manifest: {exc}") from exc
    spec = CampaignSpec(
        manifest=manifest,
        repo_root=Path(payload.repo_root),
        target=payload.target,
        contract=payload.contract,
        function=payload.function,
        source_file=payload.source_file,
        files=tuple(payload.files),
        language=payload.language,
        max_rounds=max(1, min(payload.max_rounds, 64)),
        max_engines=max(1, min(payload.max_engines, 64)),
        address=payload.address,
        chain_id=payload.chain_id,
    )
    service = get_bounty_service()
    try:
        campaign = await run_in_threadpool(
            service.create, spec, operator_identity=operator.identity
        )
    except CampaignError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CampaignPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return await run_in_threadpool(service.report, campaign.campaign_id)


@router.get("/campaigns")
async def list_campaigns(
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    service = get_bounty_service()
    try:
        ids = await run_in_threadpool(service.list_ids, operator.identity)
    except CampaignPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"campaigns": ids}


@router.get("/campaigns/{campaign_id}")
async def inspect_campaign(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().report, campaign_id)


@router.get("/campaigns/{campaign_id}/state")
async def campaign_state(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().report, campaign_id)


@router.get("/campaigns/{campaign_id}/progress")
async def campaign_progress(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().progress, campaign_id)


@router.post("/campaigns/{campaign_id}/analyze")
async def analyze_campaign(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    try:
        return await run_in_threadpool(get_bounty_service().analyze, campaign_id)
    except CampaignControlError as exc:
        raise _control_error(exc) from exc


@router.post("/campaigns/{campaign_id}/jobs")
async def start_campaign_job(
    campaign_id: str,
    payload: JobRequest,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    try:
        return await run_in_threadpool(
            get_bounty_service().start_job, campaign_id, max_steps=payload.max_steps
        )
    except CampaignControlError as exc:
        raise _control_error(exc) from exc


@router.get("/campaigns/{campaign_id}/next-action")
async def campaign_next_action(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().next_action, campaign_id)


@router.get("/campaigns/{campaign_id}/decisions")
async def campaign_decisions(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().decisions, campaign_id)


@router.post("/campaigns/{campaign_id}/execute")
async def execute_campaign(
    campaign_id: str,
    payload: ExecuteRequest,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    try:
        return await run_in_threadpool(
            get_bounty_service().step,
            campaign_id,
            capability=payload.capability,
            reason=payload.reason,
        )
    except CampaignControlError as exc:
        raise _control_error(exc) from exc


@router.post("/campaigns/{campaign_id}/suggest")
async def suggest_capability(
    campaign_id: str,
    payload: SuggestRequest,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    accepted = await run_in_threadpool(
        get_bounty_service().suggest,
        campaign_id,
        payload.capability,
        engine=payload.engine,
        reason=payload.reason,
    )
    return {
        "accepted": accepted,
        "note": "a suggestion is validated like any candidate; it cannot widen scope or budget",
    }


@router.get("/campaigns/{campaign_id}/findings")
async def campaign_findings(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().findings, campaign_id)


@router.get("/campaigns/{campaign_id}/evidence")
async def campaign_evidence(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().evidence, campaign_id)


@router.get("/campaigns/{campaign_id}/repro")
async def campaign_repro(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().repro, campaign_id)


@router.get("/campaigns/{campaign_id}/source-selection")
async def campaign_source_selection(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().source_selection, campaign_id)


@router.get("/campaigns/{campaign_id}/scope-identity")
async def campaign_scope_identity(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().scope_identity, campaign_id)


@router.get("/campaigns/{campaign_id}/advisories")
async def campaign_advisories(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().advisories, campaign_id)


@router.post("/campaigns/{campaign_id}/vfcs-feedback")
async def campaign_vfcs_feedback(
    campaign_id: str,
    payload: VfcsFeedbackRequest,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    return await run_in_threadpool(
        get_bounty_service().vfcs_feedback, campaign_id, list(payload.signals)
    )


@router.post("/campaigns/{campaign_id}/stateful-execute")
async def campaign_stateful_execute(
    campaign_id: str,
    payload: StatefulRequest,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    try:
        return await run_in_threadpool(
            get_bounty_service().stateful_execute,
            campaign_id,
            rounds=max(1, min(payload.rounds, 5)),
        )
    except CampaignControlError as exc:
        raise _control_error(exc) from exc
    except CampaignPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/campaigns/{campaign_id}/stateful")
async def campaign_stateful_status(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    """Persisted execution observations, feedback, and bundle metadata (read only)."""
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().stateful_status, campaign_id)


@router.get("/campaigns/{campaign_id}/repro-bundles/{bundle_id}")
async def campaign_repro_bundle(
    campaign_id: str,
    bundle_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    try:
        return await run_in_threadpool(get_bounty_service().repro_bundle, campaign_id, bundle_id)
    except CampaignError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/campaigns/{campaign_id}/research/coverage")
async def campaign_research_coverage(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    """RESEARCH_COVERAGE and RESEARCH_GAPS (read only; ranks gaps, runs nothing)."""
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().research_ledger, campaign_id)


@router.get("/campaigns/{campaign_id}/research/plan")
async def campaign_research_plan(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    """Cost-aware next research actions within the server-owned budget (a recommendation)."""
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().research_plan, campaign_id)


@router.get("/campaigns/{campaign_id}/research/engines")
async def campaign_research_engines(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    """Engine availability: last run's smoke-checked snapshot and the current view."""
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().research_engines, campaign_id)


@router.get("/campaigns/{campaign_id}/research/properties")
async def campaign_research_properties(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    """Declared properties/oracles per sequence with their latest observation."""
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().research_properties, campaign_id)


@router.get("/campaigns/{campaign_id}/research/evidence-graph")
async def campaign_research_evidence_graph(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    """Derived evidence graph: provenance on every edge, contradictions visible."""
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().evidence_graph, campaign_id)


@router.get("/campaigns/{campaign_id}/research/modules")
async def campaign_research_modules(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    """Evidence-triggered protocol modules and the triggering evidence."""
    _owned(campaign_id, operator)
    return await run_in_threadpool(get_bounty_service().research_modules, campaign_id)


@router.get("/campaigns/{campaign_id}/report")
async def campaign_report(
    campaign_id: str,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    pack = await run_in_threadpool(get_bounty_service().report_pack, campaign_id)
    return {
        "campaign_id": campaign_id,
        "pack_id": pack.pack_id,
        "verified": pack.verified,
        "submitted": pack.submitted,
        "truncated": pack.truncated,
        "markdown": pack.markdown,
        "report": pack.to_dict(),
    }


@router.post("/campaigns/{campaign_id}/approvals")
async def grant_campaign_approval(
    campaign_id: str,
    payload: ApprovalRequest,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    if is_ai_operator(operator.identity):
        raise HTTPException(status_code=403, detail="The AI cannot grant approvals")
    _owned(campaign_id, operator)
    if payload.capability not in APPROVABLE_CAPABILITIES:
        raise HTTPException(status_code=400, detail="capability is not approvable")
    try:
        approvals = await run_in_threadpool(
            get_bounty_service().grant_approval,
            campaign_id,
            payload.capability,
            granted_by=operator.identity,
        )
    except CampaignError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"campaign_id": campaign_id, "approvals": sorted(approvals)}


@router.post("/campaigns/{campaign_id}/pause")
async def pause_campaign(
    campaign_id: str,
    payload: ControlRequest,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    try:
        return await run_in_threadpool(
            get_bounty_service().pause, campaign_id, reason=payload.reason
        )
    except CampaignControlError as exc:
        raise _control_error(exc) from exc


@router.post("/campaigns/{campaign_id}/resume")
async def resume_campaign(
    campaign_id: str,
    payload: ControlRequest,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    try:
        return await run_in_threadpool(get_bounty_service().resume, campaign_id)
    except CampaignControlError as exc:
        raise _control_error(exc) from exc
    except CampaignError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/campaigns/{campaign_id}/stop")
async def stop_campaign(
    campaign_id: str,
    payload: ControlRequest,
    operator: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    _owned(campaign_id, operator)
    try:
        return await run_in_threadpool(
            get_bounty_service().stop, campaign_id, reason=payload.reason
        )
    except CampaignControlError as exc:
        raise _control_error(exc) from exc
