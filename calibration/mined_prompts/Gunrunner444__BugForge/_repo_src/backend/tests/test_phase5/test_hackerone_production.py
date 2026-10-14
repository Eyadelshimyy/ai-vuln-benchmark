"""Phase 5: HackerOne production readiness — persistence, auth, payload, scope."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.hackerone.client import HackerOneApiClient
from app.adapters.hackerone.errors import HackerOneError, HackerOneUrlRejectedError
from app.adapters.hackerone.provider import HackerOneProvider
from app.adapters.hackerone.reports import ReportSubmissionState, compose_vulnerability_information
from app.adapters.hackerone.weakness import map_program_weakness
from app.api.v1.endpoints.hackerone import reset_hackerone_provider
from app.domain.findings import SecurityFinding
from app.models.project import Project
from app.repositories.hackerone_repo import HackerOneRepository
from app.repositories.security_finding_repo import SecurityFindingRepository
from app.security_testing.operator_auth import OperatorSession
from tests.conftest import OPERATOR_HEADERS
from tests.test_phase4.test_hackerone import (
    MockHackerOne,
    _provider,
    _session,
    _verified_finding,
)


@pytest.mark.asyncio
async def test_fabricated_verified_finding_cannot_create_draft(client: httpx.AsyncClient) -> None:
    reset_hackerone_provider(_provider()[0])
    response = await client.post(
        "/api/v1/hackerone/reports/drafts",
        headers=OPERATOR_HEADERS,
        json={
            "program_handle": "demo",
            "finding": {
                "title": "Fabricated",
                "status": "verified",
                "target": "https://demo.example/",
                "impact": "made up",
                "reproduction": "none",
                "evidence": [
                    {"kind": "reproduction", "source": "attacker", "summary": "I said so"}
                ],
            },
        },
    )
    assert response.status_code == 422
    reset_hackerone_provider(None)


@pytest.mark.asyncio
async def test_persisted_potential_finding_cannot_create_draft(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    project = Project(name="p", repository_path="/tmp/p")
    db_session.add(project)
    await db_session.flush()
    potential = SecurityFinding.potential(
        "maybe xss",
        description="static hint",
        target="https://demo.example/",
        vulnerability_class="xss",
    )
    await SecurityFindingRepository(db_session).bulk_create(
        [potential], project_id=project.id, analysis_id=None
    )
    await db_session.commit()
    provider, _ = _provider()
    provider.sync_scope("demo")
    reset_hackerone_provider(provider)
    response = await client.post(
        "/api/v1/hackerone/reports/drafts",
        headers=OPERATOR_HEADERS,
        json={
            "program_handle": "demo",
            "finding_id": str(potential.id),
            "project_id": str(project.id),
            "severity": "high",
        },
    )
    assert response.status_code == 400
    assert "verified" in response.json()["detail"].lower()
    reset_hackerone_provider(None)


@pytest.mark.asyncio
async def test_finding_from_other_project_is_rejected(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    project_a = Project(name="a", repository_path="/tmp/a")
    project_b = Project(name="b", repository_path="/tmp/b")
    db_session.add_all([project_a, project_b])
    await db_session.flush()
    finding = _verified_finding()
    await SecurityFindingRepository(db_session).bulk_create(
        [finding], project_id=project_a.id, analysis_id=None
    )
    await db_session.commit()
    provider, _ = _provider()
    provider.sync_scope("demo")
    reset_hackerone_provider(provider)
    response = await client.post(
        "/api/v1/hackerone/reports/drafts",
        headers=OPERATOR_HEADERS,
        json={
            "program_handle": "demo",
            "finding_id": str(finding.id),
            "project_id": str(project_b.id),
            "severity": "high",
            "weakness_id": 639,
        },
    )
    assert response.status_code == 403
    reset_hackerone_provider(None)


@pytest.mark.asyncio
async def test_operator_string_does_not_approve_without_token(client: httpx.AsyncClient) -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    reset_hackerone_provider(provider)
    response = await client.post(
        "/api/v1/hackerone/reports/x/approve",
        json={"operator": "researcher", "program_handle": "demo"},
    )
    assert response.status_code == 401
    reset_hackerone_provider(None)


@pytest.mark.asyncio
async def test_draft_and_submission_persist_across_restart(db_session: AsyncSession) -> None:
    provider, mock = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    await SecurityFindingRepository(db_session).bulk_create(
        [finding], project_id=None, analysis_id=None
    )
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    provider.reports.mark_ready(draft.id, provider.programs["demo"], finding=finding)
    provider.reports.human_approve(
        draft.id, provider.programs["demo"], session=_session(), finding=finding
    )
    provider.reports.submit(
        draft.id, provider.programs["demo"], finding=finding, session=_session()
    )
    repo = HackerOneRepository(db_session)
    await repo.persist_provider(provider)
    await db_session.flush()

    restarted = HackerOneProvider(credentials=provider.credentials, client=provider.client)
    await repo.hydrate(restarted)
    loaded = restarted.reports.drafts[draft.id]
    assert loaded.hackerone_report_id == "r-100"
    assert loaded.submission_state is ReportSubmissionState.SUBMITTED
    assert loaded.approval is not None
    assert loaded.scope_snapshot is not None
    with pytest.raises(HackerOneError):
        restarted.reports.submit(
            draft.id, restarted.programs["demo"], finding=finding, session=_session()
        )
    assert mock.created_reports  # original submit only


@pytest.mark.asyncio
async def test_failed_submission_persists(db_session: AsyncSession) -> None:
    mock = MockHackerOne()
    provider, _ = _provider(mock)
    provider.sync_scope("demo")
    finding = _verified_finding()
    await SecurityFindingRepository(db_session).bulk_create(
        [finding], project_id=None, analysis_id=None
    )
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    provider.reports.human_approve(
        draft.id, provider.programs["demo"], session=_session(), finding=finding
    )
    mock.mode = "422_severity"
    result = provider.reports.submit(
        draft.id, provider.programs["demo"], finding=finding, session=_session()
    )
    assert result.submission_state is ReportSubmissionState.SUBMISSION_FAILED
    repo = HackerOneRepository(db_session)
    await repo.persist_provider(provider)
    await db_session.flush()
    restarted = HackerOneProvider(credentials=provider.credentials, client=provider.client)
    await repo.hydrate(restarted)
    assert (
        restarted.reports.drafts[draft.id].submission_state
        is ReportSubmissionState.SUBMISSION_FAILED
    )
    assert restarted.reports.drafts[draft.id].hackerone_report_id is None


@pytest.mark.asyncio
async def test_reconciliation_persists_remote_state(db_session: AsyncSession) -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    await SecurityFindingRepository(db_session).bulk_create(
        [finding], project_id=None, analysis_id=None
    )
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    provider.reports.human_approve(
        draft.id, provider.programs["demo"], session=_session(), finding=finding
    )
    submitted = provider.reports.submit(
        draft.id, provider.programs["demo"], finding=finding, session=_session()
    )
    reconciled = provider.reports.reconcile(submitted.id)
    assert reconciled.remote_state.value == "new"
    repo = HackerOneRepository(db_session)
    await repo.persist_provider(provider)
    await db_session.flush()
    restarted = HackerOneProvider(credentials=provider.credentials, client=provider.client)
    await repo.hydrate(restarted)
    assert restarted.reports.drafts[draft.id].remote_state.value == "new"


def test_json_api_payload_uses_attribute_integer_ids() -> None:
    provider, mock = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding(vulnerability_class="sql_injection")
    draft = provider.draft_from_finding(finding, "demo", severity="high")
    provider.reports.human_approve(
        draft.id, provider.programs["demo"], session=_session(), finding=finding
    )
    provider.reports.submit(
        draft.id, provider.programs["demo"], finding=finding, session=_session()
    )
    payload = mock.created_reports[0]
    assert payload == {
        "data": {
            "type": "report",
            "attributes": {
                "team_handle": "demo",
                "title": finding.title,
                "vulnerability_information": draft.vulnerability_information,
                "impact": finding.impact,
                "severity_rating": "high",
                "weakness_id": 1338,
                "structured_scope_id": 287,
            },
        }
    }


def test_scope_exclusions_are_not_target_denials() -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    program = provider.programs["demo"]
    decision = provider.scope_provider.evaluator.evaluate(
        program, "https://demo.example/login", vulnerability_class="xss"
    )
    assert decision.in_scope is True
    assert decision.target_scope is not None and decision.target_scope.in_scope is True
    dos = provider.scope_provider.evaluator.evaluate(
        program, "https://demo.example/login", vulnerability_class="denial_of_service"
    )
    assert dos.in_scope is True
    assert dos.eligible_for_submission is True
    assert dos.eligible_for_bounty is False
    assert dos.report_eligibility is not None
    assert "Denial of Service" in dos.report_eligibility.exclusion_categories


def test_weakness_requires_human_when_program_missing_cwe() -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    program = provider.programs["demo"]
    program.weaknesses = tuple(item for item in program.weaknesses if item.external_id != "cwe-89")
    mapped = map_program_weakness("sql_injection", program)
    assert mapped.requires_human_selection is True
    assert mapped.selected_id is None


def test_vulnerability_information_includes_reproduction() -> None:
    finding = _verified_finding()
    text = compose_vulnerability_information(finding, target=finding.target or "")
    assert "## Steps to reproduce" in text
    assert "GET /users/3 as user 1" in text
    assert "## Supporting evidence" in text
    assert "## Affected asset" in text


def test_client_rejects_attacker_origin() -> None:
    client = HackerOneApiClient(
        __import__(
            "app.adapters.hackerone.credentials", fromlist=["HackerOneCredentials"]
        ).HackerOneCredentials(username="u", _token="t", base_url="https://api.hackerone.com/v1"),
        transport=httpx.MockTransport(MockHackerOne()),
        min_interval_seconds=0,
    )
    with pytest.raises(HackerOneUrlRejectedError):
        client.get("https://attacker.example/")


def test_pagination_continues_with_id_gt() -> None:
    class Paged(MockHackerOne):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if "structured_scopes" in request.url.path and request.method == "GET":
                if request.url.params.get("filter[id__gt]"):
                    self.id_gt_seen.append(str(request.url.params.get("filter[id__gt]")))
                    return httpx.Response(
                        200,
                        json={
                            "data": [
                                {
                                    "id": "10001",
                                    "attributes": {
                                        "asset_type": "Domain",
                                        "asset_identifier": "more.demo.example",
                                        "eligible_for_submission": True,
                                        "eligible_for_bounty": True,
                                    },
                                }
                            ]
                        },
                    )
                data = [
                    {
                        "id": str(i),
                        "attributes": {
                            "asset_type": "Domain",
                            "asset_identifier": f"host{i}.demo.example",
                            "eligible_for_submission": True,
                            "eligible_for_bounty": True,
                        },
                    }
                    for i in range(1, 101)
                ]
                return httpx.Response(
                    200,
                    json={
                        "data": data,
                        "links": {
                            "next": "https://api.hackerone.com/v1/hackers/programs/demo/structured_scopes?page=2"
                        },
                    },
                )
            return super().__call__(request)

    mock = Paged()
    provider, _ = _provider(mock)
    # Force paginate_with_meta page_cap low by calling client directly.
    rows, meta = provider.client.paginate_with_meta(
        "hackers/programs/demo/structured_scopes", page_cap=1
    )
    assert any(item.get("id") == "10001" for item in rows)
    assert mock.id_gt_seen
    assert meta["count"] >= 101


def test_operator_session_rejects_ai() -> None:
    session = OperatorSession(
        identity="assistant", authenticated_at=datetime.now(UTC), source="operator_token"
    )
    with pytest.raises(Exception):
        session.assert_human()
