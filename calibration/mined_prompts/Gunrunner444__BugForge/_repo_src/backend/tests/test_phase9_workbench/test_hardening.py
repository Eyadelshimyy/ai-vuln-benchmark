"""Phase 9 PART A hardening: orchestrator, identities, replay, memory, export, handoff."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.mock_provider import MockLLMProvider
from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind
from app.domain.findings import FindingStatus, SecurityFinding
from app.models.security_agent import DBResearchIdentity, DBResearchMemory
from app.repositories.security_agent_repo import SecurityAgentRepository
from app.security_agent.agent import ResearchSession, SecurityResearchAgent
from app.security_agent.authorization_diff import compare_authorization
from app.security_agent.export import FindingNotFoundError, export_package
from app.security_agent.handoff import prepare_hackerone_handoff
from app.security_agent.identities import IdentityPair, ResearchIdentity
from app.security_agent.memory import ResearchMemory
from app.security_agent.oracles import AuthorizationOracle
from app.security_agent.orchestrator import AdvancedResearchOrchestrator
from app.security_agent.replay import SessionReplay
from app.security_agent.schemas import ResearchHypothesis
from app.security_agent.states import EvidenceCompleteness, HypothesisStatus, ResearchMode
from app.security_agent.strategies import ResearchStrategy, spec_for
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import RestrictedActivityError
from app.security_testing.safety import SafetyLimits
from tests.conftest import OPERATOR_HEADERS


def _session(**kwargs: object) -> ResearchSession:
    engine = SecurityTestEngine.lab(
        "lab",
        hosts=("127.0.0.1", "localhost"),
        allow_active_testing=True,
        limits=SafetyLimits.lab(),
    )
    values = dict(
        project_id="lab",
        target="http://127.0.0.1/health",
        mode=ResearchMode.LAB,
        engine=engine,
        provider=MockLLMProvider(),
    )
    values.update(kwargs)
    return ResearchSession(**values)  # type: ignore[arg-type]


def test_supporting_ids_are_not_successful_completion() -> None:
    session = _session()
    graph = session.graph
    node = graph.add(kind="observation", provenance="execution", summary="200", node_id="e1")
    hyp = ResearchHypothesis(
        title="maybe idor",
        vulnerability_class="idor",
        target="http://127.0.0.1/users/2",
        reason="guess",
        supporting_evidence_ids=(node.id,),
        status=HypothesisStatus.SUPPORTED,
    )
    session.hypotheses.append(hyp)
    graph.add(
        kind="hypothesis",
        provenance="ai_hypothesis",
        summary=hyp.title,
        node_id=hyp.id,
        extra={"hypothesis_id": hyp.id},
    )
    graph.link(hyp.id, node.id, "supports")
    orch = AdvancedResearchOrchestrator(SecurityResearchAgent(session))
    assert orch.evidence_sufficient() is False
    assert orch.classify_hypothesis(hyp) is EvidenceCompleteness.REPRODUCTION_REQUIRED
    assert any(item.endswith(":reproduction") for item in orch.identify_missing_evidence())


def test_identify_missing_evidence_checks_existence_link_and_provenance() -> None:
    session = _session()
    hyp = ResearchHypothesis(
        title="xss",
        vulnerability_class="xss",
        target="http://127.0.0.1/search",
        reason="maybe",
        supporting_evidence_ids=("missing", "ai1", "e1"),
        contradicting_evidence_ids=("c1",),
        status=HypothesisStatus.SUPPORTED,
    )
    session.hypotheses.append(hyp)
    session.graph.add(
        kind="observation", provenance="ai_hypothesis", summary="model said so", node_id="ai1"
    )
    session.graph.add(kind="observation", provenance="execution", summary="reflected", node_id="e1")
    session.graph.add(
        kind="observation", provenance="execution", summary="not reflected", node_id="c1"
    )
    orch = AdvancedResearchOrchestrator(SecurityResearchAgent(session))
    missing = orch.identify_missing_evidence()
    blob = " ".join(missing)
    assert "missing_id:missing" in blob
    assert "non_live_provenance:ai1" in blob
    assert "unlinked:e1" in blob
    assert "contradiction" in blob
    assert "reproduction" in blob


def test_identities_isolated_even_with_identical_cookies() -> None:
    pair = IdentityPair(
        context_a=ResearchIdentity(label="A", cookies={"sid": "same"}),
        context_b=ResearchIdentity(label="B", cookies={"sid": "same"}),
    )
    assert pair.isolated()
    assert pair.context_a.id != pair.context_b.id
    assert pair.context_a.browser_context_id != pair.context_b.browser_context_id
    assert pair.context_a.http_session_id != pair.context_b.http_session_id
    assert pair.context_a.storage_namespace != pair.context_b.storage_namespace


def test_identity_snapshot_omits_raw_secrets() -> None:
    ident = ResearchIdentity(
        label="A",
        cookies={"session": "lab-session"},
        headers={"Authorization": "Bearer supersecret-token"},
        storage={"token": "refresh-me"},
    )
    snap = ident.snapshot()
    blob = json.dumps(snap)
    assert "supersecret-token" not in blob
    assert "lab-session" not in blob
    assert "refresh-me" not in blob
    assert "Authorization" in snap["header_names"]
    assert "session" in snap["cookie_names"]


def test_restore_headers_fail_closed_without_secrets() -> None:
    ident = ResearchIdentity(
        label="A",
        header_names=("Authorization",),
        header_secret_refs={"Authorization": "missing-ref"},
        authentication_state="authenticated",
        credential_provenance="operator_provided",
    )
    with pytest.raises(RestrictedActivityError, match="identity_credentials_unavailable"):
        ident.restore_headers(None)


def test_authorization_oracle_is_required_for_meaningful_hypothesis() -> None:
    result = compare_authorization(
        {
            "url": "http://127.0.0.1/item/1",
            "status": 200,
            "response": {"body": '{"id":"1","ts":"2020-01-01T00:00:00Z"}'},
        },
        {
            "url": "http://127.0.0.1/item/1",
            "status": 200,
            "response": {"body": '{"id":"1","ts":"2021-02-02T00:00:00Z"}'},
        },
        expectation="",
        oracle=AuthorizationOracle.OWNER_ONLY,
    )
    assert result.is_vulnerability is False
    assert result.oracle == AuthorizationOracle.OWNER_ONLY.value
    assert result.status_difference == ""


def test_authorization_normalizes_dynamic_fields() -> None:
    result = compare_authorization(
        {
            "url": "http://127.0.0.1/item/1",
            "status": 200,
            "response": {"body": '{"id":"1","request_id":"aaa","owner":"alice"}'},
        },
        {
            "url": "http://127.0.0.1/item/1",
            "status": 200,
            "response": {"body": '{"id":"1","request_id":"bbb","owner":"alice"}'},
        },
        expectation="Only the owner may read the object",
        oracle=AuthorizationOracle.OWNER_ONLY,
    )
    assert "no meaningful difference" in result.difference
    assert result.is_vulnerability is False


@pytest.mark.asyncio
async def test_replay_restores_original_executors() -> None:
    session = _session()
    agent = SecurityResearchAgent(session)

    async def real(_arguments: dict[str, object]) -> dict[str, object]:
        return {"quality": "success", "executed": True, "live": True}

    agent.tools.replace_executor("source_inspect", real)
    original = agent.tools.executor_for("source_inspect")
    during: list[object] = []
    native_replace = agent.tools.replace_executor

    def tracking_replace(name: str, executor: object) -> None:
        native_replace(name, executor)  # type: ignore[arg-type]
        if name == "source_inspect":
            during.append(agent.tools.executor_for(name))

    agent.tools.replace_executor = tracking_replace  # type: ignore[method-assign]
    replay = SessionReplay()
    replay.record("source_inspect", {"path": "app.py"}, {"quality": "success", "executed": True})
    await replay.play(agent)
    assert during
    assert during[0] is not original
    assert agent.tools.executor_for("source_inspect") is original
    assert session.replay_mode is False
    observations = [node for node in session.graph.nodes.values() if node.kind == "observation"]
    assert observations
    assert all(node.provenance == "replay" for node in observations)


@pytest.mark.asyncio
async def test_replay_cannot_enable_live_network() -> None:
    session = _session()
    agent = SecurityResearchAgent(session)
    replay = SessionReplay(live_network=True)
    with pytest.raises(RestrictedActivityError, match="offline"):
        await replay.play(agent)
    replay.live_network = True
    with pytest.raises(RestrictedActivityError, match="offline"):
        await replay.play(agent)


def test_memory_recursively_sanitizes_and_preserves_ids() -> None:
    memory = ResearchMemory(project_id="lab")
    nested = {
        "ok": "value",
        "password": "hunter2",
        "child": {"authorization": "Bearer abc", "items": [{"token": "xyz", "n": 1}]},
        "tuple_like": ["session=abc", "plain"],
    }
    entry = memory.remember("note", "summary password=hunter2", nested)
    assert entry.extra["password"] == "[REDACTED]"
    assert entry.extra["child"]["authorization"] == "[REDACTED]"
    assert "hunter2" not in json.dumps(entry.snapshot())
    restored = ResearchMemory(project_id="lab")
    restored.restore_entry(
        entry_id=entry.id, kind=entry.kind, summary=entry.summary, extra=entry.extra
    )
    assert restored.entries[0].id == entry.id


def test_strategy_is_more_than_tool_names() -> None:
    spec = spec_for(ResearchStrategy.AUTHORIZATION)
    assert spec.goal
    assert spec.preferred_oracle
    assert spec.reproduction_requirement is True
    assert spec.minimum_evidence_expectations
    assert spec.stop_conditions


def test_export_specific_finding_and_unknown_404() -> None:
    session = _session()
    from app.domain.trusted_evidence import issue_for_finding

    shell = SecurityFinding.potential(
        "Verified IDOR", vulnerability_class="idor", target="http://127.0.0.1/users/2"
    )
    evidence = issue_for_finding(
        shell,
        Evidence(
            kind=EvidenceKind.HTTP_RESPONSE,
            source="lab",
            summary="reproduced IDOR",
            details="shown",
        ),
        "exec-export",
    )
    finding = SecurityFinding.verified(
        "Verified IDOR",
        evidence=EvidenceBundle.from_items([evidence]),
        vulnerability_class="idor",
        target="http://127.0.0.1/users/2",
    )
    other = SecurityFinding.potential("other", target="http://127.0.0.1/x")
    session.findings.extend([other, finding])
    package = export_package(session, finding_id=str(finding.id))
    assert package["finding"]["id"] == str(finding.id)
    whole = export_package(session)
    assert whole["finding"] is None
    assert len(whole["findings"]) == 2
    with pytest.raises(FindingNotFoundError):
        export_package(session, finding_id="missing")


def test_handoff_uses_evaluator_not_substring() -> None:
    session = _session()
    from app.domain.trusted_evidence import issue_for_finding

    shell = SecurityFinding.potential(
        "Verified IDOR", vulnerability_class="idor", target="http://127.0.0.1/users/2"
    )
    evidence = issue_for_finding(
        shell,
        Evidence(
            kind=EvidenceKind.HTTP_RESPONSE,
            source="lab",
            summary="reproduced IDOR",
            details="shown",
        ),
        "exec-handoff",
    )
    finding = SecurityFinding.verified(
        "Verified IDOR",
        evidence=EvidenceBundle.from_items([evidence]),
        vulnerability_class="idor",
        target="http://127.0.0.1/users/2",
    )
    session.findings.append(finding)
    payload = prepare_hackerone_handoff(session, finding)
    assert "evaluation_reason" in payload
    assert "matched_structured_scope_id" in payload
    assert payload["cannot_submit"] is True


@pytest.mark.asyncio
async def test_identity_secrets_absent_from_database(db_session: AsyncSession) -> None:
    session = _session()
    session.identities = IdentityPair(
        context_a=ResearchIdentity(
            label="A",
            cookies={"session": "lab-session"},
            headers={"Authorization": "Bearer supersecret-token"},
        ),
        context_b=ResearchIdentity(label="B", cookies={"session": "other-session"}),
    )
    repo = SecurityAgentRepository(db_session)
    await repo.save_session(session)
    await db_session.commit()
    row = (
        (
            await db_session.execute(
                select(DBResearchIdentity).where(DBResearchIdentity.session_id == session.id)
            )
        )
        .scalars()
        .all()
    )
    blob = json.dumps(
        [
            {
                "id": item.id,
                "label": item.label,
                "header_names": item.header_names,
                "header_secret_refs": item.header_secret_refs,
                "cookie_names": item.cookie_names,
                "cookie_secret_ref": item.cookie_secret_ref,
            }
            for item in row
        ]
    )
    assert "supersecret-token" not in blob
    assert "Bearer" not in blob
    assert "lab-session" not in blob
    assert "cookies" not in DBResearchIdentity.__table__.columns


@pytest.mark.asyncio
async def test_memory_ids_round_trip(db_session: AsyncSession) -> None:
    session = _session()
    memory = ResearchMemory(project_id="lab")
    entry = memory.remember("durable", "normalized fact", {"path": "/users/2"})
    session.memory = memory
    repo = SecurityAgentRepository(db_session)
    await repo.save_session(session)
    await db_session.commit()
    restored = await repo.reconstruct(session.id)
    assert restored is not None
    assert restored.session.memory is not None
    assert restored.session.memory.entries[0].id == entry.id
    stored = (
        await db_session.execute(
            select(DBResearchMemory).where(DBResearchMemory.session_id == session.id)
        )
    ).scalar_one()
    assert stored.id == entry.id


@pytest.mark.asyncio
async def test_verified_without_chain_is_downgraded(db_session: AsyncSession) -> None:
    session = _session()
    finding = SecurityFinding.potential("claimed", target="http://127.0.0.1/x")
    session.findings.append(finding)
    repo = SecurityAgentRepository(db_session)
    await repo.save_session(session)
    await db_session.commit()
    from app.models.security_agent import DBResearchFinding

    row = await db_session.get(DBResearchFinding, str(finding.id))
    assert row is not None
    row.status = FindingStatus.VERIFIED.value
    row.verification_state = FindingStatus.VERIFIED.value
    row.extra = {"evidence_items": []}
    await db_session.commit()
    restored = await repo.reconstruct(session.id)
    assert restored is not None
    assert restored.session.findings[0].status is FindingStatus.POTENTIAL
    assert "restored verifying evidence" not in str(restored.session.findings[0].evidence.items)


@pytest.mark.asyncio
async def test_export_endpoint_selects_finding(client) -> None:
    created = await client.post(
        "/api/v1/security-agent/sessions",
        headers=OPERATOR_HEADERS,
        json={"project_id": "lab", "target": "http://127.0.0.1/health", "mode": "lab"},
    )
    assert created.status_code == 200
    session_id = created.json()["id"]
    missing = await client.get(
        f"/api/v1/security-agent/sessions/{session_id}/export",
        headers=OPERATOR_HEADERS,
        params={"finding_id": "no-such-finding"},
    )
    assert missing.status_code == 404
    whole = await client.get(
        f"/api/v1/security-agent/sessions/{session_id}/export",
        headers=OPERATOR_HEADERS,
    )
    assert whole.status_code == 200
    assert whole.json()["finding"] is None


def test_operator_ownership_is_enforced() -> None:
    from datetime import UTC, datetime

    from fastapi import HTTPException

    from app.api.v1.endpoints.security_agent import _assert_operator_owns
    from app.security_testing.operator_auth import OperatorSession

    alice = OperatorSession(identity="alice", authenticated_at=datetime.now(UTC))
    bob = OperatorSession(identity="bob", authenticated_at=datetime.now(UTC))
    _assert_operator_owns("alice", alice)
    with pytest.raises(HTTPException) as exc:
        _assert_operator_owns("alice", bob)
    assert exc.value.status_code == 403
    assert exc.value.detail == "session_operator_mismatch"
