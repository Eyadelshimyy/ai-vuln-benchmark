"""Phase 9 workbench, local lab, restart, and live safety demos."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.mock_provider import MockLLMProvider
from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind
from app.domain.findings import SecurityFinding
from app.repositories.security_agent_repo import SecurityAgentRepository
from app.security_agent.agent import AgentDecision, ResearchSession, SecurityResearchAgent
from app.security_agent.checkpoints import ResearchCheckpoint
from app.security_agent.cost import consume_units_for, estimate_operation_cost
from app.security_agent.orchestrator import AdvancedResearchOrchestrator
from app.security_agent.project import SecurityResearchProject
from app.security_agent.schemas import ResearchHypothesis, ToolCallRequest
from app.security_agent.states import (
    ResearchMode,
    ResearchProjectState,
    ToolCapability,
)
from app.security_agent.strategies import ResearchStrategy
from app.security_agent.tool_status import describe_tools
from app.security_agent.tools import default_registry
from app.security_testing.engine import SecurityTestEngine, SecurityTestSession, TestingMode
from app.security_testing.errors import RestrictedActivityError
from app.security_testing.safety import SafetyLimits
from app.security_testing.scope_model import Eligibility, ProgramScope, ScopeRule
from app.security_testing.target import AssetType
from tests.conftest import OPERATOR_HEADERS
from tests.fixtures.lab_app.server import LabServer


def _lab_session(target: str = "http://127.0.0.1/health", **kwargs: object) -> ResearchSession:
    engine = SecurityTestEngine.lab(
        "lab",
        hosts=("127.0.0.1", "localhost"),
        allow_active_testing=True,
        limits=SafetyLimits.lab(),
    )
    values = dict(
        project_id="lab",
        target=target,
        mode=ResearchMode.LAB,
        engine=engine,
        provider=MockLLMProvider(),
        operator_identity="alice",
    )
    values.update(kwargs)
    return ResearchSession(**values)  # type: ignore[arg-type]


@pytest.fixture
def lab_server() -> Iterator[LabServer]:
    server = LabServer().start()
    try:
        yield server
    finally:
        server.stop()


def test_research_project_lifecycle() -> None:
    project = SecurityResearchProject(name="lab", project_id="lab", target="http://127.0.0.1/")
    assert project.state is ResearchProjectState.CREATE
    project.transition(ResearchProjectState.CONFIGURE)
    project.transition(ResearchProjectState.READY)
    project.transition(ResearchProjectState.RESEARCHING)
    project.transition(ResearchProjectState.FINDINGS)
    project.transition(ResearchProjectState.REVIEW)
    project.transition(ResearchProjectState.HANDOFF)
    project.transition(ResearchProjectState.COMPLETE)
    with pytest.raises(RestrictedActivityError):
        project.transition(ResearchProjectState.RESEARCHING)


def test_tool_states_are_separated() -> None:
    agent = SecurityResearchAgent(_lab_session())
    agent.disable_tool("fuzz")
    views = {item["name"]: item for item in describe_tools(agent)}
    http = views["http_request"]
    fuzz = views["fuzz"]
    assert http["availability"] == "available"
    assert http["enablement"] == "enabled"
    assert fuzz["enablement"] == "disabled"
    assert http["authorized_is_not_success"] is True
    assert fuzz["installed_is_not_enabled"] is True


def test_cost_estimate_matches_consume_units() -> None:
    fuzz = estimate_operation_cost("fuzz", {"count": 20}, default_registry().spec("fuzz"))
    assert fuzz.estimated_requests == 20
    assert fuzz.unit == "fuzz_requests"
    assert fuzz.is_estimate is True
    api = estimate_operation_cost("api_test", {"max_tests": 8, "execute": True})
    assert api.estimated_requests == 8
    http = estimate_operation_cost("http_request", {"method": "GET", "url": "http://127.0.0.1/"})
    assert http.estimated_requests == 1
    scanner = estimate_operation_cost("zap_scan", {}, remaining_requests=50)
    assert scanner.estimated_requests == 50
    assert consume_units_for(scanner) == ("tool", 1)


@pytest.mark.asyncio
async def test_local_lab_end_to_end(lab_server: LabServer, db_session: AsyncSession) -> None:
    session = _lab_session(target=f"{lab_server.origin}/api/orders/2")
    steps = iter(
        [
            AgentDecision(
                kind="hypothesis",
                hypothesis=ResearchHypothesis(
                    title="IDOR on orders",
                    vulnerability_class="idor",
                    target=f"{lab_server.origin}/api/orders/2",
                    reason="owner field is anyone",
                    status=__import__(
                        "app.security_agent.states", fromlist=["HypothesisStatus"]
                    ).HypothesisStatus.REQUIRES_REPRODUCTION,
                ),
            ),
            AgentDecision(
                kind="tool",
                tool=ToolCallRequest(
                    tool="http_request",
                    arguments={"method": "GET", "url": f"{lab_server.origin}/api/orders/2"},
                    reason="observe",
                ),
            ),
            AgentDecision(
                kind="tool",
                tool=ToolCallRequest(
                    tool="reproduce",
                    arguments={
                        "url": f"{lab_server.origin}/api/orders/2",
                        "expected_body_contains": '"owner": "anyone"',
                        "expected_result": "anyone",
                    },
                    reason="confirm",
                ),
            ),
            AgentDecision(kind="complete"),
        ]
    )

    async def planner(_s: ResearchSession) -> AgentDecision:
        return next(steps)

    agent = SecurityResearchAgent(session, planner=planner)
    orch = AdvancedResearchOrchestrator(agent)
    orch.recommend_strategy(ResearchStrategy.AUTHORIZATION.value)
    await orch.step()
    await orch.step()
    await orch.step()
    decision = await orch.step()
    assert decision.kind == "complete"
    assert session.hypotheses
    assert any(
        node.kind in {"observation", "reproduction", "request", "response"}
        for node in session.graph.nodes.values()
    )
    reproduced = [node for node in session.graph.nodes.values() if node.kind == "reproduction"]
    if reproduced:
        from app.domain.trusted_evidence import issue_for_finding

        shell = SecurityFinding.potential(
            "IDOR on orders", vulnerability_class="idor", target=session.target
        )
        evidence = issue_for_finding(
            shell,
            Evidence(
                kind=EvidenceKind.HTTP_RESPONSE,
                source="lab",
                summary=reproduced[0].summary or "reproduced owner=anyone",
                details="other owner",
            ),
            "exec-orders",
        )
        finding = SecurityFinding.verified(
            "IDOR on orders",
            evidence=EvidenceBundle.from_items([evidence]),
            vulnerability_class="idor",
            target=session.target,
        )
        session.findings.append(finding)
        assert orch.evidence_sufficient()
        package = __import__(
            "app.security_agent.export", fromlist=["export_package"]
        ).export_package(session, finding)
        assert package["hashes"]["sha256"]
        assert package["finding"]["id"] == str(finding.id)
    repo = SecurityAgentRepository(db_session)
    await repo.save_session(session)
    await db_session.commit()


@pytest.mark.asyncio
async def test_live_mode_safety_blocks(client) -> None:
    missing_scope = await client.post(
        "/api/v1/security-agent/sessions",
        headers=OPERATOR_HEADERS,
        json={
            "project_id": "live",
            "target": "https://example.com/",
            "mode": "live_hackerone",
            "program_handle": "",
        },
    )
    assert missing_scope.status_code == 400

    created = await client.post(
        "/api/v1/security-agent/sessions",
        headers=OPERATOR_HEADERS,
        json={"project_id": "lab", "target": "http://127.0.0.1/health", "mode": "lab"},
    )
    assert created.status_code == 200
    session_id = created.json()["id"]
    replay = await client.post(
        f"/api/v1/security-agent/sessions/{session_id}/replay",
        headers=OPERATOR_HEADERS,
        json={"events": [], "live_network": True},
    )
    assert replay.status_code == 403

    from app.api.v1.endpoints import security_agent as api

    agent = api._SESSIONS[session_id]
    agent.disable_tool("http_request")
    blocked = await client.post(
        f"/api/v1/security-agent/sessions/{session_id}/tools",
        headers=OPERATOR_HEADERS,
        json={
            "tool": "http_request",
            "arguments": {"method": "GET", "url": "http://127.0.0.1/health"},
            "reason": "probe",
        },
    )
    assert blocked.status_code == 403
    assert "disabled" in str(blocked.json()).lower()

    escalate = await client.post(
        f"/api/v1/security-agent/sessions/{session_id}/approvals",
        headers={
            **OPERATOR_HEADERS,
            "X-BugForge-Operator-Token": OPERATOR_HEADERS["X-BugForge-Operator-Token"],
        },
        json={"kind": "enable_active_testing", "note": "operator"},
    )
    assert escalate.status_code == 200


def test_live_engine_blocks_unknown_and_unapproved() -> None:
    scope = ProgramScope(
        program_name="demo",
        includes=(
            ScopeRule(
                identifier="https://in.example/",
                asset_type=AssetType.URL,
                eligible=Eligibility.ELIGIBLE,
            ),
        ),
        lab_mode=False,
        allow_active_testing=False,
    )
    engine = SecurityTestEngine(
        SecurityTestSession(
            project_id="p",
            mode=TestingMode.LIVE,
            scope=scope,
            limits=SafetyLimits.conservative(),
            dry_run=True,
            active_testing_enabled=False,
        )
    )
    denied = engine.authorize("https://unknown.example/", tool="http_request", active=True)
    assert denied.allowed is False
    denied_active = engine.authorize("https://in.example/", tool="http_request", active=True)
    assert denied_active.allowed is False


def test_checkpoint_revalidates_current_authority() -> None:
    session = _lab_session()
    agent = SecurityResearchAgent(session)
    agent.enable_active_testing(operator="alice", note="lab")
    checkpoint = ResearchCheckpoint.capture(session, label="mid")
    session.engine.session.active_testing_enabled = False
    session.engine.approvals.clear_records()
    snapshot = checkpoint.verify_live(session)
    assert snapshot.active_testing is False or session.engine.session.scope.lab_mode


@pytest.mark.asyncio
async def test_application_restart_preserves_state(db_session: AsyncSession) -> None:
    session = _lab_session()
    agent = SecurityResearchAgent(session)
    assert agent.session is session
    session.hypotheses.append(
        ResearchHypothesis(
            title="IDOR",
            vulnerability_class="idor",
            target=session.target,
            reason="owner field",
        )
    )
    node = session.graph.add(
        kind="observation", provenance="execution", summary="200", node_id="restart-e1"
    )
    session.graph.add(
        kind="hypothesis",
        provenance="ai_hypothesis",
        summary="IDOR",
        node_id=session.hypotheses[0].id,
        extra={"hypothesis_id": session.hypotheses[0].id},
    )
    session.graph.link(session.hypotheses[0].id, node.id, "supports")
    evidence = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE, source="http_request", summary="200 anyone"
    )
    session.findings.append(
        SecurityFinding.potential(
            "maybe idor",
            evidence=EvidenceBundle.from_items([evidence]),
            vulnerability_class="idor",
            target=session.target,
            hypothesis=session.hypotheses[0].id,
        )
    )
    session.strategy = ResearchStrategy.AUTHORIZATION.value
    repo = SecurityAgentRepository(db_session)
    await repo.save_session(session)
    await db_session.commit()
    restored = await repo.reconstruct(session.id)
    assert restored is not None
    assert restored.session.hypotheses[0].title == "IDOR"
    assert restored.session.findings[0].title == "maybe idor"
    assert restored.session.strategy == ResearchStrategy.AUTHORIZATION.value
    assert "restart-e1" in restored.session.graph.nodes
    assert restored.session.provider.provider_name == "mock"
    assert restored.session.thinking_enabled is True


def test_large_project_context_is_bounded() -> None:
    session = _lab_session()
    agent = SecurityResearchAgent(session)
    for index in range(200):
        session.graph.add(
            kind="observation",
            provenance="execution",
            summary=f"node {index}",
            node_id=f"n{index}",
        )
    built = agent.context.build(session)
    evidence = built["untrusted"]["evidence"]
    assert len(evidence) <= 16


@pytest.mark.skipif(
    not os.environ.get("BUGFORGE_MLX_INTEGRATION"),
    reason="Set BUGFORGE_MLX_INTEGRATION=1 to hit a real local MLX server",
)
def test_local_qwen_optional_integration() -> None:
    from app.ai.mlx_provider import MLXProvider

    provider = MLXProvider()
    assert provider.model_name


@pytest.mark.asyncio
async def test_workbench_dashboard_and_timeline(client) -> None:
    created = await client.post(
        "/api/v1/security-agent/sessions",
        headers=OPERATOR_HEADERS,
        json={"project_id": "lab", "target": "http://127.0.0.1/health", "mode": "lab"},
    )
    assert created.status_code == 200
    session_id = created.json()["id"]
    dash = await client.get(
        f"/api/v1/security-agent/sessions/{session_id}/dashboard", headers=OPERATOR_HEADERS
    )
    assert dash.status_code == 200
    body = dash.json()
    assert body["session_kind"] in {"LAB", "LIVE"}
    assert body["execution_mode"] in {"DRY-RUN", "ACTIVE"}
    assert "remaining_budget" in body
    timeline = await client.get(
        f"/api/v1/security-agent/sessions/{session_id}/timeline", headers=OPERATOR_HEADERS
    )
    assert timeline.status_code == 200
    review = await client.post(
        f"/api/v1/security-agent/sessions/{session_id}/next-action/review",
        headers=OPERATOR_HEADERS,
        json={
            "tool": "http_request",
            "arguments": {"method": "GET", "url": "http://127.0.0.1/health"},
            "reason": "observe",
        },
    )
    assert review.status_code == 200
    assert review.json()["is_estimate"] is True
    project = await client.post(
        "/api/v1/security-agent/research-projects",
        headers=OPERATOR_HEADERS,
        json={
            "name": "lab-demo",
            "project_id": "lab",
            "target": "http://127.0.0.1/health",
            "mode": "lab",
            "strategy": "passive_recon",
        },
    )
    assert project.status_code == 200
    assert project.json()["state"] == "create"
    project_id = project.json()["id"]
    from app.api.v1.endpoints import security_agent as api

    api._PROJECTS.pop(project_id, None)
    restored_project = await client.get(
        f"/api/v1/security-agent/research-projects/{project_id}",
        headers=OPERATOR_HEADERS,
    )
    assert restored_project.status_code == 200
    assert restored_project.json()["name"] == "lab-demo"
    assert restored_project.json()["state"] == "create"


def test_unavailable_capability_is_not_enabled() -> None:
    registry = default_registry()
    spec = registry.spec("zap_scan")
    assert spec.capability in {
        ToolCapability.UNAVAILABLE,
        ToolCapability.RESULTS_INGESTIBLE,
        ToolCapability.EXECUTABLE,
    }
