"""Phase 9 live-mode safety invariants."""

from __future__ import annotations

import pytest

from app.ai.mock_provider import MockLLMProvider
from app.security_agent.agent import ResearchSession, SecurityResearchAgent
from app.security_agent.orchestrator import AdvancedResearchOrchestrator
from app.security_agent.replay import SessionReplay
from app.security_agent.schemas import ToolCallRequest
from app.security_agent.states import ResearchMode
from app.security_testing.engine import SecurityTestEngine, SecurityTestSession, TestingMode
from app.security_testing.errors import RestrictedActivityError
from app.security_testing.safety import SafetyLimits
from app.security_testing.scope_model import Eligibility, ProgramScope, ScopeRule
from app.security_testing.target import AssetType


def _live_engine(
    *, includes: tuple[ScopeRule, ...] = (), active: bool = False
) -> SecurityTestEngine:
    return SecurityTestEngine(
        SecurityTestSession(
            project_id="live",
            mode=TestingMode.LIVE,
            scope=ProgramScope(
                program_name="demo",
                includes=includes,
                lab_mode=False,
                allow_active_testing=active,
            ),
            limits=SafetyLimits.conservative(),
            dry_run=True,
            active_testing_enabled=active,
        )
    )


def _live_session(**kwargs: object) -> ResearchSession:
    engine = kwargs.pop("engine", None) or _live_engine()
    values = dict(
        project_id="live",
        target="https://app.example/",
        mode=ResearchMode.LIVE_HACKERONE,
        engine=engine,
        provider=MockLLMProvider(),
        program_handle="demo",
    )
    values.update(kwargs)
    return ResearchSession(**values)  # type: ignore[arg-type]


def test_no_scope_is_blocked() -> None:
    agent = SecurityResearchAgent(_live_session())
    decision = agent._authorize_target("https://app.example/", tool="http_request")
    assert decision.value == "blocked"


def test_unknown_asset_is_blocked() -> None:
    engine = _live_engine(
        includes=(
            ScopeRule(
                identifier="https://in.example/",
                asset_type=AssetType.URL,
                eligible=Eligibility.ELIGIBLE,
            ),
        )
    )
    denied = engine.authorize("https://out.example/", tool="http_request", active=False)
    assert denied.allowed is False


def test_active_testing_disabled_is_blocked() -> None:
    engine = _live_engine(
        includes=(
            ScopeRule(
                identifier="https://in.example/",
                asset_type=AssetType.URL,
                eligible=Eligibility.ELIGIBLE,
                allow_active_testing=True,
            ),
        ),
        active=False,
    )
    denied = engine.authorize("https://in.example/", tool="http_request", active=True)
    assert denied.allowed is False


@pytest.mark.asyncio
async def test_tool_disabled_and_budget_exhausted() -> None:
    session = _live_session()
    agent = SecurityResearchAgent(session)
    agent.disable_tool("http_request")
    with pytest.raises(RestrictedActivityError, match="disabled_tool"):
        await agent.request_tool(
            ToolCallRequest(
                tool="http_request",
                arguments={"method": "GET", "url": "https://app.example/"},
                reason="x",
            )
        )
    session.budget.max_requests = 0
    session.budget.requests = 0
    session.disabled_tools.clear()
    agent.tools.enable("http_request")
    session.budget.max_iterations = 0
    session.budget.iterations = 0
    decision = await agent.step()
    assert decision.kind == "budget_exhausted"


def test_ai_cannot_verify_approve_or_submit() -> None:
    orch = AdvancedResearchOrchestrator(SecurityResearchAgent(_live_session()))
    for kind in (
        "verify_finding",
        "grant_approval",
        "enable_active_testing",
        "submit_report",
        "approve_report",
        "change_scope",
        "increase_budget",
    ):
        with pytest.raises(RestrictedActivityError):
            orch.reject_escalation(kind)


@pytest.mark.asyncio
async def test_replay_mode_has_no_live_network() -> None:
    session = _live_session()
    agent = SecurityResearchAgent(session)
    replay = SessionReplay()
    replay.record("source_inspect", {"path": "a.py"}, {"quality": "success", "executed": True})
    await replay.play(agent)
    assert session.replay_mode is False
    for node in session.graph.nodes.values():
        if node.kind == "observation":
            assert node.provenance == "replay"
