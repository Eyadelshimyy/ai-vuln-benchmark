"""Lab isolation, engine chain, HTTP client, and local vulnerable app."""

from __future__ import annotations

import pytest

from app.adapters.browsers.playwright import BrowserActionPolicy, PlaywrightBrowserAdapter
from app.domain.evidence import Evidence, EvidenceKind
from app.domain.findings import SecurityFinding
from app.security_testing.approvals import ApprovalKind
from app.security_testing.correlation import FindingCorrelationEngine, ai_only
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import AuthorizationDeniedError
from app.security_testing.failures import ToolExecutionResult
from app.security_testing.fuzzing import FuzzingEngine, FuzzLimits, MutationKind, SeedRequest
from app.security_testing.http_client import GatedHttpClient
from app.security_testing.safety import SafetyLimits
from tests.fixtures.lab_app.server import LabServer


@pytest.fixture()
def lab_server() -> LabServer:
    server = LabServer().start()
    yield server
    server.stop()


def test_lab_allows_loopback_only_when_active_enabled(lab_server: LabServer) -> None:
    blocked = SecurityTestEngine.lab(allow_active_testing=False)
    denied = blocked.authorize(lab_server.origin + "/health", tool="http")
    assert denied.allowed is False
    assert "ACTIVE-TESTING" in denied.reason

    engine = SecurityTestEngine.lab(allow_active_testing=True)
    allowed = engine.authorize(lab_server.origin + "/health", tool="http")
    assert allowed.allowed is True
    live_denied = engine.authorize("https://example.com/health", tool="http")
    assert live_denied.allowed is False


def test_live_mode_rejects_lab_bypass() -> None:
    from app.security_testing.engine import SecurityTestSession, TestingMode
    from app.security_testing.errors import RestrictedActivityError
    from app.security_testing.scope_model import ProgramScope

    with pytest.raises(RestrictedActivityError):
        SecurityTestEngine(
            SecurityTestSession(
                project_id="x",
                mode=TestingMode.LIVE,
                scope=ProgramScope(lab_mode=True, includes=()),
            )
        )


@pytest.mark.asyncio
async def test_gated_http_hits_lab_and_blocks_external(lab_server: LabServer) -> None:
    engine = SecurityTestEngine.lab(allow_active_testing=True)
    client: GatedHttpClient = engine.http("api_test")
    result = await client.request("GET", lab_server.origin + "/health")
    assert not isinstance(result, ToolExecutionResult)
    assert result.response_status == 200
    with pytest.raises(AuthorizationDeniedError):
        await client.request("GET", "https://example.com/")


@pytest.mark.asyncio
async def test_idor_and_reflected_output_fixtures(lab_server: LabServer) -> None:
    engine = SecurityTestEngine.lab(allow_active_testing=True)
    client = engine.http("api_test")
    user1 = await client.request("GET", lab_server.origin + "/users/1")
    user3 = await client.request("GET", lab_server.origin + "/users/3")
    assert not isinstance(user1, ToolExecutionResult)
    assert not isinstance(user3, ToolExecutionResult)
    assert user1.response_status == 200
    assert user3.response_status == 200
    assert "admin" in (user3.response_body or "")
    search = await client.request("GET", lab_server.origin + "/search?q=<b>x</b>")
    assert not isinstance(search, ToolExecutionResult)
    assert "<b>x</b>" in (search.response_body or "")


@pytest.mark.asyncio
async def test_fuzzing_respects_limits(lab_server: LabServer) -> None:
    engine = SecurityTestEngine.lab(
        allow_active_testing=True,
        limits=SafetyLimits.lab(),
    )
    engine.grant(ApprovalKind.ENABLE_FUZZING, operator="alice")
    fuzzer = FuzzingEngine(
        engine,
        limits=FuzzLimits(
            request_limit=3, payload_count=10, max_body_size=2048, requests_per_second=20.0
        ),
    )
    evidence = await fuzzer.fuzz(
        SeedRequest(method="GET", url=lab_server.origin + "/search?q=test"),
        kinds=(MutationKind.QUERY,),
        operator="alice",
    )
    assert isinstance(evidence, list)
    assert any(
        "request_limit" in item.summary or item.kind is EvidenceKind.FUZZING for item in evidence
    )


@pytest.mark.asyncio
async def test_browser_navigation_uses_scope_guard(lab_server: LabServer) -> None:
    engine = SecurityTestEngine.lab(allow_active_testing=True)
    browser = PlaywrightBrowserAdapter(engine=engine, policy=BrowserActionPolicy(allow_click=False))
    await browser.navigate(lab_server.origin + "/login")
    snap = await browser.snapshot()
    assert (
        lab_server.origin in snap.url
        or snap.url.endswith("/login")
        or snap.url == lab_server.origin + "/login"
    )
    with pytest.raises(AuthorizationDeniedError):
        await browser.navigate("https://example.com/")
    with pytest.raises(Exception):
        await browser.click("Delete everything")


@pytest.mark.asyncio
async def test_rate_limit_exceeded_blocks_http(lab_server: LabServer) -> None:
    engine = SecurityTestEngine.lab(
        allow_active_testing=True,
        limits=SafetyLimits(
            max_requests=50,
            requests_per_second=1.0,
            max_concurrent=2,
            timeout_seconds=5,
            allowed_http_methods=("GET", "HEAD", "OPTIONS", "POST", "PUT"),
        ),
    )
    client = engine.http("http")
    first = await client.request("GET", lab_server.origin + "/health")
    assert not isinstance(first, ToolExecutionResult)
    with pytest.raises(Exception):
        await client.request("GET", lab_server.origin + "/health")


def test_ai_only_evidence_never_verifies() -> None:
    finding = SecurityFinding.from_hypothesis("AI says XSS", "the model is sure")
    assert ai_only(finding) is True
    with pytest.raises(ValueError):
        finding.verify()
    engine = FindingCorrelationEngine()
    clusters = engine.correlate([finding])
    assert clusters[0].informational is True
    assert clusters[0].confidence == "informational"


def test_correlation_merges_equivalent_endpoints() -> None:
    one = SecurityFinding.potential(
        "IDOR",
        vulnerability_class="insecure_direct_object_reference",
        endpoint="https://lab/users/1",
        tools=("zap",),
        evidence=[Evidence(kind=EvidenceKind.SCANNER, source="zap", summary="alert")],
    )
    two = SecurityFinding.potential(
        "IDOR",
        vulnerability_class="insecure_direct_object_reference",
        endpoint="https://lab/users/2",
        tools=("api_test",),
        evidence=[Evidence(kind=EvidenceKind.API_TEST, source="api_test", summary="diff")],
    )
    clusters = FindingCorrelationEngine().correlate([one, two])
    assert len(clusters) == 1
    assert clusters[0].independent_sources >= 2
    assert clusters[0].informational is False
