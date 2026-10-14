"""Phase 3 audit fixes: scanners, redirects, browser, fuzz, path, DNS, provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.adapters.browsers.playwright import PlaywrightBrowserAdapter, sanitize_browser_evidence
from app.adapters.security_tools.nuclei import NucleiAdapter, NucleiTemplatePolicy
from app.adapters.security_tools.zap import ZapAdapter
from app.domain.evidence import EvidenceKind, EvidenceProvenance
from app.domain.scope import ScopeConstraint
from app.security_testing.audit import AuditLog, InMemoryAuditStore
from app.security_testing.browser_network import BrowserNetworkPolicy
from app.security_testing.dns import DnsAuthorizer, MappingResolver, ResolvedTargetPolicy
from app.security_testing.engine import SecurityTestEngine, SecurityTestSession, TestingMode
from app.security_testing.errors import AuthorizationDeniedError
from app.security_testing.failures import ToolExecutionResult, ToolExecutionState
from app.security_testing.fuzzing import (
    FuzzLimits,
    MutationKind,
    malformed_json_document,
    mutate_json_document,
)
from app.security_testing.process import FakeProcessRunner
from app.security_testing.safety import SafetyLimits
from app.security_testing.scope_guard import ScopeGuard
from app.security_testing.scope_model import ProgramScope, ScopeRule
from app.security_testing.screenshots import ScreenshotStore
from app.security_testing.target import AssetType, canonicalize_path
from tests.fixtures.lab_app.server import LabServer


@pytest.fixture()
def lab_server() -> LabServer:
    server = LabServer().start()
    yield server
    server.stop()


def test_zap_plan_is_not_scanner_result_and_execution_ingests_alerts() -> None:
    engine = SecurityTestEngine.lab(allow_active_testing=True)
    from app.security_testing.approvals import ApprovalKind

    engine.grant(ApprovalKind.HIGH_RISK_SCANNER, operator="alice")
    runner = FakeProcessRunner(
        stdout=json.dumps(
            {"alerts": [{"alert": "CORS", "url": "http://127.0.0.1/", "description": "x"}]}
        )
    )
    zap = ZapAdapter(engine=engine, runner=runner, binary="zap.sh")
    evidence = zap.active_scan(
        scope=ScopeConstraint(allowed_hosts=("127.0.0.1",)),
        target="http://127.0.0.1/",
    )
    kinds = {item.kind for item in evidence}
    assert EvidenceKind.SCANNER_PLAN in kinds
    assert EvidenceKind.SCANNER in kinds
    plan = next(item for item in evidence if item.kind is EvidenceKind.SCANNER_PLAN)
    assert plan.contributes_to_verification is False
    assert plan.provenance is EvidenceProvenance.SCANNER_PLAN
    result = next(item for item in evidence if item.kind is EvidenceKind.SCANNER)
    assert result.contributes_to_verification is True
    assert "authorized" not in result.summary.lower()
    assert zap.last_result is not None
    assert zap.last_result.state is ToolExecutionState.RESULTS_INGESTED
    assert "-autorun" in " ".join(runner.last_argv)


def test_zap_unavailable_does_not_invent_findings() -> None:
    engine = SecurityTestEngine.lab(allow_active_testing=True)
    from app.security_testing.approvals import ApprovalKind

    engine.grant(ApprovalKind.HIGH_RISK_SCANNER, operator="alice")
    zap = ZapAdapter(engine=engine)
    evidence = list(
        zap.active_scan(
            scope=ScopeConstraint(allowed_hosts=("127.0.0.1",)), target="http://127.0.0.1/"
        )
    )
    assert all(item.kind is not EvidenceKind.SCANNER for item in evidence)
    assert any(item.kind is EvidenceKind.SCANNER_PLAN for item in evidence)
    assert any(item.kind is EvidenceKind.TOOL_STATUS for item in evidence)
    assert all(
        not item.contributes_to_verification or item.kind is EvidenceKind.SCANNER_PLAN
        for item in evidence
    )
    assert zap.last_result is not None
    assert zap.last_result.state is ToolExecutionState.TOOL_UNAVAILABLE
    assert zap.last_result.is_finding is False


def test_nuclei_executes_and_only_output_is_scanner_evidence() -> None:
    engine = SecurityTestEngine.lab(allow_active_testing=True)
    runner = FakeProcessRunner(
        stdout='{"template-id":"ok","info":{"severity":"low","tags":["misconfig"]},"matched-at":"http://127.0.0.1/x"}\n'
    )
    adapter = NucleiAdapter(engine=engine, runner=runner, binary="nuclei")
    evidence = adapter.active_scan(
        scope=ScopeConstraint(allowed_hosts=("127.0.0.1",)), target="http://127.0.0.1/x"
    )
    scanner = [item for item in evidence if item.kind is EvidenceKind.SCANNER]
    assert len(scanner) == 1
    assert "authorized" not in scanner[0].summary.lower()
    assert any(item.kind is EvidenceKind.SCANNER_PLAN for item in evidence)
    assert "-jsonl" in runner.last_argv
    assert adapter.last_result is not None
    assert adapter.last_result.state is ToolExecutionState.RESULTS_AVAILABLE


def test_nuclei_denies_untagged_unknown_severity_and_denied_tags() -> None:
    policy = NucleiTemplatePolicy(allowed_tags=("misconfig",), max_severity="low")
    adapter = NucleiAdapter(policy=policy)
    jsonl = "\n".join(
        [
            '{"template-id":"untagged","info":{"severity":"low","tags":[]},"matched-at":"http://127.0.0.1/x"}',
            '{"template-id":"unknown-sev","info":{"severity":"mystery","tags":["misconfig"]},"matched-at":"http://127.0.0.1/x"}',
            '{"template-id":"unknown-tag","info":{"severity":"low","tags":["brand-new"]},"matched-at":"http://127.0.0.1/x"}',
            '{"template-id":"dos","info":{"severity":"low","tags":["dos"]},"matched-at":"http://127.0.0.1/x"}',
            '{"template-id":"ok","info":{"severity":"low","tags":["misconfig"]},"matched-at":"http://127.0.0.1/x"}',
        ]
    )
    kept = adapter.ingest_jsonl(jsonl)
    assert [item.metadata["template_id"] for item in kept] == ["ok"]


@pytest.mark.asyncio
async def test_http_redirects_reauthorized(lab_server: LabServer) -> None:
    engine = SecurityTestEngine.lab(
        allow_active_testing=True,
        limits=SafetyLimits(
            max_requests=50,
            requests_per_second=50.0,
            max_concurrent=4,
            timeout_seconds=5,
            max_redirects=5,
            allowed_http_methods=("GET", "HEAD", "OPTIONS", "POST", "PUT"),
        ),
    )
    client = engine.http("http")
    ok = await client.request(
        "GET", lab_server.origin + "/redirect/in-scope", follow_redirects=True
    )
    assert not isinstance(ok, ToolExecutionResult)
    assert ok.response_status == 200
    scheme = await client.request(
        "GET", lab_server.origin + "/redirect/scheme", follow_redirects=True
    )
    assert scheme is not None
    with pytest.raises(AuthorizationDeniedError):
        await client.request("GET", lab_server.origin + "/redirect/out", follow_redirects=True)
    loop = await client.request("GET", lab_server.origin + "/redirect/loop", follow_redirects=True)
    assert isinstance(loop, ToolExecutionResult)
    assert loop.state is ToolExecutionState.EXECUTION_ERROR
    port = await client.request("GET", lab_server.origin + "/redirect/port", follow_redirects=True)
    assert port is not None


@pytest.mark.asyncio
async def test_http_redirect_excluded_path(lab_server: LabServer) -> None:
    from app.security_testing.engine import SecurityTestSession
    from app.security_testing.scope_model import ProgramScope, ScopeRule

    scope = ProgramScope(
        program_name="local-lab",
        includes=(
            ScopeRule(
                identifier="127.0.0.1",
                allow_active_testing=True,
                exclusions=("/admin",),
            ),
        ),
        allow_active_testing=True,
        lab_mode=True,
    )
    engine = SecurityTestEngine(
        SecurityTestSession(
            project_id="lab",
            mode=TestingMode.LAB,
            scope=scope,
            active_testing_enabled=True,
        )
    )
    client = engine.http("http")
    with pytest.raises(AuthorizationDeniedError):
        await client.request("GET", lab_server.origin + "/redirect/excluded", follow_redirects=True)


def test_browser_network_and_redirect_policy() -> None:
    engine = SecurityTestEngine.lab(allow_active_testing=True)
    policy = BrowserNetworkPolicy(engine)
    assert policy.check("http://127.0.0.1/login", resource_type="document").allowed is True
    assert policy.check("http://127.0.0.1/app.js", resource_type="script").allowed is True
    assert policy.check("https://evil.example/", resource_type="fetch").allowed is False
    assert policy.check("https://evil.example/", resource_type="websocket").allowed is False
    live = SecurityTestEngine(
        SecurityTestSession(
            project_id="p",
            mode=TestingMode.LIVE,
            scope=ProgramScope.from_hosts(("example.com",), allow_active_testing=True),
        )
    )
    from app.security_testing.approvals import ApprovalKind

    live.grant(ApprovalKind.ENABLE_ACTIVE_TESTING, operator="alice")
    live.dns = DnsAuthorizer(
        policy=ResolvedTargetPolicy.live(),
        resolver=MappingResolver(
            {"example.com": ["93.184.216.34"], "other.example.com": ["93.184.216.34"]}
        ),
    )
    net = BrowserNetworkPolicy(live)
    assert net.check("https://example.com/", resource_type="document").allowed is True
    assert net.check("https://other.example.com/", resource_type="document").allowed is False
    https_only = SecurityTestEngine(
        SecurityTestSession(
            project_id="p2",
            mode=TestingMode.LIVE,
            scope=ProgramScope(
                includes=(
                    ScopeRule(
                        identifier="https://example.com",
                        asset_type=AssetType.URL,
                        allow_active_testing=True,
                    ),
                ),
                allow_active_testing=True,
            ),
        )
    )
    https_only.grant(ApprovalKind.ENABLE_ACTIVE_TESTING, operator="alice")
    https_only.dns = DnsAuthorizer(
        policy=ResolvedTargetPolicy.live(),
        resolver=MappingResolver({"example.com": ["93.184.216.34"]}),
    )
    policy2 = BrowserNetworkPolicy(https_only)
    assert policy2.check("https://example.com/", resource_type="document").allowed is True
    assert policy2.check("http://example.com/", resource_type="document").allowed is False


@pytest.mark.asyncio
async def test_browser_route_aborts_out_of_scope() -> None:
    engine = SecurityTestEngine.lab(allow_active_testing=True)
    adapter = PlaywrightBrowserAdapter(engine=engine)
    policy = adapter.network_policy()

    class FakeRequest:
        def __init__(self, url: str, resource_type: str = "fetch") -> None:
            self.url = url
            self.resource_type = resource_type
            self.method = "GET"

    class FakeRoute:
        def __init__(self, url: str) -> None:
            self.request = FakeRequest(url)
            self.aborted = False
            self.continued = False

        async def abort(self) -> None:
            self.aborted = True

        async def continue_(self) -> None:
            self.continued = True

    denied = FakeRoute("https://evil.example/image.png")
    await policy.handle_route(denied)
    assert denied.aborted is True
    allowed = FakeRoute("http://127.0.0.1/style.css")
    allowed.request.resource_type = "stylesheet"
    await policy.handle_route(allowed)
    assert allowed.continued is True


def test_screenshot_store_rejects_escape_and_limits(tmp_path: Path) -> None:
    store = ScreenshotStore(tmp_path / "shots", max_bytes=32, max_files=2)
    path = store.save(b"png-bytes-ok-here-limit", stem="page-1")
    assert path.parent == store.root
    with pytest.raises(ValueError):
        store.save(b"x" * 100, stem="big")
    with pytest.raises(ValueError):
        store.save(b"abc", stem="../escape")
    store.save(b"one", stem="a")
    store.save(b"two", stem="b")
    store.save(b"three", stem="c")
    assert len(list(store.root.glob("*.png"))) == 2


def test_browser_sanitizes_secrets() -> None:
    from app.adapters.browsers.playwright import BrowserEvidence
    from app.domain.http import HttpExchange, HttpHeader

    raw = BrowserEvidence(
        url="https://lab/callback?access_token=secret-token-value",
        dom="<html>Authorization: Bearer super-secret-token</html>",
        console=("password=hunter2",),
        cookies_meta=("session",),
        local_storage_meta=("api_key",),
        network=(
            HttpExchange(
                method="GET",
                url="https://lab/x",
                request_headers=(HttpHeader("Authorization", "Bearer super-secret-token"),),
            ),
        ),
        history=("https://lab/?token=abc",),
    )
    cleaned = sanitize_browser_evidence(raw)
    blob = " ".join(
        [
            cleaned.url,
            cleaned.dom or "",
            " ".join(cleaned.console),
            str(cleaned.network[0].request_headers),
            " ".join(cleaned.history),
        ]
    )
    assert "super-secret-token" not in blob
    assert "hunter2" not in blob
    assert "secret-token-value" not in blob or "REDACTED" in blob


def test_json_mutation_is_real_json_not_repr() -> None:
    seed = '{"user": {"id": 1, "admin": false, "name": "a"}, "tags": [1, null]}'
    mutated = mutate_json_document(seed, "true")
    json.loads(mutated)
    assert "True" not in mutated
    assert "'true'" not in mutated
    broken = malformed_json_document(seed, "oops")
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)
    limits = FuzzLimits(request_limit=100, requests_per_second=50)
    engine = SecurityTestEngine.lab(allow_active_testing=True)
    effective = limits.intersect(engine.safety.limits)
    assert effective.request_limit <= engine.safety.limits.max_requests
    assert effective.requests_per_second <= engine.safety.limits.requests_per_second
    assert MutationKind.MALFORMED_JSON.value == "malformed_json"


def test_path_canonicalization_cannot_bypass_scope() -> None:
    assert canonicalize_path("/api/./users") == "/api/users"
    assert canonicalize_path("/api/../admin") == "/admin"
    assert canonicalize_path("/api//users") == "/api/users"
    assert canonicalize_path("/api/") == "/api"
    assert canonicalize_path("/api%2f../admin") == "/admin"
    assert canonicalize_path("/api%252fusers") == "/api/users"
    assert canonicalize_path("/api/%2e%2e/secret") == "/secret"
    guard = ScopeGuard(
        ProgramScope(
            includes=(
                ScopeRule(
                    identifier="https://example.com/api",
                    asset_type=AssetType.URL,
                    allow_active_testing=True,
                    exclusions=("/api/admin",),
                ),
            ),
            allow_active_testing=True,
        )
    )
    assert guard.is_allowed("https://example.com/api/users", active=True) is True
    assert guard.is_allowed("https://example.com/api/../admin", active=True) is False
    assert guard.is_allowed("https://example.com/api%2f../secret", active=True) is False
    assert guard.is_allowed("https://example.com/api/admin", active=True) is False
    assert guard.is_allowed("https://example.com/api%2fadmin", active=True) is False


def test_exclusion_uses_normalizer() -> None:
    guard = ScopeGuard(
        ProgramScope(
            includes=(
                ScopeRule(
                    identifier="EXAMPLE.COM.",
                    allow_active_testing=True,
                    exclusions=("https://example.com/admin",),
                ),
            ),
            excludes=(ScopeRule(identifier="https://out.example.com/", is_exclusion=True),),
            allow_active_testing=True,
        )
    )
    assert guard.authorize("https://example.com/app", tool="http").allowed is True
    assert guard.authorize("https://example.com/admin/x", tool="http").allowed is False
    assert guard.authorize("https://out.example.com/x", tool="http").allowed is False


def test_dns_policy_blocks_private_with_mock_resolver() -> None:
    authorizer = DnsAuthorizer(
        policy=ResolvedTargetPolicy.live(),
        resolver=MappingResolver(
            {
                "public.example": ["93.184.216.34"],
                "sneaky.example": ["10.0.0.5"],
                "meta.example": ["169.254.169.254"],
            }
        ),
    )
    assert authorizer.authorize("https://public.example/").allowed is True
    denied = authorizer.authorize("https://sneaky.example/")
    assert denied.allowed is False
    assert "private" in denied.reason or "prohibited" in denied.reason
    meta = authorizer.authorize("https://meta.example/")
    assert meta.allowed is False
    lab = DnsAuthorizer(policy=ResolvedTargetPolicy.lab(), resolver=MappingResolver({}))
    assert lab.authorize("http://127.0.0.1/", lab_mode=True).allowed is True


def test_live_engine_uses_dns_stage() -> None:
    engine = SecurityTestEngine(
        SecurityTestSession(
            project_id="p",
            mode=TestingMode.LIVE,
            scope=ProgramScope.from_hosts(("sneaky.example",), allow_active_testing=True),
        )
    )
    from app.security_testing.approvals import ApprovalKind

    engine.grant(ApprovalKind.ENABLE_ACTIVE_TESTING, operator="alice")
    engine.dns = DnsAuthorizer(
        policy=ResolvedTargetPolicy.live(),
        resolver=MappingResolver({"sneaky.example": ["192.168.1.8"]}),
    )
    decision = engine.authorize("https://sneaky.example/", tool="http", active=True)
    assert decision.allowed is False


def test_audit_store_interface_and_no_secrets() -> None:
    store = InMemoryAuditStore()
    log = AuditLog(store)
    log.record(
        project="p",
        target="https://x.test/?token=abcd",
        scope_decision="allow",
        tool="http",
        action="GET token=abcd",
        result="ok",
    )
    assert log.verify_chain() is True
    assert store.load()
    dumped = json.dumps(log.entries()[0].to_mapping())
    assert "abcd" not in dumped or "REDACTED" in dumped


def test_tool_execution_states_are_not_findings() -> None:
    result = ToolExecutionResult(tool="zap", state=ToolExecutionState.TIMEOUT)
    assert result.is_finding is False
    assert ToolExecutionState.NOT_REQUESTED
    assert ToolExecutionState.WAITING_FOR_APPROVAL
    assert ToolExecutionState.RUNNING
    assert ToolExecutionState.SAFETY_BLOCKED


@pytest.mark.asyncio
async def test_playwright_screenshot_store_in_headless_mode(tmp_path: Path) -> None:
    engine = SecurityTestEngine.lab(allow_active_testing=True)
    adapter = PlaywrightBrowserAdapter(
        engine=engine, screenshot_store=ScreenshotStore(tmp_path / "shots")
    )
    await adapter.navigate("http://127.0.0.1/login")
    snap = await adapter.capture()
    # Without Playwright installed, screenshot_path stays unset; store is still valid.
    if adapter.is_available() is False:
        assert snap.screenshot_path is None
    adapter.screenshot_store().save(b"fakepng", stem="manual")
    assert list((tmp_path / "shots").glob("*.png"))
