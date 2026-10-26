"""The perimeter probes read deployment config and then attack what it declares.

Both halves matter: reading the wrong origin out of the compose file or the
charts makes the probe attack nothing, and a verdict that cannot tell a refusal
from an unreachable backend is the false pass the whole suite is built against.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probes import perimeter  # noqa: E402
from probes.base import Severity, Verdict  # noqa: E402
from probes.perimeter import (  # noqa: E402
    Origin,
    compose_origins,
    gateway_bypass_identity,
    ingress_origins,
)
from route_inventory import Route  # noqa: E402

VICTIM = "6c9a79ac-7d4e-49b7-a5c6-7eb146cd122d"


@dataclass
class StubIdentity:
    user_id: str = VICTIM


class StubContext:
    """Only the surface the perimeter probes use: a base URL and gateway requests."""

    def __init__(
        self, gateway: dict[str, int] | int = 401, base_url: str = "http://localhost:8080"
    ):
        self.base_url = base_url
        self._gateway = gateway
        self.victim = StubIdentity()
        self.requests: list[tuple[str, str]] = []

    def request(self, method: str, path: str, **_: object) -> httpx.Response:
        self.requests.append((method, path))
        status = self._gateway if isinstance(self._gateway, int) else self._gateway.get(path, 401)
        return response(status, url=f"{self.base_url}{path}", method=method)


def response(status: int, url: str = "http://backend/api/v1/files", method: str = "GET"):
    return httpx.Response(status, request=httpx.Request(method, url), json={"files": []})


def stub_direct(monkeypatch, statuses: dict[tuple[str, bool], int]) -> list[httpx.Request]:
    """Answer direct backend calls by (host, whether X-User-ID was sent)."""
    sent: list[httpx.Request] = []

    def fake(method: str, url: str, **kwargs) -> httpx.Response:
        request = httpx.Request(method, url, headers=kwargs.get("headers") or {})
        sent.append(request)
        spoofed = "x-user-id" in request.headers
        return response(statuses[(request.url.host, spoofed)], url=str(request.url))

    monkeypatch.setattr(perimeter.httpx, "request", fake)
    return sent


# ── reading the deployment config ────────────────────────────────────────────


def test_compose_origins_come_from_published_ports(tmp_path: Path) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        "services:\n"
        "  file-service:\n"
        '    ports: ["8082:8082"]\n'
        "  postgres:\n"
        '    ports: ["5432:5432"]\n'
        "  search-service:\n"
        "    expose: [8087]\n"
    )
    # postgres is not a header-trusting API, and a service that only `expose`s a
    # port is not published on the host at all.
    assert compose_origins("http://localhost:8080", compose) == [
        Origin("file-service", "http://localhost:8082", "docker-compose.yml ports", Severity.LOW)
    ]


def test_compose_origins_are_irrelevant_to_a_deployed_target(tmp_path: Path) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text('services:\n  file-service:\n    ports: ["8082:8082"]\n')
    assert compose_origins("https://api-t-example.otterworks.app", compose) == []


def test_ingress_origins_come_from_a_chart_that_publishes_its_own_host(tmp_path: Path) -> None:
    (tmp_path / "file-service").mkdir()
    (tmp_path / "file-service" / "values.yaml").write_text(
        "ingress:\n  enabled: true\n  hosts:\n    - host: files.example.test\n"
    )
    (tmp_path / "search-service").mkdir()
    (tmp_path / "search-service" / "values.yaml").write_text(
        "ingress:\n  enabled: false\n  hosts:\n    - host: search.example.test\n"
    )
    origins = ingress_origins("https://example.test", tmp_path)
    assert [(o.service, o.url, o.severity) for o in origins] == [
        ("file-service", "https://files.example.test", Severity.CRITICAL)
    ]


def test_ingress_origins_outside_the_target_site_are_left_alone(tmp_path: Path) -> None:
    """Authorization to scan one target is not authorization to scan a chart's host."""
    (tmp_path / "file-service").mkdir()
    (tmp_path / "file-service" / "values.yaml").write_text(
        "ingress:\n  enabled: true\n  hosts:\n    - host: files.somewhere-else.test\n"
    )
    assert ingress_origins("https://api.example.test", tmp_path) == []
    assert ingress_origins("http://localhost:8080", tmp_path) == []


def test_scope_is_the_target_and_what_is_under_it(monkeypatch) -> None:
    """Counting labels to guess a site puts strangers under a shared suffix in scope."""
    assert perimeter.in_scope("api.example.test", "api.example.test")
    assert perimeter.in_scope("files.api.example.test", "api.example.test")
    assert not perimeter.in_scope("files.example.test", "api.example.test")
    assert not perimeter.in_scope(
        "someone-else.eu-west-1.amazonaws.com", "us.eu-west-1.amazonaws.com"
    )
    # A host the target does not cover is attacked only once an operator names it.
    monkeypatch.setenv("DAST_ALLOW_ORIGIN_HOSTS", "files.example.test")
    assert perimeter.in_scope("files.example.test", "api.example.test")


# ── the bypass verdict ───────────────────────────────────────────────────────


@pytest.fixture
def only_file_service(monkeypatch):
    monkeypatch.setattr(perimeter, "ingress_origins", lambda target: [])
    monkeypatch.setattr(
        perimeter,
        "compose_origins",
        lambda target: [
            Origin("file-service", "http://backend:8082", "docker-compose.yml ports", Severity.LOW)
        ],
    )
    monkeypatch.setattr(perimeter, "reachable", lambda origin, timeout=3.0: True)


def test_no_declared_origin_is_secure(monkeypatch) -> None:
    monkeypatch.setattr(perimeter, "compose_origins", lambda target: [])
    monkeypatch.setattr(perimeter, "ingress_origins", lambda target: [])
    monkeypatch.setattr(perimeter, "declared_ingress_hosts", lambda: [])
    assert gateway_bypass_identity(StubContext()).verdict is Verdict.SECURE


def test_an_origin_dropped_for_being_out_of_scope_is_not_a_pass(monkeypatch) -> None:
    """The charts publish backends; this run just may not touch the hosts they name."""
    monkeypatch.setattr(perimeter, "compose_origins", lambda target: [])
    monkeypatch.setattr(perimeter, "ingress_origins", lambda target: [])
    monkeypatch.setattr(perimeter, "declared_ingress_hosts", lambda: ["files.somewhere-else.test"])
    result = gateway_bypass_identity(StubContext())
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "files.somewhere-else.test" in result.detail
    assert "DAST_ALLOW_ORIGIN_HOSTS" in result.detail


def test_a_declared_origin_that_never_answers_is_inconclusive(monkeypatch) -> None:
    """Unroutable from here is not the same as not exposed."""
    monkeypatch.setattr(
        perimeter,
        "compose_origins",
        lambda target: [
            Origin("file-service", "http://backend:8082", "docker-compose.yml ports", Severity.LOW)
        ],
    )
    monkeypatch.setattr(perimeter, "ingress_origins", lambda target: [])
    monkeypatch.setattr(perimeter, "reachable", lambda origin, timeout=3.0: False)
    assert gateway_bypass_identity(StubContext()).verdict is Verdict.INCONCLUSIVE


def test_a_chosen_identity_that_is_served_is_the_finding(only_file_service, monkeypatch) -> None:
    stub_direct(monkeypatch, {("backend", True): 200, ("backend", False): 401})
    result = gateway_bypass_identity(StubContext())
    assert result.verdict is Verdict.VULNERABLE
    assert "caller picks who they are" in result.detail
    assert result.severity is Severity.LOW  # a compose port, not a public ingress


def test_a_backend_that_needs_no_identity_is_still_the_finding(
    only_file_service, monkeypatch
) -> None:
    stub_direct(monkeypatch, {("backend", True): 200, ("backend", False): 200})
    result = gateway_bypass_identity(StubContext())
    assert result.verdict is Verdict.VULNERABLE
    assert "requires no identity at all" in result.detail


def test_a_backend_that_refuses_the_header_is_secure(only_file_service, monkeypatch) -> None:
    stub_direct(monkeypatch, {("backend", True): 401, ("backend", False): 401})
    assert gateway_bypass_identity(StubContext()).verdict is Verdict.SECURE


def test_a_route_the_gateway_serves_publicly_proves_nothing(only_file_service, monkeypatch) -> None:
    sent = stub_direct(monkeypatch, {("backend", True): 200, ("backend", False): 200})
    result = gateway_bypass_identity(StubContext(gateway=200))
    # Reaching a public route directly is not a bypass, so the backend is never
    # attacked — and a perimeter nothing was sent to is unproven, not proven.
    assert sent == []
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "was not attacked" in result.detail


def test_an_unavailable_backend_is_inconclusive(only_file_service, monkeypatch) -> None:
    stub_direct(monkeypatch, {("backend", True): 503, ("backend", False): 503})
    result = gateway_bypass_identity(StubContext())
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "503" in result.detail


def test_a_throttled_gateway_control_is_inconclusive(only_file_service, monkeypatch) -> None:
    stub_direct(monkeypatch, {("backend", True): 200, ("backend", False): 401})
    result = gateway_bypass_identity(StubContext(gateway=429))
    assert result.verdict is Verdict.INCONCLUSIVE


def test_a_public_ingress_outranks_a_local_port(monkeypatch) -> None:
    monkeypatch.setattr(
        perimeter,
        "compose_origins",
        lambda target: [
            Origin("file-service", "http://local:8082", "docker-compose.yml ports", Severity.LOW)
        ],
    )
    monkeypatch.setattr(
        perimeter,
        "ingress_origins",
        lambda target: [Origin("file-service", "https://public", "values.yaml", Severity.CRITICAL)],
    )
    monkeypatch.setattr(perimeter, "reachable", lambda origin, timeout=3.0: True)
    stub_direct(
        monkeypatch,
        {
            ("local", True): 200,
            ("local", False): 401,
            ("public", True): 200,
            ("public", False): 401,
        },
    )
    result = gateway_bypass_identity(StubContext())
    assert result.severity is Severity.CRITICAL
    assert "https://public" in result.detail


def test_the_probe_evidence_carries_no_credential(only_file_service, monkeypatch) -> None:
    stub_direct(monkeypatch, {("backend", True): 200, ("backend", False): 401})
    result = gateway_bypass_identity(StubContext())
    rendered = " ".join(f"{e.request} {e.response_excerpt} {e.note}" for e in result.evidence)
    assert "Bearer" not in rendered and "eyJ" not in rendered


# ── the anonymous sweep ──────────────────────────────────────────────────────


@pytest.fixture
def two_routes(monkeypatch):
    # The fixture's POST is only swept where writes are declared acceptable, which
    # is what most of these cases are about; the withholding case unsets it.
    monkeypatch.setenv("DAST_SWEEP_UNSAFE_METHODS", "1")
    monkeypatch.setattr(
        perimeter,
        "edge_routes",
        lambda: (
            [
                Route("POST", "/api/v1/auth/login", "auth-service", "AuthController.java"),
                Route("GET", "/api/v1/documents/{}", "document-service", "documents.py"),
            ],
            {},
        ),
    )


def test_a_route_that_would_be_performed_is_not_swept(two_routes, monkeypatch) -> None:
    """Sending an unauthenticated tenant-wide write would carry it out."""
    monkeypatch.setattr(
        perimeter,
        "sweep_exclusions",
        lambda: {"POST /api/v1/auth/login": "stands in for a tenant-wide operation"},
    )
    ctx = StubContext()
    result = perimeter.anonymous_route_sweep(ctx)
    assert ctx.requests == [("GET", f"/api/v1/documents/{perimeter.SWEEP_ID}")]
    assert "excluded from the sweep" in result.detail


def test_the_sweep_requests_every_route_with_a_placeholder_id(two_routes) -> None:
    ctx = StubContext()
    assert gateway_bypass_identity  # the module registers both probes
    result = perimeter.anonymous_route_sweep(ctx)
    assert result.verdict is Verdict.SECURE
    assert ctx.requests == [
        ("POST", "/api/v1/auth/login"),
        ("GET", f"/api/v1/documents/{perimeter.SWEEP_ID}"),
    ]


def test_a_protected_route_served_without_a_token_is_the_finding(two_routes) -> None:
    result = perimeter.anonymous_route_sweep(StubContext(gateway=200))
    assert result.verdict is Verdict.VULNERABLE
    # login is declared public, so only the document route counts.
    assert "1 of 2 route(s)" in result.detail


def test_a_route_the_middleware_protects_is_not_excused(two_routes, monkeypatch) -> None:
    """The public list is read from the gateway, so it cannot outlive the exemption."""
    monkeypatch.setattr(
        perimeter, "gateway_public_paths", lambda: ({"/api/v1/auth/login"}, ("/health",))
    )
    assert perimeter.anonymous_by_design("/api/v1/auth/login")
    assert perimeter.anonymous_by_design("/health/ready")
    assert not perimeter.anonymous_by_design("/api/v1/auth/refresh")


def test_an_unreadable_middleware_makes_the_sweep_withhold_a_verdict(
    two_routes, monkeypatch
) -> None:
    """Not knowing which routes are meant to answer makes every 2xx unjudgeable."""
    monkeypatch.setattr(perimeter, "gateway_public_paths", lambda: None)
    ctx = StubContext(gateway=200)
    result = perimeter.anonymous_route_sweep(ctx)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert ctx.requests == []


def test_every_route_unavailable_is_inconclusive(two_routes) -> None:
    assert perimeter.anonymous_route_sweep(StubContext(gateway=503)).verdict is Verdict.INCONCLUSIVE


def test_an_empty_inventory_is_inconclusive(monkeypatch) -> None:
    monkeypatch.setattr(perimeter, "edge_routes", lambda: ([], {}))
    assert perimeter.anonymous_route_sweep(StubContext()).verdict is Verdict.INCONCLUSIVE


# ── the sweep must actually reach the handler ────────────────────────────────


class RedirectingContext(StubContext):
    """A target that mounts its routes with a trailing slash, as FastAPI does."""

    def __init__(self, slashed_status: int, location: str | None = None):
        super().__init__()
        self.slashed_status = slashed_status
        self.location = location

    def request(self, method: str, path: str, **_: object) -> httpx.Response:
        self.requests.append((method, path))
        # A followed redirect is addressed absolutely, as httpx requires.
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        if path.endswith("/"):
            return response(self.slashed_status, url=url, method=method)
        headers = {"location": self.location if self.location is not None else path + "/"}
        return httpx.Response(307, headers=headers, request=httpx.Request(method, url), json={})


def test_a_route_answered_behind_a_slash_redirect_is_followed(two_routes) -> None:
    """A 307 is neither an answer nor a refusal, and counting it as one is a false pass."""
    ctx = RedirectingContext(slashed_status=200)
    result = perimeter.anonymous_route_sweep(ctx)
    slashed = f"{ctx.base_url}/api/v1/documents/{perimeter.SWEEP_ID}/"
    assert ("GET", slashed) in ctx.requests
    assert result.verdict is Verdict.VULNERABLE


def test_a_redirect_off_the_target_is_not_chased(two_routes) -> None:
    """Following it would attack a host nobody authorized, so the route is unswept."""
    ctx = RedirectingContext(slashed_status=200, location="https://elsewhere.example/api")
    result = perimeter.anonymous_route_sweep(ctx)
    assert all(host not in path for _, path in ctx.requests for host in ["elsewhere"])
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "redirected away" in result.detail


# ── reading a compose port entry ─────────────────────────────────────────────


def test_every_form_of_a_published_port_is_read(tmp_path: Path) -> None:
    """This parse is the only input to the verdict, so a missed form reads SECURE."""
    assert perimeter.published_port("8082:8082") == (None, "8082")
    assert perimeter.published_port("127.0.0.1:8082:8082") == ("127.0.0.1", "8082")
    assert perimeter.published_port({"target": 8082, "published": 8082}) == (None, "8082")
    assert perimeter.published_port({"published": 8082, "host_ip": "127.0.0.1"}) == (
        "127.0.0.1",
        "8082",
    )
    # No fixed host port to attack: compose picks an ephemeral one.
    assert perimeter.published_port("8082") is None


def test_a_port_bound_to_one_interface_is_attacked_there(tmp_path: Path) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services:\n  file-service:\n    ports:\n      - '127.0.0.1:8082:8082'\n")
    origins = perimeter.compose_origins("http://localhost:8080", compose=compose)
    assert [o.url for o in origins] == ["http://127.0.0.1:8082"]


# ── writes are not sent to a target nobody called disposable ─────────────────


def test_a_write_is_only_swept_where_it_may_be_performed(monkeypatch) -> None:
    """localhost is not evidence: the runbook reaches a live tenant by port-forward."""
    monkeypatch.delenv("DAST_SWEEP_UNSAFE_METHODS", raising=False)
    assert not perimeter.may_sweep_unsafely()
    monkeypatch.setenv("DAST_SWEEP_UNSAFE_METHODS", "1")
    assert perimeter.may_sweep_unsafely()


def test_the_sweep_withholds_writes_from_an_undeclared_target(two_routes, monkeypatch) -> None:
    """A route that answers a DELETE has been deleted, so the report says it was withheld."""
    monkeypatch.delenv("DAST_SWEEP_UNSAFE_METHODS", raising=False)
    monkeypatch.setattr(
        perimeter,
        "edge_routes",
        lambda: (
            [
                Route("GET", "/api/v1/documents/{}", "document-service", "app/api/documents.py"),
                Route("DELETE", "/api/v1/documents/{}", "document-service", "app/api/documents.py"),
            ],
            {},
        ),
    )
    ctx = StubContext(base_url="http://localhost:8080")
    result = perimeter.anonymous_route_sweep(ctx)
    assert [method for method, _ in ctx.requests] == ["GET"]
    assert "not sent because their method would write" in result.detail


def test_a_followed_redirect_is_addressed_absolutely() -> None:
    """A relative retry would be merged onto a path-routed tenant's prefix twice."""

    class Recorder(StubContext):
        def __init__(self):
            super().__init__(base_url="https://nlb.example.test/t-abc")

        def request(self, method: str, path: str, **_: object) -> httpx.Response:
            self.requests.append((method, path))
            return response(200, url=path, method=method)

    ctx = Recorder()
    redirect = httpx.Response(
        307,
        headers={"location": "/t-abc/api/v1/documents/"},
        request=httpx.Request("GET", "https://nlb.example.test/t-abc/api/v1/documents"),
    )
    perimeter.follow_once(ctx, "GET", redirect)
    # httpx merges a relative path onto the client's base path, so `/t-abc/...`
    # would be sent as `/t-abc/t-abc/...` and 404 without testing the route.
    assert ctx.requests == [("GET", "https://nlb.example.test/t-abc/api/v1/documents/")]


def test_a_backend_that_serves_a_caller_claiming_nothing_is_the_finding(
    only_file_service, monkeypatch
) -> None:
    """Rejecting an unknown identity is not protection if no identity works."""
    stub_direct(monkeypatch, {("backend", True): 403, ("backend", False): 200})
    result = gateway_bypass_identity(StubContext(gateway=401))
    assert result.verdict is Verdict.VULNERABLE
    assert "requires no identity at all" in result.detail
