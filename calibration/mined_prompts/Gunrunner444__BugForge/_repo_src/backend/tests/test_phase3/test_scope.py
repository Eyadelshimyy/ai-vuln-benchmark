"""Phase 3 scope matching, lab isolation, and default-deny rules."""

from __future__ import annotations

from app.security_testing.scope_guard import ScopeGuard
from app.security_testing.scope_model import ProgramScope, ScopeRule, TestingRestriction
from app.security_testing.target import AssetType, TargetNormalizer, hostname_matches, path_matches


def test_evil_example_does_not_match_example_com() -> None:
    assert hostname_matches("evil-example.com", ("example.com",)) is False
    assert hostname_matches("example.com.evil.test", ("example.com",)) is False
    assert hostname_matches("example.com", ("example.com",)) is True
    assert hostname_matches("EXAMPLE.COM.", ("example.com",)) is True


def test_wildcard_is_label_based() -> None:
    assert hostname_matches("foo.example.com", ("*.example.com",)) is True
    assert hostname_matches("a.b.example.com", ("*.example.com",)) is True
    assert hostname_matches("example.com", ("*.example.com",)) is True
    assert hostname_matches("evil-example.com", ("*.example.com",)) is False
    assert hostname_matches("example.com.attacker.test", ("*.example.com",)) is False


def test_path_prefix_uses_segments() -> None:
    assert path_matches("/api/users", "/api") is True
    assert path_matches("/apiary", "/api") is False
    assert path_matches("/api", "/api/") is True


def test_url_and_path_scope() -> None:
    guard = ScopeGuard(
        ProgramScope(
            program_name="demo",
            includes=(
                ScopeRule(
                    identifier="https://example.com/api",
                    asset_type=AssetType.URL,
                    allow_active_testing=True,
                ),
            ),
            allow_active_testing=True,
        )
    )
    assert guard.is_allowed("https://example.com/api/users", active=True) is True
    assert guard.is_allowed("https://example.com/other", active=True) is False


def test_exclusions_and_methods() -> None:
    guard = ScopeGuard(
        ProgramScope(
            includes=(
                ScopeRule(
                    identifier="example.com",
                    allow_active_testing=True,
                    allowed_methods=("GET",),
                    exclusions=("/admin",),
                ),
            ),
            excludes=(ScopeRule(identifier="out.example.com", is_exclusion=True),),
            allow_active_testing=True,
            allowed_methods=("GET",),
        )
    )
    assert guard.authorize("https://example.com/app", method="GET", tool="http").allowed is True
    assert guard.authorize("https://example.com/admin", method="GET", tool="http").allowed is False
    assert guard.authorize("https://example.com/app", method="DELETE", tool="http").allowed is False
    assert guard.authorize("https://out.example.com/", method="GET", tool="http").allowed is False


def test_ip_and_cidr() -> None:
    guard = ScopeGuard(
        ProgramScope(
            includes=(
                ScopeRule(
                    identifier="10.0.0.0/8", asset_type=AssetType.CIDR, allow_active_testing=True
                ),
            ),
            allow_active_testing=True,
        )
    )
    assert guard.is_allowed("http://10.1.2.3/", active=True) is True
    assert guard.is_allowed("http://11.0.0.1/", active=True) is False
    exact = ScopeGuard(
        ProgramScope(
            includes=(
                ScopeRule(
                    identifier="192.0.2.10", asset_type=AssetType.IP, allow_active_testing=True
                ),
            ),
            allow_active_testing=True,
        )
    )
    assert exact.is_allowed("http://192.0.2.10/", active=True) is True
    assert exact.is_allowed("http://192.0.2.11/", active=True) is False


def test_default_deny_no_scope_and_no_active() -> None:
    closed = ScopeGuard(ProgramScope.closed())
    denied = closed.authorize("https://example.com/", tool="zap")
    assert denied.allowed is False
    assert "NO SCOPE" in denied.reason
    scoped = ScopeGuard(ProgramScope.from_hosts(("example.com",)))
    active = scoped.authorize("https://example.com/", tool="zap", active=True)
    assert active.allowed is False
    assert "ACTIVE-TESTING" in active.reason
    unknown = ScopeGuard(ProgramScope.from_hosts(("example.com",), allow_active_testing=True))
    miss = unknown.authorize("https://other.test/", tool="zap")
    assert miss.allowed is False
    assert "UNKNOWN TARGET" in miss.reason


def test_idn_and_default_port() -> None:
    normalizer = TargetNormalizer()
    target = normalizer.normalize("https://bücher.example/")
    assert "xn--" in target.hostname or target.hostname.endswith("example")
    https = normalizer.normalize("https://Example.COM.")
    assert https.hostname == "example.com"
    assert https.port == 443
    assert https.path == "/"


def test_passive_only_restriction() -> None:
    guard = ScopeGuard(
        ProgramScope(
            includes=(
                ScopeRule(
                    identifier="example.com",
                    allow_active_testing=True,
                    restriction=TestingRestriction.PASSIVE_ONLY,
                ),
            ),
            allow_active_testing=True,
        )
    )
    assert guard.authorize("https://example.com/", tool="zap", active=True).allowed is False
    assert guard.authorize("https://example.com/", tool="har", active=False).allowed is True
