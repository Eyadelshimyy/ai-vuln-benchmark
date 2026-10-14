"""HackerOne-compatible structured scope models.

These fields prepare BugForge for later program import. This module does not
call the HackerOne API and does not submit reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.security_testing.target import AssetType


class Eligibility(StrEnum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


class ScopeMode(StrEnum):
    CLOSED = "closed"
    OPEN = "open"


class TestingRestriction(StrEnum):
    __test__ = False
    NONE = "none"
    NO_AUTOMATED_SCANNING = "no_automated_scanning"
    NO_DESTRUCTIVE = "no_destructive"
    PASSIVE_ONLY = "passive_only"
    RATE_LIMITED = "rate_limited"


@dataclass(frozen=True)
class ScopeRule:
    """One structured scope entry (include or exclude)."""

    identifier: str
    asset_type: AssetType = AssetType.DOMAIN
    eligible: Eligibility = Eligibility.ELIGIBLE
    instructions: str = ""
    exclusions: tuple[str, ...] = ()
    allowed_methods: tuple[str, ...] = ()
    allow_active_testing: bool = False
    path_prefix: str = ""
    restriction: TestingRestriction = TestingRestriction.NONE
    program_id: str | None = None
    is_exclusion: bool = False
    max_severity: str | None = None
    eligible_for_submission: bool = True
    eligible_for_bounty: bool = False
    structured_scope_id: str | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("Scope rule identifier must be non-empty")


@dataclass(frozen=True)
class ProgramScope:
    """Program or local-lab scope document. Empty includes deny everything."""

    program_id: str | None = None
    program_name: str | None = None
    includes: tuple[ScopeRule, ...] = ()
    excludes: tuple[ScopeRule, ...] = ()
    instructions: str = ""
    allow_active_testing: bool = False
    allowed_methods: tuple[str, ...] = ()
    testing_restrictions: tuple[TestingRestriction, ...] = ()
    lab_mode: bool = False
    lab_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")
    scope_mode: ScopeMode = ScopeMode.CLOSED
    open_scope_acknowledged: bool = False
    open_scope_policy: str = ""

    @classmethod
    def closed(cls) -> ProgramScope:
        return cls()

    @classmethod
    def from_hosts(
        cls,
        hosts: tuple[str, ...],
        *,
        excluded_hosts: tuple[str, ...] = (),
        allowed_methods: tuple[str, ...] = (),
        allow_active_testing: bool = False,
        program_name: str | None = None,
        instructions: str = "",
        lab_mode: bool = False,
    ) -> ProgramScope:
        includes = tuple(
            ScopeRule(
                identifier=host,
                asset_type=AssetType.WILDCARD
                if host.strip().startswith("*.")
                else AssetType.DOMAIN,
                allow_active_testing=allow_active_testing,
                allowed_methods=allowed_methods,
            )
            for host in hosts
            if host.strip()
        )
        excludes = tuple(
            ScopeRule(identifier=host, is_exclusion=True) for host in excluded_hosts if host.strip()
        )
        return cls(
            program_name=program_name,
            includes=includes,
            excludes=excludes,
            instructions=instructions,
            allow_active_testing=allow_active_testing,
            allowed_methods=allowed_methods,
            lab_mode=lab_mode,
        )


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str
    target_original: str
    method: str
    tool: str
    matched_rule: ScopeRule | None = None
    program: str | None = None
    dry_run: bool = False
    approval_required: bool = False
    approval_state: str = "not_required"
    rate_limit: str | None = None
    extra: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def deny_message(self) -> str:
        return self.reason
