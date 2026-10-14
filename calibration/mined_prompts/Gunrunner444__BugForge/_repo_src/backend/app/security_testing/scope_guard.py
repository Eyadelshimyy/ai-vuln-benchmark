"""Central ScopeGuard. Default deny. No naive substring matching."""

from __future__ import annotations

from app.domain.scope import ScopeConstraint
from app.security_testing.lab import is_loopback_or_lab_host
from app.security_testing.scope_model import (
    AuthorizationDecision,
    ProgramScope,
    ScopeMode,
    ScopeRule,
    TestingRestriction,
)
from app.security_testing.target import (
    NETWORK_ASSET_TYPES,
    AssetType,
    NormalizedTarget,
    TargetNormalizer,
    hostname_matches,
    path_matches,
)

_DEFAULT_DENY = "NO SCOPE = DENY"


class ScopeGuard:
    """Authorize a target against structured program scope.

    Defaults:
    * no include rules → deny
    * unknown target → deny
    * active testing without permission → deny
    * exclusions win
    """

    def __init__(self, scope: ProgramScope, *, normalizer: TargetNormalizer | None = None) -> None:
        self.scope = scope
        self.normalizer = normalizer or TargetNormalizer()

    @classmethod
    def from_constraint(cls, constraint: ScopeConstraint) -> ScopeGuard:
        return cls(
            ProgramScope.from_hosts(
                constraint.allowed_hosts,
                excluded_hosts=constraint.excluded_hosts,
                allowed_methods=constraint.allowed_methods,
                allow_active_testing=constraint.allow_active_testing,
                program_name=constraint.program_name,
                instructions=constraint.instructions,
            )
        )

    def authorize(
        self,
        target: str,
        *,
        method: str = "GET",
        tool: str = "unknown",
        active: bool = True,
        dry_run: bool = False,
    ) -> AuthorizationDecision:
        try:
            normalized = self.normalizer.normalize(target)
        except ValueError as exc:
            return AuthorizationDecision(
                allowed=False,
                reason=f"UNKNOWN TARGET = DENY ({exc})",
                target_original=target,
                method=method.upper(),
                tool=tool,
                dry_run=dry_run,
            )

        if self.scope.lab_mode:
            return self._authorize_lab(
                normalized, method=method, tool=tool, active=active, dry_run=dry_run
            )
        return self._authorize_live(
            normalized, method=method, tool=tool, active=active, dry_run=dry_run
        )

    def is_allowed(self, target: str, *, method: str = "GET", active: bool = True) -> bool:
        return self.authorize(target, method=method, active=active).allowed

    def _authorize_lab(
        self,
        target: NormalizedTarget,
        *,
        method: str,
        tool: str,
        active: bool,
        dry_run: bool,
    ) -> AuthorizationDecision:
        if not is_loopback_or_lab_host(target.hostname, extra=self.scope.lab_hosts):
            return AuthorizationDecision(
                allowed=False,
                reason="Local lab mode cannot target non-lab hosts",
                target_original=target.original,
                method=method.upper(),
                tool=tool,
                program=self.scope.program_name,
                dry_run=dry_run,
            )
        if self.scope.includes:
            included, _rule = self._included(target, method)
            if not included:
                # Lab may still restrict to configured lab assets.
                return self._deny(target, method, tool, "Target is not in lab scope", dry_run)
        excluded, rule = self._excluded(target, method)
        if excluded:
            return self._deny(
                target, method, tool, "Target matches a lab exclusion", dry_run, matched=rule
            )
        if active and not self.scope.allow_active_testing:
            return self._deny(target, method, tool, "NO ACTIVE-TESTING PERMISSION = DENY", dry_run)
        method_u = method.upper()
        if not self._method_allowed(method_u, None):
            return self._deny(
                target, method, tool, f"HTTP method {method_u} is not permitted", dry_run
            )
        return AuthorizationDecision(
            allowed=True,
            reason="Lab target authorized",
            target_original=target.original,
            method=method_u,
            tool=tool,
            program=self.scope.program_name or "local-lab",
            dry_run=dry_run,
        )

    def _authorize_live(
        self,
        target: NormalizedTarget,
        *,
        method: str,
        tool: str,
        active: bool,
        dry_run: bool,
    ) -> AuthorizationDecision:
        if not self.scope.includes:
            return self._deny(target, method, tool, _DEFAULT_DENY, dry_run)
        excluded, ex_rule = self._excluded(target, method)
        if excluded:
            return self._deny(
                target,
                method,
                tool,
                "Target is excluded from program scope",
                dry_run,
                matched=ex_rule,
            )
        included, rule = self._included(target, method)
        if not included or rule is None:
            if self.scope.scope_mode is ScopeMode.OPEN:
                return self._deny(
                    target,
                    method,
                    tool,
                    "Open-scope program: unknown assets are not automatically authorized. "
                    "An operator must approve an explicit active-testing policy before BugForge "
                    "may test assets that are not in the structured scope list.",
                    dry_run,
                )
            return self._deny(target, method, tool, "UNKNOWN TARGET = DENY", dry_run)
        method_u = method.upper()
        if not self._method_allowed(method_u, rule):
            return self._deny(
                target,
                method,
                tool,
                f"HTTP method {method_u} is not permitted",
                dry_run,
                matched=rule,
            )
        if active:
            if rule.asset_type is AssetType.UNSUPPORTED:
                return self._deny(
                    target,
                    method,
                    tool,
                    "UNSUPPORTED ASSET TYPE = DENY (explicit implementation required)",
                    dry_run,
                    matched=rule,
                )
            if rule.asset_type not in NETWORK_ASSET_TYPES:
                return self._deny(
                    target,
                    method,
                    tool,
                    f"Asset type {rule.asset_type.value} cannot be actively tested",
                    dry_run,
                    matched=rule,
                )
            if not self.scope.allow_active_testing and not rule.allow_active_testing:
                return self._deny(
                    target,
                    method,
                    tool,
                    "NO ACTIVE-TESTING PERMISSION = DENY",
                    dry_run,
                    matched=rule,
                )
            if rule.restriction is TestingRestriction.PASSIVE_ONLY:
                return self._deny(
                    target, method, tool, "Asset is passive-only", dry_run, matched=rule
                )
            if rule.restriction is TestingRestriction.NO_AUTOMATED_SCANNING and tool not in {
                "manual",
                "browser_evidence",
            }:
                return self._deny(
                    target,
                    method,
                    tool,
                    "Automated scanning is restricted for this asset",
                    dry_run,
                    matched=rule,
                )
        return AuthorizationDecision(
            allowed=True,
            reason="In-scope target authorized",
            target_original=target.original,
            method=method_u,
            tool=tool,
            matched_rule=rule,
            program=self.scope.program_name or self.scope.program_id,
            dry_run=dry_run,
        )

    def _included(self, target: NormalizedTarget, method: str) -> tuple[bool, ScopeRule | None]:
        for rule in self.scope.includes:
            if rule.is_exclusion:
                continue
            if self._rule_matches(rule, target, method):
                return True, rule
        return False, None

    def _excluded(self, target: NormalizedTarget, method: str) -> tuple[bool, ScopeRule | None]:
        for rule in self.scope.excludes:
            if self._rule_matches(rule, target, method):
                return True, rule
        for rule in self.scope.includes:
            for extra in rule.exclusions:
                if self._exclusion_token_matches(extra, target):
                    return True, rule
        return False, None

    def _exclusion_token_matches(self, extra: str, target: NormalizedTarget) -> bool:
        token = extra.strip()
        if not token:
            return False
        if token.startswith("/"):
            return path_matches(target.path, token)
        try:
            wanted = self.normalizer.normalize(token)
        except ValueError:
            return hostname_matches(target.hostname, (token,))
        if wanted.cidr:
            return hostname_matches(target.hostname or target.ip or "", (token,))
        if wanted.ip and not wanted.scheme:
            return hostname_matches(target.hostname or target.ip or "", (wanted.ip,))
        if wanted.hostname and wanted.hostname != target.hostname:
            return False
        prefix = wanted.path if wanted.path not in {"", "/"} else ""
        if prefix and not path_matches(target.path, prefix):
            return False
        if wanted.hostname:
            return hostname_matches(target.hostname, (wanted.hostname,))
        return False

    def _rule_matches(self, rule: ScopeRule, target: NormalizedTarget, method: str) -> bool:
        ident = rule.identifier.strip()
        if rule.asset_type is AssetType.CIDR or "/" in ident and _looks_cidr(ident):
            return hostname_matches(target.hostname or target.ip or "", (ident,))
        if rule.asset_type is AssetType.IP:
            return hostname_matches(target.hostname or target.ip or "", (ident,))
        if ident.startswith("http://") or ident.startswith("https://"):
            try:
                wanted = self.normalizer.normalize(ident)
            except ValueError:
                return False
            if target.hostname != wanted.hostname:
                return False
            if wanted.scheme and target.scheme and wanted.scheme != target.scheme:
                return False
            prefix = rule.path_prefix or wanted.path
            if prefix and not path_matches(target.path, prefix):
                return False
            return True
        if not hostname_matches(target.hostname, (ident,)):
            return False
        if rule.path_prefix and not path_matches(target.path, rule.path_prefix):
            return False
        return True

    def _method_allowed(self, method: str, rule: ScopeRule | None) -> bool:
        allowed = self.scope.allowed_methods
        if rule is not None and rule.allowed_methods:
            allowed = rule.allowed_methods
        if not allowed:
            return True
        return method in {item.strip().upper() for item in allowed}

    def _deny(
        self,
        target: NormalizedTarget,
        method: str,
        tool: str,
        reason: str,
        dry_run: bool,
        *,
        matched: ScopeRule | None = None,
    ) -> AuthorizationDecision:
        return AuthorizationDecision(
            allowed=False,
            reason=reason,
            target_original=target.original,
            method=method.upper(),
            tool=tool,
            matched_rule=matched,
            program=self.scope.program_name or self.scope.program_id,
            dry_run=dry_run,
        )


def _looks_cidr(value: str) -> bool:
    return value.count("/") == 1 and any(ch.isdigit() for ch in value)
