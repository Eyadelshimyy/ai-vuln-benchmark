"""Deterministic HackerOne scope evaluator. AI never decides scope.

Structured scope answers *where* an asset may be tested.
Scope exclusions answer *which report categories* are not bounty-eligible.
They are never treated as a target/domain deny-list.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.adapters.hackerone.hashes import scope_snapshot_hash
from app.adapters.hackerone.models import (
    HackerOneProgram,
    ScopeExclusionRecord,
    ScopeMode,
    ScopeSnapshot,
    StructuredScopeRecord,
)
from app.security_testing.target import (
    NETWORK_ASSET_TYPES,
    AssetType,
    NormalizedTarget,
    TargetNormalizer,
    hostname_matches,
    path_matches,
)


@dataclass(frozen=True)
class TargetScopeDecision:
    """Whether a *target* is listed in current structured scope."""

    in_scope: bool
    reason: str
    structured_scope_id: str | None = None
    asset_type: str | None = None
    asset_identifier: str | None = None
    eligible_for_submission: bool = False
    eligible_for_bounty: bool = False
    matched_instructions: str = ""


@dataclass(frozen=True)
class ReportEligibilityDecision:
    """Whether a *finding/report category* is eligible to submit / earn bounty."""

    eligible_for_submission: bool
    eligible_for_bounty: bool
    reason: str
    exclusion_categories: tuple[str, ...] = ()
    matching_exclusions: tuple[str, ...] = ()


@dataclass(frozen=True)
class HackerOneScopeDecision:
    allowed: bool
    reason: str
    structured_scope_id: str | None = None
    asset_type: str | None = None
    asset_identifier: str | None = None
    eligible_for_submission: bool = False
    eligible_for_bounty: bool = False
    in_scope: bool = False
    matched_instructions: str = ""
    exclusions_considered: tuple[str, ...] = ()
    target_scope: TargetScopeDecision | None = None
    report_eligibility: ReportEligibilityDecision | None = None

    def snapshot(self, program: HackerOneProgram) -> ScopeSnapshot:
        numeric = None
        if self.structured_scope_id and self.structured_scope_id.isdigit():
            numeric = int(self.structured_scope_id)
        payload = {
            "program_handle": program.handle,
            "scope_version": program.scope_version,
            "structured_scope_id": numeric,
            "asset_identifier": self.asset_identifier or "",
            "eligible_for_submission": self.eligible_for_submission,
            "eligible_for_bounty": self.eligible_for_bounty,
            "in_scope": self.in_scope,
            "matched_instructions": self.matched_instructions,
        }
        return ScopeSnapshot(
            program_handle=program.handle,
            scope_version=program.scope_version,
            structured_scope_id=numeric,
            asset_identifier=self.asset_identifier or "",
            eligible_for_submission=self.eligible_for_submission,
            eligible_for_bounty=self.eligible_for_bounty,
            in_scope=self.in_scope,
            matched_instructions=self.matched_instructions,
            snapshot_hash=scope_snapshot_hash(payload),
        )


class HackerOneScopeEvaluator:
    def __init__(self, normalizer: TargetNormalizer | None = None) -> None:
        self.normalizer = normalizer or TargetNormalizer()

    def evaluate(
        self,
        program: HackerOneProgram,
        target: str,
        *,
        vulnerability_class: str | None = None,
    ) -> HackerOneScopeDecision:
        target_decision = self.evaluate_target(program, target)
        report_decision = self.evaluate_report_eligibility(
            program, target_decision, vulnerability_class=vulnerability_class
        )
        allowed = target_decision.in_scope and report_decision.eligible_for_submission
        reason = target_decision.reason
        if target_decision.in_scope and not report_decision.eligible_for_submission:
            reason = report_decision.reason
        elif target_decision.in_scope and not report_decision.eligible_for_bounty:
            reason = (
                f"{target_decision.reason}; {report_decision.reason}"
                if report_decision.reason
                else target_decision.reason
            )
        return HackerOneScopeDecision(
            allowed=allowed,
            reason=reason,
            structured_scope_id=target_decision.structured_scope_id,
            asset_type=target_decision.asset_type,
            asset_identifier=target_decision.asset_identifier,
            eligible_for_submission=report_decision.eligible_for_submission,
            eligible_for_bounty=report_decision.eligible_for_bounty,
            in_scope=target_decision.in_scope,
            matched_instructions=target_decision.matched_instructions,
            exclusions_considered=report_decision.matching_exclusions,
            target_scope=target_decision,
            report_eligibility=report_decision,
        )

    def evaluate_target(self, program: HackerOneProgram, target: str) -> TargetScopeDecision:
        match = self._match_structured(program.structured_scopes, target)
        if match is None:
            if program.scope_mode is ScopeMode.OPEN:
                return TargetScopeDecision(
                    in_scope=False,
                    reason=(
                        "Open-scope program: unknown assets are not automatically authorized. "
                        "An operator must approve an explicit active-testing policy before "
                        "BugForge may test assets that are not listed in structured scope."
                    ),
                )
            return TargetScopeDecision(
                in_scope=False,
                reason="Closed-scope program: unknown asset is denied",
            )
        reason = "In structured scope"
        if not match.eligible_for_submission:
            reason = "Asset is in structured scope but not eligible for submission"
        return TargetScopeDecision(
            in_scope=True,
            reason=reason,
            structured_scope_id=match.id,
            asset_type=match.asset_type.value,
            asset_identifier=match.asset_identifier,
            eligible_for_submission=match.eligible_for_submission,
            eligible_for_bounty=match.eligible_for_bounty,
            matched_instructions=match.instruction,
        )

    def evaluate_report_eligibility(
        self,
        program: HackerOneProgram,
        target: TargetScopeDecision,
        *,
        vulnerability_class: str | None = None,
    ) -> ReportEligibilityDecision:
        if not target.in_scope:
            return ReportEligibilityDecision(
                eligible_for_submission=False,
                eligible_for_bounty=False,
                reason=target.reason,
            )
        submission = target.eligible_for_submission
        bounty = target.eligible_for_bounty and submission
        matched = _matching_exclusion_categories(program.exclusions, vulnerability_class)
        if matched:
            bounty = False
            categories = tuple(item.category for item in matched)
            return ReportEligibilityDecision(
                eligible_for_submission=submission,
                eligible_for_bounty=bounty,
                reason=(
                    "In scope and eligible for submission but not bounty eligible "
                    f"due to program scope exclusion categories: {', '.join(categories)}"
                    if submission
                    else "Asset is in structured scope but not eligible for submission"
                ),
                exclusion_categories=categories,
                matching_exclusions=tuple(item.id or item.category for item in matched),
            )
        if not submission:
            return ReportEligibilityDecision(
                eligible_for_submission=False,
                eligible_for_bounty=False,
                reason="Asset is in structured scope but not eligible for submission",
            )
        if not bounty:
            return ReportEligibilityDecision(
                eligible_for_submission=True,
                eligible_for_bounty=False,
                reason="In scope and eligible for submission but not bounty eligible",
            )
        return ReportEligibilityDecision(
            eligible_for_submission=True,
            eligible_for_bounty=True,
            reason="In structured scope and eligible for submission and bounty",
        )

    def _match_structured(
        self, scopes: tuple[StructuredScopeRecord, ...], target: str
    ) -> StructuredScopeRecord | None:
        network_target = self._try_network_target(target)
        for record in scopes:
            if record.asset_type not in NETWORK_ASSET_TYPES:
                if target.strip() == record.asset_identifier.strip():
                    return record
                continue
            if network_target is None:
                continue
            if self._network_match(record, network_target):
                return record
        return None

    def _network_match(self, record: StructuredScopeRecord, target: NormalizedTarget) -> bool:
        ident = record.asset_identifier.strip()
        if record.asset_type is AssetType.CIDR:
            return hostname_matches(target.hostname or target.ip or "", (ident,))
        if record.asset_type is AssetType.IP:
            return hostname_matches(target.hostname or target.ip or "", (ident,))
        if (
            record.asset_type is AssetType.URL
            or ident.startswith("http://")
            or ident.startswith("https://")
        ):
            try:
                wanted = self.normalizer.normalize(ident)
            except ValueError:
                return False
            if target.hostname != wanted.hostname:
                return False
            if wanted.scheme and target.scheme and wanted.scheme != target.scheme:
                return False
            if wanted.port and target.port and wanted.port != target.port:
                return False
            if wanted.path not in {"", "/"} and not path_matches(target.path, wanted.path):
                return False
            return True
        return hostname_matches(target.hostname, (ident,))

    def _try_network_target(self, target: str) -> NormalizedTarget | None:
        try:
            return self.normalizer.normalize(target)
        except ValueError:
            return None


def _matching_exclusion_categories(
    exclusions: tuple[ScopeExclusionRecord, ...],
    vulnerability_class: str | None,
) -> tuple[ScopeExclusionRecord, ...]:
    """Match report *categories*, never hostnames in exclusion details."""
    if not vulnerability_class:
        return ()
    key = vulnerability_class.strip().lower().replace("_", " ").replace("-", " ")
    tokens = {part for part in key.split() if part}
    aliases = {
        "sql injection": {"sqli", "sql", "injection"},
        "cross site scripting": {"xss"},
        "denial of service": {"dos", "ddos"},
    }
    hits: list[ScopeExclusionRecord] = []
    for item in exclusions:
        category = (item.category or "").strip().lower().replace("_", " ").replace("-", " ")
        if not category:
            continue
        if category == key or key in category or category in key:
            hits.append(item)
            continue
        extra = aliases.get(category, set())
        if tokens & extra or key in extra:
            hits.append(item)
    return tuple(hits)
