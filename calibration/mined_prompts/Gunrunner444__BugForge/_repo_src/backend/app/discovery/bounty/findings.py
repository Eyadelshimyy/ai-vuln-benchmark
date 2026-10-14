"""Bounty-quality research findings.

A finding here is a static candidate plus everything a reviewer needs to judge it:
scope, known-issue and duplicate status, the evidence that exists, and what is still
missing. Suppression is a statement about reporting, never about safety: a known
issue is not safe, an out-of-scope asset is not safe, and a duplicate is not safe.
Severity is only ever a candidate drawn from the program's own policy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.discovery.bounty.campaign import (
    BountyManifest,
    ImpactCategory,
    KnownIssue,
    PocRequirement,
    ScopeStatus,
)
from app.discovery.bounty.vfcs import MinimizationResult, Vfcs
from app.discovery.orchestration.codec import digest
from app.discovery.orchestration.model import (
    CampaignIdentity,
    EvidenceItem,
    EvidenceQuality,
    ResearchState,
)
from app.parsing.solidity_research import SemanticCandidate

if TYPE_CHECKING:
    from app.discovery.bounty.deployment_identity import DeploymentIdentity

DeploymentResolver = Callable[[str], "DeploymentIdentity | None"]

MAX_FINDINGS = 128
UNKNOWN = "unknown"
CONFIRMED_SEVERITY = "unconfirmed"

_RUNTIME_CATEGORIES = frozenset({"runtime", "differential", "economic"})


@dataclass(frozen=True)
class KnownIssueMatch:
    issue_id: str
    source: str
    basis: str


@dataclass(frozen=True)
class Qualification:
    """Why a candidate is or is not ready to be written up. Never a safety statement."""

    # report_candidate | needs_scope | needs_poc | known_issue | out_of_scope |
    # ambiguous_deployment | ambiguous_identity
    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ResearchFinding:
    finding_id: str
    title: str
    detector: str
    family: str
    contract: str
    function: str
    file: str
    line: int
    root_cause: str
    affected_asset: str
    impact_claim: str
    severity_candidate: str
    severity_rationale: str
    confirmed_severity: str
    scope_status: str
    scope_reason: str
    known_issue: KnownIssueMatch | None
    duplicate_of: str
    poc_status: str
    preconditions: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_quality: str
    open_contradictions: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    minimization: tuple[str, ...]
    identity: tuple[tuple[str, str], ...]
    uncertainties: tuple[str, ...]
    qualification: Qualification
    verified: bool = False
    submitted: bool = False
    root_cause_key: str = ""
    duplicate_basis: str = ""
    deployment: tuple[tuple[str, str], ...] = ()
    # Phase 52 hardening: what local execution established. A sequence that ran
    # without an oracle is ``weak_execution_only`` and never a strong candidate.
    execution_status: str = "not_executed"
    candidate_strength: str = "static_candidate"
    property_ids: tuple[str, ...] = ()
    bundle_ids: tuple[str, ...] = ()
    corroboration: str = "not_applicable"


# Facts that name the defect itself rather than the entry point. Two entry points
# that reach the same anchor (the same state variable written after a call, the
# same nested callee, the same transient slot, the same EOA assumption) share a
# root cause.
_ANCHOR_FACTS = (
    "root_cause",
    "state_variable",
    "eventual_callee",
    "uncleared_slots",
    "uncleared_vars",
    "transient_key",
    "assumption_site",
    "sink",
)


def legacy_root_cause_key(candidate: SemanticCandidate) -> str:
    """The Phase 50 key (detector, contract, function name). Kept for known issues."""
    return f"{candidate.detector}:{candidate.contract}:{candidate.function.split('(')[0]}"


def root_cause_key(candidate: SemanticCandidate, deployment: str = "") -> str:
    """Candidates that share a root cause share a key, whatever the entry point.

    The key is the detector plus the defect anchor (a fact naming the variable,
    callee, slot, or site) when the detector records one, else the function name;
    it is bound to the deployment identity when one is known, so the same source
    defect in two different deployments is not one finding.
    """
    anchor = ""
    for name in _ANCHOR_FACTS:
        value = candidate.fact(name)
        if value:
            anchor = f"{name}={value}"
            break
    base = f"{candidate.detector}:{candidate.contract}:{anchor or candidate.function.split('(')[0]}"
    return f"{base}@{deployment}" if deployment and deployment != "source" else base


def build_findings(
    candidates: Iterable[SemanticCandidate],
    *,
    manifest: BountyManifest,
    identity: CampaignIdentity,
    state: ResearchState | None = None,
    sequences: Iterable[Vfcs] = (),
    minimized: Mapping[str, MinimizationResult] | None = None,
    deployment_for: DeploymentResolver | None = None,
    ambiguous_contracts: frozenset[str] = frozenset(),
    target_address: str = "",
    target_chain_id: str = "",
    executions: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[ResearchFinding, ...]:
    sequences = tuple(sequences)
    execution_list = tuple(dict(item) for item in (executions or {}).values())
    minimized = minimized or {}
    seen_roots: dict[str, str] = {}
    seen_sites: dict[str, str] = {}
    findings: list[ResearchFinding] = []
    for candidate in sorted(
        candidates, key=lambda c: (c.file, c.line, c.contract, c.function, c.detector)
    ):
        if len(findings) >= MAX_FINDINGS:
            break
        deployment = deployment_for(candidate.contract) if deployment_for else None
        finding = _one(
            candidate,
            manifest,
            identity,
            state,
            sequences,
            minimized,
            seen_roots,
            seen_sites,
            deployment,
            candidate.contract in ambiguous_contracts,
            target_address,
            target_chain_id,
            execution_list,
        )
        findings.append(finding)
    return tuple(findings)


def _one(
    candidate: SemanticCandidate,
    manifest: BountyManifest,
    identity: CampaignIdentity,
    state: ResearchState | None,
    sequences: tuple[Vfcs, ...],
    minimized: Mapping[str, MinimizationResult],
    seen_roots: dict[str, str],
    seen_sites: dict[str, str] | None = None,
    deployment: DeploymentIdentity | None = None,
    name_ambiguous: bool = False,
    target_address: str = "",
    target_chain_id: str = "",
    executions: tuple[dict[str, Any], ...] = (),
) -> ResearchFinding:
    seen_sites = {} if seen_sites is None else seen_sites
    deployment_key = deployment.key if deployment is not None else ""
    key = root_cause_key(candidate, deployment_key)
    finding_id = f"rf_{digest((identity.campaign_id, candidate.detector, candidate.contract, candidate.function, candidate.line, candidate.file))}"
    bound_here = bool(
        deployment is not None
        and deployment.address
        and target_address
        and deployment.address.lower() == target_address.lower()
    )
    scope = manifest.scope_of(
        contract=candidate.contract,
        file=candidate.file,
        address=target_address if bound_here else "",
        chain_id=target_chain_id if bound_here else "",
    )
    known = match_known_issue(manifest.known_issues, candidate)
    # The same detector at the same source site reached through two contracts (for
    # example an inherited function) is one root cause.
    site = f"{candidate.detector}|{candidate.file}:{candidate.line}"
    duplicate = seen_sites.get(site, "") or seen_roots.get(key, "")
    basis = ""
    if duplicate:
        basis = "same_source_site" if site in seen_sites else "same_root_cause"
    seen_sites.setdefault(site, finding_id)
    seen_roots.setdefault(key, finding_id)
    related = _evidence_for(state, candidate)
    quality = _best_quality(related)
    poc = _poc_status(related)
    execution = execution_summary(candidate, executions)
    if execution.strength in {"strong_candidate", "corroborated_candidate"} and poc == "none":
        poc = "local_harness_violation_unverified"
    elif execution.strength == "mechanism_observation" and poc == "none":
        poc = "mechanism_observation_unproven"
    own = tuple(
        s
        for s in sequences
        if s.derived_from.startswith(candidate.detector + "@")
        and candidate.contract in s.derived_from
    )
    severity, rationale = severity_candidate(manifest.impact_categories, candidate)
    contradictions = _contradictions(state, candidate)
    qualification = qualify(
        manifest,
        scope.status,
        known,
        duplicate,
        poc,
        bool(own),
        deployment=deployment,
        scoped_by_address=bool(scope.asset is not None and scope.asset.kind == "address"),
        name_ambiguous=name_ambiguous,
    )
    uncertainties = _uncertainties(manifest, scope.status, candidate, poc, severity)
    uncertainties = (*uncertainties, *execution.uncertainties)
    if deployment is not None and deployment.binding != "bound":
        uncertainties = (*uncertainties, f"deployment binding is {deployment.binding}")
    if name_ambiguous:
        uncertainties = (*uncertainties, "the contract name is declared in several files")
    return ResearchFinding(
        finding_id=finding_id,
        title=candidate.title,
        detector=candidate.detector,
        family=candidate.family,
        contract=candidate.contract,
        function=candidate.function,
        file=candidate.file,
        line=candidate.line,
        root_cause=candidate.summary,
        affected_asset=_asset(manifest, candidate, deployment),
        impact_claim=_impact(candidate),
        severity_candidate=severity,
        severity_rationale=rationale,
        confirmed_severity=CONFIRMED_SEVERITY,
        scope_status=scope.status.value,
        scope_reason=scope.reason,
        known_issue=known,
        duplicate_of=duplicate,
        poc_status=poc,
        preconditions=candidate.missing,
        evidence_ids=tuple(sorted(item.evidence_id for item in related)),
        evidence_quality=quality,
        open_contradictions=contradictions,
        sequence_ids=tuple(s.sequence_id for s in own),
        minimization=tuple(
            f"{s.sequence_id}:{_describe(minimized[s.sequence_id])}"
            for s in own
            if s.sequence_id in minimized
        ),
        identity=(
            ("campaign_id", identity.campaign_id),
            ("source_snapshot", identity.source_snapshot or UNKNOWN),
            ("compiler_configuration", identity.compiler_configuration or UNKNOWN),
            ("fork_reference", identity.fork_reference or UNKNOWN),
            ("program_context", identity.program_context or UNKNOWN),
            ("deployment", deployment_key or UNKNOWN),
        ),
        uncertainties=uncertainties,
        qualification=qualification,
        root_cause_key=key,
        duplicate_basis=basis,
        deployment=tuple(
            (name, str(value))
            for name, value in (deployment.as_dict().items() if deployment else ())
            if name
            in {
                "chain_id",
                "address",
                "binding",
                "proxy_kind",
                "implementation",
                "source_matches_deployed",
                "runtime_digest",
            }
        ),
        execution_status=execution.status,
        candidate_strength=execution.strength,
        property_ids=execution.property_ids,
        bundle_ids=execution.bundle_ids,
        corroboration=execution.corroboration,
    )


@dataclass(frozen=True)
class ExecutionSummary:
    status: str
    strength: str
    property_ids: tuple[str, ...]
    bundle_ids: tuple[str, ...]
    corroboration: str
    uncertainties: tuple[str, ...]


# Strongest first. Only an oracle-judged, identity-bound violation is "strong".
_STATUS_ORDER = (
    "property_violated",
    "property_held",
    "mechanism_observed",
    "failure_propagated",
    "sequence_executed_no_oracle",
    "sequence_reverted",
    "inconclusive",
    "identity_mismatch",
    "compile_failed",
    "execution_failed",
    "timeout",
    "unavailable",
    "skipped",
)


def is_strong(record: Mapping[str, Any]) -> bool:
    """Recomputed from the structured fields; a stored flag alone is never trusted."""
    return bool(
        record.get("outcome") == "property_violated"
        and record.get("declaration") == "property_under_test"
        and record.get("verdict") == "property_violated"
        and record.get("oracle_kind") not in {"", "none", None}
        and record.get("identity_status") == "bound"
        and not record.get("stale")
    )


def execution_summary(
    candidate: SemanticCandidate, executions: tuple[dict[str, Any], ...]
) -> ExecutionSummary:
    key = f"{candidate.detector}@{candidate.contract}.{candidate.function}"
    own = [item for item in executions if item.get("derived_from") == key]
    if not own:
        return ExecutionSummary("not_executed", "static_candidate", (), (), "not_applicable", ())
    fresh = [item for item in own if not item.get("stale")]
    notes: list[str] = []
    if len(fresh) < len(own):
        notes.append("some executions ran against a source set that has since changed")
    ranked = sorted(
        fresh,
        key=lambda r: (
            _STATUS_ORDER.index(r["outcome"]) if r.get("outcome") in _STATUS_ORDER else 99
        ),
    )
    status = str(ranked[0].get("outcome")) if ranked else "stale"
    strong = [item for item in fresh if is_strong(item)]
    corroboration = "not_applicable"
    if strong:
        statuses = {str(item.get("corroboration", "")) for item in strong}
        if "disagreement" in statuses:
            corroboration = "disagreement"
            notes.append("an independent path disagrees with the local violation")
        elif "corroborated_candidate" in statuses:
            corroboration = "corroborated_candidate"
        else:
            corroboration = "single_path"
    held = any(item.get("outcome") == "property_held" for item in fresh)
    if strong and held:
        notes.append("the property held on another sequence; the contradiction stays open")
    if strong and corroboration == "corroborated_candidate":
        strength = "corroborated_candidate"
    elif strong:
        strength = "strong_candidate"
    elif any(item.get("outcome") == "mechanism_observed" for item in fresh):
        strength = "mechanism_observation"
        notes.append(
            "the callee reverted and the outer call returned normally; "
            "application-level impact and intentional design remain unproven"
        )
    elif any(item.get("outcome") == "failure_propagated" for item in fresh):
        strength = "mechanism_not_reproduced"
        notes.append(
            "the outer call also reverted, so the swallowed-call mechanism was not reproduced"
        )
    elif any(item.get("outcome") == "sequence_executed_no_oracle" for item in fresh):
        strength = "weak_execution_only"
        notes.append("the sequence executed without an oracle; execution is not a finding")
    elif any(item.get("outcome") == "property_held" for item in fresh):
        strength = "property_held"
        notes.append(
            "the probe reverted and the observed state held; a held property is not a vulnerability"
        )
    elif any(item.get("outcome") == "skipped" for item in fresh):
        strength = "static_candidate"
        skipped_record = next(item for item in fresh if item.get("outcome") == "skipped")
        reason = str(
            skipped_record.get("reason")
            or skipped_record.get("reason_code")
            or "skipped before Forge"
        )
        notes.append(reason)
    else:
        strength = "static_candidate"
    if strong:
        notes.append("local source replay only; not a deployment replay; not verified")
    return ExecutionSummary(
        status=status,
        strength=strength,
        property_ids=tuple(sorted({str(i.get("property_id", "")) for i in fresh} - {""})),
        bundle_ids=tuple(sorted({str(i.get("bundle_id", "")) for i in fresh} - {""})),
        corroboration=corroboration,
        uncertainties=tuple(notes),
    )


def quality_of(candidate_strength: str, corroboration: str) -> dict[str, Any]:
    """Three separate axes: being a candidate, being corroborated, being verified.

    Static detection always yields a *candidate*. Local execution can at most make it
    a *corroborated* candidate. *Verification* is a separate human/approved process
    that nothing in the automated path performs, so it is always ``unverified`` here.
    """
    if corroboration == "disagreement":
        corroborated = "contradicted"
    else:
        corroborated = {
            "corroborated_candidate": "corroborated",
            "strong_candidate": "single_path_execution",
            "weak_execution_only": "execution_only",
            "mechanism_observation": "mechanism_only",
            "mechanism_not_reproduced": "none",
        }.get(candidate_strength, "none")
    return {
        "candidate": "static_candidate",
        "corroboration": corroborated,
        "verification": "unverified",
        "verified": False,
    }


def match_known_issue(
    issues: Iterable[KnownIssue], candidate: SemanticCandidate
) -> KnownIssueMatch | None:
    """An issue matches only on the dimensions it declares, and all of them must agree."""
    name = candidate.function.split("(")[0]
    for issue in issues:
        checks: list[tuple[str, bool]] = []
        if issue.contracts:
            checks.append(("contract", candidate.contract in issue.contracts))
        if issue.functions:
            checks.append(
                ("function", name in issue.functions or candidate.function in issue.functions)
            )
        if issue.detectors:
            checks.append(("detector", candidate.detector in issue.detectors))
        if issue.root_cause_key:
            checks.append(
                (
                    "root_cause",
                    issue.root_cause_key
                    in {legacy_root_cause_key(candidate), root_cause_key(candidate)},
                )
            )
        if checks and all(ok for _label, ok in checks):
            return KnownIssueMatch(
                issue.issue_id, issue.source, "+".join(label for label, _ in checks)
            )
    return None


def severity_candidate(
    categories: tuple[ImpactCategory, ...], candidate: SemanticCandidate
) -> tuple[str, str]:
    """Severity comes from the program's categories. Without one it stays unknown."""
    if not categories:
        return UNKNOWN, "the program supplied no impact categories, so no severity is proposed"
    matched = [item for item in categories if set(item.tags) & set(candidate.impact_tags)]
    if not matched:
        return UNKNOWN, "no program impact category matches this candidate's impact tags"
    best = sorted(matched, key=lambda item: (-item.weight, item.name))[0]
    return best.severity, (
        f"program category {best.name} lists {best.severity}; this is an upper bound "
        "if the impact is demonstrated and not a confirmed severity"
    )


def qualify(
    manifest: BountyManifest,
    scope: ScopeStatus,
    known: KnownIssueMatch | None,
    duplicate: str,
    poc: str,
    has_sequence: bool,
    *,
    deployment: DeploymentIdentity | None = None,
    scoped_by_address: bool = False,
    name_ambiguous: bool = False,
) -> Qualification:
    """Distinct statuses: out-of-scope, known issue, unknown scope, ambiguous
    deployment or identity, missing PoC, and report candidate. None is a safety claim."""
    reasons: list[str] = []
    if scope is ScopeStatus.OUT_OF_SCOPE:
        return Qualification(
            "out_of_scope", ("the program excludes this asset; not a safety claim",)
        )
    if known is not None:
        return Qualification(
            "known_issue",
            (f"matches known issue {known.issue_id} ({known.source}); not a safety claim",),
        )
    if scope is ScopeStatus.UNKNOWN:
        return Qualification("needs_scope", ("scope for this asset is not established",))
    if deployment is not None:
        if deployment.source_matches_deployed == "no":
            return Qualification(
                "ambiguous_deployment",
                ("the deployed runtime does not match this source build",),
            )
        if scoped_by_address and deployment.binding != "bound":
            return Qualification(
                "ambiguous_deployment",
                (
                    "in scope only through its deployed address, and the source is not "
                    f"bound to that deployment ({deployment.binding})",
                ),
            )
    if name_ambiguous:
        return Qualification(
            "ambiguous_identity",
            ("several files declare this contract name; the analyzed one may not be deployed",),
        )
    if duplicate:
        reasons.append(f"shares a root cause with {duplicate}; possible duplicate, not safe")
    if poc.startswith("mechanism_observation"):
        reasons.append(
            "the callee reverted and the outer call returned normally; "
            "application-level impact and intentional design remain unproven; "
            "this is not a validated security-property violation"
        )
        return Qualification("mechanism_observation", tuple(reasons))
    if manifest.poc_requirement is PocRequirement.REQUIRED and poc == "none":
        reasons.append("the program requires a proof of concept and none exists")
        return Qualification("needs_poc", tuple(reasons))
    if poc.startswith("local_harness"):
        reasons.append(
            "the proof of concept is a local source replay of a property violation; "
            "not a deployment replay and not verified"
        )
    if manifest.poc_requirement is PocRequirement.UNKNOWN:
        reasons.append("the program's proof-of-concept requirement is unknown")
    if not has_sequence:
        reasons.append("no call sequence has been derived for this candidate")
    if deployment is not None and deployment.binding != "bound":
        reasons.append(f"deployment binding is {deployment.binding}; not proven deployed")
    return Qualification("report_candidate", tuple(reasons))


def _evidence_for(state: ResearchState | None, candidate: SemanticCandidate) -> list[EvidenceItem]:
    if state is None:
        return []
    identity = f"{candidate.contract}.{candidate.function}"
    return [
        item
        for item in state.evidence
        if item.attrs.get("identity") != "mismatch"
        and (
            item.attrs.get("detector") == candidate.detector
            and item.identity_key == identity
            or (item.identity_key == identity and item.attrs.get("category") in _RUNTIME_CATEGORIES)
        )
    ]


def _best_quality(items: list[EvidenceItem]) -> str:
    order = [
        EvidenceQuality.CORROBORATED,
        EvidenceQuality.CANDIDATE,
        EvidenceQuality.OBSERVATION,
        EvidenceQuality.INCOMPLETE,
        EvidenceQuality.UNKNOWN,
    ]
    present = {item.quality for item in items}
    for quality in order:
        if quality in present:
            return quality.value
    return "static_candidate" if not items else UNKNOWN


def _poc_status(related: list[EvidenceItem]) -> str:
    runtime = [
        item
        for item in related
        if item.attrs.get("category") in _RUNTIME_CATEGORIES and item.polarity == "positive"
    ]
    if runtime:
        return "runtime_observation_unverified"
    return "none"


def _contradictions(state: ResearchState | None, candidate: SemanticCandidate) -> tuple[str, ...]:
    if state is None:
        return ()
    identity = f"{candidate.contract}.{candidate.function}"
    return tuple(
        sorted(
            item.contradiction_id
            for item in state.contradictions
            if item.status == "open" and item.identity_key == identity
        )
    )


def _asset(
    manifest: BountyManifest,
    candidate: SemanticCandidate,
    identity: DeploymentIdentity | None = None,
) -> str:
    if identity is not None and identity.address:
        return f"{candidate.contract} at {identity.chain_id or '?'}:{identity.address}"
    named = [d for d in manifest.deployments if d.contract_name == candidate.contract]
    if len(named) == 1:
        return f"{candidate.contract} at {named[0].chain_id}:{named[0].address}"
    if len(named) > 1:
        return f"{candidate.contract} ({len(named)} deployments; which one is ambiguous)"
    return f"{candidate.contract} (deployment unknown)"


def _impact(candidate: SemanticCandidate) -> str:
    tags = ", ".join(candidate.impact_tags[:4]) or "unspecified"
    return f"potential effect on: {tags}; unproven until demonstrated"


def _uncertainties(
    manifest: BountyManifest,
    scope: ScopeStatus,
    candidate: SemanticCandidate,
    poc: str,
    severity: str,
) -> tuple[str, ...]:
    found = [f"missing: {item}" for item in candidate.missing[:3]]
    if scope is ScopeStatus.UNKNOWN:
        found.append("scope is not established")
    if poc == "none":
        found.append("no runtime evidence exists for this candidate")
    if severity == UNKNOWN:
        found.append("severity cannot be derived from the program policy")
    found.extend(f"manifest gap: {gap}" for gap in manifest.gaps()[:4])
    return tuple(found)


def _describe(result: MinimizationResult) -> str:
    return (
        f"{len(result.original)}->{len(result.minimized)} calls, completed={result.completed}, "
        f"reason={result.reason}"
    )
