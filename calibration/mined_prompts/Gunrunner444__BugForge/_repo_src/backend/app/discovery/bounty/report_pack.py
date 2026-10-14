"""Deterministic bounty report pack.

The pack is a pure function of the campaign's recorded facts. The same inputs give
the same bytes. It never submits anything, it never says a finding is verified
unless the existing verification machinery did, and every uncertainty, known issue,
and unavailable tool is written down instead of being smoothed over.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.discovery.bounty.advisories import AdvisoryReport
from app.discovery.bounty.campaign import UNKNOWN, BountyManifest
from app.discovery.bounty.compiler_diff import DifferentialResult
from app.discovery.bounty.findings import ResearchFinding
from app.discovery.bounty.priority import DISCLAIMER, PriorityResult
from app.discovery.bounty.vfcs import MinimizationResult, Vfcs
from app.discovery.orchestration.codec import digest
from app.discovery.orchestration.model import CampaignIdentity, ResearchState

SECTION_TITLES: tuple[str, ...] = (
    "Summary",
    "Program and rules",
    "Scope status",
    "Asset and source identity",
    "Affected code",
    "Root cause",
    "Impact claim",
    "Severity candidate",
    "Preconditions",
    "Call sequence",
    "Proof of concept status",
    "Evidence",
    "Contradictions and negative evidence",
    "Known-issue and duplicate status",
    "Compiler and tooling context",
    "Uncertainty and limitations",
)
MAX_FINDING_REPORTS = 32
MAX_SECTION_CHARS = 1500
MAX_PACK_CHARS = 120_000
VERIFICATION_STATEMENT = (
    "Not verified. BugForge records a static candidate and any runtime observations; "
    "confirmation requires the separate verification workflow and a human decision."
)
SUBMISSION_STATEMENT = "Not submitted. BugForge never submits reports to a bounty platform."


@dataclass(frozen=True)
class FindingReport:
    finding_id: str
    title: str
    sections: tuple[tuple[str, str], ...]
    verified: bool = False
    submitted: bool = False


@dataclass(frozen=True)
class ReportPack:
    pack_id: str
    header: tuple[tuple[str, str], ...]
    overview: tuple[tuple[str, str], ...]
    reports: tuple[FindingReport, ...]
    markdown: str
    truncated: bool
    verified: bool = False
    submitted: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "pack_id": self.pack_id,
            "header": dict(self.header),
            "overview": dict(self.overview),
            "verified": self.verified,
            "submitted": self.submitted,
            "truncated": self.truncated,
            "reports": [
                {
                    "finding_id": item.finding_id,
                    "title": item.title,
                    "verified": item.verified,
                    "submitted": item.submitted,
                    "sections": [{"title": t, "body": b} for t, b in item.sections],
                }
                for item in self.reports
            ],
        }


def build_report_pack(
    *,
    manifest: BountyManifest,
    identity: CampaignIdentity,
    findings: Iterable[ResearchFinding],
    sequences: Iterable[Vfcs] = (),
    minimized: Mapping[str, MinimizationResult] | None = None,
    state: ResearchState | None = None,
    advisories: AdvisoryReport | None = None,
    differential: DifferentialResult | None = None,
    priority: PriorityResult | None = None,
    tools: Mapping[str, str] | None = None,
) -> ReportPack:
    minimized = minimized or {}
    by_id = {item.sequence_id: item for item in sequences}
    ordered = sorted(
        findings, key=lambda f: (_status_rank(f), f.contract, f.function, f.detector, f.line)
    )
    truncated = len(ordered) > MAX_FINDING_REPORTS
    reports = tuple(
        _finding_report(f, manifest, state, by_id, minimized, advisories, differential, tools)
        for f in ordered[:MAX_FINDING_REPORTS]
    )
    header = (
        ("program", f"{manifest.platform}:{manifest.program_id}"),
        ("rules_version", manifest.rules_version),
        ("program_context", manifest.identity_digest()),
        ("campaign_id", identity.campaign_id),
        ("source_snapshot", identity.source_snapshot or UNKNOWN),
        ("compiler_configuration", identity.compiler_configuration or UNKNOWN),
        ("fork_reference", identity.fork_reference or UNKNOWN),
        ("verification", VERIFICATION_STATEMENT),
        ("submission", SUBMISSION_STATEMENT),
    )
    overview = _overview(manifest, priority, advisories, differential, tools, state)
    pack_id = (
        f"rp_{digest((header, overview, [(r.finding_id, r.sections) for r in reports]), length=20)}"
    )
    markdown = _markdown(header, overview, reports)
    if len(markdown) > MAX_PACK_CHARS:
        markdown = markdown[:MAX_PACK_CHARS] + "\n\n[pack truncated at the size limit]\n"
        truncated = True
    return ReportPack(pack_id, header, overview, reports, markdown, truncated)


def _status_rank(finding: ResearchFinding) -> int:
    return {
        "report_candidate": 0,
        "needs_poc": 1,
        "needs_scope": 2,
        "known_issue": 3,
        "out_of_scope": 4,
    }.get(finding.qualification.status, 5)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_SECTION_CHARS else text[: MAX_SECTION_CHARS - 1] + "…"


def _overview(
    manifest: BountyManifest,
    priority: PriorityResult | None,
    advisories: AdvisoryReport | None,
    differential: DifferentialResult | None,
    tools: Mapping[str, str] | None,
    state: ResearchState | None,
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = [
        ("manifest_gaps", ", ".join(manifest.gaps()) or "none"),
        (
            "tools",
            ", ".join(f"{k}={v}" for k, v in sorted((tools or {}).items())) or "not recorded",
        ),
    ]
    if priority is not None:
        top = "; ".join(f"{e.identity} ({e.score})" for e in priority.ranked[:5]) or "none"
        rows.append(("priority", f"{DISCLAIMER}. Policy: {priority.policy}. Top: {top}"))
        if priority.excluded_out_of_scope:
            rows.append(("excluded_out_of_scope", ", ".join(priority.excluded_out_of_scope[:8])))
    if advisories is not None:
        applicable = [m.uid for m in advisories.matches if m.status == "applicable_candidate"]
        rows.append(
            (
                "compiler_advisories",
                f"applicable candidates: {', '.join(applicable) or 'none'}; "
                f"unknown: {advisories.unknown}; corpus: {dict(advisories.corpus_provenance).get('source', '')}",
            )
        )
    if differential is not None:
        rows.append(
            (
                "compiler_differential",
                f"{differential.status}: {differential.reason}; differences: {len(differential.differences)}",
            )
        )
    if state is not None:
        rows.append(
            ("state", f"stop={state.stop_reason or 'none'}; evidence={len(state.evidence)}")
        )
    return tuple(rows)


def _finding_report(
    finding: ResearchFinding,
    manifest: BountyManifest,
    state: ResearchState | None,
    sequences: Mapping[str, Vfcs],
    minimized: Mapping[str, MinimizationResult],
    advisories: AdvisoryReport | None,
    differential: DifferentialResult | None,
    tools: Mapping[str, str] | None,
) -> FindingReport:
    own = [sequences[i] for i in finding.sequence_ids if i in sequences]
    sections = {
        "Summary": (
            f"{finding.title} ({finding.detector}). Qualification: {finding.qualification.status}. "
            f"{VERIFICATION_STATEMENT}"
        ),
        "Program and rules": (
            f"{manifest.platform} / {manifest.program_name or manifest.program_id}; rules "
            f"{manifest.rules_version}; proof of concept {manifest.poc_requirement.value}; "
            f"testing mode {manifest.testing_mode.value}."
        ),
        "Scope status": f"{finding.scope_status}: {finding.scope_reason}",
        "Asset and source identity": f"{finding.affected_asset}; "
        + "; ".join(f"{k}={v}" for k, v in finding.identity),
        "Affected code": f"{finding.file}:{finding.line} {finding.contract}.{finding.function}",
        "Root cause": finding.root_cause,
        "Impact claim": finding.impact_claim,
        "Severity candidate": (
            f"candidate {finding.severity_candidate}; confirmed {finding.confirmed_severity}. "
            f"{finding.severity_rationale}"
        ),
        "Preconditions": "; ".join(finding.preconditions) or "none recorded",
        "Call sequence": _sequence_text(own, minimized),
        "Proof of concept status": (
            f"{finding.poc_status}. Static analysis, a fuzzer finding, a crash, a revert, a "
            "differential divergence, and an economic divergence are not verification."
        ),
        "Evidence": (
            f"quality {finding.evidence_quality}; ids: "
            f"{', '.join(finding.evidence_ids[:12]) or 'none beyond the static candidate'}"
        ),
        "Contradictions and negative evidence": _contradiction_text(finding, state),
        "Known-issue and duplicate status": _known_text(finding),
        "Compiler and tooling context": _tooling_text(advisories, differential, tools),
        "Uncertainty and limitations": "; ".join(finding.uncertainties) or "none recorded",
    }
    ordered = tuple((title, _clip(sections[title])) for title in SECTION_TITLES)
    return FindingReport(finding.finding_id, finding.title, ordered)


def _sequence_text(sequences: list[Vfcs], minimized: Mapping[str, MinimizationResult]) -> str:
    if not sequences:
        return "No call sequence was derived for this candidate."
    parts: list[str] = []
    for item in sequences[:3]:
        calls = " -> ".join(
            f"{c.actor}:{c.identity}{' [primitive: ' + c.established_by + ']' if c.primitive else ''}"
            for c in item.calls
        )
        text = f"{item.sequence_id} ({item.template}): {calls}"
        result = minimized.get(item.sequence_id)
        if result is not None:
            text += (
                f" | minimization: {len(result.original)}->{len(result.minimized)} calls, "
                f"completed={result.completed} ({result.reason}), evaluator={result.evaluator}; "
                "a reduction is not verification"
            )
        parts.append(text)
    return " || ".join(parts)


def _contradiction_text(finding: ResearchFinding, state: ResearchState | None) -> str:
    contradictions = ", ".join(finding.open_contradictions) or "none open"
    negatives = ""
    if state is not None:
        key = f"{finding.contract}.{finding.function}"
        rows = [n for n in state.negatives if n.identity_key == key]
        negatives = "; ".join(
            f"{n.capability}: {n.statement} (does not prove safety)" for n in rows[:3]
        )
    return f"Open contradictions: {contradictions}. Negative evidence: {negatives or 'none'}."


def _known_text(finding: ResearchFinding) -> str:
    known = (
        f"matches known issue {finding.known_issue.issue_id} from {finding.known_issue.source} "
        f"on {finding.known_issue.basis}"
        if finding.known_issue
        else "no known issue matches"
    )
    duplicate = finding.duplicate_of or "none"
    return (
        f"{known}; possible duplicate of: {duplicate}. A known issue, a duplicate, or an "
        "out-of-scope asset says nothing about whether the code is safe."
    )


def _tooling_text(
    advisories: AdvisoryReport | None,
    differential: DifferentialResult | None,
    tools: Mapping[str, str] | None,
) -> str:
    parts = [
        "tools: "
        + (", ".join(f"{k}={v}" for k, v in sorted((tools or {}).items())) or "not recorded")
    ]
    if advisories is not None:
        applicable = [m.uid for m in advisories.matches if m.status == "applicable_candidate"]
        parts.append(
            f"advisory corpus {dict(advisories.corpus_provenance).get('source', 'unknown')}: "
            f"applicable {', '.join(applicable) or 'none'}, unknown {advisories.unknown}"
        )
    if differential is not None:
        parts.append(f"differential {differential.status} ({differential.reason})")
    return "; ".join(parts)


def _markdown(
    header: tuple[tuple[str, str], ...],
    overview: tuple[tuple[str, str], ...],
    reports: tuple[FindingReport, ...],
) -> str:
    lines = ["# BugForge research report pack", ""]
    lines += [f"- **{key}**: {value}" for key, value in header]
    lines += ["", "## Campaign overview", ""]
    lines += [f"- **{key}**: {value}" for key, value in overview]
    for index, report in enumerate(reports, start=1):
        lines += ["", f"# Finding {index}: {report.title}", f"`{report.finding_id}`"]
        for number, (title, body) in enumerate(report.sections, start=1):
            lines += ["", f"## {number}. {title}", body]
    return "\n".join(lines) + "\n"
