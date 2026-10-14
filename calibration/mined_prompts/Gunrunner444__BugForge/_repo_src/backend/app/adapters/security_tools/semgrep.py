"""Optional Semgrep adapter.

Missing binary yields UNAVAILABLE. Scanner JSON is untrusted static evidence
and never a verified finding. Results dedupe on rule id, path, and line.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.adapters.security_tools.base import SecurityToolAdapter, SecurityToolCapability
from app.domain.evidence import Evidence, EvidenceKind
from app.domain.scope import ScopeConstraint
from app.security_testing.failures import ToolExecutionResult, ToolExecutionState
from app.security_testing.process import ProcessRunner
from app.security_testing.sanitization import wrap_untrusted


@dataclass(frozen=True)
class SemgrepHit:
    rule_id: str
    path: str
    line: int
    message: str


def _inside_root(path: str, root: str) -> bool:
    try:
        resolved = Path(path).resolve()
        base = Path(root).resolve()
    except OSError:
        return False
    if not base.is_dir():
        return False
    try:
        resolved.relative_to(base)
    except ValueError:
        return False
    return resolved == base or base in resolved.parents


def local_ruleset() -> Path:
    return Path(__file__).with_name("semgrep_rules.yaml")


def ruleset_identity() -> tuple[str, str]:
    path = local_ruleset()
    if not path.is_file():
        return "missing", ""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "bugforge-local-1", digest


def semgrep_available(binary: str | None = None) -> bool:
    if binary:
        candidate = Path(binary)
        return candidate.is_file() or shutil.which(binary) is not None
    return shutil.which("semgrep") is not None


def normalize_semgrep(payload: str) -> tuple[str, list[SemgrepHit]]:
    """Return status and deduped hits. Invalid JSON is an error, not a finding."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("semgrep output was not JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("semgrep output was not an object")
    results = data.get("results")
    if not isinstance(results, list):
        return "UNAVAILABLE", []
    seen: set[tuple[str, str, int]] = set()
    hits: list[SemgrepHit] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        rule = str(item.get("check_id") or "")
        path = str(item.get("path") or "")
        start_raw = item.get("start")
        start: dict[str, object] = start_raw if isinstance(start_raw, dict) else {}
        line = int(str(start.get("line") or 0))
        key = (rule, path, line)
        if not rule or not path or key in seen:
            continue
        seen.add(key)
        extra_raw = item.get("extra")
        raw_extra: dict[str, object] = extra_raw if isinstance(extra_raw, dict) else {}
        message = wrap_untrusted("semgrep", str(raw_extra.get("message") or "")[:300])
        hits.append(SemgrepHit(rule, path, line, message))
    return "AVAILABLE", hits


def hits_as_evidence(hits: list[SemgrepHit]) -> list[Evidence]:
    """Static observations only. This kind cannot verify a finding."""
    evidence: list[Evidence] = []
    for hit in hits:
        evidence.append(
            Evidence(
                kind=EvidenceKind.STATIC_ANALYSIS,
                source="semgrep",
                summary=f"{hit.rule_id} {hit.path}:{hit.line}",
                details=hit.message,
                metadata={
                    "rule_id": hit.rule_id,
                    "path": hit.path,
                    "line": hit.line,
                    "is_finding": False,
                    "status": "potential",
                    "verified": False,
                },
            )
        )
    return evidence


class SemgrepAdapter(SecurityToolAdapter):
    """Local project scan plus JSON ingestion. Live targets are refused.

    A missing binary is TOOL_UNAVAILABLE. Results stay static and unverified.
    """

    def __init__(self, *, binary: str | None = None, runner: ProcessRunner | None = None) -> None:
        self._binary = binary
        self._runner = runner
        self.last_result: ToolExecutionResult | None = None

    @property
    def tool_id(self) -> str:
        return "semgrep"

    @property
    def display_name(self) -> str:
        return "Semgrep"

    def is_available(self) -> bool:
        return semgrep_available(self._binary)

    def capabilities(self) -> frozenset[SecurityToolCapability]:
        return frozenset({SecurityToolCapability.PASSIVE_EVIDENCE})

    def health(self) -> ToolExecutionResult:
        if not self.is_available():
            result = ToolExecutionResult(tool="semgrep", state=ToolExecutionState.TOOL_UNAVAILABLE)
            self.last_result = result
            return result
        result = ToolExecutionResult(tool="semgrep", state=ToolExecutionState.OK, detail="semgrep")
        self.last_result = result
        return result

    def ingest_json(self, payload: str) -> list[Evidence]:
        status, hits = normalize_semgrep(payload)
        if status != "AVAILABLE":
            self.last_result = ToolExecutionResult(
                tool="semgrep", state=ToolExecutionState.TOOL_UNAVAILABLE
            )
            return [
                Evidence(
                    kind=EvidenceKind.TOOL_STATUS,
                    source="semgrep",
                    summary="semgrep tool_unavailable",
                    metadata={"state": "tool_unavailable", "is_finding": False, "verified": False},
                )
            ]
        self.last_result = ToolExecutionResult(
            tool="semgrep",
            state=ToolExecutionState.RESULTS_AVAILABLE,
            detail=f"{len(hits)} results",
        )
        return hits_as_evidence(hits)

    def _collect_passive_evidence(self, *, scope: ScopeConstraint) -> Sequence[Evidence]:
        del scope
        if not self.is_available():
            return [
                Evidence(
                    kind=EvidenceKind.TOOL_STATUS,
                    source="semgrep",
                    summary="semgrep tool_unavailable",
                    metadata={"state": "tool_unavailable", "is_finding": False, "verified": False},
                )
            ]
        return ()

    def scan_local(self, path: str, *, root: str) -> list[Evidence]:
        """Run Semgrep on one path inside the authorized repository root."""
        if "://" in path or path.startswith("//"):
            self.last_result = ToolExecutionResult(
                tool="semgrep", state=ToolExecutionState.INVALID_SCOPE
            )
            return [
                Evidence(
                    kind=EvidenceKind.TOOL_STATUS,
                    source="semgrep",
                    summary="semgrep invalid_scope",
                    metadata={"state": "invalid_scope", "is_finding": False, "verified": False},
                )
            ]
        if not _inside_root(path, root):
            self.last_result = ToolExecutionResult(
                tool="semgrep", state=ToolExecutionState.INVALID_SCOPE
            )
            return [
                Evidence(
                    kind=EvidenceKind.TOOL_STATUS,
                    source="semgrep",
                    summary="semgrep invalid_scope",
                    metadata={"state": "invalid_scope", "is_finding": False, "verified": False},
                )
            ]
        binary = self._binary if self._binary else shutil.which("semgrep")
        installed = bool(binary) and (self._binary is not None or semgrep_available())
        if not installed or self._runner is None:
            self.last_result = ToolExecutionResult(
                tool="semgrep", state=ToolExecutionState.TOOL_UNAVAILABLE
            )
            return [
                Evidence(
                    kind=EvidenceKind.TOOL_STATUS,
                    source="semgrep",
                    summary="semgrep tool_unavailable",
                    details="No process runner is configured.",
                    metadata={"state": "tool_unavailable", "is_finding": False, "verified": False},
                )
            ]
        rules_path = local_ruleset()
        version_name, rules_hash = ruleset_identity()
        if not rules_path.is_file():
            self.last_result = ToolExecutionResult(
                tool="semgrep",
                state=ToolExecutionState.TOOL_UNAVAILABLE,
                detail="local ruleset missing",
            )
            return [
                Evidence(
                    kind=EvidenceKind.TOOL_STATUS,
                    source="semgrep",
                    summary="semgrep ruleset_missing",
                    metadata={
                        "state": "tool_unavailable",
                        "ruleset": "missing",
                        "is_finding": False,
                        "verified": False,
                    },
                )
            ]
        command = str(binary or "semgrep")
        outcome = self._runner.run(
            [
                command,
                "--json",
                "--quiet",
                "--metrics=off",
                "--config",
                str(rules_path),
                str(Path(path).resolve()),
            ],
            timeout=30,
        )
        if outcome.unavailable:
            self.last_result = ToolExecutionResult(
                tool="semgrep", state=ToolExecutionState.TOOL_UNAVAILABLE, detail=outcome.stderr
            )
            return [
                Evidence(
                    kind=EvidenceKind.TOOL_STATUS,
                    source="semgrep",
                    summary="semgrep tool_unavailable",
                    metadata={"state": "tool_unavailable", "is_finding": False, "verified": False},
                )
            ]
        version = wrap_untrusted("semgrep", (outcome.stderr or "semgrep")[:120])
        evidence = self.ingest_json(outcome.stdout or "{}")
        for item in evidence:
            item.metadata["ruleset"] = version_name
            item.metadata["ruleset_sha256"] = rules_hash
            item.metadata["tool_version"] = version
        return evidence

    def _active_scan(self, *, scope: ScopeConstraint, target: str) -> Sequence[Evidence]:
        del scope, target
        self.last_result = ToolExecutionResult(
            tool="semgrep", state=ToolExecutionState.TOOL_UNAVAILABLE
        )
        return [
            Evidence(
                kind=EvidenceKind.TOOL_STATUS,
                source="semgrep",
                summary="semgrep tool_unavailable",
                details="Semgrep does not scan live targets from this adapter.",
                metadata={"state": "tool_unavailable", "is_finding": False, "verified": False},
            )
        ]
