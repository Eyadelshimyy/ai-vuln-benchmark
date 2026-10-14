"""Nuclei adapter. Targets must pass ScopeGuard; templates are policy-gated."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.adapters.security_tools.base import SecurityToolAdapter, SecurityToolCapability
from app.domain.evidence import Evidence, EvidenceKind
from app.domain.scope import ScopeConstraint
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import AuthorizationDeniedError
from app.security_testing.failures import ToolExecutionResult, ToolExecutionState
from app.security_testing.process import ProcessRunner, SubprocessRunner
from app.security_testing.sanitization import wrap_untrusted
from app.security_testing.scanner_policy import ScannerExecutionPolicy
from app.security_testing.secrets import redact_text


def _status_evidence(state: ToolExecutionState, *, detail: str = "") -> Evidence:
    return Evidence(
        kind=EvidenceKind.TOOL_STATUS,
        source="nuclei",
        summary=f"nuclei {state.value}",
        details=detail,
        metadata={"state": state.value, "is_finding": False},
    )


@dataclass(frozen=True)
class NucleiTemplatePolicy:
    allowed_tags: tuple[str, ...] = ("misconfig", "exposure", "tech")
    denied_tags: tuple[str, ...] = ("dos", "intrusive", "takeover")
    allowed_templates: tuple[str, ...] = ()
    denied_templates: tuple[str, ...] = ()
    max_severity: str = "medium"
    deny_all: bool = False
    deny_untagged: bool = True
    deny_unknown_severity: bool = True

    _ORDER = ("info", "low", "medium", "high", "critical")

    def allows(
        self,
        *,
        tags: Sequence[str],
        severity: str,
        template_id: str = "",
    ) -> bool:
        """Default-deny unknown/unclassified active templates."""
        if self.deny_all:
            return False
        template = template_id.strip().lower()
        if template and template in {item.lower() for item in self.denied_templates}:
            return False
        if self.allowed_templates:
            if template not in {item.lower() for item in self.allowed_templates}:
                return False
        lowered = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
        if not lowered:
            return False if self.deny_untagged else False
        if lowered & {tag.lower() for tag in self.denied_tags}:
            return False
        allowed = {tag.lower() for tag in self.allowed_tags}
        if not allowed:
            return False
        unknown = lowered - allowed - {tag.lower() for tag in self.denied_tags}
        if not (lowered & allowed):
            return False
        if unknown and not (lowered & allowed):
            return False
        sev = severity.strip().lower()
        if sev not in self._ORDER:
            return False if self.deny_unknown_severity else False
        return self._ORDER.index(sev) <= self._ORDER.index(self.max_severity.lower())


class NucleiAdapter(SecurityToolAdapter):
    def __init__(
        self,
        *,
        engine: SecurityTestEngine | None = None,
        policy: NucleiTemplatePolicy | None = None,
        runner: ProcessRunner | None = None,
        execution_policy: ScannerExecutionPolicy | None = None,
        binary: str | None = None,
    ) -> None:
        self._engine = engine
        self.policy = policy or NucleiTemplatePolicy()
        self._runner = runner or SubprocessRunner()
        self._execution = execution_policy or ScannerExecutionPolicy(
            allowed_tags=self.policy.allowed_tags,
            denied_tags=self.policy.denied_tags,
            allowed_templates=self.policy.allowed_templates,
            denied_templates=self.policy.denied_templates,
        )
        self._binary = binary
        self.last_result: ToolExecutionResult | None = None

    @property
    def tool_id(self) -> str:
        return "nuclei"

    @property
    def display_name(self) -> str:
        return "Nuclei"

    def is_available(self) -> bool:
        if self._binary:
            return True
        return shutil.which("nuclei") is not None

    def capabilities(self) -> frozenset[SecurityToolCapability]:
        return frozenset(
            {SecurityToolCapability.PASSIVE_EVIDENCE, SecurityToolCapability.ACTIVE_SCAN}
        )

    def attach_engine(self, engine: SecurityTestEngine) -> None:
        self._engine = engine

    def health(self) -> ToolExecutionResult:
        if not self.is_available():
            return ToolExecutionResult(tool="nuclei", state=ToolExecutionState.TOOL_UNAVAILABLE)
        return ToolExecutionResult(
            tool="nuclei", state=ToolExecutionState.OK, detail="nuclei on PATH"
        )

    def ingest_jsonl(self, payload: str | Path) -> list[Evidence]:
        text = Path(payload).read_text(encoding="utf-8") if isinstance(payload, Path) else payload
        evidence: list[Evidence] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            info_raw = row.get("info")
            info = info_raw if isinstance(info_raw, dict) else {}
            severity = str(info.get("severity") or row.get("severity") or "")
            tags_raw = info.get("tags")
            tags = tags_raw if isinstance(tags_raw, list) else []
            template_id = str(
                row.get("template-id") or row.get("templateID") or info.get("name") or ""
            )
            if not self.policy.allows(
                tags=[str(t) for t in tags], severity=severity, template_id=template_id
            ):
                continue
            matched = str(row.get("matched-at") or row.get("host") or row.get("url") or "")
            evidence.append(
                Evidence(
                    kind=EvidenceKind.SCANNER,
                    source="nuclei",
                    summary=redact_text(f"{template_id} {severity} {matched}").strip(),
                    details=wrap_untrusted("nuclei", json.dumps(row, default=str)[:4000]),
                    metadata={
                        "template_id": template_id,
                        "severity": severity,
                        "endpoint": matched,
                    },
                )
            )
        return evidence

    def validate_targets(self, targets: Sequence[str]) -> list[str]:
        if self._engine is None:
            raise AuthorizationDeniedError("Nuclei requires a SecurityTestEngine", tool="nuclei")
        return self._engine.runner.authorize_targets(list(targets), tool="nuclei", active=True)

    def _collect_passive_evidence(self, *, scope: ScopeConstraint) -> Sequence[Evidence]:
        return ()

    def active_scan(self, *, scope: ScopeConstraint, target: str) -> Sequence[Evidence]:
        if self._engine is None:
            from app.adapters.scope.authorization import require_active_testing

            require_active_testing(scope, target)
            return ()
        allowed = self.validate_targets([target])
        if not allowed:
            self.last_result = ToolExecutionResult(
                tool="nuclei", state=ToolExecutionState.INVALID_SCOPE
            )
            return [_status_evidence(ToolExecutionState.INVALID_SCOPE)]
        evidence: list[Evidence] = [_status_evidence(ToolExecutionState.AUTHORIZED, detail=target)]
        envelope = self._execution.tighten(self._engine.safety.limits)
        argv = self._build_argv(target, envelope)
        planned = Evidence(
            kind=EvidenceKind.SCANNER_PLAN,
            source="nuclei",
            summary=f"Nuclei command planned for {target}",
            details=wrap_untrusted("nuclei-plan", " ".join(argv)),
            metadata={"state": ToolExecutionState.PLANNED.value, "is_finding": False},
        )
        evidence.append(planned)
        if self._engine.safety.dry_run:
            self.last_result = ToolExecutionResult(tool="nuclei", state=ToolExecutionState.DRY_RUN)
            evidence.append(_status_evidence(ToolExecutionState.DRY_RUN))
            return evidence
        binary = self._binary or shutil.which("nuclei")
        if not binary or not self.is_available():
            self.last_result = ToolExecutionResult(
                tool="nuclei", state=ToolExecutionState.TOOL_UNAVAILABLE
            )
            evidence.append(
                _status_evidence(
                    ToolExecutionState.TOOL_UNAVAILABLE,
                    detail="Nuclei binary not installed; targets validated but not scanned",
                )
            )
            return evidence
        evidence.append(_status_evidence(ToolExecutionState.RUNNING))
        outcome = self._runner.run(argv, timeout=envelope.max_runtime_seconds)
        if outcome.unavailable:
            self.last_result = ToolExecutionResult(
                tool="nuclei", state=ToolExecutionState.TOOL_UNAVAILABLE, detail=outcome.stderr
            )
            evidence.append(
                _status_evidence(ToolExecutionState.TOOL_UNAVAILABLE, detail=outcome.stderr)
            )
            return evidence
        if outcome.timed_out:
            self.last_result = ToolExecutionResult(
                tool="nuclei", state=ToolExecutionState.TIMEOUT, detail=outcome.stderr
            )
            evidence.append(_status_evidence(ToolExecutionState.TIMEOUT, detail=outcome.stderr))
            return evidence
        if outcome.returncode not in {0, None} and not outcome.stdout.strip():
            self.last_result = ToolExecutionResult(
                tool="nuclei",
                state=ToolExecutionState.FAILED,
                detail=outcome.stderr or f"exit {outcome.returncode}",
            )
            evidence.append(_status_evidence(ToolExecutionState.FAILED, detail=outcome.stderr))
            return evidence
        results = self.ingest_jsonl(outcome.stdout)
        evidence.extend(results)
        evidence.append(
            _status_evidence(
                ToolExecutionState.RESULTS_AVAILABLE,
                detail=f"ingested {len(results)} nuclei findings from scanner output",
            )
        )
        self.last_result = ToolExecutionResult(
            tool="nuclei",
            state=ToolExecutionState.RESULTS_AVAILABLE,
            detail=f"{len(results)} results",
        )
        return evidence

    def _build_argv(self, target: str, envelope: ScannerExecutionPolicy) -> list[str]:
        binary = self._binary or "nuclei"
        argv = [
            binary,
            "-u",
            target,
            "-jsonl",
            "-silent",
            "-rate-limit",
            str(max(1, int(envelope.requests_per_second))),
            "-c",
            str(max(1, envelope.concurrency)),
            "-timeout",
            str(max(1, int(envelope.max_runtime_seconds))),
        ]
        tags = envelope.allowed_tags or self.policy.allowed_tags
        denied = envelope.denied_tags or self.policy.denied_tags
        if tags:
            argv.extend(["-tags", ",".join(tags)])
        if denied:
            argv.extend(["-exclude-tags", ",".join(denied)])
        templates = envelope.allowed_templates or self.policy.allowed_templates
        if templates:
            for template in templates:
                argv.extend(["-t", template])
        return argv
