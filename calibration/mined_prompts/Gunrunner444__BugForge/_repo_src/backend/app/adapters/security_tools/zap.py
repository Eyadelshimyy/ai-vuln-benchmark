"""OWASP ZAP adapter using the Automation Framework (YAML plans) plus execution."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from app.adapters.security_tools.base import SecurityToolAdapter, SecurityToolCapability
from app.domain.evidence import Evidence, EvidenceKind
from app.domain.scope import ScopeConstraint
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import AuthorizationDeniedError
from app.security_testing.failures import ToolExecutionResult, ToolExecutionState
from app.security_testing.process import ProcessOutcome, ProcessRunner, SubprocessRunner
from app.security_testing.sanitization import wrap_untrusted
from app.security_testing.scanner_policy import ScannerExecutionPolicy
from app.security_testing.secrets import redact_text


def _status_evidence(state: ToolExecutionState, *, detail: str = "") -> Evidence:
    return Evidence(
        kind=EvidenceKind.TOOL_STATUS,
        source="zap",
        summary=f"zap {state.value}",
        details=detail,
        metadata={"state": state.value, "is_finding": False},
    )


class ZapAdapter(SecurityToolAdapter):
    def __init__(
        self,
        *,
        engine: SecurityTestEngine | None = None,
        runner: ProcessRunner | None = None,
        policy: ScannerExecutionPolicy | None = None,
        binary: str | None = None,
    ) -> None:
        self._engine = engine
        self._runner = runner or SubprocessRunner()
        self._policy = policy or ScannerExecutionPolicy()
        self._binary = binary
        self.last_result: ToolExecutionResult | None = None

    @property
    def tool_id(self) -> str:
        return "zap"

    @property
    def display_name(self) -> str:
        return "OWASP ZAP"

    def is_available(self) -> bool:
        if self._binary:
            return True
        return shutil.which("zap.sh") is not None or shutil.which("zap-cli") is not None

    def capabilities(self) -> frozenset[SecurityToolCapability]:
        return frozenset(
            {SecurityToolCapability.PASSIVE_EVIDENCE, SecurityToolCapability.ACTIVE_SCAN}
        )

    def attach_engine(self, engine: SecurityTestEngine) -> None:
        self._engine = engine

    def automation_plan(
        self,
        target: str,
        *,
        spider: bool = True,
        active: bool = False,
        max_duration_minutes: int = 5,
    ) -> str:
        """YAML plan whose targets must already have passed ScopeGuard.

        A generated plan is planning metadata, not scanner evidence.
        """
        jobs: list[dict[str, object]] = [
            {
                "type": "passiveScan-wait",
                "parameters": {"maxDuration": max_duration_minutes},
            }
        ]
        if spider:
            jobs.insert(0, {"type": "spider", "parameters": {"url": target, "maxDuration": 2}})
        if active:
            jobs.append(
                {
                    "type": "activeScan",
                    "parameters": {"url": target, "maxDuration": max_duration_minutes},
                }
            )
        lines = [
            "env:",
            "  contexts:",
            "    - name: bugforge",
            "      urls:",
            f"        - {target}",
            "jobs:",
        ]
        for job in jobs:
            lines.append(f"  - type: {job['type']}")
            params = job.get("parameters", {})
            if isinstance(params, dict) and params:
                lines.append("    parameters:")
                for key, value in params.items():
                    lines.append(f"      {key}: {value}")
        return "\n".join(lines) + "\n"

    def ingest_alerts(
        self, payload: str | Path | dict[str, object] | list[object]
    ) -> list[Evidence]:
        data: object
        if isinstance(payload, Path):
            data = json.loads(payload.read_text(encoding="utf-8"))
        elif isinstance(payload, str):
            stripped = payload.strip()
            if not stripped:
                return []
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                return []
        else:
            data = payload
        alerts: list[object]
        if isinstance(data, dict):
            site = data.get("site")
            if isinstance(site, list) and site and isinstance(site[0], dict):
                alerts = list(site[0].get("alerts") or [])
            else:
                alerts = list(data.get("alerts") or [])
        elif isinstance(data, list):
            alerts = data
        else:
            alerts = []
        evidence: list[Evidence] = []
        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            name = redact_text(str(alert.get("alert") or alert.get("name") or "zap-alert"))
            url = redact_text(str(alert.get("url") or ""))
            evidence.append(
                Evidence(
                    kind=EvidenceKind.SCANNER,
                    source="zap",
                    summary=f"{name} @ {url}".strip(),
                    details=wrap_untrusted("zap", str(alert.get("description") or "")),
                    metadata={
                        "risk": str(alert.get("risk") or alert.get("riskcode") or ""),
                        "pluginid": str(alert.get("pluginid") or ""),
                    },
                )
            )
        return evidence

    def _collect_passive_evidence(self, *, scope: ScopeConstraint) -> Sequence[Evidence]:
        return ()

    def active_scan(self, *, scope: ScopeConstraint, target: str) -> Sequence[Evidence]:
        if self._engine is None:
            from app.adapters.scope.authorization import require_active_testing

            require_active_testing(scope, target)
            return ()
        decision = self._engine.authorize(
            target,
            method="GET",
            tool="zap",
            active=True,
            require_live_scan_approval=self._engine.mode.value == "live",
            high_risk=True,
        )
        if not decision.allowed:
            self.last_result = ToolExecutionResult(
                tool="zap",
                state=ToolExecutionState.INVALID_SCOPE
                if "SCOPE" in decision.reason.upper() or "UNKNOWN" in decision.reason
                else ToolExecutionState.SAFETY_BLOCKED,
                detail=decision.reason,
            )
            raise AuthorizationDeniedError(decision.reason, target=target, tool="zap")

        envelope = self._policy.tighten(self._engine.safety.limits)
        duration_minutes = max(1, int(envelope.max_runtime_seconds // 60) or 1)
        plan = self.automation_plan(
            target, spider=True, active=True, max_duration_minutes=duration_minutes
        )
        planned = Evidence(
            kind=EvidenceKind.SCANNER_PLAN,
            source="zap",
            summary=f"ZAP Automation Framework plan for {target}",
            details=wrap_untrusted("zap-plan", plan),
            metadata={"state": ToolExecutionState.PLANNED.value, "is_finding": False},
        )
        if self._engine.safety.dry_run or decision.dry_run:
            self.last_result = ToolExecutionResult(tool="zap", state=ToolExecutionState.DRY_RUN)
            return [planned, _status_evidence(ToolExecutionState.DRY_RUN)]

        binary = self._binary or shutil.which("zap.sh") or shutil.which("zap-cli")
        if not binary or not self.is_available():
            self.last_result = ToolExecutionResult(
                tool="zap", state=ToolExecutionState.TOOL_UNAVAILABLE
            )
            return [
                planned,
                _status_evidence(
                    ToolExecutionState.TOOL_UNAVAILABLE,
                    detail="ZAP binary not installed; plan generated but not executed",
                ),
            ]

        outcome = self._execute(binary, plan, timeout=envelope.max_runtime_seconds)
        evidence: list[Evidence] = [
            planned,
            _status_evidence(ToolExecutionState.RUNNING, detail=" ".join(outcome.argv)),
        ]
        if outcome.unavailable:
            self.last_result = ToolExecutionResult(
                tool="zap", state=ToolExecutionState.TOOL_UNAVAILABLE, detail=outcome.stderr
            )
            evidence.append(
                _status_evidence(ToolExecutionState.TOOL_UNAVAILABLE, detail=outcome.stderr)
            )
            return evidence
        if outcome.timed_out:
            self.last_result = ToolExecutionResult(
                tool="zap", state=ToolExecutionState.TIMEOUT, detail=outcome.stderr
            )
            evidence.append(_status_evidence(ToolExecutionState.TIMEOUT, detail=outcome.stderr))
            return evidence
        if outcome.returncode not in {0, None} and not outcome.stdout.strip():
            self.last_result = ToolExecutionResult(
                tool="zap",
                state=ToolExecutionState.FAILED,
                detail=outcome.stderr or f"exit {outcome.returncode}",
            )
            evidence.append(
                _status_evidence(
                    ToolExecutionState.FAILED, detail=outcome.stderr or str(outcome.returncode)
                )
            )
            return evidence

        alerts = self.ingest_alerts(outcome.stdout)
        evidence.extend(alerts)
        evidence.append(
            _status_evidence(
                ToolExecutionState.RESULTS_INGESTED,
                detail=f"ingested {len(alerts)} alerts from ZAP output",
            )
        )
        self.last_result = ToolExecutionResult(
            tool="zap",
            state=ToolExecutionState.RESULTS_INGESTED,
            detail=f"{len(alerts)} alerts",
        )
        return evidence

    def _execute(self, binary: str, plan: str, *, timeout: float) -> ProcessOutcome:
        with tempfile.TemporaryDirectory(prefix="bugforge-zap-") as tmp:
            plan_path = Path(tmp) / "plan.yaml"
            plan_path.write_text(plan, encoding="utf-8")
            argv = [binary, "-cmd", "-autorun", str(plan_path)]
            return self._runner.run(argv, timeout=timeout)

    def unavailable(self) -> ToolExecutionResult:
        return ToolExecutionResult(tool="zap", state=ToolExecutionState.TOOL_UNAVAILABLE)
