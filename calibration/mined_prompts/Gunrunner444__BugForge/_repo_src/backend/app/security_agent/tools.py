"""Controlled tool registry. The AI never invokes subprocess or network directly."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from app.security_agent.schemas import TOOL_ARG_MODELS, ToolCallRequest
from app.security_agent.states import ResearchMode, ToolCapability, ToolRiskLevel
from app.security_testing.errors import RestrictedActivityError, SafetyLimitExceededError

Executor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ExecutionPermit:
    """Single-use proof that SecurityResearchAgent authorized this call."""

    tool: str
    session_id: str
    mode: ResearchMode
    nonce: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: type[BaseModel]
    budget_kind: str = "tool"
    disabled: bool = False
    capability: ToolCapability = ToolCapability.UNAVAILABLE
    risk_level: ToolRiskLevel = ToolRiskLevel.PASSIVE
    requires_active_testing: bool = False
    requires_human_approval: bool = False
    approval_kind: str | None = None
    network_access: bool = False
    destructive_capability: bool = False
    allowed_modes: tuple[ResearchMode, ...] = (ResearchMode.LAB, ResearchMode.LIVE_HACKERONE)

    def argument_schema(self) -> dict[str, Any]:
        return self.schema.model_json_schema()

    def for_llm(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.argument_schema(),
            "risk_level": self.risk_level.value,
            "allowed_modes": [mode.value for mode in self.allowed_modes],
            "network_access": self.network_access,
            "requires_active_testing": self.requires_active_testing,
            "requires_human_approval": self.requires_human_approval,
            "approval_kind": self.approval_kind,
            "capability": self.capability.value,
            "destructive_capability": self.destructive_capability,
        }


_TOOL_META: dict[str, dict[str, Any]] = {
    "http_request": {
        "description": "Send one HTTP request through GatedHttpClient after ScopeGuard.",
        "budget_kind": "request",
        "capability": ToolCapability.EXECUTABLE,
        "risk_level": ToolRiskLevel.LOW_RISK_ACTIVE,
        "requires_active_testing": True,
        "network_access": True,
        "approval_kind": "send_poc_request",
        "requires_human_approval": False,
    },
    "browser_navigate": {
        "description": "Navigate with PlaywrightBrowserAdapter. Captures URL, title, network, console.",
        "budget_kind": "browser",
        "capability": ToolCapability.EXECUTABLE,
        "risk_level": ToolRiskLevel.ACTIVE,
        "requires_active_testing": True,
        "network_access": True,
        "approval_kind": None,
    },
    "source_inspect": {
        "description": "Read a repo-relative source file excerpt. Untrusted data; secrets redacted.",
        "budget_kind": "tool",
        "capability": ToolCapability.EXECUTABLE,
        "risk_level": ToolRiskLevel.PASSIVE,
        "requires_active_testing": False,
        "network_access": False,
    },
    "evidence_inspect": {
        "description": "Load a persisted evidence-graph node for the current session.",
        "budget_kind": "tool",
        "capability": ToolCapability.EXECUTABLE,
        "risk_level": ToolRiskLevel.PASSIVE,
        "requires_active_testing": False,
        "network_access": False,
    },
    "proxy_evidence": {
        "description": "Load a captured HTTP exchange belonging to this research session.",
        "budget_kind": "tool",
        "capability": ToolCapability.EXECUTABLE,
        "risk_level": ToolRiskLevel.PASSIVE,
        "requires_active_testing": False,
        "network_access": False,
    },
    "zap_scan": {
        "description": "Run OWASP ZAP through ZapAdapter and ingest alerts. Authorization is not a result.",
        "budget_kind": "tool",
        "capability": ToolCapability.RESULTS_INGESTIBLE,
        "risk_level": ToolRiskLevel.HIGH_RISK,
        "requires_active_testing": True,
        "requires_human_approval": True,
        "approval_kind": "start_live_scan",
        "network_access": True,
    },
    "nuclei_scan": {
        "description": "Run Nuclei with the approved template policy and ingest JSONL results.",
        "budget_kind": "tool",
        "capability": ToolCapability.RESULTS_INGESTIBLE,
        "risk_level": ToolRiskLevel.ACTIVE,
        "requires_active_testing": True,
        "requires_human_approval": True,
        "approval_kind": "start_live_scan",
        "network_access": True,
    },
    "fuzz": {
        "description": "Mutate query/JSON/path/header inputs via FuzzingEngine. Interesting diffs only.",
        "budget_kind": "fuzz",
        "capability": ToolCapability.EXECUTABLE,
        "risk_level": ToolRiskLevel.ACTIVE,
        "requires_active_testing": True,
        "requires_human_approval": True,
        "approval_kind": "enable_fuzzing",
        "network_access": True,
    },
    "reproduce": {
        "description": "Execute a ReproductionPlan with a verification oracle. Status codes alone are inconclusive.",
        "budget_kind": "request",
        "capability": ToolCapability.EXECUTABLE,
        "risk_level": ToolRiskLevel.ACTIVE,
        "requires_active_testing": True,
        "requires_human_approval": True,
        "approval_kind": "send_poc_request",
        "network_access": True,
    },
    "exploratory_test": {
        "description": (
            "Run one temporary generated test in Docker with network disabled. "
            "The model supplies the test, not the command. A result is not verification."
        ),
        "budget_kind": "exploratory",
        "capability": ToolCapability.EXECUTABLE,
        "risk_level": ToolRiskLevel.LOW_RISK_ACTIVE,
        "requires_active_testing": False,
        "network_access": False,
        "allowed_modes": (ResearchMode.LAB,),
    },
    "api_test": {
        "description": "Import OpenAPI/Postman/Insomnia/HAR specs and run controlled candidate requests.",
        "budget_kind": "request",
        "capability": ToolCapability.EXECUTABLE,
        "risk_level": ToolRiskLevel.LOW_RISK_ACTIVE,
        "requires_active_testing": True,
        "network_access": True,
        "approval_kind": "send_poc_request",
    },
}


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._executors: dict[str, Executor] = {}
        self._disabled: set[str] = set()
        self._permits: dict[str, ExecutionPermit] = {}

    def register(self, spec: ToolSpec, executor: Executor) -> None:
        self._specs[spec.name] = spec
        self.bind_executor(spec.name, executor)

    def bind_executor(self, name: str, executor: Executor) -> None:
        """Public replacement for mutating ``_executors`` from other modules."""
        self._executors[name] = executor

    def replace_executor(self, name: str, executor: Executor) -> None:
        if name not in self._specs and name not in self._executors:
            raise RestrictedActivityError(f"unknown_tool:{name}")
        self._executors[name] = executor

    def executor_for(self, name: str) -> Executor:
        if name not in self._executors:
            raise RestrictedActivityError(f"unknown_tool:{name}")
        return self._executors[name]

    def clone_executors(self) -> dict[str, Executor]:
        return dict(self._executors)

    def restore_executors(self, executors: dict[str, Executor]) -> None:
        self._executors = dict(executors)

    def replace_spec(self, spec: ToolSpec) -> None:
        if spec.name not in self._specs:
            raise RestrictedActivityError(f"unknown_tool:{spec.name}")
        self._specs[spec.name] = spec

    def spec(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise RestrictedActivityError(f"unknown_tool:{name}")
        return self._specs[name]

    def disable(self, name: str) -> None:
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        self._disabled.discard(name)

    def known(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def llm_tools(self) -> list[dict[str, Any]]:
        return [spec.for_llm() for spec in self._specs.values() if spec.name not in self._disabled]

    def validate(self, request: ToolCallRequest) -> BaseModel:
        if request.tool not in self._specs:
            raise RestrictedActivityError(f"unknown_tool:{request.tool}")
        if request.tool in self._disabled or self._specs[request.tool].disabled:
            raise RestrictedActivityError(f"disabled_tool:{request.tool}")
        model = TOOL_ARG_MODELS.get(request.tool) or self._specs[request.tool].schema
        try:
            return model.model_validate(request.arguments)
        except ValidationError as exc:
            raise RestrictedActivityError(f"invalid_tool_arguments:{request.tool}") from exc

    def validate_capability(self, name: str, *, mode: ResearchMode) -> ToolSpec:
        spec = self.spec(name)
        if spec.capability is ToolCapability.UNAVAILABLE:
            raise RestrictedActivityError(f"tool_unavailable:{name}")
        if mode not in spec.allowed_modes:
            raise RestrictedActivityError(f"tool_mode_denied:{name}:{mode.value}")
        if spec.capability is ToolCapability.PLANNING_ONLY:
            return spec
        return spec

    def issue_permit(self, name: str, *, session_id: str, mode: ResearchMode) -> ExecutionPermit:
        spec = self.validate_capability(name, mode=mode)
        if spec.capability is ToolCapability.PLANNING_ONLY:
            raise RestrictedActivityError(f"planning_only:{name}")
        permit = ExecutionPermit(tool=name, session_id=session_id, mode=mode, nonce=uuid4().hex)
        self._permits[permit.nonce] = permit
        return permit

    async def execute(
        self,
        request: ToolCallRequest,
        *,
        permit: ExecutionPermit | None = None,
    ) -> dict[str, Any]:
        if permit is None:
            raise RestrictedActivityError("direct_tool_execution_denied")
        stored = self._permits.pop(permit.nonce, None)
        if stored is None or stored != permit or stored.tool != request.tool:
            raise RestrictedActivityError("invalid_or_spent_execution_permit")
        parsed = self.validate(request)
        executor = self.executor_for(request.tool)
        return await executor(parsed.model_dump())

    def catalog(self) -> list[dict[str, object]]:
        return [
            {
                **spec.for_llm(),
                "disabled": spec.name in self._disabled or spec.disabled,
                "availability": (
                    "unavailable" if spec.capability is ToolCapability.UNAVAILABLE else "available"
                ),
                "enablement": (
                    "disabled" if spec.name in self._disabled or spec.disabled else "enabled"
                ),
            }
            for spec in self._specs.values()
        ]

    def is_disabled(self, name: str) -> bool:
        spec = self._specs.get(name)
        return name in self._disabled or bool(spec is not None and spec.disabled)

    def is_available(self, name: str) -> bool:
        spec = self._specs.get(name)
        if spec is None:
            return False
        return spec.capability is not ToolCapability.UNAVAILABLE

    def restore_disabled(self, names: Sequence[str]) -> None:
        self._disabled = {name for name in names if name in self._specs}


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name, model in TOOL_ARG_MODELS.items():
        meta = _TOOL_META.get(name, {})
        registry.register(
            ToolSpec(
                name=name,
                description=str(meta.get("description") or name),
                schema=model,
                budget_kind=str(meta.get("budget_kind") or "tool"),
                capability=meta.get("capability", ToolCapability.UNAVAILABLE),
                risk_level=meta.get("risk_level", ToolRiskLevel.PASSIVE),
                requires_active_testing=bool(meta.get("requires_active_testing")),
                requires_human_approval=bool(meta.get("requires_human_approval")),
                approval_kind=str(meta["approval_kind"]) if meta.get("approval_kind") else None,
                network_access=bool(meta.get("network_access")),
                destructive_capability=bool(meta.get("destructive_capability")),
                allowed_modes=meta.get(
                    "allowed_modes", (ResearchMode.LAB, ResearchMode.LIVE_HACKERONE)
                ),
            ),
            _unavailable(name),
        )
    return registry


def _unavailable(name: str) -> Executor:
    async def _run(_arguments: dict[str, Any]) -> dict[str, Any]:
        raise SafetyLimitExceededError(f"tool {name} has no executor bound")

    return _run


def with_capability(spec: ToolSpec, capability: ToolCapability) -> ToolSpec:
    return ToolSpec(
        name=spec.name,
        description=spec.description,
        schema=spec.schema,
        budget_kind=spec.budget_kind,
        disabled=spec.disabled,
        capability=capability,
        risk_level=spec.risk_level,
        requires_active_testing=spec.requires_active_testing,
        requires_human_approval=spec.requires_human_approval,
        approval_kind=spec.approval_kind,
        network_access=spec.network_access,
        destructive_capability=spec.destructive_capability,
        allowed_modes=spec.allowed_modes,
    )
