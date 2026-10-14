"""
AI provider abstraction — Phase 4 AI Debugging Engine.

All interactions with language models go through this interface.
No AI vendor SDK is imported at the module level; implementations are lazy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.ai.health import AIHealthStatus
    from app.domain.evidence import EvidenceBundle

# ---------------------------------------------------------------------------
# Evidence types sent to the provider
# ---------------------------------------------------------------------------


@dataclass
class TestFailureEvidence:
    node_id: str
    test_file: str | None
    test_name: str
    traceback: str | None
    stdout: str | None
    stderr: str | None
    duration_seconds: float | None


@dataclass
class StaticFindingEvidence:
    category: str
    severity: str
    confidence: str
    file_path: str
    line: int
    message: str
    explanation: str
    evidence: str


@dataclass
class SourceFileEvidence:
    file_path: str
    content: str
    language: str = "python"


@dataclass
class DebuggingRequest:
    """Everything the AI provider receives to produce hypotheses."""

    project_name: str
    repository_path: str
    failing_tests: list[TestFailureEvidence]
    static_findings: list[StaticFindingEvidence]
    source_files: list[SourceFileEvidence]
    max_hypotheses: int = 3
    # Combined evidence for later security work. Never treated as verification.
    evidence_bundle: EvidenceBundle | None = field(default=None)


# ---------------------------------------------------------------------------
# Structured hypothesis / response models
# ---------------------------------------------------------------------------


@dataclass
class HypothesisResult:
    root_cause: str
    confidence: float  # 0.0–1.0
    confidence_label: str  # confirmed | highly_likely | likely | possible | insufficient_evidence
    affected_files: list[str]
    affected_symbols: list[str]
    evidence_summary: list[str]
    contradictory_evidence: list[str]
    reproduction_strategy: str
    recommended_tests: list[str]
    explanation: str


@dataclass
class AIUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ProviderResponse:
    """Raw output from the AI provider, before storage."""

    hypotheses: list[HypothesisResult]
    provider: str
    model: str
    usage: AIUsage
    duration_seconds: float
    raw_json: str = ""
    error: str | None = None


@dataclass
class TestGenerationResponse:
    """Structured response from generate_tests()."""

    candidates_json: str
    provider: str
    model: str
    usage: AIUsage
    duration_seconds: float
    error: str | None = None


@dataclass
class StructuredTextResponse:
    """Raw JSON response from generate_structured()."""

    content: str
    provider: str
    model: str
    usage: AIUsage
    duration_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class AICapabilities:
    """Provider-specific features. Absence of a capability is not an error.

    Callers must consult this object instead of assuming every backend
    matches OpenAI. Local models may advertise thinking, long context, or
    local execution without native JSON response_format support.
    """

    chat: bool = True
    structured_output: bool = False
    tool_calls: bool = False
    thinking: bool = False
    thinking_can_disable: bool = True
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    supports_local_models: bool = False
    notes: tuple[str, ...] = ()
    text_generation: bool = True
    structured_generation: bool = False
    reasoning: bool = False
    vision: bool = False
    long_context: bool = False
    local_execution: bool = False
    native_json_response_format: bool = False

    def advertised(self) -> tuple[str, ...]:
        labels: list[str] = []
        if self.text_generation or self.chat:
            labels.append("text_generation")
        if self.structured_generation or self.structured_output:
            labels.append("structured_generation")
        if self.reasoning or self.thinking:
            labels.append("reasoning")
        if self.tool_calls:
            labels.append("tool_calling")
        if self.vision:
            labels.append("vision")
        if self.long_context:
            labels.append("long_context")
        if self.local_execution or self.supports_local_models:
            labels.append("local_execution")
        if self.thinking:
            labels.append("thinking")
        return tuple(labels)


@dataclass
class CompletionRequest:
    """Provider-neutral completion request.

    Debugging-specific methods remain on :class:`LLMProvider`. This type is
    the foundation for later security analysis, evidence review, and reports.
    Tool calling and thinking are reserved; providers may ignore those fields.
    """

    system_prompt: str = ""
    user_message: str = ""
    json_mode: bool = False
    thinking: bool | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    tools: Sequence[Mapping[str, Any]] | None = None


@dataclass
class CompletionResponse:
    """Provider-neutral completion result."""

    content: str
    provider: str
    model: str
    usage: AIUsage
    duration_seconds: float
    thinking: str | None = None
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    error: str | None = None


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract interface for all language-model backends.

    Implementations must NOT log API keys or full prompt contents at any level.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier e.g. 'openai', 'anthropic', 'mock'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def analyze(self, request: DebuggingRequest) -> ProviderResponse:
        """Run AI analysis and return structured hypotheses.

        Must never raise on transient errors; return ProviderResponse with
        error field set instead so callers can distinguish failure types.
        """
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if this provider is usable (key set, reachable, etc.)."""
        ...

    @abstractmethod
    async def generate_tests(self, system_prompt: str, user_message: str) -> TestGenerationResponse:
        """Generate test candidates from evidence.

        system_prompt contains BugForge instructions (trusted).
        user_message contains evidence + [REPOSITORY_DATA] tagged untrusted content.
        Returns a TestGenerationResponse with the raw JSON candidates.
        """
        ...

    @abstractmethod
    async def generate_structured(
        self, system_prompt: str, user_message: str
    ) -> StructuredTextResponse:
        """Generic structured generation for reproduction planning and other tasks."""
        ...

    @abstractmethod
    async def generate_patch(self, system_prompt: str, user_message: str) -> StructuredTextResponse:
        """Generate a repair patch from evidence.

        system_prompt contains BugForge instructions (trusted).
        user_message contains evidence + [REPOSITORY_DATA] tagged untrusted content.
        Returns a StructuredTextResponse whose content is a JSON patch plan.
        """
        ...

    def capabilities(self) -> AICapabilities:
        """Advertise what this backend actually supports."""
        return AICapabilities(chat=True, thinking_can_disable=True)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generic completion used by future non-debugging workflows.

        Default implementation reuses :meth:`generate_structured` so existing
        OpenAI, Anthropic, and mock providers keep working without a rewrite.
        Tool calls are not executed in this phase.
        """
        response = await self.generate_structured(request.system_prompt, request.user_message)
        return CompletionResponse(
            content=response.content,
            provider=response.provider,
            model=response.model,
            usage=response.usage,
            duration_seconds=response.duration_seconds,
            error=response.error,
        )

    async def health(self) -> AIHealthStatus:
        """Return a health report. Implementations must never include API keys."""
        from app.ai.health import AIHealthStatus, default_capabilities

        available = await self.is_available()
        return AIHealthStatus(
            provider=self.provider_name,
            model=self.model_name,
            is_local=False,
            reachable=available,
            configured=available,
            model_available=None,
            error=None if available else "provider unavailable",
            capabilities=default_capabilities() if available else None,
        )
