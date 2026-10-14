"""Custom exceptions for the ellements library.

All ellements exceptions inherit from :class:`EllementsError`, which itself
inherits from :class:`Exception`. The taxonomy is intentionally small and
purpose-built — callers can catch the most specific class they care about,
or :class:`EllementsError` as a single umbrella.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tools import ToolCallRecord


class EllementsError(Exception):
    """Base exception for all ellements errors."""


class LLMError(EllementsError):
    """LLM call failed irrecoverably (after any retries)."""


class ConversationError(EllementsError):
    """Conversation state or input was invalid."""


class EllementsValidationError(EllementsError):
    """A validation rule inside the ellements library failed.

    Distinct from :class:`pydantic.ValidationError`, which is raised by
    Pydantic itself. Use this exception when an ellements component
    rejects an input because it violates a library-defined contract
    (e.g. a strategy expecting a non-empty prompt, a tool refusing a
    malformed argument).
    """


class StructuredOutputUnsupportedError(LLMError):
    """The selected model/provider does not support native structured outputs.

    Raised by :meth:`LLMClient.complete_structured` when the configured
    model cannot produce a JSON object matching a Pydantic schema via
    litellm's ``response_format=PydanticModel`` mechanism.
    """


class LogprobsUnsupportedError(LLMError):
    """The selected model/provider does not expose token log-probabilities.

    Raised eagerly at benchmark startup when the requested tasks need
    log-likelihoods (e.g. MMLU, HellaSwag) but the model cannot provide
    them. This is intentionally a hard failure — silently returning
    fake ``0.0`` would invalidate benchmark results.
    """


class MaxToolIterationsError(LLMError):
    """The tool-calling loop hit its iteration cap without converging.

    The error carries the partial conversation captured up to the cap
    and the list of unresolved tool calls the model produced on the
    final turn, so callers can inspect, log, or resume the loop with
    additional executors.
    """

    def __init__(
        self,
        message: str,
        *,
        max_iterations: int,
        messages: list[dict[str, Any]],
        tool_calls_made: list[ToolCallRecord],
        unresolved_tool_calls: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.max_iterations = max_iterations
        self.messages = messages
        self.tool_calls_made = tool_calls_made
        self.unresolved_tool_calls = unresolved_tool_calls


class PromptLibraryError(EllementsError):
    """Base class for errors raised by persona / guideline / prompt libraries.

    Catch this to handle any failure to resolve a named entry in any
    prompt library — useful when several lookups happen in a row and
    you want a single fallback path.
    """


class PersonaNotFoundError(PromptLibraryError):
    """The requested persona could not be located in the active library."""


class GuidelineNotFoundError(PromptLibraryError):
    """The requested guideline could not be located in the active library."""


class PromptKeyMissingError(PromptLibraryError):
    """A strategy required a named prompt that was not supplied."""


class BudgetExceededError(EllementsError):
    """A cost budget was exhausted before a request could complete.

    Raised by :class:`ellements.core.budgeting.BudgetedLLMClient` (and
    any :class:`~ellements.core.budgeting.BudgetTrackerProtocol`
    implementation) when a request would push cumulative spend past
    the configured cap.

    The error carries the budget identifier, the limit that was hit,
    and the amount that was charged so callers can surface useful
    diagnostics or implement graceful fallbacks.
    """

    def __init__(
        self,
        message: str,
        *,
        limit: float,
        spent: float,
        attempted: float,
    ) -> None:
        super().__init__(message)
        self.limit = limit
        self.spent = spent
        self.attempted = attempted


__all__ = [
    "BudgetExceededError",
    "ConversationError",
    "EllementsError",
    "EllementsValidationError",
    "GuidelineNotFoundError",
    "LLMError",
    "LogprobsUnsupportedError",
    "MaxToolIterationsError",
    "PersonaNotFoundError",
    "PromptKeyMissingError",
    "PromptLibraryError",
    "StructuredOutputUnsupportedError",
]