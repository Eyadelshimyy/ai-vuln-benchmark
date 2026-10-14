"""Structural protocols for LLM clients.

:class:`LLMClientProtocol` describes the surface that
:class:`ellements.core.LLMClient` exposes, decoupled from its concrete
implementation. Consumers that only need a subset of the API (which is
almost everyone — judges only need ``complete_structured``, strategies
only need ``complete``) can use this Protocol or a narrower one
defined locally, and tests can supply hand-rolled fakes without
inheriting from :class:`LLMClient` or pulling in litellm.

Structural typing (:pep:`544`) means **any** object with the right
methods satisfies the Protocol — the implementation does **not** need
to inherit from or know about :class:`LLMClientProtocol`. The
``@runtime_checkable`` decorator additionally enables ``isinstance``
checks at runtime, though those only verify method *names*, not
signatures.

Why a single big Protocol rather than per-method micro-Protocols?
Because the same client routinely fields many methods, and consumers
that already type-check against this Protocol can use any combination
without further imports. Tests/mocks that only implement what they
need still pass mypy (Protocols are structural, so missing methods
only error at the call site).

If you need finer-grained Protocols (interface segregation for unit
tests of small, focused functions), define them locally — they are
cheap and don't require coordination with this module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any, Protocol, TypeVar, runtime_checkable

from ..tools import ToolCallResponse, ToolDialect, ToolExecutor, ToolRegistry
from .images import ImageGenerationResponse
from .messages import Conversation, MessageInput

T = TypeVar("T")


@runtime_checkable
class LLMClientProtocol(Protocol):
    """The complete public surface of an ellements LLM client.

    Attributes:
        model: The default model identifier used when callers omit
            ``model=`` on a per-call basis.
    """

    model: str

    async def complete(
        self,
        messages: MessageInput,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a single text completion. See
        :meth:`ellements.core.LLMClient.complete`."""

    async def complete_structured(
        self,
        messages: MessageInput,
        response_model: type[T],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> T:
        """Generate a Pydantic-validated structured completion. See
        :meth:`ellements.core.LLMClient.complete_structured`."""

    async def complete_with_tools(
        self,
        messages: MessageInput,
        tools: ToolRegistry | Mapping[str, Any] | Iterable[Any],
        *,
        tool_executor: ToolExecutor | None = None,
        dialect: ToolDialect | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ToolCallResponse:
        """Run a multi-turn completion loop with tool calling. See
        :meth:`ellements.core.LLMClient.complete_with_tools`."""

    def stream(
        self,
        messages: MessageInput,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens. See
        :meth:`ellements.core.LLMClient.stream`."""

    async def continue_conversation(
        self,
        conversation: Conversation,
        user_message: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Append a user turn + assistant response. See
        :meth:`ellements.core.LLMClient.continue_conversation`."""

    async def loglikelihood(
        self,
        context: str,
        continuation: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[float, bool]:
        """Compute log-likelihood of *continuation* given *context*. See
        :meth:`ellements.core.LLMClient.loglikelihood`."""

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        style: str | None = None,
        response_format: str = "url",
        **kwargs: Any,
    ) -> ImageGenerationResponse:
        """Generate images from a text prompt. See
        :meth:`ellements.core.LLMClient.generate_image`."""


__all__ = ["LLMClientProtocol"]
