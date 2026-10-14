"""Rate-limited :class:`LLMClientProtocol` wrapper."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any, TypeVar

from ..llm.images import ImageGenerationResponse
from ..llm.messages import Conversation, MessageInput
from ..llm.protocol import LLMClientProtocol
from ..llm.wrapper import LLMClientWrapper
from ..tools import ToolDialect, ToolExecutor, ToolRegistry
from ..tools.records import ToolCallResponse
from .protocol import RateLimiterProtocol

T = TypeVar("T")


class RateLimitedLLMClient(LLMClientWrapper):
    """LLM client wrapper that acquires a permit before each call.

    Every public method calls ``await limiter.acquire(cost)`` before
    delegating to the inner client. The default ``cost`` is ``1.0``
    per call, but callers can supply a custom per-method cost map
    (e.g. ``{"generate_image": 5.0}``) to reflect the relative
    expense of each operation.

    Args:
        inner: Any object satisfying :class:`LLMClientProtocol`.
        limiter: Any object satisfying :class:`RateLimiterProtocol`. Often a
            :class:`TokenBucketRateLimiter`, but could be backed by
            Redis or an external service.
        cost_per_method: Optional ``{method_name: cost}`` overrides.
            Missing methods fall back to ``default_cost``.
        default_cost: Cost used for any method not present in
            *cost_per_method*. Defaults to ``1.0``.
    """

    def __init__(
        self,
        *,
        inner: LLMClientProtocol,
        limiter: RateLimiterProtocol,
        cost_per_method: Mapping[str, float] | None = None,
        default_cost: float = 1.0,
    ) -> None:
        super().__init__(inner=inner)
        self.limiter = limiter
        self._cost_per_method = dict(cost_per_method or {})
        self._default_cost = default_cost

    def _cost(self, method_name: str) -> float:
        return self._cost_per_method.get(method_name, self._default_cost)

    async def complete(
        self,
        messages: MessageInput,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Acquire a permit for ``complete`` then forward to the inner client."""
        await self.limiter.acquire(self._cost("complete"))
        return await self._inner.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

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
        """Acquire a permit for ``complete_structured`` then forward."""
        await self.limiter.acquire(self._cost("complete_structured"))
        return await self._inner.complete_structured(
            messages,
            response_model,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

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
        """Acquire a single permit for the whole tool-using turn.

        Note: tool follow-up calls made by the inner client are **not**
        re-throttled here. If you want per-iteration throttling, wrap
        the inner client's tool-execution path instead.
        """
        await self.limiter.acquire(self._cost("complete_with_tools"))
        return await self._inner.complete_with_tools(
            messages,
            tools,
            tool_executor=tool_executor,
            dialect=dialect,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
            **kwargs,
        )

    async def stream(
        self,
        messages: MessageInput,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Acquire a permit for ``stream`` then yield from the inner stream."""
        await self.limiter.acquire(self._cost("stream"))
        async for chunk in self._inner.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk

    async def continue_conversation(
        self,
        conversation: Conversation,
        user_message: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Acquire a permit for ``continue_conversation`` then forward."""
        await self.limiter.acquire(self._cost("continue_conversation"))
        return await self._inner.continue_conversation(
            conversation,
            user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def loglikelihood(
        self,
        context: str,
        continuation: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[float, bool]:
        """Acquire a permit for ``loglikelihood`` then forward."""
        await self.limiter.acquire(self._cost("loglikelihood"))
        return await self._inner.loglikelihood(
            context,
            continuation,
            model=model,
            max_tokens=max_tokens,
            **kwargs,
        )

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
        """Acquire a permit for ``generate_image`` then forward.

        Image generation typically costs more than text completions;
        pass a higher cost via ``cost_per_method={"generate_image": …}``
        to reflect that.
        """
        await self.limiter.acquire(self._cost("generate_image"))
        return await self._inner.generate_image(
            prompt,
            model=model,
            n=n,
            size=size,
            quality=quality,
            style=style,
            response_format=response_format,
            **kwargs,
        )


__all__ = ["RateLimitedLLMClient"]
