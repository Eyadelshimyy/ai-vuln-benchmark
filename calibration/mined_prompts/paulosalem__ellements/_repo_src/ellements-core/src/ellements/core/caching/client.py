"""Caching wrapper around any :class:`LLMClientProtocol`.

Wraps an inner LLM client and memoizes deterministic calls
(:meth:`complete`, :meth:`complete_structured`). Streams, tool calls,
``continue_conversation``, ``loglikelihood``, and ``generate_image``
are **not** cached and pass straight through — they either have
side-effects (tool calls, conversation mutation) or produce outputs
that can't be cleanly serialized into a key→value store (streams,
images).

The wrapper itself satisfies :class:`LLMClientProtocol`, so it slots
into anything that consumes a client structurally.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any, TypeVar

from ..llm.images import ImageGenerationResponse
from ..llm.messages import MessageInput, normalize_message_input
from ..llm.protocol import LLMClientProtocol
from ..llm.wrapper import LLMClientWrapper
from ..tools import ToolDialect, ToolExecutor, ToolRegistry
from ..tools.records import ToolCallResponse
from .cache import Cache
from .keys import build_cache_key

T = TypeVar("T")


class CachingLLMClient(LLMClientWrapper):
    """LLM client wrapper that memoizes deterministic completions.

    Args:
        inner: Any object satisfying :class:`LLMClientProtocol`. All
            uncached methods delegate to it unchanged.
        cache: Backend storing the cached responses. Any
            :class:`Cache` implementation works.
        ttl: Optional time-to-live (seconds) applied to every cache
            write made by this wrapper. ``None`` defers to the
            backend's policy.
        only_cache_low_temperature: When True (the default), only
            cache calls with ``temperature == 0.0`` — i.e. requests
            that are deterministic by construction. Set to False to
            cache everything (useful when callers know the model's
            seed/decoding policy is stable).

    The wrapper exposes ``inner``, ``cache``, and ``model`` as
    attributes so callers can introspect or replace pieces without
    rebuilding the wrapper. ``model`` proxies the inner client's
    default model and satisfies :class:`LLMClientProtocol`.
    """

    def __init__(
        self,
        *,
        inner: LLMClientProtocol,
        cache: Cache,
        ttl: float | None = None,
        only_cache_low_temperature: bool = True,
    ) -> None:
        super().__init__(inner=inner)
        self.cache = cache
        self.ttl = ttl
        self.only_cache_low_temperature = only_cache_low_temperature

    # ── Cached methods ────────────────────────────────────────────────

    async def complete(
        self,
        messages: MessageInput,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Return a cached completion when available; otherwise call
        through and cache the result.

        Bypasses the cache when ``only_cache_low_temperature`` is True
        and ``temperature > 0`` (non-deterministic). See
        :meth:`LLMClientProtocol.complete` for the underlying contract.
        """
        if not self._should_cache(temperature):
            return await self._inner.complete(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        resolved_model = model or self._inner.model
        key = build_cache_key(
            method="complete",
            model=resolved_model,
            messages=_messages_payload(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            extra=kwargs,
        )
        hit = await self.cache.get(key)
        if hit is not None and isinstance(hit.value, str):
            return hit.value
        result = await self._inner.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        await self.cache.set(key, result, ttl=self.ttl)
        return result

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
        """Return a cached structured completion when available; otherwise
        call through and cache the serialised result.

        Cached values are stored as ``model_dump()`` payloads and
        re-hydrated through *response_model* on read, so the cache is
        immune to in-process model identity changes. See
        :meth:`LLMClientProtocol.complete_structured` for the
        underlying contract.
        """
        if not self._should_cache(temperature):
            return await self._inner.complete_structured(
                messages,
                response_model,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        resolved_model = model or self._inner.model
        key = build_cache_key(
            method="complete_structured",
            model=resolved_model,
            messages=_messages_payload(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            extra=kwargs,
            response_model=response_model,
        )
        hit = await self.cache.get(key)
        if hit is not None:
            if isinstance(hit.value, response_model):
                return hit.value
            if isinstance(hit.value, dict):
                return response_model(**hit.value)
        result = await self._inner.complete_structured(
            messages,
            response_model,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        cacheable = result.model_dump() if hasattr(result, "model_dump") else result
        await self.cache.set(key, cacheable, ttl=self.ttl)
        return result

    # ── Uncached pass-through methods ─────────────────────────────────
    #
    # ``continue_conversation`` is inherited from LLMClientWrapper and
    # forwarded unchanged. The remaining methods below are also pure
    # passthroughs (no cache lookups would ever hit for stateful or
    # non-serializable outputs), but they are explicit here so the
    # type checker can see the LLMClientProtocol surface in full.

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
        """Forward tool-using completions to the inner client unchanged.

        Tool executions have side-effects, so the wrapper does not
        cache the response. See
        :meth:`LLMClientProtocol.complete_with_tools`.
        """
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

    def stream(
        self,
        messages: MessageInput,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Forward streamed completions to the inner client unchanged.

        Stream chunks cannot be cleanly re-played from a cache without
        materialising the full sequence in memory; callers should
        cache at a higher level if they need that. See
        :meth:`LLMClientProtocol.stream`.
        """
        return self._inner.stream(
            messages,
            model=model,
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
        """Forward log-likelihood scoring to the inner client unchanged.

        Scoring is deterministic per (model, context, continuation),
        but is not a frequent enough call to warrant a cache layer
        here. See :meth:`LLMClientProtocol.loglikelihood`.
        """
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
        """Forward image generation to the inner client unchanged.

        Image bytes / URLs are out of scope for a generic LLM cache;
        callers should use a dedicated image cache if needed. See
        :meth:`LLMClientProtocol.generate_image`.
        """
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

    # ── Internals ─────────────────────────────────────────────────────

    def _should_cache(self, temperature: float) -> bool:
        if not self.only_cache_low_temperature:
            return True
        return temperature == 0.0


def _messages_payload(messages: MessageInput) -> Any:
    """Normalize *messages* into a stable JSON-friendly payload."""
    return normalize_message_input(messages)


__all__ = ["CachingLLMClient"]
