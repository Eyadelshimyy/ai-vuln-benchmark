"""Shared base class for LLM-client wrappers.

The composition layers in :mod:`ellements.core.caching`,
:mod:`ellements.core.rate_limit`, and :mod:`ellements.core.budgeting`
all decorate a :class:`LLMClientProtocol`. They share two boring
pieces: storing the wrapped client and proxying the ``model``
attribute. This module centralizes those, plus a default
:meth:`continue_conversation` passthrough.

The base intentionally does **not** provide default passthroughs for
:meth:`complete`, :meth:`complete_structured`, :meth:`complete_with_tools`,
:meth:`stream`, :meth:`loglikelihood`, or :meth:`generate_image`.
A half-decorated wrapper that silently forwards methods it forgot to
override would defeat the purpose of the wrapper. Subclasses must
implement every method they care about, and the type checker will flag
any that are missing from :class:`LLMClientProtocol`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .messages import Conversation
    from .protocol import LLMClientProtocol


class LLMClientWrapper:
    """Storage + minimal proxying for LLM-client decorators.

    Args:
        inner: The wrapped client. Must satisfy
            :class:`~ellements.core.llm.LLMClientProtocol`.

    Subclasses are expected to expose the same methods as
    :class:`LLMClientProtocol`, overriding the ones they intercept
    and letting :meth:`continue_conversation` use the default
    passthrough provided here (or override it too if they want to
    charge/throttle multi-turn calls).
    """

    def __init__(self, *, inner: LLMClientProtocol) -> None:
        self._inner = inner

    @property
    def inner(self) -> LLMClientProtocol:
        """The wrapped client (read-only access for debugging/inspection)."""
        return self._inner

    @property
    def model(self) -> str:
        """Proxy the wrapped client's model identifier."""
        return self._inner.model

    async def continue_conversation(
        self,
        conversation: Conversation,
        user_message: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Forward to the wrapped client unchanged.

        Subclasses may override to intercept multi-turn calls (e.g. to
        charge the budget once per turn, or to count the turn toward a
        rate limit).
        """
        return await self._inner.continue_conversation(
            conversation,
            user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


__all__ = ["LLMClientWrapper"]
