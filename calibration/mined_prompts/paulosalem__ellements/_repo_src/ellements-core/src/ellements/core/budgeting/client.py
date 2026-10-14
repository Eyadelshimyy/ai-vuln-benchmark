"""Wrapping :class:`LLMClient` for budget-aware operation.

The wrapper enforces a *predictable* cost per method (the caller's
estimate, not provider-reported tokens) for two reasons:

1. Many callers want a hard ceiling on requests where the unit is
   simply "number of calls". :class:`CallCountBudget` covers that.
2. Provider-reported token usage only arrives *after* a response, so
   token-accurate budgeting cannot gate a call before it is made. For
   that workflow, attach a :class:`TokenBudget` as an observer on the
   wrapped :class:`LLMClient` directly — see
   :meth:`ellements.core.budgeting.TokenBudget.charge` and the
   :class:`~ellements.core.observability.LLMResponseEvent`'s ``usage``
   field, which carries provider-reported token counts.

The wrapper's job is the pre-call gate and the post-call accounting
of the *flat* estimate; combining the two layers gives both fast
failure and exact accounting.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from ..exceptions import BudgetExceededError
from ..llm.wrapper import LLMClientWrapper
from .protocol import BudgetTrackerProtocol

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..llm.images import ImageGenerationResponse
    from ..llm.messages import Conversation, MessageInput
    from ..llm.protocol import LLMClientProtocol
    from ..tools import ToolCallResponse, ToolDialect, ToolExecutor, ToolRegistry

T = TypeVar("T", bound="BaseModel")


class BudgetedLLMClient(LLMClientWrapper):
    """Decorate an LLM client so spend is tracked against a budget.

    Args:
        inner: The wrapped LLM client. Must satisfy
            :class:`~ellements.core.llm.LLMClientProtocol`.
        tracker: The :class:`BudgetTrackerProtocol` to consult before
            and after each call.
        cost_per_method: Optional override of the flat cost charged for
            specific methods. Keys are method names (``"complete"``,
            ``"complete_structured"``, ``"complete_with_tools"``,
            ``"continue_conversation"``, ``"loglikelihood"``,
            ``"generate_image"``). Methods not present use
            ``default_cost``.
        default_cost: Cost charged for any method without an explicit
            override. Defaults to ``1.0``.

    Streams (:meth:`stream`) are forwarded uncharged because their
    token usage is only known after the stream completes. If you need
    streaming inside a budget, charge it yourself once the stream is
    drained.
    """

    def __init__(
        self,
        *,
        inner: LLMClientProtocol,
        tracker: BudgetTrackerProtocol,
        cost_per_method: dict[str, float] | None = None,
        default_cost: float = 1.0,
    ) -> None:
        super().__init__(inner=inner)
        self._tracker = tracker
        self._cost_per_method = dict(cost_per_method or {})
        self._default_cost = float(default_cost)

    @property
    def tracker(self) -> BudgetTrackerProtocol:
        """The active budget tracker (useful for inspection or reset)."""
        return self._tracker

    def _cost_for(self, method: str) -> float:
        """Resolve the configured cost for ``method``."""
        return self._cost_per_method.get(method, self._default_cost)

    def _gate(self, method: str) -> float:
        """Pre-call check; returns the cost we plan to charge.

        Raises:
            BudgetExceededError: When :meth:`BudgetTrackerProtocol.can_afford`
                rejects the projected cost.
        """
        cost = self._cost_for(method)
        if not self._tracker.can_afford(cost):
            remaining = self._tracker.remaining()
            raise BudgetExceededError(
                f"budget exhausted before {method!r} "
                f"(remaining={remaining:.4f}, attempted={cost:.4f})",
                limit=remaining + cost,
                spent=remaining,
                attempted=cost,
            )
        return cost

    async def complete(
        self,
        messages: MessageInput,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Gate on the budget, run ``complete``, then charge the flat cost."""
        cost = self._gate("complete")
        result = await self._inner.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self._tracker.charge(cost, details={"method": "complete"})
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
        """Gate on the budget, run ``complete_structured``, then charge."""
        cost = self._gate("complete_structured")
        result = await self._inner.complete_structured(
            messages,
            response_model,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self._tracker.charge(cost, details={"method": "complete_structured"})
        return result

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
        """Charge once for the whole tool-using turn.

        Internal tool-iteration calls made by the inner client are not
        re-billed here; the cost reflects "one user-facing turn".
        """
        cost = self._gate("complete_with_tools")
        result = await self._inner.complete_with_tools(
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
        self._tracker.charge(cost, details={"method": "complete_with_tools"})
        return result

    def stream(
        self,
        messages: MessageInput,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Forward streamed completions to the inner client untouched.

        Streams are not charged because their token usage is only
        known after the stream completes. Charge yourself once the
        stream is drained if you need streaming inside a budget.
        """
        return self._inner.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def continue_conversation(
        self,
        conversation: Conversation,
        user_message: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Gate on the budget per turn, then forward."""
        cost = self._gate("continue_conversation")
        result = await self._inner.continue_conversation(
            conversation,
            user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self._tracker.charge(cost, details={"method": "continue_conversation"})
        return result

    async def loglikelihood(
        self,
        context: str,
        continuation: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[float, bool]:
        """Gate on the budget, run ``loglikelihood``, then charge."""
        cost = self._gate("loglikelihood")
        result = await self._inner.loglikelihood(
            context,
            continuation,
            model=model,
            max_tokens=max_tokens,
            **kwargs,
        )
        self._tracker.charge(cost, details={"method": "loglikelihood"})
        return result

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
        """Gate on the budget, run ``generate_image``, then charge.

        Pass a higher cost via ``cost_per_method={"generate_image": …}``
        to reflect the typically larger per-call price of images.
        """
        cost = self._gate("generate_image")
        result = await self._inner.generate_image(
            prompt,
            model=model,
            n=n,
            size=size,
            quality=quality,
            style=style,
            response_format=response_format,
            **kwargs,
        )
        self._tracker.charge(cost, details={"method": "generate_image"})
        return result
