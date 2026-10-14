"""Optional cost-tracking layer for :class:`LLMClient`.

This subpackage lets callers cap how much an :class:`LLMClient` is
allowed to spend before requests start failing fast — a defensive
measure for evaluation runs, agent loops, and any code path that might
otherwise burn through tokens without bound.

The design mirrors :mod:`ellements.core.caching` and
:mod:`ellements.core.rate_limit`:

* :class:`BudgetTrackerProtocol` — a structural :class:`typing.Protocol`
  callers can satisfy with their own bookkeeping.
* :class:`CallCountBudget` — a tiny tracker that counts requests.
* :class:`TokenBudget` — accounts for input and output tokens against
  a per-model price table (cost units are caller-defined; treat them
  as USD or anything else consistent).
* :class:`BudgetedLLMClient` — composition wrapper that satisfies
  :class:`~ellements.core.llm.LLMClientProtocol`, charging the active
  budget around each call and raising
  :class:`~ellements.core.exceptions.BudgetExceededError` when the cap
  is hit.

Costs are advisory and apply only to ``complete``, ``complete_structured``,
``complete_with_tools``, ``continue_conversation``, ``loglikelihood``,
and ``generate_image``. Streaming output is forwarded uncharged — the
provider's token counts are unavailable until the stream completes and
we deliberately keep this layer simple. Tag streaming usage manually
via :meth:`BudgetTrackerProtocol.charge` if you need it.
"""

from __future__ import annotations

from .client import BudgetedLLMClient
from .protocol import BudgetTrackerProtocol
from .trackers import CallCountBudget, TokenBudget, TokenPrice

__all__ = [
    "BudgetTrackerProtocol",
    "BudgetedLLMClient",
    "CallCountBudget",
    "TokenBudget",
    "TokenPrice",
]
