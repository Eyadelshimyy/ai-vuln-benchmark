"""Concrete :class:`BudgetTrackerProtocol` implementations.

Two strategies are provided out of the box:

* :class:`CallCountBudget` — caps the *number* of LLM calls. Useful for
  test harnesses where token counts are unpredictable but you want a
  hard ceiling on traffic.

* :class:`TokenBudget` — accounts for input and output tokens against
  a per-model price table. The unit of the price is up to the caller
  (USD, micro-dollars, abstract "credits") as long as it is consistent
  across the price table and the cap.

Both trackers are thread-/coroutine-safe within a single event loop
because they perform whole-method updates under no contention assumptions
(LLM calls are coarse-grained). If you need stricter guarantees, wrap
them in an :class:`asyncio.Lock` at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..exceptions import BudgetExceededError


@dataclass(frozen=True, slots=True)
class TokenPrice:
    """Price per token for a given model, split by direction.

    Attributes:
        input: Cost per input token (units defined by the caller).
        output: Cost per output token.
    """

    input: float
    output: float


@dataclass(slots=True)
class CallCountBudget:
    """Bound the total number of completed LLM calls.

    Args:
        limit: Maximum number of charges accepted before subsequent
            :meth:`charge` invocations raise. Must be positive.

    Each :meth:`charge` increments ``spent`` by one regardless of the
    ``cost`` argument; the tracker treats every call as identical.
    """

    limit: int
    spent: int = 0

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("limit must be positive")

    def can_afford(self, estimate: float = 0.0) -> bool:
        """Return ``True`` while we have room for at least one more call."""
        return self.spent < self.limit

    def charge(self, cost: float, *, details: dict[str, Any] | None = None) -> None:
        """Record one call against the budget.

        Args:
            cost: Ignored — kept for protocol compatibility.
            details: Ignored — kept for protocol compatibility.

        Raises:
            BudgetExceededError: When the call count has already hit
                ``limit`` (i.e. this call would push it over).
        """
        if self.spent >= self.limit:
            raise BudgetExceededError(
                f"call count budget exhausted ({self.spent}/{self.limit})",
                limit=float(self.limit),
                spent=float(self.spent),
                attempted=1.0,
            )
        self.spent += 1

    def remaining(self) -> float:
        """Return the number of calls left before the cap is hit."""
        return float(self.limit - self.spent)


@dataclass(slots=True)
class TokenBudget:
    """Bound cumulative token spend across one or more models.

    Args:
        limit: Maximum cumulative cost. Caller-defined units.
        prices: Mapping of model name to :class:`TokenPrice`. Models
            not present in the table are charged the ``default_price``.
            If neither is set, calls with unknown models raise.
        default_price: Fallback price used when a charged ``model`` is
            not in ``prices``. ``None`` means "fail on unknown models".

    Attributes:
        spent: Running total of charges accumulated through :meth:`charge`.

    Notes:
        The tracker assumes ``details`` carries integer ``input_tokens``
        and ``output_tokens`` plus a ``model`` string. Missing keys
        default to ``0`` tokens, which means unaccounted calls cost
        nothing but still count toward provider rate-limits.
    """

    limit: float
    prices: dict[str, TokenPrice] = field(default_factory=dict)
    default_price: TokenPrice | None = None
    spent: float = 0.0

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("limit must be positive")

    def can_afford(self, estimate: float = 0.0) -> bool:
        """Return ``True`` while ``spent + estimate`` is within the cap."""
        return self.spent + max(estimate, 0.0) <= self.limit

    def charge(self, cost: float, *, details: dict[str, Any] | None = None) -> None:
        """Record token-derived spend against the budget.

        The actual charge is computed from ``details`` when it carries
        ``input_tokens`` (and optionally ``output_tokens`` plus
        ``model``) so providers' usage data drives the bill. Otherwise
        the supplied ``cost`` is used as a flat fallback — this is the
        path taken by :class:`BudgetedLLMClient`, which knows the
        per-method estimate but not the exact token usage.

        Args:
            cost: Flat fallback charge applied when ``details`` lacks
                token counts.
            details: Optional mapping. When it contains an
                ``input_tokens`` key the charge is computed as
                ``input_tokens * price.input + output_tokens * price.output``
                using ``prices[details["model"]]`` (falling back to
                ``default_price``); otherwise ``details`` is ignored
                for the math.

        Raises:
            BudgetExceededError: When the new charge would push ``spent``
                past ``limit``.
            KeyError: When ``details`` carries token counts for a model
                that is not in ``prices`` and ``default_price`` is unset.
        """
        if details is None or "input_tokens" not in details:
            charge = max(cost, 0.0)
        else:
            input_tokens = int(details.get("input_tokens", 0))
            output_tokens = int(details.get("output_tokens", 0))
            model = str(details.get("model", ""))
            price = self.prices.get(model, self.default_price)
            if price is None:
                raise KeyError(
                    f"no price configured for model {model!r}; set default_price "
                    "or add it to prices"
                )
            charge = input_tokens * price.input + output_tokens * price.output

        projected = self.spent + charge
        if projected > self.limit:
            raise BudgetExceededError(
                f"token budget exhausted "
                f"(spent={self.spent:.4f}, attempted={charge:.4f}, limit={self.limit:.4f})",
                limit=self.limit,
                spent=self.spent,
                attempted=charge,
            )
        self.spent = projected

    def remaining(self) -> float:
        """Return the unspent portion of the budget (clipped at zero)."""
        return max(self.limit - self.spent, 0.0)
