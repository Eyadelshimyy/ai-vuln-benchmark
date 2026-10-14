"""Protocol defining how a budget tracker is consumed by wrappers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BudgetTrackerProtocol(Protocol):
    """Structural type for cost trackers consumed by :class:`BudgetedLLMClient`.

    Implementations are expected to be lightweight, in-process bookkeepers.
    They are *not* concurrency-safe across multiple processes; if you need
    cross-process budgets, persist spend yourself and rebuild the tracker
    on startup.

    Methods
    -------
    can_afford
        Predicate evaluated *before* a request is dispatched. Trackers
        that gate on call count or a hard limit should refuse early
        rather than after the spend has occurred. ``estimate`` is the
        caller's best guess at the upcoming cost; trackers may ignore
        it.
    charge
        Records actual spend after a request completes. ``details`` is
        a free-form mapping (e.g. ``{"input_tokens": 42, "output_tokens": 11,
        "model": "openai/gpt-4o"}``) so trackers can compute the true
        cost from raw provider usage data when an estimate is not enough.
        Must raise :class:`~ellements.core.exceptions.BudgetExceededError`
        when the recorded charge pushes spend past the configured cap.
    remaining
        Returns the headroom left under the cap, or ``math.inf`` for
        unlimited trackers. Useful for diagnostics and for callers that
        want to short-circuit large batches before they start.
    """

    def can_afford(self, estimate: float = 0.0) -> bool:
        """Return ``True`` if ``estimate`` cost is plausibly affordable."""
        ...

    def charge(self, cost: float, *, details: dict[str, Any] | None = None) -> None:
        """Record ``cost`` against the budget, raising if the cap is exceeded."""
        ...

    def remaining(self) -> float:
        """Return the unspent portion of the budget (``math.inf`` if uncapped)."""
        ...


__all__ = ["BudgetTrackerProtocol"]
