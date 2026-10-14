"""Tests for :mod:`ellements.core.budgeting`."""

from __future__ import annotations

from typing import Any

import pytest
from ellements.core.budgeting import (
    BudgetedLLMClient,
    BudgetTrackerProtocol,
    CallCountBudget,
    TokenBudget,
    TokenPrice,
)
from ellements.core.exceptions import BudgetExceededError

# ── Protocol conformance ──────────────────────────────────────────────


def test_call_count_budget_satisfies_protocol() -> None:
    assert isinstance(CallCountBudget(limit=1), BudgetTrackerProtocol)


def test_token_budget_satisfies_protocol() -> None:
    assert isinstance(TokenBudget(limit=1.0), BudgetTrackerProtocol)


# ── CallCountBudget ───────────────────────────────────────────────────


def test_call_count_rejects_nonpositive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        CallCountBudget(limit=0)


def test_call_count_charge_increments_until_limit() -> None:
    budget = CallCountBudget(limit=2)
    budget.charge(0.0)
    budget.charge(0.0)
    assert budget.spent == 2
    assert budget.remaining() == 0.0
    with pytest.raises(BudgetExceededError):
        budget.charge(0.0)


def test_call_count_can_afford_reflects_state() -> None:
    budget = CallCountBudget(limit=1)
    assert budget.can_afford() is True
    budget.charge(0.0)
    assert budget.can_afford() is False


# ── TokenBudget ───────────────────────────────────────────────────────


def test_token_budget_rejects_nonpositive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        TokenBudget(limit=0.0)


def test_token_budget_charges_from_token_details() -> None:
    budget = TokenBudget(
        limit=10.0,
        prices={"openai/gpt-4o": TokenPrice(input=0.01, output=0.03)},
    )
    budget.charge(
        0.0,
        details={"model": "openai/gpt-4o", "input_tokens": 100, "output_tokens": 50},
    )
    # 100 * 0.01 + 50 * 0.03 = 2.5
    assert pytest.approx(budget.spent) == 2.5
    assert pytest.approx(budget.remaining()) == 7.5


def test_token_budget_raises_when_charge_exceeds_limit() -> None:
    budget = TokenBudget(
        limit=1.0,
        prices={"openai/gpt-4o": TokenPrice(input=0.01, output=0.03)},
    )
    with pytest.raises(BudgetExceededError) as excinfo:
        budget.charge(
            0.0,
            details={"model": "openai/gpt-4o", "input_tokens": 100, "output_tokens": 100},
        )
    err = excinfo.value
    assert err.limit == 1.0
    assert err.spent == 0.0
    assert pytest.approx(err.attempted) == 4.0


def test_token_budget_falls_back_to_default_price() -> None:
    budget = TokenBudget(
        limit=10.0,
        default_price=TokenPrice(input=0.005, output=0.01),
    )
    budget.charge(
        0.0,
        details={"model": "some/unknown-model", "input_tokens": 200, "output_tokens": 100},
    )
    # 200 * 0.005 + 100 * 0.01 = 2.0
    assert pytest.approx(budget.spent) == 2.0


def test_token_budget_unknown_model_with_no_default_raises_keyerror() -> None:
    budget = TokenBudget(limit=10.0)
    with pytest.raises(KeyError, match="no price configured"):
        budget.charge(
            0.0,
            details={"model": "openai/gpt-4o", "input_tokens": 5, "output_tokens": 5},
        )


def test_token_budget_uses_fallback_cost_when_no_details() -> None:
    budget = TokenBudget(limit=10.0)
    budget.charge(3.0, details=None)
    assert pytest.approx(budget.spent) == 3.0


def test_token_budget_can_afford_factors_in_estimate() -> None:
    budget = TokenBudget(limit=10.0)
    budget.charge(8.0, details=None)
    assert budget.can_afford(1.0) is True
    assert budget.can_afford(2.0) is True  # exactly at limit
    assert budget.can_afford(3.0) is False


# ── BudgetedLLMClient ─────────────────────────────────────────────────


class _FakeLLM:
    """Minimal LLMClientProtocol stub."""

    model: str = "fake/model"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, messages: Any, **kwargs: Any) -> str:
        self.calls.append("complete")
        return "ok"

    async def complete_structured(self, messages: Any, response_model: type, **kwargs: Any) -> Any:
        self.calls.append("complete_structured")
        return response_model()

    async def complete_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("complete_with_tools")
        return ("ok", [])

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("stream")
        async def _gen() -> Any:
            yield "x"
        async for chunk in _gen():
            yield chunk

    async def continue_conversation(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append("continue_conversation")
        return "ok"

    async def loglikelihood(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("loglikelihood")
        return (0.0, True)

    async def generate_image(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("generate_image")
        return "img"


@pytest.mark.asyncio
async def test_budgeted_client_charges_after_complete() -> None:
    inner = _FakeLLM()
    budget = CallCountBudget(limit=2)
    client = BudgetedLLMClient(inner=inner, tracker=budget)
    await client.complete("hi")
    assert budget.spent == 1


@pytest.mark.asyncio
async def test_budgeted_client_raises_when_gate_refuses() -> None:
    inner = _FakeLLM()
    budget = CallCountBudget(limit=1)
    client = BudgetedLLMClient(inner=inner, tracker=budget)
    await client.complete("hi")  # exhausts budget
    with pytest.raises(BudgetExceededError):
        await client.complete("hi again")
    assert inner.calls == ["complete"]  # second call was blocked


@pytest.mark.asyncio
async def test_budgeted_client_respects_per_method_cost() -> None:
    inner = _FakeLLM()
    budget = TokenBudget(limit=5.0)
    client = BudgetedLLMClient(
        inner=inner,
        tracker=budget,
        cost_per_method={"generate_image": 3.0},
        default_cost=1.0,
    )
    await client.generate_image("a cat")
    await client.complete("hi")
    assert pytest.approx(budget.spent) == 4.0


@pytest.mark.asyncio
async def test_budgeted_client_stream_bypasses_charging() -> None:
    inner = _FakeLLM()
    budget = CallCountBudget(limit=1)
    client = BudgetedLLMClient(inner=inner, tracker=budget)
    chunks = [chunk async for chunk in client.stream("hi")]
    assert chunks == ["x"]
    assert budget.spent == 0  # streams are not charged


@pytest.mark.asyncio
async def test_budgeted_client_proxies_model_and_tracker() -> None:
    inner = _FakeLLM()
    budget = CallCountBudget(limit=10)
    client = BudgetedLLMClient(inner=inner, tracker=budget)
    assert client.model == "fake/model"
    assert client.tracker is budget


@pytest.mark.asyncio
async def test_budgeted_client_charges_all_chargeable_methods() -> None:
    inner = _FakeLLM()
    budget = CallCountBudget(limit=10)
    client = BudgetedLLMClient(inner=inner, tracker=budget)

    class _M:
        pass

    await client.complete("hi")
    await client.complete_structured("hi", _M)
    await client.complete_with_tools("hi", tools=[])
    await client.continue_conversation([], "more")
    await client.loglikelihood("ctx", "cont")
    await client.generate_image("a cat")
    assert budget.spent == 6
