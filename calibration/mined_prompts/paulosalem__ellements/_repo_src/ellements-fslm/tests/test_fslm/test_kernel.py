from __future__ import annotations

import pytest
from ellements.fslm import FSLMEvent, FSLMKernel, machine


@pytest.mark.asyncio
async def test_kernel_applies_self_transition_effects() -> None:
    m = machine("demo", initial="working")
    with m.state("working") as s:
        s.on("continue").stay("keep_working").effects(count=1)
    spec = m.build()
    snapshot = spec.initial_snapshot()

    result = await FSLMKernel(spec).step(snapshot, FSLMEvent(type="continue"))

    assert result.status == "transitioned"
    assert result.selected_transition == "keep_working"
    assert result.new_snapshot.current_state == "working"
    assert result.new_snapshot.variables["count"] == 1


@pytest.mark.asyncio
async def test_kernel_blocks_when_guard_fails() -> None:
    m = machine("demo", initial="a")
    with m.state("a") as s:
        s.on("ready").to("b", "go").when("approved")
    m.state("b")
    spec = m.build()

    result = await FSLMKernel(spec).step(
        spec.initial_snapshot(),
        FSLMEvent(type="ready", payload={"guards": {"approved": False}}),
    )

    assert result.status == "no_transition"
    assert result.new_snapshot.current_state == "a"


@pytest.mark.asyncio
async def test_kernel_uses_seeded_probabilistic_transition() -> None:
    m = machine("demo", initial="a")
    with m.state("a") as s:
        s.on("tick").choose((0.8, "a", "stay"), (0.2, "b", "leave"))
    m.state("b")
    spec = m.build()

    result = await FSLMKernel(spec).step(
        spec.initial_snapshot(random_seed=0),
        FSLMEvent(type="tick"),
    )

    assert result.selected_transition == "leave"
    assert result.random["seed"] == 0
    assert result.new_snapshot.random_draws == 1


@pytest.mark.asyncio
async def test_kernel_reports_invariant_violation_after_transition() -> None:
    m = machine("demo", initial="a")
    with m.state("a") as s:
        s.on("go").to("b", "go_b")
    with m.state("b") as s:
        s.invariant("healthy")
    spec = m.build()

    result = await FSLMKernel(spec).step(
        spec.initial_snapshot(),
        FSLMEvent(type="go", payload={"invariants": {"healthy": False}}),
    )

    assert result.status == "blocked"
    assert result.violations == ["healthy"]
    assert result.new_snapshot.current_state == "b"
