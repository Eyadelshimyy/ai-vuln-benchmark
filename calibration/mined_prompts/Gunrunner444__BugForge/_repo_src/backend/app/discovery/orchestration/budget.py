"""Campaign budget construction and cost estimates.

The orchestrator shares the scheduler's `max_engines` and `max_rounds`. Every
other dimension is a smaller cap derived from them or from existing settings. A
caller may tighten a limit and can never raise one.
"""

from __future__ import annotations

from app.discovery.orchestration.model import BudgetLedger
from app.discovery.scheduler import DiscoveryScheduler
from app.discovery.sequences import MAX_EXECUTIONS

DEFAULT_RUN_CAPS = {
    "fuzz_runs": 3,
    "symbolic_runs": 2,
    "test_runs": 3,
    "economic_runs": 2,
    "protocol_runs": 2,
}
RUNTIME_SECONDS_PER_RUN = 60
HARD_MAX_ROUNDS = 16

_CAPABILITY_DIMENSION = {
    "fuzzing": "fuzz_runs",
    "symbolic_execution": "symbolic_runs",
    "test_execution": "test_runs",
    "economic_simulation": "economic_runs",
    "cross_contract_analysis": "protocol_runs",
}
_RUNTIME = frozenset({"runtime_validation", "fork_validation", "differential_validation"})


def new_ledger(
    scheduler: DiscoveryScheduler,
    *,
    max_rounds: int | None = None,
    caps: dict[str, int] | None = None,
    runtime_seconds: int | None = None,
) -> BudgetLedger:
    """Build limits from the scheduler. Requested values can only be lower."""
    engines = max(0, scheduler.max_engines)
    rounds = min(max(0, scheduler.max_rounds), HARD_MAX_ROUNDS)
    if max_rounds is not None:
        rounds = min(rounds, max(0, max_rounds))
    ceiling_seconds = runtime_seconds if runtime_seconds is not None else _scan_seconds()
    limits = {
        "engines": engines,
        "rounds": rounds,
        "executions": engines,
        "attempts": max(1, 2 * rounds + 2),
        "runtime_executions": min(MAX_EXECUTIONS, engines * 2),
        "runtime_seconds": max(0, min(int(ceiling_seconds), _scan_seconds())),
    }
    for name, default in DEFAULT_RUN_CAPS.items():
        limits[name] = min(default, engines)
    for name, value in (caps or {}).items():
        if name in limits:
            limits[name] = min(limits[name], max(0, int(value)))
    return BudgetLedger(limits=limits)


def estimate_cost(capability: str, *, runs: int = 1) -> dict[str, int]:
    """What starting one engine for this capability reserves.

    Unavailable or refused work never reaches this reservation, so it consumes
    nothing. A failure after the engine started still settles the reservation.
    """
    cost = {"attempts": 1, "engines": 1, "rounds": 1, "executions": 1}
    dimension = _CAPABILITY_DIMENSION.get(capability)
    if dimension:
        cost[dimension] = 1
    if capability in _RUNTIME:
        cost["runtime_executions"] = max(1, runs)
        cost["runtime_seconds"] = max(1, runs) * RUNTIME_SECONDS_PER_RUN
    return cost


def actual_cost(capability: str, *, executed: bool, runs: int) -> dict[str, int]:
    """Consumed cost. A run that never started consumes only an attempt."""
    if not executed:
        return {"attempts": 1}
    return estimate_cost(capability, runs=runs)


def _scan_seconds() -> int:
    from app.core.config import get_settings

    return int(get_settings().security_agent_max_scan_seconds)
