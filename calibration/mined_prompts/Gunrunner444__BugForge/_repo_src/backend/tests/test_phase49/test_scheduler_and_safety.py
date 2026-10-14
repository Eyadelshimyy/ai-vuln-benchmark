from __future__ import annotations

import ast
import itertools
from pathlib import Path

from app.discovery.capabilities import EngineCapability, ResultStatus
from app.discovery.orchestration import dump_state
from app.discovery.orchestration.budget import HARD_MAX_ROUNDS, estimate_cost, new_ledger
from app.discovery.orchestration.model import CampaignIdentity, ResearchState
from app.discovery.orchestration.planner import CAPABILITY_MODE
from tests.test_phase49.phase49_runtime import runtime_engine
from tests.test_phase49.phase49_support import (
    FakeEngine,
    fuzz_engine,
    make_orchestrator,
    make_request,
    make_scheduler,
    static_engine,
    symbolic_engine,
)

C = EngineCapability
PACKAGE = Path(__file__).resolve().parents[2] / "app/discovery/orchestration"


# ---- scheduler additions ------------------------------------------------------------------


def test_run_engine_counts_only_executed_runs(tmp_path) -> None:
    down = static_engine(engine_id="slither")
    down.available = False
    scheduler = make_scheduler([down, static_engine()], max_engines=2)
    request = make_request(tmp_path)
    assert scheduler.run_engine("slither", request).status is ResultStatus.UNAVAILABLE
    assert scheduler.engines_started == 0
    assert scheduler.run_engine("bugforge-static", request).executed
    assert scheduler.engines_started == 1


def test_run_engine_refuses_beyond_the_campaign_budget_and_unknown_engines(tmp_path) -> None:
    engine = static_engine()
    scheduler = make_scheduler([engine], max_engines=1)
    request = make_request(tmp_path)
    scheduler.run_engine("bugforge-static", request)
    refused = scheduler.run_engine("bugforge-static", request)
    assert refused.executed is False and refused.provenance == "budget_exhausted"
    assert len(engine.calls) == 1
    ghost = scheduler.run_engine("ghost", request)
    assert ghost.executed is False and ghost.provenance == "unregistered"


def test_decide_reuses_the_scheduler_preconditions(tmp_path) -> None:
    scheduler = make_scheduler(
        [
            FakeEngine("wake", {C.STATIC_ANALYSIS}),
            FakeEngine("foundry", {C.TEST_EXECUTION, C.FUZZING}),
            static_engine(),
        ]
    )
    request = make_request(tmp_path)
    assert scheduler.decide("wake", request).code == "optional"
    assert scheduler.decide("foundry", request).code == "framework"
    assert scheduler.decide("bugforge-static", request).action == "run"
    assert scheduler.decide("ghost", request).code == "unregistered"


def test_phase48_scheduler_behaviour_is_unchanged(tmp_path) -> None:
    scheduler = make_scheduler([static_engine(), fuzz_engine()])
    decisions = scheduler.select(make_request(tmp_path))
    assert [d.engine_id for d in decisions if d.action == "run"][0] == "bugforge-static"
    assert all(d.code for d in decisions)


# ---- budget -------------------------------------------------------------------------------


def test_the_orchestrator_shares_the_scheduler_budget_and_can_only_tighten(tmp_path) -> None:
    scheduler = make_scheduler([static_engine()], max_engines=3, max_rounds=2)
    ledger = new_ledger(scheduler)
    assert ledger.limits["engines"] == 3 and ledger.limits["rounds"] == 2
    tighter = new_ledger(scheduler, max_rounds=1, caps={"fuzz_runs": 1, "unknown": 9})
    assert tighter.limits["rounds"] == 1 and tighter.limits["fuzz_runs"] == 1
    assert "unknown" not in tighter.limits
    looser = new_ledger(scheduler, max_rounds=50, caps={"fuzz_runs": 50}, runtime_seconds=10_000)
    assert looser.limits["rounds"] == 2
    assert looser.limits["fuzz_runs"] == 3
    assert looser.limits["runtime_seconds"] <= 300
    huge = make_scheduler([], max_rounds=1000, max_engines=1000)
    assert new_ledger(huge).limits["rounds"] == HARD_MAX_ROUNDS
    assert new_ledger(make_scheduler([], max_engines=-3)).limits["engines"] == 0


def test_a_scheduler_that_already_ran_engines_starts_with_that_consumption(tmp_path) -> None:
    scheduler = make_scheduler([static_engine()], max_engines=2)
    scheduler.run_engine("bugforge-static", make_request(tmp_path))
    orch = make_orchestrator([static_engine()], make_request(tmp_path), max_engines=2)
    orch.scheduler.engines_started = 2
    orch.state.budget = orch._new_ledger()
    assert orch.state.budget.consumed["engines"] == 2
    assert orch.run().stop_reason == "execution_cap_exhausted"


def test_costs_and_actual_consumption() -> None:
    assert estimate_cost("static_analysis") == {
        "attempts": 1,
        "engines": 1,
        "rounds": 1,
        "executions": 1,
    }
    assert estimate_cost("fuzzing")["fuzz_runs"] == 1
    assert estimate_cost("differential_validation", runs=2)["runtime_seconds"] == 120
    from app.discovery.orchestration.budget import actual_cost

    assert actual_cost("fuzzing", executed=False, runs=1) == {"attempts": 1}
    assert actual_cost("fuzzing", executed=True, runs=1) == estimate_cost("fuzzing")


def test_an_engine_that_runs_longer_than_reserved_is_recorded_as_an_overrun(tmp_path) -> None:
    from tests.test_phase49.phase49_runtime import runtime_result

    request = make_request(tmp_path, sequence_id="s", runtime="true")

    def many(req):
        return runtime_result(req, mode="local", executions=8)

    orch = make_orchestrator([static_engine(), fuzz_engine(), runtime_engine(many)], request)
    state = orch.run()
    assert state.budget.consumed["runtime_executions"] == 8
    assert state.budget.overrun is True


def test_every_orchestratable_capability_has_a_known_budget_dimension() -> None:
    for capability in CAPABILITY_MODE:
        cost = estimate_cost(capability)
        assert cost["attempts"] == 1 and cost["rounds"] == 1


# ---- safety -------------------------------------------------------------------------------


def test_orchestration_imports_no_network_process_or_model_modules() -> None:
    forbidden = {
        "requests",
        "httpx",
        "aiohttp",
        "urllib",
        "urllib3",
        "socket",
        "subprocess",
        "openai",
        "anthropic",
        "litellm",
        "langchain",
        "app.ai",
        "ssl",
        "http",
        "asyncio",
    }
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not any(name == bad or name.startswith(f"{bad}.") for bad in forbidden), (
                    path.name,
                    name,
                )


def test_orchestration_has_no_api_key_or_model_endpoint_settings() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.glob("*.py")).lower()
    for needle in ("api_key", "apikey", "bearer", "private_key", "mnemonic", "chat/completions"):
        assert needle not in text


def test_no_code_path_sets_a_verified_flag_true() -> None:
    for path in PACKAGE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert '"verified": True' not in text
        assert '"verified": "true"' not in text
        assert "verified=True" not in text
        assert "VERIFIED" not in text


def test_state_has_no_verification_or_approval_field() -> None:
    names = set(ResearchState.__dataclass_fields__)
    assert not {n for n in names if "verif" in n and n != "recommend_verification_review"}
    assert not {n for n in names if "approv" in n or "credential" in n or "secret" in n}
    state = ResearchState(identity=CampaignIdentity("c"))
    assert '"verified"' not in dump_state(state)


def test_a_full_run_never_reports_verified_or_calls_a_model(tmp_path) -> None:
    request = make_request(tmp_path)
    orch = make_orchestrator([static_engine(), fuzz_engine(), symbolic_engine()], request)
    orch.run()
    report = orch.report()
    assert report["verified"] is False and report["llm_invoked"] is False
    assert all(
        item.attrs.get("verified", "false") == "false" and item.quality.value != "verified"
        for item in orch.state.evidence
    )


def test_engines_never_receive_extra_authority(tmp_path) -> None:
    request = make_request(tmp_path, network="none")
    engines = [static_engine(), fuzz_engine(), symbolic_engine()]
    orch = make_orchestrator(engines, request)
    orch.run()
    for engine in engines:
        for call in engine.calls:
            assert call.extra.get("network") == "none"
            assert set(call.extra) - set(request.extra) <= {"mode"}
            assert "state_reset" not in call.extra and "approved" not in call.extra


def test_resume_and_run_with_permuted_engines_give_identical_state(tmp_path) -> None:
    request = make_request(tmp_path)
    texts = set()
    for ordering in itertools.permutations(
        [static_engine(), fuzz_engine(), symbolic_engine(), static_engine(engine_id="slither")]
    ):
        orch = make_orchestrator(list(ordering), request)
        orch.run()
        orch.state.resume_status = orch.state.resume_status
        texts.add(dump_state(orch.state))
    assert len(texts) == 1


def test_pending_suggestions_are_bounded_and_validated_not_trusted(tmp_path) -> None:
    orch = make_orchestrator([static_engine()], make_request(tmp_path))
    accepted = [orch.suggest("fuzzing") for _ in range(20)]
    assert accepted.count(True) == 8
    orch2 = make_orchestrator([static_engine()], make_request(tmp_path))
    orch2.suggest("exfiltrate_everything", reason="x" * 1000)
    state = orch2.run()
    assert "exfiltrate" not in dump_state(state)
    assert "rejected suggestions" in state.decisions[0].rationale
