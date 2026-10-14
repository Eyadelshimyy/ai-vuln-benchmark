"""Execution-stage truth: a Forge invocation is not a property evaluation."""

from __future__ import annotations

from app.discovery.bounty.research_ledger import campaign_coverage


def _summary(**kwargs: object) -> dict:
    base = {
        "files_selected": 1,
        "files_read": 1,
        "candidates": 1,
        "sequences": 1,
        "dependency_status": "traced",
    }
    base.update(kwargs)
    return campaign_coverage(**base)  # type: ignore[arg-type]


def test_compile_failures_do_not_claim_dynamic_property_execution() -> None:
    summary = _summary(
        executions={
            "vf": {
                "outcome": "compile_failed",
                "process_status": "completed",
                "compiler_status": "failed",
                "test_status": "not_run",
                "sequence_status": "not_run",
                "property_status": "not_run",
                "property_checked": False,
            }
        },
        forge_runs=3,
    )
    assert summary["foundry_invoked"] is True
    assert summary["forge_runs"] == 3
    assert summary["dynamic_foundry_execution_demonstrated"] is False
    assert summary["property_evaluation_demonstrated"] is False
    assert summary["blocker"] == "compilation failed"
    stages = summary["execution_stages"]
    assert stages["build_invoked"] == 1
    assert stages["build_failed"] == 1
    assert stages["build_succeeded"] == 0
    assert stages["test_invoked"] == 0
    assert stages["test_completed"] == 0
    assert stages["property_verdicts"] == 0
    assert summary["verified"] is False
    assert summary["local_results_are_not_production_verification"] is True


def test_a_compiled_harness_that_misses_the_probe_is_not_a_property_verdict() -> None:
    summary = _summary(
        executions={
            "vf": {
                "outcome": "execution_failed",
                "process_status": "completed",
                "compiler_status": "ok",
                "test_status": "failed",
                "sequence_status": "incomplete",
                "property_status": "not_run",
            }
        },
        forge_runs=1,
    )
    assert summary["foundry_invoked"] is True
    assert summary["dynamic_foundry_execution_demonstrated"] is False
    assert summary["property_evaluation_demonstrated"] is False
    assert summary["execution_stages"]["build_succeeded"] == 1
    assert summary["execution_stages"]["test_completed"] == 1
    assert summary["execution_stages"]["probe_reached"] == 0
    assert "did not reach" in summary["blocker"]


def test_a_property_verdict_is_distinct_from_production_verification() -> None:
    summary = _summary(
        executions={
            "vf": {
                "outcome": "property_violated",
                "process_status": "completed",
                "compiler_status": "ok",
                "test_status": "passed",
                "sequence_status": "completed",
                "property_status": "violated",
                "property_checked": True,
            }
        },
        forge_runs=1,
        persistence_status="persisted",
    )
    assert summary["blocker"] == ""
    assert summary["dynamic_foundry_execution_demonstrated"] is True
    assert summary["property_evaluation_demonstrated"] is True
    assert summary["execution_stages"]["property_verdicts"] == 1
    assert summary["results_persisted"] is True
    assert summary["verified"] is False


def test_missing_tools_are_not_executed_tests() -> None:
    summary = _summary(
        executions={"vf": {"outcome": "unavailable"}},
        forge_runs=0,
    )
    assert summary["foundry_invoked"] is False
    assert summary["dynamic_foundry_execution_demonstrated"] is False
    assert summary["execution_stages"]["tool_unavailable"] == 1
    assert summary["execution_stages"]["test_invoked"] == 0
    assert summary["execution_stages"]["test_completed"] == 0
    assert summary["blocker"] == "Foundry or solc is unavailable"


def test_timeout_and_process_failure_stay_distinct() -> None:
    timed = _summary(
        executions={
            "vf": {
                "outcome": "timeout",
                "process_status": "timeout",
                "compiler_status": "ok",
                "test_status": "not_run",
                "sequence_status": "not_run",
                "property_status": "not_run",
            }
        },
        forge_runs=1,
    )
    assert timed["dynamic_foundry_execution_demonstrated"] is False
    assert timed["execution_stages"]["test_timed_out"] == 1
    assert timed["execution_stages"]["test_completed"] == 0
    assert "timed out" in timed["blocker"]
    failed = _summary(
        executions={
            "vf": {
                "outcome": "execution_failed",
                "process_status": "spawn_failed",
                "compiler_status": "not_run",
                "test_status": "not_run",
                "sequence_status": "not_run",
                "property_status": "not_run",
            }
        },
        forge_runs=1,
    )
    assert failed["execution_stages"]["process_failed"] == 1
    assert failed["execution_stages"]["test_timed_out"] == 0
    assert failed["dynamic_foundry_execution_demonstrated"] is False


def test_a_mechanism_observation_is_not_a_property_violation() -> None:
    summary = _summary(
        executions={
            "vf": {
                "outcome": "mechanism_observed",
                "process_status": "completed",
                "compiler_status": "ok",
                "test_status": "passed",
                "sequence_status": "completed",
                "property_status": "mechanism_observed",
            }
        },
        forge_runs=1,
    )
    assert summary["dynamic_foundry_execution_demonstrated"] is True
    assert summary["property_evaluation_demonstrated"] is False
    assert summary["execution_stages"]["mechanism_observations"] == 1
    assert summary["execution_stages"]["property_verdicts"] == 0
    assert "mechanism" in summary["blocker"]
    assert summary["by_outcome"]["mechanism_observed"] == 1


def test_a_completed_test_without_an_oracle_is_not_a_property_judgment() -> None:
    summary = _summary(
        executions={
            "vf": {
                "outcome": "sequence_executed_no_oracle",
                "process_status": "completed",
                "compiler_status": "ok",
                "test_status": "passed",
                "sequence_status": "completed",
                "property_status": "no_oracle",
            }
        },
        forge_runs=1,
    )
    assert summary["dynamic_foundry_execution_demonstrated"] is True
    assert summary["property_evaluation_demonstrated"] is False
    assert "oracle" in summary["blocker"]


def test_an_identity_mismatch_invokes_no_forge_process() -> None:
    summary = _summary(
        executions={
            "vf": {
                "outcome": "identity_mismatch",
                "process_status": None,
                "compiler_status": None,
                "test_status": None,
                "sequence_status": None,
                "property_status": None,
            }
        },
        forge_runs=0,
    )
    assert summary["foundry_invoked"] is False
    assert summary["forge_runs"] == 0
    assert summary["dynamic_foundry_execution_demonstrated"] is False
    assert summary["execution_stages"]["identity_mismatch"] == 1
    assert summary["execution_stages"]["build_invoked"] == 0
    assert "identity" in summary["blocker"]


def test_a_pragma_mismatch_is_not_a_missing_tool_or_an_executed_test() -> None:
    summary = _summary(
        executions={
            "vf": {
                "outcome": "unavailable",
                "reason_code": "compiler_unavailable",
                "process_status": None,
                "compiler_status": None,
                "test_status": None,
                "sequence_status": None,
                "property_status": None,
            }
        },
        forge_runs=0,
    )
    assert summary["foundry_invoked"] is False
    assert summary["dynamic_foundry_execution_demonstrated"] is False
    assert summary["execution_stages"]["test_invoked"] == 0
    assert summary["execution_stages"]["tool_unavailable"] == 0
    assert summary["execution_stages"]["compiler_unavailable"] == 1
    assert "pragma" in summary["blocker"]


def test_a_policy_block_is_not_described_as_an_inconclusive_test() -> None:
    summary = _summary(
        executions={
            "vf": {
                "outcome": "inconclusive",
                "reason_code": "blocked_by_policy:cheatcode_in_source",
            }
        },
        forge_runs=0,
    )
    assert summary["foundry_invoked"] is False
    assert summary["dynamic_foundry_execution_demonstrated"] is False
    assert "policy" in summary["blocker"]


def test_invocation_count_does_not_hide_the_blocking_stage() -> None:
    summary = _summary(
        executions={"vf": {"outcome": "compile_failed"}},
        forge_runs=4,
    )
    assert summary["foundry_invoked"] is True
    assert summary["dynamic_foundry_execution_demonstrated"] is False
    assert summary["blocker"] == "compilation failed"
    assert summary["blocker"] != ""
