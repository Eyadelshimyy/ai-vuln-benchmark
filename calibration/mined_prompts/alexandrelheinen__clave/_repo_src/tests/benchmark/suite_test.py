"""Tests for turning runs into scored records, and for the evidence pack.

These use recorded reports rather than a live loop, which needs MuJoCo and a GL
backend and is already covered at v0.9.0.
"""

import json
from pathlib import Path

import pytest

from clave.benchmark.config import BenchmarkConfig, Configuration
from clave.benchmark.pack import (
    UNMEASURABLE_METRICS,
    UNMEASURED,
    EvidencePack,
    recommend,
    score,
    table,
)
from clave.benchmark.suite import ConfigurationResult, outcomes_for
from clave.runtime.loop import DecisionRecord, RunReport
from clave.validation.gates import GateConfig
from clave.validation.outcomes import OutcomeSet

ROOT = Path(__file__).resolve().parents[2]
GATES = GateConfig.load(ROOT / "configs" / "validation" / "gates.yml")


def decision(object_id: int, true_class: str, predicted: str) -> DecisionRecord:
    """One decision the loop published."""
    return DecisionRecord(
        object_id=object_id,
        true_class=true_class,
        predicted_class=predicted,
        latency_seconds=0.05,
        verdict="accepted",
        channel=1,
    )


def report(**overrides: object) -> RunReport:
    """A run that presented three objects and decided about two of them."""
    built = RunReport(predictor="test", belt_speed=0.2, window_length=0.34)
    built.presented = [(0, "M-01"), (1, "M-05"), (2, "M-07")]
    built.decisions = [
        decision(0, "M-01", "M-01"),
        decision(1, "M-05", "M-01"),
    ]
    built.proposals = 2
    built.latencies_seconds = [0.04, 0.05]
    for key, value in overrides.items():
        setattr(built, key, value)
    return built


def test_one_record_per_object_that_entered_the_window() -> None:
    """The population is what the validation harness already defines."""
    outcomes, outside = outcomes_for("row", [(0, report())])
    assert len(outcomes) == 3
    assert outside == 0
    assert {record.true_class for record in outcomes.outcomes} == {
        "M-01",
        "M-05",
        "M-07",
    }


def test_an_object_nobody_decided_about_carries_no_prediction() -> None:
    """A miss is a miss, not an instant decision about nothing."""
    outcomes, _ = outcomes_for("row", [(0, report())])
    undecided = [r for r in outcomes.outcomes if r.predicted_class is None]
    assert len(undecided) == 1
    assert undecided[0].true_class == "M-07"
    assert undecided[0].decision_latency_seconds is None


def test_every_record_says_nothing_was_picked() -> None:
    """This is what makes pick success unmeasurable, not zero."""
    outcomes, _ = outcomes_for("row", [(0, report())])
    assert all(not record.picked for record in outcomes.outcomes)
    assert all(record.routed_channel is None for record in outcomes.outcomes)
    assert all(record.cycle_time_seconds is None for record in outcomes.outcomes)


def test_the_last_decision_about_an_object_is_the_one_scored() -> None:
    """A line acts on the latest decision, so that is the one to score."""
    run = report()
    run.decisions = [decision(0, "M-01", "M-05"), decision(0, "M-01", "M-01")]
    outcomes, _ = outcomes_for("row", [(0, run)])
    scored = {r.object_id: r.predicted_class for r in outcomes.outcomes}
    assert scored["seed0-object0"] == "M-01"


def test_a_decision_about_an_object_outside_the_window_is_counted() -> None:
    """Proposing an unreachable object is a fault the safety layer catches."""
    run = report()
    run.decisions = [*run.decisions, decision(9, "M-02", "M-02")]
    _, outside = outcomes_for("row", [(0, run)])
    assert outside == 1


def test_records_from_two_seeds_do_not_collide() -> None:
    """The object pool numbers slots, and two seeds reuse the same numbers."""
    outcomes, _ = outcomes_for("row", [(0, report()), (1, report())])
    assert len(outcomes) == 6
    assert len({record.object_id for record in outcomes.outcomes}) == 6


def result(name: str, **overrides: object) -> ConfigurationResult:
    """A scored row, with fields replaced for the case under test."""
    outcomes, _ = outcomes_for(name, [(0, report())])
    fields: dict[str, object] = {
        "configuration": Configuration(name, "checkpoints", "p", "q", ""),
        "available": True,
        "unavailable_reason": None,
        "presented": 3,
        "decided": 2,
        "proposals": 2,
        "overridden": 1,
        "outside_window": 0,
        "simulated_seconds": 20.0,
        "outcomes": outcomes,
        "latencies_seconds": (0.04, 0.05),
        "belt_speeds": (0.2,),
    }
    fields.update(overrides)
    return ConfigurationResult(**fields)  # type: ignore[arg-type]


def test_an_unavailable_row_is_recorded_with_its_reason() -> None:
    """One missing checkpoint does not cost the comparison."""
    scored = score(
        result(
            "missing",
            available=False,
            unavailable_reason="no checkpoint at runs/ghost.pt",
            outcomes=OutcomeSet("missing", []),
        ),
        GATES,
    )
    assert scored.summary is None
    assert scored.gates == ()
    assert "ghost" in (scored.result.unavailable_reason or "")
    assert "Unavailable" in table([scored])


def test_the_table_reports_what_cannot_be_measured_rather_than_a_zero() -> None:
    """A rate of zero would read as a result."""
    rendered = table([score(result("row"), GATES)])
    assert rendered.count(UNMEASURED) >= 2
    assert "pick success rate" in UNMEASURABLE_METRICS


def test_the_recommendation_names_a_row_that_actually_ran() -> None:
    """A recommendation is a choice between measured rows."""
    rows = [score(result("first"), GATES), score(result("second"), GATES)]
    name, reason = recommend(rows)
    assert name in {"first", "second"}
    assert "accuracy" in reason


def test_nothing_is_recommended_when_nothing_trained_ran() -> None:
    """Silence beats naming a row that never produced a decision."""
    control = Configuration("scripted-expert", "scripted", None, None, "")
    name, reason = recommend([score(result("control", configuration=control), GATES)])
    assert name == "none"
    assert "nothing to choose" in reason


def test_the_evidence_pack_round_trips_through_json(tmp_path: Path) -> None:
    """A later reader parses it without this package."""
    config = BenchmarkConfig.load(ROOT / "configs" / "benchmark" / "default.yml")
    pack = EvidencePack(
        config=config,
        scored=[score(result("row"), GATES)],
        machine="x86_64 on Linux",
        threads=8,
        environment={"python": "3.12"},
        world_digest="abc",
    )
    path = tmp_path / "benchmark.json"
    pack.write(path)
    read = json.loads(path.read_text())
    assert read["seeds"] == list(config.seeds)
    assert read["benchmark_config_digest"] == config.digest
    assert read["configurations"][0]["pick_success_rate"] == UNMEASURED
    assert read["unmeasurable"]["cycle time"]
    assert read["recommendation"]["configuration"] == "row"
    assert "What could not be measured" in pack.render()


def test_the_pack_states_the_bar_a_constant_predictor_would_clear() -> None:
    """A model below the majority class share learned nothing.

    The fixture presents one M-01, one M-05 and one M-07, so a constant
    predictor scores one in three.
    """
    scored = score(result("row"), GATES)
    assert scored.majority_class_share == pytest.approx(1 / 3)
    assert scored.class_support == {"M-01": 1, "M-05": 1, "M-07": 1}

    config = BenchmarkConfig.load(ROOT / "configs" / "benchmark" / "default.yml")
    pack = EvidencePack(
        config=config,
        scored=[scored],
        machine="x86_64 on Linux",
        threads=8,
        environment={},
        world_digest="abc",
    )
    assert "always named the most common class" in pack.render()


def test_a_scripted_row_builds_without_a_checkpoint() -> None:
    """The control needs no trained model, which is what makes it a control."""
    from clave.benchmark.suite import build_predictor
    from clave.runtime.inference import ScriptedPredictor
    from clave.runtime.loop import RuntimeSettings

    settings = RuntimeSettings.load(ROOT / "configs" / "runtime" / "sitl.yml")
    control = Configuration("scripted-expert", "scripted", None, None, "")
    assert isinstance(build_predictor(control, settings, ROOT), ScriptedPredictor)


def test_an_empty_run_reports_no_rate_rather_than_dividing_by_zero() -> None:
    """A configuration that presented nothing is a zero, not a crash."""
    empty = result("empty", presented=0, simulated_seconds=0.0)
    assert empty.decision_rate == 0.0
    assert empty.coverage == 0.0


def test_a_row_whose_checkpoint_is_missing_is_recorded_and_skipped(
    tmp_path: Path,
) -> None:
    """One absent dependency does not cost the comparison.

    The row below names a checkpoint that does not exist. On a machine with no
    deep learning framework it fails on the import instead, which is the same
    kind of absence and is reported the same way.
    """
    from clave.benchmark.suite import run_configuration

    config = BenchmarkConfig.load(ROOT / "configs" / "benchmark" / "default.yml")
    ghost = Configuration("ghost", "checkpoints", "resnet50-baseline", "nothing", "")
    outcome = run_configuration(ROOT, config, ghost, tmp_path)
    assert not outcome.available
    assert outcome.unavailable_reason
    assert outcome.presented == 0
    assert len(outcome.outcomes) == 0
