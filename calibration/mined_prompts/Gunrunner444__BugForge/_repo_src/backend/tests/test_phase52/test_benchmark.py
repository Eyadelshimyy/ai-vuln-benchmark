"""Phase 52 Slice D: the local regression benchmark over compact incident fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.discovery.bounty.benchmark import BENCHMARK_SCHEMA, DISCLAIMER, run_benchmark
from app.discovery.bounty.stateful import StatefulExecutor
from tests.test_phase52.phase52_support import requires_forge

HISTORY = Path(__file__).resolve().parents[1] / "fixtures" / "phase52_history"


def test_static_benchmark_detects_every_vulnerable_case_and_no_safe_twin() -> None:
    report = run_benchmark(HISTORY)
    assert report["schema"] == BENCHMARK_SCHEMA
    cases = json.loads((HISTORY / "benchmark.json").read_text())["cases"]
    assert report["totals"] == {"tp": len(cases), "fn": 0, "fp": 0, "tn": len(cases)}
    assert all(case["static"] == "pass" for case in report["cases"])
    # honesty: a fixture regression signal, not a real-world rate; never verified
    assert report["real_world_rate"] is None and report["verified"] is False
    assert report["disclaimer"] == DISCLAIMER and "not a real-world" in DISCLAIMER
    assert report["stateful"]["not_run"] == sum(1 for c in cases if c.get("stateful"))
    assert report["emission_is_not_vulnerability_precision"] is True
    assert run_benchmark(HISTORY)["digest"] == report["digest"]


def test_benchmark_refuses_foreign_schema_and_escaping_paths(tmp_path: Path) -> None:
    (tmp_path / "benchmark.json").write_text(json.dumps({"schema": "other", "cases": []}))
    with pytest.raises(ValueError):
        run_benchmark(tmp_path)
    (tmp_path / "benchmark.json").write_text(
        json.dumps(
            {
                "schema": BENCHMARK_SCHEMA,
                "cases": [
                    {
                        "id": "x",
                        "vulnerable": "../outside.sol",
                        "safe": "s.sol",
                        "detector": "d",
                    }
                ],
            }
        )
    )
    (tmp_path.parent / "outside.sol").write_text("contract A {}")
    (tmp_path / "s.sol").write_text("contract A {}")
    with pytest.raises(ValueError):
        run_benchmark(tmp_path)


def test_a_detector_regression_shows_up_as_a_failed_case(tmp_path: Path) -> None:
    for item in HISTORY.iterdir():
        (tmp_path / item.name).write_bytes(item.read_bytes())
    # make the "safe" twin identical to the vulnerable one: the case must fail (fp)
    (tmp_path / "bridge_receiver_safe.sol").write_bytes(
        (HISTORY / "bridge_receiver_vulnerable.sol").read_bytes()
    )
    report = run_benchmark(tmp_path)
    assert report["totals"]["fp"] == 1
    failed = [c["id"] for c in report["cases"] if c["static"] == "fail"]
    assert failed == ["bridge_receiver"]
    assert report["digest"] != run_benchmark(HISTORY)["digest"]


@requires_forge
def test_stateful_twins_agree_with_their_expected_outcomes() -> None:
    report = run_benchmark(HISTORY, executor=StatefulExecutor())
    cases = json.loads((HISTORY / "benchmark.json").read_text())["cases"]
    assert report["stateful"]["disagree"] == 0
    assert report["stateful"]["agree"] == sum(1 for case in cases if case.get("stateful"))
    assert "identity_rejection" not in report["stateful"]
    assert report["identity_validation"]["outcome"] == "identity_mismatch"
    assert report["identity_validation"]["executed"] is False
    assert report["identity_validation"]["forge_delta"] == 0
    assert report["fixture_precision"] != report.get("real_world_rate")
    lines = ["benchmark (emission is not vulnerability precision)"]
    for case in report["cases"]:
        if "stateful" in case:
            assert case["stateful"] == case["stateful_expected"]
            assert case["checks"]["overstated"] is False
            assert case["vulnerability_label"] != "verified"
        else:
            assert case["execution"] == "unsupported_or_not_requested"
            assert case["vulnerability_label"] == "unknown"
            assert case["ground_truth"] == "unknown"
        lines.append(
            f"{case['id']}: static={case['static']} label={case['vulnerability_label']} "
            f"execution={case['execution']} truth={case['ground_truth']}"
        )
    print("\n".join(lines))
