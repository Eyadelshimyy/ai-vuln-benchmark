"""Local regression benchmark over compact incident-class fixtures (Phase 52, Slice D).

Each case is a vulnerable fixture and its safe twin, the detector that should fire
on the first and not on the second, and optionally the stateful outcome its property
should reach on each (the safe twin runs the *same* sequence, rebuilt from the
vulnerable candidate, against the fixed code). Everything is local: fixtures are read
from a directory on disk; nothing is downloaded.

The report is a regression signal for these fixtures only. It is **not** a real-world
detection rate and must never be presented as one.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.discovery.bounty.properties import build_property
from app.discovery.bounty.vfcs import generate
from app.discovery.orchestration.codec import digest
from app.parsing.solidity_research import build_research_model
from app.parsing.solidity_research_suite import run_suite

BENCHMARK_SCHEMA = "bugforge.local_benchmark/1"
DISCLAIMER = (
    "local fixture regression only: these compact reconstructions measure BugForge "
    "against itself; this is not a real-world detection rate"
)
MAX_CASES = 64


def _fires(text: str, file: str, detector: str, contract: str) -> tuple[bool, Any, Any]:
    model = build_research_model({file: text})
    hits = [
        c
        for c in run_suite(model).candidates
        if c.detector == detector and (not contract or c.contract == contract)
    ]
    return bool(hits), (hits[0] if hits else None), model


def run_benchmark(directory: Path, *, executor: Any = None) -> dict[str, Any]:
    """Run every case in ``directory/benchmark.json``. ``executor`` enables stateful checks."""
    spec = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    if spec.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError("not a BugForge local benchmark")
    cases: list[dict[str, Any]] = []
    totals = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    stateful = {"agree": 0, "disagree": 0, "not_run": 0}
    identity_validation = {"outcome": "not_run", "executed": None, "forge_delta": None}
    for case in list(spec.get("cases", []))[:MAX_CASES]:
        detector = str(case["detector"])
        contract = str(case.get("contract", ""))
        files = {role: (directory / str(case[role])).resolve() for role in ("vulnerable", "safe")}
        if any(directory.resolve() not in path.parents for path in files.values()):
            raise ValueError("benchmark fixtures must live inside the benchmark directory")
        texts = {role: path.read_text(encoding="utf-8") for role, path in files.items()}
        vuln_hit, candidate, vuln_model = _fires(
            texts["vulnerable"], files["vulnerable"].name, detector, contract
        )
        safe_hit, _, safe_model = _fires(texts["safe"], files["safe"].name, detector, contract)
        totals["tp" if vuln_hit else "fn"] += 1
        totals["fp" if safe_hit else "tn"] += 1
        result: dict[str, Any] = {
            "id": case["id"],
            "detector": detector,
            "vulnerable_fires": vuln_hit,
            "safe_fires": safe_hit,
            "static": "pass" if vuln_hit and not safe_hit else "fail",
            "rationale": str(case.get("rationale", "")),
            "ground_truth": str(case.get("ground_truth", "unknown")),
        }
        result["vulnerability_label"] = "not_a_confirmed_vulnerability"
        result["execution"] = "not_requested"
        if not case.get("stateful"):
            result["execution"] = "unsupported_or_not_requested"
            result["vulnerability_label"] = "unknown"
        expected = dict(case.get("stateful") or {})
        if expected and executor is not None and candidate is not None:
            outcomes: dict[str, str] = {}
            observed: dict[str, Any] = {}
            vulnerable_sequence = None
            for role, model, text, path in (
                ("vulnerable", vuln_model, texts["vulnerable"], files["vulnerable"]),
                ("safe", safe_model, texts["safe"], files["safe"]),
            ):
                twin = replace(candidate, file=path.name)
                role_sources = {path.name: text}
                built = generate(model, [twin], sources=role_sources)
                if not built.sequences:
                    outcomes[role] = "no_sequence"
                    continue
                sequence = built.sequences[0]
                spec_ = build_property(sequence, model, twin)
                obs = executor.execute(sequence, model, {path.name: text}, spec=spec_)
                outcomes[role] = obs.outcome.value
                observed[role] = obs
                if role == "vulnerable":
                    vulnerable_sequence = sequence
            result["stateful"] = outcomes
            result["stateful_expected"] = expected
            refused = bool(outcomes) and all(value == "no_sequence" for value in outcomes.values())
            result["execution"] = (
                "sequence_refused"
                if refused
                else "foundry"
                if executor.available()
                else "unavailable"
            )
            labels = {
                "property_violated": "local_property_violation_unverified",
                "property_held": "property_held",
                "mechanism_observed": "mechanism_only",
                "failure_propagated": "mechanism_not_reproduced",
                "no_sequence": "not_a_confirmed_vulnerability",
            }
            result["vulnerability_label"] = labels.get(
                outcomes.get("vulnerable", ""), "unclassified"
            )
            vulnerable_obs = observed.get("vulnerable")
            if vulnerable_sequence is not None and identity_validation["outcome"] == "not_run":
                altered = {files["vulnerable"].name: texts["vulnerable"] + "\n"}
                before = getattr(executor, "runs", None)
                rejected = executor.execute(
                    vulnerable_sequence,
                    vuln_model,
                    altered,
                    spec=build_property(vulnerable_sequence, vuln_model),
                )
                identity_validation = {
                    "outcome": rejected.outcome.value,
                    "executed": rejected.executed,
                    "forge_delta": (None if before is None else executor.runs - before),
                }
                result["identity_rejection"] = rejected.outcome.value
            agreed = all(outcomes.get(role) == want for role, want in expected.items())
            executed = any(getattr(item, "executed", False) for item in observed.values())
            verified = any(getattr(item, "verified", False) for item in observed.values())
            result["checks"] = {
                "static_candidate": "emitted" if vuln_hit else "absent",
                "sequence": "refused" if refused else "generated",
                "oracle": (
                    "none"
                    if refused or vulnerable_obs is None
                    else str(getattr(vulnerable_obs, "oracle_kind", ""))
                ),
                "harness": (
                    "not_built"
                    if refused or vulnerable_obs is None or vulnerable_obs.harness is None
                    else "built"
                    if vulnerable_obs.harness.buildable
                    else "failed"
                ),
                "foundry": "executed" if executed else "not_executed",
                "outcome": "matched" if agreed else "differed",
                "classification": result["vulnerability_label"],
                "persisted": "not_applicable",
                "overstated": verified,
            }
            unavailable = any(v in {"unavailable"} for v in outcomes.values())
            if unavailable:
                stateful["not_run"] += 1
            elif agreed:
                stateful["agree"] += 1
            else:
                stateful["disagree"] += 1
        elif expected:
            stateful["not_run"] += 1
        cases.append(result)
    judged = totals["tp"] + totals["fp"]
    report = {
        "schema": BENCHMARK_SCHEMA,
        "cases": cases,
        "totals": totals,
        "fixture_precision": (totals["tp"] / judged) if judged else None,
        "fixture_recall": (
            totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else None
        ),
        "stateful": stateful,
        "identity_validation": identity_validation,
        "disclaimer": DISCLAIMER,
        "emission_is_not_vulnerability_precision": True,
        "real_world_rate": None,
        "verified": False,
    }
    report["digest"] = digest(
        {
            "cases": [
                {
                    "id": case["id"],
                    "static": case["static"],
                    "vulnerable_fires": case["vulnerable_fires"],
                    "safe_fires": case["safe_fires"],
                    "execution": case["execution"],
                    "vulnerability_label": case["vulnerability_label"],
                    "ground_truth": case["ground_truth"],
                    "stateful": case.get("stateful"),
                    "identity_rejection": case.get("identity_rejection"),
                }
                for case in cases
            ],
            "totals": totals,
            "stateful": stateful,
            "identity_validation": identity_validation,
            "fixture_precision": report["fixture_precision"],
            "fixture_recall": report["fixture_recall"],
            "real_world_rate": None,
            "verified": False,
        }
    )
    return report


def pipeline_coverage_matrix(directory: Path | None = None) -> dict[str, Any]:
    """Registry-derived view of which detectors have fixtures and executable oracles.

    Untested cells stay ``untested``. This does not run Foundry and does not turn
    a fixture into a passing execution result.
    """
    from app.discovery.bounty.properties import _ORACLE_BUILDERS
    from app.discovery.bounty.vfcs import _BUILDERS

    cases: list[dict[str, Any]] = []
    if directory is not None and (directory / "benchmark.json").is_file():
        spec = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
        cases = list(spec.get("cases", []))
    by_detector = {str(case.get("detector")): case for case in cases}
    rows = []
    for detector in sorted(_BUILDERS):
        case = by_detector.get(detector)
        stateful = bool(case and case.get("stateful"))
        rows.append(
            {
                "detector": detector,
                "registered": True,
                "sequence_builder": True,
                "positive_fixture": "present" if case else "untested",
                "safe_control": "present" if case else "untested",
                "oracle_expected": "specified" if stateful else "untested",
                "foundry_execution": "untested",
                "persistence": "untested",
                "qualification": "untested",
            }
        )
    return {
        "rows": rows,
        "oracle_templates": sorted(_ORACLE_BUILDERS),
        "builders_without_fixture": [
            row["detector"] for row in rows if row["positive_fixture"] == "untested"
        ],
        "note": (
            "foundry_execution, persistence, and qualification stay untested in this "
            "view; a present fixture is not an execution result and not a vulnerability"
        ),
        "verified": False,
    }
