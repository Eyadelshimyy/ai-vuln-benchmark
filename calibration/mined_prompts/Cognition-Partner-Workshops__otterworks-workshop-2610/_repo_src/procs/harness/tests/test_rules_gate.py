from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[3]))

import procs.harness.rules_gate as gate


def fixture_tree(tmp_path: Path, **rule_updates):
    root = tmp_path
    scenarios = root / "procs" / "scenarios" / "plans"
    rules = root / "procs" / "rules"
    proc_dir = root / "services" / "legacy-billing" / "db" / "procs"
    tests = root / "services" / "billing-service" / "tests"
    scenarios.mkdir(parents=True)
    rules.mkdir(parents=True)
    proc_dir.mkdir(parents=True)
    tests.mkdir(parents=True)
    (scenarios / "001.yaml").write_text("id: PLANS-001\n")
    (proc_dir / "plans.sql").write_text("line one\nline two\n")
    (tests / "test_rule.py").write_text('pytest.mark.rule("PLANS-001")\n')
    rule = {
        "id": "PLANS-001",
        "statement": "A statement",
        "source": {
            "file": "services/legacy-billing/db/procs/plans.sql",
            "lines": [1, 2],
        },
        "inputs": ["input"],
        "outputs": ["output"],
        "confidence": "high",
        "scenarios": ["PLANS-001"],
        "decision": {"status": "approved", "reviewer": "product-owner", "date": "2026-01-01"},
    }
    rule.update(rule_updates)
    (rules / "plans.rules.yaml").write_text(yaml.safe_dump({"rules": [rule]}))
    return root


def patch_tree(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(gate, "ROOT", root)
    monkeypatch.setattr(gate, "SCENARIOS", root / "procs" / "scenarios")
    monkeypatch.setattr(gate, "RULES", root / "procs" / "rules")
    monkeypatch.setattr(gate, "TESTS", root / "services" / "billing-service" / "tests")


def test_valid_rules_gate(monkeypatch, tmp_path):
    root = fixture_tree(tmp_path)
    patch_tree(monkeypatch, root)
    assert gate.validate_module("plans") == (0, [])


def test_question_requires_explicit_answer(monkeypatch, tmp_path):
    root = fixture_tree(tmp_path, question="What?", answer=None)
    patch_tree(monkeypatch, root)
    code, errors = gate.validate_module("plans")
    assert code == gate.DECISION_INVALID
    assert "explicit answer" in errors[0]


def test_source_must_be_module_proc(monkeypatch, tmp_path):
    root = fixture_tree(
        tmp_path,
        source={"file": "services/legacy-billing/db/procs/rating.sql", "lines": [1, 2]},
    )
    patch_tree(monkeypatch, root)
    code, _ = gate.validate_module("plans")
    assert code == gate.SOURCE_INVALID


def test_required_fields_are_validated(monkeypatch, tmp_path):
    root = fixture_tree(tmp_path, outputs=[])
    patch_tree(monkeypatch, root)
    code, _ = gate.validate_module("plans")
    assert code == gate.SCHEMA_INVALID


def test_scenario_claim_is_required(monkeypatch, tmp_path):
    root = fixture_tree(tmp_path, scenarios=[])
    patch_tree(monkeypatch, root)
    code, _ = gate.validate_module("plans")
    assert code == gate.COVERAGE_INVALID


def test_target_marker_is_required(monkeypatch, tmp_path):
    root = fixture_tree(tmp_path)
    (root / "services" / "billing-service" / "tests" / "test_rule.py").write_text("")
    patch_tree(monkeypatch, root)
    code, _ = gate.validate_module("plans")
    assert code == gate.MARKER_INVALID


def test_unknown_target_marker_is_rejected(monkeypatch, tmp_path):
    root = fixture_tree(tmp_path)
    (root / "services" / "billing-service" / "tests" / "test_rule.py").write_text(
        'pytest.mark.rule("PLANS-999")\n'
    )
    patch_tree(monkeypatch, root)
    code, errors = gate.validate_module("plans")
    assert code == gate.MARKER_INVALID
    assert "no ledger rule" in errors[0]


def test_markers_from_other_ledgers_do_not_fail_this_module(monkeypatch, tmp_path):
    root = fixture_tree(tmp_path)
    (root / "procs" / "scenarios" / "rating").mkdir()
    (root / "procs" / "scenarios" / "rating" / "001.yaml").write_text(
        "id: RATING-001\n"
    )
    (root / "services" / "legacy-billing" / "db" / "procs" / "rating.sql").write_text(
        "line one\nline two\n"
    )
    rating_rule = {
        "id": "RATING-001",
        "statement": "A rating statement",
        "source": {
            "file": "services/legacy-billing/db/procs/rating.sql",
            "lines": [1, 2],
        },
        "inputs": ["input"],
        "outputs": ["output"],
        "confidence": "high",
        "scenarios": ["RATING-001"],
        "decision": {
            "status": "approved",
            "reviewer": "product-owner",
            "date": "2026-01-01",
        },
    }
    (root / "procs" / "rules" / "rating.rules.yaml").write_text(
        yaml.safe_dump({"rules": [rating_rule]})
    )
    (root / "services" / "billing-service" / "tests" / "test_rating.py").write_text(
        'pytest.mark.rule("RATING-001")\n'
    )
    patch_tree(monkeypatch, root)
    assert gate.validate_module("plans") == (0, [])
    assert gate.validate_module("rating") == (0, [])
