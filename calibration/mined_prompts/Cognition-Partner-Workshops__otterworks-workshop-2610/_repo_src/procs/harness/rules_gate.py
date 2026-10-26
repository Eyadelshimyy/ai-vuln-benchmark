from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "procs" / "scenarios"
RULES = ROOT / "procs" / "rules"
TESTS = ROOT / "services" / "billing-service" / "tests"

MISSING_LEDGER = 2
DECISION_INVALID = 3
COVERAGE_INVALID = 4
SOURCE_INVALID = 5
MARKER_INVALID = 6
SCHEMA_INVALID = 7


def validate_module(module: str) -> tuple[int, list[str]]:
    ledger_path = RULES / f"{module}.rules.yaml"
    if not ledger_path.exists():
        return MISSING_LEDGER, [f"no ledger for module {module}"]
    ledger = yaml.safe_load(ledger_path.read_text()) or {}
    errors: dict[int, list[str]] = {
        DECISION_INVALID: [],
        COVERAGE_INVALID: [],
        SOURCE_INVALID: [],
        MARKER_INVALID: [],
        SCHEMA_INVALID: [],
    }
    rules = ledger.get("rules")
    if not isinstance(rules, list):
        errors[SCHEMA_INVALID].append("rules must be a list")
        return SCHEMA_INVALID, errors[SCHEMA_INVALID]
    scenario_paths = sorted((SCENARIOS / module).glob("*.yaml"))
    scenarios = [yaml.safe_load(path.read_text()) for path in scenario_paths]
    scenario_ids = {scenario.get("id") for scenario in scenarios}
    claimed: set[str] = set()
    markers = {
        rule_id
        for path in TESTS.rglob("*.py")
        for rule_id in re.findall(r'pytest\.mark\.rule\(["\']([^"\']+)["\']\)', path.read_text())
    }
    all_rule_ids: set[str] = set()
    for other_ledger_path in RULES.glob("*.rules.yaml"):
        other_ledger = yaml.safe_load(other_ledger_path.read_text()) or {}
        other_rules = other_ledger.get("rules", [])
        if isinstance(other_rules, list):
            all_rule_ids.update(
                rule.get("id")
                for rule in other_rules
                if isinstance(rule, dict) and isinstance(rule.get("id"), str)
            )
    for marker in sorted(markers - all_rule_ids):
        errors[MARKER_INVALID].append(f"{marker}: target test marker has no ledger rule")
    proc_path = ROOT / "services" / "legacy-billing" / "db" / "procs" / f"{module}.sql"
    for index, rule in enumerate(rules):
        prefix = f"rule[{index}]"
        if not isinstance(rule, dict):
            errors[SCHEMA_INVALID].append(f"{prefix}: must be a mapping")
            continue
        required = {
            "id": isinstance(rule.get("id"), str) and bool(rule["id"].strip()),
            "statement": isinstance(rule.get("statement"), str) and bool(rule["statement"].strip()),
            "source": isinstance(rule.get("source"), dict),
            "outputs": isinstance(rule.get("outputs"), list) and bool(rule["outputs"]),
            "confidence": isinstance(rule.get("confidence"), str) and bool(rule["confidence"].strip()),
        }
        missing = [field for field, valid in required.items() if not valid]
        if missing:
            errors[SCHEMA_INVALID].append(f"{prefix}: invalid required fields {', '.join(missing)}")
            continue
        rule_id = rule["id"]
        decision = rule.get("decision")
        if not isinstance(decision, dict):
            errors[DECISION_INVALID].append(f"{rule_id}: decision is required")
        else:
            if decision.get("status") in {None, "pending"}:
                errors[DECISION_INVALID].append(f"{rule_id}: decision is pending")
            if not decision.get("reviewer"):
                errors[DECISION_INVALID].append(f"{rule_id}: reviewer is required")
            if not decision.get("date"):
                errors[DECISION_INVALID].append(f"{rule_id}: date is required")
        if "question" in rule and not rule.get("answer"):
            errors[DECISION_INVALID].append(f"{rule_id}: question has no explicit answer")
        scenarios_for_rule = rule.get("scenarios")
        if not isinstance(scenarios_for_rule, list) or not scenarios_for_rule:
            errors[COVERAGE_INVALID].append(f"{rule_id}: must cite a scenario")
        else:
            claimed.update(scenarios_for_rule)
            for scenario_id in scenarios_for_rule:
                if scenario_id not in scenario_ids:
                    errors[COVERAGE_INVALID].append(f"{rule_id}: unknown scenario {scenario_id}")
        source = rule["source"]
        lines = source.get("lines")
        source_file = source.get("file")
        if (
            source_file != f"services/legacy-billing/db/procs/{module}.sql"
            or not isinstance(lines, list)
            or len(lines) != 2
            or not all(isinstance(line, int) for line in lines)
            or not proc_path.exists()
        ):
            errors[SOURCE_INVALID].append(f"{rule_id}: source must resolve inside {module}.sql")
        elif lines[0] < 1 or lines[0] > lines[1] or lines[1] > len(proc_path.read_text().splitlines()):
            errors[SOURCE_INVALID].append(f"{rule_id}: source line range is invalid")
        if rule_id not in markers:
            errors[MARKER_INVALID].append(f"{rule_id}: no target test marker")
    for scenario_id in sorted(scenario_ids - claimed):
        errors[COVERAGE_INVALID].append(f"{scenario_id}: not claimed by a rule")
    for code in (SCHEMA_INVALID, DECISION_INVALID, COVERAGE_INVALID, SOURCE_INVALID, MARKER_INVALID):
        if errors[code]:
            return code, errors[code]
    return 0, []


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--module")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    modules = (
        sorted(path.name.removesuffix(".rules.yaml") for path in RULES.glob("*.rules.yaml"))
        if args.all
        else [args.module]
    )
    failures: list[tuple[str, int, list[str]]] = []
    for module in modules:
        code, errors = validate_module(module)
        if code:
            failures.append((module, code, errors))
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
        else:
            print(f"Rules gate PASS: {module}")
    if failures:
        return failures[0][1]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
