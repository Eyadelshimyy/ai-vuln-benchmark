from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "procs" / "rules"


def scenario_rule_map(rules_root: Path = RULES) -> dict[str, list[str]]:
    claims: dict[str, set[str]] = {}
    for ledger_path in sorted(rules_root.glob("*.rules.yaml")):
        ledger = yaml.safe_load(ledger_path.read_text()) or {}
        for rule in ledger.get("rules", []):
            if not isinstance(rule, dict) or not isinstance(rule.get("id"), str):
                continue
            for scenario in rule.get("scenarios", []):
                claims.setdefault(scenario, set()).add(rule["id"])
    return {scenario: sorted(rule_ids) for scenario, rule_ids in claims.items()}
