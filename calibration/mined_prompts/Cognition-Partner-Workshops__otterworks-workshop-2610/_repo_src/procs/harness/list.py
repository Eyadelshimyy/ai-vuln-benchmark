from __future__ import annotations

import argparse
from pathlib import Path

import yaml
try:
    from procs.harness.ledger import scenario_rule_map
    from procs.harness.status import VALID_STATUSES, status_for
except ModuleNotFoundError:
    from ledger import scenario_rule_map
    from status import VALID_STATUSES, status_for

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "procs" / "scenarios"
ROUTES = ROOT / "procs" / "routes.yaml"
RULES = ROOT / "procs" / "rules"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module")
    args = parser.parse_args()
    routes = yaml.safe_load(ROUTES.read_text()).get("modules", {})
    rules_by_scenario = scenario_rule_map(RULES)
    scenario_modules = {
        path.name for path in SCENARIOS.iterdir() if path.is_dir()
    }
    modules = [args.module] if args.module else sorted(set(routes) | scenario_modules)
    for module in modules:
        module_config = routes.get(module)
        module_status = status_for(module_config)
        if module_config is not None and module_status not in VALID_STATUSES:
            parser.error(f"invalid status for module {module}: {module_status!r}")
        paths = sorted((SCENARIOS / module).glob("*.yaml"))
        if not paths:
            parser.error(f"unknown module: {module}")
        scenarios = [yaml.safe_load(path.read_text()) for path in paths]
        rules = sorted(
            {rule for scenario in scenarios for rule in rules_by_scenario.get(scenario["id"], [])}
        )
        print(
            f"{module:12} {module_status:9} "
            f"scenarios={len(scenarios):2} rule_claims={len(rules):2}"
        )
        for scenario in scenarios:
            print(f"  {scenario['id']}: {', '.join(rules_by_scenario.get(scenario['id'], []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
