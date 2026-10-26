from __future__ import annotations

from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[3]


def test_probe_shapes_are_stable_per_entrypoint() -> None:
    for module_path in sorted((ROOT / "procs" / "scenarios").iterdir()):
        if not module_path.is_dir():
            continue
        shapes: dict[str, set[tuple[tuple[str, bool], ...]]] = {}
        for scenario_path in sorted(module_path.glob("*.yaml")):
            scenario = yaml.safe_load(scenario_path.read_text())
            assert "setup_sql" not in scenario
            assert "before_sql" not in scenario
            probes = scenario.get("probes", [])
            for probe in probes:
                query = probe["query"]
                assert "string_agg" not in query.lower()
                assert not re.search(r"::text\s*\|\||\|\|\s*':'", query)
            shape = tuple(
                sorted(
                    (
                        probe["id"],
                        bool(probe.get("collect_rows")),
                    )
                    for probe in probes
                )
            )
            shapes.setdefault(scenario["entrypoint"], set()).add(shape)
            assert "rules" not in scenario
        assert all(len(entrypoint_shapes) == 1 for entrypoint_shapes in shapes.values())
