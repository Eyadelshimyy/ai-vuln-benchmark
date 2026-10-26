from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[3]))

import procs.harness.list as scenario_list


def test_listing_reflects_contract_status_flip(tmp_path, monkeypatch, capsys) -> None:
    scenarios = tmp_path / "scenarios" / "plans"
    scenarios.mkdir(parents=True)
    (scenarios / "PLANS-001.yaml").write_text(
        yaml.safe_dump({"id": "PLANS-001", "rules": ["PLANS-001"]})
    )
    routes = tmp_path / "routes.yaml"
    routes.write_text(yaml.safe_dump({"modules": {"plans": {"status": "pending"}}}))
    monkeypatch.setattr(scenario_list, "SCENARIOS", tmp_path / "scenarios")
    monkeypatch.setattr(scenario_list, "ROUTES", routes)
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "plans.rules.yaml").write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {
                        "id": "PLANS-001",
                        "scenarios": ["PLANS-001"],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(scenario_list, "RULES", rules)
    monkeypatch.setattr(sys, "argv", ["list.py"])

    assert scenario_list.main() == 0
    assert "plans        pending" in capsys.readouterr().out

    routes.write_text(yaml.safe_dump({"modules": {"plans": {"status": "extracted"}}}))
    assert scenario_list.main() == 0
    assert "plans        extracted" in capsys.readouterr().out
