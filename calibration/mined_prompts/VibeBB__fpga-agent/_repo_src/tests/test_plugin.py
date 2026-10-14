from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from fpga import cli, mcp_server, service

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "fpga"
HOOKS = PLUGIN / "hooks" / "scripts"
LAUNCHER = PLUGIN / "scripts" / "fpga_launcher.py"


def _hook(script: str, payload: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOKS / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )


def _terminal(command: str) -> dict[str, object]:
    return {"tool_name": "terminal", "tool_input": {"command": command}}


def test_mcp_tools_registered_without_program() -> None:
    names = {tool.name for tool in mcp_server.tool_specs()}
    assert names == {
        "fpga_doctor",
        "fpga_validate",
        "fpga_check",
        "fpga_gates",
        "fpga_constraints",
        "fpga_pinmap_export",
        "fpga_regmap_export",
        "fpga_lint",
        "fpga_sim",
        "fpga_formal",
        "fpga_build",
        "fpga_production_export",
        "fpga_request",
        "fpga_sim_thermal_request",
        "fpga_sim_thermal_check",
        "fpga_profile",
        "fpga_render",
        "fpga_record_decision",
        "fpga_record_impression",
        "fpga_record_vision_review",
        "fpga_records_status",
        "fpga_record_reconcile",
        "fpga_record_insight",
        "fpga_record_song_receipt",
        "fpga_record_read",
        "fpga_records_digest",
        "fpga_records_search",
        "fpga_ux_inbox",
        "fpga_ux_respond",
    }
    assert mcp_server.dispatch("fpga_program", {})["verdict"] == "fail"


def test_mcp_dispatch_validate(ulx3s: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(ulx3s.parents[2]))
    payload = mcp_server.dispatch("fpga_validate", {"contract_path": str(ulx3s)})
    assert payload["verdict"] == "pass"
    assert payload["profile"] == "ecp5-lfe5u-85f-cabga381"


def test_cli_exit_codes(ulx3s: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["validate", str(ulx3s)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "pass"
    assert cli.main(["profile", "ice40-up5k-sg48"]) == 0
    assert json.loads(capsys.readouterr().out)["family"] == "ice40"
    assert cli.main(["profile", "nope"]) != 0
    assert cli.main(["sim", str(ulx3s), "--id", "nope"]) != 0
    capsys.readouterr()


def test_pinmap_export(ulx3s: Path) -> None:
    payload = service.pinmap_payload(ulx3s, None)
    assert payload["verdict"] == "pass"
    pinmap = json.loads((ulx3s.parent / "fpga-reports" / "blinky.fpga-pinmap.json").read_text())
    assert pinmap["artifact_kind"] == "fpga_pinmap"
    assert {"port": "clk_25mhz", "package_pin": "G2"}.items() <= pinmap["pins"][0].items()


def _decision_event(root: Path) -> str:
    """Append a minimal decision record so requests have a ref to cite."""
    from typing import Any

    from fpga import records

    root.joinpath("out").mkdir(exist_ok=True)
    root.joinpath("out/design.v").write_text("module m; endmodule\n", encoding="utf-8")
    result = records.record_decision(
        {
            "id": "fpga-request-test",
            "stage": "design",
            "question": "Which pin should carry the LED?",
            "principles": ["an I/O pin must meet the required voltage standard"],
            "options": [
                {"name": "G2", "pros": ["free"], "cons": ["shared"]},
                {"name": "T1", "pros": ["dedicated"], "cons": ["far"]},
            ],
            "chosen": "G2",
            "rationale": (
                "G2 already carries the LED net on the reference board, meets the"
                " 3.3 V I/O standard, and requires no rework of the carrier board,"
                " which keeps the change minimal and the risk of a new constraint"
                " mistake close to zero for this particular board revision."
            ),
            "risks": ["pin is shared"],
            "revisit_when": "the carrier board changes",
            "evidence": [{"path": "out/design.v"}],
        },
        root=root,
    )
    return cast(dict[str, Any], result["record"])["event_id"]


def test_request_artifact(ulx3s: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = ulx3s.parent
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(root))
    decision = _decision_event(root)
    missing_ref = service.request_payload(
        ulx3s,
        None,
        target="circuit",
        risk="low",
        change="x".ljust(10),
        rationale="y".ljust(10),
        nets=[],
        failing_checks=[],
    )
    assert missing_ref["verdict"] == "fail"
    payload = service.request_payload(
        ulx3s,
        None,
        target="circuit",
        risk="high",
        change="route LED0 to a clock-capable pin",
        rationale="the LED0 path misses timing",
        nets=["LED0"],
        failing_checks=["fpga.timing"],
        decision_refs=[decision],
    )
    assert payload["verdict"] == "pass"
    written = Path(cast(list[str], payload["written"])[0])
    assert written.name.endswith(".fpga-request.json")
    request = json.loads(written.read_text())
    assert request["target"] == "circuit" and request["failing_checks"] == ["fpga.timing"]
    assert request["schema_version"] == 2
    assert request["decision_refs"] == [decision]
    assert request["inputs"][0]["path"] == ulx3s.name
    bad = service.request_payload(
        ulx3s,
        None,
        target="nobody",
        risk="low",
        change="x",
        rationale="y",
        nets=[],
        failing_checks=[],
    )
    assert bad["verdict"] == "fail"


def test_program_requires_passing_report(ulx3s: Path) -> None:
    payload = service.program_payload(ulx3s, None, "0" * 64, dry_run=True)
    assert payload["verdict"] == "fail" and "gate report" in str(payload["detail"])


@pytest.mark.parametrize(
    "path",
    [
        "examples/a/a.fpga.pcf",
        "b.fpga.lpf",
        "c.fpga.cst",
        "fpga-reports/a.fpga-report.md",
        "fpga-reports/a.fpga-pinmap.json",
        "fpga-reports/a.fpga-regmap.json",
        "fpga-reports/a.fpga-production.json",
        "fpga-reports/sim-count.log",
        "observations/fpga/image-observations.jsonl",
        "intake/attachments/manifest.jsonl",
    ],
)
def test_protect_generated_blocks_edits(path: str) -> None:
    result = _hook(
        "protect_generated.py",
        {"tool_name": "file_editor", "tool_input": {"command": "create", "path": path}},
    )
    assert result.returncode == 2


def test_protect_generated_blocks_terminal_redirect() -> None:
    assert _hook("protect_generated.py", _terminal("echo x > a.fpga.pcf")).returncode == 2


@pytest.mark.parametrize(
    "payload",
    [
        _terminal("cat a.fpga.pcf"),
        {"tool_name": "file_editor", "tool_input": {"command": "create", "path": "rtl/a.vhdl"}},
        {"tool_name": "apply_patch", "tool_input": {"patch": "*** Update File: a.fpga.json\n"}},
    ],
)
def test_protect_generated_allows(payload: object) -> None:
    assert _hook("protect_generated.py", payload).returncode == 0


@pytest.mark.parametrize(
    ("script", "command"),
    [
        ("deny_programming.py", "openFPGALoader -b ulx3s build/blinky.bit"),
        ("deny_programming.py", "sudo iceprog build/a.bin"),
        ("deny_programming.py", "cd x && ecpprog a.bit"),
        ("deny_programming.py", "fpga program a.fpga.json --confirm-sha256 abc"),
        ("deny_programming.py", "python -m fpga program a.fpga.json --confirm-sha256 abc"),
        (
            "deny_programming.py",
            "python3 plugins/fpga/scripts/fpga_launcher.py program a.fpga.json --confirm-sha256 x",
        ),
        ("safety_rail.py", "git push --force origin feature"),
        ("safety_rail.py", "git reset --hard HEAD"),
        ("safety_rail.py", "rm -rf /"),
    ],
)
def test_safety_rail_denies(script: str, command: str) -> None:
    assert _hook(script, _terminal(command)).returncode == 2


@pytest.mark.parametrize(
    ("script", "command"),
    [
        ("deny_programming.py", "fpga program a.fpga.json --confirm-sha256 abc --dry-run"),
        ("safety_rail.py", "fpga gates a.fpga.json"),
        ("safety_rail.py", "echo openFPGALoader"),
        ("safety_rail.py", "yosys -p 'synth_ice40' a.v"),
    ],
)
def test_safety_rail_allows(script: str, command: str) -> None:
    assert _hook(script, _terminal(command)).returncode == 0


def test_report_status_lists_failures(tmp_path: Path) -> None:
    reports = tmp_path / "fpga-reports"
    reports.mkdir()
    (reports / "k.fpga-report.json").write_text(
        json.dumps({"verdict": "fail", "checks": [{"id": "fpga.timing", "status": "fail"}]})
    )
    result = _hook("report_fpga_status.py", {"working_dir": str(tmp_path)})
    assert result.returncode == 0
    context = json.loads(result.stdout)["additionalContext"]
    assert "verdict=fail" in context and "fpga.timing" in context


def _launch(
    args: list[str], tmp_path: Path, plugin_root: Path = PLUGIN
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "FPGA_SRC": str(ROOT / "src"),
        "FPGA_TOOLS_IMAGE": "",
        "PYTHONPATH": ":".join(sys.path),
    }
    return subprocess.run(
        [sys.executable, str(plugin_root / "scripts" / "fpga_launcher.py"), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_launcher_fails_closed_without_image(ulx3s: Path, tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugins" / "fpga"
    shutil.copytree(PLUGIN, plugin_root)
    (plugin_root / "tools-image.json").write_text(
        json.dumps(
            {
                "image": "ghcr.io/vibebb/fpga-tools",
                "digest": None,
                "tag": None,
            }
        ),
        encoding="utf-8",
    )
    result = _launch(["validate", str(ulx3s)], tmp_path, plugin_root)
    assert result.returncode == 1
    result = _launch(["validate", str(ulx3s), "--warn"], tmp_path, plugin_root)
    assert result.returncode == 0
    assert json.loads(result.stdout)["verdict"] == "fail"


def test_launcher_program_fails_closed_on_host(ulx3s: Path, tmp_path: Path) -> None:
    result = _launch(["program", str(ulx3s), "--confirm-sha256", "0" * 64, "--dry-run"], tmp_path)
    assert result.returncode != 0
    assert json.loads(result.stdout)["verdict"] == "fail"
