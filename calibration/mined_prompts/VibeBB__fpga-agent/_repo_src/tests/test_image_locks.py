from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.measure_image_tools import measure
from scripts.print_locked_image import locked_image
from scripts.pull_locked_image import main as pull_main
from scripts.update_image_digest_lock import update_lock

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "docker" / "image-digests.json"
PIN = ROOT / "plugins" / "fpga" / "tools-image.json"
NULL_LOCK = {
    "fpga_tools": {
        "image": "ghcr.io/vibebb/fpga-tools",
        "digest": None,
        "tag": None,
    }
}


def test_fpga_lock_entry_has_valid_shape() -> None:
    entries = json.loads(LOCK.read_text(encoding="utf-8"))
    assert "fpga_tools" in entries
    lock_entry = entries["fpga_tools"]
    assert lock_entry["image"] == "ghcr.io/vibebb/fpga-tools"
    if lock_entry["digest"] is None:
        assert lock_entry["tag"] is None
    else:
        assert isinstance(lock_entry["digest"], str)
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", lock_entry["digest"])
        assert isinstance(lock_entry["tag"], str) and lock_entry["tag"]

    pin_entry = json.loads(PIN.read_text(encoding="utf-8"))
    assert {key: pin_entry[key] for key in ("image", "digest", "tag")} == {
        key: lock_entry[key] for key in ("image", "digest", "tag")
    }


def test_launcher_ignores_initial_null_lock_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FPGA_TOOLS_IMAGE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plugin_root = tmp_path / "plugins" / "fpga"
    plugin_root.mkdir(parents=True)
    (plugin_root / "tools-image.json").write_text(
        json.dumps(NULL_LOCK["fpga_tools"]),
        encoding="utf-8",
    )
    repository_lock = tmp_path / "docker" / "image-digests.json"
    repository_lock.parent.mkdir(parents=True)
    repository_lock.write_text(json.dumps(NULL_LOCK), encoding="utf-8")
    code = (
        "import importlib.util, json, pathlib, sys; "
        "spec = importlib.util.spec_from_file_location('fpga_launcher', sys.argv[1]); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "print(json.dumps(module.image_ref(pathlib.Path(sys.argv[2]))))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(ROOT / "plugins/fpga/scripts/fpga_launcher.py"),
            str(plugin_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "null"


def test_print_locked_image_rejects_initial_null_entry(tmp_path: Path) -> None:
    lock = tmp_path / "image-digests.json"
    lock.write_text(json.dumps(NULL_LOCK), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        locked_image(lock, "fpga_tools")


def test_update_initial_null_entry(tmp_path: Path) -> None:
    lock = tmp_path / "image-digests.json"
    lock.write_text(json.dumps(NULL_LOCK), encoding="utf-8")
    digest = "sha256:" + "b" * 64
    changed = update_lock(
        lock,
        entry="fpga_tools",
        image="ghcr.io/vibebb/fpga-tools",
        tag="abc123-tools",
        digest=digest,
        published_at="2026-09-30T00:00:00Z",
        workflow_run="https://github.com/VibeBB/fpga-agent/actions/runs/1",
        dockerfile="docker/fpga-tools.Dockerfile",
        tools={"python": "python --version: Python 3.12.14"},
    )
    assert changed
    data = json.loads(lock.read_text(encoding="utf-8"))
    assert data["fpga_tools"]["image"] == "ghcr.io/vibebb/fpga-tools"
    assert data["fpga_tools"]["digest"] == digest
    assert data["fpga_tools"]["tag"] == "abc123-tools"
    assert data["fpga_tools"]["tools"] == {"python": "python --version: Python 3.12.14"}


def test_update_lock_records_optional_attestation(tmp_path: Path) -> None:
    lock = tmp_path / "image-digests.json"
    lock.write_text(json.dumps(NULL_LOCK), encoding="utf-8")
    attestation = "https://github.com/VibeBB/fpga-agent/attestations/1"

    assert update_lock(
        lock,
        entry="fpga_tools",
        image="ghcr.io/vibebb/fpga-tools",
        tag="abc123-tools",
        digest="sha256:" + "b" * 64,
        published_at="2026-09-30T00:00:00Z",
        workflow_run="https://github.com/VibeBB/fpga-agent/actions/runs/1",
        dockerfile="docker/fpga-tools.Dockerfile",
        tools={"python": "python --version: Python 3.12.14"},
        attestation=attestation,
    )
    data = json.loads(lock.read_text(encoding="utf-8"))
    assert data["fpga_tools"]["attestation"] == attestation
    assert locked_image(lock, "fpga_tools") == ("ghcr.io/vibebb/fpga-tools@sha256:" + "b" * 64)


def test_update_rejects_wrong_image(tmp_path: Path) -> None:
    lock = tmp_path / "image-digests.json"
    lock.write_text(json.dumps(NULL_LOCK), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected image"):
        update_lock(
            lock,
            entry="fpga_tools",
            image="ghcr.io/vibebb/other-tools",
            tag="abc123-tools",
            digest="sha256:" + "b" * 64,
            published_at="2026-09-30T00:00:00Z",
            workflow_run="https://github.com/VibeBB/fpga-agent/actions/runs/1",
            dockerfile="docker/fpga-tools.Dockerfile",
            tools={"python": "python --version: Python 3.12.14"},
        )


def test_pull_locked_image_uses_digest_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lock = tmp_path / "lock.json"
    digest = "sha256:" + "c" * 64
    lock.write_text(
        json.dumps(
            {
                "fpga_tools": {
                    "image": "ghcr.io/vibebb/fpga-tools",
                    "digest": digest,
                    "tag": "abc123-tools",
                }
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def run(command: list[str], **_kwargs: object) -> Result:
        calls.append(command)
        return Result()

    monkeypatch.setattr("scripts.pull_locked_image.subprocess.run", run)
    record = tmp_path / "record.json"
    assert pull_main(["--lock", str(lock), "--entry", "fpga_tools", "--record", str(record)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert calls == [["docker", "pull", f"ghcr.io/vibebb/fpga-tools@{digest}"]]
    assert output["status"] == "pulled"
    assert json.loads(record.read_text(encoding="utf-8")) == output


def test_measure_records_required_probe_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "".join(
        f"{name}\t{command}\t{version}\n"
        for name, command, version in (
            ("python", "python --version", "Python 3.12.14"),
            ("uv", "uv --version", "uv 0.12.22"),
            ("fpga-agent", "python -c ...", "0.1.0"),
            ("ghdl-yosys-plugin", "yosys -m ghdl -p help ghdl -q", "GHDL plugin"),
            ("yosys", "yosys --version", "Yosys 0.62"),
            ("nvc", "nvc --version", "NVC 1.23.0"),
            ("iverilog", "iverilog -V", "Icarus Verilog version 14.0"),
            ("vvp", "vvp -V", "Icarus Verilog runtime version 14.0"),
            ("verilator", "verilator --version", "Verilator 5.0"),
            ("sby", "sby --help", "usage: sby"),
            ("yosys-smtbmc", "yosys-smtbmc --help", "yosys-smtbmc [options]"),
            ("yices-smt2", "yices-smt2 --version", "Yices 2.7"),
            ("nextpnr-ice40", "nextpnr-ice40 --version", "nextpnr-ice40 0.8"),
            ("nextpnr-ecp5", "nextpnr-ecp5 --version", "nextpnr-ecp5 0.8"),
            ("nextpnr-himbaechel", "nextpnr-himbaechel --version", "nextpnr-himbaechel 0.8"),
            ("icepack", "icepack -h", "Usage: icepack"),
            ("ecppack", "ecppack --help", "Usage: ecppack"),
            ("gowin_pack", "gowin_pack -h", "Usage: gowin_pack"),
            ("ghdl", "ghdl --version", "GHDL 5.0"),
            ("z3", "z3 --version", "Z3 version 4.13"),
            ("boolector", "boolector --version", "Boolector 3.2"),
            ("openFPGALoader", "openFPGALoader -V", "openFPGALoader 1.1"),
        )
    )

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        script = command[-1]
        for tool in (
            "yosys",
            "ghdl",
            "vvp",
            "sby",
            "yosys-smtbmc",
            "yices-smt2",
            "nextpnr-ice40",
            "nextpnr-ecp5",
            "nextpnr-himbaechel",
            "icepack",
            "ecppack",
            "gowin_pack",
            "iverilog",
            "verilator",
            "nvc",
            "z3",
            "boolector",
            "openFPGALoader",
        ):
            assert tool in script
        assert "--network" in command
        assert command[command.index("--network") + 1] == "none"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("scripts.measure_image_tools.subprocess.run", run)
    values = measure("fpga-tools:local")
    assert set(values) == {
        "ecppack",
        "fpga-agent",
        "ghdl-yosys-plugin",
        "ghdl",
        "gowin_pack",
        "boolector",
        "icepack",
        "iverilog",
        "nextpnr-ecp5",
        "nextpnr-himbaechel",
        "nextpnr-ice40",
        "nvc",
        "openFPGALoader",
        "python",
        "sby",
        "uv",
        "vvp",
        "verilator",
        "yosys",
        "yosys-smtbmc",
        "yices-smt2",
        "z3",
    }
    assert "3.12.14" in values["python"]
    assert "0.8" in values["nextpnr-ecp5"]
    assert "1.23.0" in values["nvc"]
