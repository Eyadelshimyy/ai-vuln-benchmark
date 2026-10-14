from __future__ import annotations

import json

from ellements.fslm import FSLMEvent, LocalFSLMStore, load_machine_spec, machine
from ellements.fslm.cli import main


def test_local_store_roundtrips_snapshot(tmp_path) -> None:
    m = machine("demo", initial="a")
    m.state("a")
    spec = m.build()
    snapshot = spec.initial_snapshot()
    store = LocalFSLMStore(tmp_path)

    import asyncio

    asyncio.run(store.save_snapshot(snapshot))
    loaded = asyncio.run(store.load_snapshot())

    assert loaded is not None
    assert loaded.current_state == "a"


def test_load_machine_spec_accepts_python_builder_export(tmp_path) -> None:
    spec_path = tmp_path / "demo_machine.py"
    spec_path.write_text(
        "from ellements.fslm import machine\n\n"
        "machine_spec = machine('demo', initial='a')\n"
        "machine_spec.state('a')\n",
        encoding="utf-8",
    )

    spec = load_machine_spec(spec_path)

    assert spec.name == "demo"
    assert spec.initial == "a"


def test_fslm_validate_cli_accepts_python_machine(tmp_path, capsys) -> None:
    spec_path = tmp_path / "demo_machine.py"
    spec_path.write_text(
        "from ellements.fslm import machine\n\n"
        "def build():\n"
        "    m = machine('demo', initial='a')\n"
        "    m.state('a')\n"
        "    return m.build()\n",
        encoding="utf-8",
    )

    assert main(["validate", str(spec_path)]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["ok"] is True


def test_fslm_step_cli_runs_one_transition(tmp_path, capsys) -> None:
    m = machine("demo", initial="a")
    with m.state("a") as s:
        s.on("go").to("b", "go_b")
    m.state("b")
    spec = m.build()
    spec_path = tmp_path / "demo_machine.py"
    snapshot_path = tmp_path / "snapshot.json"
    event_path = tmp_path / "event.json"
    spec_path.write_text(
        "from ellements.fslm import machine\n\n"
        "def build():\n"
        "    m = machine('demo', initial='a')\n"
        "    with m.state('a') as s:\n"
        "        s.on('go').to('b', 'go_b')\n"
        "    m.state('b')\n"
        "    return m.build()\n",
        encoding="utf-8",
    )
    snapshot_path.write_text(
        json.dumps(spec.initial_snapshot().model_dump(mode="json")),
        encoding="utf-8",
    )
    event_path.write_text(
        json.dumps(FSLMEvent(type="go").model_dump(mode="json")),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "step",
                str(spec_path),
                "--snapshot",
                str(snapshot_path),
                "--event",
                str(event_path),
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["selected_transition"] == "go_b"
    assert out["new_snapshot"]["current_state"] == "b"


def test_fslm_init_state_cli_accepts_machine_id(tmp_path, capsys) -> None:
    spec_path = tmp_path / "demo_machine.py"
    spec_path.write_text(
        "from ellements.fslm import machine\n\n"
        "def build():\n"
        "    m = machine('demo', initial='a')\n"
        "    m.state('a')\n"
        "    return m.build()\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "init-state",
                str(spec_path),
                "--machine-id",
                "asset-1",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["machine_id"] == "asset-1"


def test_fslm_step_cli_renders_rich_transition(tmp_path, capsys) -> None:
    m = machine("demo", initial="a")
    with m.state("a") as s:
        s.on("go").to("b", "go_b")
    m.state("b")
    spec = m.build()
    spec_path = tmp_path / "demo_machine.py"
    snapshot_path = tmp_path / "snapshot.json"
    event_path = tmp_path / "event.json"
    spec_path.write_text(
        "from ellements.fslm import machine\n\n"
        "def build():\n"
        "    m = machine('demo', initial='a')\n"
        "    with m.state('a') as s:\n"
        "        s.on('go').to('b', 'go_b')\n"
        "    m.state('b')\n"
        "    return m.build()\n",
        encoding="utf-8",
    )
    snapshot_path.write_text(
        json.dumps(spec.initial_snapshot(machine_id="asset-1").model_dump(mode="json")),
        encoding="utf-8",
    )
    event_path.write_text(
        json.dumps(FSLMEvent(type="go").model_dump(mode="json")),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "step",
                str(spec_path),
                "--snapshot",
                str(snapshot_path),
                "--event",
                str(event_path),
                "--format",
                "rich",
                "--color",
                "never",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "FSLM State Movement" in out
    assert "go_b" in out
    assert "transitioned" in out
