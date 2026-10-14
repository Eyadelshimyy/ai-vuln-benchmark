from __future__ import annotations

from pathlib import Path

import pytest
from ellements.fslm import FSLMEvent, FSLMKernel, det, load_machine_definition, machine


def has_flag(ctx) -> bool:
    return bool(ctx.event.payload.get("flag"))


def mark_seen(ctx) -> dict[str, bool]:
    return {"seen": True}


def emit_payload(ctx) -> dict[str, str]:
    return {"state": ctx.snapshot.current_state, "event": ctx.event.type}


def write_marker(ctx) -> dict[str, str]:
    path = Path(ctx.event.payload["marker"])
    path.write_text("marked", encoding="utf-8")
    return {"path": str(path)}


@pytest.mark.asyncio
async def test_deterministic_callable_guard_effect_output_and_action(tmp_path: Path) -> None:
    m = machine("callable-demo", initial="a")
    with m.state("a") as s:
        s.on("go").to("b", "go_b").when(
            det.guard("has_flag", has_flag)
        ).effect(
            det.effect("mark_seen", mark_seen)
        ).emit(
            det.output("DemoOutput", emit_payload)
        ).call(
            write_marker,
            name="write_marker",
        )
    m.state("b")

    marker = tmp_path / "marker.txt"
    result = await FSLMKernel(m.build()).step(
        m.build().initial_snapshot(),
        FSLMEvent(type="go", payload={"flag": True, "marker": str(marker)}),
    )

    assert result.status == "transitioned"
    assert result.new_snapshot.variables["seen"] is True
    assert result.outputs[0].payload == {"event": "go", "state": "b"}
    assert result.actions[0].status == "executed"
    assert marker.read_text(encoding="utf-8") == "marked"


@pytest.mark.asyncio
async def test_yaml_uses_companion_binding_module(tmp_path: Path) -> None:
    bindings = tmp_path / "bindings.py"
    bindings.write_text(
        "def allowed(ctx):\n"
        "    return ctx.event.payload.get('ok') is True\n"
        "\n"
        "def patch(ctx):\n"
        "    return {'patched': ctx.event.type}\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "machine.yaml"
    spec_path.write_text(
        """
name: yaml-demo
initial: start
bindings:
  imports:
    custom: ./bindings.py
states:
  start:
    transitions:
      - name: finish
        event: go
        target: done
        guards:
          - id: allowed
            kind: deterministic
            ref: custom.allowed
        effect_calls:
          - id: patch
            ref: custom.patch
  done:
    terminal: true
""",
        encoding="utf-8",
    )

    definition = load_machine_definition(spec_path)
    result = await FSLMKernel(definition).step(
        definition.initial_snapshot(),
        FSLMEvent(type="go", payload={"ok": True}),
    )

    assert result.selected_transition == "finish"
    assert result.new_snapshot.current_state == "done"
    assert result.new_snapshot.variables["patched"] == "go"


@pytest.mark.asyncio
async def test_yaml_uses_builtin_payload_guard() -> None:
    spec = {
        "name": "builtin-demo",
        "initial": "start",
        "states": {
            "start": {
                "transitions": [
                    {
                        "name": "finish",
                        "event": "go",
                        "target": "done",
                        "guards": [
                            {
                                "id": "ok",
                                "kind": "deterministic",
                                "ref": "builtin.payload_equals",
                                "args": {
                                    "path": "$.event.payload.ok",
                                    "value": True,
                                },
                            }
                        ],
                    }
                ]
            },
            "done": {},
        },
    }
    from ellements.fslm import MachineSpec

    machine_spec = MachineSpec.model_validate(spec)
    result = await FSLMKernel(machine_spec).step(
        machine_spec.initial_snapshot(),
        FSLMEvent(type="go", payload={"ok": True}),
    )

    assert result.status == "transitioned"
    assert result.new_snapshot.current_state == "done"
