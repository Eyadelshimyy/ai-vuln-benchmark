from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).parents[1] / "plugins" / "fpga" / "hooks" / "scripts"
VISION_SCRIPT = SCRIPTS / "record_vision_tool_event.py"
OBSERVATION_SCRIPT = SCRIPTS / "record_image_observation.py"
ATTACHMENT_SCRIPT = SCRIPTS / "intake_attachments.py"

_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c626001000000ffff03000006000557bfabd40000000049"
    "454e44ae426082"
)


def _run(
    script: Path,
    payload: dict[str, Any] | str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_env = dict(os.environ)
    process_env.pop("OPENHANDS_PROJECT_DIR", None)
    process_env.pop("FPGA_IMAGE_OBSERVATIONS", None)
    process_env.pop("FPGA_VISION_TOOL_EVENTS", None)
    if env:
        process_env.update(env)
    return subprocess.run(
        [sys.executable, str(script)],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env=process_env,
    )


def _vision_payload(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "working_dir": str(tmp_path),
        "session_id": "session-1",
        "tool_name": "inspect_image_with_vision",
        "tool_input": {"image_index": 0, "question": "Check the PWM waveform"},
        "tool_response": {
            "answer": "The PWM signal is high for the stated duty cycle.",
            "profile_name": "vision",
            "model": "vision-model-1",
        },
    }
    payload.update(overrides)
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _vision_events(tmp_path: Path) -> list[dict[str, Any]]:
    return _jsonl(tmp_path / "observations" / "fpga" / "vision-tool-events.jsonl")


def _image_observations(tmp_path: Path) -> list[dict[str, Any]]:
    return _jsonl(tmp_path / "observations" / "fpga" / "image-observations.jsonl")


def test_record_vision_tool_event_appends_provenance(tmp_path: Path) -> None:
    payload = _vision_payload(tmp_path, agent_name="fpga-review", tool_call_id="call-42")

    result = _run(VISION_SCRIPT, payload)

    assert result.returncode == 0
    records = _vision_events(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["tool_name"] == "inspect_image_with_vision"
    assert record["profile_name"] == "vision"
    assert record["model"] == "vision-model-1"
    assert record["image_index"] == 0
    assert record["question"] == "Check the PWM waveform"
    assert record["response_sha256"].startswith("sha256:")
    assert record["actor"] == {"agent_name": "fpga-review", "tool_call_id": "call-42"}
    assert record["tool_call_id"] == "call-42"
    assert record["sequence"] == 1


def test_record_vision_tool_event_skips_errors_and_invalid_responses(
    tmp_path: Path,
) -> None:
    for response in (
        {"error": "vision profile missing"},
        {"is_error": True, "answer": "x", "profile_name": "vision", "model": "m"},
        {"answer": " ", "profile_name": "vision", "model": "m"},
        {"answer": "ok", "profile_name": "", "model": "m"},
        {"answer": "ok", "profile_name": "vision"},
        "not-a-dict",
    ):
        assert (
            _run(
                VISION_SCRIPT,
                _vision_payload(tmp_path, tool_response=response),
            ).returncode
            == 0
        )
    assert _run(VISION_SCRIPT, "{not-json").returncode == 0
    assert _vision_events(tmp_path) == []


def test_record_vision_tool_event_honors_path_override(tmp_path: Path) -> None:
    override = tmp_path / "custom" / "vision.jsonl"

    result = _run(
        VISION_SCRIPT,
        _vision_payload(tmp_path),
        env={"FPGA_VISION_TOOL_EVENTS": str(override)},
    )

    assert result.returncode == 0
    assert len(_jsonl(override)) == 1
    assert not (tmp_path / "observations").exists()


def test_record_image_observation_logs_file_editor_view(tmp_path: Path) -> None:
    image = tmp_path / "intake" / "attachments" / "capture.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(_PNG)

    result = _run(
        OBSERVATION_SCRIPT,
        {
            "working_dir": str(tmp_path),
            "session_id": "session-1",
            "tool_name": "file_editor",
            "tool_input": {"command": "view", "path": "intake/attachments/capture.png"},
            "tool_response": {"output": "ok"},
            "subagent_type": "fpga-review",
            "action_id": "act-8",
        },
    )

    assert result.returncode == 0
    records = _image_observations(tmp_path)
    assert len(records) == 1
    assert records[0]["image_path"] == str(image)
    assert records[0]["image_sha256"] == hashlib.sha256(_PNG).hexdigest()
    assert records[0]["actor"] == {"action_id": "act-8", "subagent_type": "fpga-review"}


def test_record_image_observation_skips_non_views_and_errors(tmp_path: Path) -> None:
    image = tmp_path / "capture.png"
    image.write_bytes(_PNG)
    for payload in (
        {
            "working_dir": str(tmp_path),
            "tool_name": "file_editor",
            "tool_input": {"command": "create", "path": str(image)},
            "tool_response": {"output": "ok"},
        },
        {
            "working_dir": str(tmp_path),
            "tool_name": "file_editor",
            "tool_input": {"command": "view", "path": "capture.svg"},
            "tool_response": {"output": "ok"},
        },
        {
            "working_dir": str(tmp_path),
            "tool_name": "file_editor",
            "tool_input": {"command": "view", "path": str(image)},
            "tool_response": {"error": "view failed"},
        },
        {
            "working_dir": str(tmp_path),
            "tool_name": "terminal",
            "tool_input": {"command": "cat capture.png"},
            "tool_response": {"output": "ok"},
        },
        {
            "working_dir": str(tmp_path),
            "tool_name": "fpga_check",
            "tool_input": {},
            "tool_response": {"output": str(image)},
        },
    ):
        assert _run(OBSERVATION_SCRIPT, payload).returncode == 0
    assert _run(OBSERVATION_SCRIPT, "{not-json").returncode == 0
    assert _image_observations(tmp_path) == []


def _write_event(events: Path, name: str, source: str, image_urls: list[str]) -> None:
    event = {
        "id": name,
        "source": source,
        "llm_message": {
            "role": "user",
            "content": (
                [{"type": "image", "image_urls": image_urls}]
                if image_urls
                else [{"type": "text", "text": "hello"}]
            ),
        },
    }
    (events / name).write_text(json.dumps(event), encoding="utf-8")


def _run_attachments(
    payload: dict[str, Any],
    events_dir: Path | None,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_env = dict(os.environ)
    process_env.pop("FPGA_AGENT_EVENTS_DIR", None)
    process_env.pop("FPGA_INTAKE_ATTACHMENTS_DIR", None)
    if events_dir is not None:
        process_env["FPGA_AGENT_EVENTS_DIR"] = str(events_dir)
    if env:
        process_env.update(env)
    return subprocess.run(
        [sys.executable, str(ATTACHMENT_SCRIPT)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        encoding="utf-8",
        check=False,
        env=process_env,
    )


def test_intake_attachments_materializes_user_images_idempotently(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events"
    events.mkdir()
    encoded = "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii")
    _write_event(events, "event-1.json", "user", [encoded])
    _write_event(events, "event-2.json", "agent", [encoded])
    workdir = tmp_path / "work"
    workdir.mkdir()

    result = _run_attachments({"working_dir": str(workdir)}, events)

    assert result.returncode == 0
    attachments = workdir / "intake" / "attachments"
    images = list(attachments.glob("*.png"))
    assert len(images) == 1
    assert images[0].read_bytes() == _PNG
    manifest = attachments / "manifest.jsonl"
    records = _jsonl(manifest)
    assert len(records) == 1
    assert records[0]["sha256"] == hashlib.sha256(_PNG).hexdigest()
    assert records[0]["materialized"] is True

    assert _run_attachments({"working_dir": str(workdir)}, events).returncode == 0
    assert len(list(attachments.glob("*.png"))) == 1
    assert len(_jsonl(manifest)) == 1


def test_intake_attachments_records_unmaterialized_urls(tmp_path: Path) -> None:
    events = tmp_path / "events"
    events.mkdir()
    _write_event(events, "event-1.json", "user", ["https://example.test/board.png"])
    workdir = tmp_path / "work"
    workdir.mkdir()

    result = _run_attachments({"working_dir": str(workdir)}, events)

    assert result.returncode == 0
    record = _jsonl(workdir / "intake" / "attachments" / "manifest.jsonl")[0]
    assert record["materialized"] is False
    assert record["reason"] == "non-data-url"


def test_intake_attachments_uses_session_event_path_and_override(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    events = home / ".openhands" / "agent-canvas" / "dev_conversations" / "session-9" / "events"
    events.mkdir(parents=True)
    encoded = "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii")
    _write_event(events, "event-1.json", "user", [encoded])
    workdir = tmp_path / "work"
    workdir.mkdir()

    result = _run_attachments(
        {"working_dir": str(workdir), "session_id": "session-9"},
        None,
        env={"HOME": str(home)},
    )

    assert result.returncode == 0
    assert list((workdir / "intake" / "attachments").glob("*.png"))


def test_intake_attachments_fails_open_without_events(tmp_path: Path) -> None:
    result = _run_attachments({"working_dir": str(tmp_path)}, None)

    assert result.returncode == 0
    assert not (tmp_path / "intake").exists()


def test_plugin_and_agent_vision_hooks_are_declared() -> None:
    plugin_root = Path(__file__).resolve().parents[1] / "plugins" / "fpga"
    hooks = json.loads((plugin_root / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    expected = {
        "session_start": {
            "fpga-doctor",
            "intake-attachments",
            "ensure-llm-profiles",
            "ensure-agent-profiles",
            "require-records",
        },
        "user_prompt_submit": {"intake-attachments"},
        "stop": {"require-records", "report-fpga-status", "intake-attachments"},
        "post_tool_use": {"record-image-observation", "record-vision-tool-event"},
    }
    for event, names in expected.items():
        actual = {hook["name"] for group in hooks[event] for hook in group["hooks"]}
        assert actual == names
    for name in ("fpga-architect", "fpga-blind-review", "fpga-developer", "fpga-review"):
        text = (plugin_root / "agents" / f"{name}.md").read_text(encoding="utf-8")
        assert "name: record-image-observation" in text
        assert "name: record-vision-tool-event" in text
