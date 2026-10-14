"""RED integration contracts for ignored Rust operation-specific deadlines."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_ignored_rust_shard.py"
_SPEC = importlib.util.spec_from_file_location("ignored_rust_deadlines", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
sharder = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sharder
_SPEC.loader.exec_module(sharder)


def _metadata() -> str:
    """Return one default-tested workspace library target."""
    return json.dumps(
        {
            "workspace_members": ["core-id"],
            "packages": [
                {
                    "id": "core-id",
                    "name": "mlsirm-core",
                    "targets": [
                        {
                            "name": "mlsirm_core",
                            "kind": ["lib"],
                            "crate_types": ["lib"],
                            "test": True,
                            "doctest": False,
                        }
                    ],
                }
            ],
        }
    )


def test_runner_routes_metadata_inventory_and_study_through_distinct_operations(monkeypatch) -> None:
    """Each subprocess class uses the deadline policy matching its expected duration."""
    calls: list[tuple[tuple[str, ...], object]] = []

    def fake_run(command, *, operation, **_kwargs):
        calls.append((tuple(command), operation))
        if command[:2] == ["cargo", "metadata"]:
            return subprocess.CompletedProcess(command, 0, _metadata(), "")
        if "--list" in command:
            return subprocess.CompletedProcess(command, 0, "study::one: test\n", "")
        return subprocess.CompletedProcess(command, 0, None, None)

    monkeypatch.setattr(sharder, "run_bounded", fake_run)

    assert sharder.run_shard(0, 1, []) == 0
    assert [operation for _command, operation in calls] == [
        sharder.SubprocessOperation.CARGO_METADATA,
        sharder.SubprocessOperation.CARGO_TEST_LIST,
        sharder.SubprocessOperation.STATISTICAL_TEST,
    ]


def test_inventory_uses_capture_and_check_with_bounded_list_operation(monkeypatch) -> None:
    """Ignored-test discovery retains exact stdout parsing and fail-fast exit semantics."""
    target = sharder.CargoTarget("mlsirm-core", "lib", "mlsirm_core")
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "alpha::one: test\n", "")

    monkeypatch.setattr(sharder, "run_bounded", fake_run)
    inventory = sharder.inventory_ignored_tests([target])

    assert [test.test_name for test in inventory] == ["alpha::one"]
    assert observed["operation"] is sharder.SubprocessOperation.CARGO_TEST_LIST
    assert observed["check"] is True
    assert observed["capture_output"] is True
    assert observed["text"] is True


def test_main_emits_redacted_machine_timeout_evidence(monkeypatch, capsys) -> None:
    """A timeout exits distinctly and reports only package-owned bounded evidence."""
    error = sharder.BoundedSubprocessTimeout(
        operation=sharder.SubprocessOperation.STATISTICAL_TEST,
        timeout_seconds=1800.0,
    )
    monkeypatch.setattr(
        sharder,
        "run_shard",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    assert sharder.main(["--shard", "0", "--shards", "1"]) == 124
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload == error.as_dict()
    assert "command" not in captured.err.lower()
