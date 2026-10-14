"""Runtime path through the real DockerTestExecutor with a stand-in Docker process.

These tests exercise the real argument building, mount layout, exit-status
mapping, and artifact collection. They do NOT run Docker or an EVM. They are
not the Docker integration test; that one is in test_runtime_integration.py.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from app.adapters.discovery.runtime import RuntimeEngine
from app.discovery.capabilities import ResultStatus
from tests.test_phase48.phase48_support import configure, document, request, row


class _Process:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b""

    def kill(self) -> None:
        return None


def _fake_docker(monkeypatch: Any, *, exit_code: int, write: bool = True, rows=None):
    seen: list[list[str]] = []

    async def create(*args: str, **kwargs: Any) -> _Process:
        del kwargs
        seen.append(list(args))
        mounts = [args[index + 1] for index, item in enumerate(args) if item == "-v"]
        output = next(item.split(":")[0] for item in mounts if ":/bugforge-output:rw" in item)
        command = shlex.split(args[-1])
        manifest = json.loads(Path(command[command.index("--input") + 1]).read_text())
        if write:
            text = document(manifest, rows(manifest) if rows else None)
            Path(output, "runtime.json").write_text(text, encoding="utf-8")
        return _Process(exit_code)

    monkeypatch.setattr("app.execution.docker_executor.asyncio.create_subprocess_exec", create)
    return seen


def test_real_executor_path_collects_the_artifact_and_binds_identity(
    tmp_path: Path, monkeypatch
) -> None:
    configure(monkeypatch)
    calls = _fake_docker(monkeypatch, exit_code=0)
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.INGESTED
    assert result.metadata["transaction_index"] == "1"
    args = calls[0]
    assert args[:2] == ["docker", "run"]
    assert args[args.index("--network") + 1] == "none"
    assert args[args.index("--memory") + 1] == "512m"
    assert args[args.index("--cpus") + 1] == "1.0"
    assert "--read-only" in args
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in args
    joined = " ".join(args)
    assert "docker.sock" not in joined
    assert " pull" not in joined
    assert f"{tmp_path}:/bugforge-src:ro" in args
    assert any(item.endswith(":/bugforge-output:rw") for item in args)
    assert args[args.index("local/runtime:pinned") + 1 :][:2] == ["sh", "-c"]
    assert "--input" in shlex.split(args[-1])


def test_real_executor_path_maps_a_nonzero_exit_to_a_tool_failure(
    tmp_path: Path, monkeypatch
) -> None:
    configure(monkeypatch)
    _fake_docker(monkeypatch, exit_code=7)
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.TOOL_FAILURE
    assert result.exit_code == 7
    assert result.runtime_evidence is None
    assert result.metadata["document_present"] == "true"


def test_real_executor_path_without_an_artifact_creates_no_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    configure(monkeypatch)
    _fake_docker(monkeypatch, exit_code=0, write=False)
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.runtime_evidence is None


def test_real_executor_path_rejects_a_document_for_another_transaction(
    tmp_path: Path, monkeypatch
) -> None:
    configure(monkeypatch)
    _fake_docker(monkeypatch, exit_code=0, rows=lambda manifest: [row(manifest, index="0")])
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.metadata["binding"].startswith("not_found")
