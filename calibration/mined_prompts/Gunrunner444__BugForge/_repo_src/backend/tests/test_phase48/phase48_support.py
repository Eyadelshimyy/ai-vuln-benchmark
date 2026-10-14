"""Shared Phase 48 test helpers. These build scripted sandbox output, not a real EVM."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.discovery.engine import AnalysisRequest
from app.execution.base import ExecutionConfig, ExecutionResult

IDENTITY: dict[str, str] = {
    "project_id": "p",
    "source_snapshot": "snap",
    "compiler_configuration": "cfg",
    "runtime_configuration": "pinned-local",
    "function_identity": "Vault.sol::Vault.deposit(uint256)",
    "deployment_address": "0x1",
    "sequence_id": "seq",
    "transaction_index": "1",
    "actor": "user",
    "state_snapshot": "genesis",
    "chain_id": "31337",
    "block_number": "1",
}


def request(tmp_path: Path, **overrides: str) -> AnalysisRequest:
    extra = {**IDENTITY, **{key: value for key, value in overrides.items() if key != "contract"}}
    extra = {key: value for key, value in extra.items() if value != "<drop>"}
    return AnalysisRequest(
        tmp_path,
        "solidity",
        target="Vault",
        contract=overrides.get("contract", "Vault"),
        extra=extra,
    )


def configure(monkeypatch: Any, *, fork_source: str = "", fork_enabled: bool = False) -> None:
    prefix = "app.adapters.discovery.runtime."
    monkeypatch.setattr(prefix + "_runtime_image", lambda: "local/runtime:pinned")
    monkeypatch.setattr(prefix + "_docker_present", lambda: True)
    monkeypatch.setattr(prefix + "_local_image", lambda: True)
    monkeypatch.setattr(prefix + "_fork_enabled", lambda: fork_enabled)
    monkeypatch.setattr(prefix + "_fork_source", lambda: fork_source)


def manifest_of(config: ExecutionConfig) -> dict[str, str]:
    value = json.loads(Path(config.command[2]).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def row(manifest: dict[str, str], **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "index": manifest["transaction_index"],
        "sequence_id": manifest["sequence_id"],
        "actor": manifest["actor"] or "user",
        "contract": manifest["contract"] or "Vault",
        "address": manifest["deployment_address"] or "0x1",
        "function_identity": manifest["function_identity"] or "Vault.sol::Vault.deposit(uint256)",
        "identity_established": True,
        "selector": "0xb6b55f25",
        "success": True,
        "return_data": "0x",
        "gas_used": "21000",
        "call_type": "CALL",
    }
    base.update(overrides)
    return base


def document(
    manifest: dict[str, str],
    rows: list[dict[str, object]] | None = None,
    **shared: object,
) -> str:
    contract = manifest["contract"] or "Vault"
    address = manifest["deployment_address"] or "0x1"
    payload: dict[str, object] = {
        "schema": "bugforge-runtime-v1",
        "request_identity": manifest["request_identity"],
        "execution_id": manifest["execution_id"],
        "mode": manifest["mode"],
        "target": manifest["target"],
        "project": manifest["project"],
        "source_snapshot": manifest["source_snapshot"],
        "compiler_configuration": manifest["compiler_configuration"],
        "runtime_environment": "local",
        "runtime_configuration": manifest["runtime_configuration"],
        "chain_id": manifest["chain_id"] or "31337",
        "block_number": manifest["block_number"] or "1",
        "block_timestamp": "0",
        "state_snapshot": manifest["state_snapshot"] or "genesis",
        "tool": "bugforge-runtime",
        "tool_version": "bugforge-runtime-v1",
        "duration": "0.01",
        "sequence_id": manifest["sequence_id"],
        "deployments": [{"address": address, "contract": contract, "identity": contract}],
        "transactions": rows if rows is not None else [row(manifest)],
    }
    payload.update(shared)
    return json.dumps(payload)


Builder = Callable[[dict[str, str], int], ExecutionResult]


class Runner:
    """Scripted stand-in for the executor boundary. Records every config it sees."""

    def __init__(self, builder: Builder | None = None) -> None:
        self.configs: list[ExecutionConfig] = []
        self.manifests: list[dict[str, str]] = []
        self.builder = builder or self.ok

    @staticmethod
    def ok(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest)}
        )

    def __call__(self, config: ExecutionConfig) -> ExecutionResult:
        self.configs.append(config)
        manifest = manifest_of(config)
        self.manifests.append(manifest)
        return self.builder(manifest, len(self.configs) - 1)


def install(monkeypatch: Any, runner: Runner) -> None:
    monkeypatch.setattr("app.adapters.discovery.runtime._execute", runner)
