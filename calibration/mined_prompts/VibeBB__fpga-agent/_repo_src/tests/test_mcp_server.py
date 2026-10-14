from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import mcp.types
import pytest

from fpga import mcp_server, service


def _call_result(name: str, arguments: dict[str, object]) -> mcp.types.CallToolResult:
    async def invoke() -> mcp.types.CallToolResult:
        result = await mcp_server.call_tool(name, arguments)
        assert isinstance(result, mcp.types.CallToolResult)
        return result

    return asyncio.run(invoke())


def _payload(result: mcp.types.CallToolResult) -> dict[str, Any]:
    assert result.content
    assert isinstance(result.content[0], mcp.types.TextContent)
    return json.loads(result.content[0].text)


def test_unknown_tool_returns_transport_error() -> None:
    result = _call_result("fpga_missing", {})
    assert result.isError is True
    assert _payload(result) == {
        "verdict": "fail",
        "detail": "unknown tool fpga_missing",
        "error_type": "unknown_tool",
    }


def test_handler_exception_returns_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_doctor() -> service.Json:
        raise RuntimeError("handler failed")

    monkeypatch.setattr(service, "doctor_payload", fail_doctor)
    result = _call_result("fpga_doctor", {})
    assert result.isError is True
    assert _payload(result) == {
        "verdict": "fail",
        "detail": "fpga_doctor error: handler failed",
        "error_type": "RuntimeError",
    }


@pytest.mark.parametrize("verdict", ["fail", "unknown"])
def test_gate_verdicts_are_not_transport_errors(
    verdict: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))

    def gate_result(_contract_path: Path) -> service.Json:
        return {"verdict": verdict}

    monkeypatch.setattr(service, "validate_payload", gate_result)
    result = _call_result("fpga_validate", {"contract_path": "design.fpga.json"})
    assert result.isError is False
    assert _payload(result)["verdict"] == verdict


def test_mcp_contains_relative_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))
    captured: dict[str, Path | None] = {}

    def capture_gates(contract_path: Path, out_dir: Path | None, *, full: bool) -> service.Json:
        captured["contract_path"] = contract_path
        captured["out_dir"] = out_dir
        assert full is False
        return {"verdict": "unknown"}

    monkeypatch.setattr(service, "gates_payload", capture_gates)
    result = _call_result(
        "fpga_check",
        {"contract_path": "contracts/device.fpga.json", "out_dir": "reports"},
    )
    assert result.isError is False
    assert captured == {
        "contract_path": tmp_path / "contracts" / "device.fpga.json",
        "out_dir": tmp_path / "reports",
    }


def test_mcp_accepts_absolute_contract_inside_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))
    captured: dict[str, Path] = {}

    def capture_validate(contract_path: Path) -> service.Json:
        captured["contract_path"] = contract_path
        return {"verdict": "pass"}

    monkeypatch.setattr(service, "validate_payload", capture_validate)
    contract_path = tmp_path / "contracts" / "device.fpga.json"
    result = _call_result("fpga_validate", {"contract_path": str(contract_path)})
    assert result.isError is False
    assert captured == {"contract_path": contract_path}


@pytest.mark.parametrize("contract_path", ["../outside.fpga.json", "/tmp/outside.fpga.json"])
def test_mcp_rejects_contract_outside_workspace(
    contract_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))
    result = _call_result("fpga_validate", {"contract_path": contract_path})
    assert result.isError is True
    assert _payload(result)["error_type"] == "ValueError"
    assert "path is outside the workspace" in _payload(result)["detail"]


def test_mcp_rejects_out_dir_outside_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))
    result = _call_result(
        "fpga_check",
        {
            "contract_path": "design.fpga.json",
            "out_dir": str(tmp_path.parent / "outside-reports"),
        },
    )
    assert result.isError is True
    assert _payload(result)["error_type"] == "ValueError"
    assert "path is outside the workspace" in _payload(result)["detail"]
