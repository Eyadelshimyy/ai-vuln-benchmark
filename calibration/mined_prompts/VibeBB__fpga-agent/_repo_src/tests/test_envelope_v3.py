"""Interchange envelope v3: fpga sidecars and the circuit import check."""

from __future__ import annotations

import json
from pathlib import Path

from fpga._envelope import sidecar_path
from fpga.gates import run_gates


def _provenance(contract: Path) -> tuple[str, str]:
    report = run_gates(contract, contract.parent / "fpga-reports", full=False)
    check = next(c for c in report.checks if c.id == "fpga.import_provenance")
    return check.status, check.detail


def test_circuit_import_provenance_fails_closed(uart_echo: Path) -> None:
    assert _provenance(uart_echo)[0] == "pass"
    side = sidecar_path(uart_echo.parent / "circuit" / "icebreaker.firmware.json")
    body = json.loads(side.read_text(encoding="utf-8"))
    side.write_text(json.dumps(body | {"producer": "mech"}), encoding="utf-8")
    assert _provenance(uart_echo)[0] == "fail"
    side.unlink()
    status, detail = _provenance(uart_echo)
    assert (status, detail.startswith("unknown: ")) == ("fail", True)
