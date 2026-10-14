"""Sibling interchange: the circuit connectivity export the FPGA agent
consumes and the pin map export the circuit agent consumes back. Both are
plain JSON files; no sibling code is imported.

electrical-circuit-agent writes one ``<design>.firmware.json``
(``circuit_firmware_connectivity``) listing every programmable device it
knows about; an FPGA appears there under its reference designator exactly
like an MCU, so both firmware-agent and fpga-agent read the same export.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CircuitDevicePin(_Strict):
    pin: str = Field(min_length=1)
    function: str | None = None
    net: str | None = None
    signal_class: str | None = None
    voltage_v: float | None = Field(default=None, ge=0)


class CircuitDevice(_Strict):
    ref: str
    lib_id: str
    value: str | None = None
    footprint: str
    pins: list[CircuitDevicePin]


class CircuitConnectivity(_Strict):
    """``<design>.firmware.json`` written by electrical-circuit-agent."""

    schema_version: Literal[1]
    system: Literal["circuit"]
    artifact_kind: Literal["circuit_firmware_connectivity"]
    design: str
    source: Literal["brief", "netlist"]
    brief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    netlist_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mcus: list[CircuitDevice] = Field(min_length=1)

    def device(self, ref: str) -> CircuitDevice | None:
        return next((m for m in self.mcus if m.ref == ref), None)


class PinmapEntry(_Strict):
    port: str
    package_pin: str
    net: str | None
    io_standard: str | None
    pull: str | None


class FpgaPinmap(_Strict):
    """``<name>.fpga-pinmap.json`` exported for electrical-circuit-agent."""

    schema_version: Literal[1] = 1
    system: Literal["fpga"] = "fpga"
    artifact_kind: Literal["fpga_pinmap"] = "fpga_pinmap"
    design: str
    contract_sha256: str
    circuit_sha256: str | None = None
    device_ref: str
    device_profile: str
    part: str
    package: str
    io_voltage_max_v: float
    pins: list[PinmapEntry]
    free_pins: list[str]


class RegmapField(_Strict):
    name: str
    lsb: int
    width: int
    mask: int
    access: Literal["ro", "rw", "wo", "w1c"]
    description: str


class RegmapRegister(_Strict):
    name: str
    offset: int
    access: Literal["ro", "rw", "wo", "w1c"]
    reset: int
    description: str
    fields: list[RegmapField]


class FpgaRegmap(_Strict):
    """``<name>.fpga-regmap.json`` exported for firmware-agent.

    Field ``access`` is resolved (a field without its own inherits the
    register's) and ``mask`` is precomputed so the reader re-derives nothing.
    """

    schema_version: Literal[1] = 1
    system: Literal["fpga"] = "fpga"
    artifact_kind: Literal["fpga_regmap"] = "fpga_regmap"
    design: str
    contract_sha256: str
    device_ref: str
    bus: Literal["spi", "i2c", "uart"]
    i2c_address: int | None
    data_width: int
    address_width: int
    registers: list[RegmapRegister]


class FpgaProduction(_Strict):
    """``<name>.fpga-production.json``: the gated bitstream and how to load it,
    exported for production-engineering-agent's programming operation.

    ``bitstream`` and the last ``argv`` element are relative to the directory
    holding this file.
    """

    schema_version: Literal[1] = 1
    system: Literal["fpga"] = "fpga"
    artifact_kind: Literal["fpga_production"] = "fpga_production"
    design: str
    contract_sha256: str
    gate_report_sha256: str
    device_ref: str
    device_profile: str
    part: str
    package: str
    bitstream: str
    bitstream_sha256: str
    bitstream_bytes: int
    target: Literal["sram", "flash"]
    tool: Literal["openFPGALoader"] = "openFPGALoader"
    board: str | None
    cable: str | None
    argv: list[str]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_circuit(path: Path) -> CircuitConnectivity:
    return CircuitConnectivity.model_validate(json.loads(path.read_text(encoding="utf-8")))
