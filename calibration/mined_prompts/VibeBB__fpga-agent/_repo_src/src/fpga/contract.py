"""The FPGA contract (``<name>.fpga.json``): the single source of truth.

The contract declares the target device, the HDL sources and external
libraries (with their licenses), the top-level design, clocks, the pin map
(top-level port bit -> package pin -> circuit net), resource budgets, and
the simulation and formal runs. Every gate reads this file; the generated
constraint file, pin map export and reports are projections of it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1
IDENT = r"^[a-z][a-z0-9_]*$"
HDL_NAME = r"^[A-Za-z][A-Za-z0-9_]*$"
PORT_BIT = r"^[A-Za-z][A-Za-z0-9_]*(\[[0-9]+\])?$"

Language = Literal["vhdl", "verilog", "systemverilog"]
Caution = Literal["config", "jtag"]
ParamValue = int | bool | str


def _relative(value: str) -> str:
    if not value or Path(value).is_absolute() or "\\" in value:
        raise ValueError(f"{value!r} must be a relative POSIX path")
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Device(_Strict):
    profile: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    ref: str = Field(default="U1", min_length=1)


class Source(_Strict):
    path: str
    language: Language
    library: str = Field(default="work", pattern=IDENT)

    _path = field_validator("path")(_relative)


class Library(_Strict):
    """An external HDL library used unmodified (e.g. Colibri).

    ``files`` are listed in compilation order relative to ``root``.
    ``components`` names design units (VHDL entities/packages, Verilog
    modules, ``<name>.f`` file lists, or ``all``) resolved to ordered
    files by scanning ``root``; both may be combined. ``exclude`` lists
    root-relative glob patterns skipped while scanning (e.g. upstream
    test setups that are not part of the synthesizable library).
    ``license_file``
    points at a root-relative license text covering files that carry no
    SPDX header. The license, source URL and pinned revision are recorded
    so the obligation each library carries is visible in every report.
    """

    name: str = Field(pattern=IDENT)
    root: str
    language: Language = "vhdl"
    files: list[str] = Field(default_factory=list[str])
    components: list[str] = Field(default_factory=list[str])
    exclude: list[str] = Field(default_factory=list[str])
    license: str = Field(min_length=2)
    license_file: str | None = None
    source_url: str = Field(pattern=r"^https://")
    revision: str = Field(min_length=7)

    _root = field_validator("root")(_relative)

    @field_validator("license_file")
    @classmethod
    def _license_file(cls, value: str | None) -> str | None:
        return _relative(value) if value is not None else value

    @field_validator("files")
    @classmethod
    def _files(cls, value: list[str]) -> list[str]:
        return [_relative(item) for item in value]

    @field_validator("exclude")
    @classmethod
    def _exclude(cls, value: list[str]) -> list[str]:
        for item in value:
            if item.startswith("/") or ".." in PurePosixPath(item).parts:
                raise ValueError(f"invalid exclude pattern {item!r}")
        return value

    @field_validator("components")
    @classmethod
    def _components(cls, value: list[str]) -> list[str]:
        for item in value:
            if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.$-]*", item):
                raise ValueError(f"invalid component name {item!r}")
        return value

    @field_validator("name")
    @classmethod
    def _not_work(cls, value: str) -> str:
        if value in ("work", "std", "ieee"):
            raise ValueError(f"library name {value!r} is reserved")
        return value

    @model_validator(mode="after")
    def _some_input(self) -> Library:
        if not self.files and not self.components:
            raise ValueError(f"library {self.name}: declare files or components")
        return self


class Clock(_Strict):
    port: str = Field(pattern=PORT_BIT)
    frequency_mhz: float = Field(gt=0, le=1000)
    # nextpnr times one corner of the profile speed grade; the guard band
    # covers voltage/temperature/process corners it does not model.
    margin_pct: float = Field(default=0, ge=0, lt=100)

    @property
    def required_mhz(self) -> float:
        return round(self.frequency_mhz * (1 + self.margin_pct / 100), 9)


class Pin(_Strict):
    port: str = Field(pattern=PORT_BIT)
    package_pin: str = Field(min_length=1)
    net: str | None = None
    io_standard: str | None = None
    pull: Literal["up", "down", "none"] | None = None
    acknowledge: list[Caution] = Field(default_factory=list[Caution])
    rationale: str = ""


class Budget(_Strict):
    logic_pct: float = Field(default=80, gt=0, le=100)
    ram_pct: float = Field(default=90, gt=0, le=100)
    dsp_pct: float = Field(default=90, gt=0, le=100)
    io_pct: float = Field(default=100, gt=0, le=100)
    clock_pct: float = Field(default=100, gt=0, le=100)
    pll_pct: float = Field(default=100, gt=0, le=100)

    def limit(self, resource_class: str) -> float | None:
        return {
            "logic": self.logic_pct,
            "ram": self.ram_pct,
            "dsp": self.dsp_pct,
            "io": self.io_pct,
            "clock": self.clock_pct,
            "pll": self.pll_pct,
        }.get(resource_class)


class Build(_Strict):
    dir: str = "build"
    constraints: str
    bitstream: str
    vhdl_standard: Literal["93", "08"] = "08"
    parameters: dict[str, ParamValue] = Field(default_factory=dict[str, ParamValue])
    defines: dict[str, str] = Field(default_factory=dict[str, str])
    seed: int = Field(default=1, ge=0)
    budget: Budget = Field(default_factory=Budget)

    _paths = field_validator("dir", "constraints", "bitstream")(_relative)

    @field_validator("parameters", "defines")
    @classmethod
    def _names(cls, value: dict[str, object]) -> dict[str, object]:
        for key in value:
            if not re.match(HDL_NAME, key):
                raise ValueError(f"{key!r} is not an HDL identifier")
        return value


class SimCoverage(_Strict):
    """Structural coverage floors for one simulation, measured on ``measure`` design sources.

    Verilog runs are rebuilt with Verilator ``--coverage-line`` (Icarus has no
    coverage); VHDL runs are elaborated with NVC ``--cover=statement,branch``.
    """

    measure: list[str] = Field(min_length=1)
    min_line_pct: float = Field(ge=0, le=100)
    min_branch_pct: float = Field(ge=0, le=100)
    rationale: str = Field(min_length=12)


class Simulation(_Strict):
    id: str = Field(pattern=IDENT)
    runner: Literal["nvc", "iverilog"]
    top: str = Field(pattern=HDL_NAME)
    sources: list[Source] = Field(min_length=1)
    expect: list[str] = Field(min_length=1)
    forbid: list[str] = Field(default_factory=list[str])
    timeout_s: float = Field(default=120, gt=0, le=3600)
    waveform: bool = False
    stop_time: str | None = Field(default=None, pattern=r"^[0-9]+(fs|ps|ns|us|ms)$")
    coverage: SimCoverage | None = None


class Formal(_Strict):
    id: str = Field(pattern=IDENT)
    top: str = Field(pattern=HDL_NAME)
    mode: Literal["prove", "bmc", "cover"]
    depth: int = Field(ge=1, le=1000)
    engine: Literal["smtbmc yices", "smtbmc z3", "smtbmc boolector", "abc pdr"] = "smtbmc yices"
    sources: list[Source] = Field(default_factory=list[Source])
    parameters: dict[str, ParamValue] = Field(default_factory=dict[str, ParamValue])
    timeout_s: float = Field(default=300, gt=0, le=7200)


class CircuitLink(_Strict):
    connectivity: str
    unused_pins: list[str] = Field(default_factory=list[str])

    _path = field_validator("connectivity")(_relative)


class Programmer(_Strict):
    """openFPGALoader target used by the host-only ``fpga program`` command."""

    board: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")
    cable: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")
    write_flash: bool = False


class PowerEstimate(_Strict):
    """Cited coefficients for a bottom-up power estimate from placed-cell counts.

    nextpnr reports no power; ``uw_per_mhz`` is the dynamic power of one used
    cell per MHz at 100 % toggle, copied with ``static_w`` from the vendor
    power calculator named in ``source``.
    """

    static_w: float = Field(ge=0)
    toggle_rate: float = Field(gt=0, le=1)
    uw_per_mhz: dict[str, float] = Field(min_length=1)
    source: str = Field(min_length=1)

    @field_validator("uw_per_mhz")
    @classmethod
    def _positive(cls, value: dict[str, float]) -> dict[str, float]:
        bad = sorted(cell for cell, coefficient in value.items() if coefficient <= 0)
        if bad:
            raise ValueError(f"uw_per_mhz must be > 0: {', '.join(bad)}")
        return value


class Thermal(_Strict):
    """Authored package thermal facts handed to simulation-agent; nothing is estimated here.

    ``power_w`` comes from the vendor power estimator or a measurement named in
    ``source``; the theta path is passed on as authored, and simulation-agent
    reports a package without one as ``unknown``.
    """

    ambient_c: float
    power_w: float = Field(ge=0)
    tj_max_c: float
    derating_margin_c: float = Field(default=0, ge=0)
    theta_ja_c_per_w: float | None = Field(default=None, ge=0)
    theta_jc: float | None = Field(default=None, ge=0)
    theta_cs: float | None = Field(default=None, ge=0)
    theta_sa: float | None = Field(default=None, ge=0)
    source: str = Field(min_length=1)
    response_path: str | None = Field(default=None, min_length=1)
    estimate: PowerEstimate | None = None

    @model_validator(mode="after")
    def _theta_path(self) -> Thermal:
        if self.tj_max_c - self.derating_margin_c <= self.ambient_c:
            raise ValueError("tj_max_c minus derating_margin_c must exceed ambient_c")
        return self

    @field_validator("response_path")
    @classmethod
    def _response(cls, value: str | None) -> str | None:
        return None if value is None else _relative(value)


Access = Literal["ro", "rw", "wo", "w1c"]
REG_NAME = r"^[a-z](?:_?[a-z0-9])*$"


class RegisterField(_Strict):
    name: str = Field(pattern=REG_NAME)
    lsb: int = Field(ge=0)
    width: int = Field(ge=1)
    access: Access | None = None
    description: str = ""


class Register(_Strict):
    name: str = Field(pattern=REG_NAME)
    offset: int = Field(ge=0)
    access: Access
    reset: int = Field(default=0, ge=0)
    description: str = ""
    fields: list[RegisterField] = Field(default_factory=list[RegisterField])


class RegisterMap(_Strict):
    """Registers the host MCU reaches over ``bus``; ``offset`` is the register address.

    ``hdl_package`` is the generated constants file the RTL must use (a VHDL
    package listed in ``sources`` or a Verilog include), so the HDL and the
    firmware header come from the same numbers.
    """

    bus: Literal["spi", "i2c", "uart"]
    i2c_address: int | None = Field(default=None, ge=0x08, le=0x77)
    data_width: Literal[8, 16, 32]
    address_width: int = Field(ge=1, le=16)
    hdl_package: str
    registers: list[Register] = Field(min_length=1)

    _hdl_package = field_validator("hdl_package")(_relative)

    @model_validator(mode="after")
    def _layout(self) -> RegisterMap:
        if (self.bus == "i2c") != (self.i2c_address is not None):
            raise ValueError("registers: i2c_address is required for i2c and only for i2c")
        names = [r.name for r in self.registers]
        offsets = [str(r.offset) for r in self.registers]
        for label, values in (("register name", names), ("register offset", offsets)):
            duplicates = sorted({v for v in values if values.count(v) > 1})
            if duplicates:
                raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")
        for reg in self.registers:
            if reg.offset >= 1 << self.address_width:
                raise ValueError(
                    f"register {reg.name}: offset {reg.offset} needs more than "
                    f"{self.address_width} address bits"
                )
            if reg.reset >= 1 << self.data_width:
                raise ValueError(f"register {reg.name}: reset does not fit {self.data_width} bits")
            used = 0
            field_names = [f.name for f in reg.fields]
            if len(set(field_names)) != len(field_names):
                raise ValueError(f"register {reg.name}: duplicate field name")
            for fld in reg.fields:
                if fld.lsb + fld.width > self.data_width:
                    raise ValueError(
                        f"register {reg.name}.{fld.name}: bits {fld.lsb}+{fld.width} "
                        f"exceed {self.data_width}"
                    )
                mask = ((1 << fld.width) - 1) << fld.lsb
                if used & mask:
                    raise ValueError(f"register {reg.name}.{fld.name}: overlaps another field")
                used |= mask
        _unique_identifiers(self.registers)
        return self


def _unique_identifiers(registers: list[Register]) -> None:
    """Reject maps whose generated HDL or firmware C names would collide.

    ``a`` + field ``b_lsb`` and register ``a_b`` + field ``lsb`` both yield
    ``A_B_LSB_...`` style names, so every suffix the generators emit is
    enumerated here once.
    """
    seen: dict[str, str] = {}
    for reg in registers:
        base = reg.name.upper()
        names = [base, f"{base}_ADDR", f"{base}_RESET", f"{base}_WRITABLE"]
        for fld in reg.fields:
            stem = f"{base}_{fld.name.upper()}"
            names += [f"{stem}_{s}" for s in ("LSB", "WIDTH", "SHIFT", "MASK")]
        for name in names:
            owner = seen.setdefault(name, reg.name)
            if owner != reg.name or names.count(name) > 1:
                raise ValueError(f"register {reg.name}: generated name {name} collides")


class FpgaContract(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    system: Literal["fpga"] = "fpga"
    artifact_kind: Literal["fpga_contract"] = "fpga_contract"
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    description: str = ""
    device: Device
    top: str = Field(pattern=HDL_NAME)
    sources: list[Source] = Field(min_length=1)
    libraries: list[Library] = Field(default_factory=list[Library])
    clocks: list[Clock] = Field(default_factory=list[Clock])
    pins: list[Pin] = Field(min_length=1)
    build: Build
    simulations: list[Simulation] = Field(default_factory=list[Simulation])
    formal: list[Formal] = Field(default_factory=list[Formal])
    circuit: CircuitLink | None = None
    programmer: Programmer | None = None
    thermal: Thermal | None = None
    registers: RegisterMap | None = None

    @model_validator(mode="after")
    def _invariants(self) -> FpgaContract:
        for label, values in (
            ("pin port", [p.port for p in self.pins]),
            ("package pin", [p.package_pin for p in self.pins]),
            ("clock port", [c.port for c in self.clocks]),
            ("library", [lib.name for lib in self.libraries]),
            ("simulation id", [s.id for s in self.simulations]),
            ("formal id", [f.id for f in self.formal]),
        ):
            duplicates = sorted({v for v in values if values.count(v) > 1})
            if duplicates:
                raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")
        family = self.design_language
        if any(_family(s.language) != family for s in self.sources):
            raise ValueError("design sources must be all VHDL or all Verilog/SystemVerilog")
        if family == "vhdl":
            declared = {lib.name for lib in self.libraries} | {"work"}
            for source in self.sources:
                if source.library not in declared:
                    raise ValueError(f"{source.path}: library {source.library} is not declared")
        else:
            if any(s.library != "work" for s in self.sources):
                raise ValueError("Verilog sources have no library; leave library as work")
            if any(lib.language == "vhdl" for lib in self.libraries):
                raise ValueError("VHDL libraries need a VHDL design")
        for sim in self.simulations:
            expected = "nvc" if family == "vhdl" else "iverilog"
            if sim.runner != expected:
                raise ValueError(f"simulation {sim.id}: {family} designs run on {expected}")
            if any(_family(s.language) != family for s in sim.sources):
                raise ValueError(f"simulation {sim.id}: testbench language differs from design")
            if sim.coverage is not None:
                stray = sorted(set(sim.coverage.measure) - {s.path for s in self.sources})
                if stray:
                    raise ValueError(f"simulation {sim.id}: coverage measures non-design {stray}")
        if self.registers is not None:
            package = self.registers.hdl_package
            suffixes = ("_regs_pkg.vhd",) if family == "vhdl" else ("_regs.vh",)
            if not package.endswith(suffixes):
                raise ValueError(f"registers.hdl_package must end with {' or '.join(suffixes)}")
            listed = any(s.path == package for s in self.sources)
            if family == "vhdl" and not listed:
                raise ValueError("registers.hdl_package must be listed in sources")
            if family == "verilog" and listed:
                raise ValueError("registers.hdl_package is a Verilog include, not a source")
        for run in self.formal:
            if any(_family(s.language) != family for s in run.sources):
                raise ValueError(f"formal {run.id}: property language differs from design")
            if run.engine == "abc pdr" and run.mode != "prove":
                raise ValueError(f"formal {run.id}: abc pdr only proves")
        return self

    @property
    def design_language(self) -> Literal["vhdl", "verilog"]:
        return _family(self.sources[0].language)

    def pin(self, port: str) -> Pin | None:
        return next((p for p in self.pins if p.port == port), None)

    def simulation(self, sim_id: str) -> Simulation | None:
        return next((s for s in self.simulations if s.id == sim_id), None)

    def formal_run(self, run_id: str) -> Formal | None:
        return next((f for f in self.formal if f.id == run_id), None)


def _family(language: Language) -> Literal["vhdl", "verilog"]:
    return "vhdl" if language == "vhdl" else "verilog"


def load_contract(path: Path) -> FpgaContract:
    return FpgaContract.model_validate(json.loads(path.read_text(encoding="utf-8")))


def resolve(contract_path: Path, relative: str) -> Path:
    return (contract_path.parent / relative).resolve()
