"""Unit tests for ``fpga.libraries`` component resolution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from fpga.contract import Library, load_contract
from fpga.libraries import resolve_contract_libraries, resolve_library, scan_library


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _vhdl(path: Path, decl: str, uses: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join([*(f"use {u};" for u in uses or []), decl, ""])
    path.write_text(body, encoding="utf-8")
    return path


def _lib(tmp_path: Path, name: str, **kw: object) -> Library:
    fields: dict[str, object] = {
        "name": name,
        "root": f"third_party/{name}",
        "language": "vhdl",
        "components": [],
        "files": ["dummy.vhdl"],
        "license": "MIT",
        "source_url": "https://example.com/x",
        "revision": "0123456789abcdef",
    }
    fields.update(kw)
    (tmp_path / "third_party" / name).mkdir(parents=True, exist_ok=True)
    return Library.model_validate(fields)


def test_scan_indexes_entities_and_packages(tmp_path: Path) -> None:
    root = tmp_path / "third_party" / "colibri"
    _vhdl(root / "src/io/uart.vhdl", "entity uart is\nend entity;")
    _vhdl(root / "src/common/types.vhdl", "package types is\nend package;")
    _vhdl(root / "src/common/types_b.vhdl", "package body types is\nend package body;")
    _vhdl(root / "sim/tb.vhdl", "entity tb is\nend entity;")
    index = scan_library(root, "vhdl")
    assert set(index.providers) == {"uart", "types"}
    assert "types" in index.bodies
    assert "tb" not in index.providers


def test_components_expand_dependencies(tmp_path: Path) -> None:
    root = tmp_path / "third_party" / "colibri"
    _vhdl(root / "src/common/utils.vhdl", "package utils is\nend package;")
    _vhdl(root / "src/common/utils_b.vhdl", "package body utils is\nend package body;")
    _vhdl(root / "src/io/uart_rx.vhdl", "entity uart_rx is\nend entity;", ["colibri.utils"])
    _vhdl(root / "src/io/uart_tx.vhdl", "entity uart_tx is\nend entity;", ["colibri.utils"])
    uart = _vhdl(
        root / "src/io/uart.vhdl",
        "entity uart is\nend entity;",
        ["entity colibri.uart_rx", "entity colibri.uart_tx"],
    )
    lib = _lib(tmp_path, "colibri", components=["uart"], files=[])
    res = resolve_library(tmp_path / "x.fpga.json", lib, ["colibri"])
    assert res.problems == []
    assert res.files[-1] == uart
    names = [p.name for p in res.files]
    assert names == [
        "utils.vhdl",
        "utils_b.vhdl",
        "uart_rx.vhdl",
        "uart_tx.vhdl",
        "uart.vhdl",
    ]


def test_all_component_selects_everything(tmp_path: Path) -> None:
    root = tmp_path / "third_party" / "lib"
    _vhdl(root / "a/one.vhdl", "entity one is\nend entity;")
    _vhdl(root / "b/two.vhdl", "entity two is\nend entity;")
    lib = _lib(tmp_path, "lib", components=["all"], files=[])
    res = resolve_library(tmp_path / "x.fpga.json", lib, ["lib"])
    assert res.problems == []
    assert len(res.files) == 2


def test_example_dirs_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "third_party" / "lib"
    _vhdl(root / "a/one.vhdl", "entity one is\nend entity;")
    _vhdl(root / "example/demo.vhdl", "entity demo is\nend entity;")
    lib = _lib(tmp_path, "lib", components=["all"], files=[])
    res = resolve_library(tmp_path / "x.fpga.json", lib, ["lib"])
    assert res.problems == []
    assert len(res.files) == 1


def test_unknown_component_reports_unit_names(tmp_path: Path) -> None:
    root = tmp_path / "third_party" / "lib"
    _vhdl(root / "a/one.vhdl", "entity one is\nend entity;")
    lib = _lib(tmp_path, "lib", components=["nosuch"], files=[])
    res = resolve_library(tmp_path / "x.fpga.json", lib, ["lib"])
    assert len(res.problems) == 1
    assert "'nosuch' not found" in res.problems[0]
    assert "one" in res.problems[0]


def test_cross_library_reference_needs_earlier_declaration(tmp_path: Path) -> None:
    root_common = tmp_path / "third_party" / "common"
    root_fifo = tmp_path / "third_party" / "fifo"
    _vhdl(root_common / "rtl/types_pkg.vhdl", "package types_pkg is\nend package;")
    _vhdl(
        root_fifo / "rtl/fifo.vhdl",
        "entity fifo is\nend entity;",
        ["common.types_pkg"],
    )
    fifo = _lib(tmp_path, "fifo", components=["fifo"], files=[])
    res = resolve_library(tmp_path / "x.fpga.json", fifo, ["fifo"])
    assert res.problems
    assert "uses library 'common' which is not declared" in res.problems[0]
    common = _lib(tmp_path, "common", components=["types_pkg"], files=[])
    ordered = resolve_contract_libraries(
        _contract(tmp_path, [common, fifo]), tmp_path / "x.fpga.json"
    )
    assert ordered["fifo"].problems == []
    assert ordered["common"].external == set()


def test_verilog_filelist_and_modules(tmp_path: Path) -> None:
    root = tmp_path / "third_party" / "axis"
    (root / "rtl").mkdir(parents=True)
    (root / "rtl/axis_fifo.v").write_text("module axis_fifo();\nendmodule\n")
    (root / "rtl/axis_adapter.v").write_text(
        '`include "axis_fifo.v"\nmodule axis_adapter();\nendmodule\n'
    )
    (root / "rtl/axis.f").write_text("axis_adapter.v\n")
    lib = _lib(tmp_path, "axis", language="verilog", components=["axis.f"], files=[])
    res = resolve_library(tmp_path / "x.fpga.json", lib, ["axis"])
    assert res.problems == []
    assert [p.name for p in res.files] == ["axis_fifo.v", "axis_adapter.v"]


def test_verilog_instantiation_expands_dependencies(tmp_path: Path) -> None:
    root = tmp_path / "third_party" / "uart"
    (root / "rtl").mkdir(parents=True)
    (root / "rtl/uart_rx.v").write_text("module uart_rx();\nendmodule\n")
    (root / "rtl/uart_tx.v").write_text("module uart_tx();\nendmodule\n")
    (root / "rtl/uart.v").write_text(
        "module uart();\n"
        "uart_rx uart_rx_inst ();\n"
        "uart_tx #(.WIDTH(8)) uart_tx_inst ();\n"
        "endmodule\n"
    )
    lib = _lib(tmp_path, "uart", language="verilog", components=["uart"], files=[])
    res = resolve_library(tmp_path / "x.fpga.json", lib, ["uart"])
    assert res.problems == []
    assert [p.name for p in res.files] == ["uart_rx.v", "uart_tx.v", "uart.v"]


def test_exclude_globs_drop_paths(tmp_path: Path) -> None:
    root = tmp_path / "third_party" / "lib"
    (root / "rtl").mkdir(parents=True)
    (root / "rtl/good.v").write_text("module good();\nendmodule\n")
    (root / "bench").mkdir()
    (root / "bench/tb.v").write_text("module tb();\nendmodule\n")
    lib = _lib(
        tmp_path,
        "lib",
        language="verilog",
        components=["all"],
        files=[],
        exclude=["bench/**"],
    )
    res = resolve_library(tmp_path / "x.fpga.json", lib, ["lib"])
    assert res.problems == []
    assert [p.name for p in res.files] == ["good.v"]


def test_self_reference_and_header_includes(tmp_path: Path) -> None:
    root = tmp_path / "third_party" / "lib"
    (root / "hdl").mkdir(parents=True)
    (root / "hdl/cfg.vh").write_text("`define W 8\n")
    (root / "hdl/core.v").write_text(
        '`include "cfg.vh"\n'
        "module core();\n"
        "core_sub u0 ();\n"
        "`ifdef FORMAL\n"
        '`include "formal_only.vh"\n'
        "`endif\n"
        "endmodule\n"
        "module core_sub();\nendmodule\n"
    )
    lib = _lib(tmp_path, "lib", language="verilog", components=["core"], files=[])
    res = resolve_library(tmp_path / "x.fpga.json", lib, ["lib"])
    # Instantiating a unit declared in the same file is not a cycle; the
    # ifdef-guarded include is skipped, cfg.vh lands in include dirs only.
    assert res.problems == []
    assert [p.name for p in res.files] == ["core.v"]
    assert res.includes == {(root / "hdl").resolve()}


def test_filelist_keywords_and_env_prefix(tmp_path: Path) -> None:
    root = tmp_path / "third_party" / "lib"
    (root / "rtl").mkdir(parents=True)
    (root / "rtl/a.vhdl").write_text("entity a is\nend entity;")
    (root / "rtl/all.f").write_text("file $LIB_HOME/rtl/a.vhdl\n")
    lib = _lib(tmp_path, "lib", components=["all.f"], files=[])
    res = resolve_library(tmp_path / "x.fpga.json", lib, ["lib"])
    assert res.problems == []
    assert [p.name for p in res.files] == ["a.vhdl"]


def test_index_records_unit_kinds(tmp_path: Path) -> None:
    from fpga.libraries import scan_library

    root = tmp_path / "third_party" / "lib"
    _vhdl(root / "a/pkg.vhdl", "package pkg is\nend package;")
    _vhdl(root / "a/pkg_body.vhdl", "package body pkg is\nend package body;")
    _vhdl(root / "a/one.vhdl", "entity one is\nend entity;")
    index = scan_library(root, "vhdl")
    assert index.kinds["pkg"].startswith("package")
    assert index.kinds["one"] == "entity"


def test_files_still_appended(tmp_path: Path) -> None:
    root = tmp_path / "third_party" / "lib"
    extra = _vhdl(root / "extra/extra.vhdl", "entity extra is\nend entity;")
    lib = _lib(tmp_path, "lib", components=[], files=["extra/extra.vhdl"])
    res = resolve_library(tmp_path / "x.fpga.json", lib, ["lib"])
    assert res.problems == []
    assert res.files == [extra]


def test_missing_root_is_problem(tmp_path: Path) -> None:
    lib = _lib(tmp_path, "gone", components=["all"], files=[])
    (tmp_path / "third_party" / "gone").rmdir()
    res = resolve_library(tmp_path / "x.fpga.json", lib, ["gone"])
    assert res.problems == ["gone: root third_party/gone does not exist"]


def _contract(tmp_path: Path, libs: list[Library]):
    contract = {
        "name": "test",
        "device": {"profile": "ice40-up5k-sg48", "ref": "U1"},
        "top": "top",
        "sources": [{"path": "top.vhdl", "language": "vhdl"}],
        "clocks": [{"port": "clk", "frequency_mhz": 12}],
        "pins": [{"port": "clk", "package_pin": "A1"}],
        "build": {"constraints": "x.fpga.pcf", "bitstream": "build/x.bin"},
        "libraries": [json.loads(lib.model_dump_json()) for lib in libs],
    }
    path = tmp_path / "x.fpga.json"
    write_json(path, contract)
    (tmp_path / "top.vhdl").write_text("entity top is\nend entity;\n")
    return load_contract(path)
