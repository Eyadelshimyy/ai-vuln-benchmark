"""Deterministic PNG renders: canvas primitives, each view, VCD parsing, caps."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any, cast

import pytest

from fpga import render, service
from fpga.contract import load_contract
from fpga.devices import load_profile

EXAMPLES = Path(__file__).parents[1] / "examples"
DATA = Path(__file__).parent / "data"

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _png_check(data: bytes) -> tuple[int, int]:
    assert data.startswith(PNG_SIG)
    assert data[12:16] == b"IHDR"
    width, height, depth, colour = struct.unpack(">IIBB", data[16:26])
    assert depth == 8 and colour == 2
    offset = 8
    idat = b""
    saw_iend = False
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        tag = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        if tag == b"IDAT":
            idat += body
        if tag == b"IEND":
            saw_iend = True
        offset += 12 + length
    assert saw_iend
    raw = zlib.decompress(idat)
    assert len(raw) == height * (1 + width * 3)
    return width, height


def _contract(name: str) -> Any:
    directory = EXAMPLES / name
    return load_contract(directory / next(p.name for p in directory.glob("*.fpga.json")))


def test_canvas_png_valid_and_deterministic() -> None:
    canvas = render.Canvas(64, 32)
    canvas.fill_rect(2, 2, 10, 8, (255, 0, 0))
    canvas.rect(0, 0, 63, 31, (0, 0, 0))
    canvas.line(0, 0, 63, 31, (0, 0, 255))
    canvas.text(4, 20, "A1 ?", (0, 0, 0), 2)
    data = canvas.png_bytes()
    assert _png_check(data) == (64, 32)
    again = render.Canvas(64, 32)
    again.fill_rect(2, 2, 10, 8, (255, 0, 0))
    again.rect(0, 0, 63, 31, (0, 0, 0))
    again.line(0, 0, 63, 31, (0, 0, 255))
    again.text(4, 20, "A1 ?", (0, 0, 0), 2)
    assert again.png_bytes() == data
    different = render.Canvas(64, 32)
    different.text(4, 20, "a1 ?", (0, 0, 0), 2)  # lowercase has real glyphs
    assert different.png_bytes() != data
    with pytest.raises(ValueError):
        render.Canvas(5000, 10)


def test_font_covers_printable_ascii_distinct() -> None:
    import string

    font = render.FONT_ROWS
    for char in string.printable[:95]:
        assert char in font, f"missing glyph {char!r}"
        bitmap = font[char]
        assert len(bitmap) == 35
        assert set(bitmap) <= {" ", "#"}
    bitmaps = {char: font[char] for char in font}
    assert len(set(bitmaps.values())) == len(bitmaps), "duplicate glyphs"


def test_font_specific_bitmaps() -> None:
    def rows(char: str) -> list[str]:
        bitmap = render.FONT_ROWS[char]
        return [bitmap[r * 5 : r * 5 + 5] for r in range(7)]

    assert rows("5") == [
        "#####",
        "#    ",
        "#### ",
        "    #",
        "    #",
        "#   #",
        " ### ",
    ]
    assert rows("S") == [
        " ####",
        "#    ",
        "#    ",
        " ### ",
        "    #",
        "    #",
        "#### ",
    ]
    assert rows("E") == [
        "#####",
        "#    ",
        "#    ",
        "#### ",
        "#    ",
        "#    ",
        "#####",
    ]
    assert rows("6") == [
        "  ## ",
        " #   ",
        "#    ",
        "#### ",
        "#   #",
        "#   #",
        " ### ",
    ]
    assert rows("9") == [
        " ### ",
        "#   #",
        "#   #",
        " ####",
        "    #",
        "   # ",
        " ##  ",
    ]
    assert rows("0") == [
        " ### ",
        "#   #",
        "#  ##",
        "# # #",
        "##  #",
        "#   #",
        " ### ",
    ]
    assert rows("O") == [
        " ### ",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        " ### ",
    ]
    # '-' only the middle row lit; '_' only the bottom row lit
    assert rows("-") == ["     "] * 3 + ["#####"] + ["     "] * 3
    assert rows("_") == ["     "] * 6 + ["#####"]
    assert rows(".") == ["     "] * 5 + [" ##  ", " ##  "]


def test_layout_problems_detects_overlap() -> None:
    canvas = render.Canvas(200, 100)
    canvas.text(4, 4, "alpha", (0, 0, 0), 2)
    canvas.text(8, 8, "beta", (0, 0, 0), 2)
    canvas.text(0, 92, "outside?", (0, 0, 0), 2)
    canvas.fill_rect(120, 4, 20, 20, (0, 0, 0))
    canvas.solid(120, 4, 20, 20)
    canvas.text(124, 8, "on", (0, 0, 0), 2)
    problems = canvas.layout_problems()
    assert len(problems) >= 3


def test_pinmap_canvas() -> None:
    contract = _contract("blinky-tangnano9k")
    profile = load_profile(contract.device.profile, [EXAMPLES / "blinky-tangnano9k"])
    canvas = render.pinmap_canvas(contract, profile)
    _png_check(canvas.png_bytes())
    assert canvas.layout_problems() == []


def test_pinmap_canvas_perimeter() -> None:
    contract = _contract("blinky-tangnano9k")
    profile = load_profile(contract.device.profile, [EXAMPLES / "blinky-tangnano9k"])
    pins = [{"name": str(i + 1)} for i in range(48)]
    perimeter = profile.model_copy(update={"pins": [], "package": "LQFP-48"})
    perimeter = perimeter.model_copy(
        update={"pins": [type(profile.pins[0]).model_validate(p) for p in pins]}
    )
    canvas = render.pinmap_canvas(contract, perimeter)
    _png_check(canvas.png_bytes())
    assert canvas.layout_problems() == []


def _nextpnr_report() -> dict[str, Any]:
    return {
        "utilization": {
            "ICE40_LC": {"used": 120, "available": 5280},
            "ICESTORM_RAM": {"used": 2, "available": 30},
        },
        "fmax": {"clk": {"achieved": 62.5, "constraint": 50.0}},
        "critical_paths": {
            "clk": {
                "from": "count[0]$Q",
                "to": "count[7]$D",
                "total_delay_ns": 15.9,
                "segments": ["seg1", "seg2"],
            }
        },
    }


def test_utilization_and_timing_canvases() -> None:
    contract = _contract("uart-echo-icebreaker")
    profile = load_profile(contract.device.profile, [EXAMPLES / "uart-echo-icebreaker"])
    report = _nextpnr_report()
    utilization = render.utilization_canvas(contract, profile, report)
    timing = render.timing_canvas(contract, report)
    _png_check(utilization.png_bytes())
    _png_check(timing.png_bytes())
    assert utilization.layout_problems() == []
    assert timing.layout_problems() == []


def test_timing_worst_path_total_from_segments() -> None:
    contract = _contract("uart-echo-icebreaker")
    report = _nextpnr_report()
    report["critical_paths"] = {
        "clk": {
            "from": "count[0]$Q",
            "to": "count[7]$D",
            "segments": [
                {"type": "setup", "delay": 10.5, "net": "n1"},
                {"type": "route", "delay": 5.4, "net": "n2"},
            ],
        }
    }
    canvas = render.timing_canvas(contract, report)
    texts = [box[4] for box in canvas.text_boxes]
    assert any("15.90 ns" in t for t in texts), texts
    assert canvas.layout_problems() == []


def test_floorplan_canvas() -> None:
    contract = _contract("uart-echo-icebreaker")
    profile = load_profile(contract.device.profile, [EXAMPLES / "uart-echo-icebreaker"])
    placed = {
        "modules": {
            contract.top: {
                "cells": {
                    "c0": {"type": "ICE40_LC", "attributes": {"NEXTPNR_BEL": "X1/Y2/lc0"}},
                    "c1": {"type": "ICESTORM_RAM", "attributes": {"NEXTPNR_BEL": "X3/Y4"}},
                    "c2": {"type": "SB_IO", "attributes": {"NEXTPNR_BEL": "X0/Y7"}},
                }
            }
        }
    }
    _png_check(render.floorplan_canvas(placed, profile, contract.top).png_bytes())
    _png_check(render.floorplan_canvas({"modules": {}}, profile, contract.top).png_bytes())


def test_floorplan_canvas_real_fixture_classes() -> None:
    contract = _contract("uart-echo-icebreaker")
    profile = load_profile(contract.device.profile, [EXAMPLES / "uart-echo-icebreaker"])
    placed = json.loads((DATA / "uart-echo.placed.json").read_text(encoding="utf-8"))
    canvas = render.floorplan_canvas(placed, profile, contract.top)
    _png_check(canvas.png_bytes())
    assert canvas.layout_problems() == []
    legend = [box[4] for box in canvas.text_boxes]
    classes = {t.split(" (")[0] for t in legend if " (" in t and t.endswith(")")}
    assert len(classes) >= 3, legend


def test_floorplan_canvas_gowin_fixture_classes() -> None:
    profile = load_profile("gowin-gw1nr9c-qn88p", [EXAMPLES / "blinky-tangnano9k"])
    placed = json.loads((DATA / "blinky-tangnano9k.pnr.json").read_text(encoding="utf-8"))
    canvas = render.floorplan_canvas(placed, profile, "blinky")
    _png_check(canvas.png_bytes())
    assert canvas.layout_problems() == []
    legend = [box[4] for box in canvas.text_boxes]
    classes = {t.split(" (")[0] for t in legend if " (" in t and t.endswith(")")}
    assert "other" not in classes, legend
    assert {"logic", "io"} <= classes, legend


def test_floorplan_regex_fallback_keeps_types() -> None:
    data = render.placed_cells(
        '{"modules": {"top": {"cells": {'
        '"a": {"type": "SB_GB", "attributes": {"NEXTPNR_BEL": "X16/Y0"}},'
        '"b": {"type": "SB_IO", "attributes": {"NEXTPNR_BEL": "X0/Y5"}}'
        "}}}",
        "top",
    )
    cells = data["modules"]["top"]["cells"]
    assert {c["type"] for c in cells.values()} == {"SB_GB", "SB_IO"}


def _write_vcd(path: Path, rows: int = 4096) -> None:
    lines = [
        "$timescale 1ns $end",
        "$scope module top $end",
        "$var wire 1 ! clk $end",
        '$var wire 8 " data [7:0] $end',
        "$upscope $end",
        "$enddefinitions $end",
        "#0",
        "0!",
        'b00000000 "',
    ]
    t = 0
    for i in range(rows):
        t += 5
        lines.append(f"#{t}")
        lines.append("1!" if i % 2 == 0 else "0!")
        lines.append(f'b{i % 256:08b} "')
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_waveform_canvas_and_parser(tmp_path: Path) -> None:
    vcd = tmp_path / "sim-count.vcd"
    _write_vcd(vcd, rows=20)
    signals, timescale, truncated = render.parse_vcd(vcd)
    assert timescale == "1ns" and not truncated
    assert [s.name for s in signals] == ["clk", "data"]
    canvas = render.waveform_canvas(vcd, "uart-echo wave count")
    _png_check(canvas.png_bytes())
    assert canvas.layout_problems() == []


def test_waveform_string_and_scope_vars(tmp_path: Path) -> None:
    vcd = tmp_path / "sim-echo.vcd"
    vcd.write_text(
        "\n".join(
            [
                "$timescale 1us $end",
                "$scope module uart_echo $end",
                "$var wire 1 ! clk $end",
                '$var wire 8 " tx [7:0] $end',
                "$var string 1 # done $end",
                "$var string 1 $ state $end",
                "$scope module inner $end",
                "$var wire 8 % tx [7:0] $end",
                "$upscope $end",
                "$upscope $end",
                "$enddefinitions $end",
                "#0",
                "0!",
                'b00000000 "',
                "sfalse #",
                "ss_reset $",
                "b00000000 %",
                "#100",
                "1!",
                'b10101010 "',
                "strue #",
                "ss_run $",
                "b11110000 %",
                "#200",
                "0!",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    signals, timescale, _ = render.parse_vcd(vcd)
    assert timescale == "1us"
    assert any(s.width == 0 for s in signals)
    canvas = render.waveform_canvas(vcd, "uart-echo wave echo")
    _png_check(canvas.png_bytes())
    assert canvas.layout_problems() == []
    texts = {box[4] for box in canvas.text_boxes}
    assert any("us" in t for t in texts), "engineering-unit tick labels"
    assert any(t.startswith("inner.tx") or t == "inner.tx [8]" for t in texts), (
        "duplicate names must be scope-qualified"
    )
    assert any(t == "true" for t in texts), "string value drawn as text"


def test_waveform_row_cap_and_performance(tmp_path: Path) -> None:
    vcd = tmp_path / "sim-big.vcd"
    _write_vcd(vcd, rows=4096)
    import time

    start = time.monotonic()
    canvas = render.waveform_canvas(vcd, "big")
    assert time.monotonic() - start < 10
    _png_check(canvas.png_bytes())
    assert len(render.parse_vcd(vcd)[0]) <= render.MAX_WAVE_ROWS


def test_report_canvas() -> None:
    report = {
        "design": "blinky",
        "scope": "full",
        "verdict": "fail",
        "checks": [
            {"id": "fpga.contract", "subject": "blinky", "status": "pass", "detail": ""},
            {"id": "fpga.timing", "subject": "top", "status": "fail", "detail": "slow"},
            {"id": "fpga.sim", "subject": "", "status": "not_applicable", "detail": "none"},
        ],
    }
    _png_check(render.report_canvas(report).png_bytes())


def test_render_view_pngs_pins_only(tmp_path: Path) -> None:
    contract = _contract("uart-echo-icebreaker")
    profile = load_profile(contract.device.profile, [EXAMPLES / "uart-echo-icebreaker"])
    rendered, skipped = render.render_view_pngs(
        contract, profile, tmp_path, build_dir=tmp_path / "build", views=["all"]
    )
    views = {r["view"] for r in rendered}
    assert "pinmap" in views
    assert {s["view"] for s in skipped} == {
        "utilization",
        "timing",
        "floorplan",
        "waveform",
        "report",
    }
    for item in rendered:
        _png_check(Path(str(item["path"])).read_bytes())


def test_render_failure_never_changes_verdict(tmp_path: Path) -> None:
    contract = _contract("uart-echo-icebreaker")
    profile = load_profile(contract.device.profile, [EXAMPLES / "uart-echo-icebreaker"])
    bad = tmp_path / "build"
    bad.mkdir()
    (bad / f"{contract.name}.nextpnr-report.json").write_text("{not json", encoding="utf-8")
    rendered, skipped = render.render_view_pngs(
        contract, profile, tmp_path, build_dir=bad, views=["all"]
    )
    assert {s["view"] for s in skipped} >= {"utilization", "timing"}
    assert all(r["view"] != "timing" for r in rendered)


def test_mcp_call_returns_image_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    import base64
    import shutil

    import mcp.types

    from fpga import mcp_server

    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))
    work = tmp_path / "proj"
    work.mkdir()
    shutil.copytree(EXAMPLES / "uart-echo-icebreaker", work, dirs_exist_ok=True)

    async def invoke() -> mcp.types.CallToolResult:
        result = await mcp_server.call_tool(
            "fpga_render",
            {"contract_path": str(work / "uart-echo.fpga.json"), "view": "pinmap"},
        )
        assert isinstance(result, mcp.types.CallToolResult)
        return result

    result = asyncio.run(invoke())
    assert result.isError is False
    images = [c for c in result.content if isinstance(c, mcp.types.ImageContent)]
    assert images and images[0].mimeType == "image/png"
    _png_check(base64.b64decode(images[0].data))


def test_render_cli_payload(tmp_path: Path) -> None:
    import shutil

    work = tmp_path / "proj"
    shutil.copytree(EXAMPLES / "uart-echo-icebreaker", work)
    contract = work / "uart-echo.fpga.json"
    payload = service.render_payload(contract, None, "pinmap")
    assert payload["verdict"] == "pass"
    rendered = payload["rendered"]
    assert isinstance(rendered, list)
    first = cast("dict[str, Any]", rendered[0])
    assert isinstance(first, dict)
    record = first
    assert record["view"] == "pinmap"
    path = record["path"]
    assert isinstance(path, str)
    _png_check(Path(path).read_bytes())
