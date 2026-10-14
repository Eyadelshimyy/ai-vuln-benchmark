"""Tests for simulation rendering modes ('lite' vs 'demo'/'realistic').

Covers CLI flag parsing, simulation engine execution, presentation lighting
and shadowing toggles, ML tracking overlay painting, and report serialization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clave.cli import _build_parser
from clave.sim.debug_run import run
from clave.sim.overlay import _with_shadows, _without_shadows, paint_boxes
from clave.sim.report import DebugRunError, _report_document, report_lines

ROOT = Path(__file__).resolve().parents[2]


def test_cli_parser_defaults_to_none_and_resolves_lite() -> None:
    """Default CLI arguments result in lite render_mode."""
    parser = _build_parser()
    args = parser.parse_args(["sim", "--seconds", "1.0", "--no-window"])
    assert args.render_mode is None
    assert args.demo_mode is False
    assert args.lite is False


def test_cli_parser_handles_render_mode_choices_and_flags() -> None:
    """CLI arguments support --mode, --lite, and --demo/--realistic flags."""
    parser = _build_parser()

    args_lite = parser.parse_args(["sim", "--mode", "lite"])
    assert args_lite.render_mode == "lite"

    args_demo = parser.parse_args(["sim", "--mode", "demo"])
    assert args_demo.render_mode == "demo"

    args_realistic = parser.parse_args(["sim", "--mode", "realistic"])
    assert args_realistic.render_mode == "realistic"

    args_flag_lite = parser.parse_args(["sim", "--lite"])
    assert args_flag_lite.lite is True

    args_flag_demo = parser.parse_args(["sim", "--demo"])
    assert args_flag_demo.demo_mode is True

    args_flag_realistic = parser.parse_args(["sim", "--realistic"])
    assert args_flag_realistic.demo_mode is True


def test_run_executes_in_lite_mode(tmp_path: Path) -> None:
    """run() with render_mode='lite' completes and reports render_mode='lite'."""
    report = run(
        root=ROOT,
        out=tmp_path,
        seconds=0.4,
        seed=0,
        window=False,
        video=False,
        frames=False,
        render_mode="lite",
        progress=False,
    )
    assert report.render_mode == "lite"
    lines = report_lines(report)
    assert any("render mode     lite" in line for line in lines)


def test_run_executes_in_demo_mode(tmp_path: Path) -> None:
    """run() with render_mode='demo' builds presentation scene and reports demo."""
    report = run(
        root=ROOT,
        out=tmp_path,
        seconds=0.4,
        seed=0,
        window=False,
        video=False,
        frames=False,
        render_mode="demo",
        progress=False,
    )
    assert report.render_mode == "demo"
    lines = report_lines(report)
    assert any("render mode     demo" in line for line in lines)
    document = _report_document(report)
    assert document["render_mode"] == "demo"


def test_run_refuses_invalid_render_mode(tmp_path: Path) -> None:
    """run() raises DebugRunError when an unknown render_mode is given."""
    with pytest.raises(DebugRunError, match="unknown render_mode"):
        run(
            root=ROOT,
            out=tmp_path,
            seconds=0.2,
            seed=0,
            window=False,
            video=False,
            frames=False,
            render_mode="invalid_mode",
            progress=False,
        )


def test_shadow_toggles_modify_renderer_scene_flags() -> None:
    """_with_shadows and _without_shadows toggle the mjRND_SHADOW flag."""
    mujoco = pytest.importorskip("mujoco")

    class MockScene:
        def __init__(self) -> None:
            self.flags = [0] * 10

    class MockRenderer:
        def __init__(self) -> None:
            self.scene = MockScene()

    renderer = MockRenderer()
    _with_shadows(renderer)
    assert renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] == 1

    _without_shadows(renderer)
    assert renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] == 0


def test_paint_boxes_renders_in_lite_and_demo_modes() -> None:
    """paint_boxes() modifies frame pixels in both lite and demo modes."""
    frame_lite = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [(10, 10, 50, 50)]
    labels = ["OBJ_01"]

    painted_lite = paint_boxes(frame_lite, boxes, labels=labels, render_mode="lite")
    assert painted_lite.sum() > 0

    frame_demo = np.zeros((100, 100, 3), dtype=np.uint8)
    painted_demo = paint_boxes(frame_demo, boxes, labels=labels, render_mode="demo")
    assert painted_demo.sum() > 0
