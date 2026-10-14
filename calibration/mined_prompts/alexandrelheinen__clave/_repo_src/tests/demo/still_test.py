"""Tests for the still capture.

A still is presentation, so what these guard is that it stays presentation:
the lighting reaches the world as an argument and never as a change to the
configuration every dataset, training run and benchmark reads.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from clave.demo.still import (
    StillError,
    StillScenario,
    _let_the_arm_pass_through,
    write_png,
)

ROOT = Path(__file__).resolve().parents[2]
STILLS = ROOT / "configs" / "stills"


def test_every_shipped_still_loads() -> None:
    """A still is named and nothing else has to be supplied."""

    files = sorted(STILLS.glob("*.yml"))
    assert files, "no still ships with the project"
    for path in files:
        scenario = StillScenario.load(path)
        assert scenario.cameras
        assert scenario.width > 0 and scenario.height > 0


def test_the_thumbnail_is_sixteen_by_nine() -> None:
    """The card plate crops anything else."""
    scenario = StillScenario.load(STILLS / "thumbnail.yml")
    assert scenario.width * 9 == scenario.height * 16


def test_every_camera_sits_inside_the_stated_ranges() -> None:
    """The ranges are what keep the arm readable rather than a dark smudge.

    The distance bound tracks the line it frames rather than a fixed number.
    It was 0.8 to 1.8 m for a 1.20 m belt with a 0.25 m arm; the line is now
    3.00 m long with a UR10e reaching 1.25 m from a pedestal beside it, and a
    camera still inside the old bound would sit inside the machine.
    """
    scenario = StillScenario.load(STILLS / "thumbnail.yml")
    assert len(scenario.cameras) >= 3
    for camera in scenario.cameras:
        assert 2.2 <= camera.distance <= 4.0, camera.name
        assert -35.0 <= camera.elevation <= -10.0, camera.name
        assert 20.0 <= camera.fovy <= 60.0, camera.name
    # Three points of view rather than one frame rendered three times.
    assert len({camera.azimuth for camera in scenario.cameras}) == len(scenario.cameras)


def test_a_missing_key_fails_naming_itself(tmp_path: Path) -> None:
    """A still carries no default, as no configuration here does."""
    raw = yaml.safe_load((STILLS / "thumbnail.yml").read_text())
    del raw["still"]["seed"]
    path = tmp_path / "broken.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(StillError, match="still.seed"):
        StillScenario.load(path)


def test_the_lighting_never_reaches_the_shared_world() -> None:
    """The world every measurement reads must not carry presentation values.

    This is the constraint the whole feature is built around: a frame that
    shifts the benchmark's world is worth less than no frame.
    """
    world = yaml.safe_load(
        (ROOT / "configs" / "world" / "sorting_line.yml").read_text()
    )
    scenario = StillScenario.load(STILLS / "thumbnail.yml")
    assert scenario.lighting, "the still declares no lighting of its own"
    for key in ("headlight", "background", "lights"):
        assert key not in world, f"{key} leaked into the shared world"
    # The world's own lighting section is the one it always had.
    assert set(world["lighting"]) == {"diffuse"}


def test_a_world_built_without_a_still_is_unchanged() -> None:
    """The presentation argument defaults to adding nothing."""
    pytest.importorskip("mujoco")
    import mujoco

    from clave.world import config, scene

    raw = config.load(ROOT / "configs" / "world" / "sorting_line.yml")
    plain, _, _ = scene.build(raw, np.random.default_rng(0), ROOT)
    lit, _, _ = scene.build(
        raw,
        np.random.default_rng(0),
        ROOT,
        presentation=StillScenario.load(STILLS / "thumbnail.yml").lighting,
    )
    # The count is not asserted absolutely: the vendored arm ships its own
    # tracking light, so the shared world's total is the scene's plus whatever
    # the manipulator brings. What matters is that presentation lighting is
    # absent from one and present in the other.
    assert lit.nlight > plain.nlight, "the still added none"
    # Nothing but lighting differs: same bodies, same geoms, same degrees of
    # freedom, so no trajectory can depend on which one was built.
    assert (plain.nbody, plain.nq, plain.nv) == (lit.nbody, lit.nq, lit.nv)
    names = [
        mujoco.mj_id2name(plain, mujoco.mjtObj.mjOBJ_LIGHT, i)
        for i in range(plain.nlight)
    ]
    assert not any(n and n.startswith("presentation_") for n in names)


@pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg is not installed here",
)
def test_a_written_still_is_the_pixels_it_was_given(tmp_path: Path) -> None:
    """Nothing is drawn on a frame after the renderer made it.

    One flat gray frame in, one flat gray frame out. PNG is lossless, so an
    overlay, a caption or a watermark would show up exactly.
    """
    path = tmp_path / "flat.png"
    write_png(np.full((48, 64, 3), 137, dtype=np.uint8), path, 64, 48)
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    pixels = np.frombuffer(decoded, dtype=np.uint8)
    assert pixels.size == 48 * 64 * 3
    assert int(pixels.min()) == 137
    assert int(pixels.max()) == 137


def _rendering_available() -> bool:
    """Whether offscreen rendering works here.

    MuJoCo aborts the process when no GL backend is present, so the probe runs
    in a subprocess where an abort kills the child rather than the session.
    """
    import sys

    model = '<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>'
    probe = (
        "import os;os.environ.setdefault('MUJOCO_GL','osmesa');"
        "import mujoco;"
        f"m=mujoco.MjModel.from_xml_string({model!r});"
        "mujoco.Renderer(m,height=8,width=8)"
    )
    try:
        return (
            subprocess.run(
                [sys.executable, "-c", probe], capture_output=True, timeout=60
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def test_a_capture_writes_one_frame_per_camera(tmp_path: Path) -> None:
    """The command produces a file for every point of view it declares."""
    pytest.importorskip("mujoco")
    if not _rendering_available():
        pytest.skip("no offscreen GL backend here")
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        pytest.skip("ffmpeg is not installed here")

    from clave.demo.still import capture

    # A short, small version of the shipped still, because what this proves is
    # that the path works rather than what the frame looks like.
    raw = yaml.safe_load((STILLS / "thumbnail.yml").read_text())
    raw["still"]["capture_at_seconds"] = 1.5
    raw["still"]["width"] = 160
    raw["still"]["height"] = 90
    raw["still"]["expect"] = {"packages": 0, "classes": 0}
    path = tmp_path / "short.yml"
    path.write_text(yaml.safe_dump(raw))

    scenario = StillScenario.load(path)
    written = capture(ROOT, scenario, tmp_path)

    assert len(written) == len(scenario.cameras)
    for file in written:
        assert file.is_file() and file.stat().st_size > 0
        size = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(file),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert size == "160,90"
    # Three cameras, three different pictures.
    assert len({file.read_bytes() for file in written}) == len(written)


def test_a_still_naming_no_camera_is_refused(tmp_path: Path) -> None:
    """A still with no point of view renders nothing."""
    raw = yaml.safe_load((STILLS / "thumbnail.yml").read_text())
    raw["cameras"] = []
    path = tmp_path / "blind.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(StillError, match="no camera"):
        StillScenario.load(path)


def test_a_camera_needs_three_numbers_to_look_at(tmp_path: Path) -> None:
    """Two coordinates name no point in a scene."""
    raw = yaml.safe_load((STILLS / "thumbnail.yml").read_text())
    raw["cameras"][0]["lookat_meters"] = [0.0, 0.0]
    path = tmp_path / "flat.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(StillError, match="three numbers"):
        StillScenario.load(path)


def test_the_published_figures_are_named_as_the_page_names_them() -> None:
    """One scenario, one command, the three files the page points at."""
    scenario = StillScenario.load(STILLS / "clave.yml")
    written = [f"{scenario.name}-{camera.name}.png" for camera in scenario.cameras]
    assert written == ["clave-line.png", "clave-overhead.png", "clave-window.png"]
    assert scenario.width * 9 == scenario.height * 16


def test_the_figures_carry_annotations_and_the_thumbnail_does_not() -> None:
    """The annotations are a property of the figure, not of every still."""
    figures = StillScenario.load(STILLS / "clave.yml")
    assert figures.annotations
    assert set(figures.annotations) >= {
        "channel_colors",
        "reach_ring",
        "window_edges",
        "class_markers",
    }
    assert not StillScenario.load(STILLS / "thumbnail.yml").annotations


def test_no_annotation_reaches_the_shared_world() -> None:
    """The world every measurement reads must carry no explanatory geometry."""
    world = yaml.safe_load(
        (ROOT / "configs" / "world" / "sorting_line.yml").read_text()
    )
    assert "annotations" not in world


def test_the_figures_show_the_manipulator_the_repository_simulates() -> None:
    """Established by the model the scene loads, not by inspection."""
    pytest.importorskip("mujoco")
    import mujoco

    from clave.world import arm, config, scene

    raw = config.load(ROOT / "configs" / "world" / "sorting_line.yml")
    scenario = StillScenario.load(STILLS / "clave.yml")
    model, _, _ = scene.build(
        raw,
        np.random.default_rng(scenario.seed),
        ROOT,
        presentation=scenario.lighting,
        annotations=scenario.annotations,
    )
    for joint in arm.ARM_JOINTS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) >= 0, joint


def test_presentation_lighting_suppresses_scene_lights() -> None:
    """AC-STILL-02: only presentation lights stay active on a still build."""
    pytest.importorskip("mujoco")
    import mujoco

    from clave.world import config, scene

    raw = config.load(ROOT / "configs" / "world" / "sorting_line.yml")
    scenario = StillScenario.load(STILLS / "thumbnail.yml")
    model, _, _ = scene.build(
        raw,
        np.random.default_rng(0),
        ROOT,
        presentation=scenario.lighting,
    )
    active = []
    for index in range(model.nlight):
        if not model.light_active[index]:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, index)
        active.append(name)
    assert active
    assert all(name and name.startswith("presentation_") for name in active)
    assert model.vis.quality.shadowsize == 0
    assert model.vis.quality.offsamples == 8


def test_presentation_lights_may_be_directional() -> None:
    """AC-STILL-03: a still may declare directional fills."""
    pytest.importorskip("mujoco")
    import mujoco

    from clave.world import config, scene

    raw = config.load(ROOT / "configs" / "world" / "sorting_line.yml")
    scenario = StillScenario.load(STILLS / "thumbnail.yml")
    assert any(light.get("directional") for light in scenario.lighting["lights"])
    model, _, _ = scene.build(
        raw,
        np.random.default_rng(0),
        ROOT,
        presentation=scenario.lighting,
    )
    types = [
        int(model.light_type[index])
        for index in range(model.nlight)
        if model.light_active[index]
    ]
    assert int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL) in types


def test_every_shipped_still_matches_its_belt_claim() -> None:
    """AC-STILL-04: the capture instant matches still.expect."""
    pytest.importorskip("mujoco")

    from clave.demo.still import _simulate

    for path in sorted(STILLS.glob("*.yml")):
        scenario = StillScenario.load(path)
        _mujoco, _model, _data, _plan, conveyor, _indices = _simulate(ROOT, scenario)
        packages = len(conveyor.active)
        classes = len({item.material_class for item in conveyor.active})
        assert packages == scenario.expect.packages, path.name
        assert classes == scenario.expect.classes, path.name


def test_still_is_its_own_subcommand() -> None:
    """AC-STILL-01: stills are a subcommand, not a flag on sim."""
    from clave.cli import _build_parser

    parser = _build_parser()
    still = parser.parse_args(["still", "clave", "--out", "runs/stills"])
    assert still.command == "still"
    assert still.scenario == "clave"
    sim = parser.parse_args(["sim", "--seconds", "1"])
    assert sim.command == "sim"
    assert not hasattr(sim, "still")


def test_the_posed_arm_does_not_collide_with_the_belt() -> None:
    """A still's arm is scenery, so it cannot eject the packages it photographs."""
    pytest.importorskip("mujoco")
    import mujoco

    from clave.world import config, scene

    raw = config.load(ROOT / "configs" / "world" / "sorting_line.yml")
    model, _, _ = scene.build(raw, np.random.default_rng(0), ROOT)

    def _is_arm(geom: int) -> bool:
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])
        )
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        return (body is not None and body.startswith("arm_")) or (
            name is not None and name.startswith("arm_")
        )

    colliding = [geom for geom in range(model.ngeom) if _is_arm(geom)]
    assert colliding
    assert any(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) is None
        for geom in colliding
    )
    _let_the_arm_pass_through(mujoco, model)
    for geom in colliding:
        assert int(model.geom_contype[geom]) == 0
        assert int(model.geom_conaffinity[geom]) == 0


def test_the_arm_serves_the_belt_at_capture(tmp_path: Path) -> None:
    """AC-STILL-05: at capture the flange rides above a reachable package.

    `capture` itself refuses when the arm misses the belt, so a clean write
    is the acceptance check. The short path in the other capture test has no
    packages yet and is allowed to stay parked.
    """
    pytest.importorskip("mujoco")
    if not _rendering_available():
        pytest.skip("no offscreen GL backend here")
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        pytest.skip("ffmpeg is not installed here")

    from clave.demo.still import capture

    scenario = StillScenario.load(STILLS / "thumbnail.yml")
    assert scenario.expect.packages >= 1
    written = capture(ROOT, scenario, tmp_path)
    assert written
    assert all(path.is_file() for path in written)
