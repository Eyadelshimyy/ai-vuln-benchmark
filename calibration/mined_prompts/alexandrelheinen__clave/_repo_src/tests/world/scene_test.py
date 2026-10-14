"""Tests for scene assembly and the belt.

These need MuJoCo and the pinned menagerie submodule, and skip when either is
absent, matching how the candidate tests treat their optional libraries.
"""

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from clave.world import belt, config, scene
from clave.world.scene import SceneLayout, _two_narrowest_sides_fit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "world" / "sorting_line.yml"


def built(
    seed: int,
) -> tuple[dict[str, Any], np.random.Generator, Any, Any, SceneLayout]:
    """Build the shipped world at one seed."""

    raw = config.load(CONFIG)
    rng = np.random.default_rng(seed)
    model, data, plan = scene.build(raw, rng, ROOT)
    return raw, rng, model, data, plan


def conveyor_for(
    raw: dict[str, Any], rng: np.random.Generator, plan: SceneLayout
) -> belt.Conveyor:
    """Build a conveyor from configuration."""
    spawn = raw["spawn"]
    assert isinstance(spawn, dict)
    return belt.Conveyor(
        plan,
        rng,
        config.require_range(spawn, "spacing_meters", "spawn"),
        config.require_range(spawn, "lateral_offset_meters", "spawn"),
        config.require_range(spawn, "drop_height_meters", "spawn"),
        entry_margin=float(spawn["entry_margin_meters"]),
    )


def test_both_narrowest_mesh_sides_must_fit_the_gripper() -> None:
    """A single thin side must not make a broad flat object graspable."""
    assert _two_narrowest_sides_fit((0.020, 0.080, 0.180), 0.085)
    assert not _two_narrowest_sides_fit((0.020, 0.120, 0.180), 0.085)


def test_the_scene_builds_one_chute_per_channel() -> None:
    """AC-DROP-01: the scene builds one chute per channel."""
    pytest.importorskip("mujoco")
    import mujoco

    _, _, model, _, plan = built(0)
    for channel in plan.channels:
        mouth = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"chute_{channel}_mouth"
        )
        assert mouth >= 0
        # AC-DROP-11: and a take-away conveyor under its throat.
        assert (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"takeaway_{channel}")
            >= 0
        )
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "belt") >= 0
    # The manipulator arrived from the submodule under its attachment prefix.
    assert any("arm_" in (model.joint(i).name or "") for i in range(model.njnt))


def test_the_scene_exposes_its_timestep() -> None:
    """The scene exposes its timestep."""
    pytest.importorskip("mujoco")
    _, _, model, _, plan = built(0)
    assert plan.timestep > 0
    assert model.opt.timestep == pytest.approx(plan.timestep)


def test_the_scene_steps_without_instability() -> None:
    """Parked pool bodies used to fall forever into a NaN."""
    pytest.importorskip("mujoco")
    import mujoco

    raw, rng, model, data, plan = built(0)
    conveyor = conveyor_for(raw, rng, plan)
    for _ in range(4000):
        mujoco.mj_step(model, data)
        conveyor.step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))


def test_one_seed_twice_places_objects_identically() -> None:
    """One seed twice places objects identically."""
    pytest.importorskip("mujoco")
    _, _, _, _, first = built(5)
    _, _, _, _, second = built(5)
    assert first.belt.speed == second.belt.speed


def test_two_seeds_place_objects_differently() -> None:
    """The guard above would pass for a constant."""
    pytest.importorskip("mujoco")
    _, _, _, _, first = built(5)
    _, _, _, _, second = built(6)
    assert first.belt.speed != second.belt.speed


def test_the_belt_is_drawn_as_the_modules_the_configuration_names() -> None:
    """The module count and the belt length have to describe one belt."""
    pytest.importorskip("mujoco")
    import mujoco

    raw = config.load(CONFIG)
    model, _, plan = scene.build(raw, np.random.default_rng(0), ROOT)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in range(model.ngeom)
    ]
    drawn = [n for n in names if n and n.startswith("conveyor_module_")]
    expected = int(config.require(config.require(raw, "belt"), "modules", "belt"))
    if not (ROOT / scene.CONVEYOR_MODULE).is_file():
        assert not drawn
        return
    assert len(drawn) == expected
    # The modules are what a viewer sees; the box underneath keeps the physics.
    belt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "belt")
    assert model.geom_rgba[belt][3] == 0.0
    assert model.geom_contype[belt] != 0
    for name in drawn:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_contype[g] == 0, "a module must not collide"


def test_a_module_count_of_zero_is_refused() -> None:
    """Zero modules describes no belt."""
    pytest.importorskip("mujoco")
    import copy

    raw = copy.deepcopy(config.load(CONFIG))
    raw["belt"]["modules"] = 0
    with pytest.raises(config.WorldConfigError, match="positive"):
        scene.build(raw, np.random.default_rng(0), ROOT)


def test_the_arm_keeps_its_pedestal_whatever_draws_the_belt() -> None:
    """The arm stands on something, whatever draws the belt."""
    pytest.importorskip("mujoco")
    import mujoco

    raw = config.load(CONFIG)
    model, _, _ = scene.build(raw, np.random.default_rng(0), ROOT)
    for part in ("arm_pedestal", "arm_pedestal_foot"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, part) >= 0


def test_the_pedestal_reaches_the_floor() -> None:
    """An arm floating at working height describes no installation."""
    pytest.importorskip("mujoco")
    import mujoco

    raw = config.load(CONFIG)
    model, _, plan = scene.build(raw, np.random.default_rng(0), ROOT)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arm_pedestal")
    half_height = float(model.geom_size[geom][2])
    centre_z = float(model.geom_pos[geom][2])
    assert abs(centre_z - half_height) < 1e-9, (
        "the pedestal does not stand on the floor"
    )
    assert abs((centre_z + half_height) - plan.arm_base[2]) < 1e-9, (
        "the pedestal does not reach the arm it carries"
    )


def test_the_pedestal_stands_clear_of_the_arm_sweep() -> None:
    """A support the arm drives into is worse than none.

    The gantry this replaced was first placed at the belt edge, inside the
    arm's own annulus, where it stalled short of every target beyond. Here the
    pedestal sits under the base, so what keeps it clear is the inner radius.
    """
    pytest.importorskip("mujoco")
    import mujoco

    raw = config.load(CONFIG)
    model, _, plan = scene.build(raw, np.random.default_rng(0), ROOT)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arm_pedestal")
    width, depth = float(model.geom_size[geom][0]), float(model.geom_size[geom][1])
    half_diagonal = float(np.hypot(width, depth))
    assert half_diagonal < plan.reach_min, (
        f"the pedestal reaches {half_diagonal:.3f} m from the base, inside the "
        f"{plan.reach_min:.2f} m the arm sweeps"
    )
    assert plan.pedestal == (width * 2.0, depth * 2.0)


def test_the_arm_model_is_committed_rather_than_ignored() -> None:
    """A clone has to carry the arm, and CI proved that is not automatic.

    The model first landed under `assets/`, which `.gitignore` excludes because
    everything else there is generated by `scripts/import_scene_assets.py`. It
    ran locally and every test touching the world failed on a fresh checkout.
    This asserts the file git actually tracks, not the file on this disk.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(scene.ARM_MODEL)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked.returncode == 0, (
        f"{scene.ARM_MODEL} is not tracked by git, so a clone cannot build the "
        f"world: {tracked.stderr.strip()}"
    )


def test_the_arm_meshes_are_committed_rather_than_ignored() -> None:
    """A clone has to carry the arm's meshes, and CI proved that is not automatic.

    The model file was tracked while its `assets/` directory was not, because
    `.gitignore` carried an unanchored `assets/` rule meant for one generated
    directory at the repository root. It matched the vendored manipulator's own
    mesh directory too, and every test that builds a world failed on a fresh
    checkout while passing here. This asserts what git tracks, not what happens
    to be on this disk.
    """
    meshes = subprocess.run(
        ["git", "ls-files", str(scene.ARM_MODEL.parent / "assets")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tracked = [line for line in meshes.stdout.splitlines() if line.strip()]
    assert len(tracked) >= 20, (
        f"only {len(tracked)} arm meshes are tracked, so a clone cannot build the world"
    )


def annotated() -> tuple[Any, Any, SceneLayout, dict[str, Any]]:
    """Build the shipped world with the published figures' annotations."""
    import yaml

    scenario = yaml.safe_load((ROOT / "configs" / "stills" / "clave.yml").read_text())
    annotations = scenario["annotations"]
    raw = config.load(CONFIG)
    model, data, plan = scene.build(
        raw, np.random.default_rng(0), ROOT, annotations=annotations
    )
    return model, data, plan, annotations


def geom_named(model: Any, name: str) -> int:
    """The id of a geom, or -1 when the model has none by that name."""
    import mujoco

    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))


def test_annotations_add_geometry_and_no_freedom() -> None:
    """Annotations add geometry and no freedom.

    The figures are renders, so the explanation has to be geometry the renderer
    sees. What it must not be is a change to the world: a marker that added a
    body, a coordinate or a degree of freedom would put the published figure in
    a different simulation from the one every measurement runs in.
    """
    pytest.importorskip("mujoco")
    _, _, plain, _, _ = built(0)
    model, _, _, _ = annotated()
    assert model.ngeom > plain.ngeom
    assert (model.nbody, model.nq, model.nv) == (plain.nbody, plain.nq, plain.nv)
    assert geom_named(plain, "annotation_window_open") < 0
    assert geom_named(model, "annotation_window_open") >= 0


def test_the_ring_stands_at_the_radii_the_layout_carries() -> None:
    """A ring cannot outlive the workspace it describes."""
    pytest.importorskip("mujoco")
    model, _, plan, _ = annotated()
    for label, radius in (("inner", plan.reach_min), ("outer", plan.reach_max)):
        geom = geom_named(model, f"annotation_ring_{label}_0")
        assert geom >= 0, label
        position = model.geom_pos[geom]
        offset = np.hypot(
            position[0] - plan.arm_base[0], position[1] - plan.arm_base[1]
        )
        assert offset == pytest.approx(radius, abs=1e-6), label


def test_the_window_edges_stand_where_the_sweep_puts_them() -> None:
    """The figure and the safety layer read one measurement."""
    pytest.importorskip("mujoco")
    model, _, plan, _ = annotated()
    opens, closes = belt.reach_report(plan).window_edges
    assert model.geom_pos[geom_named(model, "annotation_window_open")][0] == (
        pytest.approx(opens)
    )
    assert model.geom_pos[geom_named(model, "annotation_window_close")][0] == (
        pytest.approx(closes)
    )


def test_a_marker_carries_its_channel_color_and_no_mass() -> None:
    """A marker carries its channel color and no mass.

    The color is read from the object's own channel rather than assigned per
    pool slot, so the legend cannot drift from the object set. The mass is what
    keeps an annotated object falling exactly as an unannotated one does.
    """
    pytest.importorskip("mujoco")
    import mujoco

    _, _, plain, _, plan = built(0)
    model, _, _, annotations = annotated()
    colors = annotations["channel_colors"]
    for index in range(plan.pool_size):
        geom = geom_named(model, f"object_{index}_marker")
        assert geom >= 0, index
        channel = plan.objects[index % len(plan.objects)].channel
        assert list(model.geom_rgba[geom][:3]) == pytest.approx(colors[channel])
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"object_{index}")
        assert model.body_mass[body] == pytest.approx(plain.body_mass[body])


def test_a_channel_with_no_color_is_refused() -> None:
    """A figure that draws one class in the default color teaches a wrong legend."""
    pytest.importorskip("mujoco")
    raw = config.load(CONFIG)
    with pytest.raises(Exception, match="CH-"):
        scene.build(
            raw,
            np.random.default_rng(0),
            ROOT,
            annotations={
                "channel_colors": {"CH-HDPE": [1.0, 0.0, 0.0]},
                "class_markers": {
                    "radius_meters": 0.03,
                    "height_above_meters": 0.16,
                },
            },
        )
