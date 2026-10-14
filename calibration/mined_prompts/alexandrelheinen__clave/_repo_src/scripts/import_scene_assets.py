#!/usr/bin/env python3
"""Convert the scene assets MuJoCo cannot read directly.

Two sources, neither in a format MuJoCo loads:

* The MIT-0 AWS RoboMaker warehouse props are COLLADA, pinned as a submodule.
* The CC BY 4.0 Open-RMF conveyor module is glTF, published on Gazebo Fuel,
  which is not a git host. It is downloaded and checked against a digest
  recorded here, the same way a corpus is.

The output under `assets/` is generated rather than committed, for the same
reason a dataset is: it is derived from bytes that are already pinned, and
committing it would put a second copy in the history that no digest describes.
The world builds without any of it and says so.

Run it once::

    git submodule update --init --recursive
    python scripts/import_scene_assets.py
"""

from __future__ import annotations

import hashlib
import io
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _log_output(*values: object, file: object = None) -> None:
    """Write importer output through logging at its appropriate severity."""
    level = logging.WARNING if file is sys.stderr else logging.INFO
    LOGGER.log(level, " ".join(str(value) for value in values))

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "third_party" / "aws-robomaker-small-warehouse-world" / "models"
OUT = ROOT / "assets" / "warehouse"
CONVEYOR_OUT = ROOT / "assets" / "conveyor"

CONVEYOR_URL = (
    "https://fuel.gazebosim.org/1.0/Open-RMF/models/conveyor_block/4/"
    "conveyor_block.zip"
)
"""The pinned version of the module, not whatever Fuel serves as latest."""

CONVEYOR_SHA256 = "ab6f108ba144bc076b26166619f27ec1af479a9d33e875a8ef61d3e418460a54"
"""What those bytes hash to. A download that does not match is refused."""

CONVEYOR_MESH = "meshes/conveyor_block_visual.glb"
"""The visual mesh inside the archive."""

CENTIMETERS_TO_METERS = 0.01
"""The upstream meshes are authored in centimeters."""

MESHES = {
    "shelf.obj": "aws_robomaker_warehouse_ShelfD_01",
    "shelf_e.obj": "aws_robomaker_warehouse_ShelfE_01",
    "trash_can.obj": "aws_robomaker_warehouse_TrashCanC_01",
    "pallet_jack.obj": "aws_robomaker_warehouse_PalletJackB_01",
    "bucket.obj": "aws_robomaker_warehouse_Bucket_01",
}
"""What CLAVE dresses its scene with, and where each comes from upstream."""

TEXTURES = {
    "ground.png": (
        "aws_robomaker_warehouse_GroundB_01",
        "aws_robomaker_warehouse_GroundB_01.png",
    ),
    "wall.png": (
        "aws_robomaker_warehouse_WallB_01",
        "aws_robomaker_warehouse_WallB_01.png",
    ),
}
"""The two surfaces that carry a texture rather than a mesh."""


def convert(name: str, model: str) -> tuple[str, tuple[float, float, float]]:
    """Convert one visual mesh to OBJ in meters.

    Args:
        name: Output file name.
        model: Upstream model directory.

    Returns:
        The output name and the converted extents, for the manifest.

    Raises:
        FileNotFoundError: If the upstream mesh is absent.
    """
    import trimesh

    LOGGER.debug("loading warehouse mesh %s from %s", name, model)
    candidates = sorted((SOURCE / model / "meshes").glob("*visual*.DAE"))
    if not candidates:
        raise FileNotFoundError(f"{model} has no visual mesh")
    mesh = trimesh.load(candidates[0], force="mesh")
    mesh.apply_scale(CENTIMETERS_TO_METERS)
    # Origin at the footprint centre and the base on the floor, so a pose in the
    # world configuration names where the prop stands rather than where its
    # author happened to put the origin.
    mesh.apply_translation(
        [-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(), -mesh.bounds[0][2]]
    )
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    mesh.export(OUT / "meshes" / name)
    extents = tuple(round(float(value), 3) for value in mesh.extents)
    return name, (extents[0], extents[1], extents[2])


def import_conveyor() -> tuple[float, float, float] | None:
    """Fetch the Open-RMF conveyor module and convert it to OBJ.

    Gazebo Fuel is not a git host, so this is a download rather than a
    submodule, checked against a digest exactly as a corpus is. A mismatch is
    refused rather than used, because a scene asset that silently changed is a
    scene nobody can reproduce.

    Returns:
        The module's extents in meters, or None when the download failed.
    """
    import trimesh

    LOGGER.debug("loading conveyor archive from %s", CONVEYOR_URL)
    try:
        with urllib.request.urlopen(CONVEYOR_URL, timeout=120) as response:
            payload = response.read()
    except OSError as error:
        _log_output(f"  could not reach Gazebo Fuel: {error}", file=sys.stderr)
        return None

    digest = hashlib.sha256(payload).hexdigest()
    if digest != CONVEYOR_SHA256:
        _log_output(
            f"  the conveyor archive hashes to {digest}, and this build pins\n"
            f"  {CONVEYOR_SHA256}. Refusing bytes nobody described.",
            file=sys.stderr,
        )
        return None

    archive = zipfile.ZipFile(io.BytesIO(payload))
    mesh = trimesh.load(
        io.BytesIO(archive.read(CONVEYOR_MESH)), file_type="glb", force="mesh"
    )
    # Origin at the footprint centre with the belt surface at z = 0, so the
    # world places a module by naming where its surface goes rather than by
    # guessing where its author put the origin.
    mesh.apply_translation(
        [
            -mesh.bounds[:, 0].mean(),
            -mesh.bounds[:, 1].mean(),
            -mesh.bounds[1][2],
        ]
    )
    CONVEYOR_OUT.mkdir(parents=True, exist_ok=True)
    mesh.export(CONVEYOR_OUT / "module.obj")
    extents = tuple(round(float(value), 4) for value in mesh.extents)
    (CONVEYOR_OUT / "IMPORTED.md").write_text(
        "\n".join(
            [
                "# Imported Open-RMF conveyor module",
                "",
                f"Source: {CONVEYOR_URL}",
                f"sha256: `{CONVEYOR_SHA256}`",
                "License: CC BY 4.0, Open Robotics",
                "",
                f"One module measures {extents[0]:.3f} x {extents[1]:.3f} x "
                f"{extents[2]:.3f} m as published. The world scales it to the",
                "belt it configures; see the README.",
                "",
                "Not committed. Regenerate with scripts/import_scene_assets.py.",
            ]
        )
        + "\n"
    )
    return (extents[0], extents[1], extents[2])


def _fetch_objects() -> int:
    """Download the pinned object files. Returns a process exit code."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from clave.assets.fetch import fetch_objects

    manifest = ROOT / "configs" / "assets" / "objects.yml"
    downloaded, present = fetch_objects(ROOT, manifest)
    _log_output(
        f"  objects         {downloaded} downloaded, {present} already matched"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Convert warehouse meshes and fetch the pinned object files.

    Args:
        argv: Arguments, defaulting to the process arguments. `--objects`
            fetches only the pinned meshes. `--warehouse` converts only the
            warehouse props. With neither, both run.

    Returns:
        A process exit code.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Import scene assets")
    parser.add_argument(
        "--objects",
        action="store_true",
        help="download the pinned object meshes and textures",
    )
    parser.add_argument(
        "--warehouse",
        action="store_true",
        help="convert the warehouse props and the conveyor module",
    )
    args = parser.parse_args(argv)
    objects = args.objects or not args.warehouse
    warehouse = args.warehouse or not args.objects

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if objects and _fetch_objects() != 0:
        return 1
    if not warehouse:
        return 0
    LOGGER.debug("initializing asset importer: source=%s output=%s", SOURCE, OUT)
    if not SOURCE.is_dir():
        _log_output(
            "the warehouse submodule is not checked out. Run:\n"
            "  git submodule update --init "
            "third_party/aws-robomaker-small-warehouse-world",
            file=sys.stderr,
        )
        return 1
    try:
        import trimesh  # noqa: F401
    except ImportError:
        _log_output(
            'trimesh and pycollada are needed. Run: uv pip install "clave[assets]"',
            file=sys.stderr,
        )
        return 1

    rows = [convert(name, model) for name, model in sorted(MESHES.items())]
    (OUT / "textures").mkdir(parents=True, exist_ok=True)
    for name, (model, upstream) in sorted(TEXTURES.items()):
        source = SOURCE / model / "materials" / "textures" / upstream
        if not source.is_file():
            _log_output(f"  missing texture {source}", file=sys.stderr)
            continue
        shutil.copyfile(source, OUT / "textures" / name)

    manifest = [
        "# Imported AWS RoboMaker warehouse assets",
        "",
        "Generated by `scripts/import_warehouse_assets.py` from the submodule at",
        "`third_party/aws-robomaker-small-warehouse-world`, MIT-0.",
        "",
        "Nothing here is committed. The world builds without it and says so.",
        "",
        "| File | Upstream model | Extents (m) |",
        "| --- | --- | --- |",
    ]
    for name, extents in rows:
        model = MESHES[name]
        size = " x ".join(f"{value:.3f}" for value in extents)
        manifest.append(f"| `meshes/{name}` | `{model}` | {size} |")
    for name, (model, _) in sorted(TEXTURES.items()):
        manifest.append(f"| `textures/{name}` | `{model}` | texture |")
    (OUT / "IMPORTED.md").write_text("\n".join(manifest) + "\n")

    _log_output(f"  wrote {len(rows)} meshes and {len(TEXTURES)} textures to {OUT}")
    for name, extents in rows:
        _log_output(
            f"    {name:18s} {extents[0]:.3f} x {extents[1]:.3f} x "
            f"{extents[2]:.3f} m"
        )

    module = import_conveyor()
    if module is None:
        _log_output("  the conveyor module was not imported; the belt stays a box")
        return 0
    _log_output(f"  wrote the conveyor module to {CONVEYOR_OUT}")
    _log_output(
        f"    module.obj         {module[0]:.3f} x {module[1]:.3f} x "
        f"{module[2]:.3f} m"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - a one-off import step
    raise SystemExit(main())
