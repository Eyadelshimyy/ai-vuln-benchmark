"""Pinned object downloads."""

from io import BytesIO
from pathlib import Path

import pytest
import yaml

from clave.assets.fetch import AssetError, fetch_objects, load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "assets" / "objects.yml"
WORLD = ROOT / "configs" / "world" / "sorting_line.yml"


def test_the_manifest_covers_every_spawned_object_path() -> None:
    """AC-ASSET-01: every path the world and the barcode fixtures name is pinned."""
    declared = {item.path for item in load_manifest(MANIFEST)}
    world = WORLD.read_text()
    for line in world.splitlines():
        stripped = line.strip()
        if stripped.startswith("mesh:") or stripped.startswith("texture:"):
            path = stripped.split(":", 1)[1].strip()
            assert path in declared
    fixtures = (
        "third_party/ycb_sim/textures/008_pudding_box.png",
        "third_party/scanned_objects/models/JarroSil_Activated_Silicon/texture.png",
    )
    for path in fixtures:
        assert path in declared


def test_a_matching_file_is_not_downloaded_again(tmp_path: Path) -> None:
    """AC-ASSET-02: a file whose digest already matches is left alone."""
    payload = b"mesh-bytes"
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    manifest = tmp_path / "objects.yml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "sources": {
                    "scanned_objects": {
                        "repository": "example/objects",
                        "commit": "abc",
                    }
                },
                "files": [
                    {
                        "path": "third_party/scanned_objects/models/one/model.obj",
                        "source": "scanned_objects",
                        "sha256": digest,
                    }
                ],
            }
        )
    )
    destination = tmp_path / "third_party/scanned_objects/models/one/model.obj"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)

    def opener(url: str, timeout: int = 0) -> BytesIO:
        raise AssertionError(f"should not fetch {url}")

    downloaded, present = fetch_objects(tmp_path, manifest, opener=opener)
    assert (downloaded, present) == (0, 1)


def test_a_bad_download_is_refused(tmp_path: Path) -> None:
    """AC-ASSET-03: mismatched bytes are not written."""
    manifest = tmp_path / "objects.yml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "sources": {"ycb_sim": {"repository": "example/ycb", "commit": "abc"}},
                "files": [
                    {
                        "path": "third_party/ycb_sim/meshes/can.msh",
                        "source": "ycb_sim",
                        "sha256": "0" * 64,
                    }
                ],
            }
        )
    )

    class _Response:
        def read(self) -> bytes:
            return b"not-the-file"

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    destination = tmp_path / "third_party/ycb_sim/meshes/can.msh"
    with pytest.raises(AssetError, match="downloaded as"):
        fetch_objects(tmp_path, manifest, opener=lambda url, timeout=0: _Response())
    assert not destination.exists()
