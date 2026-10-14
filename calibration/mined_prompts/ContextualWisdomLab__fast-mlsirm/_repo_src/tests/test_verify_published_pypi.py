from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/ci/verify_published_pypi.py"
SPEC = importlib.util.spec_from_file_location("verify_published_pypi", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_public_release_must_match_all_admitted_files(tmp_path: Path) -> None:
    version = "0.11.5"
    names = [f"fast_mlsirm-{version}-cp{minor}-cp{minor}-platform{platform}.whl"
             for minor in (312, 313, 314) for platform in range(4)]
    names.append(f"fast_mlsirm-{version}.tar.gz")
    manifest = tmp_path / "admitted-manifest.tsv"
    manifest.write_text("".join(f"{'a' * 64}  {name}\n" for name in names))
    expected = verifier.admitted_files(manifest, version)
    release = {"info": {"version": version}, "urls": [
        {"filename": name, "digests": {"sha256": digest}, "yanked": False}
        for name, digest in expected.items()]}
    verifier.verify_published(release, expected, version)

    release["urls"][0]["digests"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="differ"):
        verifier.verify_published(release, expected, version)
    release["urls"][0]["digests"]["sha256"] = "a" * 64
    release["urls"].pop()
    with pytest.raises(ValueError, match="thirteen"):
        verifier.verify_published(release, expected, version)
