"""The release capture must include every declared Python extra."""

import hashlib
from pathlib import Path
import runpy

import pytest


def test_release_capture_exports_all_extras(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/ci"))
    capture = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/ci/capture_release_runtime.py"))
    source = tmp_path / "source"
    source.mkdir()
    wheel = tmp_path / "fixture.whl"
    wheel.write_bytes(b"wheel")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    row = tmp_path / "linux-py3.12.tsv"
    row.write_text(f"linux-py3.12\ttrue\tclean-target-repeat-same-env\t{sha}\t{sha}\t{wheel.name}\trunner:test\n")
    source_sha = "a" * 40
    exports = []

    class ExportReached(Exception):
        pass

    def fake_run(*args, cwd):
        if args[:2] == ("git", "rev-parse"):
            return source_sha + "\n"
        if args[:2] == ("uv", "--version"):
            return "uv 0.12.5\n"
        exports.append(args)
        raise ExportReached

    monkeypatch.setitem(capture["capture"].__globals__, "_run", fake_run)
    project = source / "pyproject.toml"
    project.write_text('[project]\nname = "fast-mlsirm"\n[project.optional-dependencies]\ndev = []\nfuzz = []\n')
    with pytest.raises(ExportReached):
        capture["capture"](row, tmp_path, source, tmp_path, source_sha)
    assert exports and "--all-extras" in exports[0]
