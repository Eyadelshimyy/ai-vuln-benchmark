"""Actual transport policy over synthetic ZIPs; no network or install."""
import hashlib
import io
from pathlib import Path
import runpy
import zipfile

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/ci/release_artifact_transport.py"))


def _zip(name="wheel.whl", data=b"synthetic"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, data)
    return buffer.getvalue()


def _selection(payload):
    return [{"id": 41, "name": "dist-wheel-leg", "digest": "sha256:" + hashlib.sha256(payload).hexdigest()}]


def test_actual_id_download_and_member_binding(tmp_path):
    payload = _zip()
    selected = _selection(payload)
    calls = []
    def fetch(repo, artifact_id, output):
        calls.append((repo, artifact_id))
        output.write(payload)
    receipt = MODULE["materialize"](selected, "owner/repo", tmp_path / "d", fetch)
    assert calls == [("owner/repo", 41)]
    MODULE["verify_materialized"](selected, receipt, tmp_path / "d")
    receipt[0]["id"] = 42
    with pytest.raises(ValueError, match="immutable IDs"):
        MODULE["verify_materialized"](selected, receipt, tmp_path / "d")
    receipt[0]["id"] = 41
    (tmp_path / "d/dist-wheel-leg/wheel.whl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="members changed"):
        MODULE["verify_materialized"](selected, receipt, tmp_path / "d")


@pytest.mark.parametrize("artifact_id", [None, True, 0, -1, "41"])
def test_invalid_id_never_downloads(tmp_path, artifact_id):
    selected = _selection(_zip())
    selected[0]["id"] = artifact_id
    with pytest.raises(ValueError, match="artifact ID"):
        MODULE["materialize"](selected, "owner/repo", tmp_path, lambda *_: pytest.fail("unexpected fetch"))


def test_digest_mismatch_stops_before_extract(tmp_path):
    with pytest.raises(ValueError, match="ZIP digest"):
        MODULE["materialize"](_selection(_zip()), "owner/repo", tmp_path, lambda repo, ident, output: output.write(_zip(data=b"other")))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["../escape", "/absolute", "dir/file", ".."])
def test_unsafe_zip_member_is_never_written(tmp_path, name):
    payload = _zip(name)
    with pytest.raises(ValueError, match="ZIP member"):
        MODULE["materialize"](_selection(payload), "owner/repo", tmp_path, lambda repo, ident, output: output.write(payload))
    assert list(tmp_path.iterdir()) == []


def test_download_failure_propagates(tmp_path):
    def fail(*_):
        raise RuntimeError("download failed")
    with pytest.raises(RuntimeError, match="download failed"):
        MODULE["materialize"](_selection(_zip()), "owner/repo", tmp_path, fail)


def test_no_full_member_or_file_read(tmp_path, monkeypatch):
    payload = _zip(data=b"chunked" * 100)
    def forbidden(*args, **kwargs):
        pytest.fail("whole-file read is forbidden")
    monkeypatch.setattr(zipfile.ZipFile, "read", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    root = tmp_path / "downloaded"
    selected = _selection(payload)
    receipt = MODULE["materialize"](selected, "owner/repo", root,
        lambda repo, ident, output: output.write(payload))
    MODULE["verify_materialized"](selected, receipt, root)
    assert receipt[0]["members"]["wheel.whl"] == hashlib.sha256(b"chunked" * 100).hexdigest()


@pytest.mark.parametrize("member", ["../escape", "/absolute", "pkg//file", "pkg/./file"])
def test_bundle_inventory_rejects_unsafe_member(tmp_path, member):
    wheel = tmp_path / "pkg-1-cp312-cp312-manylinux2014_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, b"unsafe")
    with pytest.raises(ValueError, match="unsafe or duplicate"):
        MODULE["bundle_inventory"](wheel, "x86_64-unknown-linux-gnu-py3.12", "a" * 40, "container:image")


def test_partial_fetch_never_materializes(tmp_path):
    def fail(repo, ident, output):
        output.write(b"partial")
        raise OSError("interrupted download")
    with pytest.raises(OSError, match="interrupted"):
        MODULE["materialize"](_selection(_zip()), "owner/repo", tmp_path, fail)
    assert list(tmp_path.iterdir()) == []


def test_partial_member_failure_cleans_private_staging(tmp_path, monkeypatch):
    payload = _zip()
    original = MODULE["copy_and_hash"]
    def fail(source, destination=None):
        if destination is not None:
            destination.write(b"partial")
            raise OSError("member read failed")
        return original(source)
    monkeypatch.setitem(MODULE["materialize"].__globals__, "copy_and_hash", fail)
    with pytest.raises(OSError, match="member read"):
        MODULE["materialize"](_selection(payload), "owner/repo", tmp_path,
            lambda repo, ident, output: output.write(payload))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", [None, "nonzero", "interrupt"])
def test_fetch_streams_and_reaps_child(monkeypatch, failure):
    import subprocess
    class Stream(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= MODULE["CHUNK_BYTES"]
            if failure == "interrupt":
                raise KeyboardInterrupt()
            return super().read(size)
    class Process:
        args = ["synthetic-gh"]
        stdout = Stream(b"small ZIP placeholder")
        killed = False
        waits = 0
        def wait(self):
            self.waits += 1
            return 7 if failure == "nonzero" else 0
        def poll(self):
            return None if failure == "interrupt" and not self.killed else 0
        def kill(self):
            self.killed = True
    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    output = io.BytesIO()
    if failure:
        expected = KeyboardInterrupt if failure == "interrupt" else subprocess.CalledProcessError
        with pytest.raises(expected):
            MODULE["fetch"]("owner/repo", 1, output)
    else:
        MODULE["fetch"]("owner/repo", 1, output)
        assert output.getvalue() == b"small ZIP placeholder"
    assert process.waits >= 1
    assert process.stdout.closed
    assert process.killed is (failure == "interrupt")
