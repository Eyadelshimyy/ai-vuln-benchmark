"""Coverage for artifact persistence and bounded loaders (io.py)."""

from __future__ import annotations

import io as _io
import json
import os
import zipfile

import numpy as np
import pytest

import struct

import fast_mlsirm.io as io_mod
from fast_mlsirm.config import MLS2PLMConfig
from fast_mlsirm.io import (
    _arrays_to_lists,
    _atomic_write,
    _load_json_bounded,
    _load_numpy_bounded,
    _read_text_bounded,
    _validate_npy_header,
    _validate_numpy_file,
    _write_factor_csv,
    load_factor_csv,
    load_params,
    save_dimensionality_diagnostics,
    save_fit_diagnostics,
    save_fit_result,
    save_simulation,
)
from fast_mlsirm.simulation import simulate
from fast_mlsirm.types import (
    DimensionalityDiagnostics,
    FitDiagnostics,
    FitResult,
    MLSIRMParams,
)


def _params(n_persons=8, n_items=6, latent=2, seed=0):
    rng = np.random.default_rng(seed)
    return MLSIRMParams(
        theta=rng.standard_normal((n_persons, 1)),
        alpha=np.log(0.8 + 0.3 * rng.random(n_items)),
        b=np.linspace(-1.0, 1.0, n_items),
        xi=rng.standard_normal((n_persons, latent)),
        zeta=rng.standard_normal((n_items, latent)),
        tau=0.1,
    )


def _fit_result(with_population=True, with_ic=True, trace=(-10.0, -9.0)):
    pop = None
    if with_population:
        pop = {
            "kind": "multilevel",
            "mu": np.zeros(1),
            "sigma": np.ones(1),
            "sigma_u": 0.5,  # float summary key present
        }  # u_eap / theta_sd / icc intentionally absent to hit the missing-key arcs
    ic = {"aic": 1.0, "bic": 2.0, "n_parameters": 5} if with_ic else None
    return FitResult(
        params=_params(),
        model="MLS2PLM",
        optimizer="adam",
        backend="rust",
        rust_device="cpu",
        objective=1.23,
        loglik_trace=list(trace),
        objective_trace=list(trace),
        convergence_status="converged",
        n_iter=3,
        population=pop,
        ic=ic,
    )


def test_save_simulation_and_load_factor_csv_roundtrip(tmp_path):
    data = simulate(MLS2PLMConfig(n_persons=30, n_dims=2, items_per_dim=3, seed=1))
    save_simulation(data, tmp_path)
    assert (tmp_path / "manifest.json").exists()
    factors = load_factor_csv(tmp_path / "item_factor.csv")
    assert factors.shape == data.factor_id.shape
    np.testing.assert_array_equal(factors, data.factor_id)


def test_save_fit_result_with_population_and_load_params(tmp_path):
    save_fit_result(_fit_result(), tmp_path)
    summary = json.loads((tmp_path / "fit_summary.json").read_text())
    assert summary["population"]["kind"] == "multilevel"
    assert "information_criteria" in summary
    restored = load_params(tmp_path / "params.npz")
    assert isinstance(restored, MLSIRMParams)
    assert restored.theta.shape == (8, 1)


def test_save_fit_result_without_population_or_ic(tmp_path):
    save_fit_result(
        _fit_result(with_population=False, with_ic=False, trace=()), tmp_path
    )
    summary = json.loads((tmp_path / "fit_summary.json").read_text())
    assert summary["final_loglik"] is None
    assert "population" not in summary
    assert "information_criteria" not in summary


def test_save_fit_diagnostics_and_dimensionality(tmp_path):
    diagnostics = FitDiagnostics(
        itemfit={"outfit": np.array([1.0, 1.1])},
        personfit={"z": np.array([0.1, -0.2])},
        model_fit={"rmsea": 0.05},
        factorfit={"load": np.array([0.7])},
    )
    save_fit_diagnostics(diagnostics, tmp_path)
    payload = json.loads((tmp_path / "fit_diagnostics.json").read_text())
    assert payload["itemfit"]["outfit"] == [1.0, 1.1]

    dim = DimensionalityDiagnostics(
        candidates=[{"n_dims": 1, "bic": 10.0}], best={"n_dims": 1}
    )
    save_dimensionality_diagnostics(dim, tmp_path)
    assert (tmp_path / "dimension_diagnostics.json").exists()


def test_arrays_to_lists_helper():
    out = _arrays_to_lists({"a": np.array([1, 2, 3])})
    assert out == {"a": [1.0, 2.0, 3.0]}


def test_atomic_write_cleans_up_on_writer_failure(tmp_path):
    target = tmp_path / "dest.bin"

    def boom(stream):
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError):
        _atomic_write(target, boom)
    assert not target.exists()
    # no stray temp files left behind
    assert list(tmp_path.glob(".dest.bin.*")) == []


def test_write_factor_csv_roundtrip(tmp_path):
    factor_id = np.array([0, 1, 0, 1, 2])
    path = tmp_path / "f.csv"
    _write_factor_csv(path, factor_id)
    np.testing.assert_array_equal(load_factor_csv(path), factor_id)


# --------------------------- bounded text/json loaders ---------------------------

def test_read_text_bounded_limit_and_utf8(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("hello", encoding="utf-8")
    assert _read_text_bounded(good, source="t", max_bytes=100) == "hello"
    with pytest.raises(ValueError):
        _read_text_bounded(good, source="t", max_bytes=2)
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError):
        _read_text_bounded(bad, source="t", max_bytes=100)


def test_load_json_bounded_happy_and_depth(tmp_path):
    ok = tmp_path / "ok.json"
    ok.write_text('{"a": [1, 2, "x\\"y"], "b": true}', encoding="utf-8")
    assert _load_json_bounded(ok, source="j")["b"] is True
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 200 + "]" * 200, encoding="utf-8")
    with pytest.raises(ValueError):
        _load_json_bounded(deep, source="j")


@pytest.mark.parametrize(
    "payload",
    [
        '{"status":"ok","status":"failed"}',
        '{"outer":{"id":1,"id":2}}',
    ],
    ids=["root", "nested"],
)
def test_load_json_bounded_rejects_duplicate_object_members(tmp_path, payload):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="contains a duplicate JSON object member"):
        _load_json_bounded(duplicate, source="artifact JSON")


def test_load_json_bounded_byte_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(io_mod, "MAX_JSON_INPUT_BYTES", 3)
    big = tmp_path / "big.json"
    big.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        _load_json_bounded(big, source="j")


def test_load_factor_csv_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(io_mod, "MAX_FACTOR_CSV_BYTES", 4)
    path = tmp_path / "big.csv"
    path.write_text("item_id,factor_id\n0,0\n1,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_factor_csv(path)


def test_load_factor_csv_rejects_empty(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_factor_csv(path)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW unavailable")
@pytest.mark.parametrize("loader", [
    "json",
    "csv",
    "numpy",
    "validate_numpy",
    "params",
])
def test_bounded_loaders_reject_leaf_symlinks(tmp_path, loader):
    """Input readers never follow a symlink to an attacker-chosen file."""
    target = tmp_path / f"target-{loader}"
    if loader == "json":
        target.write_text('{"safe": true}', encoding="utf-8")
    elif loader == "csv":
        target.write_text("item_id,factor_id\n0,0\n", encoding="utf-8")
    elif loader in {"numpy", "validate_numpy"}:
        target = target.with_suffix(".npy")
        _save_npy(target, np.zeros(1))
    else:
        target = target.with_suffix(".npz")
        np.savez(
            target,
            theta=np.zeros((1, 1)),
            alpha=np.zeros(1),
            b=np.zeros(1),
            xi=np.zeros((1, 1)),
            zeta=np.zeros((1, 1)),
            tau=np.array(1.0),
        )
    link = tmp_path / f"link-{loader}"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="stable regular file"):
        if loader == "json":
            _load_json_bounded(link, source="test JSON")
        elif loader == "csv":
            load_factor_csv(link)
        elif loader == "numpy":
            _load_numpy_bounded(link)
        elif loader == "validate_numpy":
            _validate_numpy_file(link)
        else:
            load_params(link)


# --------------------------- numpy file validation ---------------------------

def _save_npy(path, arr):
    with path.open("wb") as fh:
        np.save(fh, arr)


def test_validate_numpy_accepts_valid_npy(tmp_path):
    path = tmp_path / "ok.npy"
    _save_npy(path, np.zeros(5))
    assert _validate_numpy_file(path) is None  # returns cleanly


def test_validate_numpy_accepts_zero_size_npy(tmp_path):
    path = tmp_path / "empty_arr.npy"
    _save_npy(path, np.zeros((0, 3)))
    assert _validate_numpy_file(path) is None


def test_validate_npy_header_rejects_negative_dimension():
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': (-1, 3), }"
    magic = b"\x93NUMPY\x01\x00"
    total = len(magic) + 2 + len(header) + 1
    header = header + " " * ((64 - total % 64) % 64) + "\n"
    hbytes = header.encode("latin1")
    stream = _io.BytesIO(magic + struct.pack("<H", len(hbytes)) + hbytes)
    with pytest.raises(ValueError):
        _validate_npy_header(stream, "neg")


def test_validate_numpy_rejects_wrong_suffix(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"nope")
    with pytest.raises(ValueError):
        _validate_numpy_file(path)


def test_validate_numpy_rejects_object_dtype(tmp_path):
    path = tmp_path / "obj.npy"
    with path.open("wb") as fh:
        np.save(fh, np.array([{"k": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        _validate_numpy_file(path)


def test_validate_numpy_rejects_oversized_declaration(tmp_path, monkeypatch):
    monkeypatch.setattr(io_mod, "MAX_NUMPY_ARRAY_ELEMENTS", 3)
    path = tmp_path / "big.npy"
    _save_npy(path, np.zeros(10))
    with pytest.raises(ValueError):
        _validate_numpy_file(path)


def test_validate_numpy_rejects_oversized_bytes(tmp_path, monkeypatch):
    # element count stays under its cap but the declared byte size exceeds the
    # byte cap -> the post-loop nbytes guard fires
    monkeypatch.setattr(io_mod, "MAX_NUMPY_ARRAY_BYTES", 8)
    path = tmp_path / "bytes.npy"
    _save_npy(path, np.zeros(10))
    with pytest.raises(ValueError):
        _validate_numpy_file(path)


def test_validate_numpy_rejects_file_too_large(tmp_path, monkeypatch):
    monkeypatch.setattr(io_mod, "MAX_NUMPY_ARCHIVE_BYTES", 8)
    path = tmp_path / "f.npy"
    _save_npy(path, np.zeros(100))
    with pytest.raises(ValueError):
        _validate_numpy_file(path)


def test_validate_numpy_rejects_truncated_npy(tmp_path):
    path = tmp_path / "trunc.npy"
    _save_npy(path, np.zeros(100))
    with path.open("r+b") as fh:
        fh.seek(0, _io.SEEK_END)
        size = fh.tell()
        fh.truncate(size - 400)  # drop half the declared data
    with pytest.raises(ValueError):
        _validate_numpy_file(path)


def test_validate_npy_header_version_paths():
    # version 2.0 header is accepted
    buf = _io.BytesIO()
    np.lib.format.write_array(buf, np.zeros(4), version=(2, 0))
    buf.seek(0)
    nbytes, _ = _validate_npy_header(buf, "v2")
    assert nbytes == 32
    # an unsupported major version is rejected
    with pytest.raises(ValueError):
        _validate_npy_header(_io.BytesIO(b"\x93NUMPY\x03\x00"), "v3")


def _npy_bytes(arr):
    b = _io.BytesIO()
    np.save(b, arr)
    return b.getvalue()


def test_validate_npz_member_checks(tmp_path):
    # empty archive
    empty = tmp_path / "empty.npz"
    with zipfile.ZipFile(empty, "w") as zf:
        pass
    with pytest.raises(ValueError):
        _validate_numpy_file(empty)

    # non-npy member
    nonnpy = tmp_path / "nonnpy.npz"
    with zipfile.ZipFile(nonnpy, "w") as zf:
        zf.writestr("note.txt", b"hello")
    with pytest.raises(ValueError):
        _validate_numpy_file(nonnpy)

    # truncated member (header claims more than stored)
    trunc = tmp_path / "truncmember.npz"
    full = _npy_bytes(np.zeros(100))
    with zipfile.ZipFile(trunc, "w") as zf:
        zf.writestr("arr.npy", full[:-400])
    with pytest.raises(ValueError):
        _validate_numpy_file(trunc)


def test_validate_npz_member_count_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(io_mod, "MAX_NUMPY_ARCHIVE_MEMBERS", 1)
    path = tmp_path / "multi.npz"
    np.savez(path, a=np.zeros(3), b=np.ones(3))
    with pytest.raises(ValueError):
        _validate_numpy_file(path)


def test_validate_npz_total_bytes_limit(tmp_path, monkeypatch):
    # compressed file stays under the archive-byte cap while the declared
    # (uncompressed) member bytes exceed it, exercising the running-total guard
    monkeypatch.setattr(io_mod, "MAX_NUMPY_ARCHIVE_BYTES", 5000)
    path = tmp_path / "compressed.npz"
    np.savez_compressed(path, a=np.zeros(400), b=np.zeros(400))
    assert path.stat().st_size < 5000
    with pytest.raises(ValueError):
        _validate_numpy_file(path)


def test_validate_npz_member_too_large(tmp_path, monkeypatch):
    # threshold is MAX_NUMPY_ARRAY_BYTES + MAX_NUMPY_HEADER_BYTES; a ~72 KB member
    # exceeds it once the array cap is dropped, hitting the per-member size guard
    # before the header is even parsed
    monkeypatch.setattr(io_mod, "MAX_NUMPY_ARRAY_BYTES", 8)
    path = tmp_path / "member.npz"
    np.savez(path, a=np.zeros(9000))
    with pytest.raises(ValueError):
        _validate_numpy_file(path)
