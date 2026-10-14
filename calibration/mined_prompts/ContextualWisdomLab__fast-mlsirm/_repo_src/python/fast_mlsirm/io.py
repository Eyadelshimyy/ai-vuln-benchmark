from __future__ import annotations

import io
import json
import os
import stat
import tempfile
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np

from .types import DimensionalityDiagnostics, FitDiagnostics, FitResult, MLSIRMParams, SimulationData


MAX_NUMPY_ARRAY_ELEMENTS = 50_000_000
MAX_NUMPY_ARRAY_BYTES = 512 * 1024 * 1024
MAX_NUMPY_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_NUMPY_ARCHIVE_MEMBERS = 256
MAX_NUMPY_HEADER_BYTES = 64 * 1024
MAX_JSON_INPUT_BYTES = 32 * 1024 * 1024
MAX_JSON_NESTING_DEPTH = 128
MAX_FACTOR_CSV_BYTES = 16 * 1024 * 1024


def _input_open_flags() -> int:
    """Open input without following a leaf symlink where supported."""
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _validate_input_identity(
    path: Path,
    descriptor_status: os.stat_result,
    *,
    source: str,
) -> None:
    """Require the requested path to still name the opened regular file."""
    try:
        path_status = os.lstat(path)
    except OSError:
        raise ValueError(f"{source} path changed during the read") from None
    if not stat.S_ISREG(path_status.st_mode):
        raise ValueError(f"{source} must be a stable regular file")
    if (path_status.st_dev, path_status.st_ino) != (
        descriptor_status.st_dev,
        descriptor_status.st_ino,
    ):
        raise ValueError(f"{source} path changed during the read")


def _open_stable_input(
    path: str | Path,
    *,
    source: str,
) -> tuple[Path, BinaryIO, os.stat_result]:
    """Open one regular input descriptor without following its leaf symlink."""
    input_path = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        try:
            if input_path.is_symlink():
                raise ValueError(f"{source} must be a stable regular file")
        except OSError:
            raise ValueError(f"{source} could not be opened safely") from None
    try:
        file_descriptor = os.open(input_path, _input_open_flags())
    except FileNotFoundError:
        raise
    except OSError:
        raise ValueError(f"{source} must be a stable regular file") from None
    try:
        descriptor_status = os.fstat(file_descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise ValueError(f"{source} must be a stable regular file")
        _validate_input_identity(input_path, descriptor_status, source=source)
        stream = os.fdopen(file_descriptor, "rb")
    except BaseException:
        os.close(file_descriptor)
        raise
    return input_path, stream, descriptor_status


def _atomic_write(path: str | Path, writer: Callable[[BinaryIO], object]) -> None:
    """Write a child artifact without following a pre-existing leaf symlink."""
    destination = Path(path)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_text(path: str | Path, content: str) -> None:
    """Atomically write UTF-8 ``content`` to ``path`` (temp file + rename)."""
    _atomic_write(path, lambda stream: stream.write(content.encode("utf-8")))


def _json_dumps_strict(payload: object, *, indent: int = 2) -> str:
    """Serialize package-owned JSON without non-finite numeric extensions."""
    try:
        return json.dumps(payload, indent=indent, allow_nan=False)
    except ValueError as exc:
        if str(exc).startswith("Out of range float values"):
            raise ValueError("artifact contains a non-finite JSON numeric value") from None
        raise ValueError("artifact could not be serialized as strict JSON") from None


def _validate_npy_header(stream: BinaryIO, source: str) -> tuple[int, int]:
    """Read only an NPY header and reject unsafe declared allocations."""
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(
            stream, max_header_size=MAX_NUMPY_HEADER_BYTES
        )
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(
            stream, max_header_size=MAX_NUMPY_HEADER_BYTES
        )
    else:
        raise ValueError(f"{source} uses unsupported NPY format version {version}")
    if dtype.hasobject:
        raise ValueError(f"{source} contains an object dtype")

    elements = 1
    for dim in shape:
        if dim < 0:
            raise ValueError(f"{source} declares a negative array dimension")
        if dim == 0:
            elements = 0
        elif elements and elements > MAX_NUMPY_ARRAY_ELEMENTS // dim:
            raise ValueError(
                f"{source} declares more than {MAX_NUMPY_ARRAY_ELEMENTS} array elements"
            )
        else:
            elements *= dim
    nbytes = elements * int(dtype.itemsize)
    if elements > MAX_NUMPY_ARRAY_ELEMENTS or nbytes > MAX_NUMPY_ARRAY_BYTES:
        raise ValueError(
            f"{source} declares {elements} elements / {nbytes} bytes, above the safe limit"
        )
    return nbytes, stream.tell()


def _validate_numpy_file(path: Path) -> None:
    """Validate a ``.npy``/``.npz`` file before loading it.

    Enforces file-size, member-count, and declared-array element/byte caps and
    rejects object dtypes or truncated arrays, guarding against
    memory-exhaustion and unpickling attacks from untrusted numpy inputs.
    """
    input_path, stream, descriptor_status = _open_stable_input(
        path,
        source="NumPy input",
    )
    try:
        _validate_numpy_stream(
            stream,
            source=input_path.name,
            suffix=input_path.suffix.lower(),
            file_size=descriptor_status.st_size,
        )
    finally:
        stream.close()


def _validate_numpy_stream(
    stream: BinaryIO,
    *,
    source: str,
    suffix: str,
    file_size: int,
) -> None:
    """Validate one already-open NPY/NPZ stream and its declared arrays."""
    if file_size > MAX_NUMPY_ARCHIVE_BYTES:
        raise ValueError(
            f"NumPy input exceeds the {MAX_NUMPY_ARCHIVE_BYTES}-byte file limit"
        )
    stream.seek(0)
    if suffix == ".npy":
        nbytes, header_end = _validate_npy_header(stream, source)
        if file_size - header_end < nbytes:
            raise ValueError(
                f"{source} is truncated relative to its declared array shape"
            )
        return
    if suffix != ".npz":
        raise ValueError("NumPy input must use a .npy or .npz suffix")

    with zipfile.ZipFile(stream) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if not members or len(members) > MAX_NUMPY_ARCHIVE_MEMBERS:
            raise ValueError(
                f"NPZ archive must contain 1..{MAX_NUMPY_ARCHIVE_MEMBERS} members"
            )
        total_bytes = 0
        for info in members:
            if not info.filename.endswith(".npy"):
                raise ValueError(
                    f"NPZ archive member {info.filename!r} is not an NPY array"
                )
            if info.file_size > MAX_NUMPY_ARRAY_BYTES + MAX_NUMPY_HEADER_BYTES:
                raise ValueError(
                    f"NPZ member {info.filename!r} exceeds the safe byte limit"
                )
            with archive.open(info) as stream:
                nbytes, header_end = _validate_npy_header(stream, info.filename)
            if info.file_size - header_end < nbytes:
                raise ValueError(
                    f"NPZ member {info.filename!r} is truncated relative to its declared array shape"
                )
            total_bytes += nbytes
            if total_bytes > MAX_NUMPY_ARCHIVE_BYTES:
                raise ValueError(
                    "NPZ archive declares more array bytes than the safe limit"
                )


def _read_text_bounded(
    path: str | Path,
    *,
    source: str,
    max_bytes: int,
) -> str:
    """Read UTF-8 text without permitting an unbounded in-memory payload."""
    input_path, stream, descriptor_status = _open_stable_input(
        path,
        source=source,
    )
    try:
        payload = stream.read(max_bytes + 1)
        _validate_input_identity(input_path, descriptor_status, source=source)
    finally:
        stream.close()
    if len(payload) > max_bytes:
        raise ValueError(f"{source} exceeds the {max_bytes}-byte input limit")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} must be valid UTF-8") from exc


def _load_json_bounded(
    path: str | Path,
    *,
    source: str,
    parse_constant=None,
):
    """Deserialize JSON only after enforcing bounded, unambiguous input."""
    content = _read_text_bounded(
        path,
        source=source,
        max_bytes=MAX_JSON_INPUT_BYTES,
    )
    depth = 0
    in_string = False
    escaped = False
    for char in content:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_NESTING_DEPTH:
                raise ValueError(
                    f"{source} exceeds the maximum JSON nesting depth "
                    f"of {MAX_JSON_NESTING_DEPTH}"
                )
        elif char in "]}":
            depth -= 1

    def reject_duplicate_members(pairs):
        """Build one JSON object while rejecting repeated member names."""
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source} contains a duplicate JSON object member")
            result[key] = value
        return result

    def reject_nonfinite_constant(_literal: str):
        """Reject Python JSON decoder extensions outside interoperable JSON."""
        raise ValueError(f"{source} contains a non-finite JSON numeric value")

    kwargs = {
        "parse_constant": (
            reject_nonfinite_constant if parse_constant is None else parse_constant
        ),
        "object_pairs_hook": reject_duplicate_members,
    }
    return json.loads(content, **kwargs)


def _load_numpy_bounded(path: str | Path):
    """Load NPY/NPZ only after validating headers and allocation bounds."""
    source, stream, descriptor_status = _open_stable_input(
        path,
        source="NumPy input",
    )
    loaded = None
    owns_stream = False
    try:
        _validate_numpy_stream(
            stream,
            source=source.name,
            suffix=source.suffix.lower(),
            file_size=descriptor_status.st_size,
        )
        stream.seek(0)
        loaded = np.load(
            stream,
            allow_pickle=False,
            max_header_size=MAX_NUMPY_HEADER_BYTES,
        )
        _validate_input_identity(source, descriptor_status, source="NumPy input")
        if isinstance(loaded, np.lib.npyio.NpzFile):
            # np.load(file_object) does not own the stream; transfer ownership
            # so the existing NpzFile context-manager contract still closes it.
            loaded.fid = stream
            owns_stream = True
        return loaded
    except BaseException:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()
        raise
    finally:
        if not owns_stream:
            stream.close()


def save_simulation(data: SimulationData, run_dir: str | Path) -> None:
    """Persist a simulation to ``run_dir``.

    Writes the config, responses, ground-truth parameters, item-to-factor CSV,
    and a manifest describing the run.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out / "config.json", _json_dumps_strict(asdict(data.config)))
    _atomic_write(out / "responses.npy", lambda stream: np.save(stream, data.Y))
    _atomic_write(
        out / "truth.npz",
        lambda stream: np.savez(
            stream,
            theta=data.truth.theta,
            alpha=data.truth.alpha,
            a=data.truth.a,
            b=data.truth.b,
            xi=data.truth.xi,
            zeta=data.truth.zeta,
            tau=np.array(data.truth.tau),
            gamma=np.array(data.truth.gamma),
            factor_id=data.factor_id,
            Phi=data.Phi,
        ),
    )
    _write_factor_csv(out / "item_factor.csv", data.factor_id)
    manifest = {
        "package": "fast-mlsirm",
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "MLS2PLM",
        "n_persons": int(data.Y.shape[0]),
        "n_items": int(data.Y.shape[1]),
        "n_dims": int(data.config.n_dims),
        "latent_dim": int(data.config.latent_dim),
        "gamma": float(data.config.gamma),
        "phi": float(data.config.phi),
        "seed": int(data.config.seed),
        "files": {"responses": "responses.npy", "truth": "truth.npz", "factors": "item_factor.csv"},
    }
    _atomic_write_text(out / "manifest.json", _json_dumps_strict(manifest))


def save_fit_result(result: FitResult, run_dir: str | Path) -> None:
    """Persist a fit to ``run_dir``.

    Writes the estimated parameter arrays to ``params.npz`` and a JSON summary
    (optimizer/backend metadata, objective, convergence, information criteria,
    and any marginal population structure) to ``fit_summary.json``.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = result.params
    arrays = dict(theta=p.theta, alpha=p.alpha, a=p.a, b=p.b, xi=p.xi, zeta=p.zeta, tau=p.tau, gamma=p.gamma)
    summary = {
        "model": result.model,
        "optimizer": result.optimizer,
        "backend": result.backend,
        "rust_device": result.rust_device,
        "objective": result.objective,
        "convergence_status": result.convergence_status,
        "n_iter": result.n_iter,
        "final_loglik": result.loglik_trace[-1] if result.loglik_trace else None,
    }
    if result.ic is not None:
        summary["information_criteria"] = {
            key: (float(v) if isinstance(v, float) else v)
            for key, v in result.ic.items()
        }
    if result.population is not None:
        pop = result.population
        summary["population"] = {"kind": pop["kind"]}
        for key in ("mu", "sigma", "u_eap", "theta_sd"):
            if key in pop:
                arrays[f"pop_{key}"] = np.asarray(pop[key])
        for key in ("sigma_u", "icc"):
            if key in pop:
                summary["population"][key] = float(pop[key])
    _atomic_write(out / "params.npz", lambda stream: np.savez(stream, **arrays))
    _atomic_write_text(out / "fit_summary.json", _json_dumps_strict(summary))


def save_fit_diagnostics(diagnostics: FitDiagnostics, run_dir: str | Path) -> None:
    """Write fit-diagnostic tables to ``fit_diagnostics.json`` in ``run_dir``."""
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "itemfit": _arrays_to_lists(diagnostics.itemfit),
        "personfit": _arrays_to_lists(diagnostics.personfit),
        "factorfit": _arrays_to_lists(diagnostics.factorfit or {}),
        "categoryfit": _arrays_to_lists(diagnostics.categoryfit or {}),
        "groupfit": _arrays_to_lists(diagnostics.groupfit or {}),
        "clusterfit": _arrays_to_lists(diagnostics.clusterfit or {}),
        "group_itemfit": _arrays_to_lists(diagnostics.group_itemfit or {}),
        "cluster_itemfit": _arrays_to_lists(diagnostics.cluster_itemfit or {}),
        "model_fit": diagnostics.model_fit,
    }
    _atomic_write_text(out / "fit_diagnostics.json", _json_dumps_strict(payload))


def save_dimensionality_diagnostics(diagnostics: DimensionalityDiagnostics, run_dir: str | Path) -> None:
    """Write the dimensionality-search candidates/best to JSON in ``run_dir``."""
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"candidates": diagnostics.candidates, "best": diagnostics.best}
    _atomic_write_text(out / "dimension_diagnostics.json", _json_dumps_strict(payload))


def load_params(path: str | Path) -> MLSIRMParams:
    """Load an :class:`MLSIRMParams` from a bounded ``params.npz`` file."""
    with _load_numpy_bounded(path) as data:
        return MLSIRMParams(theta=data["theta"], alpha=data["alpha"], b=data["b"], xi=data["xi"], zeta=data["zeta"], tau=float(data["tau"]))


def load_factor_csv(path: str | Path) -> np.ndarray:
    """Load the ``factor_id`` column from a bounded item-factor CSV file."""
    import warnings

    content = _read_text_bounded(
        path,
        source="factor CSV",
        max_bytes=MAX_FACTOR_CSV_BYTES,
    )
    if not content or content.isspace():
        raise ValueError("factor CSV is empty")
    # NumPy's loadtxt tokenizer mishandles supplementary-plane characters:
    # some segfault the process and others are silently converted to garbage
    # integers, so they must never reach it.
    if max(content) > "\uffff":
        raise ValueError(
            "factor CSV contains characters outside the Basic Multilingual Plane"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.loadtxt(
            io.StringIO(content),
            delimiter=",",
            skiprows=1,
            usecols=1,
            dtype=np.int64,
            ndmin=1,
        )


def _write_factor_csv(path: Path, factor_id: np.ndarray) -> None:
    """Atomically write an ``item_id,factor_id`` CSV for the given assignment."""
    item_ids = np.arange(len(factor_id))
    data = np.column_stack((item_ids, factor_id))

    def write_csv(stream: BinaryIO) -> None:
        """Serialize the item/factor table as CSV to ``stream``."""
        text_stream = io.TextIOWrapper(stream, encoding="utf-8", newline="")
        try:
            np.savetxt(
                text_stream,
                data,
                delimiter=",",
                header="item_id,factor_id",
                comments="",
                fmt="%d",
            )
            text_stream.flush()
        finally:
            text_stream.detach()

    _atomic_write(path, write_csv)


def _arrays_to_lists(values: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """Convert a dict of numeric arrays to JSON-serializable nested lists."""
    return {key: np.asarray(value, dtype=float).tolist() for key, value in values.items()}
