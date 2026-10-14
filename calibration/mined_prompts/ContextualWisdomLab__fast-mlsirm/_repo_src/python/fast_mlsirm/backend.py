from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType


VALID_PRODUCTION_BACKENDS = {"rust", "auto"}
VALID_REFERENCE_BACKENDS = {"numpy"}
VALID_KERNEL_BACKENDS = VALID_PRODUCTION_BACKENDS | VALID_REFERENCE_BACKENDS
VALID_BACKENDS = VALID_PRODUCTION_BACKENDS
# Execution device for the Rust backend. This is a sub-option of the ``rust``
# backend (CPU vs. wgpu GPGPU), not a separate compute-backend axis.
# The production backend axis is {rust, auto}; NumPy is retained only for
# low-level parity kernels and the explicit fit_reference API.
VALID_DEVICES = {"cpu", "gpu", "auto"}
CORE_MODULE = "fast_mlsirm._core"
# Stable, non-reflective auto-resolution error. Do not interpolate paths,
# ABI details, environment data, or import exception text into this text.
AUTO_BACKEND_UNAVAILABLE_MESSAGE = (
    "compiled Rust core is required for automatic backend resolution; "
    "install a wheel or editable build that provides fast_mlsirm._core, "
    "or use fast_mlsirm.fit_reference (backend='numpy') for the explicit "
    "reference/parity path"
)
# Stable, non-reflective explicit-rust-request error. Keeps the missing
# artifact name for operator diagnostics and adds the customer next action.
RUST_BACKEND_UNAVAILABLE_MESSAGE = (
    "Rust backend requested but fast_mlsirm._core is unavailable; install a "
    "wheel or editable build of this package that provides the compiled Rust "
    "extension, or rerun through the explicit NumPy reference path "
    "(fast_mlsirm.fit_reference, or --reference on the CLI)"
)
_REFERENCE_BACKEND_ACTIVE: ContextVar[bool] = ContextVar(
    "fast_mlsirm_reference_backend_active",
    default=False,
)


def _normalize_builtin_text(value: object, *, field: str) -> str:
    """Normalize an exact built-in string without dispatching caller callbacks."""
    if type(value) is not str:
        raise ValueError(f"{field} must be a built-in string")
    return value.strip().lower()


@contextmanager
def _reference_backend_scope() -> Iterator[None]:
    """Temporarily authorize NumPy resolution for the named reference API only."""
    token = _REFERENCE_BACKEND_ACTIVE.set(True)
    try:
        yield
    finally:
        _REFERENCE_BACKEND_ACTIVE.reset(token)


def normalize_backend(name: str) -> str:
    """Normalize an internal objective backend.

    ``numpy`` remains here only for low-level parity kernels. Public fitting
    configuration must use :func:`normalize_production_backend`.
    """
    backend = _normalize_builtin_text(name, field="backend")
    if backend not in VALID_KERNEL_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(VALID_KERNEL_BACKENDS)}")
    return backend


def normalize_production_backend(name: str) -> str:
    """Normalize a backend accepted by the production fitting contract."""
    backend = normalize_backend(name)
    if backend not in VALID_PRODUCTION_BACKENDS:
        raise ValueError(
            "production backend must be one of "
            f"{sorted(VALID_PRODUCTION_BACKENDS)}; use fit_reference for parity"
        )
    return backend


def normalize_device(name: str) -> str:
    """Lower-case and validate a Rust-backend device against ``{cpu, gpu, auto}``."""
    device = _normalize_builtin_text(name, field="rust_device")
    if device not in VALID_DEVICES:
        raise ValueError(f"rust_device must be one of {sorted(VALID_DEVICES)}")
    return device


def resolve_backend(name: str) -> str:
    """Resolve a requested numerical backend to the concrete backend that runs.

    ``rust`` requires the compiled :mod:`fast_mlsirm._core`
    extension. ``auto`` is the production convenience choice: it resolves to
    ``rust`` when the compiled core is available and fails closed otherwise.
    Automatic resolution never silently changes the numerical owner to NumPy.
    """
    backend = normalize_production_backend(name)
    core = _load_core()
    if backend == "rust":
        if core is None:
            raise RuntimeError(
                RUST_BACKEND_UNAVAILABLE_MESSAGE
            )
        return "rust"
    if core is None:
        raise RuntimeError(AUTO_BACKEND_UNAVAILABLE_MESSAGE)
    return "rust"


def resolve_reference_backend(name: str = "numpy") -> str:
    """Resolve an explicitly requested NumPy parity kernel.

    This low-level resolver intentionally has no fitting-scope requirement so
    callers can compare the Rust kernel with the independent NumPy kernel.
    High-level reference fitting uses :func:`_resolve_scoped_reference_backend`
    to keep NumPy out of the production fitting contract.
    """
    backend = normalize_backend(name)
    if backend not in VALID_REFERENCE_BACKENDS:
        raise ValueError("reference backend must be 'numpy'")
    return backend


def _resolve_scoped_reference_backend(name: str = "numpy") -> str:
    """Resolve NumPy only while the named reference fitting API is active."""
    backend = resolve_reference_backend(name)
    if not _REFERENCE_BACKEND_ACTIVE.get():
        raise RuntimeError(
            "NumPy fitting is available only through fast_mlsirm.fit_reference"
        )
    return backend


def load_rust_core() -> ModuleType:
    """Import and return the compiled Rust core, raising if it is unavailable."""
    core = _load_core()
    if core is None:
        raise RuntimeError(RUST_BACKEND_UNAVAILABLE_MESSAGE)
    return core


def _load_core() -> ModuleType | None:
    """Import the Rust core or normalize native loader failures.

    A missing extension is represented as ``None``. If discovery succeeds but
    the native module cannot be loaded, fail closed with a package-owned error
    and preserve the loader exception as the cause for operator diagnostics.
    """
    if importlib.util.find_spec(CORE_MODULE) is None:
        return None
    try:
        return importlib.import_module(CORE_MODULE)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "compiled Rust core is present but could not be imported"
        ) from exc
