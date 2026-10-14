"""
Single source of truth for the TraceBi version.

The version lives in ``pyproject.toml`` and is read back from the installed
distribution metadata. The CLI, the web API, and ``tracebi.__version__`` all
resolve through here so they can never drift apart.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version


def get_version() -> str:
    """Return the installed TraceBi version, or ``"unknown"`` if not installed."""
    try:
        return _pkg_version("tracebi")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()
