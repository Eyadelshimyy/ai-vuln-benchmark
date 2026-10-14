from __future__ import annotations

from typing import TYPE_CHECKING

from app.plugins.errors import (
    ActiveTestingNotPermittedError,
    AdapterConflictError,
    AdapterError,
    AdapterNotFoundError,
    AdapterNotImplementedError,
    DuplicateAdapterError,
    OutOfScopeError,
    UnsupportedCapabilityError,
)
from app.plugins.registry import AdapterRegistry

if TYPE_CHECKING:
    from app.plugins.catalog import PluginCatalog

__all__ = [
    "ActiveTestingNotPermittedError",
    "AdapterConflictError",
    "AdapterError",
    "AdapterNotFoundError",
    "AdapterNotImplementedError",
    "AdapterRegistry",
    "DuplicateAdapterError",
    "OutOfScopeError",
    "PluginCatalog",
    "UnsupportedCapabilityError",
    "get_plugin_catalog",
    "reset_plugin_catalog",
]


def get_plugin_catalog() -> PluginCatalog:
    from app.plugins.catalog import get_plugin_catalog as _get

    return _get()


def reset_plugin_catalog() -> None:
    from app.plugins.catalog import reset_plugin_catalog as _reset

    _reset()


def __getattr__(name: str) -> object:
    if name == "PluginCatalog":
        from app.plugins.catalog import PluginCatalog

        return PluginCatalog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
