"""Curated root API and namespace bootstrap for the public Ellements repo."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from pkgutil import extend_path
from typing import Any

__path__ = extend_path(__path__, __name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_ROOTS = [
    "ellements-agents/src/ellements",
    "ellements-benchmarking/src/ellements",
    "ellements-cli/src/ellements",
    "ellements-core/src/ellements",
    "ellements-domain-specific/src/ellements",
    "ellements-execution/src/ellements",
    "ellements-fslm/src/ellements",
    "ellements-reporting/src/ellements",
    "ellements-standard-tools/src/ellements",
]

for relative_root in _SOURCE_ROOTS:
    candidate = _REPO_ROOT / relative_root
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in __path__:
            __path__.append(candidate_str)

_ROOT_EXPORTS: dict[str, tuple[str, str]] = {
    "Conversation": ("ellements.core", "Conversation"),
    "ImageInput": ("ellements.core", "ImageInput"),
    "LLMClient": ("ellements.core", "LLMClient"),
    "SimpleTool": ("ellements.core", "SimpleTool"),
}

try:
    __version__ = version("ellements")
except PackageNotFoundError:
    __version__ = "0.2.10"

__all__ = [*sorted(_ROOT_EXPORTS), "__version__"]


def __getattr__(name: str) -> Any:
    """Resolve curated public root exports lazily."""
    if name not in _ROOT_EXPORTS:
        raise AttributeError(f"module 'ellements' has no attribute {name!r}")
    module_name, attr_name = _ROOT_EXPORTS[name]
    return getattr(import_module(module_name), attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ROOT_EXPORTS))
