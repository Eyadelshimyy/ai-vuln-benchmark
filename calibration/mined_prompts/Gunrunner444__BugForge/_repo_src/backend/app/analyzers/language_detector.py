from __future__ import annotations

from pathlib import Path

from app.adapters.languages.registry import LanguageRegistry
from app.domain.language import LanguageStats

__all__ = ["LanguageStats", "detect_languages", "language_for_path", "extension_map"]


def _languages() -> LanguageRegistry:
    from app.plugins import get_plugin_catalog

    return get_plugin_catalog().languages


def language_for_path(path: Path) -> str | None:
    return _languages().language_id_for_path(path)


def detect_languages(file_paths: list[Path]) -> list[LanguageStats]:
    return _languages().detect_languages(file_paths)


def _legacy_extension_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for adapter in _languages().all_adapters():
        for ext in adapter.file_extensions:
            mapping[ext] = adapter.language_id
    return mapping


def extension_map() -> dict[str, str]:
    """Compatibility snapshot of adapter extensions. The registry is authoritative."""
    return _legacy_extension_map()
