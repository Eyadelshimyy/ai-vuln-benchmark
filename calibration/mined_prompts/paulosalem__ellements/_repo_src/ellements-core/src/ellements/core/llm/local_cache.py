"""Explicit local caching configuration for :class:`LLMClient`."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import litellm
from litellm import Cache as LiteLLMCache

if TYPE_CHECKING:
    from ..caching.cache import Cache

_cache_lock = threading.Lock()
_configured_response_directory: Path | None = None
_configured_litellm_cache: Any | None = None


@dataclass(frozen=True, slots=True)
class LocalCacheConfig:
    """Configure exact, persistent local caching for an LLM client.

    Args:
        directory: Root directory for cached responses.
        cache_responses: Cache text, structured, streaming, tool-loop, and
            log-likelihood API responses through LiteLLM's disk cache.
        cache_images: Cache normalized image-generation and image-edit responses
            through Ellements' JSON disk cache. Responses containing only
            provider-hosted URLs are not cached because those URLs expire.
    """

    directory: Path
    cache_responses: bool = True
    cache_images: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory).expanduser())

    @property
    def response_directory(self) -> Path:
        """Return the LiteLLM response-cache directory."""

        return self.directory / "responses"

    @property
    def image_directory(self) -> Path:
        """Return the Ellements image-cache directory."""

        return self.directory / "images"


def configure_local_response_cache(config: LocalCacheConfig) -> None:
    """Configure LiteLLM's process-wide disk cache for one explicit directory."""

    if not config.cache_responses:
        return

    directory = config.response_directory.resolve()
    _prepare_private_directory(directory)

    global _configured_litellm_cache, _configured_response_directory
    with _cache_lock:
        if (
            directory == _configured_response_directory
            and litellm.cache is _configured_litellm_cache
        ):
            return
        configured = LiteLLMCache(type="disk", disk_cache_dir=str(directory))
        litellm.cache = configured
        litellm.enable_caching_on_provider_specific_optional_params = True
        _configured_response_directory = directory
        _configured_litellm_cache = configured


def create_local_image_cache(config: LocalCacheConfig) -> Cache | None:
    """Create the image cache requested by *config*, if enabled."""

    if not config.cache_images:
        return None
    from ..caching.disk import JsonDiskCache

    return JsonDiskCache(config.image_directory)


def _prepare_private_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        directory.chmod(0o700)


__all__ = ["LocalCacheConfig"]
