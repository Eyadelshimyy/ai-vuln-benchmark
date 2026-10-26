"""Data-access providers. The concrete one is chosen by ``DATA_PROVIDER`` env."""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .base import ExposureDataProvider
from .connection_registry import load_registry_from_settings
from .fact_cache import get_global_fact_cache, reset_global_fact_cache


@lru_cache(maxsize=1)
def get_provider() -> ExposureDataProvider:
    """Factory — returns the configured provider as a process-wide singleton.

    Caching matters: providers hold in-memory state (dataset groups, connection
    pools) that would otherwise vanish when FastAPI re-resolves
    ``Depends(get_provider)``. Tests clear via :func:`get_provider.cache_clear`
    and :func:`reset_global_fact_cache`.
    """
    settings = get_settings()
    # Ensure process-wide cache is sized from settings on first provider build.
    cache = get_global_fact_cache(
        max_datasets=settings.fact_cache_max_datasets,
        ttl_seconds=settings.fact_cache_ttl_seconds,
    )
    workers = settings.fact_load_max_workers

    if settings.data_provider == "mock":
        from .mock import MockExposureDataProvider

        return MockExposureDataProvider(
            settings.mock_data_dir,
            fact_cache=cache,
            max_workers=workers,
            cache_max_datasets=settings.fact_cache_max_datasets,
            cache_ttl_seconds=settings.fact_cache_ttl_seconds,
        )

    if settings.data_provider in ("sqlserver", "hybrid"):
        from .sqlserver import SqlServerExposureDataProvider

        registry = load_registry_from_settings(
            servers_file=settings.sqlserver_servers_file,
            default_user=settings.sqlserver_default_user,
            default_password=settings.sqlserver_default_password,
            default_driver=settings.sqlserver_default_driver,
            inline_json=settings.sqlserver_servers_json,
        )
        # hybrid: optional mock fallback; sqlserver: never mock.
        fallback = (
            settings.hybrid_fallback_on_sql_error
            if settings.data_provider == "hybrid"
            else False
        )
        return SqlServerExposureDataProvider(
            catalog_dir=settings.mock_data_dir,
            registry=registry,
            fact_cache=cache,
            max_workers=workers,
            evolution_table_pattern=settings.sqlserver_evolution_table_pattern,
            fallback_mock=fallback,
            cache_max_datasets=settings.fact_cache_max_datasets,
            cache_ttl_seconds=settings.fact_cache_ttl_seconds,
        )

    if settings.data_provider == "databricks":  # pragma: no cover — v2
        raise NotImplementedError("DatabricksExposureDataProvider lands in Phase 12.")

    raise ValueError(f"Unknown DATA_PROVIDER: {settings.data_provider!r}")


def clear_provider_state() -> None:
    """Test helper — drop provider singleton + fact cache."""
    get_provider.cache_clear()
    reset_global_fact_cache()


__all__ = [
    "ExposureDataProvider",
    "get_provider",
    "clear_provider_state",
    "reset_global_fact_cache",
]
