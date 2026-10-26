"""MockExposureDataProvider — JSON fact files + shared catalog base."""

from __future__ import annotations

from .catalog import JsonCatalogProvider
from .fact_cache import FactCache
from ..models.exposure import ExposureFactNormalized


class MockExposureDataProvider(JsonCatalogProvider):
    """Fixture-backed provider: catalog + lazy ``exposure_facts/<id>.json``."""

    def __init__(
        self,
        mock_data_dir: str,
        *,
        fact_cache: FactCache | None = None,
        max_workers: int = 16,
        cache_max_datasets: int = 1024,
        cache_ttl_seconds: float = 3600.0,
    ) -> None:
        super().__init__(
            mock_data_dir,
            fact_cache=fact_cache,
            max_workers=max_workers,
            cache_max_datasets=cache_max_datasets,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    def _load_facts_uncached(self, dataset_id: str) -> list[ExposureFactNormalized]:
        return self._load_mock_fact_file(dataset_id)

    def _has_facts_available(self, dataset_id: str) -> bool:
        return dataset_id in self._fact_files or self._cache.get(dataset_id) is not None
