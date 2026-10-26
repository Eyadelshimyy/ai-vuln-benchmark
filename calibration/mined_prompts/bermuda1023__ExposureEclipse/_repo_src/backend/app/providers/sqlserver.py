"""SqlServer / hybrid multi-EDM provider — catalog base + SQL fact load."""

from __future__ import annotations

import logging

from ..models.exposure import ExposureFactNormalized
from .catalog import JsonCatalogProvider
from .connection_registry import ConnectionRegistry
from .fact_cache import FactCache
from .sql_row_map import map_sql_row_to_fact, rows_from_cursor

logger = logging.getLogger(__name__)


class SqlServerExposureDataProvider(JsonCatalogProvider):
    """JSON catalog + SQL facts (optional mock fallback for hybrid demos)."""

    def __init__(
        self,
        *,
        catalog_dir: str,
        registry: ConnectionRegistry,
        fact_cache: FactCache | None = None,
        max_workers: int = 16,
        evolution_table_pattern: str = "{edm}__EVOLUTION",
        fallback_mock: bool = True,
        cache_max_datasets: int = 1024,
        cache_ttl_seconds: float = 3600.0,
    ) -> None:
        super().__init__(
            catalog_dir,
            fact_cache=fact_cache,
            max_workers=max_workers,
            cache_max_datasets=cache_max_datasets,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self._registry = registry
        self._table_pattern = evolution_table_pattern
        self._fallback_mock = fallback_mock
        # Track hybrid fallbacks so callers can warn (dataset_id → reason).
        self.last_fallback_reasons: dict[str, str] = {}

    @property
    def connection_registry(self) -> ConnectionRegistry:
        return self._registry

    def _has_facts_available(self, dataset_id: str) -> bool:
        if self._cache.get(dataset_id) is not None:
            return True
        programme = self._programme_by_dataset_id.get(dataset_id)
        if programme is None:
            return self._fallback_mock and dataset_id in self._fact_files
        if programme.edm.server_name in self._registry:
            return True
        return self._fallback_mock and dataset_id in self._fact_files

    def _load_facts_uncached(self, dataset_id: str) -> list[ExposureFactNormalized]:
        self.last_fallback_reasons.pop(dataset_id, None)
        programme = self._programme_by_dataset_id.get(dataset_id)
        if programme is None:
            if self._fallback_mock:
                rows = self._load_mock_fact_file(dataset_id)
                if rows:
                    self.last_fallback_reasons[dataset_id] = "no programme in catalog; used mock file"
                return rows
            raise KeyError(f"Dataset {dataset_id!r} has no programme in the catalog")

        edm = programme.edm
        if edm.server_name in self._registry:
            try:
                return self._load_sql(
                    dataset_id=dataset_id,
                    server_name=edm.server_name,
                    database_name=edm.edm_database_name,
                    currency=edm.currency,
                )
            except Exception as exc:
                logger.exception(
                    "SQL load failed for %s @ %s/%s",
                    dataset_id,
                    edm.server_name,
                    edm.edm_database_name,
                )
                if not self._fallback_mock:
                    raise
                self.last_fallback_reasons[dataset_id] = (
                    f"SQL failed ({type(exc).__name__}); used mock file"
                )

        if self._fallback_mock:
            if edm.server_name not in self._registry:
                self.last_fallback_reasons[dataset_id] = (
                    f"server {edm.server_name!r} not in registry; used mock file"
                )
            return self._load_mock_fact_file(dataset_id)

        logger.warning(
            "No registry entry for server %s (dataset %s) and fallback_mock=False",
            edm.server_name,
            dataset_id,
        )
        raise RuntimeError(
            f"Server {edm.server_name!r} is not registered for dataset {dataset_id!r}"
        )

    def _evolution_table_name(self, edm_database_name: str) -> str:
        return self._table_pattern.format(edm=edm_database_name)

    def _load_sql(
        self,
        *,
        dataset_id: str,
        server_name: str,
        database_name: str,
        currency: str,
    ) -> list[ExposureFactNormalized]:
        table = self._evolution_table_name(database_name)
        safe = table.replace("]", "]]")
        # Reject obviously unsafe identifiers.
        if any(c in safe for c in (";", "--", "/*", "*/")):
            raise ValueError(f"Refusing unsafe table name: {safe!r}")

        raw_rows: list[dict] | None = None
        source_table = safe
        with self._registry.connection(server_name, database_name) as conn:
            for candidate in ("ee_exposure_facts", safe):
                cur = conn.cursor()
                try:
                    cur.execute(f"SELECT * FROM [{candidate}]")
                    raw_rows = rows_from_cursor(cur)
                    source_table = candidate
                    cur.close()
                    break
                except Exception:
                    try:
                        cur.close()
                    except Exception:
                        pass
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    continue
        if raw_rows is None:
            raise RuntimeError(
                f"No readable fact table on {server_name}/{database_name} "
                f"(tried ee_exposure_facts and {safe})"
            )

        facts = [
            map_sql_row_to_fact(
                row,
                dataset_id=dataset_id,
                server_name=server_name,
                database_name=database_name,
                source_table_name=source_table,
                default_currency=currency,
            )
            for row in raw_rows
        ]
        logger.info(
            "Loaded %d facts from %s/%s.%s for %s",
            len(facts),
            server_name,
            database_name,
            source_table,
            dataset_id,
        )
        return facts
