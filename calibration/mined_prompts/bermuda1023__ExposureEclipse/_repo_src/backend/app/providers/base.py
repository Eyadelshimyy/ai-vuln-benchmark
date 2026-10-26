"""ExposureDataProvider — the boundary between API/services and the data source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from threading import Lock
from typing import Protocol, runtime_checkable

from ..models.cedent import Cedent, Programme, ProgrammeChain
from ..models.dataset import (
    DatasetGroup,
    DatasetGroupCreate,
    DatasetStatusResponse,
)
from ..models.exposure import ExposureFactNormalized, IEDIndustryRow
from .fact_load import FactBatch


class ExposureDataProvider(ABC):
    """Abstract contract every provider (mock, sqlserver, hybrid) must satisfy.

    Prefer :meth:`get_facts_for_datasets` for multi-programme scopes.
    Prefer :meth:`portfolio_dataset_ids` for the in-force portfolio universe
    (map view and share-metric denominators share this set).
    """

    # Parallelism for bulk loads; subclasses set from settings.
    fact_load_workers: int = 16

    @abstractmethod
    def list_cedents(self) -> list[Cedent]: ...

    @abstractmethod
    def get_cedent(self, cedent_id: str) -> Cedent | None: ...

    @abstractmethod
    def get_chain(self, chain_id: str) -> ProgrammeChain | None: ...

    @abstractmethod
    def get_programme(self, programme_id: str) -> Programme | None: ...

    @abstractmethod
    def get_programme_by_dataset_id(self, dataset_id: str) -> Programme | None: ...

    @abstractmethod
    def get_dataset_status(self, dataset_id: str) -> DatasetStatusResponse: ...

    @abstractmethod
    def list_dataset_groups(self) -> list[DatasetGroup]: ...

    @abstractmethod
    def get_dataset_group(self, dataset_group_id: str) -> DatasetGroup | None: ...

    @abstractmethod
    def create_dataset_group(self, payload: DatasetGroupCreate) -> DatasetGroup: ...

    @abstractmethod
    def get_facts_for_dataset(self, dataset_id: str) -> list[ExposureFactNormalized]:
        """Normalized fact rows for one EDM. Implementations SHOULD cache."""

    def get_facts_for_datasets(self, dataset_ids: Sequence[str]) -> FactBatch:
        """Bulk-load many EDMs (parallel). Returns facts + per-dataset errors."""
        from .fact_load import LoadError
        from .parallel_load import load_many

        errors: list[LoadError] = []
        errors_lock = Lock()

        def loader(ds_id: str) -> list[ExposureFactNormalized]:
            try:
                return self.get_facts_for_dataset(ds_id)
            except Exception as exc:
                with errors_lock:
                    errors.append(
                        LoadError(dataset_id=ds_id, message=f"{type(exc).__name__}: {exc}")
                    )
                return []

        by_id = load_many(
            list(dataset_ids),
            loader,
            max_workers=self.fact_load_workers,
            empty=[],
            on_error="empty",
        )
        return FactBatch(by_id=by_id, errors=errors)

    def portfolio_dataset_ids(self, *, in_force_only: bool = True) -> list[str]:
        """Dataset ids in the portfolio universe (default: in-force BOUND only)."""
        ids: list[str] = []
        seen: set[str] = set()
        for cedent in self.list_cedents():
            for chain in cedent.chains:
                for prog in chain.programmes:
                    if not prog.include_in_portfolio:
                        continue
                    if in_force_only and not prog.is_in_force():
                        continue
                    if prog.dataset_id not in seen:
                        seen.add(prog.dataset_id)
                        ids.append(prog.dataset_id)
        return ids

    @abstractmethod
    def get_portfolio_facts(self) -> list[ExposureFactNormalized]:
        """Facts for share-metric denominators — must match portfolio_dataset_ids()."""

    @abstractmethod
    def get_ied_industry(self) -> list[IEDIndustryRow]: ...

    @abstractmethod
    def get_geometry_availability(self) -> set[str]: ...


@runtime_checkable
class ProviderOps(Protocol):
    """Optional multi-EDM ops surface (cache / warmup / SQL registry)."""

    @property
    def fact_cache(self): ...

    def warmup(
        self,
        dataset_ids: Sequence[str] | None = None,
        *,
        in_force_only: bool = True,
    ) -> dict[str, int]: ...


@runtime_checkable
class ProviderWithRegistry(Protocol):
    @property
    def connection_registry(self): ...
