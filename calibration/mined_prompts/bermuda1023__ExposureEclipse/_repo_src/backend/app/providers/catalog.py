"""Shared JSON-catalog provider base.

Both mock and SQL providers load the same small catalog (cedents tree,
dataset groups, IED, geometry ids). Only fact loading differs.

Subclass implements :meth:`_load_facts_uncached` and optionally
:meth:`_has_facts_available` for portfolio discovery without loading.
"""

from __future__ import annotations

import csv
import json
import secrets
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..ert.expected_tables import ALL_TABLE_TYPES, EXPECTED_ERT_TABLES, REQUIRED_TABLE_TYPES
from ..models.cedent import Cedent, Programme, ProgrammeChain
from ..models.dataset import (
    DatasetGroup,
    DatasetGroupCreate,
    DatasetStatusResponse,
    ExpectedTableStatus,
)
from ..models.enums import ErtStatus, WarningCode
from ..models.exposure import ExposureFactNormalized, IEDIndustryRow
from ..models.warnings import make_warning
from .base import ExposureDataProvider
from .fact_cache import FactCache, get_global_fact_cache
from .fact_load import FactBatch, LoadError
from .parallel_load import load_many


def resolve_data_dir(path: str) -> Path:
    """Resolve a data dir relative to backend/ when not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    backend_dir = Path(__file__).resolve().parents[2]
    return (backend_dir / p).resolve()


class JsonCatalogProvider(ExposureDataProvider):
    """Catalog from mockdata-style fixtures; facts via subclass + FactCache."""

    fact_load_workers: int = 16

    def __init__(
        self,
        data_dir: str,
        *,
        fact_cache: FactCache | None = None,
        max_workers: int = 16,
        cache_max_datasets: int = 1024,
        cache_ttl_seconds: float = 3600.0,
    ) -> None:
        self._root = resolve_data_dir(data_dir)
        if not self._root.exists():
            raise FileNotFoundError(f"Catalog data dir not found: {self._root}")
        self.fact_load_workers = max(1, max_workers)
        self._cache: FactCache = fact_cache or get_global_fact_cache(
            max_datasets=cache_max_datasets, ttl_seconds=cache_ttl_seconds
        )

        cedents_raw = self._load_json(self._root / "cedents.json")
        self._cedents: list[Cedent] = [Cedent.model_validate(c) for c in cedents_raw]
        self._cedent_by_id: dict[str, Cedent] = {c.cedent_id: c for c in self._cedents}
        self._chain_by_id: dict[str, ProgrammeChain] = {}
        self._programme_by_id: dict[str, Programme] = {}
        self._programme_by_dataset_id: dict[str, Programme] = {}
        for c in self._cedents:
            for ch in c.chains:
                self._chain_by_id[ch.chain_id] = ch
                for p in ch.programmes:
                    self._programme_by_id[p.programme_id] = p
                    self._programme_by_dataset_id[p.dataset_id] = p

        self._dataset_groups: list[DatasetGroup] = [
            DatasetGroup.model_validate(g)
            for g in self._load_json(self._root / "dataset_groups.json", default=[])
        ]
        self._ied: list[IEDIndustryRow] = self._load_ied(self._root / "ied_industry.csv")
        self._geometry_ids: set[str] = self._scan_geometry(self._root / "geo")

        facts_dir = self._root / "exposure_facts"
        self._fact_files: set[str] = set()
        if facts_dir.exists():
            self._fact_files = {p.stem for p in facts_dir.glob("*.json")}
            self._fact_files.update(
                p.name[: -len(".json.gz")] for p in facts_dir.glob("*.json.gz")
            )

    # ── subclass hooks ──────────────────────────────────────────

    @abstractmethod
    def _load_facts_uncached(self, dataset_id: str) -> list[ExposureFactNormalized]:
        """Load one EDM's facts (no cache). May raise on hard failure."""

    def _has_facts_available(self, dataset_id: str) -> bool:
        """True if this dataset can contribute facts without forcing a load."""
        return dataset_id in self._fact_files or self._cache.get(dataset_id) is not None

    # ── loaders ─────────────────────────────────────────────────

    @staticmethod
    def _load_json(path: Path, *, default=None):
        if not path.exists():
            if default is not None:
                return default
            raise FileNotFoundError(f"Required fixture missing: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_ied(path: Path) -> list[IEDIndustryRow]:
        if not path.exists():
            return []
        rows: list[IEDIndustryRow] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            lines = [ln for ln in f if not ln.lstrip().startswith("#")]
        reader = csv.DictReader(lines)
        for raw in reader:
            rows.append(
                IEDIndustryRow.model_validate(
                    {
                        "geography_level": raw["geographyLevel"],
                        "geography_id": raw["geographyId"],
                        "occupancy_segment": raw["occupancySegment"],
                        "industry_tiv": float(raw["industryTIV"]),
                        "currency": raw["currency"],
                        "source_year": int(raw["sourceYear"]) if raw.get("sourceYear") else None,
                    }
                )
            )
        return rows

    @staticmethod
    def _scan_geometry(geo_dir: Path) -> set[str]:
        ids: set[str] = set()
        if not geo_dir.exists():
            return ids
        for fp in geo_dir.glob("*.geojson"):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    gj = json.load(f)
            except json.JSONDecodeError:
                continue
            for feat in gj.get("features", []) or []:
                gid = (feat.get("properties") or {}).get("geographyId")
                if gid:
                    ids.add(gid)
        return ids

    def _load_mock_fact_file(self, dataset_id: str) -> list[ExposureFactNormalized]:
        facts_dir = self._root / "exposure_facts"
        plain = facts_dir / f"{dataset_id}.json"
        packed = facts_dir / f"{dataset_id}.json.gz"
        if plain.exists():
            rows = self._load_json(plain, default=[])
        elif packed.exists():
            import gzip

            with gzip.open(packed, "rt", encoding="utf-8") as handle:
                rows = json.load(handle)
        else:
            return []
        return [ExposureFactNormalized.model_validate(r) for r in rows]

    # ── catalog ABC ─────────────────────────────────────────────

    def list_cedents(self) -> list[Cedent]:
        return list(self._cedents)

    def get_cedent(self, cedent_id: str) -> Cedent | None:
        return self._cedent_by_id.get(cedent_id)

    def get_chain(self, chain_id: str) -> ProgrammeChain | None:
        return self._chain_by_id.get(chain_id)

    def get_programme(self, programme_id: str) -> Programme | None:
        return self._programme_by_id.get(programme_id)

    def get_programme_by_dataset_id(self, dataset_id: str) -> Programme | None:
        return self._programme_by_dataset_id.get(dataset_id)

    def get_dataset_status(self, dataset_id: str) -> DatasetStatusResponse:
        programme = self._programme_by_dataset_id.get(dataset_id)
        if programme is None:
            status = ErtStatus.ERT_NOT_FOUND
            present: set[str] = set()
            edm_name = dataset_id
        else:
            status = ErtStatus(programme.edm.ert_status)
            edm_name = programme.edm.edm_database_name
            if status == ErtStatus.ERT_NOT_FOUND:
                present = set()
            elif status == ErtStatus.ERT_PARTIAL:
                present = set(REQUIRED_TABLE_TYPES) - {"PERIL_DETAILS"}
            else:
                present = set(ALL_TABLE_TYPES)

        table_rows = [
            ExpectedTableStatus(
                table_type=t.table_type,
                name=t.table_name_pattern.format(edm=edm_name, table_type=t.table_type),
                exists=t.table_type in present,
                row_count=(
                    self._cached_row_count(dataset_id) if t.table_type in present else 0
                ),
            )
            for t in EXPECTED_ERT_TABLES
        ]

        warnings = []
        if status == ErtStatus.ERT_NOT_FOUND:
            warnings.append(
                make_warning(WarningCode.WARN_ERT_NOT_FOUND, context={"datasetId": dataset_id})
            )
        elif status == ErtStatus.ERT_PARTIAL:
            missing = sorted(REQUIRED_TABLE_TYPES - present)
            warnings.append(
                make_warning(
                    WarningCode.WARN_ERT_TABLES_PARTIAL,
                    context={"datasetId": dataset_id, "missingTableTypes": missing},
                )
            )

        return DatasetStatusResponse(
            dataset_id=dataset_id,
            ert_status=status,
            tables=table_rows,
            warnings=warnings,
        )

    def _cached_row_count(self, dataset_id: str) -> int:
        cached = self._cache.get(dataset_id)
        if cached is not None:
            return len(cached)
        return 1 if dataset_id in self._fact_files else 0

    def list_dataset_groups(self) -> list[DatasetGroup]:
        return list(self._dataset_groups)

    def get_dataset_group(self, dataset_group_id: str) -> DatasetGroup | None:
        return next(
            (g for g in self._dataset_groups if g.dataset_group_id == dataset_group_id), None
        )

    def create_dataset_group(self, payload: DatasetGroupCreate) -> DatasetGroup:
        new_id = f"grp-{secrets.token_hex(4)}"
        group = DatasetGroup(
            dataset_group_id=new_id,
            group_name=payload.group_name,
            currency=payload.currency,
            combination_method=payload.combination_method,
            members=list(payload.members),
            distinct_segments_confirmed=payload.distinct_segments_confirmed,
            created_at=datetime.now(timezone.utc),
            cedent_name=payload.cedent_name,
            programme_name=payload.programme_name,
            year_of_account=payload.year_of_account,
            notes=payload.notes,
        )
        self._dataset_groups.append(group)
        return group

    # ── facts ───────────────────────────────────────────────────

    def get_facts_for_dataset(self, dataset_id: str) -> list[ExposureFactNormalized]:
        # Cache holds the list; treat as immutable — return as-is (no full copy).
        return self._cache.get_or_load(
            dataset_id, lambda: self._load_facts_uncached(dataset_id)
        )

    def get_facts_for_datasets(self, dataset_ids: Sequence[str]) -> FactBatch:
        errors: list[LoadError] = []

        def loader(ds_id: str) -> list[ExposureFactNormalized]:
            try:
                return self.get_facts_for_dataset(ds_id)
            except Exception as exc:
                errors.append(LoadError(dataset_id=ds_id, message=f"{type(exc).__name__}: {exc}"))
                return []

        by_id = load_many(
            list(dataset_ids),
            loader,
            max_workers=self.fact_load_workers,
            empty=[],
            on_error="empty",
        )
        # load_many may also swallow — surface empty+exception as error if not already
        return FactBatch(by_id=by_id, errors=errors)

    def portfolio_dataset_ids(self, *, in_force_only: bool = True) -> list[str]:
        """Canonical portfolio universe — Option A: in-force BOUND, no orphans."""
        ids: list[str] = []
        seen: set[str] = set()
        for p in self._programme_by_id.values():
            if not p.include_in_portfolio:
                continue
            if in_force_only and not p.is_in_force():
                continue
            if p.edm.ert_status == ErtStatus.ERT_NOT_FOUND:
                continue
            if p.dataset_id not in seen:
                seen.add(p.dataset_id)
                ids.append(p.dataset_id)
        return ids

    def get_portfolio_facts(self) -> list[ExposureFactNormalized]:
        """Share-metric denominator: same universe as portfolio map view."""
        ids = self.portfolio_dataset_ids(in_force_only=True)
        batch = self.get_facts_for_datasets(ids)
        return batch.flatten(ids)

    def get_ied_industry(self) -> list[IEDIndustryRow]:
        return list(self._ied)

    def get_geometry_availability(self) -> set[str]:
        ids: set[str] = set(self._geometry_ids)
        for ds_id in self._cache.cached_keys():
            rows = self._cache.get(ds_id)
            if not rows:
                continue
            for f in rows:
                ids.add(f.geography_id)
        return ids

    # ── ops surface ─────────────────────────────────────────────

    @property
    def fact_cache(self) -> FactCache:
        return self._cache

    def warmup(
        self,
        dataset_ids: Sequence[str] | None = None,
        *,
        in_force_only: bool = True,
    ) -> dict[str, int]:
        if dataset_ids is None:
            ids = self.portfolio_dataset_ids(in_force_only=in_force_only)
            if not in_force_only:
                ids = list(self._programme_by_dataset_id.keys())
        else:
            ids = list(dataset_ids)
        batch = self.get_facts_for_datasets(ids)
        return {k: len(v or []) for k, v in batch.by_id.items()}

    def is_always_fails(self, dataset_id: str) -> bool:
        programme = self._programme_by_dataset_id.get(dataset_id)
        if programme is None:
            return False
        return "AlwaysFails" in programme.edm.edm_database_name
