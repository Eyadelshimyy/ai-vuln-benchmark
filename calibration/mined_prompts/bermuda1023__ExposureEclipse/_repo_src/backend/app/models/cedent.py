"""Cedent / ProgrammeChain / Programme model.

Replaces the flat `DatasetRegistry` row as the primary navigation entity.

Hierarchy
─────────
  Cedent (Farmers Group)
  └── ProgrammeChain  "BDA · Farmers Nationwide WS"   ← unit of YoY comparison
      ├── Programme 2027  (programmeId=P-..)  → EDMRef → SQL pointer
      ├── Programme 2026                     → auto-prior in this chain
      └── …

Why a chain?
  In reinsurance, the same "deal slot" renews year over year with a fresh
  programmeId — but is conceptually the same programme. YoY comparisons run
  within a chain. Each office writes its own chain; one cedent can have many
  chains (different perils, different territories, different offices).

EDMRef
  The SQL pointer that used to live on `DatasetRegistry`. A programme has
  exactly one EDM; two programmes (e.g. different years in a chain) may
  reference the same EDM if appropriate, but typically each year has its own
  data cut.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .common import CamelModel
from .enums import AggregationLevel, ErtStatus, Peril, ProgrammeStatus


class EDMRef(CamelModel):
    """The SQL data-source pointer. Frontend never imports this directly — it
    rides inside a Programme."""

    server_name: str
    edm_database_name: str
    currency: str
    ert_status: ErtStatus
    available_granularity: list[AggregationLevel]
    last_generated_at: datetime | None = None
    exposure_data_cutoff_date: datetime | None = None


class Programme(CamelModel):
    """A specific bound (or quoted) programme for one treaty year.

    `perils` is the canonical list of perils this programme's EDM carries (an
    office's annual EDM is typically multi-peril — WS + EQ + CS together). The
    peril selector at the top of the page filters which of these get rendered.
    `peril` is kept as legacy metadata for single-peril cases.
    """

    programme_id: str
    chain_id: str
    cedent_id: str
    programme_name: str
    treaty_year: int
    perils: list[Peril] = []
    peril: Peril = Peril.ALL
    office: str
    underwriter: str
    status: ProgrammeStatus = ProgrammeStatus.BOUND
    layer: str | None = None
    signed_share: float | None = None
    inception_date: datetime | None = None
    expiry_date: datetime | None = None
    notes: str | None = None
    # Reference books (the Industry cedant) are selectable but are not
    # part of the in-force client portfolio. Client deals default to True.
    include_in_portfolio: bool = True
    edm: EDMRef
    # `dataset_id` is the legacy stable id used by exposure_facts/<id>.json. We
    # keep it so we don't have to rename every fact file when promoting to the
    # cedent model — many programmes will share their dataset_id with their
    # programme_id, but a chain that renames itself can keep historical facts.
    dataset_id: str

    def is_in_force(self, as_of: datetime | None = None) -> bool:
        """A programme is in-force iff status=BOUND and ``as_of`` falls within
        [inception_date, expiry_date]. Missing dates → treated as bound forever.
        ``as_of`` defaults to ``datetime.now(timezone.utc)``."""
        if self.status != ProgrammeStatus.BOUND:
            return False
        now = as_of or datetime.now(timezone.utc)
        # Some mock dates are naive; coerce both sides to UTC-aware for comparison.
        if self.inception_date:
            inc = self.inception_date
            if inc.tzinfo is None:
                inc = inc.replace(tzinfo=timezone.utc)
            if now < inc:
                return False
        if self.expiry_date:
            exp = self.expiry_date
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if now > exp:
                return False
        return True


class ProgrammeChain(CamelModel):
    """The renewal lineage for one deal slot. Comparison is within the chain."""

    chain_id: str
    cedent_id: str
    chain_name: str
    office: str
    default_peril: Peril
    programmes: list[Programme]  # ordered newest-first


class Cedent(CamelModel):
    """Top-level grouping above chains. One company; many chains."""

    cedent_id: str
    cedent_name: str
    chains: list[ProgrammeChain]
    region: str | None = None  # short bucket: "Nationwide" / "California" / "Southeast"
    notes: str | None = None


class CedentTreeResponse(CamelModel):
    """`GET /api/cedents` response shape."""

    cedents: list[Cedent]
