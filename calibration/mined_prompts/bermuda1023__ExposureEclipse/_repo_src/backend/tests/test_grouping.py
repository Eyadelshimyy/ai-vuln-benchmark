"""Tests for ``app.services.grouping`` — dataset-group combination methods.

Centerpiece is the worked example from CALCULATION_RULES.md
§"Worked example (max-across-perils)". The combiner MUST agree with the
table in the docs exactly — that's the canonical proof that the default
multi-peril combination behavior obeys CLAUDE.md rule 3.
"""

from __future__ import annotations

import pytest

from app.models.enums import (
    AggregationLevel,
    CombinationMethod,
    OccupancySegment,
    Peril,
)
from app.models.exposure import ExposureFactNormalized
from app.services.grouping import combine_at_grain, location_count_at_max_peril


def fact(
    *,
    statecode: str = "FL",
    peril: Peril = Peril.WS,
    tiv: float = 0.0,
    locations: int = 0,
    occupancy_segment: OccupancySegment = OccupancySegment.UNKNOWN,
    occupancy: str | None = None,
    aggregation: AggregationLevel = AggregationLevel.STATE,
    dataset_id: str = "ds-test",
    currency: str = "USD",
) -> ExposureFactNormalized:
    """Minimal fact for grouping tests; ``geography_id = US-<statecode>``."""
    return ExposureFactNormalized(
        dataset_id=dataset_id,
        portname="09302025",
        source_server_name="srv",
        source_database_name="db",
        source_table_name="tbl",
        aggregation=aggregation,
        geography_level=aggregation,
        country="US",
        country_name="United States",
        statecode=statecode,
        state_name=statecode,
        geography_id=f"US-{statecode}",
        peril=peril,
        occupancy=occupancy,
        occupancy_segment=occupancy_segment,
        tiv=tiv,
        location_count=locations,
        currency=currency,
    )


# ───────────────────────── Worked example ─────────────────────────


def test_max_across_perils_at_state_grain_matches_worked_example() -> None:
    """CALCULATION_RULES.md §Worked example — canonical proof of CLAUDE.md rule 3."""
    facts = [
        fact(statecode="FL", peril=Peril.WS, tiv=12_400_000_000, locations=42_318),
        fact(statecode="FL", peril=Peril.EQ, tiv=9_100_000_000, locations=42_900),
        fact(statecode="FL", peril=Peril.CS, tiv=7_800_000_000, locations=41_500),
        fact(statecode="CA", peril=Peril.WS, tiv=3_000_000_000, locations=10_100),
        fact(statecode="CA", peril=Peril.EQ, tiv=14_200_000_000, locations=10_500),
        fact(statecode="CA", peril=Peril.CS, tiv=2_100_000_000, locations=9_900),
    ]
    combined = combine_at_grain(
        facts,
        grain=("statecode",),
        method=CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN,
    )
    assert combined[("FL",)] == 12_400_000_000  # WS wins for FL
    assert combined[("CA",)] == 14_200_000_000  # EQ wins for CA

    # Location count = count from the EDM (peril) that supplied the max TIV.
    counts = location_count_at_max_peril(facts, grain=("statecode",))
    assert counts[("FL",)] == 42_318  # WS
    assert counts[("CA",)] == 10_500  # EQ


# ───────────────────────── Grain sensitivity ─────────────────────────


def test_max_across_perils_recomputed_when_grain_changes() -> None:
    """Drill-down changes the key — max is per (key) not per state. CALCULATION_RULES.md."""
    facts = [
        # FL Residential: WS dominates
        fact(statecode="FL", peril=Peril.WS, occupancy="ResA", tiv=10),
        fact(statecode="FL", peril=Peril.EQ, occupancy="ResA", tiv=3),
        # FL Commercial: EQ dominates
        fact(statecode="FL", peril=Peril.WS, occupancy="ComA", tiv=4),
        fact(statecode="FL", peril=Peril.EQ, occupancy="ComA", tiv=8),
    ]
    by_state = combine_at_grain(
        facts,
        grain=("statecode",),
        method=CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN,
    )
    # At state grain WS=14 (10+4) vs EQ=11 (3+8) → WS wins, max = 14
    assert by_state[("FL",)] == 14

    by_state_occ = combine_at_grain(
        facts,
        grain=("statecode", "occupancy"),
        method=CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN,
    )
    assert by_state_occ[("FL", "ResA")] == 10  # WS
    assert by_state_occ[("FL", "ComA")] == 8  # EQ


# ───────────────────────── Sum-distinct-segments guard ─────────────────────────


def test_sum_distinct_segments_requires_explicit_confirmation() -> None:
    """CLAUDE.md rule 3 — summing across peril EDMs requires explicit opt-in."""
    facts = [
        fact(statecode="FL", peril=Peril.WS, tiv=10),
        fact(statecode="FL", peril=Peril.EQ, tiv=20),
    ]
    with pytest.raises(ValueError):
        combine_at_grain(
            facts,
            grain=("statecode",),
            method=CombinationMethod.SUM_DISTINCT_SEGMENTS,
        )


def test_sum_distinct_segments_with_confirmation_sums() -> None:
    facts = [
        fact(statecode="FL", peril=Peril.WS, tiv=10),
        fact(statecode="FL", peril=Peril.EQ, tiv=20),
        fact(statecode="CA", peril=Peril.EQ, tiv=5),
    ]
    out = combine_at_grain(
        facts,
        grain=("statecode",),
        method=CombinationMethod.SUM_DISTINCT_SEGMENTS,
        distinct_segments_confirmed=True,
    )
    assert out[("FL",)] == 30
    assert out[("CA",)] == 5


# ───────────────────────── Keep perils separate ─────────────────────────


def test_keep_perils_separate_returns_one_entry_per_grain_and_peril() -> None:
    facts = [
        fact(statecode="FL", peril=Peril.WS, tiv=10),
        fact(statecode="FL", peril=Peril.EQ, tiv=20),
        fact(statecode="FL", peril=Peril.CS, tiv=30),
        fact(statecode="CA", peril=Peril.WS, tiv=4),
    ]
    out = combine_at_grain(
        facts,
        grain=("statecode",),
        method=CombinationMethod.KEEP_PERILS_SEPARATE,
    )
    # Entries keyed by (grain_key + peril)
    assert out[("FL", Peril.WS)] == 10
    assert out[("FL", Peril.EQ)] == 20
    assert out[("FL", Peril.CS)] == 30
    assert out[("CA", Peril.WS)] == 4
    # Exactly four entries — no implicit combination.
    assert len(out) == 4


# ───────────────────────── SELECTED_EDM_AS_BASE ─────────────────────────


def test_selected_edm_as_base_requires_base_dataset_id() -> None:
    facts = [fact(statecode="FL", peril=Peril.WS, tiv=10, dataset_id="ds-1")]
    with pytest.raises(ValueError):
        combine_at_grain(
            facts,
            grain=("statecode",),
            method=CombinationMethod.SELECTED_EDM_AS_BASE,
        )


def test_selected_edm_as_base_uses_only_base_dataset_tiv() -> None:
    facts = [
        fact(statecode="FL", peril=Peril.WS, tiv=10, dataset_id="ds-base"),
        fact(statecode="FL", peril=Peril.EQ, tiv=20, dataset_id="ds-other"),
        fact(statecode="CA", peril=Peril.EQ, tiv=5, dataset_id="ds-other"),
        fact(statecode="CA", peril=Peril.WS, tiv=7, dataset_id="ds-base"),
    ]
    out = combine_at_grain(
        facts,
        grain=("statecode",),
        method=CombinationMethod.SELECTED_EDM_AS_BASE,
        base_dataset_id="ds-base",
    )
    assert out[("FL",)] == 10
    assert out[("CA",)] == 7
