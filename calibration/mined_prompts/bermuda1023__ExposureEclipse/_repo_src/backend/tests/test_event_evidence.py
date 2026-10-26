"""Post-event evidence: ASOS QC, Storm Events parsing, miss attribution."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

from app.services.event_evidence import county_evidence, miss_peril

_SPEC = importlib.util.spec_from_file_location(
    "build_event_evidence",
    Path(__file__).resolve().parents[1] / "scripts" / "build_event_evidence.py",
)
ev = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ev
_SPEC.loader.exec_module(ev)

T0 = datetime(2024, 10, 9, 20, 0)


def _at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def test_qc_drops_sustained_above_its_gust_and_lone_spikes() -> None:
    rows = [
        (_at(0), 20.0, 30.0),
        (_at(60), 74.0, 48.0),   # sustained above its own gust: bad report
        (_at(120), 25.0, 0.0),
        (_at(180), 70.0, 0.0),   # no gust, neighbours at 25 kt: spike
        (_at(240), 22.0, 31.0),
    ]
    sustained, gust, rejected = ev._qc_peaks(rows)
    assert sustained == 25.0
    assert gust == 48.0
    assert rejected == 2


def test_qc_keeps_an_eyewall_peak_before_the_station_fails() -> None:
    # Sarasota, Milton: 60 kt with an 89 kt gust, then the record stops.
    rows = [(_at(0), 35.0, 63.0), (_at(89), 60.0, 89.0)]
    sustained, gust, _ = ev._qc_peaks(rows)
    assert sustained == 60.0 and gust == 89.0


def test_storm_events_damage_and_county_names() -> None:
    assert ev._damage_usd("10.00K") == 10_000
    assert ev._damage_usd("1.5M") == 1_500_000
    assert ev._damage_usd("") == 0.0
    assert ev._county_key("St. Johns County") == ev._county_key("Saint Johns")
    assert ev._county_key("Orleans Parish") == "orleans"


def test_miss_peril_prefers_tornado_then_flood() -> None:
    doc = {"counties": {
        "US-NC-37021": {"stormEvents": {"flood": {"n": 4}, "wind": {"n": 1}},
                        "hwm": {"n": 3, "riverine": 3, "coastal": 0}},
        "US-GA-13021": {"stormEvents": {"tornado": {"n": 1, "maxEf": 1}, "flood": {"n": 2}}},
    }}
    assert miss_peril(county_evidence(doc, "US-NC-37021")) == "flood"
    assert miss_peril(county_evidence(doc, "US-GA-13021")) == "tornado"
    assert miss_peril(county_evidence(doc, "US-FL-12086")) == "noEvidence"
    assert county_evidence(None, "US-FL-12086") is None
