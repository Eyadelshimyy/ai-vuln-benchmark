"""Scoring and bootstrap logic of the segment-curve holdout comparison."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_segment_curves",
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_segment_curves.py",
)
seg = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = seg
_SPEC.loader.exec_module(seg)


def _cell(res: tuple[float, float], com: tuple[float, float], auto: tuple[float, float]) -> dict:
    return {
        "segments": {"RESIDENTIAL": list(res), "COMMERCIAL": list(com), "AUTO": list(auto)},
        "county": [math.log(res[1] / res[0])],
    }


def _cells(separate_commercial_factor: float) -> dict:
    cells = {}
    for i, event in enumerate(["A", "B", "C", "D", "E", "F"]):
        over = 1.5 if i % 2 else 0.7
        shared = _cell((10, 10 * over), (5, 5 * over * 2), (2, 2 * over))
        separate = _cell((10, 10 * over), (5, 5 * over * separate_commercial_factor), (2, 2 * over))
        cells[event] = {"shared": shared, "level": shared, "separate": separate}
    return cells


def test_identical_models_score_identically_with_zero_bootstrap_change():
    report = seg.summarise(_cells(2.0), draws=200, seed=1)
    assert report["scores"]["separate"] == report["scores"]["shared"]
    assert report["eventWinsVsShared"]["separate"] == 0
    for low, high in report["bootstrapDiffVsShared95"]["separate"].values():
        assert low == high == 0


def test_better_commercial_split_shows_in_segment_score_and_interval():
    report = seg.summarise(_cells(1.0), draws=500, seed=1)
    shared, separate = report["scores"]["shared"], report["scores"]["separate"]
    assert separate["eventSegmentLogRms"] < shared["eventSegmentLogRms"]
    assert report["segmentEventLogRms"]["separate"]["COMMERCIAL"] < (
        report["segmentEventLogRms"]["shared"]["COMMERCIAL"]
    )
    assert report["segmentEventLogRms"]["separate"]["RESIDENTIAL"] == (
        report["segmentEventLogRms"]["shared"]["RESIDENTIAL"]
    )
    assert report["bootstrapDiffVsShared95"]["separate"]["eventSegmentLogRms"][1] < 0
