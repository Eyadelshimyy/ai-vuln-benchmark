"""Post-event evidence baked by ``scripts/build_event_evidence.py``.

One JSON per storm in ``mockdata/event_evidence/``: ASOS station winds next to
the modelled wind, NCEI Storm Events reports by county and peril, and USGS
high-water marks. Diagnostic only. It explains backtest misses (rain flood,
tornadoes, a wind field that is too strong) and never changes a loss.
"""

from __future__ import annotations

import json
from functools import lru_cache

from . import industry_losses

EVIDENCE_DIR = "event_evidence"
# Order matters: the first peril with a report names a footprint miss.
MISS_PERILS = ("tornado", "flood", "surge", "wind")


@lru_cache(maxsize=32)
def load_evidence(storm_id: str) -> dict | None:
    path = industry_losses.data_dir() / EVIDENCE_DIR / f"{(storm_id or '').upper()}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def clear_caches() -> None:
    load_evidence.cache_clear()


def county_evidence(doc: dict | None, geography_id: str) -> dict | None:
    """Flat per-county summary for a backtest row."""
    if not doc:
        return None
    rec = doc.get("counties", {}).get(geography_id)
    if not rec:
        return None
    se = rec.get("stormEvents") or {}
    asos = rec.get("asos") or {}
    hwm = rec.get("hwm") or {}
    return {
        "asosMaxSustainedKt": asos.get("maxSustainedKt"),
        "asosMaxGustKt": asos.get("maxGustKt"),
        "asosModelLandKt": asos.get("modelLandKt"),
        "tornadoes": (se.get("tornado") or {}).get("n", 0),
        "maxEf": (se.get("tornado") or {}).get("maxEf"),
        "floodReports": (se.get("flood") or {}).get("n", 0),
        "surgeReports": (se.get("surge") or {}).get("n", 0),
        "windReports": (se.get("wind") or {}).get("n", 0),
        "hwmCount": hwm.get("n", 0),
        "hwmRiverine": hwm.get("riverine", 0),
        "hwmCoastal": hwm.get("coastal", 0),
        "hwmMaxHeightAboveGroundFt": hwm.get("maxHeightAboveGroundFt"),
    }


def miss_peril(ev: dict | None) -> str:
    """Which reported peril most likely explains loss outside the wind field."""
    if not ev:
        return "noEvidence"
    counts = {
        "tornado": ev["tornadoes"],
        "flood": ev["floodReports"] + ev["hwmRiverine"],
        "surge": ev["surgeReports"] + ev["hwmCoastal"],
        "wind": ev["windReports"],
    }
    for peril in MISS_PERILS:
        if counts[peril] > 0:
            return peril
    return "noEvidence"


def summary(doc: dict | None) -> dict | None:
    if not doc:
        return None
    return {
        "builtAt": doc.get("builtAt"),
        "window": doc.get("window"),
        "sources": doc.get("sources"),
        "windBias": doc.get("windBias"),
        "stormEventTotals": doc.get("stormEventTotals"),
        "currency": doc.get("currency", "USD"),
        "stations": len(doc.get("stations") or []),
    }
