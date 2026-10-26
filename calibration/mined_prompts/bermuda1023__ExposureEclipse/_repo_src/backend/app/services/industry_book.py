"""County TIV from the Industry client book.

The book is the supplied US windstorm TIV workbook: residential, commercial,
and auto by county. Counties that are not on that sheet are not in the book.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from ..config import get_settings


@lru_cache(maxsize=1)
def _by_geography() -> dict[str, tuple[int, int, int, int]]:
    path = Path(get_settings().mock_data_dir) / "industry_exposure_counties.csv"
    if not path.exists():
        return {}
    out: dict[str, tuple[int, int, int, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gid = row.get("geographyId") or ""
            if not gid:
                continue
            res = int(float(row["residentialTiv"]))
            com = int(float(row["commercialTiv"]))
            auto = int(float(row.get("autoTiv") or 0))
            out[gid] = (res, com, auto, res + com + auto)
    return out


def county_industry_tiv(geography_id: str) -> tuple[int, int, int, int] | None:
    """Residential, commercial, auto, and bundled TIV for one county.

    Accepts ``US-FL-12086`` or a raw 5-digit GEOID. Bundled is the sum of
    the three segments.
    """
    if not geography_id:
        return None
    book = _by_geography()
    if geography_id in book:
        return book[geography_id]
    tail = geography_id.split("-")[-1]
    for gid, values in book.items():
        if gid.endswith("-" + tail):
            return values
    return None
