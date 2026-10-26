"""IBTrACS parse: best-track wind radii land on the track points."""

from __future__ import annotations

import io

import app.services.ibtracs as ib

_QUADS = ("NE", "SE", "SW", "NW")
_COLUMNS = [
    "USA_ATCF_ID", "ISO_TIME", "USA_RMW", "SEASON", "NAME", "USA_LAT", "USA_LON",
    "USA_WIND", "USA_PRES", "USA_STATUS", "USA_RECORD",
    *(f"USA_R{r}_{q}" for r in (64, 50, 34) for q in _QUADS),
]


def _row(**values: str) -> str:
    return ",".join(values.get(c, "") for c in _COLUMNS)


def _csv() -> bytes:
    lines = [
        ",".join(_COLUMNS),
        ",".join("" for _ in _COLUMNS),
        _row(
            USA_ATCF_ID="AL092020", ISO_TIME="2020-08-04 03:00:00", SEASON="2020",
            NAME="ISAIAS", USA_LAT="33.9", USA_LON="-78.3", USA_WIND="75",
            USA_PRES="988", USA_STATUS="HU", USA_RECORD="L", USA_RMW="30",
            USA_R64_NE="20", USA_R64_SE="20", USA_R64_SW="0", USA_R64_NW="0",
            USA_R50_NE="60", USA_R50_SE="40", USA_R50_SW="0", USA_R50_NW="20",
            USA_R34_NE="160", USA_R34_SE="150", USA_R34_SW="60", USA_R34_NW="80",
        ),
        _row(
            USA_ATCF_ID="AL092020", ISO_TIME="2020-08-04 12:00:00", SEASON="2020",
            NAME="ISAIAS", USA_LAT="37.5", USA_LON="-76.9", USA_WIND="60",
            USA_PRES="992", USA_STATUS="TS",
            USA_R34_NE="180", USA_R34_SE="160", USA_R34_SW="0", USA_R34_NW="40",
        ),
    ]
    return ("\n".join(lines) + "\n").encode()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def test_r34_and_r50_quadrants_are_kept_on_each_fix(monkeypatch) -> None:
    monkeypatch.setattr(ib.urllib.request, "urlopen", lambda *a, **k: _Resp(_csv()))
    ib._parse_csv.cache_clear()
    try:
        storm = ib._storms_index()["AL092020"]
        landfall, inland = storm.track
        assert landfall.r34_quads_nm == (160.0, 150.0, 60.0, 80.0)
        assert landfall.r50_quads_nm == (60.0, 40.0, 0.0, 20.0)
        assert inland.r34_quads_nm == (180.0, 160.0, 0.0, 40.0)
        assert inland.r50_quads_nm is None
        # Best-track radii are not live NHC radii.
        assert landfall.radii_source is None and landfall.r64_quads_nm is None
        assert ib._r64_quads_index()[("AL092020", "202008040300")] == (20.0, 20.0, 0.0, 0.0)
    finally:
        ib._parse_csv.cache_clear()
