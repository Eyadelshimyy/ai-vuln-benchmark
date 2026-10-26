"""Live-storm reliability: lagged ensembles, caches that forget failures,
wind-radii parsing, and a shared deadline that keeps finished work.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error

from app.api import live as live_api
from app.services import atcf_adecks, nhc_gis, wind_forecast
from app.services.atcf_adecks import ModelFix, ModelTrack, fetch_model_tracks
from app.services.ensemble_envelope import build_envelope
from app.services.nhc_gis import fetch_prior_forecast_tracks, parse_wind_radii_kml


def _row(init: str, tech: str, tau: int, lat: int = 220, lon: int = 941) -> str:
    return (
        f"AL, 09, {init}, 03, {tech:>4}, {tau:>3}, "
        f"{lat:>3}N, {lon:>4}W,  35, 1004, TS\n"
    )


def _lagged_adeck() -> bytes:
    """Early aids on 06Z, GEFS still on 00Z, one GraphCast, one ancient member."""
    rows: list[str] = []
    taus = (12, 24, 36, 48, 72)
    for tau in taus:
        rows.append(_row("2026100706", "OFCL", tau, lat=220 + tau // 12))
        rows.append(_row("2026100706", "AVNI", tau, lat=222))
        rows.append(_row("2026100706", "GDMI", tau, lat=221, lon=940 + tau // 12))
    for member in range(1, 7):
        for tau in taus:
            rows.append(_row(
                "2026100700", f"AP{member:02d}", tau,
                lat=218 + member, lon=930 + tau // 6,
            ))
    # 48 h behind the newest cycle — outside the lag window.
    for tau in taus:
        rows.append(_row("2026100506", "AP30", tau, lat=200))
    return "".join(rows).encode("ascii")


def test_lagged_gefs_still_builds_envelopes(monkeypatch) -> None:
    monkeypatch.setattr(
        atcf_adecks, "_download_adeck",
        lambda basin, cy, year, *, refresh=False: _lagged_adeck(),
    )
    tracks = fetch_model_tracks("AL092026")
    by_tech = {t.tech_id: t for t in tracks}
    assert by_tech["OFCL"].init_cycle == "2026-10-07T06Z"
    assert by_tech["AP01"].init_cycle == "2026-10-07T00Z"
    assert by_tech["AP01"].family == "gefs_ens"
    assert "AP30" not in by_tech
    assert by_tech["GDMI"].family == "ai"

    env = build_envelope(tracks)
    assert env is not None
    assert env.members_used >= 5
    assert env.ring[0] == env.ring[-1]

    ai = build_envelope(tracks, include_families=frozenset({"ai"}), min_members=1)
    assert ai is not None
    assert ai.members_used == 1
    assert len(ai.ring) >= 4

    pinned = fetch_model_tracks("AL092026", init_cycle="2026100706")
    assert "AP01" not in {t.tech_id for t in pinned}
    assert "GDMI" in {t.tech_id for t in pinned}


def test_single_ai_track_draws_a_corridor() -> None:
    track = ModelTrack(
        tech_id="GDMI", label="GraphCast", family="ai",
        init_cycle="2026-10-07T06Z",
        fixes=[
            ModelFix(hours_out=h, lat=22.0, lon=-94.0 - h * 0.05, wind_kt=35, pressure_mb=1004)
            for h in (12, 24, 36, 48, 72)
        ],
    )
    env = build_envelope([track], include_families=frozenset({"ai"}), min_members=1)
    assert env is not None
    assert env.ring[0] == env.ring[-1]
    assert len(env.ring) >= 4


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Body":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def test_adeck_failure_is_not_cached(monkeypatch) -> None:
    atcf_adecks._ADECK_CACHE.clear()
    state = {"mode": "fail", "n": 0}

    def urlopen(*_args: object, **_kwargs: object) -> _Body:
        state["n"] += 1
        if state["mode"] == "fail":
            raise urllib.error.URLError("down")
        return _Body(gzip.compress(b"adeck-bytes"))

    monkeypatch.setattr(atcf_adecks.urllib.request, "urlopen", urlopen)
    assert atcf_adecks._download_adeck("al", 9, 2026) is None
    assert atcf_adecks._download_adeck("al", 9, 2026) is None
    assert state["n"] == 2
    assert atcf_adecks._ADECK_CACHE.get(("al", 9, 2026)) is None

    state["mode"] = "ok"
    payload = atcf_adecks._download_adeck("al", 9, 2026)
    assert payload == b"adeck-bytes"
    cached_calls = state["n"]
    assert atcf_adecks._download_adeck("al", 9, 2026) == payload
    assert state["n"] == cached_calls
    assert atcf_adecks._download_adeck("al", 9, 2026, refresh=True) == payload
    assert state["n"] == cached_calls + 1


def test_null_wind_chunk_is_not_cached(monkeypatch) -> None:
    wind_forecast._bulk_cache.clear()
    calls = {"n": 0}
    null = json.dumps({
        "hourly": {
            "time": ["2026-10-07T12:00"],
            "wind_speed_10m": [None],
            "wind_direction_10m": [None],
        },
    }).encode()
    real = json.dumps({
        "hourly": {
            "time": ["2026-10-07T12:00"],
            "wind_speed_10m": [8.0],
            "wind_direction_10m": [180],
        },
    }).encode()

    def urlopen(*_args: object, **_kwargs: object) -> _Body:
        calls["n"] += 1
        return _Body(null if calls["n"] == 1 else real)

    monkeypatch.setattr(wind_forecast.urllib.request, "urlopen", urlopen)
    wind_forecast._fetch_bulk_chunk("22.000", "-94.000", "gfs_seamless")
    assert calls["n"] == 1
    wind_forecast._fetch_bulk_chunk("22.000", "-94.000", "gfs_seamless")
    assert calls["n"] == 2
    wind_forecast._fetch_bulk_chunk("22.000", "-94.000", "gfs_seamless")
    assert calls["n"] == 2
    wind_forecast._fetch_bulk_chunk("22.000", "-94.000", "gfs_seamless", refresh=True)
    assert calls["n"] == 3


_RADII_KML = b"""<?xml version="1.0"?>
<kml xmlns='http://www.opengis.net/kml/2.2'>
<Document>
<Placemark><name>64</name>
<Polygon><outerBoundaryIs><LinearRing><coordinates>
-94.2,21.2,0 -93.8,21.2,0 -93.8,21.8,0 -94.2,21.8,0
</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark>
<Placemark><name>34</name>
<Polygon><outerBoundaryIs><LinearRing><coordinates>
-95,21,0 -93,21,0 -93,23,0 -95,23,0 -95,21,0
</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark>
<Placemark><name>notes</name>
<Polygon><outerBoundaryIs><LinearRing><coordinates>
0,0,0 1,0,0 1,1,0 0,0,0
</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark>
</Document></kml>
"""


def test_parse_wind_radii_kml_keeps_34_50_64_only() -> None:
    parsed = parse_wind_radii_kml(_RADII_KML)
    assert [wind for wind, _ring in parsed] == [34, 64]
    for _wind, ring in parsed:
        assert ring[0] == ring[-1]
        assert len(ring) >= 4


def test_prior_track_timeout_does_not_raise(monkeypatch) -> None:
    def _slow(_url: str, *, refresh: bool = False) -> list:
        time.sleep(0.4)
        return []

    monkeypatch.setattr(nhc_gis, "fetch_forecast_track", _slow)
    out = fetch_prior_forecast_tracks(
        "AL092026", "003", n_prior=1, timeout_s=0.05,
    )
    assert out == []


def test_gather_keeps_a_finished_job_when_an_earlier_one_hogs_the_deadline() -> None:
    def slow() -> str:
        time.sleep(0.4)
        return "slow"

    def fast() -> str:
        return "fast"

    out = live_api._gather(
        {
            "slow": (slow, "default-slow"),
            "fast": (fast, "default-fast"),
        },
        0.1,
    )
    assert out["slow"] == "default-slow"
    assert out["fast"] == "fast"
