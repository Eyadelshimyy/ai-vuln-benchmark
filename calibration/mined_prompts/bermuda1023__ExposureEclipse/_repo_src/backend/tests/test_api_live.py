"""Smoke tests for the live + replay hurricane endpoints. Hits real NOAA /
NWS / NHC endpoints over the network — these are integration-flavoured."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import live_hurricane

client = TestClient(app)


_FAKE_NHC_STORM = {
    "id": "AL022026",
    "binNumber": "AT2",
    "name": "TESTSTORM",
    "classification": "HU",
    "intensity": "95",
    "pressure": "965",
    "latitudeNumeric": 25.4,
    "longitudeNumeric": -78.3,
    "latitude": "25.4N",
    "longitude": "78.3W",
    "movementDir": 315,
    "movementSpeed": 12,
    "lastUpdate": "2026-09-15T15:00:00Z",
}


def test_storm_list_has_no_replay_candidates() -> None:
    r = client.get("/api/live/storms")
    assert r.status_code == 200
    body = r.json()
    assert body["replay"] == []
    assert "invests" not in body
    if body["hasActive"] is False:
        assert body["note"]


def test_replay_bundle_returns_observed_and_forecasts() -> None:
    r = client.get(
        "/api/live/storms/AL092022",
        params={"includeObs": "false", "includeAlerts": "false", "includeLand": "false", "includeSst": "false"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["storm"]["name"].upper() == "IAN"
    assert len(b["observedTrack"]) > 5
    assert len(b["forecasts"]) >= 1
    # Latest advisory should have the highest advisory_number.
    advs = sorted(b["forecasts"], key=lambda a: -a["advisoryNumber"])
    assert advs[0]["advisoryNumber"] == max(a["advisoryNumber"] for a in b["forecasts"])
    assert b["bbox"][0] < b["bbox"][2]  # west < east
    assert b["bbox"][1] < b["bbox"][3]  # south < north
    # Replay storms have no live hunter mission.
    assert b["recon"] == []
    assert b["vortex"] is None


def test_replay_bundle_unknown_storm_404() -> None:
    r = client.get("/api/live/storms/AL999999", params={"includeObs": "false", "includeAlerts": "false", "includeLand": "false", "includeSst": "false"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_replay_bundle_sst_layer_bounded() -> None:
    r = client.get(
        "/api/live/storms/AL092022",
        params={"includeObs": "false", "includeAlerts": "false", "includeLand": "false", "includeSst": "true"},
    )
    assert r.status_code == 200
    b = r.json()
    # SST grid is bounded to the storm bbox — even a basin-wide replay stays
    # under 10k cells thanks to the adaptive step (0.25° / 0.5° / 1.0°).
    assert 0 < len(b["sst"]) < 10000
    assert b["sstMinC"] is not None and b["sstMaxC"] is not None
    assert b["sstMinC"] <= b["sstMaxC"]


def test_live_bundle_builds_from_nhc_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A storm currently in NHC's CurrentStorms.json must resolve via the live
    path, not fall through to the replay-set 404. This is the regression case
    for AL022026 → 'not found in IBTrACS replay set.'"""
    monkeypatch.setattr(
        live_hurricane,
        "_fetch_current_storms_raw",
        lambda *, refresh=False: {"activeStorms": [_FAKE_NHC_STORM]},
    )
    r = client.get(
        "/api/live/storms/AL022026",
        params={
            "includeObs": "false",
            "includeAlerts": "false",
            "includeLand": "false",
            "includeSst": "false",
            "includeSurge": "false",
            "includeWindMap": "false",
        },
    )
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["storm"]["stormId"] == "AL022026"
    assert b["storm"]["isLive"] is True
    assert b["storm"]["year"] == 2026
    # Observed track = 2 back-projected + 1 current fix.
    assert len(b["observedTrack"]) == 3
    # Fake NHC entry has no forecastTrack KMZ → the motion-vector fallback
    # produces one 8-anchor advisory. Real live storms may also carry prior
    # advisories (up to 5 total); the mock does not exercise that path.
    assert len(b["forecasts"]) == 1
    assert len(b["forecasts"][0]["points"]) == 8
    # Wind cones built off the observed fixes.
    assert len(b["observedWindField"]["outerRings"]) >= 1


@pytest.mark.parametrize("flag", ["includeObs", "includeAlerts", "includeLand", "includeSst"])
def test_replay_bundle_individual_layer_toggles(flag: str) -> None:
    # Each layer can be turned off without affecting the rest.
    off = {k: "false" for k in ("includeObs", "includeAlerts", "includeLand", "includeSst")}
    off[flag] = "false"
    other_on = {k: ("true" if k != flag else "false") for k in off}
    r = client.get("/api/live/storms/AL092022", params=other_on)
    assert r.status_code == 200
