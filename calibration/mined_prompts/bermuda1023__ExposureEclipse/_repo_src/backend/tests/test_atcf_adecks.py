"""Unit tests for the ATCF a-deck parser + ensemble envelope + endpoint.

The parser is exercised with a hand-rolled fixture a-deck instead of a real
network fetch; that keeps tests deterministic and fast. The endpoint is
smoke-tested by monkeypatching _download_adeck to return the fixture bytes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import atcf_adecks
from app.services.atcf_adecks import (
    MODEL_FAMILY,
    ModelFix,
    ModelTrack,
    fetch_model_tracks,
    list_available_cycles,
)
from app.services.ensemble_envelope import build_envelope


# ─────────────────────────── fixture a-deck ───────────────────────────


_INIT_CYCLE = "2024092612"


def _adeck_row(tech: str, tau: int, lat_tenths: int, lon_tenths: int,
               wind: int = 65, pres: int = 990) -> str:
    return (
        f"AL, 09, {_INIT_CYCLE}, 03, {tech:>4}, {tau:>3}, "
        f"{lat_tenths:>3}N, {lon_tenths:>4}W, {wind:>3}, {pres:>4}, HU\n"
    )


def _make_adeck() -> bytes:
    """Build a synthetic a-deck that exercises every family bucket."""
    rows: list[str] = []
    # NHC official — tightly clustered, ranked highest
    rows += [_adeck_row("OFCL", tau, 245 + tau // 6, 830 + tau)
             for tau in (12, 24, 36, 48, 72, 96, 120)]
    # GFS deterministic
    rows += [_adeck_row("AVNO", tau, 246 + tau // 6, 832 + tau)
             for tau in (12, 24, 36, 48, 72, 96, 120)]
    # ECMWF-HRES
    rows += [_adeck_row("ECMF", tau, 244 + tau // 6, 828 + tau)
             for tau in (12, 24, 36, 48, 72, 96, 120)]
    # GEFS 10 members + control
    for member in ["AC00"] + [f"AP{i:02d}" for i in range(1, 11)]:
        # Perturb slightly per member so the envelope has area
        offset = int(member[-2:]) if member != "AC00" else 0
        rows += [_adeck_row(member, tau,
                            245 + tau // 6 + offset % 3,
                            830 + tau + offset)
                 for tau in (12, 24, 36, 48, 72, 96, 120)]
    # ECMWF-ENS 5 members + mean
    for member in ["EEMN"] + [f"EE{i:02d}" for i in range(1, 6)]:
        offset = int(member[-2:]) if member != "EEMN" else 0
        rows += [_adeck_row(member, tau,
                            243 + tau // 6 + offset % 2,
                            828 + tau + offset)
                 for tau in (12, 24, 36, 48, 72, 96, 120)]
    # AI models
    for ai in ("GRAP", "GENC", "AIFS", "FNV3", "PANG"):
        rows += [_adeck_row(ai, tau, 245 + tau // 6, 831 + tau)
                 for tau in (12, 24, 36, 48, 72, 96, 120)]
    # Regional
    rows += [_adeck_row("HWRF", tau, 246 + tau // 6, 829 + tau)
             for tau in (12, 24, 36, 48, 72, 96, 120)]
    # Baseline (default off)
    rows += [_adeck_row("CLIP", tau, 240 + tau // 6, 825 + tau)
             for tau in (12, 24, 36, 48, 72, 96, 120)]

    # Also inject a second (older) init cycle so the "latest cycle" filter is
    # non-trivial.
    older = "2024092606"
    rows.append(
        f"AL, 09, {older}, 03, OFCL,  12, 245N,  825W,  60,  990, HU\n"
    )

    # _download_adeck returns the DECOMPRESSED payload; the monkeypatch
    # replaces the whole function, so we return uncompressed bytes here.
    return "".join(rows).encode("ascii")


@pytest.fixture(autouse=True)
def _patch_download(monkeypatch: pytest.MonkeyPatch) -> None:
    # Also clear the lru_cache so tests are independent.
    atcf_adecks._download_adeck.cache_clear()
    monkeypatch.setattr(
        atcf_adecks, "_download_adeck",
        lambda basin, cy, year, *, refresh=False: _make_adeck(),
    )


# ─────────────────────────── parser ───────────────────────────


def test_parser_picks_latest_cycle_by_default() -> None:
    tracks = fetch_model_tracks("AL092024")
    assert tracks
    # Every track's init_cycle is the LATEST — the older 2024092606 row must
    # be filtered out.
    for t in tracks:
        assert t.init_cycle == "2024-09-26T12Z"


def test_parser_groups_by_tech_and_sorts_by_lead() -> None:
    tracks = fetch_model_tracks("AL092024")
    by_tech = {t.tech_id: t for t in tracks}
    ofcl = by_tech["OFCL"]
    assert [f.hours_out for f in ofcl.fixes] == [12, 24, 36, 48, 72, 96, 120]
    # Lat parsing: fixture stores tenths — "247N" (tau=12: 245 + 2) → 24.7.
    assert ofcl.fixes[0].lat == pytest.approx((245 + 12 // 6) / 10.0)
    # Lon parsing: "8XXW" → -8X.X (all fixture longitudes are western)
    assert ofcl.fixes[0].lon < 0


def test_parser_assigns_families_correctly() -> None:
    tracks = fetch_model_tracks("AL092024")
    fam_by_tech = {t.tech_id: t.family for t in tracks}
    assert fam_by_tech["OFCL"] == "official"
    assert fam_by_tech["AVNO"] == "gfs_det"
    assert fam_by_tech["ECMF"] == "ecmwf_det"
    assert fam_by_tech["AC00"] == "gefs_ens"
    assert fam_by_tech["AP01"] == "gefs_ens"
    assert fam_by_tech["EEMN"] == "ecmwf_mean"
    assert fam_by_tech["EE01"] == "ecmwf_ens"
    assert fam_by_tech["GRAP"] == "ai"
    assert fam_by_tech["GENC"] == "ai"
    assert fam_by_tech["AIFS"] == "ai"
    assert fam_by_tech["FNV3"] == "ai"
    assert fam_by_tech["PANG"] == "ai"
    assert fam_by_tech["HWRF"] == "regional"


def test_parser_excludes_baselines_by_default() -> None:
    tracks = fetch_model_tracks("AL092024")
    techs = {t.tech_id for t in tracks}
    assert "CLIP" not in techs
    # Opt-in via include_baselines=True should surface them.
    with_baselines = fetch_model_tracks("AL092024", include_baselines=True)
    assert "CLIP" in {t.tech_id for t in with_baselines}


def test_list_available_cycles_orders_newest_first() -> None:
    cycles = list_available_cycles("AL092024")
    assert cycles[0] == "2024-09-26T12Z"
    assert cycles[-1] == "2024-09-26T06Z"


# ─────────────────────────── envelope ───────────────────────────


def test_envelope_returns_closed_ring_across_ensemble_members() -> None:
    tracks = fetch_model_tracks("AL092024")
    env = build_envelope(tracks)
    assert env is not None
    # 11 GEFS + 6 ECMWF-ENS + 5 AI = 22 members that qualify.
    assert env.members_used >= 15
    # Ring is closed.
    assert env.ring[0] == env.ring[-1]
    # Non-degenerate polygon.
    assert len(env.ring) >= 4


def test_ai_only_envelope_uses_ai_bucket() -> None:
    tracks = fetch_model_tracks("AL092024")
    ai_env = build_envelope(
        tracks, include_families=frozenset({"ai"}), min_members=2,
    )
    assert ai_env is not None
    assert ai_env.members_used == 5   # GRAP, GENC, AIFS, FNV3, PANG


def test_envelope_returns_none_below_min_members() -> None:
    # A track list with only OFCL isn't an ensemble.
    tracks = [
        ModelTrack(
            tech_id="OFCL", label="NHC Official", family="official",
            init_cycle="2024-09-26T12Z",
            fixes=[ModelFix(hours_out=h, lat=25.0, lon=-83.0, wind_kt=65, pressure_mb=990)
                   for h in (12, 24, 36)],
        )
    ]
    assert build_envelope(tracks) is None


# ─────────────────────────── endpoint ───────────────────────────


def test_model_tracks_endpoint_returns_all_families() -> None:
    client = TestClient(app)
    r = client.get("/api/live/storms/AL092024/model-tracks")
    assert r.status_code == 200
    body = r.json()
    assert body["initCycle"] == "2024-09-26T12Z"
    assert body["ensembleEnvelope"] is not None
    assert body["aiEnvelope"] is not None
    families = {f["family"] for f in body["families"]}
    # Every family bucket present in the fixture must surface.
    assert families >= {"official", "ai", "gfs_det", "gefs_ens", "ecmwf_ens"}


def test_model_tracks_endpoint_include_baselines_toggle() -> None:
    client = TestClient(app)
    r = client.get(
        "/api/live/storms/AL092024/model-tracks",
        params={"includeBaselines": "true"},
    )
    body = r.json()
    techs = {t["techId"] for t in body["tracks"]}
    assert "CLIP" in techs


def test_model_tracks_endpoint_handles_bad_atcf_id() -> None:
    client = TestClient(app)
    r = client.get("/api/live/storms/not-real-id/model-tracks")
    assert r.status_code == 200  # graceful — endpoint returns empty
    body = r.json()
    assert body["tracks"] == []
    assert body["ensembleEnvelope"] is None
    assert body["notes"]  # explanation surfaced


def test_parser_reads_wind_radii_and_rmw() -> None:
    from app.services.atcf_adecks import _parse_lines

    line = (
        "AL, 09, 2024092612, 03, OFCL,  12, 245N,  830W,  90,  960, HU, "
        "64, NEQ,  40,  30,  25,  35, 1012, 180,  15,  110,\n"
    )
    rows = list(_parse_lines(line.encode("ascii")))
    assert len(rows) == 1
    r = rows[0]
    assert r["rad"] == 64
    assert r["quads"] == (40, 30, 25, 35)
    assert r["rmw_nm"] == 15
    assert r["ty"] == "HU"


def test_fetch_official_fixes_merges_rad_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services.atcf_adecks import fetch_official_fixes

    payload = (
        "AL, 09, 2024092612, 03, OFCL,   0, 245N,  830W, 100,  950, HU, "
        "34, NEQ, 120, 100,  80, 110, 1012, 200,  18,  120,\n"
        "AL, 09, 2024092612, 03, OFCL,   0, 245N,  830W, 100,  950, HU, "
        "64, NEQ,  45,  35,  30,  40, 1012, 200,  18,  120,\n"
        "AL, 09, 2024092612, 03, OFCL,  24, 255N,  820W,  85,  965, HU, "
        "64, NEQ,  50,  40,  20,  35, 1012, 180,  20,  100,\n"
    ).encode("ascii")
    monkeypatch.setattr(
        atcf_adecks, "_download_adeck",
        lambda basin, cy, year, *, refresh=False: payload,
    )
    fixes = fetch_official_fixes("AL092024")
    by_tau = {f.hours_out: f for f in fixes}
    assert 0 in by_tau and 24 in by_tau
    now = by_tau[0]
    assert now.rmw_nm == 18
    assert now.r34_quads == (120, 100, 80, 110)
    assert now.r64_quads == (45, 35, 30, 40)
    assert now.wind_kt == 100
