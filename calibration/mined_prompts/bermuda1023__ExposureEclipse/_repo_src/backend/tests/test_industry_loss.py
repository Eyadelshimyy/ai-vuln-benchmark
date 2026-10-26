"""Industry loss: supplied actuals + TIV loaders, event loss, curve fit, API."""

from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import damage_curves, event_evidence, event_loss, industry_losses, onlevel
from app.services.county_wind import SpeedBin
from app.services.curve_fit import (
    FitOptions,
    Observation,
    _pav_non_decreasing,
    fit_curve,
    fit_with_holdout,
    residuals,
)
from app.services.damage_curves import BAND_MPH_LO, get_curve_set, vendor_curve
from app.services.event_loss import lognormal_band
from app.services.hurricane_impact import CountyImpact
from app.services.ibtracs import Storm
from app.services.industry_losses import (
    IndustryDataError,
    IndustryTiv,
    normalize_geography,
)
from app.services.loss_calibration import match_event

client = TestClient(app)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(industry_losses, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(damage_curves, "data_dir", lambda: tmp_path)
    industry_losses.clear_caches()
    onlevel.clear_caches()
    event_evidence.clear_caches()
    damage_curves.load_curve_sets.cache_clear()
    yield tmp_path
    industry_losses.clear_caches()
    onlevel.clear_caches()
    damage_curves.load_curve_sets.cache_clear()


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _impact(gid: str, bins: list[tuple[int, float]], wind: int = 100) -> CountyImpact:
    usps, geoid = gid.split("-")[1], gid.split("-")[2]
    return CountyImpact(
        geoid=geoid,
        geography_id=gid,
        name=f"County {geoid}",
        state_usps=usps,
        centroid_lat=27.0,
        centroid_lon=-82.0,
        max_wind_kt=wind,
        max_category=3,
        closest_distance_nm=10.0,
        rmax_at_closest_nm=20.0,
        rmax_source="ibtracs",
        tiv=0.0,
        location_count=0,
        has_data=False,
        speed_bins=[
            SpeedBin(mph_lo=lo, mph_hi=lo + 9, category=3, area_fraction=f, max_wind_kt=wind)
            for lo, f in bins
        ],
    )


# ─────────────────────────────── loaders ───────────────────────────────


def test_normalize_geography_variants() -> None:
    assert normalize_geography("US-FL-12086").geography_id == "US-FL-12086"
    assert normalize_geography("12086").geography_id == "US-FL-12086"
    assert normalize_geography("1001").geography_id == "US-AL-01001"
    assert normalize_geography("US-FL").level == "STATE"
    assert normalize_geography("fl").geography_id == "US-FL"
    assert normalize_geography("US-TX-12086") is None
    assert normalize_geography("nowhere") is None


def test_actuals_load_and_totals(data_dir) -> None:
    _write(
        data_dir / "industry_losses" / "ian.csv",
        """
        # comment lines are ignored
        stormId,geographyId,segment,loss,currency,asOfDate,source
        AL092022,US-FL-12071,RESIDENTIAL,"8,000,000,000",USD,2024-06-30,PCS
        AL092022,12071,COMMERCIAL,4000000000,USD,,
        AL092022,US-FL-12015,AUTO,500000000,USD,,
        AL092022,US-SC,RESIDENTIAL,100000000,USD,,
        AL092022,US-FL,RESIDENTIAL,999,USD,,
        """,
    )
    ev = industry_losses.actuals_for("al092022")
    assert ev is not None
    assert ev.source == "PCS" and ev.as_of == "2024-06-30"
    seg = ev.segment_totals()
    # FL residential has county rows, so the FL state row is not double counted.
    assert seg["RESIDENTIAL"] == 8_000_000_000 + 100_000_000
    assert seg["COMMERCIAL"] == 4_000_000_000
    assert seg["AUTO"] == 500_000_000
    assert industry_losses.actuals_for("AL012000") is None


def test_actuals_unallocated_claims_and_basis_flags(data_dir) -> None:
    _write(
        data_dir / "industry_losses" / "pcs.csv",
        """
        stormId,geographyId,segment,loss,claims,currency,source,includesFlood,includesSurge,includesNfip,includesLae,unallocated
        AL112017,US-FL-12087,RESIDENTIAL,1000,10,USD,PCS 1744,true,true,false,false,
        AL112017,US-VI-78010,COMMERCIAL,500,5,USD,PCS 1744,true,true,false,false,
        AL112017,US-PR-72127,AUTO,0,0,USD,PCS 1744,true,true,false,false,
        AL112017,US-FL,RESIDENTIAL,200,2,USD,PCS 1744,true,true,false,false,true
        """,
    )
    ev = industry_losses.actuals_for("AL112017")
    # Unallocated adds even though FL residential has county rows.
    assert ev.segment_totals() == {"RESIDENTIAL": 1200, "COMMERCIAL": 500, "AUTO": 0}
    assert ev.unallocated_total() == 200
    assert ev.claims == {"RESIDENTIAL": 12, "COMMERCIAL": 5, "AUTO": 0}
    assert ev.states() == ["FL", "PR", "VI"]
    assert (ev.includes_flood, ev.includes_nfip, ev.includes_lae) == (True, False, False)
    assert ("US-PR-72127", "AUTO") in ev.county  # a reported zero is kept as a known zero


def test_actuals_reject_mixed_fx_conversions(data_dir) -> None:
    _write(
        data_dir / "industry_losses" / "pcs.csv",
        """
        stormId,geographyId,segment,loss,currency,reportedLoss,reportedCurrency,fxRate
        AL142024,US-FL-12057,RESIDENTIAL,1325,USD,1000,GBP,1.325
        AL142024,US-FL-12103,RESIDENTIAL,1300,USD,1000,GBP,1.30
        """,
    )
    with pytest.raises(IndustryDataError, match="mixes FX"):
        industry_losses.load_actuals()


def test_unallocated_row_must_be_state(data_dir) -> None:
    _write(
        data_dir / "industry_losses" / "bad.csv",
        "stormId,geographyId,segment,loss,currency,unallocated\nAL092022,US-FL-12071,AUTO,1,USD,true",
    )
    with pytest.raises(IndustryDataError, match="unallocated rows must use a state"):
        industry_losses.load_actuals()


def test_actuals_reject_bad_segment(data_dir) -> None:
    _write(
        data_dir / "industry_losses" / "bad.csv",
        "stormId,geographyId,segment,loss,currency\nAL092022,US-FL-12071,MARINE,1,USD",
    )
    with pytest.raises(IndustryDataError, match="bad.csv:2"):
        industry_losses.load_actuals()


def test_tiv_vintage_selection(data_dir) -> None:
    body = "geographyId,segment,tiv,currency\nUS-FL-12071,RESIDENTIAL,100,USD"
    _write(data_dir / "industry_tiv" / "2018.csv", body)
    _write(data_dir / "industry_tiv" / "2022.csv", body)
    assert industry_losses.tiv_for_year(2022).year == 2022
    assert industry_losses.tiv_for_year(2020).year == 2018
    assert industry_losses.tiv_for_year(2017).year == 2018
    assert industry_losses.tiv_for_year(2026).year == 2022
    assert industry_losses.tiv_for_year(2022).kind == "supplied"


def test_tiv_falls_back_to_book_when_nothing_supplied(data_dir) -> None:
    _write(
        data_dir / "industry_exposure_counties.csv",
        "geographyId,residentialTiv,commercialTiv,autoTiv\nUS-FL-12071,10,5,2",
    )
    tiv = industry_losses.tiv_for_year(2022)
    assert tiv.kind == "book"
    assert tiv.county["US-FL-12071"] == {"RESIDENTIAL": 10.0, "COMMERCIAL": 5.0, "AUTO": 2.0}
    assert tiv.segments_present() == {"RESIDENTIAL", "COMMERCIAL", "AUTO"}


def test_panel_band_reports_honest_full_event_coverage() -> None:
    """The displayed range is checked against full PCS totals, not fit rows."""
    curve = get_curve_set("calibrated-20261020")
    band = curve.fit["panelBand"]
    trained = json.loads(
        (
            industry_losses.data_dir()
            / "damage_assumptions"
            / "sshws_calibrated.json"
        ).read_text(encoding="utf-8")
    )
    errors = trained["segments"]["ALL"]["eventLogErrorHoldout"]
    events = {event["stormId"]: event for event in trained["events"]}
    factors = {
        storm_id: event["actualTotal"] / (event["inField"] * math.exp(errors[storm_id]))
        for storm_id, event in events.items()
        if storm_id in errors and event["inField"] > 0
    }
    covered = {
        storm_id
        for storm_id, factor in factors.items()
        if band["lowFactor"] <= factor <= band["highFactor"]
    }
    assert len(factors) == band["scorableEvents"] == 14
    assert len(events) == band["suppliedEvents"] == 15
    assert len(covered) == band["coveredEvents"] == 12
    assert set(factors) - covered == set(band["outsideEvents"]) == {
        "AL042024",
        "AL282020",
    }


# ─────────────────────────────── event loss ───────────────────────────────


def _plain(curve_id: str = "kcc"):
    """Copy of a built-in set with no land factor and no deductible."""
    base = get_curve_set(curve_id)
    return damage_curves.CurveSet(
        id="plain", label="plain", kind="vendor",
        segments={k: list(v) for k, v in base.segments.items()},
        wind_factor=1.0, deductible={},
    )


def test_event_loss_sums_bins_by_segment(data_dir) -> None:
    curve = _plain()
    imp = _impact("US-FL-12071", [(101, 0.5), (121, 0.25)])
    tiv = IndustryTiv(
        kind="supplied", year=2022, currency="USD",
        county={"US-FL-12071": {"RESIDENTIAL": 1_000.0, "COMMERCIAL": 400.0, "AUTO": 100.0}},
    )
    res = event_loss.event_loss([imp], tiv, curve)
    k = vendor_curve("kcc")
    # Band midpoints 105.5 and 125.5 mph sit exactly on curve nodes 6 and 8.
    dr = 0.5 * k[6] + 0.25 * k[8]
    assert res.by_segment["RESIDENTIAL"] == pytest.approx(1_000 * dr)
    assert res.by_segment["COMMERCIAL"] == pytest.approx(400 * dr)
    assert res.by_segment["AUTO"] == pytest.approx(100 * dr * 0.15)
    half = event_loss.event_loss([imp], tiv, curve, {"12071": 0.5})
    assert half.total == pytest.approx(res.total * 0.5)


def test_wind_factor_moves_marine_wind_onto_land_curve() -> None:
    curve = _plain()
    curve.wind_factor = 0.8
    k = vendor_curve("kcc")
    # 125.5 mph marine x 0.8 = 100.4 mph over land, between nodes 95.5 and 105.5.
    t = (100.4 - 95.5) / 10
    assert curve.ground_up_dr("RESIDENTIAL", 125.5) == pytest.approx(k[5] * (1 - t) + k[6] * t)
    assert curve.ground_up_dr("RESIDENTIAL", 45.0) == 0.0  # 36 mph over land: no damage


def test_deductible_turns_ground_up_into_insured() -> None:
    assert damage_curves.insured_dr(0.3, 0.0) == 0.3
    high = damage_curves.insured_dr(0.5, 0.013)
    assert 0.48 < high < 0.5
    low = damage_curves.insured_dr(0.005, 0.013)
    assert 0 < low < 0.005  # some locations still exceed the deductible
    assert damage_curves.insured_dr(0.02, 0.03) < damage_curves.insured_dr(0.02, 0.013)
    kcc = get_curve_set("kcc")
    assert kcc.dr("RESIDENTIAL", 150.0) < kcc.ground_up_dr("RESIDENTIAL", 150.0)


def test_event_loss_reports_counties_without_tiv() -> None:
    kcc = get_curve_set("kcc")
    tiv = IndustryTiv(kind="supplied", year=2022, currency="USD", county={})
    res = event_loss.event_loss([_impact("US-FL-12071", [(101, 1.0)])], tiv, kcc)
    assert res.missing_tiv == ["US-FL-12071"]
    assert set(res.segments_without_tiv) == {"RESIDENTIAL", "COMMERCIAL", "AUTO"}


def test_lognormal_band_keeps_mean() -> None:
    band = lognormal_band(100.0, 0.5)
    assert band["p10"] < band["p50"] < band["mean"] < band["p90"]
    assert band["p50"] == pytest.approx(100 * math.exp(-0.125))
    assert lognormal_band(0.0, 0.5)["p90"] == 0.0


# ─────────────────────────────── curve fit ───────────────────────────────


def test_pav_is_non_decreasing() -> None:
    assert _pav_non_decreasing([1, 3, 2, 4]) == [1, 2.5, 2.5, 4]
    assert _pav_non_decreasing([5, 4, 3]) == [4, 4, 4]


def _synthetic(truth: list[float]) -> list[Observation]:
    obs = []
    for e, event in enumerate(("EV1", "EV2", "EV3")):
        for i in range(40):
            hi = 3 + (i + 3 * e) % 9  # bands 71..151 mph
            lo = hi - 2
            bins = [(lo, 0.3), (hi - 1, 0.3), (hi, 0.2)]
            tiv = 1e9 * (1 + (i % 7))
            actual = tiv * sum(f * truth[b] for b, f in bins)
            obs.append(Observation(event, f"US-FL-{12000 + i:05d}", tiv, bins, actual))
    return obs


def test_fit_recovers_scaled_curve() -> None:
    prior = vendor_curve("kcc")
    truth = [v * 0.6 for v in prior]
    obs = _synthetic(truth)
    res = fit_curve(obs, prior)
    assert all(b >= a for a, b in zip(res.curve, res.curve[1:]))
    for ev, err in res.event_errors.items():
        assert abs(err) < 0.05, ev
    assert res.county_sigma is not None and res.county_sigma < 0.1
    # Bands with plenty of data land near the truth.
    for b in range(3, 11):
        assert res.curve[b] == pytest.approx(truth[b], rel=0.2)


def test_fit_with_holdout_scores_each_event() -> None:
    prior = vendor_curve("kcc")
    obs = _synthetic([v * 1.5 for v in prior])
    res = fit_with_holdout(obs, prior)
    assert set(res.oos_event_errors) == {"EV1", "EV2", "EV3"}
    assert all(abs(e) < 0.15 for e in res.oos_event_errors.values())


def test_below_materiality_rows_are_upper_bounds() -> None:
    prior = vendor_curve("kcc")
    truth = [v * 0.6 for v in prior]
    clean = _synthetic(truth)
    # Token PCS allocations in barely touched counties: $5k on $100M of TIV
    # in the 41-60 mph bands, where the true loss is about $66k-$156k.
    def noise(upper: float | None) -> list[Observation]:
        return [
            Observation(f"EV{1 + i % 3}", f"US-GA-{13000 + i:05d}", 1e8, [(i % 2, 1.0)], 5_000.0,
                        upper_bound=upper)
            for i in range(60)
        ]

    exact = fit_curve(clean + noise(None), prior)
    censored = fit_curve(clean + noise(250_000.0), prior)
    # Read as exact, the noise implies a 0.005% DR and drags the low bands down.
    assert exact.curve[0] < 0.6 * censored.curve[0]
    assert censored.curve[0] >= 0.5 * truth[0]
    for b in range(3, 11):
        assert censored.curve[b] == pytest.approx(truth[b], rel=0.2)
    # A model under the ceiling costs nothing.
    o = Observation("EV", "US-FL-1", 1e9, [(5, 1.0)], 5_000.0, upper_bound=250_000.0)
    assert residuals([o], [0.0001] * len(prior))[0] == []


def test_match_event_censors_small_rows_and_low_claim_counts() -> None:
    impact = _impact("US-FL-12071", [(131, 1.0)])
    tiv = IndustryTiv(
        kind="supplied", year=2022, currency="USD",
        county={"US-FL-12071": {"RESIDENTIAL": 1e10, "COMMERCIAL": 5e9, "AUTO": 1e9}},
    )
    actuals = industry_losses.EventActuals(
        storm_id="AL092022", currency="USD",
        county={
            ("US-FL-12071", "RESIDENTIAL"): 2e9,
            ("US-FL-12071", "COMMERCIAL"): 40_000.0,
            ("US-FL-12071", "AUTO"): 900_000.0,
        },
        county_claims={("US-FL-12071", "AUTO"): 12.0},
    )
    m = match_event(actuals, [impact], tiv, 0.8)
    assert m.observations["RESIDENTIAL"][0].upper_bound is None
    assert m.observations["COMMERCIAL"][0].upper_bound == 250_000.0
    assert m.observations["AUTO"][0].upper_bound == 900_000.0


def test_fit_without_data_returns_prior() -> None:
    prior = vendor_curve("kcc")
    assert fit_curve([], prior).curve == prior


def test_fixed_band_holds_and_the_rest_refit_around_it() -> None:
    prior = vendor_curve("kcc")
    truth = [v * 0.6 for v in prior]
    obs = _synthetic(truth)
    res = fit_with_holdout(obs, prior, FitOptions(fixed=((0, 0.0004),)))
    assert res.curve[0] == pytest.approx(0.0004)
    assert all(c[0] == pytest.approx(0.0004) for c in res.oos_curves.values())
    assert set(res.oos_curves) == {"EV1", "EV2", "EV3"}
    assert all(b >= a for a, b in zip(res.curve, res.curve[1:]))
    for b in range(3, 11):
        assert res.curve[b] == pytest.approx(truth[b], rel=0.2)


def test_unsupported_tail_follows_the_prior_shape() -> None:
    prior = vendor_curve("kcc")
    truth = [v * 1.5 for v in prior]
    # Exposure only in the bottom eight bands.
    obs = [o for o in _synthetic(truth) if all(b < 8 for b, _ in o.bins)]
    res = fit_curve(obs, prior, FitOptions(smooth=2.0, tail_min_weight=5e-4))
    last = max(b for b, w in enumerate(res.band_weight) if w >= 5e-4)
    assert last < len(prior) - 1
    scale = res.curve[last] / prior[last]
    for b in range(last + 1, len(prior)):
        assert res.curve[b] == pytest.approx(min(prior[b] * scale, 1.0), rel=1e-6)


def test_calibrated_set_becomes_default(data_dir) -> None:
    assert damage_curves.default_curve_set_id() == "kcc"
    payload = {
        "id": "calibrated-test",
        "kind": "calibrated",
        "createdAt": "2026-10-13T00:00:00+00:00",
        "segments": {"RESIDENTIAL": [0.01] * len(BAND_MPH_LO)},
        "sigmaLogCounty": 0.4,
        "sigmaLogEvent": 0.2,
    }
    _write(data_dir / "damage_curves" / "calibrated-test.json", json.dumps(payload))
    damage_curves.load_curve_sets.cache_clear()
    assert damage_curves.default_curve_set_id() == "calibrated-test"
    cs = get_curve_set(None)
    assert cs.sigma_log_event == 0.2
    # Segments missing from the file fall back to the insured KCC shape.
    assert cs.segments["AUTO"] == damage_curves.insured_prior(get_curve_set("kcc"), "AUTO")
    assert cs.wind_factor == damage_curves.DEFAULT_WIND_FACTOR


# ─────────────────────────────── API ───────────────────────────────


@pytest.fixture
def fake_storm(monkeypatch):
    from app.api import industry_loss as api

    storm = Storm(storm_id="AL092022", name="IAN", year=2022)
    impacts = [
        _impact("US-FL-12071", [(121, 0.6), (131, 0.2)], wind=120),
        _impact("US-FL-12015", [(91, 0.5)], wind=80),
        _impact("US-FL-12115", [(61, 0.4)], wind=55),
    ]
    monkeypatch.setattr(api, "resolve_storm", lambda sid: (storm, "ibtracs") if sid.upper() == "AL092022" else None)
    monkeypatch.setattr(api, "storm_impacts", lambda s, src, m: impacts)
    return storm


def _seed_files(root) -> None:
    _write(
        root / "industry_tiv" / "2022.csv",
        """
        geographyId,segment,tiv,currency
        US-FL-12071,RESIDENTIAL,100000000000,USD
        US-FL-12071,COMMERCIAL,60000000000,USD
        US-FL-12071,AUTO,8000000000,USD
        US-FL-12015,RESIDENTIAL,40000000000,USD
        US-FL-12015,COMMERCIAL,20000000000,USD
        US-FL-12115,RESIDENTIAL,90000000000,USD
        """,
    )
    _write(
        root / "industry_losses" / "ian.csv",
        """
        stormId,geographyId,segment,loss,currency,source
        AL092022,US-FL-12071,RESIDENTIAL,20000000000,USD,PCS
        AL092022,US-FL-12071,COMMERCIAL,9000000000,USD,PCS
        AL092022,US-FL-12071,AUTO,300000000,USD,PCS
        AL092022,US-FL-12015,RESIDENTIAL,1500000000,USD,PCS
        AL092022,US-FL-12105,RESIDENTIAL,200000000,USD,PCS
        """,
    )


def test_api_industry_loss(data_dir, fake_storm) -> None:
    _seed_files(data_dir)
    r = client.post("/api/hurricanes/AL092022/industry-loss", json={"curveSetId": "kcc"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["currency"] == "USD"
    assert body["tivSource"] == {"kind": "supplied", "year": 2022, "file": "2022.csv",
                                 "currency": "USD", "asOfDate": "2022-07-01"}
    assert body["actuals"]["onLevel"] is None  # no indices installed
    assert body["curveSet"]["id"] == "kcc"
    seg = body["summary"]["bySegment"]
    assert seg["RESIDENTIAL"]["mean"] > 0 and seg["AUTO"]["mean"] > 0
    total = sum(s["mean"] for s in seg.values())
    assert body["summary"]["total"]["mean"] == pytest.approx(total)
    assert body["actuals"]["total"] == pytest.approx(31_000_000_000)
    assert body["counties"][0]["geographyId"] == "US-FL-12071"


def test_api_onlevels_actuals_to_tiv_valuation_date(data_dir, fake_storm) -> None:
    _seed_files(data_dir)
    _write(data_dir / "industry_tiv" / "2022.csv",
           "geographyId,segment,tiv,currency,asOfDate\nUS-FL-12071,RESIDENTIAL,1e11,USD,2025-07-01")
    _write(data_dir / "onlevel" / "county_indices.csv", """
        geographyId,metric,2021,2022,2023,2024,2025
        US-FL-12071,HU,100,100,105,110,120
        US-FL,HU,100,100,100,100,100
        US-FL,CONS_PAY,50,50,50,50,50
        US-FL,POP,1,1,1,1,1
        US-FL,EMP,1000,1000,1000,1000,1000
        US-FL,WAGES,1,1,1,1,1
    """)
    _write(data_dir / "onlevel" / "monthly_indices.csv", """
        month,PPI_RES_INPUTS,PPI_NONRES_INPUTS,CPI_USED_VEHICLES
        2022-07,100,100,100
        2025-07,100,100,100
    """)
    body = client.post("/api/hurricanes/AL092022/industry-loss", json={"curveSetId": "kcc"}).json()
    info = body["actuals"]["onLevel"]
    assert info["eventDate"] == "2022-07-01" and info["targetDate"] == "2025-07-01"
    # Lee residential units grew 100 -> 120; everything else is flat.
    assert body["actuals"]["bySegment"]["RESIDENTIAL"] == pytest.approx(20e9 * 1.2 + 1.5e9 + 2e8, rel=1e-3)
    assert info["reportedTotal"] == pytest.approx(31e9)
    off = client.post("/api/hurricanes/AL092022/industry-loss",
                      json={"curveSetId": "kcc", "onLevel": False}).json()
    assert off["actuals"]["total"] == pytest.approx(31e9) and off["actuals"]["onLevel"] is None


def test_api_unknown_curve_and_storm(data_dir, fake_storm) -> None:
    assert client.post("/api/hurricanes/AL092022/industry-loss", json={"curveSetId": "nope"}).status_code == 404
    assert client.post("/api/hurricanes/AL012000/industry-loss", json={}).status_code == 404


def test_api_backtest(data_dir, fake_storm) -> None:
    _seed_files(data_dir)
    listing = client.get("/api/industry-losses").json()
    assert [e["stormId"] for e in listing["events"]] == ["AL092022"]
    r = client.get("/api/industry-losses/AL092022/backtest?curveSetId=kcc")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"]["actual"] == pytest.approx(31_000_000_000)
    assert body["segments"]["RESIDENTIAL"]["countiesScored"] == 2
    # Polk reported a loss but sits outside the modelled wind field.
    assert body["footprintMisses"] == [
        {
            "geographyId": "US-FL-12105",
            "segment": "RESIDENTIAL",
            "actual": 200_000_000,
            "belowMateriality": False,
            "evidencePeril": None,
        }
    ]
    assert body["materiality"]["currency"] == "USD"
    assert {c["geographyId"] for c in body["unreportedCounties"]} == {"US-FL-12115"}
    assert body["evidence"] is None and body["footprintMissByPeril"] is None
    assert client.get("/api/industry-losses/AL012000/backtest").status_code == 404


def test_api_backtest_explains_misses_with_evidence(data_dir, fake_storm) -> None:
    _seed_files(data_dir)
    (data_dir / "event_evidence").mkdir()
    (data_dir / "event_evidence" / "AL092022.json").write_text(json.dumps({
        "stormId": "AL092022",
        "currency": "USD",
        "windBias": {"n": 12, "medianObsOverModel": 0.8},
        "stormEventTotals": {"flood": {"n": 3, "damage": 1e6}},
        "stations": [],
        "counties": {
            "US-FL-12105": {
                "stormEvents": {"flood": {"n": 3, "damage": 1e6, "mentionsStorm": 3}},
                "hwm": {"n": 2, "coastal": 0, "riverine": 2, "maxHeightAboveGroundFt": 4.1},
            },
            "US-FL-12071": {"asos": {"n": 1, "maxSustainedKt": 70.0, "maxGustKt": 110.0,
                                     "modelLandKt": 92.0}},
        },
    }))
    body = client.get("/api/industry-losses/AL092022/backtest?curveSetId=kcc").json()
    assert body["evidence"]["windBias"]["medianObsOverModel"] == 0.8
    assert body["footprintMisses"][0]["evidencePeril"] == "flood"
    assert body["footprintMissByPeril"] == {"flood": 200_000_000}
    lee = next(r for r in body["counties"] if r["geographyId"] == "US-FL-12071")
    assert lee["evidence"]["asosMaxSustainedKt"] == 70.0
    assert lee["evidence"]["tornadoes"] == 0


def test_api_damage_curves(data_dir) -> None:
    body = client.get("/api/damage-curves").json()
    assert body["defaultId"] == "kcc"
    ids = {c["id"] for c in body["curveSets"]}
    assert {"kcc", "air", "rms"} <= ids
