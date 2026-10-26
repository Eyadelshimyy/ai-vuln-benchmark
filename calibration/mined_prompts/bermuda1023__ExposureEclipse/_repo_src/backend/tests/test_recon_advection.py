"""Storm-relative placement of hunter surface samples."""

from datetime import datetime, timedelta, timezone

from app.services.wind_field_map import (
    AGE_TAU_H,
    WindObs,
    _SURFACE_CACHE,
    advect_recon_obs,
    interpolate_obs,
    regrid_with_recon,
    remember_surface_obs,
)
from app.services.wind_forecast import choose_model_step
import math


NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
CENTERS = [
    ("2026-10-07T10:00:00Z", 22.0, -94.0),
    ("2026-10-07T12:00:00Z", 23.0, -94.0),
]


def test_vortex_sample_slides_with_the_center():
    # 0.3° east of the center at 10Z. The center then moves 1° north.
    moved = advect_recon_obs(
        [(22.0, -93.7, 90.0, 180.0, "2026-10-07T10:00:00Z", "AF309-10:00 adv")],
        CENTERS,
        NOW,
    )
    assert len(moved) == 1
    assert moved[0].lat == 23.0
    assert moved[0].lon == -93.7
    assert moved[0].wind_kt == 90.0
    assert moved[0].wind_dir_deg == 180.0
    assert moved[0].source == "recon"
    # Two hours old, tau = 3 h.
    assert moved[0].weight == round(math.exp(-2 / AGE_TAU_H), 3)


def test_ferry_leg_is_not_dragged_with_the_vortex():
    # 8° north of the center is the environment, and 2 h is stale.
    dropped = advect_recon_obs(
        [(30.0, -94.0, 25.0, 90.0, "2026-10-07T10:00:00Z", "ferry")],
        CENTERS,
        NOW,
    )
    assert dropped == []


def test_fresh_environment_sample_stays_where_it_was_measured():
    fresh_now = datetime(2026, 10, 7, 10, 20, tzinfo=timezone.utc)
    kept = advect_recon_obs(
        [(30.0, -94.0, 18.0, 45.0, "2026-10-07T10:00:00Z", "ferry")],
        CENTERS,
        fresh_now,
    )
    assert len(kept) == 1
    assert kept[0].lat == 30.0
    assert kept[0].lon == -94.0


def test_no_motion_estimate_drops_a_stale_sample():
    # A single center fix cannot say where the storm was two hours ago.
    dropped = advect_recon_obs(
        [(22.2, -94.0, 70.0, 180.0, "2026-10-07T10:00:00Z", "AF309")],
        [("2026-10-07T12:00:00Z", 23.0, -94.0)],
        NOW,
    )
    assert dropped == []


def test_forecast_leg_places_the_center_between_advisories():
    # 10Z at 22N, forecast valid 16Z at 25N. "Now" is 13Z — halfway.
    centers = [
        ("2026-10-07T10:00:00Z", 22.0, -94.0),
        ("2026-10-07T16:00:00Z", 25.0, -94.0),
    ]
    moved = advect_recon_obs(
        [(22.0, -94.0, 60.0, 0.0, "2026-10-07T10:00:00Z", "AF309")],
        centers,
        datetime(2026, 10, 7, 13, 0, tzinfo=timezone.utc),
    )
    assert len(moved) == 1
    assert moved[0].lat == 23.5
    assert moved[0].lon == -94.0


def test_hunter_sample_reaches_a_cell_the_buoys_do_not():
    # One strong sample, tight recon radius, cell on top of it.
    from app.services.wind_field_map import WindObs
    obs = [
        WindObs(
            lat=22.0, lon=-94.0, wind_kt=80.0, wind_dir_deg=90.0,
            source="recon", station_id="AF", observed_at="2026-10-07T12:00:00Z",
            weight=1.0,
        )
    ]
    cells = interpolate_obs(-94.5, 21.5, -93.5, 22.5, obs, step=0.5)
    assert cells
    on_top = next(c for c in cells if abs(c.lat - 22.0) < 0.01 and abs(c.lon + 94.0) < 0.01)
    assert on_top.wind_kt > 50


def test_distant_gradient_does_not_zero_a_close_hunter():
    """A hunter on the cell stays HIGH even if the rest of the storm differs."""
    from app.services.wind_field_map import WindObs
    obs = [
        WindObs(
            lat=22.0, lon=-94.0, wind_kt=90.0, wind_dir_deg=90.0,
            source="recon", station_id="AF", observed_at="2026-10-07T12:00:00Z",
            weight=1.0,
        ),
        WindObs(
            lat=22.05, lon=-94.05, wind_kt=88.0, wind_dir_deg=80.0,
            source="recon", station_id="AF2", observed_at="2026-10-07T12:00:00Z",
            weight=1.0,
        ),
        # Far enough to be outside the 0.5° agreement neighborhood, inside
        # the 3° buoy radius. These used to drive agreement to zero.
        WindObs(
            lat=24.2, lon=-94.0, wind_kt=15.0, wind_dir_deg=0.0,
            source="buoy", station_id="B1", observed_at="2026-10-07T12:00:00Z",
            weight=1.0,
        ),
        WindObs(
            lat=22.0, lon=-91.6, wind_kt=20.0, wind_dir_deg=180.0,
            source="buoy", station_id="B2", observed_at="2026-10-07T12:00:00Z",
            weight=1.0,
        ),
    ]
    cells = interpolate_obs(-94.5, 21.5, -93.5, 22.5, obs, step=0.5)
    on_top = next(c for c in cells if abs(c.lat - 22.0) < 0.01 and abs(c.lon + 94.0) < 0.01)
    assert on_top.nearest_obs_km is not None and on_top.nearest_obs_km < 20
    assert on_top.confidence >= 0.5
    assert on_top.agreement_score >= 0.5


def test_nearby_hunter_disagreement_stays_low():
    from app.services.wind_field_map import WindObs
    obs = [
        WindObs(
            lat=22.0, lon=-94.0, wind_kt=30.0, wind_dir_deg=90.0,
            source="recon", station_id="AF", observed_at="2026-10-07T12:00:00Z",
            weight=1.0,
        ),
        WindObs(
            lat=22.15, lon=-94.0, wind_kt=95.0, wind_dir_deg=90.0,
            source="recon", station_id="AF2", observed_at="2026-10-07T12:00:00Z",
            weight=1.0,
        ),
    ]
    cells = interpolate_obs(-94.5, 21.5, -93.5, 22.5, obs, step=0.5)
    on_top = next(c for c in cells if abs(c.lat - 22.0) < 0.01 and abs(c.lon + 94.0) < 0.01)
    assert on_top.confidence < 0.25



def test_model_step_coarsens_a_basin_sized_bbox():
    step = choose_model_step(-100.0, 10.0, -60.0, 45.0)
    nlat = int(35.0 / step) + 1
    nlon = int(40.0 / step) + 1
    assert nlat * nlon <= 1000
    assert step >= 0.5


def test_model_step_keeps_a_gulf_cone_under_the_minute_budget():
    # The Isaias-sized probe bbox. Both models have to fit in 600
    # location-weighted calls, so one cone stays at or under the cap.
    from app.services.wind_forecast import _MAX_MODEL_CELLS

    step = choose_model_step(-97.0, 19.0, -81.0, 41.0)
    nlat = int(22.0 / step) + 1
    nlon = int(16.0 / step) + 1
    assert nlat * nlon <= _MAX_MODEL_CELLS
    assert step == 1.25


def test_model_step_uses_quarter_degree_on_a_tight_view():
    # A zoomed 2° box. 0.25° is 9×9 = 81 locations, under the cap.
    assert choose_model_step(-88.0, 27.0, -86.0, 29.0) == 0.25


def test_model_step_stays_half_degree_on_a_small_box():
    assert choose_model_step(-90.0, 24.0, -84.0, 30.0) == 0.5


def test_cached_surface_field_accepts_a_later_hunter_pass():
    _SURFACE_CACHE.clear()
    assert regrid_with_recon(-95, 21, -93, 24, []) is None
    remember_surface_obs(-95, 21, -93, 24, [
        WindObs(
            lat=21.2, lon=-94.8, wind_kt=12.0, wind_dir_deg=90.0,
            source="buoy", station_id="41000",
            observed_at="2026-10-07T12:00:00Z", weight=1.0,
        ),
    ])
    moved = advect_recon_obs(
        [(22.0, -93.7, 90.0, 180.0, "2026-10-07T10:00:00Z", "AF309 adv")],
        CENTERS,
        NOW,
    )
    blended = regrid_with_recon(-95, 21, -93, 24, moved)
    assert blended is not None
    _cells, step, obs = blended
    assert step == 0.25
    assert any(o.source == "buoy" for o in obs)
    assert any(o.source == "recon" and o.lat == 23.0 for o in obs)


def _hourly(speed_ms: float) -> dict:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    times = [
        (now + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M")
        for h in range(0, 130)
    ]
    return {
        "hourly": {
            "time": times,
            "wind_speed_10m": [speed_ms] * len(times),
            "wind_direction_10m": [90.0] * len(times),
        }
    }


def test_failed_model_chunk_is_a_gap_not_calm(monkeypatch):
    import app.services.wind_forecast as wf

    wf._GRID_CACHE.clear()
    wf._bulk_cache.clear()
    wf._location_events.clear()
    monkeypatch.setattr(wf, "_CHUNK_SIZE", 2)

    def fake(lat_str, lon_str, model_key, refresh=False, **_kwargs):
        lats = [float(x) for x in lat_str.split(",") if x]
        if lats and abs(lats[0] - 24.0) < 1e-6:
            raise TimeoutError("chunk died")
        return tuple(_hourly(10.0) for _ in lats)

    monkeypatch.setattr(wf, "_fetch_bulk_chunk", fake)
    grid = wf.fetch_model_wind_grid(-90.0, 24.0, -89.0, 25.0, "gfs")
    assert grid.cells
    assert all(abs(c.lat - 24.0) > 1e-6 for c in grid.cells)
    assert any(kt > 0 for frame in grid.frames for kt in frame.wind_kt)
    for frame in grid.frames:
        assert len(frame.wind_kt) == len(grid.cells)
        assert len(frame.wind_dir_deg) == len(grid.cells)


def test_empty_model_grid_is_not_cached(monkeypatch):
    import app.services.wind_forecast as wf

    wf._GRID_CACHE.clear()
    wf._bulk_cache.clear()
    wf._location_events.clear()
    calls = {"n": 0}

    def fake(lat_str, lon_str, model_key, refresh=False, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return ()
        return tuple(_hourly(8.0) for _ in lat_str.split(",") if _)

    monkeypatch.setattr(wf, "_fetch_bulk_chunk", fake)
    first = wf.fetch_model_wind_grid(-80.0, 25.0, -79.5, 25.5, "ecmwf")
    assert not any(kt > 0 for frame in first.frames for kt in frame.wind_kt)
    assert not first.rate_limited
    second = wf.fetch_model_wind_grid(-80.0, 25.0, -79.5, 25.5, "ecmwf")
    assert any(kt > 0 for frame in second.frames for kt in frame.wind_kt)
    assert calls["n"] >= 2


def test_upstream_429_is_rate_limited_and_not_cached(monkeypatch):
    import app.services.wind_forecast as wf

    wf._GRID_CACHE.clear()
    wf._bulk_cache.clear()
    wf._location_events.clear()
    calls = {"n": 0}

    def fake(*_args, **_kwargs):
        calls["n"] += 1
        raise wf.OpenMeteoRateLimited()

    monkeypatch.setattr(wf, "_fetch_bulk_chunk", fake)
    grid = wf.fetch_model_wind_grid(-80.0, 25.0, -79.5, 25.5, "ecmwf")
    assert grid.rate_limited
    assert grid.cells == []
    seen = calls["n"]
    again = wf.fetch_model_wind_grid(-80.0, 25.0, -79.5, 25.5, "ecmwf")
    assert again.rate_limited
    assert calls["n"] > seen


def test_location_budget_refuses_the_next_grid_without_a_request(monkeypatch):
    import app.services.wind_forecast as wf

    wf._GRID_CACHE.clear()
    wf._bulk_cache.clear()
    wf._location_events.clear()
    # The inclusive 0.25-degree grid has 3 x 3 = 9 cells. Admit exactly one
    # grid, then verify that the next one is refused before an upstream call.
    monkeypatch.setattr(wf, "_LOCATION_BUDGET", 9)
    calls = {"n": 0}

    def fake(lat_str, lon_str, model_key, refresh=False, **_kwargs):
        calls["n"] += 1
        return tuple(_hourly(8.0) for _ in lat_str.split(",") if _)

    monkeypatch.setattr(wf, "_fetch_bulk_chunk", fake)
    first = wf.fetch_model_wind_grid(-80.0, 25.0, -79.5, 25.5, "gfs")
    assert any(kt > 0 for frame in first.frames for kt in frame.wind_kt)
    second = wf.fetch_model_wind_grid(-70.0, 25.0, -69.5, 25.5, "ecmwf")
    assert second.rate_limited
    assert second.cells == []
    assert calls["n"] == 1
    wf._location_events.clear()
