"""#121 (Wartopia): the Monthly Report's AVG consumption was blank ("—") for a user's first partial
month. The report's getEC window began at the 1st of the month, but a fresh install mid-month leaves
the month's head with no trips — so the guard withheld the average (full-window cloud energy would
pair with partial trip distance → a nonsense figure). The fix CLAMPS the window start to the
first-ever recorded trip, so cloud energy and trip distance span the SAME period and the real getEC
average shows. An established user's month (first trip predates it) is untouched.

Needs web.main (fastapi); the minimal CI env skips this module cleanly."""
import pytest

pytest.importorskip("fastapi", reason="web.main needs fastapi (absent in the minimal CI test env)")
import main

DAY = 86400


# ── the clamp helper ─────────────────────────────────────────────────────────

def test_clamp_moves_start_to_first_trip_for_a_first_partial_month(monkeypatch):
    first = 1_000_000                                   # first-ever recorded trip
    monkeypatch.setattr(main.db_reader, "get_first_trip_ts", lambda: first)
    month_start = first - 9 * DAY                       # month began 9 days before the first trip
    assert main._clamp_begin_to_first_trip(month_start) == first   # clamped up to the first trip


def test_clamp_is_noop_for_an_established_user(monkeypatch):
    first = 1_000_000
    monkeypatch.setattr(main.db_reader, "get_first_trip_ts", lambda: first)
    month_start = first + 60 * DAY                      # a month well after the first trip
    assert main._clamp_begin_to_first_trip(month_start) == month_start   # unchanged


def test_clamp_is_noop_when_no_trips_yet(monkeypatch):
    monkeypatch.setattr(main.db_reader, "get_first_trip_ts", lambda: None)
    assert main._clamp_begin_to_first_trip(500) == 500


# ── the Wartopia scenario, end to end through the enrich ──────────────────────

def test_wartopia_scenario_dash_before_real_avg_after(monkeypatch):
    """Reproduce Wartopia: 7 trips / 344 km all in the first partial month, getEC = 61.6 kWh."""
    first = 1_000_000                                   # first trip (mid-month)
    month_start = first - 9 * DAY
    end = first + 21 * DAY
    monkeypatch.setattr(main.db_reader, "get_first_trip_ts", lambda: first)
    monkeypatch.setattr(main.db_reader, "get_trip_totals_between",
                        lambda b, e: {"distance_km": 344.0, "duration_min": 400})
    eb = {"total_kwh": 61.6}

    # BEFORE the fix — the report queried getEC from the month start → the guard blanks the average
    before = main._enrich_eb_with_trip_totals(dict(eb), month_start, end)
    assert "avg_kwh100" not in before                  # → the "—" Wartopia reported

    # THE FIX — clamp the window start, then enrich
    clamped = main._clamp_begin_to_first_trip(month_start)
    after = main._enrich_eb_with_trip_totals(dict(eb), clamped, end)
    assert after["avg_kwh100"] == 17.9                 # 61.6 / 344 * 100 — the real getEC average
    assert after["distance_km"] == 344.0


def test_established_user_month_shows_avg_unchanged(monkeypatch):
    first = 1_000_000
    month_start = first + 60 * DAY                      # normal later month
    end = month_start + 30 * DAY
    monkeypatch.setattr(main.db_reader, "get_first_trip_ts", lambda: first)
    monkeypatch.setattr(main.db_reader, "get_trip_totals_between",
                        lambda b, e: {"distance_km": 500.0, "duration_min": 600})
    clamped = main._clamp_begin_to_first_trip(month_start)
    assert clamped == month_start                       # no clamp for an established user
    eb = main._enrich_eb_with_trip_totals({"total_kwh": 90.0}, clamped, end)
    assert eb["avg_kwh100"] == 18.0                     # 90 / 500 * 100


def test_parked_month_no_driving_shows_no_avg(monkeypatch):
    """Silvio's edge case: an established user parks the car all month → no driving. getEC ~0 → no
    average, correctly, and NOT via the guard (the empty-total_kwh short-circuit fires first)."""
    monkeypatch.setattr(main.db_reader, "get_first_trip_ts", lambda: 1000)     # old first trip
    eb = main._enrich_eb_with_trip_totals({"total_kwh": 0}, 5_000_000, 5_000_000 + 30 * DAY)
    assert "avg_kwh100" not in eb                       # "—" is correct: nothing was driven this month
