"""_report_month_bounds() feeds the live getEC query behind the Monthly Report's Consumo-medio/
Energia tiles. The window must grow day by day for the month currently in progress (end = now,
not the month's future end) but be the true full month once it's closed — otherwise a past month
would silently miss its last days' driving.

Needs web.main (fastapi); the minimal CI env skips this module cleanly.
"""
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi", reason="web.main needs fastapi (absent in the minimal CI test env)")

import main


def test_current_month_in_progress_ends_at_now():
    now = datetime(2026, 7, 1, 14, 18, 0, tzinfo=timezone.utc)
    begin_ts, end_ts = main._report_month_bounds("2026-07", now)
    assert begin_ts == int(datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    assert end_ts == int(now.timestamp())   # NOT July's last moment — the month isn't over yet


def test_closed_past_month_ends_at_its_last_moment():
    now = datetime(2026, 7, 1, 14, 18, 0, tzinfo=timezone.utc)   # viewing June from July
    begin_ts, end_ts = main._report_month_bounds("2026-06", now)
    assert begin_ts == int(datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    assert end_ts == int(datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc).timestamp())


def test_february_uses_the_real_last_day():
    now = datetime(2026, 3, 15, tzinfo=timezone.utc)
    _, end_ts = main._report_month_bounds("2026-02", now)
    assert end_ts == int(datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc).timestamp())
