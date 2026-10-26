"""850–200 hPa deep-layer shear."""

from app.services.wind_forecast import deep_layer_shear


def test_westerly_upper_wind_is_westerly_shear():
    # 850 calm, 200 hPa 40 kt from the west. Arrow (+180) points east.
    mag, frm = deep_layer_shear(0.0, 0.0, 40.0, 270.0)
    assert mag == 40.0
    assert frm == 270.0


def test_matched_levels_have_no_shear():
    mag, frm = deep_layer_shear(20.0, 90.0, 20.0, 90.0)
    assert mag == 0.0
    assert frm is None


def test_calm_aloft_reverses_the_lower_wind():
    # 20 kt from the east at 850, calm at 200. The difference points east,
    # which is a from-the-west shear of 20 kt.
    mag, frm = deep_layer_shear(20.0, 90.0, 0.0, 0.0)
    assert mag == 20.0
    assert frm == 270.0


def test_missing_level_is_a_gap_not_zero_shear():
    from datetime import datetime, timezone

    from app.services.wind_forecast import _extract_shear_frames

    now = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
    times = [f"2026-10-07T{12 + h:02d}:00" for h in range(7)]
    n = len(times)
    upper = [20.0] * n
    upper[6] = None
    item = {
        "hourly": {
            "time": times,
            "wind_speed_850hPa": [10.0] * n,
            "wind_direction_850hPa": [90.0] * n,
            "wind_speed_200hPa": upper,
            "wind_direction_200hPa": [90.0] * n,
        },
    }
    coords, frames = _extract_shear_frames([(25.0, -90.0)], [item], now)
    assert len(coords) == 1
    kts0, dirs0, vt0 = frames[0]
    kts6, dirs6, _vt6 = frames[6]
    assert kts0[0] is not None and kts0[0] > 0
    assert dirs0[0] is not None
    assert vt0
    assert kts6[0] is None
    assert dirs6[0] is None
