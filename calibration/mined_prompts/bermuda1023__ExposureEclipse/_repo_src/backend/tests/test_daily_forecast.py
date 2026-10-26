"""Five-day point forecast. No network: Open-Meteo is stubbed."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app
from app.services.wind_forecast import (
    DailyDay,
    DailyForecast,
    ModelDaily,
    _DAILY_CACHE,
    daily_forecast,
)

client = TestClient(app)


class _Body:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_Body":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _payload(model_hint: str) -> dict:
    hotter = 80.0 if "gfs" in model_hint else 70.0
    return {
        "timezone": "America/New_York",
        "daily": {
            "time": ["2026-10-08", "2026-10-09", "2026-10-10", "2026-10-11", "2026-10-12"],
            "weather_code": [1, 61, 3, 0, 95],
            "temperature_2m_max": [hotter, hotter - 1, None, 60.2, 55],
            "temperature_2m_min": [60.4, 58, 50, 48.2, 40],
            "precipitation_sum": [0, 0.426, 0.1, 0, 1.2],
            "wind_speed_10m_max": [12.24, 18, 9, 8, 22],
            "wind_gusts_10m_max": [20, None, 14, 11, 35],
            "wind_direction_10m_dominant": [180.2, 210, 90, 0, 359],
        },
    }


def test_daily_forecast_keeps_both_models_and_units(monkeypatch) -> None:
    _DAILY_CACHE.clear()

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        url = req.full_url
        return _Body(_payload(url))

    monkeypatch.setattr(
        "app.services.wind_forecast.urllib.request.urlopen", fake_urlopen,
    )
    result = daily_forecast(40.712, -74.006)
    assert [m.model for m in result.models] == ["gfs", "ecmwf"]
    assert result.timezone == "America/New_York"
    gfs = result.models[0].days
    assert len(gfs) == 5
    assert gfs[0].temp_max_f == 80.0
    assert gfs[0].temp_min_f == 60.4
    assert gfs[1].precip_in == 0.43
    assert gfs[0].wind_max_kt == 12.2
    assert gfs[1].gust_max_kt is None
    assert gfs[0].wind_dir_deg == 180.0
    assert gfs[2].temp_max_f is None
    assert gfs[4].weather_code == 95
    assert result.models[1].days[0].temp_max_f == 70.0


def test_daily_forecast_omits_a_failed_model(monkeypatch) -> None:
    _DAILY_CACHE.clear()

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        if "ecmwf" in req.full_url:
            raise TimeoutError("slow")
        return _Body(_payload(req.full_url))

    monkeypatch.setattr(
        "app.services.wind_forecast.urllib.request.urlopen", fake_urlopen,
    )
    result = daily_forecast(41.0, -73.0)
    assert [m.model for m in result.models] == ["gfs"]


def test_daily_forecast_endpoint_is_camel_case(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.live.daily_forecast",
        lambda lat, lon: DailyForecast(
            lat=lat,
            lon=lon,
            timezone="Asia/Tokyo",
            fetched_at_utc="2026-10-08T00:00:00Z",
            models=[
                ModelDaily(
                    model="gfs",
                    days=[
                        DailyDay(
                            date="2026-10-08",
                            weather_code=3,
                            temp_max_f=72.0,
                            temp_min_f=61.0,
                            precip_in=0.1,
                            wind_max_kt=15.0,
                            gust_max_kt=24.0,
                            wind_dir_deg=90.0,
                        ),
                    ],
                ),
            ],
        ),
    )
    r = client.get("/api/live/daily-forecast", params={"lat": 35.6, "lon": 139.7})
    assert r.status_code == 200
    body = r.json()
    assert body["timezone"] == "Asia/Tokyo"
    assert body["models"][0]["model"] == "gfs"
    day = body["models"][0]["days"][0]
    assert day["tempMaxF"] == 72.0
    assert day["precipIn"] == 0.1
    assert day["windMaxKt"] == 15.0
    assert "temp_max_f" not in day
