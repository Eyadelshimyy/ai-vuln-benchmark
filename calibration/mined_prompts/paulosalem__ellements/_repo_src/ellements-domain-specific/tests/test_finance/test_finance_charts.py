"""Focused tests for finance charting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ellements.domain_specific.finance.charts import technical_chart_assets
from ellements.domain_specific.finance.technical_indicators import talib_available
from ellements.domain_specific.finance.yahoo_finance_models import HistoricalData


def _history_from_frame(frame: pd.DataFrame) -> HistoricalData:
    prices = []
    for idx, row in frame.iterrows():
        prices.append(
            {
                "date": idx.to_pydatetime(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
                "adj_close": float(row["close"]),
            }
        )
    return HistoricalData(symbol="CHART", period="3mo", interval="1d", prices=prices)


def _sample_frame(length: int = 90) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=length, freq="D")
    base = np.linspace(50.0, 75.0, length)
    wobble = np.cos(np.linspace(0, 10, length))
    close = base + wobble
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(100, 1000, length),
        },
        index=idx,
    )


@pytest.mark.skipif(not talib_available(), reason="TA-Lib not installed")
def test_technical_chart_assets_payload_shape():
    history = _history_from_frame(_sample_frame())
    payload = technical_chart_assets(history, title="My Chart")

    assert payload["symbol"] == "CHART"
    assert payload["title"] == "My Chart"
    assert str(payload["data_uri"]).startswith("data:image/png;base64,")
    assert "canvas_chart_payload" in payload
    assert "snapshot" in payload
    assert "indicators" in payload
    assert isinstance(payload["indicators"], list)
