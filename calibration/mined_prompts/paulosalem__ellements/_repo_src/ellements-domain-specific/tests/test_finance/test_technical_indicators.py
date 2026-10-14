"""Tests for technical indicators and finance chart helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ellements.domain_specific.finance.charts import (
    render_technical_chart_from_dataframe,
)
from ellements.domain_specific.finance.technical_indicators import (
    compute_technical_indicators,
    talib_available,
)


def _sample_price_frame(length: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    base = np.linspace(100.0, 130.0, num=length)
    noise = np.sin(np.linspace(0, 12, num=length)) * 1.5
    close = base + noise
    open_ = close - 0.4
    high = close + 1.1
    low = close - 1.1
    volume = np.linspace(1000, 5000, num=length)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


@pytest.mark.skipif(not talib_available(), reason="TA-Lib not installed")
def test_compute_indicators_from_dataframe_preserves_index():
    frame = _sample_price_frame()
    indicators, snapshot = compute_technical_indicators(frame)

    assert len(indicators) == len(frame)
    assert indicators.index.equals(frame.index)
    expected_columns = {
        "rsi",
        "macd",
        "macd_signal",
        "macd_histogram",
        "sma",
        "ema",
        "bollinger_upper",
        "bollinger_middle",
        "bollinger_lower",
        "stochastic_k",
        "stochastic_d",
        "atr",
        "adx",
    }
    assert expected_columns.issubset(set(indicators.columns))
    assert snapshot.sample_size == len(frame)
    assert snapshot.rsi is None or isinstance(snapshot.rsi, float)


@pytest.mark.skipif(not talib_available(), reason="TA-Lib not installed")
def test_compute_indicators_from_series_supported():
    frame = _sample_price_frame()
    indicators, _snapshot = compute_technical_indicators(frame["close"])

    assert len(indicators) == len(frame)
    assert indicators.index.equals(frame.index)
    # Without OHLC columns, ATR/ADX/Stochastic should be all NaN
    assert indicators["atr"].isna().all()
    assert indicators["adx"].isna().all()
    assert indicators["stochastic_k"].isna().all()
    assert indicators["stochastic_d"].isna().all()


@pytest.mark.skipif(not talib_available(), reason="TA-Lib not installed")
def test_render_technical_chart_from_dataframe_returns_artifacts():
    frame = _sample_price_frame()
    result = render_technical_chart_from_dataframe(
        frame,
        symbol="TEST",
        title="TEST Technical",
        image_format="png",
    )

    assert result.symbol == "TEST"
    assert result.title == "TEST Technical"
    assert "data_uri" in result.chart_assets
    assert str(result.chart_assets["data_uri"]).startswith("data:image/png;base64,")
    assert "canvas_chart_payload" in result.chart_assets
    assert len(result.indicators) == len(frame)
    assert isinstance(result.snapshot.sample_size, int)


def test_compute_indicators_invalid_params():
    frame = _sample_price_frame()
    with pytest.raises(ValueError, match="rsi_period must be > 0"):
        compute_technical_indicators(frame, rsi_period=0)

    with pytest.raises(ValueError, match="bollinger_dev must be > 0"):
        compute_technical_indicators(frame, bollinger_dev=0.0)
