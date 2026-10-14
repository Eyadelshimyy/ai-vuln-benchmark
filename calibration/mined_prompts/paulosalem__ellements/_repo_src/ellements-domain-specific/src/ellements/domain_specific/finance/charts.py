"""Finance chart helpers aligned with the existing matplotlib chart pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from ellements.reporting.charts import create_chart_artifacts
from pydantic import BaseModel

from .technical_indicators import (
    TechnicalIndicatorSnapshot,
    compute_technical_indicators,
)

if TYPE_CHECKING:
    from .yahoo_finance_models import HistoricalData


class TechnicalChartResult(BaseModel):
    """Structured technical chart output."""

    symbol: str
    title: str
    chart_assets: dict[str, Any]
    indicators: list[dict[str, Any]]
    snapshot: TechnicalIndicatorSnapshot


def mplfinance_available() -> bool:
    """Return True if mplfinance is importable."""
    try:
        import mplfinance  # noqa: F401
    except ImportError:
        return False
    return True


def _historical_to_dataframe(history: HistoricalData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for point in history.prices:
        if point.date is None:
            continue
        rows.append(
            {
                "date": pd.Timestamp(point.date),
                "open": point.open,
                "high": point.high,
                "low": point.low,
                "close": point.close if point.close is not None else point.adj_close,
                "adj_close": point.adj_close,
                "volume": point.volume,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Historical data contains no rows.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").set_index("date")
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["close"].isna().all():
        raise ValueError("Historical data has no usable close prices.")
    return frame


def _normalize_chart_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    frame = data.copy()
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.drop(columns=["date"])
        frame.index = dates
    else:
        frame.index = pd.to_datetime(frame.index, errors="coerce")

    frame = frame[~pd.isna(frame.index)].sort_index()
    if frame.empty:
        raise ValueError("Input DataFrame contains no valid datetime rows.")

    normalized_columns: dict[object, str] = {}
    for original in frame.columns:
        raw = str(original).strip().lower().replace(" ", "_")
        if raw in {"open", "high", "low", "close", "volume"}:
            normalized_columns[original] = raw
        elif raw in {"adj_close", "adjusted_close", "adjclose"}:
            normalized_columns[original] = "adj_close"

    if not normalized_columns:
        raise ValueError("Input DataFrame must include close prices (close/adj_close).")

    working = frame.rename(columns=normalized_columns)
    selected_columns = [
        col
        for col in ["open", "high", "low", "close", "adj_close", "volume"]
        if col in working.columns
    ]
    working = working[selected_columns].copy()

    if "close" not in working.columns:
        if "adj_close" not in working.columns:
            raise ValueError("Input DataFrame must include close prices (close/adj_close).")
        working["close"] = working["adj_close"]
    elif "adj_close" in working.columns:
        working["close"] = working["close"].fillna(working["adj_close"])

    for column in working.columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    if working["close"].isna().all():
        raise ValueError("Input DataFrame has no usable close prices.")
    return working


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN check
        return None
    return parsed


def _build_indicator_series_payloads(
    indicators: pd.DataFrame,
) -> list[dict[str, object]]:
    aligned = indicators.dropna(how="all")
    if aligned.empty:
        return []
    x_axis = [stamp.isoformat() for stamp in pd.DatetimeIndex(aligned.index).to_pydatetime()]
    payloads: list[dict[str, object]] = []
    colors = {
        "rsi": "#8b5cf6",
        "macd": "#3b82f6",
        "macd_signal": "#f59e0b",
        "macd_histogram": "#94a3b8",
        "stochastic_k": "#0ea5e9",
        "stochastic_d": "#f97316",
        "atr": "#ef4444",
        "adx": "#14b8a6",
    }
    for column in aligned.columns:
        if column in {
            "sma",
            "ema",
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower",
        }:
            continue
        values = aligned[column].tolist()
        if pd.Series(values).dropna().empty:
            continue
        payloads.append(
            {
                "name": column,
                "x": x_axis,
                "y": values,
                "color": colors.get(column),
            }
        )
    return payloads


def _serialize_indicator_rows(indicators_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in indicators_df.iterrows():
        idx_ts = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)
        payload: dict[str, Any] = {"date": idx_ts.isoformat()}
        for col in indicators_df.columns:
            payload[col] = _float_or_none(row[col])
        records.append(payload)
    return records


def _render_technical_chart_from_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    title: str = "",
    image_format: str = "png",
    dpi: int = 120,
    include_ohlc_fallback: bool = False,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 20,
    ema_period: int = 20,
    bollinger_window: int = 20,
    bollinger_dev: float = 2.0,
    stochastic_window: int = 14,
    stochastic_smooth_window: int = 3,
    atr_window: int = 14,
    adx_window: int = 14,
) -> TechnicalChartResult:
    indicators_df, snapshot = compute_technical_indicators(
        frame,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        sma_period=sma_period,
        ema_period=ema_period,
        bollinger_window=bollinger_window,
        bollinger_dev=bollinger_dev,
        stochastic_window=stochastic_window,
        stochastic_smooth_window=stochastic_smooth_window,
        atr_window=atr_window,
        adx_window=adx_window,
    )

    resolved_title = title.strip() or f"{symbol} Technical Indicators"
    x_axis = [stamp.isoformat() for stamp in pd.DatetimeIndex(frame.index).to_pydatetime()]
    base_series: list[dict[str, object]] = [
        {
            "name": "close",
            "x": x_axis,
            "y": frame["close"].tolist(),
            "color": "#2563eb",
        },
        {
            "name": f"sma_{sma_period}",
            "x": x_axis,
            "y": indicators_df["sma"].tolist(),
            "color": "#f97316",
        },
        {
            "name": f"ema_{ema_period}",
            "x": x_axis,
            "y": indicators_df["ema"].tolist(),
            "color": "#22c55e",
        },
        {
            "name": "bb_upper",
            "x": x_axis,
            "y": indicators_df["bollinger_upper"].tolist(),
            "color": "#94a3b8",
        },
        {
            "name": "bb_middle",
            "x": x_axis,
            "y": indicators_df["bollinger_middle"].tolist(),
            "color": "#64748b",
        },
        {
            "name": "bb_lower",
            "x": x_axis,
            "y": indicators_df["bollinger_lower"].tolist(),
            "color": "#94a3b8",
        },
    ]
    series = base_series + _build_indicator_series_payloads(indicators_df)

    chart_spec: dict[str, Any] = {
        "type": "line",
        "title": resolved_title,
        "x_label": "Time",
        "y_label": "Value",
        "series": series,
        "legend": True,
    }
    if include_ohlc_fallback:
        chart_spec["caption"] = (
            "Candlestick rendering can be layered later via mplfinance. "
            "Current output keeps compatibility with the existing chart artifact pipeline."
        )

    assets = create_chart_artifacts(
        chart_spec,
        title=resolved_title,
        image_format=image_format,
        dpi=dpi,
    )

    return TechnicalChartResult(
        symbol=symbol,
        title=resolved_title,
        chart_assets=dict(assets),
        indicators=_serialize_indicator_rows(indicators_df),
        snapshot=snapshot,
    )


def render_technical_chart(
    history: HistoricalData,
    *,
    title: str = "",
    image_format: str = "png",
    dpi: int = 120,
    include_ohlc_fallback: bool = False,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 20,
    ema_period: int = 20,
    bollinger_window: int = 20,
    bollinger_dev: float = 2.0,
    stochastic_window: int = 14,
    stochastic_smooth_window: int = 3,
    atr_window: int = 14,
    adx_window: int = 14,
) -> TechnicalChartResult:
    """Render a technical chart artifact and indicator data.

    Harmonization note:
        Uses the existing reports chart artifact pipeline (matplotlib -> data_uri),
        so output is directly consumable by existing GUI/chat chart widgets.
    """
    frame = _historical_to_dataframe(history)
    return _render_technical_chart_from_frame(
        frame,
        symbol=history.symbol,
        title=title,
        image_format=image_format,
        dpi=dpi,
        include_ohlc_fallback=include_ohlc_fallback,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        sma_period=sma_period,
        ema_period=ema_period,
        bollinger_window=bollinger_window,
        bollinger_dev=bollinger_dev,
        stochastic_window=stochastic_window,
        stochastic_smooth_window=stochastic_smooth_window,
        atr_window=atr_window,
        adx_window=adx_window,
    )


def render_technical_chart_from_dataframe(
    data: pd.DataFrame,
    *,
    symbol: str = "DATAFRAME",
    title: str = "",
    image_format: str = "png",
    dpi: int = 120,
    include_ohlc_fallback: bool = False,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 20,
    ema_period: int = 20,
    bollinger_window: int = 20,
    bollinger_dev: float = 2.0,
    stochastic_window: int = 14,
    stochastic_smooth_window: int = 3,
    atr_window: int = 14,
    adx_window: int = 14,
) -> TechnicalChartResult:
    """Render a technical chart artifact directly from a pandas DataFrame."""
    frame = _normalize_chart_dataframe(data)
    return _render_technical_chart_from_frame(
        frame,
        symbol=symbol,
        title=title,
        image_format=image_format,
        dpi=dpi,
        include_ohlc_fallback=include_ohlc_fallback,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        sma_period=sma_period,
        ema_period=ema_period,
        bollinger_window=bollinger_window,
        bollinger_dev=bollinger_dev,
        stochastic_window=stochastic_window,
        stochastic_smooth_window=stochastic_smooth_window,
        atr_window=atr_window,
        adx_window=adx_window,
    )


def technical_chart_assets(
    history: HistoricalData,
    *,
    title: str = "",
    image_format: str = "png",
    dpi: int = 120,
    include_ohlc_fallback: bool = False,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 20,
    ema_period: int = 20,
    bollinger_window: int = 20,
    bollinger_dev: float = 2.0,
    stochastic_window: int = 14,
    stochastic_smooth_window: int = 3,
    atr_window: int = 14,
    adx_window: int = 14,
) -> dict[str, Any]:
    """Convenience wrapper returning chart assets + indicator payloads."""
    result = render_technical_chart(
        history,
        title=title,
        image_format=image_format,
        dpi=dpi,
        include_ohlc_fallback=include_ohlc_fallback,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        sma_period=sma_period,
        ema_period=ema_period,
        bollinger_window=bollinger_window,
        bollinger_dev=bollinger_dev,
        stochastic_window=stochastic_window,
        stochastic_smooth_window=stochastic_smooth_window,
        atr_window=atr_window,
        adx_window=adx_window,
    )
    payload = dict(result.chart_assets)
    payload["symbol"] = result.symbol
    payload["snapshot"] = result.snapshot.model_dump()
    payload["indicators"] = result.indicators
    return payload
