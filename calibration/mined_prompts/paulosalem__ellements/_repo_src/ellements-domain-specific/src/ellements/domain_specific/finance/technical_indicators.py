"""Technical analysis indicators with pandas-first APIs backed by TA-Lib."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel


def talib_available() -> bool:
    """Return True when TA-Lib can be imported."""
    try:
        import talib  # noqa: F401
    except ImportError:
        return False
    return True


def _load_talib() -> Any:
    try:
        import talib
    except ImportError as exc:
        raise RuntimeError(
            "TA-Lib is required for technical indicators. "
            "Install with: pip install TA-Lib"
        ) from exc
    return talib


class TechnicalIndicatorSnapshot(BaseModel):
    """Latest values for the core technical indicator set."""

    sample_size: int = 0
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    sma: float | None = None
    ema: float | None = None
    bollinger_upper: float | None = None
    bollinger_middle: float | None = None
    bollinger_lower: float | None = None
    stochastic_k: float | None = None
    stochastic_d: float | None = None
    atr: float | None = None
    adx: float | None = None


def _as_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")


def _normalize_price_input(
    data: pd.Series | pd.DataFrame | Sequence[float],
) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        frame = pd.DataFrame({"close": _as_float_series(data)})
        frame.index = data.index
        return frame.sort_index()

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        if frame.empty:
            raise ValueError("Input DataFrame is empty.")

        normalized_columns: dict[object, str] = {}
        for original in frame.columns:
            raw = str(original).strip().lower().replace(" ", "_")
            if raw in {"open", "high", "low", "close", "volume"}:
                normalized_columns[original] = raw
            elif raw in {"adj_close", "adjusted_close", "adjclose"}:
                normalized_columns[original] = "adj_close"

        if not normalized_columns:
            raise ValueError(
                "Input DataFrame must include close prices (close/adj_close)."
            )

        working = frame.rename(columns=normalized_columns)
        selected_columns = [
            col
            for col in ["open", "high", "low", "close", "adj_close", "volume"]
            if col in working.columns
        ]
        working = working[selected_columns].copy()

        if "close" not in working.columns:
            if "adj_close" not in working.columns:
                raise ValueError(
                    "Input DataFrame must include close prices (close/adj_close)."
                )
            working["close"] = working["adj_close"]
        elif "adj_close" in working.columns:
            working["close"] = working["close"].fillna(working["adj_close"])

        for column in working.columns:
            working[column] = _as_float_series(working[column])

        return working.sort_index()

    if not isinstance(data, Sequence):
        raise ValueError("Unsupported input type for technical indicators.")

    series = pd.Series(list(data), dtype="float64")
    return pd.DataFrame({"close": _as_float_series(series)})


def _last_finite(series: pd.Series) -> float | None:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def latest_indicator_snapshot(indicators: pd.DataFrame) -> TechnicalIndicatorSnapshot:
    """Extract latest finite values for each indicator column."""
    return TechnicalIndicatorSnapshot(
        sample_size=int(len(indicators)),
        rsi=_last_finite(indicators["rsi"]),
        macd=_last_finite(indicators["macd"]),
        macd_signal=_last_finite(indicators["macd_signal"]),
        macd_histogram=_last_finite(indicators["macd_histogram"]),
        sma=_last_finite(indicators["sma"]),
        ema=_last_finite(indicators["ema"]),
        bollinger_upper=_last_finite(indicators["bollinger_upper"]),
        bollinger_middle=_last_finite(indicators["bollinger_middle"]),
        bollinger_lower=_last_finite(indicators["bollinger_lower"]),
        stochastic_k=_last_finite(indicators["stochastic_k"]),
        stochastic_d=_last_finite(indicators["stochastic_d"]),
        atr=_last_finite(indicators["atr"]),
        adx=_last_finite(indicators["adx"]),
    )


def compute_technical_indicators(
    data: pd.Series | pd.DataFrame | Sequence[float],
    *,
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
) -> tuple[pd.DataFrame, TechnicalIndicatorSnapshot]:
    """Compute the technical indicator set on Series/DataFrame inputs.

    Returns:
        tuple[pd.DataFrame, TechnicalIndicatorSnapshot]:
            - DataFrame with one column per indicator and aligned index.
            - Snapshot with latest finite values.
    """
    _validate_positive_int("rsi_period", rsi_period)
    _validate_positive_int("macd_fast", macd_fast)
    _validate_positive_int("macd_slow", macd_slow)
    _validate_positive_int("macd_signal", macd_signal)
    _validate_positive_int("sma_period", sma_period)
    _validate_positive_int("ema_period", ema_period)
    _validate_positive_int("bollinger_window", bollinger_window)
    _validate_positive_int("stochastic_window", stochastic_window)
    _validate_positive_int("stochastic_smooth_window", stochastic_smooth_window)
    _validate_positive_int("atr_window", atr_window)
    _validate_positive_int("adx_window", adx_window)
    if bollinger_dev <= 0:
        raise ValueError("bollinger_dev must be > 0.")

    talib = _load_talib()
    frame = _normalize_price_input(data)
    if frame.empty:
        raise ValueError("No price data available to compute indicators.")

    close = frame["close"].to_numpy(dtype="float64")
    rsi = talib.RSI(close, timeperiod=rsi_period)
    macd, macd_signal_line, macd_hist = talib.MACD(
        close,
        fastperiod=macd_fast,
        slowperiod=macd_slow,
        signalperiod=macd_signal,
    )
    sma = talib.SMA(close, timeperiod=sma_period)
    ema = talib.EMA(close, timeperiod=ema_period)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        close,
        timeperiod=bollinger_window,
        nbdevup=bollinger_dev,
        nbdevdn=bollinger_dev,
    )

    nan_line = np.full(len(frame), np.nan, dtype="float64")
    stochastic_k = nan_line.copy()
    stochastic_d = nan_line.copy()
    atr = nan_line.copy()
    adx = nan_line.copy()

    if {"high", "low"}.issubset(frame.columns):
        high = frame["high"].to_numpy(dtype="float64")
        low = frame["low"].to_numpy(dtype="float64")
        stochastic_k, stochastic_d = talib.STOCH(
            high,
            low,
            close,
            fastk_period=stochastic_window,
            slowk_period=stochastic_smooth_window,
            slowd_period=stochastic_smooth_window,
        )
        atr = talib.ATR(high, low, close, timeperiod=atr_window)
        adx = talib.ADX(high, low, close, timeperiod=adx_window)

    indicators = pd.DataFrame(
        {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal_line,
            "macd_histogram": macd_hist,
            "sma": sma,
            "ema": ema,
            "bollinger_upper": bb_upper,
            "bollinger_middle": bb_middle,
            "bollinger_lower": bb_lower,
            "stochastic_k": stochastic_k,
            "stochastic_d": stochastic_d,
            "atr": atr,
            "adx": adx,
        },
        index=frame.index,
        dtype="float64",
    )
    return indicators, latest_indicator_snapshot(indicators)
