"""
Feature engineering pipeline ported from Phase 2 forecasting notebook.

Constructs technical indicators, cross-asset features, and macro transforms
from market data. Applies the "Grand Shift" to enforce strict causality:
features at row t use only data available at t-1.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RSI_PERIOD = 14
VOLATILITY_WINDOW = 20
ROC_WINDOWS = [5, 10, 20]
MA_WINDOWS = [5, 20, 50]
FEATURE_SHIFT = 1

EXCLUDE_FROM_GRAND_SHIFT = [
    "target_ret_next",
    "target_return",
    "target_dir_next",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def compute_rsi(price_series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = price_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period // 2).mean()
    avg_loss = loss.rolling(window=period, min_periods=period // 2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def engineer_features(
    df: pd.DataFrame,
    sentiment_daily: Optional[pd.DataFrame] = None,
    proxy_columns: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build feature set from market data and optional sentiment scores.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at minimum 'Close', 'Volume' columns with a DatetimeIndex.
    sentiment_daily : pd.DataFrame, optional
        Daily sentiment with columns like 'news_sentiment', 'social_sentiment'.
        Index should be dates matching df.
    proxy_columns : dict, optional
        Mapping of safe ticker name -> return column name already in df,
        e.g. {"QTUM": "ret_qtum"}.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (df with features + targets, list of feature column names)
    """
    df = df.copy()

    # --- Returns ---
    if "ret" in df.columns:
        df["ret_ionq"] = df["ret"]
    else:
        df["ret_ionq"] = df["Close"].pct_change()

    if proxy_columns:
        for safe_ticker, ret_col in proxy_columns.items():
            if ret_col in df.columns:
                pass  # already present
            elif safe_ticker in df.columns:
                df[ret_col] = df[safe_ticker].pct_change()

    # --- Technical indicators ---
    df["ionq_vol_20"] = df["ret_ionq"].rolling(
        window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW // 2
    ).std()

    df["ionq_rsi_14"] = compute_rsi(df["Close"], period=RSI_PERIOD)

    for window in ROC_WINDOWS:
        df[f"ionq_roc_{window}"] = df["Close"].pct_change(periods=window)

    for window in MA_WINDOWS:
        df[f"ionq_ma_{window}"] = df["Close"].rolling(
            window=window, min_periods=window // 2
        ).mean()

    df["ionq_dist_ma20"] = (df["Close"] - df["ionq_ma_20"]) / df["ionq_ma_20"]

    # --- Cross-asset spreads ---
    if proxy_columns:
        for safe_ticker, ret_col in proxy_columns.items():
            if ret_col in df.columns:
                df[f"spread_ionq_{safe_ticker.lower()}"] = df["ret_ionq"] - df[ret_col]

    # --- Macro transforms ---
    for indicator in ["VIX", "US10Y", "US3M", "TERM_SPREAD"]:
        if indicator in df.columns:
            df[f"d{indicator}"] = df[indicator].diff()

    if "VIX" in df.columns:
        df["VIX_squared"] = df["VIX"] ** 2

    # --- Sentiment features (from QSent pipeline) ---
    if sentiment_daily is not None:
        for col in sentiment_daily.columns:
            if col not in df.columns:
                df[col] = sentiment_daily[col].reindex(df.index)
        # Lagged sentiment to avoid look-ahead
        for col in sentiment_daily.columns:
            lag_col = f"{col}_lag1"
            if lag_col not in EXCLUDE_FROM_GRAND_SHIFT:
                EXCLUDE_FROM_GRAND_SHIFT.append(lag_col)
            df[lag_col] = df[col].shift(1)

    # --- Identify feature columns ---
    feature_cols = [
        col for col in df.columns
        if col not in EXCLUDE_FROM_GRAND_SHIFT
        and not col.startswith("target_")
        and col not in ["Open", "High", "Low", "Close", "Volume", "Adj_Close", "ret"]
    ]

    # --- Grand Shift: enforce causality ---
    logger.info("Applying Grand Shift to %d feature columns", len(feature_cols))
    for col in feature_cols:
        df[col] = df[col].shift(FEATURE_SHIFT)

    return df, feature_cols
