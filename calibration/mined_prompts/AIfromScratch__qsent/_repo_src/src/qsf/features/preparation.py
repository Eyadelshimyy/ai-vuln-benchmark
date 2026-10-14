"""
Data preparation: chronological splits, scaling, and data bundle assembly.

All scaling is fit on training data only to prevent information leakage.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_FRAC = 0.60
DEFAULT_VAL_FRAC = 0.20
DEFAULT_TEST_FRAC = 0.20

MAX_NAN_FRACTION = 0.10

EXCLUDE_OHLCV = {"Open", "High", "Low", "Close", "Volume", "Adj_Close", "Adj Close", "ret"}


def _select_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Identify feature and target columns."""
    target_cols = [c for c in df.columns if c.startswith("target_")]
    if feature_cols is None:
        excluded = set(target_cols) | EXCLUDE_OHLCV
        feature_cols = [c for c in df.columns if c not in excluded]
    return feature_cols, target_cols


def prepare_data_bundle(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    primary_target: str = "target_return",
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> dict[str, Any]:
    """Prepare train/val/test splits with leak-free scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with targets.
    feature_cols : list[str] or None
        If None, auto-detected by excluding targets and raw OHLCV.
    primary_target : str
        Regression target column name.
    train_frac, val_frac, test_frac : float
        Split fractions (must sum to 1.0).

    Returns
    -------
    dict
        Keys: X_train, X_val, X_test (scaled numpy arrays),
        y_train, y_val, y_test (unscaled numpy arrays),
        train_df, val_df, test_df (DataFrames),
        feature_cols, target_cols, primary_target, scaler.
    """
    feature_cols, target_cols = _select_features(df, feature_cols)

    # Clean NaNs
    nan_per_row = df[feature_cols].isna().sum(axis=1)
    max_nan = int(len(feature_cols) * MAX_NAN_FRACTION)
    df_clean = df[nan_per_row <= max_nan].dropna(subset=[primary_target]).copy()

    remaining_nans = df_clean[feature_cols].isna()
    if remaining_nans.any().any():
        df_clean[feature_cols] = df_clean[feature_cols].fillna(0.0)

    logger.info(
        "Data after cleaning: %d rows, %d features", len(df_clean), len(feature_cols)
    )

    # Chronological splits
    n = len(df_clean)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df_clean.iloc[:train_end].copy()
    val_df = df_clean.iloc[train_end:val_end].copy()
    test_df = df_clean.iloc[val_end:].copy()

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    y_train = train_df[primary_target].values
    y_val = val_df[primary_target].values
    y_test = test_df[primary_target].values

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "primary_target": primary_target,
        "scaler": scaler,
    }
