from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    values = series.astype(float)
    if ascending:
        ranks = values.rank(method="dense", ascending=True)
        max_rank = values.rank(method="dense", ascending=True).max()
    else:
        ranks = values.rank(method="min", ascending=False)
        max_rank = values.rank(method="min", ascending=False).max()
    if max_rank <= 1:
        return pd.Series(100.0, index=series.index)
    return ((ranks - 1) / (max_rank - 1)) * 100


def rank_desc_normalized(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="min", ascending=False)
    total = len(series)
    if total <= 1:
        return pd.Series(100.0, index=series.index)
    return ((total - ranks) / (total - 1)) * 100


def zscore(value: float, pool: pd.Series, clip: float = 2.0) -> float:
    med = float(pool.median())
    std = float(pool.std(ddof=1))
    if not std:
        return 0.0
    z = (value - med) / std
    return float(max(min(z, clip), -clip))


def composite_z(
    values: list[float],
    pools: list[pd.Series],
    weights: list[float],
    clip: float = 2.0,
) -> float:
    total = 0.0
    for value, pool, weight in zip(values, pools, weights):
        total += zscore(value, pool, clip=clip) * weight
    return total


def apply_bonus(base: float) -> float:
    if base > 90:
        return base * 1.10
    if base > 80:
        return base * 1.05
    return base


def normalize_rating_component(base: float) -> float:
    return 5 + apply_bonus(base) * 0.045


def rank_players(series: pd.Series) -> pd.Series:
    return series.rank(method="min", ascending=False).astype(int)


def zscore_linear_100(series: pd.Series, spread: float = 15.0) -> pd.Series:
    """Map pool values to 0–100 via z-score (50 = mean), preserving rank order."""
    values = series.astype(float)
    std = float(values.std(ddof=1))
    if not std:
        return pd.Series(50.0, index=series.index)
    z = (values - values.mean()) / std
    return (50 + z * spread).clip(0, 100)
