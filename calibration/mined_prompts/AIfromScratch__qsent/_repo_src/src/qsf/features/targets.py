"""
Target variable creation for forecasting models.

Creates regression (next-day return) and classification (direction) targets.
Optionally creates a ternary target with a dead zone around zero.
"""
import numpy as np
import pandas as pd

DEAD_ZONE_STD_MULTIPLIER = 0.5


def create_targets(
    df: pd.DataFrame,
    return_col: str = "ret_ionq",
    dead_zone_multiplier: float = DEAD_ZONE_STD_MULTIPLIER,
) -> tuple[pd.DataFrame, float]:
    """Create prediction targets from return series.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``return_col``.
    return_col : str
        Column with daily returns.
    dead_zone_multiplier : float
        Fraction of return std to use as dead-zone threshold for ternary target.

    Returns
    -------
    tuple[pd.DataFrame, float]
        (df with target columns added, dead-zone threshold tau)
    """
    df = df.copy()

    df["target_ret_next"] = df[return_col].shift(-1)
    df["target_return"] = df["target_ret_next"]
    df["target_dir_next"] = (df["target_ret_next"] > 0).astype(int)

    tau = df[return_col].std() * dead_zone_multiplier
    df["target_ternary"] = 0
    df.loc[df["target_ret_next"] > tau, "target_ternary"] = 1
    df.loc[df["target_ret_next"] < -tau, "target_ternary"] = -1

    df = df.dropna(subset=["target_ret_next"])

    return df, tau
