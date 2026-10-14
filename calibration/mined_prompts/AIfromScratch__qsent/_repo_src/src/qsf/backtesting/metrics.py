"""
Trading and risk metrics ported from the Phase 2 notebook.
"""
import numpy as np
import pandas as pd

ANNUALIZATION_FACTOR = 252.0


def compute_risk_metrics(strategy_returns: np.ndarray) -> dict[str, float]:
    """Compute comprehensive risk metrics from a strategy return series.

    Parameters
    ----------
    strategy_returns : np.ndarray
        Daily strategy returns (net of costs).

    Returns
    -------
    dict with keys: total_return_pct, sharpe, sortino, max_drawdown_pct,
    win_rate_pct, profit_factor, volatility_annual_pct.
    """
    equity = np.cumprod(1.0 + strategy_returns)
    total_return = float((equity[-1] - 1.0) * 100)

    mean_d = np.mean(strategy_returns)
    std_d = np.std(strategy_returns, ddof=0)

    sharpe = 0.0
    if std_d > 0:
        sharpe = float(
            (mean_d * ANNUALIZATION_FACTOR) / (std_d * np.sqrt(ANNUALIZATION_FACTOR))
        )

    downside = strategy_returns[strategy_returns < 0]
    downside_std = np.std(downside, ddof=0) if len(downside) > 0 else 0.0
    sortino = 0.0
    if downside_std > 0:
        sortino = float(
            (mean_d * ANNUALIZATION_FACTOR)
            / (downside_std * np.sqrt(ANNUALIZATION_FACTOR))
        )

    cummax = np.maximum.accumulate(equity)
    drawdowns = (equity - cummax) / cummax
    max_dd = float(np.min(drawdowns) * 100) if len(drawdowns) > 0 else 0.0

    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    win_rate = float(len(wins) / len(strategy_returns) * 100) if len(strategy_returns) > 0 else 0.0

    gross_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    vol_annual = float(std_d * np.sqrt(ANNUALIZATION_FACTOR) * 100)

    return {
        "total_return_pct": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "volatility_annual_pct": vol_annual,
    }


def simulate_strategy(
    actual_returns: np.ndarray,
    predictions: np.ndarray,
    cost_bps: float = 20.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Simulate a sign-based long/flat strategy with transaction costs.

    Parameters
    ----------
    actual_returns : np.ndarray
        Actual daily returns.
    predictions : np.ndarray
        Model predictions (positive = go long, else flat).
    cost_bps : float
        Round-trip transaction cost in basis points.

    Returns
    -------
    tuple[np.ndarray, dict]
        (strategy_returns, risk_metrics)
    """
    position = (predictions > 0).astype(float)
    turnover = np.abs(np.diff(position, prepend=position[0]))
    cost_per_side = (cost_bps / 2.0) / 10_000.0

    lagged_pos = np.roll(position, 1)
    lagged_pos[0] = 0.0

    strategy_returns = lagged_pos * actual_returns - turnover * cost_per_side
    metrics = compute_risk_metrics(strategy_returns)

    return strategy_returns, metrics
