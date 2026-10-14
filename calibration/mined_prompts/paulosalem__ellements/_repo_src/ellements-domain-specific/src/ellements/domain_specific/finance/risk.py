"""Risk analytics and portfolio optimization utilities.

This module intentionally keeps risk assessment separate from valuation logic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
from pydantic import BaseModel


class RiskMetrics(BaseModel):
    """Collection of risk metrics for a return series."""

    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    beta: float | None = None
    annualized_return: float | None = None
    annualized_volatility: float | None = None
    value_at_risk_95: float | None = None
    value_at_risk_99: float | None = None
    conditional_value_at_risk_95: float | None = None
    conditional_value_at_risk_99: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    quantstats_used: bool = False


class PortfolioResult(BaseModel):
    """Portfolio return and risk calculation."""

    expected_return: float
    volatility: float
    sharpe_ratio: float | None = None
    weights: list[float]


class PortfolioOptimizationResult(BaseModel):
    """Mean-variance optimization result."""

    weights: list[float]
    expected_return: float
    volatility: float
    sharpe_ratio: float | None = None
    method: str
    target_return: float | None = None


def sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Sharpe ratio — risk-adjusted return."""

    arr = np.array(returns, dtype=float)
    if len(arr) < 2:
        return 0.0
    excess = arr - risk_free_rate
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std)


def sortino_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
) -> float:
    """Sortino ratio — downside-risk-adjusted return."""

    arr = np.array(returns, dtype=float)
    if len(arr) < 2:
        return 0.0
    downside = arr[arr < target_return] - target_return
    if len(downside) == 0:
        return float("inf") if np.mean(arr) > risk_free_rate else 0.0
    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std == 0:
        return 0.0
    return float((np.mean(arr) - risk_free_rate) / downside_std)


def calculate_beta(
    asset_returns: Sequence[float],
    market_returns: Sequence[float],
) -> float:
    """Beta — systematic risk relative to the market."""

    a = np.array(asset_returns, dtype=float)
    m = np.array(market_returns, dtype=float)
    if len(a) != len(m):
        raise ValueError("Asset and market return series must have equal length.")
    if len(a) < 2:
        raise ValueError("At least 2 return periods required.")

    cov_matrix = np.cov(a, m, ddof=1)
    var_market = cov_matrix[1, 1]
    if var_market == 0:
        return 0.0
    return float(cov_matrix[0, 1] / var_market)


def value_at_risk(
    returns: Sequence[float],
    confidence_level: float = 0.95,
    method: str = "historical",
) -> float:
    """Value at Risk as a positive loss threshold."""

    arr = np.array(returns, dtype=float)
    if len(arr) == 0:
        return 0.0

    percentile = (1 - confidence_level) * 100
    method_normalized = method.lower().strip()
    if method_normalized == "gaussian":
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        z = NormalDist().inv_cdf(1 - confidence_level)
        var = mean + z * std
        return float(max(-var, 0.0))

    var = np.percentile(arr, percentile)
    return float(max(-var, 0.0))


def conditional_value_at_risk(
    returns: Sequence[float],
    confidence_level: float = 0.95,
    method: str = "historical",
) -> float:
    """Conditional VaR (Expected Shortfall) as a positive expected tail loss."""

    arr = np.array(returns, dtype=float)
    if len(arr) == 0:
        return 0.0

    method_normalized = method.lower().strip()
    if method_normalized == "gaussian":
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        alpha = 1 - confidence_level
        if alpha <= 0 or std == 0:
            return max(-mean, 0.0)
        z = NormalDist().inv_cdf(alpha)
        pdf = math.exp(-(z**2) / 2) / math.sqrt(2 * math.pi)
        es = -(mean - std * (pdf / alpha))
        return float(max(es, 0.0))

    threshold = np.percentile(arr, (1 - confidence_level) * 100)
    tail = arr[arr <= threshold]
    if len(tail) == 0:
        return value_at_risk(arr.tolist(), confidence_level, method="historical")
    return float(max(-np.mean(tail), 0.0))


def max_drawdown(returns: Sequence[float]) -> float:
    """Maximum drawdown from a return series."""

    arr = np.array(returns, dtype=float)
    if len(arr) == 0:
        return 0.0
    cumulative = np.cumprod(1 + arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / np.maximum(running_max, 1e-12)
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def calmar_ratio(
    returns: Sequence[float],
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio = annualized return / max drawdown."""

    arr = np.array(returns, dtype=float)
    if len(arr) < 2:
        return 0.0
    ann_ret = float((np.prod(1 + arr) ** (periods_per_year / len(arr))) - 1)
    mdd = max_drawdown(arr.tolist())
    if mdd == 0:
        return 0.0
    return ann_ret / mdd


def risk_metrics(
    returns: Sequence[float],
    market_returns: Sequence[float] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    var_method: str = "historical",
) -> RiskMetrics:
    """Calculate a comprehensive set of risk metrics."""

    arr = np.array(returns, dtype=float)
    n = len(arr)
    if n < 2:
        return RiskMetrics()

    ann_ret = float((np.prod(1 + arr) ** (periods_per_year / n)) - 1)
    ann_vol = float(np.std(arr, ddof=1) * np.sqrt(periods_per_year))
    win_rate = float(np.mean(arr > 0))

    arr_list = arr.tolist()
    result = RiskMetrics(
        sharpe_ratio=sharpe_ratio(arr_list, risk_free_rate),
        sortino_ratio=sortino_ratio(arr_list, risk_free_rate),
        calmar_ratio=calmar_ratio(arr_list, periods_per_year=periods_per_year),
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        value_at_risk_95=value_at_risk(arr_list, 0.95, method=var_method),
        value_at_risk_99=value_at_risk(arr_list, 0.99, method=var_method),
        conditional_value_at_risk_95=conditional_value_at_risk(
            arr_list, 0.95, method=var_method
        ),
        conditional_value_at_risk_99=conditional_value_at_risk(
            arr_list, 0.99, method=var_method
        ),
        max_drawdown=max_drawdown(arr_list),
        win_rate=win_rate,
    )

    if market_returns is not None and len(market_returns) == n:
        result.beta = calculate_beta(arr_list, market_returns)

    return result


def portfolio_return(
    weights: Sequence[float],
    returns: Sequence[float],
) -> float:
    """Expected portfolio return from weights and expected returns."""

    w = np.array(weights, dtype=float)
    r = np.array(returns, dtype=float)
    if len(w) != len(r):
        raise ValueError("weights and returns must have the same length.")
    return float(np.dot(w, r))


def portfolio_volatility(
    weights: Sequence[float],
    cov_matrix: Sequence[Sequence[float]],
) -> float:
    """Portfolio volatility from weights and covariance matrix."""

    w = np.array(weights, dtype=float)
    cov = np.array(cov_matrix, dtype=float)
    variance = float(w @ cov @ w)
    return math.sqrt(max(variance, 0))


def portfolio_analysis(
    weights: Sequence[float],
    expected_returns: Sequence[float],
    cov_matrix: Sequence[Sequence[float]],
    risk_free_rate: float = 0.0,
) -> PortfolioResult:
    """Full portfolio return, volatility, and Sharpe ratio."""

    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    sr = (ret - risk_free_rate) / vol if vol > 0 else 0.0

    return PortfolioResult(
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sr,
        weights=list(weights),
    )


def optimize_portfolio_min_variance(
    expected_returns: Sequence[float],
    cov_matrix: Sequence[Sequence[float]],
    target_return: float | None = None,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
) -> PortfolioOptimizationResult:
    """Solve a simple mean-variance portfolio optimization problem."""

    mu = np.array(expected_returns, dtype=float)
    cov = np.array(cov_matrix, dtype=float)
    n = len(mu)
    if n == 0 or cov.shape != (n, n):
        raise ValueError("expected_returns and cov_matrix dimensions are inconsistent.")

    inv_cov = np.linalg.pinv(cov)
    ones = np.ones(n)

    if target_return is None:
        raw_weights = inv_cov @ ones
        method = "minimum_variance"
    else:
        a = float(ones @ inv_cov @ ones)
        b = float(ones @ inv_cov @ mu)
        c = float(mu @ inv_cov @ mu)
        det = a * c - b * b
        if abs(det) < 1e-12:
            raw_weights = inv_cov @ ones
            method = "minimum_variance_fallback"
        else:
            lam = (c - b * target_return) / det
            gam = (a * target_return - b) / det
            raw_weights = inv_cov @ (lam * ones + gam * mu)
            method = "minimum_variance_target_return"

    if not allow_short:
        raw_weights = np.clip(raw_weights, 0.0, None)

    total = float(np.sum(raw_weights))
    weights = (raw_weights / total) if total > 0 else np.full(n, 1.0 / n)

    result = portfolio_analysis(
        weights=weights.tolist(),
        expected_returns=mu.tolist(),
        cov_matrix=cov.tolist(),
        risk_free_rate=risk_free_rate,
    )
    return PortfolioOptimizationResult(
        weights=result.weights,
        expected_return=result.expected_return,
        volatility=result.volatility,
        sharpe_ratio=result.sharpe_ratio,
        method=method,
        target_return=target_return,
    )


def simulated_portfolio_scenarios(
    periods: int = 252,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Generate deterministic simulated return scenarios for testing and demos."""

    rng = np.random.default_rng(seed)
    balanced = rng.normal(0.0006, 0.0100, periods)
    concentrated_growth = rng.normal(0.0009, 0.0180, periods)
    defensive_income = rng.normal(0.0003, 0.0060, periods)
    high_volatility = rng.normal(0.0004, 0.0280, periods)

    first = periods // 2
    regime_shift = np.concatenate(
        [
            rng.normal(0.0009, 0.0090, first),
            rng.normal(-0.0006, 0.0180, periods - first),
        ]
    )

    return {
        "balanced": balanced.tolist(),
        "concentrated_growth": concentrated_growth.tolist(),
        "defensive_income": defensive_income.tolist(),
        "high_volatility": high_volatility.tolist(),
        "regime_shift": regime_shift.tolist(),
    }
