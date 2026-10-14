"""Quantitative portfolio analytics with optional QuantStats integration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .risk import RiskMetrics, risk_metrics


def quantstats_available() -> bool:
    """Return True if quantstats is importable."""

    try:
        import quantstats  # noqa: F401

        return True
    except Exception:
        return False


class QuantRiskSummary(BaseModel):
    """Quantitative risk summary, with or without QuantStats backing."""

    sample_size: int
    periods_per_year: int
    quantstats_used: bool = False
    benchmark_symbol: str | None = None
    annualized_return: float | None = None
    annualized_volatility: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    max_drawdown: float | None = None
    value_at_risk_95: float | None = None
    value_at_risk_99: float | None = None
    conditional_value_at_risk_95: float | None = None
    conditional_value_at_risk_99: float | None = None
    win_rate: float | None = None
    beta: float | None = None


class AlignedReturnSeries(BaseModel):
    """Aligned portfolio and benchmark return series."""

    portfolio_returns: list[float] = Field(default_factory=list)
    benchmark_returns: list[float] | None = None
    sample_size: int = 0


def _to_series(returns: Sequence[float] | pd.Series, name: str) -> pd.Series:
    if isinstance(returns, pd.Series):
        series = returns.astype(float).copy()
    else:
        series = pd.Series(list(returns), dtype="float64", name=name)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.name is None:
        series.name = name
    return series


def align_return_series(
    portfolio_returns: Sequence[float] | pd.Series,
    benchmark_returns: Sequence[float] | pd.Series | None = None,
) -> AlignedReturnSeries:
    """Align return series and remove NaN/inf values."""

    p = _to_series(portfolio_returns, "portfolio")
    if benchmark_returns is None:
        return AlignedReturnSeries(
            portfolio_returns=p.tolist(),
            benchmark_returns=None,
            sample_size=len(p),
        )

    b = _to_series(benchmark_returns, "benchmark")
    if isinstance(portfolio_returns, pd.Series) and isinstance(
        benchmark_returns, pd.Series
    ):
        aligned = pd.concat([p, b], axis=1, join="inner").dropna()
        p = aligned.iloc[:, 0]
        b = aligned.iloc[:, 1]
    else:
        n = min(len(p), len(b))
        p = p.iloc[-n:] if n > 0 else p.iloc[:0]
        b = b.iloc[-n:] if n > 0 else b.iloc[:0]

    return AlignedReturnSeries(
        portfolio_returns=p.tolist(),
        benchmark_returns=b.tolist(),
        sample_size=len(p),
    )


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)  # type: ignore[arg-type]
        if np.isnan(result):
            return None
        return result
    except Exception:
        return None


def summarize_quant_risk(
    portfolio_returns: Sequence[float] | pd.Series,
    benchmark_returns: Sequence[float] | pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    benchmark_symbol: str | None = None,
    var_method: str = "historical",
) -> QuantRiskSummary:
    """Summarize risk metrics, using QuantStats when available."""

    aligned = align_return_series(portfolio_returns, benchmark_returns)
    market = aligned.benchmark_returns

    base = risk_metrics(
        aligned.portfolio_returns,
        market_returns=market,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        var_method=var_method,
    )

    summary = QuantRiskSummary(
        sample_size=aligned.sample_size,
        periods_per_year=periods_per_year,
        quantstats_used=False,
        benchmark_symbol=benchmark_symbol,
        annualized_return=base.annualized_return,
        annualized_volatility=base.annualized_volatility,
        sharpe_ratio=base.sharpe_ratio,
        sortino_ratio=base.sortino_ratio,
        calmar_ratio=base.calmar_ratio,
        max_drawdown=base.max_drawdown,
        value_at_risk_95=base.value_at_risk_95,
        value_at_risk_99=base.value_at_risk_99,
        conditional_value_at_risk_95=base.conditional_value_at_risk_95,
        conditional_value_at_risk_99=base.conditional_value_at_risk_99,
        win_rate=base.win_rate,
        beta=base.beta,
    )

    if not quantstats_available() or aligned.sample_size < 2:
        return summary

    try:
        import quantstats as qs

        series = _to_series(aligned.portfolio_returns, "portfolio")
        benchmark_series = (
            _to_series(market, "benchmark") if market is not None else None
        )

        qs_sharpe = _safe_float(
            qs.stats.sharpe(series, rf=risk_free_rate, periods=periods_per_year)
        )
        qs_sortino = _safe_float(
            qs.stats.sortino(series, rf=risk_free_rate, periods=periods_per_year)
        )
        qs_calmar = _safe_float(qs.stats.calmar(series))
        qs_cagr = _safe_float(qs.stats.cagr(series, periods=periods_per_year))
        qs_vol = _safe_float(qs.stats.volatility(series, periods=periods_per_year))
        qs_mdd = _safe_float(abs(qs.stats.max_drawdown(series)))
        qs_win_rate = _safe_float(qs.stats.win_rate(series))
        qs_beta = (
            _safe_float(qs.stats.beta(series, benchmark_series))
            if benchmark_series is not None
            else summary.beta
        )

        summary.sharpe_ratio = (
            qs_sharpe if qs_sharpe is not None else summary.sharpe_ratio
        )
        summary.sortino_ratio = (
            qs_sortino if qs_sortino is not None else summary.sortino_ratio
        )
        summary.calmar_ratio = (
            qs_calmar if qs_calmar is not None else summary.calmar_ratio
        )
        summary.annualized_return = (
            qs_cagr if qs_cagr is not None else summary.annualized_return
        )
        summary.annualized_volatility = (
            qs_vol if qs_vol is not None else summary.annualized_volatility
        )
        summary.max_drawdown = qs_mdd if qs_mdd is not None else summary.max_drawdown
        summary.win_rate = qs_win_rate if qs_win_rate is not None else summary.win_rate
        summary.beta = qs_beta if qs_beta is not None else summary.beta
        summary.quantstats_used = True
        return summary
    except Exception:
        return summary


def risk_metrics_from_quant_summary(summary: QuantRiskSummary) -> RiskMetrics:
    """Convert a QuantRiskSummary into the generic RiskMetrics model."""

    return RiskMetrics(
        sharpe_ratio=summary.sharpe_ratio,
        sortino_ratio=summary.sortino_ratio,
        calmar_ratio=summary.calmar_ratio,
        beta=summary.beta,
        annualized_return=summary.annualized_return,
        annualized_volatility=summary.annualized_volatility,
        value_at_risk_95=summary.value_at_risk_95,
        value_at_risk_99=summary.value_at_risk_99,
        conditional_value_at_risk_95=summary.conditional_value_at_risk_95,
        conditional_value_at_risk_99=summary.conditional_value_at_risk_99,
        max_drawdown=summary.max_drawdown,
        win_rate=summary.win_rate,
        quantstats_used=summary.quantstats_used,
    )


def _ensure_datetime_index(series: pd.Series) -> pd.Series:
    """Return *series* with a ``DatetimeIndex``.

    quantstats reporting helpers require a ``DatetimeIndex`` so they can
    format the report's date range. When the caller supplied a plain
    integer index (the common case for synthetic return streams), we
    synthesize a daily business-day index ending today.
    """
    if isinstance(series.index, pd.DatetimeIndex):
        return series
    new_index = pd.bdate_range(
        end=pd.Timestamp.today().normalize(), periods=len(series)
    )
    return series.set_axis(new_index)


def generate_quantstats_tear_sheet(
    portfolio_returns: Sequence[float] | pd.Series,
    benchmark_returns: Sequence[float] | pd.Series | None = None,
    output_path: str | Path | None = None,
    title: str = "Portfolio Risk Tear Sheet",
) -> str:
    """Generate an HTML tear sheet and return path to generated file."""

    if not quantstats_available():
        raise RuntimeError("quantstats is not available in the current environment.")

    import quantstats as qs

    aligned = align_return_series(portfolio_returns, benchmark_returns)
    if aligned.sample_size < 2:
        raise ValueError("At least 2 return periods are required for a tear sheet.")

    portfolio_series = _ensure_datetime_index(
        _to_series(aligned.portfolio_returns, "portfolio")
    )
    benchmark_series = (
        _ensure_datetime_index(_to_series(aligned.benchmark_returns, "benchmark"))
        if aligned.benchmark_returns is not None
        else None
    )

    if output_path is None:
        with NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            output = Path(tmp.name)
    else:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

    qs.reports.html(
        portfolio_series,
        benchmark=benchmark_series,
        output=str(output),
        title=title,
    )
    return str(output)
