"""Tests for quant_analysis optional QuantStats integration."""

from pathlib import Path

import pytest
from ellements.domain_specific.finance.quant_analysis import (
    QuantRiskSummary,
    align_return_series,
    generate_quantstats_tear_sheet,
    quantstats_available,
    risk_metrics_from_quant_summary,
    summarize_quant_risk,
)


def test_align_return_series_handles_mismatch():
    aligned = align_return_series(
        portfolio_returns=[0.01, 0.02, -0.01, 0.015],
        benchmark_returns=[0.005, 0.01, -0.008],
    )
    assert aligned.sample_size == 3
    assert len(aligned.portfolio_returns) == 3
    assert aligned.benchmark_returns is not None
    assert len(aligned.benchmark_returns) == 3


def test_quant_summary_fallback_metrics_are_present():
    summary = summarize_quant_risk(
        portfolio_returns=[0.01, -0.02, 0.015, 0.008, -0.005, 0.011],
        benchmark_returns=[0.008, -0.015, 0.01, 0.007, -0.004, 0.009],
        periods_per_year=252,
    )
    assert isinstance(summary, QuantRiskSummary)
    assert summary.sample_size == 6
    assert summary.value_at_risk_95 is not None
    assert summary.conditional_value_at_risk_95 is not None
    assert summary.conditional_value_at_risk_95 >= summary.value_at_risk_95
    assert summary.sharpe_ratio is not None


def test_quant_summary_to_risk_metrics_conversion():
    summary = QuantRiskSummary(
        sample_size=10,
        periods_per_year=252,
        quantstats_used=False,
        sharpe_ratio=1.2,
        value_at_risk_95=0.03,
        conditional_value_at_risk_95=0.04,
    )
    risk = risk_metrics_from_quant_summary(summary)
    assert risk.sharpe_ratio == pytest.approx(1.2)
    assert risk.value_at_risk_95 == pytest.approx(0.03)
    assert risk.conditional_value_at_risk_95 == pytest.approx(0.04)


@pytest.mark.skipif(not quantstats_available(), reason="quantstats not installed")
def test_generate_quantstats_tear_sheet(tmp_path: Path):
    output_path = tmp_path / "tearsheet.html"
    try:
        output = generate_quantstats_tear_sheet(
            portfolio_returns=[0.01, -0.02, 0.015, 0.01, -0.005, 0.007, 0.004, 0.009],
            benchmark_returns=[
                0.008,
                -0.015,
                0.012,
                0.008,
                -0.004,
                0.006,
                0.003,
                0.007,
            ],
            output_path=output_path,
        )
    except ValueError as exc:
        message = str(exc)
        if "Invalid frequency" in message or "frequency string" in message:
            pytest.skip(
                f"quantstats/pandas frequency-string mismatch in this "
                f"environment: {exc}"
            )
        raise
    assert Path(output).exists()
    assert Path(output).suffix == ".html"
