"""Contracts for the safe marginal-distance engineering benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np


def _load_benchmark_module():
    """Load the repository benchmark without requiring it to be a Python package."""
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_marginal_distance_workspaces.py"
    spec = importlib.util.spec_from_file_location("marginal_distance_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_reports_stable_kernel_and_exact_pairwise_peak() -> None:
    """Benchmark metadata must describe the final coordinate-subtraction kernel."""
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        n_left=4,
        n_right=5,
        latent_dim=2,
        warmups=1,
        repetitions=1,
        seed=563,
        output=None,
    )

    report = benchmark.build_report(args)

    assert "bounded_coordinate_subtraction" in report
    assert "bounded_squared_norm" not in report
    assert (
        "eliminates_the_identified_high_offset_cancellation_mechanism"
        in report["claims"]
    )
    assert report["inputs"]["pairwise_kernel_peak_bytes"] == 2 * 4 * 5 * np.dtype(np.float64).itemsize
    assert report["inputs"]["pairwise_output_bytes"] == 4 * 5 * np.dtype(np.float64).itemsize
    assert report["inputs"]["private_ceiling_bytes"] == benchmark.MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES


def test_benchmark_reference_comparison_remains_below_safety_ceiling() -> None:
    """Small benchmark runs compare stable and explicit-broadcast distances exactly enough."""
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        n_left=3,
        n_right=7,
        latent_dim=3,
        warmups=1,
        repetitions=1,
        seed=563,
        output=None,
    )

    report = benchmark.build_report(args)

    assert report["legacy_broadcast"] is not None
    assert report["legacy_broadcast"]["maximum_absolute_difference"] <= 1e-12


def test_numpy_configuration_supports_legacy_show_config(monkeypatch) -> None:
    benchmark = _load_benchmark_module()
    configuration_calls = []

    def legacy_show_config():
        configuration_calls.append(None)
        print("configuration " + "x" * 32768 + " complete legacy configuration")

    monkeypatch.setattr(benchmark.np, "show_config", legacy_show_config)
    configuration_text = benchmark._numpy_configuration()
    assert configuration_text.endswith("complete legacy configuration")
    assert len(configuration_text) > 32768
    assert configuration_calls == [None]


def test_numpy_configuration_modern_mode_is_warning_free() -> None:
    benchmark = _load_benchmark_module()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        configuration_text = benchmark._numpy_configuration()
    assert configuration_text


def test_numpy_configuration_preserves_modern_dictionary(monkeypatch) -> None:
    """Keep the requested mode and returned build metadata in the report."""
    import ast

    benchmark = _load_benchmark_module()
    configuration_calls = []
    build_configuration = {"Build Dependencies": {"blas": {"name": "fixture-blas"}}}

    def modern_show_config(*, mode):
        configuration_calls.append(mode)
        return build_configuration

    monkeypatch.setattr(benchmark.np, "show_config", modern_show_config)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        configuration_text = benchmark._numpy_configuration()
    assert configuration_calls == ["dicts"]
    assert ast.literal_eval(configuration_text) == build_configuration
