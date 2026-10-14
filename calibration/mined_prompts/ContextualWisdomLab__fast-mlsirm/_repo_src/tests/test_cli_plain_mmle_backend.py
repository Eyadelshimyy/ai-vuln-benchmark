"""CLI regressions for exact resolved-backend reporting."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

from fast_mlsirm.cli import main


def test_cli_plain_unidimensional_mmle_auto_reports_rust(
    tmp_path, capsys, monkeypatch
) -> None:
    """Plain MMLE must report the resolved Rust backend, never the selector ``auto``."""

    pytest.importorskip("fast_mlsirm._core")
    monkeypatch.chdir(tmp_path)
    sim_dir = tmp_path / "sim_out"
    fit_dir = tmp_path / "fit_out"

    with patch.object(
        sys,
        "argv",
        [
            "fast-mlsirm",
            "simulate",
            "--persons",
            "50",
            "--dims",
            "1",
            "--items-per-dim",
            "2",
            "--out",
            str(sim_dir),
        ],
    ):
        assert main() == 0
    capsys.readouterr()

    args = [
        "fit",
        "--responses",
        str(sim_dir / "responses.npy"),
        "--factors",
        str(sim_dir / "item_factor.csv"),
        "--model",
        "ULS2PLM",
        "--estimator",
        "mmle",
        "--max-iter",
        "1",
        "--backend",
        "auto",
        "--out",
        str(fit_dir),
        "--json",
    ]
    with patch.object(sys, "argv", ["fast-mlsirm", *args]):
        assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "rust"
    summary = json.loads((fit_dir / "fit_summary.json").read_text(encoding="utf-8"))
    assert summary["backend"] == "rust"
