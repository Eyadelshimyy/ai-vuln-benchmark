"""Coverage for :mod:`fast_mlsirm.cli` error-handler, helper, and argv branches.

The CLI dispatch wraps every command's input loading in typed ``try/except``
blocks that print a sanitized message and return exit code 1 (or re-raise under
``FAST_MLSIRM_DEBUG``). These tests patch each command's loader to raise the
exact exception type its handlers catch, in both non-debug (returns 1) and
debug (re-raises) modes, and exercise the CLI helper functions directly.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm import cli
from fast_mlsirm.cli import main

TARGET_RF = "fast_mlsirm.cli._load_response_and_factors"
TARGET_NPY = "fast_mlsirm.cli._load_numpy_bounded"
TARGET_REPORT = "fast_mlsirm.cli.render_diagnostics_report"

SIM = ["simulate", "--persons", "4", "--dims", "1", "--items-per-dim", "2", "--out", "out"]
DFIT = ["diagnose-fit", "--responses", "r.npy", "--factors", "f.csv", "--params", "p.npz", "--out", "out"]
DDIM = ["diagnose-dimensions", "--responses", "r.npy", "--factors", "f.csv", "--out", "out"]
DPROC = ["diagnose-response-process", "--responses", "r.npy", "--probabilities", "p.npy", "--out", "out"]
DCAND = ["diagnose-response-candidates", "--responses", "r.npy", "--candidate", "x=c.npy", "--out", "out"]
DFIX = ["diagnose-fixed-item-calibration", "--responses", "r.npy", "--candidate", "x=c.npy", "--out", "out"]
DREP = ["render-report", "--diagnostics", "d.json", "--out", "o.html"]
FIT = ["fit", "--responses", "r.npy", "--factors", "f.csv", "--out", "out"]


def _fnf():
    """A FileNotFoundError carrying a filename for the ``e.filename`` print."""
    return FileNotFoundError(2, "missing", "r.npy")


def _val():
    """A ValueError standing in for invalid input data."""
    return ValueError("bad value")


def _rt():
    """A RuntimeError routed to each command's generic Exception handler."""
    return RuntimeError("boom")


def _os():
    """An OSError standing in for a failed simulation write."""
    return OSError("disk full")


# (id, argv, patch target, exception factory, stderr substring)
LOADER_CASES = [
    ("simulate_value", SIM, "fast_mlsirm.cli.simulate", _val, "Invalid configuration"),
    ("simulate_os", SIM, "fast_mlsirm.cli.simulate", _os, "Failed to save simulation"),
    ("dfit_fnf", DFIT, TARGET_RF, _fnf, "Could not find file"),
    ("dfit_val", DFIT, TARGET_RF, _val, "Invalid input data"),
    ("dfit_exc", DFIT, TARGET_RF, _rt, "Failed to load data"),
    ("ddim_fnf", DDIM, TARGET_RF, _fnf, "Could not find file"),
    ("ddim_val", DDIM, TARGET_RF, _val, "Invalid input data"),
    ("ddim_exc", DDIM, TARGET_RF, _rt, "Failed to load data"),
    ("dproc_fnf", DPROC, TARGET_NPY, _fnf, "Could not find file"),
    ("dproc_val", DPROC, TARGET_NPY, _val, "Invalid input data"),
    ("dproc_exc", DPROC, TARGET_NPY, _rt, "Failed to load data"),
    ("dcand_fnf", DCAND, TARGET_NPY, _fnf, "Could not find file"),
    ("dcand_val", DCAND, TARGET_NPY, _val, "Invalid input data"),
    ("dcand_exc", DCAND, TARGET_NPY, _rt, "Failed to load data"),
    ("dfix_fnf", DFIX, TARGET_NPY, _fnf, "Could not find file"),
    ("dfix_val", DFIX, TARGET_NPY, _val, "Invalid input data"),
    ("dfix_exc", DFIX, TARGET_NPY, _rt, "Failed to load data"),
    ("drep_fnf", DREP, TARGET_REPORT, _fnf, "Could not find file"),
    ("drep_val", DREP, TARGET_REPORT, _val, "Invalid diagnostics data"),
    ("drep_exc", DREP, TARGET_REPORT, _rt, "Failed to render report"),
    ("fit_exc", FIT, TARGET_RF, _rt, "Failed to load data"),
]


@pytest.mark.parametrize(
    ("argv", "target", "make_exc", "substr"),
    [case[1:] for case in LOADER_CASES],
    ids=[case[0] for case in LOADER_CASES],
)
def test_command_loader_error_returns_one(argv, target, make_exc, substr, capsys):
    with patch(target, side_effect=make_exc()):
        assert main(argv) == 1
    captured = capsys.readouterr()
    assert substr in captured.err
    assert "Traceback" not in captured.err


# Debug re-raise: every loader case above, plus the fit-load FileNotFoundError
# and ValueError debug branches whose non-debug side other tests already cover.
DEBUG_CASES = [(case[0], case[1], case[2], case[3]) for case in LOADER_CASES] + [
    ("fit_fnf_debug", FIT, TARGET_RF, _fnf),
    ("fit_val_debug", FIT, TARGET_RF, _val),
]


@pytest.mark.parametrize(
    ("argv", "target", "make_exc"),
    [case[1:] for case in DEBUG_CASES],
    ids=[case[0] for case in DEBUG_CASES],
)
def test_command_loader_error_debug_reraises(argv, target, make_exc, monkeypatch):
    monkeypatch.setenv("FAST_MLSIRM_DEBUG", "1")
    exc = make_exc()
    with patch(target, side_effect=exc):
        with pytest.raises(type(exc)):
            main(argv)


# --- fit() call handlers (ValueError / RuntimeError) -------------------------

def _valid_load():
    """Patch context returning a valid (responses, factors) pair for fit()."""
    responses = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
    return patch(TARGET_RF, return_value=(responses, np.array([0, 0])))


def test_fit_call_value_error_returns_one(capsys):
    with _valid_load(), patch("fast_mlsirm.cli.fit", side_effect=ValueError("cfg")):
        assert main(FIT) == 1
    assert "Invalid fit configuration" in capsys.readouterr().err


def test_fit_call_value_error_debug_reraises(monkeypatch):
    monkeypatch.setenv("FAST_MLSIRM_DEBUG", "1")
    with _valid_load(), patch("fast_mlsirm.cli.fit", side_effect=ValueError("cfg")):
        with pytest.raises(ValueError):
            main(FIT)


def test_fit_call_runtime_error_returns_one(capsys):
    with _valid_load(), patch("fast_mlsirm.cli.fit", side_effect=RuntimeError("nope")):
        assert main(FIT) == 1
    assert "Fit failed" in capsys.readouterr().err


def test_fit_call_runtime_error_debug_reraises(monkeypatch):
    monkeypatch.setenv("FAST_MLSIRM_DEBUG", "1")
    with _valid_load(), patch("fast_mlsirm.cli.fit", side_effect=RuntimeError("nope")):
        with pytest.raises(RuntimeError):
            main(FIT)


# --- top-level main() KeyboardInterrupt --------------------------------------

def test_keyboard_interrupt_returns_130(capsys):
    with patch("fast_mlsirm.cli.simulate", side_effect=KeyboardInterrupt):
        assert main(SIM) == 130
    assert "Interrupted by user" in capsys.readouterr().err


# --- argv handling -----------------------------------------------------------

def test_main_empty_list_prints_help_returns_two(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


# --- _validate_response_and_factors ------------------------------------------

def test_validate_rejects_non_2d_responses():
    with pytest.raises(ValueError, match="responses must be a 2D"):
        cli._validate_response_and_factors(np.zeros(3), np.array([0, 0, 0]))


def test_validate_rejects_non_1d_factors():
    with pytest.raises(ValueError, match="factor_id must be a 1D"):
        cli._validate_response_and_factors(np.zeros((2, 3)), np.zeros((3, 1)))


# --- _load_fit_context -------------------------------------------------------

def test_load_fit_context_without_summary(tmp_path):
    params = tmp_path / "params.npz"
    np.savez(params, theta=np.zeros((1, 1)))
    assert cli._load_fit_context(params) == (None, None, None)


def test_load_fit_context_empty_convergence_status(tmp_path):
    params = tmp_path / "params.npz"
    np.savez(params, theta=np.zeros((1, 1)))
    (tmp_path / "fit_summary.json").write_text(
        json.dumps({"optimizer": "", "convergence_status": ""}), encoding="utf-8"
    )
    estimator, population, convergence = cli._load_fit_context(params)
    assert estimator is None
    assert population is None
    assert convergence is None


def test_load_fit_context_population_without_arrays(tmp_path):
    # population present but no pop_mu / pop_sigma in the params archive: the
    # two ``in arrays`` guards take their skip branches.
    params = tmp_path / "params.npz"
    np.savez(params, theta=np.zeros((1, 1)))
    (tmp_path / "fit_summary.json").write_text(
        json.dumps(
            {
                "optimizer": "adam_lbfgs",
                "convergence_status": "converged",
                "population": {"kind": "multigroup"},
            }
        ),
        encoding="utf-8",
    )
    estimator, population, convergence = cli._load_fit_context(params)
    assert estimator == "jmle"
    assert population == {"kind": "multigroup"}
    assert convergence == "converged"


def test_load_fit_context_reads_population_arrays(tmp_path):
    params = tmp_path / "params.npz"
    np.savez(
        params,
        theta=np.zeros((1, 1)),
        pop_mu=np.array([0.5]),
        pop_sigma=np.array([1.5]),
    )
    (tmp_path / "fit_summary.json").write_text(
        json.dumps(
            {
                "optimizer": "mmle_em/numpy",
                "convergence_status": "Converged",
                "population": {"kind": "multigroup"},
            }
        ),
        encoding="utf-8",
    )
    estimator, population, convergence = cli._load_fit_context(params)
    assert estimator == "mmle"
    assert convergence == "converged"
    assert np.allclose(population["mu"], [0.5])
    assert np.allclose(population["sigma"], [1.5])


# --- _load_candidate_probabilities -------------------------------------------

def test_load_candidates_rejects_too_many():
    specs = [f"c{i}=/nonexistent{i}.npy" for i in range(cli.MAX_CANDIDATE_COUNT + 1)]
    with pytest.raises(ValueError, match="candidate count exceeds"):
        cli._load_candidate_probabilities(specs)


def test_load_candidates_rejects_empty_label():
    with pytest.raises(ValueError, match="candidate label must not be empty"):
        cli._load_candidate_probabilities(["=/some/path.npy"])


def test_load_candidates_rejects_npz_archive(tmp_path):
    archive = tmp_path / "multi.npz"
    np.savez(archive, a=np.zeros(2), b=np.ones(2))
    with pytest.raises(ValueError, match=r"must be single \.npy arrays"):
        cli._load_candidate_probabilities([f"lab={archive}"])


def test_load_candidates_rejects_aggregate_size(tmp_path, monkeypatch):
    array_path = tmp_path / "cand.npy"
    np.save(array_path, np.zeros((2, 2)))
    monkeypatch.setattr(cli, "MAX_CANDIDATE_ELEMENTS", 1)
    with pytest.raises(ValueError, match="exceed the aggregate size limit"):
        cli._load_candidate_probabilities([f"lab={array_path}"])
