from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .config import FitConfig, MLS2PLMConfig
from .diagnostics import (
    dimensionality_diagnostics,
    fit_diagnostics,
    fixed_item_calibration_diagnostics,
    response_process_dimensionality_diagnostics,
    response_process_fit_diagnostics,
)
from .fit import fit
from .reference import fit_reference
from .irt_contract import fit_irt_experiment
from .io import (
    _atomic_write_text,
    _load_json_bounded,
    _load_numpy_bounded,
    load_factor_csv,
    load_params,
    save_dimensionality_diagnostics,
    save_fit_diagnostics,
    save_fit_result,
    save_simulation,
)
from .report import render_diagnostics_report
from .simulation import simulate


MAX_CANDIDATE_COUNT = 128
MAX_CANDIDATE_ELEMENTS = 50_000_000
MAX_CANDIDATE_BYTES = 512 * 1024 * 1024


def _confine_cli_path(path: str | Path, *, kind: str) -> str:
    """Keep a CLI path inside the caller's current working directory.

    The lexical path is returned so bounded input readers can still reject a
    leaf symlink with ``O_NOFOLLOW``.  Canonical resolution is used only for
    the containment check, which also rejects ``..`` and symlinked parents.
    """
    requested = os.fspath(path)
    if not requested.strip():
        raise ValueError(
            f"{kind} path must not be empty; pass the file or directory "
            "path after the matching option"
        )
    try:
        root = Path.cwd().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            "current working directory could not be resolved safely; rerun "
            "the command from a valid working directory"
        ) from exc
    lexical = Path(os.path.abspath(requested))
    try:
        resolved = lexical.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"{kind} path could not be resolved safely; check that the path "
            "and its parent folders exist and are readable, then retry"
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            f"{kind} path must remain within the current working directory; "
            "move or copy the file under the current working directory (or "
            "rerun the command from its folder), then retry"
        ) from None
    return requested


def _confine_candidate_specs(specs: list[str]) -> list[str]:
    """Confine every ``label=path`` candidate while preserving its label."""
    confined: list[str] = []
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
            if not label:
                raise ValueError(
                    "candidate label must not be empty; pass candidates as "
                    "label=path.npy"
                )
            confined.append(
                f"{label}={_confine_cli_path(path, kind='candidate probability')}"
            )
        else:
            confined.append(
                _confine_cli_path(spec, kind="candidate probability")
            )
    return confined


def _confine_cli_paths(args: argparse.Namespace) -> None:
    """Apply one workspace boundary to every path accepted by the CLI."""
    input_fields = {
        "fit": ("responses", "factors", "group_id", "cluster_id"),
        "score": ("bundle", "responses"),
        "diagnose-fit": ("responses", "factors", "params", "group_id", "cluster_id"),
        "diagnose-dimensions": ("responses", "factors"),
        "diagnose-response-process": (
            "responses",
            "probabilities",
            "group_id",
            "cluster_id",
        ),
        "diagnose-response-candidates": ("responses",),
        "diagnose-fixed-item-calibration": ("responses", "fixed_items"),
        "render-report": ("diagnostics",),
    }
    for field in input_fields.get(args.command, ()):
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, _confine_cli_path(value, kind=field))
    if hasattr(args, "candidate"):
        args.candidate = _confine_candidate_specs(args.candidate)
    if hasattr(args, "out") and args.out is not None:
        args.out = _confine_cli_path(args.out, kind="output")


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    """Register the shared ``--json`` machine-readable-output flag on a parser."""
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write one machine-readable JSON status object to stdout.",
    )


def _load_fit_context(
    params_path: str | Path,
) -> tuple[str | None, dict | None, str | None]:
    """Recover estimator, population, and convergence metadata beside params."""
    path = Path(params_path)
    summary_path = path.with_name("fit_summary.json")
    if not summary_path.exists():
        return None, None, None
    summary = _load_json_bounded(summary_path, source="fit summary")
    optimizer = str(summary.get("optimizer", "")).lower()
    estimator = "mmle" if optimizer.startswith("mmle") else "jmle" if optimizer else None
    raw_status = summary.get("convergence_status")
    convergence_status = (
        str(raw_status).strip().lower() if raw_status is not None else None
    )
    if not convergence_status:
        convergence_status = None
    population = summary.get("population")
    if population is not None:
        population = dict(population)
        with _load_numpy_bounded(path) as arrays:
            if "pop_mu" in arrays:
                population["mu"] = np.asarray(arrays["pop_mu"], dtype=float)
            if "pop_sigma" in arrays:
                population["sigma"] = np.asarray(arrays["pop_sigma"], dtype=float)
    return estimator, population, convergence_status


def _progress(args: argparse.Namespace, message: str) -> None:
    """Print a human-readable progress line unless ``--json`` mode is active."""
    if not getattr(args, "json", False):
        print(message)


def _complete(args: argparse.Namespace, message: str, payload: dict[str, object]) -> int:
    """Emit the command result and return exit code 0.

    Prints the JSON ``payload`` in ``--json`` mode, otherwise the human-readable
    ``message``.
    """
    if getattr(args, "json", False):
        print(json.dumps(payload, sort_keys=True))
    else:
        print(message)
    return 0


def _output_file(run_dir: str, filename: str) -> str:
    """Join an output ``run_dir`` and ``filename`` into a path string."""
    return str(Path(run_dir) / filename)


def _load_response_and_factors(responses_path: str, factors_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and cross-validate the response ``.npy`` and factor CSV inputs."""
    responses = _load_numpy_bounded(responses_path)
    factors = load_factor_csv(factors_path)
    _validate_response_and_factors(responses, factors)
    return responses, factors


def _validate_response_and_factors(responses: np.ndarray, factors: np.ndarray) -> None:
    """Check the response matrix is 2D and the factor vector matches item count."""
    if responses.ndim != 2:
        raise ValueError("responses must be a 2D persons x items array")
    if factors.ndim != 1:
        raise ValueError("factor_id must be a 1D item vector")
    if factors.shape[0] != responses.shape[1]:
        raise ValueError(
            f"factor_id length ({factors.shape[0]}) must match response item count ({responses.shape[1]})"
        )


def _main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand.

    Builds the full ``fast-mlsirm`` argument parser (simulate, fit, and the
    diagnose/render subcommands), runs the chosen command, and returns its
    process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="fast-mlsirm",
        description="Fast simulation, fitting, and recovery diagnostics for MLSIRM/MLS2PLM models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser(
        "simulate",
        help="Simulate binary responses for the MLS2PLM model.",
        description="Simulate binary responses for the MLS2PLM model.",
    )
    sim.add_argument("--persons", type=int, default=500, help="Number of persons to simulate (default: 500).")
    sim.add_argument("--dims", type=int, default=2, help="Number of true item dimensions (default: 2).")
    sim.add_argument("--items-per-dim", type=int, default=8, help="Number of items per dimension (default: 8).")
    sim.add_argument("--latent-dim", type=int, default=2, help="Latent dimensionality for person traits (default: 2).")
    sim.add_argument("--phi", type=float, default=0.3, help="Variance of item intercept factors (default: 0.3).")
    sim.add_argument("--gamma", type=float, default=1.5, help="Variance of person trait coordinates (default: 1.5).")
    sim.add_argument("--seed", type=int, default=1, help="Random seed for simulation (default: 1).")
    sim.add_argument("--out", required=True, help="Directory path to save simulated output (responses, factors, etc.).")
    _add_json_flag(sim)

    fit_cmd = sub.add_parser(
        "fit",
        help="Fit an MLSIRM model to response data.",
        description="Fit an MLSIRM model to response data.",
    )
    fit_cmd.add_argument("--responses", required=True, help="Path to the responses numpy array file (.npy).")
    fit_cmd.add_argument("--factors", required=True, help="Path to the item factors CSV file.")
    fit_cmd.add_argument("--model", default="MLS2PLM", help="Model type to fit (default: MLS2PLM).")
    fit_cmd.add_argument("--latent-dim", type=int, default=2, help="Latent dimensionality for person traits (default: 2).")
    fit_cmd.add_argument("--estimator", choices=["jmle", "mmle"], default="jmle", help="Estimator: penalized joint MLE, or marginal MLE via EM (persons integrated out; default: jmle).")
    fit_cmd.add_argument("--group-id", help="Optional .npy person group IDs: estimation-level multigroup calibration (estimator=mmle; group 0 is the N(0,1) reference).")
    fit_cmd.add_argument("--cluster-id", help="Optional .npy person cluster IDs: estimation-level multilevel random intercept (estimator=mmle).")
    fit_cmd.add_argument("--q-theta", type=int, default=21, help="Marginal estimator: Gauss-Hermite nodes per trait dimension (default: 21).")
    fit_cmd.add_argument("--q-xi", type=int, default=11, help="Marginal estimator: Gauss-Hermite nodes per latent-space axis (default: 11).")
    fit_cmd.add_argument("--q-u", type=int, default=15, help="Marginal estimator: Gauss-Hermite nodes for the multilevel intercept (default: 15).")
    fit_cmd.add_argument("--tolerance", type=float, default=1e-6, help="Convergence tolerance (default: 1e-6).")
    fit_cmd.add_argument("--optimizer", choices=["adam", "lbfgs", "adam_lbfgs"], default="adam_lbfgs", help="Optimizer to use (default: adam_lbfgs).")
    fit_cmd.add_argument("--max-iter", type=int, default=100, help="Maximum number of iterations for the optimizer (default: 100).")
    fit_cmd.add_argument("--n-restarts", type=int, default=1, help="Number of random restarts (default: 1).")
    fit_cmd.add_argument("--seed", type=int, default=1, help="Random seed for fitting (default: 1).")
    fit_cmd.add_argument("--backend", choices=["rust", "auto"], default="auto", help="Rust numerical owner to use (default: auto; fails closed otherwise; use --reference for the explicit NumPy parity path).")
    fit_cmd.add_argument("--reference", action="store_true", help="Use the explicit non-production NumPy parity reference.")
    fit_cmd.add_argument("--rust-device", choices=["auto", "cpu", "gpu"], default="auto", help="Execution device for the Rust backend (default: auto). Ignored by --reference.")
    fit_cmd.add_argument("--out", required=True, help="Directory path to save the fitted parameters.")
    _add_json_flag(fit_cmd)

    score_cmd = sub.add_parser(
        "score",
        help="Score new respondents against a frozen serving bundle (EAP).",
        description="Score new respondents against a frozen serving bundle (EAP, item parameters fixed).",
    )
    score_cmd.add_argument("--bundle", required=True, help="Path to a serving bundle JSON (see fast_mlsirm.serving.export_serving_bundle).")
    score_cmd.add_argument("--responses", required=True, help="Responses: a JSON file (dict or list of dicts mapping item code -> 0/1) or a .npy matrix in bundle item order (NaN = missing).")
    score_cmd.add_argument("--out", help="Optional path for the scores JSON (default: stdout).")
    _add_json_flag(score_cmd)

    diagnose = sub.add_parser(
        "diagnose-fit",
        help="Compute item, person, and model fit diagnostics for fitted parameters.",
        description="Compute item, person, and model fit diagnostics for fitted parameters.",
    )
    diagnose.add_argument("--responses", required=True, help="Path to the responses numpy array file (.npy).")
    diagnose.add_argument("--factors", required=True, help="Path to the item factors CSV file.")
    diagnose.add_argument("--params", required=True, help="Path to fitted params.npz.")
    diagnose.add_argument("--model", default="MLS2PLM", help="Model type used for the fitted parameters (default: MLS2PLM).")
    diagnose.add_argument(
        "--estimator",
        choices=["jmle", "cmle", "mmle"],
        help="Estimator used for calibration; inferred from fit_summary.json when omitted.",
    )
    diagnose.add_argument("--group-id", help="Optional .npy person group IDs for multigroup summaries.")
    diagnose.add_argument("--cluster-id", help="Optional .npy person cluster IDs for multilevel summaries.")
    diagnose.add_argument(
        "--limited-information",
        action="store_true",
        help=(
            "Also compute M2, RMSEA, SRMR, CFI, and TLI. Multiple-group fits "
            "use stacked moments; multilevel fits use cluster-robust covariance."
        ),
    )
    diagnose.add_argument("--out", required=True, help="Directory path to save fit_diagnostics.json.")
    _add_json_flag(diagnose)

    dim = sub.add_parser(
        "diagnose-dimensions",
        help="Compare latent-space dimensionality with K-fold held-out likelihood.",
        description="Compare latent-space dimensionality with K-fold held-out likelihood.",
    )
    dim.add_argument("--responses", required=True, help="Path to the responses numpy array file (.npy).")
    dim.add_argument("--factors", required=True, help="Path to the item factors CSV file.")
    dim.add_argument("--latent-dims", default="1,2,3", help="Comma-separated latent dimensions to compare (default: 1,2,3).")
    dim.add_argument("--folds", type=int, default=5, help="Number of validation folds (default: 5).")
    dim.add_argument("--model", default="MLS2PLM", help="Model type to fit (default: MLS2PLM).")
    dim.add_argument("--optimizer", choices=["adam", "lbfgs", "adam_lbfgs"], default="adam_lbfgs", help="Optimizer to use (default: adam_lbfgs).")
    dim.add_argument("--max-iter", type=int, default=100, help="Maximum iterations per fold fit (default: 100).")
    dim.add_argument("--n-restarts", type=int, default=1, help="Random restarts per fold fit (default: 1).")
    dim.add_argument("--seed", type=int, default=1, help="Random seed for folds and fitting (default: 1).")
    dim.add_argument("--out", required=True, help="Directory path to save dimension_diagnostics.json.")
    _add_json_flag(dim)

    process = sub.add_parser(
        "diagnose-response-process",
        help="Compute dichotomous or polytomous fit diagnostics from category probabilities.",
        description="Compute dichotomous or polytomous fit diagnostics from category probabilities.",
    )
    process.add_argument("--responses", required=True, help="Path to the responses numpy array file (.npy).")
    process.add_argument("--probabilities", required=True, help="Path to probabilities .npy, either persons x items or persons x items x categories.")
    process.add_argument("--item-type", choices=["dichotomous", "polytomous"], default="polytomous", help="Item type for metadata validation.")
    process.add_argument("--response-process", choices=["ideal_point", "cumulative"], default="cumulative", help="Response process represented by the probabilities.")
    process.add_argument("--group-id", help="Optional .npy person group IDs for multigroup summaries.")
    process.add_argument("--cluster-id", help="Optional .npy person cluster IDs for multilevel summaries.")
    process.add_argument("--out", required=True, help="Directory path to save fit_diagnostics.json.")
    _add_json_flag(process)

    candidates = sub.add_parser(
        "diagnose-response-candidates",
        help="Compare response-process probability candidates with held-out likelihood.",
        description="Compare candidate category probability tensors for dimensionality or response-process checks.",
    )
    candidates.add_argument("--responses", required=True, help="Path to the responses numpy array file (.npy).")
    candidates.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate probability file as label=path.npy. Repeat for each candidate.",
    )
    candidates.add_argument("--item-type", choices=["dichotomous", "polytomous"], default="polytomous", help="Item type for metadata validation.")
    candidates.add_argument("--response-process", choices=["ideal_point", "cumulative"], default="cumulative", help="Response process represented by the candidates.")
    candidates.add_argument("--out", required=True, help="Directory path to save dimension_diagnostics.json.")
    _add_json_flag(candidates)

    fixed_calibration = sub.add_parser(
        "diagnose-fixed-item-calibration",
        help="Select a response-process candidate using fixed-item calibration diagnostics.",
        description="Score candidate probability tensors on fixed evaluation items using item-fit risk weighting.",
    )
    fixed_calibration.add_argument("--responses", required=True, help="Path to the responses numpy array file (.npy).")
    fixed_calibration.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate probability file as label=path.npy. Repeat for each candidate.",
    )
    fixed_calibration.add_argument("--fixed-items", help="Optional .npy boolean mask or item-index vector for fixed items.")
    fixed_calibration.add_argument("--item-type", choices=["dichotomous", "polytomous"], default="polytomous", help="Item type for metadata validation.")
    fixed_calibration.add_argument("--response-process", choices=["ideal_point", "cumulative"], default="cumulative", help="Response process represented by the candidates.")
    fixed_calibration.add_argument("--itemfit-penalty-weight", type=float, default=1.0, help="Weight applied to mean |outfit_mnsq - 1| in the calibration score.")
    fixed_calibration.add_argument("--out", required=True, help="Directory path to save dimension_diagnostics.json.")
    _add_json_flag(fixed_calibration)

    report = sub.add_parser(
        "render-report",
        help="Render saved diagnostics JSON as a standalone HTML report.",
        description="Render fit_diagnostics.json or dimension_diagnostics.json as a standalone HTML report.",
    )
    report.add_argument("--diagnostics", required=True, help="Path to fit_diagnostics.json or dimension_diagnostics.json.")
    report.add_argument("--out", required=True, help="Path to write the diagnostics HTML report.")
    report.add_argument("--title", help="Optional report title.")
    _add_json_flag(report)

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 2

    args = parser.parse_args(argv)
    try:
        _confine_cli_paths(args)
    except ValueError as e:
        if os.environ.get("FAST_MLSIRM_DEBUG"):
            raise
        print(f"❌ Error: Invalid path - {str(e)}", file=sys.stderr)
        return 1
    if args.command == "fit" and args.reference and args.backend != "auto":
        print("❌ Error: --reference cannot be combined with --backend rust", file=sys.stderr)
        return 2
    if args.command == "simulate":
        _progress(args, f"⏳ Simulating {args.persons} persons and {args.dims} dimensions...")
        try:
            data = simulate(
                MLS2PLMConfig(
                    n_persons=args.persons,
                    n_dims=args.dims,
                    items_per_dim=args.items_per_dim,
                    latent_dim=args.latent_dim,
                    phi=args.phi,
                    gamma=args.gamma,
                    seed=args.seed,
                )
            )
            save_simulation(data, args.out)
        except ValueError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Invalid configuration - {str(e)}", file=sys.stderr)
            return 1
        except OSError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Failed to save simulation - {str(e)}", file=sys.stderr)
            return 1

        return _complete(
            args,
            f"✅ Simulation successfully saved to {args.out}",
            {
                "command": "simulate",
                "status": "ok",
                "out": str(args.out),
                "n_persons": int(data.Y.shape[0]),
                "n_items": int(data.Y.shape[1]),
                "n_dims": int(data.config.n_dims),
                "files": {
                    "responses": _output_file(args.out, "responses.npy"),
                    "factors": _output_file(args.out, "item_factor.csv"),
                    "truth": _output_file(args.out, "truth.npz"),
                    "manifest": _output_file(args.out, "manifest.json"),
                },
            },
        )

    if args.command == "score":
        from .serving import load_serving_bundle, score_respondents

        _progress(args, "⏳ Scoring respondents against the serving bundle...")
        try:
            bundle = load_serving_bundle(args.bundle)
            if args.responses.endswith(".npy"):
                payload = _load_numpy_bounded(args.responses)
            else:
                payload = _load_json_bounded(
                    args.responses,
                    source="response JSON",
                )
            scores = score_respondents(bundle, payload)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Scoring failed - {str(e)}", file=sys.stderr)
            return 1
        if args.out:
            _atomic_write_text(
                args.out, json.dumps(scores, ensure_ascii=False, indent=2)
            )
        return _complete(
            args,
            json.dumps(scores, ensure_ascii=False, indent=2)
            if not args.out
            else f"✅ Scores written to {args.out}",
            {
                "command": "score",
                "status": "ok",
                "n_scored": len(scores),
                "scores": scores,
            },
        )

    if args.command == "diagnose-fit":
        _progress(args, f"⏳ Computing {args.model} fit diagnostics...")
        try:
            responses, factors = _load_response_and_factors(args.responses, args.factors)
            params = load_params(args.params)
            saved_estimator, population, convergence_status = _load_fit_context(args.params)
            group_id = _load_optional_npy(args.group_id)
            cluster_id = _load_optional_npy(args.cluster_id)
            if (
                args.limited_information
                and convergence_status is not None
                and convergence_status != "converged"
            ):
                raise ValueError(
                    "limited-information diagnostics require converged parameters; "
                    "the fitted model did not converge "
                    f"(status={convergence_status})"
                )
        except FileNotFoundError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except ValueError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Invalid input data - {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
            return 1

        diagnostics = fit_diagnostics(
            responses=responses,
            params=params,
            factor_id=factors,
            model=args.model,
            group_id=group_id,
            cluster_id=cluster_id,
            include_m2=args.limited_information,
            estimator=args.estimator or saved_estimator,
            population=population,
            convergence_status=convergence_status,
        )
        save_fit_diagnostics(diagnostics, args.out)
        return _complete(
            args,
            f"✅ Fit diagnostics successfully saved to {args.out}",
            {
                "command": "diagnose-fit",
                "status": "ok",
                "out": str(args.out),
                "model": args.model,
                "files": {"diagnostics": _output_file(args.out, "fit_diagnostics.json")},
            },
        )

    if args.command == "diagnose-dimensions":
        _progress(args, f"⏳ Comparing {args.model} latent dimensions {args.latent_dims}...")
        try:
            responses, factors = _load_response_and_factors(args.responses, args.factors)
            latent_dims = [int(value) for value in args.latent_dims.split(",")]
        except FileNotFoundError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except ValueError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Invalid input data - {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
            return 1

        diagnostics = dimensionality_diagnostics(
            responses=responses,
            factor_id=factors,
            latent_dims=latent_dims,
            model=args.model,
            k_folds=args.folds,
            seed=args.seed,
            require_experiment_readiness=True,
            config=FitConfig(
                model=args.model,
                optimizer=args.optimizer,
                max_iter=args.max_iter,
                n_restarts=args.n_restarts,
                seed=args.seed,
            ),
        )
        save_dimensionality_diagnostics(diagnostics, args.out)
        return _complete(
            args,
            f"✅ Dimension diagnostics successfully saved to {args.out}",
            {
                "command": "diagnose-dimensions",
                "status": "ok",
                "out": str(args.out),
                "model": args.model,
                "best_latent_dim": int(diagnostics.best["latent_dim"]),
                "files": {"diagnostics": _output_file(args.out, "dimension_diagnostics.json")},
            },
        )

    if args.command == "diagnose-response-process":
        _progress(args, f"⏳ Computing {args.item_type} {args.response_process} fit diagnostics...")
        try:
            responses = _load_numpy_bounded(args.responses)
            probabilities = _load_numpy_bounded(args.probabilities)
            group_id = _load_optional_npy(args.group_id)
            cluster_id = _load_optional_npy(args.cluster_id)
        except FileNotFoundError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except ValueError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Invalid input data - {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
            return 1

        diagnostics = response_process_fit_diagnostics(
            responses=responses,
            probabilities=probabilities,
            item_type=args.item_type,
            response_process=args.response_process,
            group_id=group_id,
            cluster_id=cluster_id,
        )
        save_fit_diagnostics(diagnostics, args.out)
        return _complete(
            args,
            f"✅ Response process diagnostics successfully saved to {args.out}",
            {
                "command": "diagnose-response-process",
                "status": "ok",
                "out": str(args.out),
                "item_type": args.item_type,
                "response_process": args.response_process,
                "files": {"diagnostics": _output_file(args.out, "fit_diagnostics.json")},
            },
        )

    if args.command == "diagnose-response-candidates":
        _progress(args, f"⏳ Comparing {args.item_type} {args.response_process} response candidates...")
        try:
            responses = _load_numpy_bounded(args.responses)
            candidate_probabilities = _load_candidate_probabilities(args.candidate)
        except FileNotFoundError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except ValueError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Invalid input data - {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
            return 1

        diagnostics = response_process_dimensionality_diagnostics(
            responses=responses,
            candidate_probabilities=candidate_probabilities,
            item_type=args.item_type,
            response_process=args.response_process,
        )
        save_dimensionality_diagnostics(diagnostics, args.out)
        return _complete(
            args,
            f"✅ Response candidate diagnostics successfully saved to {args.out}",
            {
                "command": "diagnose-response-candidates",
                "status": "ok",
                "out": str(args.out),
                "item_type": args.item_type,
                "response_process": args.response_process,
                "best_candidate": diagnostics.best["candidate_label"],
                "files": {"diagnostics": _output_file(args.out, "dimension_diagnostics.json")},
            },
        )

    if args.command == "diagnose-fixed-item-calibration":
        _progress(args, "⏳ Scoring fixed-item calibration candidates...")
        try:
            responses = _load_numpy_bounded(args.responses)
            candidate_probabilities = _load_candidate_probabilities(args.candidate)
            fixed_items = _load_optional_npy(args.fixed_items)
        except FileNotFoundError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except ValueError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Invalid input data - {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
            return 1

        diagnostics = fixed_item_calibration_diagnostics(
            responses=responses,
            candidate_probabilities=candidate_probabilities,
            fixed_items=fixed_items,
            item_type=args.item_type,
            response_process=args.response_process,
            itemfit_penalty_weight=args.itemfit_penalty_weight,
        )
        save_dimensionality_diagnostics(diagnostics, args.out)
        return _complete(
            args,
            f"✅ Fixed-item calibration diagnostics successfully saved to {args.out}",
            {
                "command": "diagnose-fixed-item-calibration",
                "status": "ok",
                "out": str(args.out),
                "item_type": args.item_type,
                "response_process": args.response_process,
                "best_candidate": diagnostics.best["candidate_label"],
                "calibration_score": diagnostics.best["calibration_score"],
                "files": {"diagnostics": _output_file(args.out, "dimension_diagnostics.json")},
            },
        )

    if args.command == "render-report":
        _progress(args, f"⏳ Rendering diagnostics report from {args.diagnostics}...")
        try:
            report_path = render_diagnostics_report(args.diagnostics, args.out, title=args.title)
        except FileNotFoundError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except ValueError as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Invalid diagnostics data - {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            if os.environ.get("FAST_MLSIRM_DEBUG"):
                raise
            print(f"❌ Error: Failed to render report - {str(e)}", file=sys.stderr)
            return 1

        return _complete(
            args,
            f"✅ Diagnostics report successfully saved to {report_path}",
            {
                "command": "render-report",
                "status": "ok",
                "out": str(report_path),
                "files": {"report": str(report_path), "diagnostics": str(args.diagnostics)},
            },
        )

    _progress(args, f"⏳ Fitting {args.model} model to responses...")
    try:
        responses, factors = _load_response_and_factors(args.responses, args.factors)
    except FileNotFoundError as e:
        if os.environ.get("FAST_MLSIRM_DEBUG"):
            raise
        print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
        return 1
    except ValueError as e:
        if os.environ.get("FAST_MLSIRM_DEBUG"):
            raise
        print(f"❌ Error: Invalid input data - {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        if os.environ.get("FAST_MLSIRM_DEBUG"):
            raise
        print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
        return 1

    try:
        result = fit_irt_experiment(
            fit_reference if args.reference else fit,
            responses,
            "dichotomous",
            factor_ids=factors,
            factor_id=factors,
            config=FitConfig(
                model=args.model,
                latent_dim=args.latent_dim,
                optimizer=args.optimizer,
                estimator=args.estimator,
                max_iter=args.max_iter,
                n_restarts=args.n_restarts,
                seed=args.seed,
                backend=args.backend,
                rust_device=args.rust_device,
                q_theta=args.q_theta,
                q_xi=args.q_xi,
                q_u=args.q_u,
                tolerance=args.tolerance,
            ),
            group_id=_load_optional_npy(args.group_id),
            cluster_id=_load_optional_npy(args.cluster_id),
        )
    except ValueError as e:
        if os.environ.get("FAST_MLSIRM_DEBUG"):
            raise
        print(f"❌ Error: Invalid fit configuration - {str(e)}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        if os.environ.get("FAST_MLSIRM_DEBUG"):
            raise
        print(f"❌ Error: Fit failed - {str(e)}", file=sys.stderr)
        return 1
    save_fit_result(result, args.out)
    return _complete(
        args,
        f"✅ Fit result successfully saved to {args.out}",
        {
            "command": "fit",
            "status": "ok",
            "out": str(args.out),
            "model": result.model,
            "optimizer": result.optimizer,
            "backend": result.backend,
            "rust_device": result.rust_device,
            "objective": float(result.objective),
            "convergence_status": result.convergence_status,
            "n_iter": int(result.n_iter),
            "files": {
                "params": _output_file(args.out, "params.npz"),
                "summary": _output_file(args.out, "fit_summary.json"),
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run :func:`_main` with top-level error handling.

    Maps ``KeyboardInterrupt`` to exit code 130 and any other exception to exit
    code 1 (re-raising when ``FAST_MLSIRM_DEBUG`` is set).
    """
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("❌ Error: Interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:
        if os.environ.get("FAST_MLSIRM_DEBUG"):
            raise
        print(f"❌ Error: Unexpected failure - {exc}", file=sys.stderr)
        return 1


def _load_optional_npy(path: str | None) -> np.ndarray | None:
    """Load a bounded ``.npy`` array, or return ``None`` when no path is given."""
    return None if path is None else _load_numpy_bounded(path)


def _load_candidate_probabilities(specs: list[str]) -> dict[str, np.ndarray]:
    """Load labelled candidate-probability arrays from ``label=path`` specs.

    Enforces the candidate-count and aggregate element/byte caps and rejects
    empty or duplicate labels, returning a mapping of label to loaded array.
    """
    if len(specs) > MAX_CANDIDATE_COUNT:
        raise ValueError(
            f"candidate count exceeds the {MAX_CANDIDATE_COUNT}-candidate "
            "limit; combine related candidates or pass fewer --candidate flags"
        )
    candidates = {}
    total_elements = 0
    total_bytes = 0
    for spec in specs:
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        if not label:
            raise ValueError(
                "candidate label must not be empty; pass candidates as "
                "label=path.npy"
            )
        if label in candidates:
            raise ValueError(
                f"duplicate candidate label: {label}; give each --candidate "
                "flag a unique label"
            )
        candidate = _load_numpy_bounded(path)
        if not isinstance(candidate, np.ndarray):
            candidate.close()
            raise ValueError(
                "candidate probability inputs must be single .npy arrays; "
                "export each candidate as a plain .npy array and retry"
            )
        total_elements += int(candidate.size)
        total_bytes += int(candidate.nbytes)
        if (
            total_elements > MAX_CANDIDATE_ELEMENTS
            or total_bytes > MAX_CANDIDATE_BYTES
        ):
            raise ValueError(
                "candidate probability inputs exceed the aggregate size "
                "limit; score fewer or smaller candidate sets per run"
            )
        candidates[label] = candidate
    return candidates


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover
