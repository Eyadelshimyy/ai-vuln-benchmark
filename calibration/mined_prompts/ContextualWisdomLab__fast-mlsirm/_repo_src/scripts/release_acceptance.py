#!/usr/bin/env python
"""Release acceptance smoke test for fast-mlsirm.

This script runs an end-to-end CLI workflow that mirrors a minimal production
verification path:
simulate -> fit (auto and optionally rust) -> diagnostics -> report rendering.
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    from scripts._bounded_json import parse_json_bounded, read_json_object
except ModuleNotFoundError:  # executed from scripts/ or loaded via importlib
    from _bounded_json import parse_json_bounded, read_json_object


RELEASE_ACCEPTANCE_TIMEOUT_SECONDS = {
    "simulate": 120.0,
    "fit_auto": 900.0,
    "fit_rust": 900.0,
    "diagnose-fit": 300.0,
    "diagnose-dimensions": 900.0,
    "render-report-fit": 120.0,
    "render-report-dimensions": 120.0,
}


def _cli_timeout_seconds(out_label: str) -> float:
    """Return the bounded subprocess deadline for one acceptance operation."""
    try:
        return RELEASE_ACCEPTANCE_TIMEOUT_SECONDS[out_label]
    except KeyError:
        raise RuntimeError("unsupported release acceptance operation") from None


def _require_auto_fit_resolved_to_rust(
    summary: dict[str, object], fit_payload: dict[str, object]
) -> None:
    """Fail closed unless the auto fit recorded Rust as the numerical owner.

    Automatic production resolution never selects NumPy. A purchaser running
    release acceptance without ``--require-rust`` must still see Rust on the
    ``fit --backend auto`` path. Pass ``backend="numpy"`` only as an explicit
    reference/parity command, not as an accepted automatic outcome.
    """
    if summary.get("backend") != "rust":
        raise RuntimeError("fit auto backend must resolve to rust")
    if fit_payload.get("backend") != "rust":
        raise RuntimeError("fit_payload backend does not match fit_summary backend")


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_python = Path(__file__).resolve().parents[1] / "python"
    has_pkg = importlib.util.find_spec("fast_mlsirm") is not None
    if not has_pkg:
        existing = env.get("PYTHONPATH", "")
        if existing:
            env["PYTHONPATH"] = f"{existing}{os.pathsep}{repo_python}"
        else:
            env["PYTHONPATH"] = str(repo_python)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _run_cli(
    args: list[str], out_label: str, *, require_json: bool = True
) -> dict[str, object]:
    command = [sys.executable, "-m", "fast_mlsirm.cli", *args]
    if require_json and "--json" not in command:
        command.append("--json")
    started = time.perf_counter()
    timeout_seconds = _cli_timeout_seconds(out_label)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=_cli_env(),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{out_label} timed out") from None
    duration_seconds = round(time.perf_counter() - started, 6)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        raise RuntimeError(f"{out_label} failed ({command}): {stderr or stdout}")
    raw_output = completed.stdout.strip().splitlines()
    if not raw_output:
        raise RuntimeError(f"{out_label} succeeded without JSON output")
    if not require_json:
        return {
            "status": "ok",
            "command": out_label,
            "stdout": raw_output[-1],
            "duration_seconds": duration_seconds,
        }
    try:
        payload = parse_json_bounded(raw_output[-1])
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        payload["duration_seconds"] = duration_seconds
        return payload
    except ValueError as exc:
        raise RuntimeError(
            f"{out_label} produced non-JSON output: {raw_output[-1]}"
        ) from exc


def _run_acceptance(args: argparse.Namespace) -> dict[str, object]:
    acceptance_started = time.perf_counter()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "command": "release_acceptance",
        "status": "ok",
        "out": str(out_dir),
        "steps": [],
    }

    sim_out = out_dir / "simulate"
    fit_auto_out = out_dir / "fit_auto"
    fit_rust_out = out_dir / "fit_rust"
    diag_fit_out = out_dir / "diagnostics_fit"
    diag_dim_out = out_dir / "diagnostics_dimensions"
    report_fit_out = out_dir / "fit_report.html"
    report_dim_out = out_dir / "dimension_report.html"

    simulate_payload = _run_cli(
        [
            "simulate",
            "--persons",
            str(args.persons),
            "--dims",
            str(args.dims),
            "--items-per-dim",
            str(args.items_per_dim),
            "--latent-dim",
            str(args.latent_dim),
            "--seed",
            str(args.seed),
            "--out",
            str(sim_out),
        ],
        "simulate",
    )
    report["steps"].append(simulate_payload)

    fit_auto_payload = _run_cli(
        [
            "fit",
            "--responses",
            str(sim_out / "responses.npy"),
            "--factors",
            str(sim_out / "item_factor.csv"),
            "--max-iter",
            str(args.max_iter),
            "--n-restarts",
            str(args.n_restarts),
            "--optimizer",
            "adam",
            "--backend",
            "auto",
            "--latent-dim",
            str(args.latent_dim),
            "--seed",
            str(args.seed),
            "--out",
            str(fit_auto_out),
        ],
        "fit_auto",
    )
    report["steps"].append(fit_auto_payload)

    summary = read_json_object(fit_auto_out / "fit_summary.json")
    _require_auto_fit_resolved_to_rust(summary, fit_auto_payload)

    if args.require_rust:
        fit_rust_payload = _run_cli(
            [
                "fit",
                "--responses",
                str(sim_out / "responses.npy"),
                "--factors",
                str(sim_out / "item_factor.csv"),
                "--max-iter",
                str(args.max_iter),
                "--n-restarts",
                str(args.n_restarts),
                "--optimizer",
                "adam",
                "--backend",
                "rust",
                "--latent-dim",
                str(args.latent_dim),
                "--seed",
                str(args.seed),
                "--out",
                str(fit_rust_out),
            ],
            "fit_rust",
        )
        report["steps"].append(fit_rust_payload)
        rust_summary = read_json_object(fit_rust_out / "fit_summary.json")
        if rust_summary.get("backend") != "rust":
            raise RuntimeError("rust fit did not report rust backend")

    fit_diagnostics_payload = _run_cli(
        [
            "diagnose-fit",
            "--responses",
            str(sim_out / "responses.npy"),
            "--factors",
            str(sim_out / "item_factor.csv"),
            "--params",
            str(fit_auto_out / "params.npz"),
            "--out",
            str(diag_fit_out),
        ],
        "diagnose-fit",
    )
    report["steps"].append(fit_diagnostics_payload)

    dimensionality_payload = _run_cli(
        [
            "diagnose-dimensions",
            "--responses",
            str(sim_out / "responses.npy"),
            "--factors",
            str(sim_out / "item_factor.csv"),
            "--latent-dims",
            args.latent_dims,
            "--folds",
            str(args.folds),
            "--max-iter",
            str(args.max_iter),
            "--seed",
            str(args.seed),
            "--out",
            str(diag_dim_out),
        ],
        "diagnose-dimensions",
    )
    report["steps"].append(dimensionality_payload)

    render_fit_payload = _run_cli(
        [
            "render-report",
            "--diagnostics",
            str(diag_fit_out / "fit_diagnostics.json"),
            "--out",
            str(report_fit_out),
            "--json",
        ],
        "render-report-fit",
        require_json=True,
    )
    report["steps"].append(render_fit_payload)

    render_dim_payload = _run_cli(
        [
            "render-report",
            "--diagnostics",
            str(diag_dim_out / "dimension_diagnostics.json"),
            "--out",
            str(report_dim_out),
            "--json",
        ],
        "render-report-dimensions",
        require_json=True,
    )
    report["steps"].append(render_dim_payload)

    for path in [
        sim_out / "responses.npy",
        sim_out / "item_factor.csv",
        fit_auto_out / "params.npz",
        fit_auto_out / "fit_summary.json",
        diag_fit_out / "fit_diagnostics.json",
        diag_dim_out / "dimension_diagnostics.json",
        report_fit_out,
        report_dim_out,
    ]:
        if not path.exists():
            raise RuntimeError(f"expected artifact missing: {path}")

    if args.require_rust:
        for path in [
            fit_rust_out / "params.npz",
            fit_rust_out / "fit_summary.json",
        ]:
            if not path.exists():
                raise RuntimeError(f"expected rust artifact missing: {path}")

    summary_path = out_dir / "acceptance_summary.json"
    summary_payload = {
        "status": "ok",
        "out": str(out_dir),
        "steps": report["steps"],
        "total_duration_seconds": round(time.perf_counter() - acceptance_started, 6),
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"status": "ok", "out": str(out_dir), "report": str(summary_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fast-mlsirm release acceptance smoke test."
    )
    parser.add_argument(
        "--out",
        default="acceptance_check",
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--persons", type=int, default=12, help="Number of persons to simulate."
    )
    parser.add_argument(
        "--dims", type=int, default=1, help="Simulated true item dimensions."
    )
    parser.add_argument(
        "--items-per-dim",
        type=int,
        default=2,
        help="Items per dimension for simulation.",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=1, help="Latent dimension for fitting."
    )
    parser.add_argument("--seed", type=int, default=1, help="Simulation seed.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1,
        help="Max optimization iterations for fitting.",
    )
    parser.add_argument(
        "--n-restarts", type=int, default=1, help="Optimization restarts."
    )
    parser.add_argument(
        "--latent-dims",
        default="1,2",
        help="Comma-separated latent dims for dimension diagnostics.",
    )
    parser.add_argument(
        "--folds", type=int, default=2, help="CV folds for dimensionality diagnostics."
    )
    parser.add_argument(
        "--require-rust",
        action="store_true",
        help="Also run fit in explicit rust backend mode; fail if unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _run_acceptance(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
