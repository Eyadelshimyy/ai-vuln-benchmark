#!/usr/bin/env python
"""Build benchmark evidence from a release acceptance summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

GIT_METADATA_TIMEOUT_SECONDS = 5

try:
    from scripts._bounded_json import read_json_object
except ModuleNotFoundError:
    from _bounded_json import read_json_object


def _read_json(path: Path) -> dict[str, Any]:
    """Read one descriptor-safe bounded JSON object from disk."""
    return read_json_object(path)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one benchmark evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(repo_root: Path) -> str:
    """Return the exact HEAD SHA, failing closed when Git provenance is unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_METADATA_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("source commit lookup timed out") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("source commit lookup failed") from exc

    source_commit = completed.stdout.strip()
    if len(source_commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise RuntimeError("source commit lookup returned invalid identity")
    return source_commit


def _content_security_policy() -> str:
    """Return the restrictive policy used by the self-contained benchmark report."""
    return "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def _format_value(value: Any) -> str:
    """Format a benchmark value for a human-readable report cell."""
    if isinstance(value, bool):
        return "go" if value else "failed"
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _step_files(step: dict[str, Any]) -> list[str]:
    """Extract declared artifact paths from one acceptance step."""
    files = step.get("files")
    if not isinstance(files, dict):
        return []
    return [str(path) for path in files.values()]


def _observed_backends(steps: list[dict[str, Any]]) -> list[str]:
    """Return the sorted backend identities observed in fit steps."""
    observed: set[str] = set()
    for step in steps:
        if step.get("command") != "fit":
            continue
        backend = step.get("backend")
        if isinstance(backend, str) and backend:
            observed.add(backend)
        if "fit_auto" in str(step.get("out", "")):
            observed.add("auto")
    return sorted(observed)


def _required_backends(benchmark: dict[str, Any]) -> list[str]:
    """Return the sorted backend identities required by benchmark scenarios."""
    scenarios = benchmark.get("scenarios", [])
    if not isinstance(scenarios, list):
        return []
    backends = {
        scenario.get("backend")
        for scenario in scenarios
        if isinstance(scenario, dict) and isinstance(scenario.get("backend"), str)
    }
    return sorted(backends)


def _artifact_coverage(
    benchmark: dict[str, Any], steps: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare required benchmark artifacts with acceptance outputs on disk."""
    required = benchmark.get("required_artifacts", [])
    if not isinstance(required, list):
        required = []
    required_names = [str(name) for name in required]
    observed_paths = [path for step in steps for path in _step_files(step)]
    observed_names = {Path(path).name for path in observed_paths}
    present = [name for name in required_names if name in observed_names]
    missing = [name for name in required_names if name not in observed_names]
    missing_paths = [path for path in observed_paths if not Path(path).exists()]
    return {
        "required": required_names,
        "present": present,
        "missing": missing,
        "missing_paths": missing_paths,
        "observed_paths": observed_paths,
    }


def _command_durations(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project acceptance steps into stable benchmark duration rows."""
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        rows.append(
            {
                "index": index,
                "command": step.get("command", ""),
                "backend": step.get("backend", ""),
                "duration_seconds": step.get("duration_seconds"),
                "out": step.get("out", ""),
            }
        )
    return rows


def _render_rows(
    rows: list[dict[str, Any]], columns: list[tuple[str, str]]
) -> list[str]:
    """Render benchmark records as escaped accessible HTML table rows."""
    rendered = []
    for row in rows:
        cells = []
        for key, label in columns:
            value = escape(_format_value(row.get(key)))
            if not cells:
                cells.append(f'<th scope="row">{value}</th>')
            else:
                cells.append(f"<td>{value}</td>")
        rendered.append("<tr>" + "".join(cells) + "</tr>")
    return rendered


def _render_report_html(report: dict[str, Any]) -> str:
    """Render benchmark evidence as a portable accessible HTML report."""
    command_rows = _render_rows(
        report.get("command_durations", []),
        [
            ("index", "Step"),
            ("command", "Command"),
            ("backend", "Backend"),
            ("duration_seconds", "Seconds"),
            ("out", "Output"),
        ],
    )
    artifact_rows = [
        "<tr>"
        f'<th scope="row">{escape(name)}</th>'
        f"<td>{'go' if name in set(report.get('artifact_coverage', {}).get('present', [])) else 'missing'}</td>"
        "</tr>"
        for name in report.get("artifact_coverage", {}).get("required", [])
    ]
    scenario = report.get("scenario_coverage", {})
    cards = [
        ("Status", report.get("status", "")),
        ("Runtime Budget", f"{report.get('runtime_budget_seconds', '')}s"),
        ("Total Runtime", f"{report.get('total_duration_seconds', '')}s"),
        ("Budget Result", report.get("budget_ok")),
        (
            "Observed Backends",
            ", ".join(scenario.get("observed_backends", []))
            if isinstance(scenario, dict)
            else "",
        ),
        ("Source Commit", report.get("source_commit", "")),
    ]
    card_markup = [
        "\n".join(
            [
                '<article class="metric-card">',
                f"<span>{escape(label)}</span>",
                f"<strong>{escape(_format_value(value))}</strong>",
                "</article>",
            ]
        )
        for label, value in cards
    ]
    caveats = report.get("caveats", [])
    caveat_items = []
    if isinstance(caveats, list):
        caveat_items = [f"<li>{escape(str(caveat))}</li>" for caveat in caveats]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f'<meta http-equiv="Content-Security-Policy" content="{escape(_content_security_policy(), quote=True)}">',
            "<title>fast-mlsirm Benchmark Evidence Report</title>",
            "<style>",
            _report_css(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            '<section class="hero">',
            "<p>fast-mlsirm release benchmark</p>",
            "<h1>Benchmark Evidence Report</h1>",
            f"<span>Generated: {escape(str(report.get('generated_at', '')))}</span>",
            "</section>",
            '<section class="report-section">',
            "<h2>Runtime Summary</h2>",
            '<div class="metrics-grid">',
            *card_markup,
            "</div>",
            "</section>",
            '<section class="report-section">',
            "<h2>Command Durations</h2>",
            '<div class="table-wrap" role="region" aria-label="Command duration table" tabindex="0">',
            "<table>",
            "<caption>Command duration table</caption>",
            '<thead><tr><th scope="col">Step</th><th scope="col">Command</th><th scope="col">Backend</th><th scope="col">Seconds</th><th scope="col">Output</th></tr></thead>',
            "<tbody>",
            *command_rows,
            "</tbody>",
            "</table>",
            "</div>",
            "</section>",
            '<section class="report-section">',
            "<h2>Required Artifact Coverage</h2>",
            '<div class="table-wrap" role="region" aria-label="Required artifact coverage table" tabindex="0">',
            "<table>",
            "<caption>Required artifact coverage table</caption>",
            '<thead><tr><th scope="col">Artifact</th><th scope="col">Status</th></tr></thead>',
            "<tbody>",
            *artifact_rows,
            "</tbody>",
            "</table>",
            "</div>",
            "<ul>",
            *caveat_items,
            "</ul>",
            '<p class="note">This report is benchmark evidence for procurement review, not a production performance guarantee.</p>',
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _report_css() -> str:
    """Return the small inline stylesheet used by the benchmark report."""
    return """
:root {
  color: #172026;
  background: #f5f7f8;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
}
main {
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 20px 56px;
}
.hero {
  background: #12343b;
  color: #ffffff;
  border-radius: 8px;
  padding: 28px;
}
.hero p,
.hero h1 {
  margin: 0;
}
.hero p {
  color: #b7d7d0;
  font-size: 0.86rem;
  font-weight: 700;
  text-transform: uppercase;
}
.hero h1 {
  margin-top: 8px;
  font-size: 2rem;
}
.hero span {
  display: inline-block;
  margin-top: 14px;
  color: #dce8e5;
}
.report-section {
  margin-top: 22px;
  background: #ffffff;
  border: 1px solid #d8e1e3;
  border-radius: 8px;
  padding: 22px;
}
.report-section h2 {
  margin: 0 0 16px;
  font-size: 1.16rem;
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.metric-card {
  border: 1px solid #d8e1e3;
  border-radius: 8px;
  padding: 14px;
}
.metric-card span {
  display: block;
  color: #5e6f76;
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: uppercase;
}
.metric-card strong {
  display: block;
  margin-top: 8px;
  overflow-wrap: anywhere;
}
.table-wrap {
  overflow-x: auto;
  border: 1px solid #d8e1e3;
  border-radius: 8px;
}
.table-wrap:focus-visible {
  outline: 3px solid #0f766e;
  outline-offset: 3px;
}
table {
  width: 100%;
  min-width: 680px;
  border-collapse: collapse;
}
caption {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
th,
td {
  padding: 10px 12px;
  border-bottom: 1px solid #e8edef;
  text-align: left;
  vertical-align: top;
  font-variant-numeric: tabular-nums;
}
tbody tr {
  transition: background-color 0.15s ease-in-out;
}
tbody tr:hover {
  background: #fbfcfa;
}
tbody tr:last-child th,
tbody tr:last-child td {
  border-bottom: 0;
}
.note {
  color: #5e6f76;
  margin-bottom: 0;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

@media print {
  * {
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }
  body {
    background: #ffffff;
  }
  .report-section,
  .metric-card,
  .table-wrap {
    break-inside: avoid;
  }
}
"""


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """Build and return benchmark evidence from acceptance and scenario manifests."""
    repo_root = Path(args.repo_root).resolve()
    acceptance_path = Path(args.acceptance).resolve()
    benchmark_manifest_path = Path(args.benchmark_manifest).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "benchmark_report.json"
    html_path = out_dir / "benchmark_report.html"

    acceptance = _read_json(acceptance_path)
    benchmark = _read_json(benchmark_manifest_path)
    steps = [step for step in acceptance.get("steps", []) if isinstance(step, dict)]
    runtime_budget = benchmark.get("runtime_budget_seconds")
    total_duration = acceptance.get("total_duration_seconds")
    budget_ok = (
        isinstance(runtime_budget, (int, float))
        and isinstance(total_duration, (int, float))
        and total_duration <= runtime_budget
    )
    required_backends = _required_backends(benchmark)
    observed_backends = _observed_backends(steps)
    missing_backends = [
        backend for backend in required_backends if backend not in observed_backends
    ]
    artifact_coverage = _artifact_coverage(benchmark, steps)
    status_ok = (
        acceptance.get("status") == "ok"
        and budget_ok
        and not missing_backends
        and not artifact_coverage["missing"]
        and not artifact_coverage["missing_paths"]
    )
    report: dict[str, Any] = {
        "command": "build_benchmark_report",
        "status": "ok" if status_ok else "failed",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": _source_commit(repo_root),
        "acceptance": str(acceptance_path),
        "benchmark_manifest": str(benchmark_manifest_path),
        "benchmark_scope": benchmark.get("benchmark_scope", ""),
        "runtime_budget_seconds": runtime_budget,
        "total_duration_seconds": total_duration,
        "budget_ok": budget_ok,
        "scenario_coverage": {
            "required_backends": required_backends,
            "observed_backends": observed_backends,
            "missing_backends": missing_backends,
        },
        "artifact_coverage": artifact_coverage,
        "command_durations": _command_durations(steps),
        "caveats": benchmark.get("caveats", []),
        "html_report_file": str(html_path),
    }
    html_path.write_text(_render_report_html(report), encoding="utf-8")
    report["html_report_sha256"] = _sha256(html_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for benchmark report generation."""
    parser = argparse.ArgumentParser(
        description="Build fast-mlsirm benchmark evidence from release acceptance output."
    )
    parser.add_argument(
        "--repo-root", default=".", help="Repository root used to record source commit."
    )
    parser.add_argument(
        "--acceptance", required=True, help="Path to release acceptance_summary.json."
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="examples/enterprise_demo/benchmark_manifest.json",
        help="Path to benchmark_manifest.json.",
    )
    parser.add_argument(
        "--out",
        default="benchmark-evidence",
        help="Output directory for benchmark report files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run benchmark report generation and print a machine-readable summary."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_report(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": str(Path(args.out).resolve()),
                "report": str(Path(args.out).resolve() / "benchmark_report.json"),
                "html": report["html_report_file"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
