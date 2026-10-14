#!/usr/bin/env python
"""Build one commercial release evidence bundle for fast-mlsirm."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any, Callable

GIT_METADATA_TIMEOUT_SECONDS = 5

try:
    from scripts._bounded_json import parse_json_bounded, read_json_object
    from scripts._bounded_subprocess import BoundedSubprocessOutputError, run_bounded_capture
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts", "scripts._bounded_json", "scripts._bounded_subprocess"}:
        raise
    from _bounded_json import parse_json_bounded, read_json_object
    from _bounded_subprocess import BoundedSubprocessOutputError, run_bounded_capture

try:
    from scripts.release_acceptance import (
        RELEASE_ACCEPTANCE_TIMEOUT_SECONDS as _INNER_RELEASE_ACCEPTANCE_TIMEOUT_SECONDS,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts", "scripts.release_acceptance"}:
        raise
    from release_acceptance import (
        RELEASE_ACCEPTANCE_TIMEOUT_SECONDS as _INNER_RELEASE_ACCEPTANCE_TIMEOUT_SECONDS,
    )


Runner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]


class _ReleaseCompletedProcess(subprocess.CompletedProcess[str]):
    """Completed process carrying a stable commercial-release failure category."""

    failure_kind: str | None

    def __init__(
        self,
        args: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
        *,
        failure_kind: str | None = None,
    ) -> None:
        super().__init__(args=args, returncode=returncode, stdout=stdout, stderr=stderr)
        self.failure_kind = failure_kind


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one release artifact."""
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


def _read_json(path: Path) -> dict[str, Any]:
    """Read one descriptor-safe bounded JSON object from disk."""
    return read_json_object(path)


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    """Resolve a path relative to the repository root when it is not absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _content_security_policy() -> str:
    """Return the restrictive policy used by the self-contained release report."""
    return "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


_DEFAULT_STAGE_TIMEOUT_SECONDS = 300.0
_BUILD_DIST_TIMEOUT_SECONDS = 900.0
_RELEASE_ACCEPTANCE_ORCHESTRATION_MARGIN_SECONDS = 60.0
_RELEASE_ACCEPTANCE_TIMEOUT_SECONDS = (
    sum(_INNER_RELEASE_ACCEPTANCE_TIMEOUT_SECONDS.values())
    + _RELEASE_ACCEPTANCE_ORCHESTRATION_MARGIN_SECONDS
)


def _is_build_dist_command(command: list[str]) -> bool:
    """Return True for the `python -m build` dist-packaging stage command."""
    return len(command) >= 3 and command[1:3] == ["-m", "build"]


def _is_release_acceptance_command(command: list[str]) -> bool:
    """Return True if the command is running release_acceptance.py."""
    return len(command) >= 2 and Path(command[1]).name == "release_acceptance.py"


def _run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one release-stage command and capture its text output."""
    timeout_seconds = _DEFAULT_STAGE_TIMEOUT_SECONDS
    if _is_build_dist_command(command):
        timeout_seconds = _BUILD_DIST_TIMEOUT_SECONDS
    elif _is_release_acceptance_command(command):
        timeout_seconds = _RELEASE_ACCEPTANCE_TIMEOUT_SECONDS

    try:
        return run_bounded_capture(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=10 * 1024 * 1024,
            max_stderr_bytes=10 * 1024 * 1024,
        )
    except BoundedSubprocessOutputError as exc:
        return _ReleaseCompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr=str(exc),
            failure_kind="output_limit",
        )
    except subprocess.TimeoutExpired as exc:
        return _ReleaseCompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr=str(exc),
            failure_kind="timeout",
        )


def _parse_last_json_line(stdout: str) -> dict[str, Any] | None:
    """Parse the final JSON object emitted by a release-stage command."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for index in range(len(lines)):
        candidate = "\n".join(lines[index:])
        try:
            payload = parse_json_bounded(candidate)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _tail(text: str, limit: int = 1200) -> str:
    """Return a bounded suffix suitable for release-stage diagnostics."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _stage(
    name: str,
    command: list[str],
    *,
    repo_root: Path,
    runner: Runner,
) -> dict[str, Any]:
    """Execute one release stage and return stable status and diagnostics."""
    started = time.perf_counter()
    completed = runner(command, repo_root)
    duration = round(time.perf_counter() - started, 6)
    parsed = _parse_last_json_line(completed.stdout)
    stage = {
        "name": name,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "duration_seconds": duration,
        "command": command,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }
    if completed.returncode != 0:
        stage["failure_kind"] = getattr(
            completed, "failure_kind", None
        ) or "subprocess_exit"
    if parsed is not None:
        stage["result"] = parsed
    return stage


def _artifact(path: Path) -> dict[str, Any]:
    """Describe one optional release artifact with existence and digest metadata."""
    return {
        "path": str(path),
        "name": path.name,
        "exists": path.exists() and path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": _sha256(path) if path.exists() and path.is_file() else None,
    }


def _render_html(manifest: dict[str, Any]) -> str:
    """Render the commercial release manifest as a portable HTML report."""
    rows = []
    for stage in manifest.get("stages", []):
        if not isinstance(stage, dict):
            continue
        rows.append(
            "<tr>"
            f'<th scope="row">{escape(str(stage.get("name", "")))}</th>'
            f"<td>{escape(str(stage.get('status', '')))}</td>"
            f"<td>{escape(str(stage.get('duration_seconds', '')))}</td>"
            f"<td><code>{escape(' '.join(str(part) for part in stage.get('command', [])))}</code></td>"
            "</tr>"
        )
    artifact_rows = []
    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, dict):
        for name, artifact in sorted(artifacts.items()):
            if not isinstance(artifact, dict):
                continue
            artifact_rows.append(
                "<tr>"
                f'<th scope="row">{escape(name.replace("_", " ").title())}</th>'
                f"<td>{escape(str(artifact.get('exists', '')))}</td>"
                f"<td>{escape(str(artifact.get('name', '')))}</td>"
                f"<td><code>{escape(str(artifact.get('sha256', '')))}</code></td>"
                "</tr>"
            )
    cards = [
        ("Status", manifest.get("status", "")),
        (
            "Contract Value",
            f"KRW {manifest.get('contract_value_krw', ''):,}"
            if isinstance(manifest.get("contract_value_krw"), int)
            else "",
        ),
        ("Source Commit", manifest.get("source_commit", "")),
        (
            "Stages",
            len(manifest.get("stages", []))
            if isinstance(manifest.get("stages"), list)
            else "",
        ),
        ("Failed Stage", manifest.get("failed_stage", "")),
    ]
    card_markup = [
        "\n".join(
            [
                '<article class="metric-card">',
                f"<span>{escape(label)}</span>",
                f"<strong>{escape(str(value))}</strong>",
                "</article>",
            ]
        )
        for label, value in cards
    ]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f'<meta http-equiv="Content-Security-Policy" content="{escape(_content_security_policy(), quote=True)}">',
            "<title>fast-mlsirm Commercial Release Summary</title>",
            "<style>",
            _report_css(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            '<section class="hero">',
            "<p>fast-mlsirm commercial release</p>",
            "<h1>Commercial Release Summary</h1>",
            f"<span>Generated: {escape(str(manifest.get('generated_at', '')))}</span>",
            "</section>",
            '<section class="report-section">',
            "<h2>Decision Summary</h2>",
            '<div class="metrics-grid">',
            *card_markup,
            "</div>",
            "</section>",
            '<section class="report-section">',
            "<h2>Stage Results</h2>",
            '<div class="table-wrap" role="region" aria-label="Commercial release stage table" tabindex="0">',
            "<table>",
            "<caption>Commercial release stage table</caption>",
            '<thead><tr><th scope="col">Stage</th><th scope="col">Status</th><th scope="col">Seconds</th><th scope="col">Command</th></tr></thead>',
            "<tbody>",
            *rows,
            "</tbody>",
            "</table>",
            "</div>",
            "</section>",
            '<section class="report-section">',
            "<h2>Evidence Artifacts</h2>",
            '<div class="table-wrap" role="region" aria-label="Commercial release artifact table" tabindex="0">',
            "<table>",
            "<caption>Commercial release artifact table</caption>",
            '<thead><tr><th scope="col">Artifact</th><th scope="col">Exists</th><th scope="col">File</th><th scope="col">SHA256</th></tr></thead>',
            "<tbody>",
            *artifact_rows,
            "</tbody>",
            "</table>",
            "</div>",
            '<p class="note">This summary coordinates procurement evidence only. It is not a valuation guarantee or regulated-use approval.</p>',
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _report_css() -> str:
    """Return the small inline stylesheet used by the release report."""
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
  min-width: 760px;
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
code {
  overflow-wrap: anywhere;
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


def _commands(
    args: argparse.Namespace, repo_root: Path, out_dir: Path
) -> list[tuple[str, list[str]]]:
    """Build the ordered core release command stages from CLI options."""
    python = args.python
    scripts = repo_root / "scripts"
    dist_dir = _resolve_path(args.dist, base=repo_root).resolve()
    acceptance_dir = out_dir / "release-acceptance"
    benchmark_dir = acceptance_dir / "benchmark"
    packet_dir = out_dir / "buyer-evidence-packet"
    index_dir = out_dir / "release-evidence-index"
    final_sales = acceptance_dir / "final_sales_readiness_manifest.json"
    stages: list[tuple[str, list[str]]] = []
    if not args.skip_build:
        stages.append(
            ("build_dist", [python, "-m", "build", "--outdir", str(dist_dir)])
        )
    acceptance = [
        python,
        str(scripts / "release_acceptance.py"),
        "--out",
        str(acceptance_dir),
    ]
    if args.require_rust:
        acceptance.append("--require-rust")
    stages.append(("release_acceptance", acceptance))
    stages.append(
        (
            "benchmark_report",
            [
                python,
                str(scripts / "build_benchmark_report.py"),
                "--acceptance",
                str(acceptance_dir / "acceptance_summary.json"),
                "--out",
                str(benchmark_dir),
            ],
        )
    )
    sales = [
        python,
        str(scripts / "sales_readiness.py"),
        "--acceptance",
        str(acceptance_dir / "acceptance_summary.json"),
        "--dist",
        str(dist_dir),
        "--require-20b-product",
        "--benchmark-report",
        str(benchmark_dir / "benchmark_report.json"),
        "--require-benchmark-report",
        "--out",
        str(acceptance_dir / "sales_readiness_manifest.json"),
    ]
    if args.require_rust:
        sales.append("--require-rust")
    if args.check_import:
        sales.append("--check-import")
    stages.append(("sales_readiness", sales))
    stages.append(
        (
            "buyer_packet",
            [
                python,
                str(scripts / "build_buyer_packet.py"),
                "--acceptance",
                str(acceptance_dir / "acceptance_summary.json"),
                "--sales-readiness",
                str(acceptance_dir / "sales_readiness_manifest.json"),
                "--dist",
                str(dist_dir),
                "--benchmark-report",
                str(benchmark_dir / "benchmark_report.json"),
                "--out",
                str(packet_dir),
                "--contract-value-krw",
                str(args.contract_value_krw),
            ],
        )
    )
    stages.append(
        (
            "release_evidence_index",
            [
                python,
                str(scripts / "build_release_evidence_index.py"),
                "--acceptance",
                str(acceptance_dir / "acceptance_summary.json"),
                "--sales-readiness",
                str(acceptance_dir / "sales_readiness_manifest.json"),
                "--dist",
                str(dist_dir),
                "--benchmark-report",
                str(benchmark_dir / "benchmark_report.json"),
                "--buyer-packet-manifest",
                str(packet_dir / "buyer_evidence_manifest.json"),
                "--out",
                str(index_dir),
                "--contract-value-krw",
                str(args.contract_value_krw),
            ],
        )
    )
    final_gate = [
        python,
        str(scripts / "sales_readiness.py"),
        "--acceptance",
        str(acceptance_dir / "acceptance_summary.json"),
        "--dist",
        str(dist_dir),
        "--require-20b-product",
        "--benchmark-report",
        str(benchmark_dir / "benchmark_report.json"),
        "--require-benchmark-report",
        "--buyer-packet-manifest",
        str(packet_dir / "buyer_evidence_manifest.json"),
        "--require-buyer-packet",
        "--release-evidence-index",
        str(index_dir / "release_evidence_index.json"),
        "--require-release-evidence-index",
        "--out",
        str(final_sales),
    ]
    if args.require_rust:
        final_gate.append("--require-rust")
    if args.check_import:
        final_gate.append("--check-import")
    stages.append(("final_sales_readiness", final_gate))
    return stages


def _procurement_command(
    args: argparse.Namespace, repo_root: Path, out_dir: Path
) -> list[str]:
    """Build the optional procurement due-diligence command."""
    command = [
        args.python,
        str(repo_root / "scripts" / "build_procurement_due_diligence.py"),
        "--repo-root",
        str(repo_root),
        "--dist",
        str(_resolve_path(args.dist, base=repo_root).resolve()),
        "--commercial-release-manifest",
        str(out_dir / "commercial_release_manifest.json"),
        "--out",
        str(out_dir / "procurement-due-diligence"),
        "--repo",
        getattr(args, "repo", "ContextualWisdomLab/fast-mlsirm"),
        "--contract-value-krw",
        str(args.contract_value_krw),
    ]
    if getattr(args, "offline_github", False):
        command.append("--offline-github")
    return command


def _pr_queue_command(
    args: argparse.Namespace, repo_root: Path, out_dir: Path
) -> list[str]:
    """Build the optional PR queue governance command."""
    command = [
        args.python,
        str(repo_root / "scripts" / "build_pr_queue_governance.py"),
        "--repo-root",
        str(repo_root),
        "--out",
        str(out_dir / "pr-queue-governance"),
        "--repo",
        getattr(args, "repo", "ContextualWisdomLab/fast-mlsirm"),
        "--contract-value-krw",
        str(args.contract_value_krw),
        "--max-stale-days",
        str(getattr(args, "pr_queue_max_stale_days", 14)),
    ]
    offline_snapshot = getattr(args, "pr_queue_offline_snapshot", None)
    if offline_snapshot:
        command.extend(
            [
                "--offline-snapshot",
                str(_resolve_path(offline_snapshot, base=repo_root).resolve()),
            ]
        )
    return command


def _figma_sync_command(
    args: argparse.Namespace, repo_root: Path, out_dir: Path
) -> list[str]:
    """Build the optional Figma evidence synchronization command."""
    command = [
        args.python,
        str(repo_root / "scripts" / "build_figma_evidence_sync.py"),
        "--repo-root",
        str(repo_root),
        "--packet",
        str(repo_root / "examples" / "enterprise_demo" / "figma_design_packet.json"),
        "--out",
        str(out_dir / "figma-evidence-sync"),
        "--contract-value-krw",
        str(args.contract_value_krw),
        "--figma-url",
        getattr(
            args, "figma_url", "https://www.figma.com/design/qD34PfMH8Kr41tFdqLCkem"
        ),
    ]
    metadata_snapshot = getattr(args, "figma_metadata_snapshot", None)
    if metadata_snapshot:
        command.extend(
            [
                "--metadata-snapshot",
                str(_resolve_path(metadata_snapshot, base=repo_root).resolve()),
            ]
        )
    return command


def _artifacts(args: argparse.Namespace, out_dir: Path) -> dict[str, dict[str, Any]]:
    """Collect known release artifact paths and their current digests."""
    acceptance_dir = out_dir / "release-acceptance"
    repo_root = Path(args.repo_root).resolve()
    dist_dir = _resolve_path(args.dist, base=repo_root).resolve()
    return {
        "acceptance_summary": _artifact(acceptance_dir / "acceptance_summary.json"),
        "benchmark_report": _artifact(
            acceptance_dir / "benchmark" / "benchmark_report.json"
        ),
        "benchmark_html": _artifact(
            acceptance_dir / "benchmark" / "benchmark_report.html"
        ),
        "sales_readiness": _artifact(acceptance_dir / "sales_readiness_manifest.json"),
        "buyer_packet_manifest": _artifact(
            out_dir / "buyer-evidence-packet" / "buyer_evidence_manifest.json"
        ),
        "buyer_packet_zip": _artifact(
            out_dir / "buyer-evidence-packet" / "fast_mlsirm_buyer_evidence_packet.zip"
        ),
        "buyer_packet_html": _artifact(
            out_dir / "buyer-evidence-packet" / "buyer_evidence_report.html"
        ),
        "release_evidence_index": _artifact(
            out_dir / "release-evidence-index" / "release_evidence_index.json"
        ),
        "release_evidence_html": _artifact(
            out_dir / "release-evidence-index" / "release_evidence_index.html"
        ),
        "final_sales_readiness": _artifact(
            acceptance_dir / "final_sales_readiness_manifest.json"
        ),
        "procurement_due_diligence": _artifact(
            out_dir
            / "procurement-due-diligence"
            / "procurement_due_diligence_manifest.json"
        ),
        "procurement_due_diligence_html": _artifact(
            out_dir
            / "procurement-due-diligence"
            / "procurement_due_diligence_report.html"
        ),
        "pr_queue_governance": _artifact(
            out_dir / "pr-queue-governance" / "pr_queue_governance_manifest.json"
        ),
        "pr_queue_governance_html": _artifact(
            out_dir / "pr-queue-governance" / "pr_queue_governance_report.html"
        ),
        "figma_evidence_sync": _artifact(
            out_dir / "figma-evidence-sync" / "figma_evidence_sync_manifest.json"
        ),
        "figma_evidence_sync_html": _artifact(
            out_dir / "figma-evidence-sync" / "figma_evidence_sync_report.html"
        ),
        "wheel": _artifact(
            next(iter(sorted(dist_dir.glob("*.whl"))), dist_dir / "missing.whl")
        ),
        "sdist": _artifact(
            next(iter(sorted(dist_dir.glob("*.tar.gz"))), dist_dir / "missing.tar.gz")
        ),
    }


def _write_outputs(
    manifest: dict[str, Any], manifest_path: Path, html_path: Path
) -> None:
    """Write the release manifest and its digest-bearing HTML report."""
    html_path.write_text(_render_html(manifest), encoding="utf-8")
    manifest["html_report_sha256"] = _sha256(html_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def build_commercial_release(
    args: argparse.Namespace, *, runner: Runner = _run_command
) -> dict[str, Any]:
    """Run commercial release stages and return the final evidence manifest."""
    repo_root = Path(args.repo_root).resolve()
    out_dir = _resolve_path(args.out, base=repo_root).resolve()
    dist_dir = _resolve_path(args.dist, base=repo_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "commercial_release_manifest.json"
    html_path = out_dir / "commercial_release_report.html"
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "command": "build_commercial_release",
        "status": "failed",
        "contract_value_krw": args.contract_value_krw,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": _source_commit(repo_root),
        "repo_root": str(repo_root),
        "out": str(out_dir),
        "dist": str(dist_dir),
        "stages": [],
        "failed_stage": None,
        "artifacts": {},
        "html_report_file": str(html_path),
    }
    for name, command in _commands(args, repo_root, out_dir):
        stage = _stage(name, command, repo_root=repo_root, runner=runner)
        manifest["stages"].append(stage)
        if stage["status"] != "ok":
            manifest["failed_stage"] = name
            break
    manifest["total_duration_seconds"] = round(time.perf_counter() - started, 6)
    manifest["artifacts"] = _artifacts(args, out_dir)
    if manifest["failed_stage"] is None and all(
        stage.get("status") == "ok" for stage in manifest["stages"]
    ):
        final_path = (
            out_dir / "release-acceptance" / "final_sales_readiness_manifest.json"
        )
        final_status = (
            _read_json(final_path).get("status") if final_path.exists() else None
        )
        manifest["status"] = "ok" if final_status == "ok" else "failed"
        if final_status != "ok":
            manifest["failed_stage"] = "final_sales_readiness"
    _write_outputs(manifest, manifest_path, html_path)
    if manifest["status"] == "ok" and not getattr(
        args, "skip_procurement_due_diligence", False
    ):
        stage = _stage(
            "procurement_due_diligence",
            _procurement_command(args, repo_root, out_dir),
            repo_root=repo_root,
            runner=runner,
        )
        manifest["stages"].append(stage)
        if stage["status"] != "ok":
            manifest["status"] = "failed"
            manifest["failed_stage"] = "procurement_due_diligence"
        manifest["total_duration_seconds"] = round(time.perf_counter() - started, 6)
        manifest["artifacts"] = _artifacts(args, out_dir)
        _write_outputs(manifest, manifest_path, html_path)
    if manifest["status"] == "ok" and not getattr(
        args, "skip_pr_queue_governance", False
    ):
        stage = _stage(
            "pr_queue_governance",
            _pr_queue_command(args, repo_root, out_dir),
            repo_root=repo_root,
            runner=runner,
        )
        manifest["stages"].append(stage)
        if stage["status"] != "ok":
            manifest["status"] = "failed"
            manifest["failed_stage"] = "pr_queue_governance"
        manifest["total_duration_seconds"] = round(time.perf_counter() - started, 6)
        manifest["artifacts"] = _artifacts(args, out_dir)
        _write_outputs(manifest, manifest_path, html_path)
    if manifest["status"] == "ok" and not getattr(
        args, "skip_figma_evidence_sync", False
    ):
        stage = _stage(
            "figma_evidence_sync",
            _figma_sync_command(args, repo_root, out_dir),
            repo_root=repo_root,
            runner=runner,
        )
        manifest["stages"].append(stage)
        if stage["status"] != "ok":
            manifest["status"] = "failed"
            manifest["failed_stage"] = "figma_evidence_sync"
        manifest["total_duration_seconds"] = round(time.perf_counter() - started, 6)
        manifest["artifacts"] = _artifacts(args, out_dir)
        _write_outputs(manifest, manifest_path, html_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for commercial release generation."""
    parser = argparse.ArgumentParser(
        description="Build the full fast-mlsirm commercial release evidence bundle."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--out",
        default="commercial-release",
        help="Output directory for all commercial release evidence.",
    )
    parser.add_argument(
        "--dist",
        default="dist",
        help="Directory containing or receiving built wheel and sdist artifacts.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for subcommands.",
    )
    parser.add_argument(
        "--contract-value-krw",
        type=int,
        default=2_000_000_000,
        help="Target contract value.",
    )
    parser.add_argument(
        "--require-rust",
        action="store_true",
        help="Require explicit Rust backend evidence.",
    )
    parser.add_argument(
        "--check-import",
        action="store_true",
        help="Validate importability in sales readiness gates.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use existing dist artifacts instead of running python -m build.",
    )
    parser.add_argument(
        "--skip-procurement-due-diligence",
        action="store_true",
        help="Skip the procurement due-diligence evidence stage.",
    )
    parser.add_argument(
        "--skip-pr-queue-governance",
        action="store_true",
        help="Skip the PR queue governance evidence stage.",
    )
    parser.add_argument(
        "--skip-figma-evidence-sync",
        action="store_true",
        help="Skip the Figma evidence sync stage.",
    )
    parser.add_argument(
        "--offline-github",
        action="store_true",
        help="Use offline GitHub snapshot mode for due diligence.",
    )
    parser.add_argument(
        "--pr-queue-offline-snapshot",
        help="Optional offline PR queue snapshot JSON for governance evidence.",
    )
    parser.add_argument(
        "--pr-queue-max-stale-days",
        type=int,
        default=14,
        help="Age in days after which an open PR is stale.",
    )
    parser.add_argument(
        "--figma-metadata-snapshot",
        help="Optional exported live Figma metadata snapshot JSON.",
    )
    parser.add_argument(
        "--figma-url",
        default="https://www.figma.com/design/qD34PfMH8Kr41tFdqLCkem",
        help="Fallback Figma design URL for the evidence sync stage.",
    )
    parser.add_argument(
        "--repo",
        default="ContextualWisdomLab/fast-mlsirm",
        help="GitHub repository for due-diligence snapshots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run commercial release generation and print a machine-readable summary."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_commercial_release(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "out": manifest["out"],
                "manifest": str(
                    Path(manifest["out"]) / "commercial_release_manifest.json"
                ),
                "html": manifest["html_report_file"],
                "failed_stage": manifest["failed_stage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
