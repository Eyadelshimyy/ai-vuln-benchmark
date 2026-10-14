#!/usr/bin/env python
"""Build a release evidence index for fast-mlsirm procurement review."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

try:
    from scripts._bounded_json import read_json_object
except ModuleNotFoundError:
    from _bounded_json import read_json_object


GIT_METADATA_TIMEOUT_SECONDS = 5


REQUIRED_COVERAGE = {
    "acceptance_summary",
    "sales_readiness_manifest",
    "benchmark_report",
    "benchmark_html_report",
    "buyer_packet_manifest",
    "buyer_packet_zip",
    "buyer_packet_html_report",
    "wheel",
    "sdist",
}


def _read_json(path: Path) -> dict[str, Any]:
    """Read one descriptor-safe bounded JSON object from disk."""
    return read_json_object(path)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one release evidence file."""
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


def _project_version(repo_root: Path) -> str:
    """Read the package version from the repository's ``pyproject.toml``."""
    pyproject = repo_root / "pyproject.toml"
    try:
        import tomllib
    except ModuleNotFoundError:
        return _parse_project_version(pyproject.read_text(encoding="utf-8"))
    with pyproject.open("rb") as fh:
        payload = tomllib.load(fh)
    return str(payload["project"]["version"])


def _parse_project_version(pyproject_text: str) -> str:
    """Extract a project version without requiring the TOML parser."""
    in_project = False
    for raw_line in pyproject_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project:
            key, separator, value = line.partition("=")
            if separator and key.strip() == "version":
                return value.strip().strip("\"'")
    return "unknown"


def _resolve_artifact_path(value: Any, *, base: Path) -> Path | None:
    """Resolve a manifest artifact path relative to its manifest directory."""
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _file_entry(role: str, path: Path) -> dict[str, Any]:
    """Describe one release artifact with size and content digest."""
    return {
        "role": role,
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _dist_entries(dist_dir: Path) -> list[dict[str, Any]]:
    """Collect wheel and source-distribution files from a distribution folder."""
    entries: list[dict[str, Any]] = []
    for path in sorted(dist_dir.glob("*.whl")):
        entry = _file_entry("wheel", path)
        entry["kind"] = "wheel"
        entries.append(entry)
    for path in sorted(dist_dir.glob("*.tar.gz")):
        entry = _file_entry("sdist", path)
        entry["kind"] = "sdist"
        entries.append(entry)
    return entries


def _content_security_policy() -> str:
    """Return the restrictive policy used by the self-contained HTML index."""
    return "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def _format_value(value: Any) -> str:
    """Format a status value for a human-readable evidence table."""
    if isinstance(value, bool):
        return "go" if value else "missing"
    if value is None:
        return ""
    return str(value)


def _render_rows(
    rows: list[dict[str, Any]], columns: list[tuple[str, str]]
) -> list[str]:
    """Render evidence records as escaped accessible HTML table rows."""
    rendered = []
    for row in rows:
        cells = []
        for key, _label in columns:
            value = escape(_format_value(row.get(key)))
            if not cells:
                cells.append(f'<th scope="row">{value}</th>')
            else:
                cells.append(f"<td>{value}</td>")
        rendered.append("<tr>" + "".join(cells) + "</tr>")
    return rendered


def _render_report_html(index: dict[str, Any]) -> str:
    """Render the release evidence index as portable self-contained HTML."""
    coverage_rows = []
    coverage = index.get("coverage", {})
    if isinstance(coverage, dict):
        for name, ok in sorted(coverage.items()):
            coverage_rows.append(
                "<tr>"
                f'<th scope="row">{escape(name.replace("_", " ").title())}</th>'
                f"<td>{escape(_format_value(ok))}</td>"
                "</tr>"
            )

    file_rows = _render_rows(
        index.get("files", [])[:30],
        [
            ("role", "Role"),
            ("name", "Name"),
            ("size_bytes", "Bytes"),
            ("sha256", "SHA256"),
        ],
    )
    cards = [
        ("Status", index.get("status", "")),
        (
            "Contract Value",
            f"KRW {index.get('contract_value_krw', ''):,}"
            if isinstance(index.get("contract_value_krw"), int)
            else "",
        ),
        ("Version", index.get("project_version", "")),
        ("Source Commit", index.get("source_commit", "")),
        (
            "Dist Artifacts",
            len(index.get("dist", {}).get("artifacts", []))
            if isinstance(index.get("dist"), dict)
            else "",
        ),
        (
            "Failed Checks",
            len(index.get("failures", []))
            if isinstance(index.get("failures"), list)
            else "",
        ),
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
    failures = index.get("failures", [])
    failure_items = []
    if isinstance(failures, list):
        failure_items = [f"<li>{escape(str(item))}</li>" for item in failures]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f'<meta http-equiv="Content-Security-Policy" content="{escape(_content_security_policy(), quote=True)}">',
            "<title>fast-mlsirm Release Evidence Index</title>",
            "<style>",
            _report_css(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            '<section class="hero">',
            "<p>fast-mlsirm release review</p>",
            "<h1>Release Evidence Index</h1>",
            f"<span>Generated: {escape(str(index.get('generated_at', '')))}</span>",
            "</section>",
            '<section class="report-section">',
            "<h2>Decision Summary</h2>",
            '<div class="metrics-grid">',
            *card_markup,
            "</div>",
            "</section>",
            '<section class="report-section">',
            "<h2>Required Evidence Coverage</h2>",
            '<div class="table-wrap" role="region" aria-label="Required evidence coverage table" tabindex="0">',
            "<table>",
            "<caption>Required evidence coverage table</caption>",
            '<thead><tr><th scope="col">Evidence</th><th scope="col">Status</th></tr></thead>',
            "<tbody>",
            *coverage_rows,
            "</tbody>",
            "</table>",
            "</div>",
            "</section>",
            '<section class="report-section">',
            "<h2>Artifact Digests</h2>",
            '<div class="table-wrap" role="region" aria-label="Release artifact digest table" tabindex="0">',
            "<table>",
            "<caption>Release artifact digest table</caption>",
            '<thead><tr><th scope="col">Role</th><th scope="col">Name</th><th scope="col">Bytes</th><th scope="col">SHA256</th></tr></thead>',
            "<tbody>",
            *file_rows,
            "</tbody>",
            "</table>",
            "</div>",
            "</section>",
            '<section class="report-section">',
            "<h2>Failures</h2>",
            "<ul>",
            *failure_items,
            "</ul>",
            '<p class="note">This index organizes release evidence for procurement review. It is not a valuation guarantee or regulated-use approval.</p>',
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _report_css() -> str:
    """Return the small inline stylesheet used by the evidence index."""
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


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    """Build, validate, and return the release evidence index."""
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "release_evidence_index.json"
    html_path = out_dir / "release_evidence_index.html"

    acceptance_path = Path(args.acceptance).resolve()
    sales_path = Path(args.sales_readiness).resolve()
    benchmark_path = Path(args.benchmark_report).resolve()
    buyer_packet_path = Path(args.buyer_packet_manifest).resolve()
    dist_dir = Path(args.dist).resolve()

    acceptance = _read_json(acceptance_path)
    sales = _read_json(sales_path)
    benchmark = _read_json(benchmark_path)
    buyer_packet = _read_json(buyer_packet_path)

    benchmark_html = _resolve_artifact_path(
        benchmark.get("html_report_file"), base=benchmark_path.parent
    )
    buyer_zip = _resolve_artifact_path(
        buyer_packet.get("zip_file"), base=buyer_packet_path.parent
    )
    buyer_html = _resolve_artifact_path(
        buyer_packet.get("report_file"), base=buyer_packet_path.parent
    )
    dist_artifacts = _dist_entries(dist_dir)

    files: list[dict[str, Any]] = []
    failures: list[str] = []
    for role, path in [
        ("acceptance_summary", acceptance_path),
        ("sales_readiness_manifest", sales_path),
        ("benchmark_report", benchmark_path),
        ("buyer_packet_manifest", buyer_packet_path),
    ]:
        if path.exists() and path.is_file():
            files.append(_file_entry(role, path))
        else:
            failures.append(f"missing {role}: {path}")
    for role, path in [
        ("benchmark_html_report", benchmark_html),
        ("buyer_packet_zip", buyer_zip),
        ("buyer_packet_html_report", buyer_html),
    ]:
        if path is not None and path.exists() and path.is_file():
            files.append(_file_entry(role, path))
        else:
            failures.append(f"missing {role}: {path}")
    files.extend(dist_artifacts)

    coverage = {
        "acceptance_summary": acceptance_path.exists() and acceptance_path.is_file(),
        "sales_readiness_manifest": sales_path.exists() and sales_path.is_file(),
        "benchmark_report": benchmark_path.exists() and benchmark_path.is_file(),
        "benchmark_html_report": benchmark_html is not None
        and benchmark_html.exists()
        and benchmark_html.is_file(),
        "buyer_packet_manifest": buyer_packet_path.exists()
        and buyer_packet_path.is_file(),
        "buyer_packet_zip": buyer_zip is not None
        and buyer_zip.exists()
        and buyer_zip.is_file(),
        "buyer_packet_html_report": buyer_html is not None
        and buyer_html.exists()
        and buyer_html.is_file(),
        "wheel": any(entry.get("kind") == "wheel" for entry in dist_artifacts),
        "sdist": any(entry.get("kind") == "sdist" for entry in dist_artifacts),
    }

    benchmark_html_sha = (
        _sha256(benchmark_html)
        if coverage["benchmark_html_report"] and benchmark_html is not None
        else None
    )
    if benchmark_html_sha != benchmark.get("html_report_sha256"):
        failures.append("benchmark HTML SHA256 does not match benchmark_report.json")
    buyer_zip_sha = (
        _sha256(buyer_zip)
        if coverage["buyer_packet_zip"] and buyer_zip is not None
        else None
    )
    if buyer_zip_sha != buyer_packet.get("zip_sha256"):
        failures.append(
            "buyer packet ZIP SHA256 does not match buyer_evidence_manifest.json"
        )
    buyer_html_sha = (
        _sha256(buyer_html)
        if coverage["buyer_packet_html_report"] and buyer_html is not None
        else None
    )
    if buyer_html_sha != buyer_packet.get("report_sha256"):
        failures.append(
            "buyer packet HTML SHA256 does not match buyer_evidence_manifest.json"
        )

    if acceptance.get("status") != "ok":
        failures.append("acceptance_summary.json status is not ok")
    if sales.get("status") != "ok":
        failures.append("sales_readiness_manifest.json status is not ok")
    if benchmark.get("status") != "ok" or benchmark.get("budget_ok") is not True:
        failures.append("benchmark_report.json status or budget is not ok")
    if buyer_packet.get("status") != "ok":
        failures.append("buyer_evidence_manifest.json status is not ok")
    missing_coverage = [
        name for name in sorted(REQUIRED_COVERAGE) if coverage.get(name) is not True
    ]
    if missing_coverage:
        failures.append(
            f"missing required release evidence coverage: {missing_coverage}"
        )

    index: dict[str, Any] = {
        "command": "build_release_evidence_index",
        "status": "failed",
        "contract_value_krw": args.contract_value_krw,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": _source_commit(repo_root),
        "project_version": _project_version(repo_root),
        "repo_root": str(repo_root),
        "coverage": coverage,
        "dist": {
            "directory": str(dist_dir),
            "wheel_count": sum(
                1 for entry in dist_artifacts if entry.get("kind") == "wheel"
            ),
            "sdist_count": sum(
                1 for entry in dist_artifacts if entry.get("kind") == "sdist"
            ),
            "artifacts": dist_artifacts,
        },
        "acceptance": {
            "file": str(acceptance_path),
            "status": acceptance.get("status"),
            "total_duration_seconds": acceptance.get("total_duration_seconds"),
            "command_count": len(
                [step for step in acceptance.get("steps", []) if isinstance(step, dict)]
            ),
        },
        "sales_readiness": {
            "file": str(sales_path),
            "status": sales.get("status"),
            "failed_check_count": len(sales.get("failed_checks", []))
            if isinstance(sales.get("failed_checks"), list)
            else None,
            "require_20b_product": sales.get("require_20b_product"),
            "require_buyer_packet": sales.get("require_buyer_packet"),
            "require_benchmark_report": sales.get("require_benchmark_report"),
        },
        "benchmark": {
            "file": str(benchmark_path),
            "status": benchmark.get("status"),
            "budget_ok": benchmark.get("budget_ok"),
            "runtime_budget_seconds": benchmark.get("runtime_budget_seconds"),
            "total_duration_seconds": benchmark.get("total_duration_seconds"),
            "html_report_file": str(benchmark_html)
            if benchmark_html is not None
            else None,
            "html_report_sha256": benchmark.get("html_report_sha256"),
        },
        "buyer_packet": {
            "manifest_file": str(buyer_packet_path),
            "status": buyer_packet.get("status"),
            "artifact_count": buyer_packet.get("artifact_count"),
            "zip_file": str(buyer_zip) if buyer_zip is not None else None,
            "zip_sha256": buyer_packet.get("zip_sha256"),
            "report_file": str(buyer_html) if buyer_html is not None else None,
            "report_sha256": buyer_packet.get("report_sha256"),
        },
        "files": sorted(
            files,
            key=lambda item: (str(item.get("role", "")), str(item.get("name", ""))),
        ),
        "failures": sorted(set(failures)),
        "html_report_file": str(html_path),
    }
    index["status"] = (
        "ok" if not index["failures"] and all(index["coverage"].values()) else "failed"
    )
    html_path.write_text(_render_report_html(index), encoding="utf-8")
    index["html_report_sha256"] = _sha256(html_path)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for release evidence index generation."""
    parser = argparse.ArgumentParser(
        description="Build a release evidence index for fast-mlsirm."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to record source commit and version.",
    )
    parser.add_argument(
        "--acceptance", required=True, help="Path to acceptance_summary.json."
    )
    parser.add_argument(
        "--sales-readiness",
        required=True,
        help="Path to sales_readiness_manifest.json.",
    )
    parser.add_argument(
        "--benchmark-report", required=True, help="Path to benchmark_report.json."
    )
    parser.add_argument(
        "--buyer-packet-manifest",
        required=True,
        help="Path to buyer_evidence_manifest.json.",
    )
    parser.add_argument(
        "--dist",
        required=True,
        help="Directory containing release wheel and sdist artifacts.",
    )
    parser.add_argument(
        "--out",
        default="release-evidence-index",
        help="Output directory for release evidence index files.",
    )
    parser.add_argument(
        "--contract-value-krw",
        type=int,
        default=2_000_000_000,
        help="Target contract value.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run release evidence indexing and print a machine-readable summary."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        index = build_index(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": index["status"],
                "out": str(Path(args.out).resolve()),
                "index": str(Path(args.out).resolve() / "release_evidence_index.json"),
                "html": index["html_report_file"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if index["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
