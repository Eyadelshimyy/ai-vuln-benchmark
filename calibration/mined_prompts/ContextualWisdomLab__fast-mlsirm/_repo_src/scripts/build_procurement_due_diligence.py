#!/usr/bin/env python
"""Build procurement due-diligence evidence for a fast-mlsirm release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import zipfile
from datetime import UTC, datetime
from email.parser import Parser
from html import escape
from pathlib import Path
from typing import Any

GIT_METADATA_TIMEOUT_SECONDS = 5
_FULL_OBJECT_ID_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_GH_COMMAND_TIMEOUT_SECONDS = 60

try:
    from scripts._bounded_json import MAX_JSON_BYTES, parse_json_bounded, read_json_object
    from scripts._bounded_subprocess import BoundedSubprocessOutputError, run_bounded_capture
except ModuleNotFoundError:
    from _bounded_json import MAX_JSON_BYTES, parse_json_bounded, read_json_object
    from _bounded_subprocess import BoundedSubprocessOutputError, run_bounded_capture

_GH_STDOUT_MAX_BYTES = MAX_JSON_BYTES
_GH_STDERR_MAX_BYTES = 1024 * 1024


POLICY_FILES = [
    "README.md",
    "LICENSE",
    "SECURITY.md",
    "SUPPORT.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "docs/commercial_readiness.md",
    "docs/enterprise_sales_readiness.md",
    "docs/release_acceptance.md",
    "docs/20b_product_readiness.md",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one descriptor-safe bounded JSON object from disk."""
    return read_json_object(path)


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _metadata_dict(text: str) -> dict[str, str]:
    parsed = Parser().parsestr(text)
    return {
        key: parsed.get(key, "")
        for key in ["Name", "Version", "License", "Requires-Python"]
    }


def parse_wheel(path: Path) -> dict[str, Any]:
    required = {"METADATA": None, "WHEEL": None, "RECORD": None}
    metadata: dict[str, str] = {}
    members: list[str] = []
    if not path.exists():
        return {
            "ok": False,
            "path": str(path),
            "missing": sorted(required),
            "metadata": metadata,
        }
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        for name in members:
            for required_name in list(required):
                if name.endswith(f".dist-info/{required_name}"):
                    required[required_name] = name
        if required["METADATA"] is not None:
            metadata = _metadata_dict(
                archive.read(required["METADATA"]).decode("utf-8", errors="replace")
            )
    missing = sorted(name for name, member in required.items() if member is None)
    return {
        "ok": not missing,
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "metadata": metadata,
        "dist_info": required,
        "missing": missing,
        "member_count": len(members),
    }


def parse_sdist(path: Path) -> dict[str, Any]:
    metadata: dict[str, str] = {}
    pkg_info_name = None
    if not path.exists():
        return {
            "ok": False,
            "path": str(path),
            "missing": ["PKG-INFO"],
            "metadata": metadata,
        }
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name.endswith("PKG-INFO"):
                pkg_info_name = member.name
                fh = archive.extractfile(member)
                if fh is not None:
                    metadata = _metadata_dict(
                        fh.read().decode("utf-8", errors="replace")
                    )
                break
    return {
        "ok": pkg_info_name is not None,
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "metadata": metadata,
        "pkg_info": pkg_info_name,
        "missing": [] if pkg_info_name is not None else ["PKG-INFO"],
    }


def _project_metadata(repo_root: Path) -> dict[str, str]:
    pyproject = repo_root / "pyproject.toml"
    try:
        import tomllib
    except ModuleNotFoundError:
        return _parse_project_metadata(pyproject.read_text(encoding="utf-8"))
    with pyproject.open("rb") as fh:
        project = tomllib.load(fh).get("project", {})
    return {
        "name": str(project.get("name", "")),
        "version": str(project.get("version", "")),
        "requires_python": str(project.get("requires-python", "")),
    }


def _parse_project_metadata(text: str) -> dict[str, str]:
    in_project = False
    values = {"name": "", "version": "", "requires_python": ""}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if not in_project:
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        normalized = key.strip().replace("-", "_")
        if normalized in values:
            values[normalized] = value.strip().strip("\"'")
    return values


def _source_commit(repo_root: Path) -> str:
    """Return the canonical full commit SHA for the checked-out HEAD.

    The value must reconstruct the exact procurement-evidence source, so every
    failure mode fails closed: timeouts, missing executables, non-zero exits,
    empty output, and any identity that is not a full lowercase SHA-1 or
    SHA-256 object name raise :class:`RuntimeError` instead of degrading to an
    unreconstructable placeholder.
    """
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
    candidate = completed.stdout.strip()
    if not _FULL_OBJECT_ID_PATTERN.fullmatch(candidate):
        raise RuntimeError("source commit is not a full lowercase SHA-1 or SHA-256 object id")
    return candidate


def _check(
    name: str, category: str, ok: bool, detail: str, **metadata: Any
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "category": category,
        "ok": ok,
        "detail": detail,
    }
    payload.update(metadata)
    return payload


def _policy_checks(repo_root: Path) -> list[dict[str, Any]]:
    checks = []
    for relative in POLICY_FILES:
        path = repo_root / relative
        checks.append(
            _check(
                f"policy_file:{relative}",
                "policy",
                path.exists() and path.is_file() and path.stat().st_size > 0,
                "required procurement policy file exists",
                path=str(path),
            )
        )
    workflow = repo_root / ".github" / "workflows" / "ci.yml"
    checks.append(
        _check(
            "workflow:ci",
            "github",
            workflow.exists(),
            "CI workflow file exists",
            path=str(workflow),
        )
    )
    return checks


def _commercial_checks(
    path: Path, *, contract_value_krw: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        return {}, [
            _check(
                "commercial_release:manifest",
                "commercial_release",
                False,
                f"missing manifest: {path}",
            )
        ]
    payload = _read_json(path)
    checks = [
        _check(
            "commercial_release:status",
            "commercial_release",
            payload.get("status") == "ok",
            "commercial release manifest status is ok",
            actual=payload.get("status"),
        ),
        _check(
            "commercial_release:contract_value",
            "commercial_release",
            payload.get("contract_value_krw") == contract_value_krw,
            "commercial release contract value matches due-diligence gate",
            expected=contract_value_krw,
            actual=payload.get("contract_value_krw"),
        ),
    ]
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, dict):
        for name in ["wheel", "sdist", "final_sales_readiness"]:
            artifact = artifacts.get(name, {})
            ok = (
                isinstance(artifact, dict)
                and artifact.get("exists") is True
                and isinstance(artifact.get("sha256"), str)
            )
            checks.append(
                _check(
                    f"commercial_release:artifact:{name}",
                    "commercial_release",
                    ok,
                    "commercial release manifest records required artifact digest",
                    artifact=artifact,
                )
            )
    return payload, checks


def _bounded_json_snapshot(command: list[str]) -> dict[str, Any]:
    """Run one GitHub JSON command with bounded output and stable failures."""
    try:
        completed = run_bounded_capture(
            command,
            timeout_seconds=_GH_COMMAND_TIMEOUT_SECONDS,
            max_stdout_bytes=_GH_STDOUT_MAX_BYTES,
            max_stderr_bytes=_GH_STDERR_MAX_BYTES,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": 124,
            "data": None,
            "stderr": f"command timed out after {_GH_COMMAND_TIMEOUT_SECONDS} seconds",
        }
    except BoundedSubprocessOutputError as exc:
        return {
            "ok": False,
            "returncode": 75,
            "data": None,
            "stderr": str(exc),
        }
    if completed.returncode != 0:
        return {
            "ok": False,
            "returncode": completed.returncode,
            "data": None,
            "stderr": completed.stderr.strip(),
        }
    try:
        data = (
            parse_json_bounded(completed.stdout, max_bytes=_GH_STDOUT_MAX_BYTES)
            if completed.stdout.strip()
            else None
        )
    except ValueError as exc:
        return {
            "ok": False,
            "returncode": 65,
            "data": None,
            "stderr": str(exc),
        }
    return {
        "ok": True,
        "returncode": 0,
        "data": data,
        "stderr": completed.stderr.strip(),
    }


def _bounded_lines_snapshot(command: list[str]) -> dict[str, Any]:
    """Run one GitHub text command with bounded output and stable failures."""
    try:
        completed = run_bounded_capture(
            command,
            timeout_seconds=_GH_COMMAND_TIMEOUT_SECONDS,
            max_stdout_bytes=_GH_STDOUT_MAX_BYTES,
            max_stderr_bytes=_GH_STDERR_MAX_BYTES,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": 124,
            "lines": [],
            "stderr": f"command timed out after {_GH_COMMAND_TIMEOUT_SECONDS} seconds",
        }
    except BoundedSubprocessOutputError as exc:
        return {
            "ok": False,
            "returncode": 75,
            "lines": [],
            "stderr": str(exc),
        }
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "lines": [line for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode == 0
        else [],
        "stderr": completed.stderr.strip(),
    }


def _github_snapshot(repo: str, *, offline: bool) -> dict[str, Any]:
    if offline:
        return {"mode": "offline", "repo": repo, "checks": {"snapshot_recorded": True}}
    snapshot: dict[str, Any] = {"mode": "live", "repo": repo}
    commands = {
        "repo": [
            "gh",
            "repo",
            "view",
            repo,
            "--json",
            "nameWithOwner,defaultBranchRef,visibility,isArchived,pushedAt,updatedAt,url",
        ],
        "open_prs": [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,headRefName,baseRefName,isDraft,mergeStateStatus,reviewDecision,updatedAt,url",
        ],
    }
    for name, command in commands.items():
        snapshot[name] = _bounded_json_snapshot(command)
    snapshot["releases"] = _bounded_lines_snapshot(
        ["gh", "release", "list", "--repo", repo, "--limit", "20"]
    )
    return snapshot


def _github_checks(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    if snapshot.get("mode") == "offline":
        return [
            _check(
                "github:snapshot",
                "github",
                True,
                "GitHub snapshot intentionally recorded in offline mode",
            )
        ]
    repo_data = snapshot.get("repo", {})
    repo_payload = repo_data.get("data", {}) if isinstance(repo_data, dict) else {}
    checks = [
        _check(
            "github:repo_snapshot",
            "github",
            bool(repo_data.get("ok")),
            "GitHub repository metadata was read",
        ),
        _check(
            "github:not_archived",
            "github",
            isinstance(repo_payload, dict) and repo_payload.get("isArchived") is False,
            "GitHub repository is not archived",
            actual=repo_payload.get("isArchived")
            if isinstance(repo_payload, dict)
            else None,
        ),
        _check(
            "github:open_pr_snapshot",
            "github",
            bool(snapshot.get("open_prs", {}).get("ok")),
            "Open PR state was recorded",
        ),
        _check(
            "github:release_snapshot",
            "github",
            bool(snapshot.get("releases", {}).get("ok")),
            "Release list state was recorded",
        ),
    ]
    return checks


def _content_security_policy() -> str:
    return "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def _report_css() -> str:
    return """
:root { color: #172026; background: #f5f7f8; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 20px 56px; }
.hero { background: #12343b; color: #fff; border-radius: 8px; padding: 28px; }
.hero p, .hero h1 { margin: 0; }
.hero p { color: #b7d7d0; font-size: 0.86rem; font-weight: 700; text-transform: uppercase; }
.hero h1 { margin-top: 8px; font-size: 2rem; }
.hero span { display: inline-block; margin-top: 14px; color: #dce8e5; }
.report-section { margin-top: 22px; background: #fff; border: 1px solid #d8e1e3; border-radius: 8px; padding: 22px; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.metric-card { border: 1px solid #d8e1e3; border-radius: 8px; padding: 14px; }
.metric-card span { display: block; color: #5e6f76; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; }
.metric-card strong { display: block; margin-top: 8px; overflow-wrap: anywhere; }
.table-wrap { overflow-x: auto; border: 1px solid #d8e1e3; border-radius: 8px; }
.table-wrap:focus-visible { outline: 3px solid #0f766e; outline-offset: 3px; }
table { width: 100%; min-width: 760px; border-collapse: collapse; }
caption { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
th, td { padding: 10px 12px; border-bottom: 1px solid #e8edef; text-align: left; vertical-align: top;
  font-variant-numeric: tabular-nums; }
tbody tr { transition: background-color 0.15s ease-in-out; }
tbody tr:hover { background: #fbfcfa; }
code { overflow-wrap: anywhere; }
.note { color: #5e6f76; margin-bottom: 0; }

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


def _render_report(manifest: dict[str, Any]) -> str:
    checks = manifest.get("checks", [])
    rows = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        rows.append(
            "<tr>"
            f'<th scope="row">{escape(str(check.get("name", "")))}</th>'
            f"<td>{escape(str(check.get('category', '')))}</td>"
            f"<td>{escape('go' if check.get('ok') else 'failed')}</td>"
            f"<td>{escape(str(check.get('detail', '')))}</td>"
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
        ("Project Version", manifest.get("project", {}).get("version", "")),
        ("Source Commit", manifest.get("source_commit", "")),
        ("Checks", len(checks) if isinstance(checks, list) else ""),
        (
            "Failed",
            len(manifest.get("failed_checks", []))
            if isinstance(manifest.get("failed_checks"), list)
            else "",
        ),
    ]
    card_markup = [
        '<article class="metric-card">'
        f"<span>{escape(label)}</span><strong>{escape(str(value))}</strong></article>"
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
            "<title>fast-mlsirm Procurement Due Diligence</title>",
            "<style>",
            _report_css(),
            "</style>",
            "</head>",
            "<body><main>",
            '<section class="hero"><p>fast-mlsirm procurement review</p><h1>Procurement Due Diligence</h1>',
            f"<span>Generated: {escape(str(manifest.get('generated_at', '')))}</span></section>",
            '<section class="report-section"><h2>Decision Summary</h2><div class="metrics-grid">',
            *card_markup,
            "</div></section>",
            '<section class="report-section"><h2>Due-Diligence Checks</h2>',
            '<div class="table-wrap" role="region" aria-label="Procurement due diligence check table" tabindex="0">',
            "<table><caption>Procurement due diligence check table</caption>",
            '<thead><tr><th scope="col">Check</th><th scope="col">Category</th><th scope="col">Status</th><th scope="col">Detail</th></tr></thead><tbody>',
            *rows,
            "</tbody></table></div>",
            '<p class="note">This report records procurement evidence only. It is not a valuation guarantee or regulated-use approval.</p>',
            "</section></main></body></html>",
        ]
    )


def build_procurement_due_diligence(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    dist_dir = _resolve_path(args.dist, base=repo_root).resolve()
    commercial_path = _resolve_path(
        args.commercial_release_manifest, base=repo_root
    ).resolve()
    out_dir = _resolve_path(args.out, base=repo_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    project = _project_metadata(repo_root)
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheel = (
        parse_wheel(wheels[0])
        if wheels
        else {"ok": False, "missing": ["wheel"], "metadata": {}}
    )
    sdist = (
        parse_sdist(sdists[0])
        if sdists
        else {"ok": False, "missing": ["sdist"], "metadata": {}}
    )
    commercial, checks = _commercial_checks(
        commercial_path, contract_value_krw=args.contract_value_krw
    )
    checks = [
        _check(
            "dist:wheel",
            "package",
            bool(wheels),
            "wheel artifact exists",
            count=len(wheels),
        ),
        _check(
            "dist:sdist",
            "package",
            bool(sdists),
            "source distribution artifact exists",
            count=len(sdists),
        ),
        _check(
            "wheel:metadata",
            "package",
            bool(wheel.get("ok")),
            "wheel contains METADATA, WHEEL, and RECORD",
            missing=wheel.get("missing"),
        ),
        _check(
            "sdist:metadata",
            "package",
            bool(sdist.get("ok")),
            "source distribution contains PKG-INFO",
            missing=sdist.get("missing"),
        ),
        _check(
            "package:version",
            "package",
            project.get("version")
            == wheel.get("metadata", {}).get("Version")
            == sdist.get("metadata", {}).get("Version"),
            "pyproject, wheel, and source distribution versions match",
            pyproject=project.get("version"),
            wheel=wheel.get("metadata", {}).get("Version"),
            sdist=sdist.get("metadata", {}).get("Version"),
        ),
        *checks,
        *_policy_checks(repo_root),
    ]
    github = _github_snapshot(args.repo, offline=args.offline_github)
    checks.extend(_github_checks(github))
    failed = [check for check in checks if not check["ok"]]
    manifest: dict[str, Any] = {
        "command": "build_procurement_due_diligence",
        "status": "ok" if not failed else "failed",
        "contract_value_krw": args.contract_value_krw,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": _source_commit(repo_root),
        "repo_root": str(repo_root),
        "dist": str(dist_dir),
        "project": project,
        "package": {"wheel": wheel, "sdist": sdist},
        "commercial_release": commercial,
        "github": github,
        "checks": checks,
        "failed_checks": failed,
    }
    html_path = out_dir / "procurement_due_diligence_report.html"
    manifest_path = out_dir / "procurement_due_diligence_manifest.json"
    html_path.write_text(_render_report(manifest), encoding="utf-8")
    manifest["html_report_file"] = str(html_path)
    manifest["html_report_sha256"] = _sha256(html_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build procurement due-diligence evidence for fast-mlsirm."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--dist", default="dist", help="Directory containing wheel and sdist artifacts."
    )
    parser.add_argument(
        "--commercial-release-manifest",
        default="commercial-release/commercial_release_manifest.json",
        help="Path to commercial_release_manifest.json.",
    )
    parser.add_argument(
        "--out", default="procurement-due-diligence", help="Output directory."
    )
    parser.add_argument(
        "--repo",
        default="ContextualWisdomLab/fast-mlsirm",
        help="GitHub repository name.",
    )
    parser.add_argument(
        "--contract-value-krw",
        type=int,
        default=2_000_000_000,
        help="Target contract value.",
    )
    parser.add_argument(
        "--offline-github",
        action="store_true",
        help="Record an offline GitHub snapshot instead of calling gh.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_procurement_due_diligence(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "out": str(Path(args.out).resolve()),
                "manifest": str(
                    Path(args.out).resolve() / "procurement_due_diligence_manifest.json"
                ),
                "html": manifest["html_report_file"],
                "failed_checks": len(manifest["failed_checks"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
