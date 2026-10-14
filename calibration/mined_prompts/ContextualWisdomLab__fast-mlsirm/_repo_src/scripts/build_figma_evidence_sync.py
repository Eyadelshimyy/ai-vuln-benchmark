#!/usr/bin/env python
"""Build Figma evidence sync checks for a fast-mlsirm buyer packet."""

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


REQUIRED_FRAME_IDS = [
    "01-package-evidence",
    "02-synthetic-demo-run",
    "03-fit-diagnostics",
    "04-dimensionality-review",
    "05-report-export",
    "06-procurement-packet",
    "07-irt-stability-review",
    "08-fixed-item-calibration",
    "09-afipc-product-design-spec",
    "10-afipc-wireframe-user-stories",
]

REQUIRED_TOKENS = [
    "buyer packet",
    "release evidence index",
    "procurement due diligence",
    "pr queue governance",
    "fixed item calibration",
    "information architecture",
    "wireframe",
    "user stories",
]


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one Figma evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one descriptor-safe bounded JSON object from disk."""
    return read_json_object(path)


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    """Resolve a packet or output path relative to the repository root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _source_commit(repo_root: Path) -> str:
    """Return the canonical full Git object identity for HEAD or fail closed."""
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
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError("source commit lookup failed") from exc
    source_commit = completed.stdout.strip()
    if len(source_commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise RuntimeError("source commit lookup returned invalid object id")
    return source_commit


def _check(
    name: str, category: str, ok: bool, detail: str, **metadata: Any
) -> dict[str, Any]:
    """Build one named Figma evidence check with optional diagnostic metadata."""
    payload: dict[str, Any] = {
        "name": name,
        "category": category,
        "ok": ok,
        "detail": detail,
    }
    payload.update(metadata)
    return payload


def _normalize(value: str) -> str:
    """Normalize human-readable evidence text for token matching."""
    return " ".join(value.lower().replace("-", " ").replace("_", " ").split())


def _coverage(text: str, tokens: list[str]) -> dict[str, Any]:
    """Return required evidence tokens missing from normalized packet text."""
    normalized = _normalize(text)
    missing = [token for token in tokens if _normalize(token) not in normalized]
    return {"required": tokens, "missing": missing}


def _frame_coverage(packet: dict[str, Any]) -> dict[str, Any]:
    """Return present and missing buyer-review frame identifiers."""
    frames = packet.get("frames", [])
    ids = {
        frame.get("id")
        for frame in frames
        if isinstance(frame, dict) and isinstance(frame.get("id"), str)
    }
    return {
        "required": REQUIRED_FRAME_IDS,
        "present": sorted(ids),
        "missing": sorted(set(REQUIRED_FRAME_IDS) - ids),
    }


def _snapshot_text(snapshot: dict[str, Any]) -> str:
    """Flatten meaningful text fields from an exported Figma metadata snapshot."""
    parts: list[str] = []

    def walk(value: Any) -> None:
        """Visit nested snapshot values and collect meaningful text fields."""
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"characters", "name", "text"} and isinstance(child, str):
                    parts.append(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(snapshot)
    return "\n".join(parts)


def _content_security_policy() -> str:
    """Return the restrictive policy used by the self-contained HTML report."""
    return "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def _report_css() -> str:
    """Return the small inline stylesheet used by the Figma evidence report."""
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

/* Non-zero durations keep transitionend/animationend events firing. */
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
    """Render Figma evidence checks as a portable accessible HTML report."""
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
            "Code Connect",
            "disabled"
            if manifest.get("code_connect") is False
            else manifest.get("code_connect", ""),
        ),
        ("Frames", len(manifest.get("frame_coverage", {}).get("present", []))),
        (
            "Missing Tokens",
            len(manifest.get("required_token_coverage", {}).get("missing", [])),
        ),
        (
            "Metadata Snapshot",
            "checked"
            if manifest.get("metadata_snapshot", {}).get("checked")
            else "not required",
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
            "<title>fast-mlsirm Figma Evidence Sync</title>",
            "<style>",
            _report_css(),
            "</style>",
            "</head>",
            "<body><main>",
            '<section class="hero"><p>fast-mlsirm design evidence</p><h1>Figma Evidence Sync</h1>',
            f"<span>Generated: {escape(str(manifest.get('generated_at', '')))}</span></section>",
            '<section class="report-section"><h2>Decision Summary</h2><div class="metrics-grid">',
            *card_markup,
            "</div></section>",
            '<section class="report-section"><h2>Sync Checks</h2>',
            '<div class="table-wrap" role="region" aria-label="Figma evidence sync table" tabindex="0">',
            "<table><caption>Figma evidence sync table</caption>",
            '<thead><tr><th scope="col">Check</th><th scope="col">Category</th><th scope="col">Status</th><th scope="col">Detail</th></tr></thead><tbody>',
            *rows,
            "</tbody></table></div>",
            '<p class="note">This report verifies repo-local Figma packet evidence. Live Figma metadata is optional and supplied as an exported snapshot.</p>',
            "</section></main></body></html>",
        ]
    )


def build_figma_evidence_sync(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the Figma packet and write its manifest and HTML report."""
    repo_root = Path(args.repo_root).resolve()
    packet_path = _resolve_path(args.packet, base=repo_root).resolve()
    out_dir = _resolve_path(args.out, base=repo_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    packet = _read_json(packet_path)
    packet_text = json.dumps(packet, ensure_ascii=False)
    frame_coverage = _frame_coverage(packet)
    token_coverage = _coverage(packet_text, REQUIRED_TOKENS)
    code_connect = packet.get("code_connect")
    figma_url = str(packet.get("figma_artifact_url") or args.figma_url)
    url_ok = figma_url.startswith("https://www.figma.com/design/")
    snapshot_info: dict[str, Any] = {
        "checked": False,
        "path": None,
        "token_coverage": {"missing": []},
    }
    metadata_snapshot = getattr(args, "metadata_snapshot", None)
    metadata_check_ok = True
    if metadata_snapshot:
        snapshot_path = _resolve_path(metadata_snapshot, base=repo_root).resolve()
        snapshot = _read_json(snapshot_path)
        snapshot_coverage = _coverage(_snapshot_text(snapshot), REQUIRED_TOKENS)
        snapshot_info = {
            "checked": True,
            "path": str(snapshot_path),
            "token_coverage": snapshot_coverage,
        }
        metadata_check_ok = snapshot_coverage["missing"] == []

    checks = [
        _check(
            "figma:packet",
            "figma_packet",
            packet_path.exists(),
            "Figma design packet JSON exists",
            path=str(packet_path),
        ),
        _check(
            "figma:code_connect_disabled",
            "figma_policy",
            code_connect is False,
            "Figma Code Connect remains disabled",
            actual=code_connect,
        ),
        _check(
            "figma:artifact_url",
            "figma_packet",
            url_ok,
            "Figma artifact URL points to a design file",
            actual=figma_url,
        ),
        _check(
            "figma:frame_coverage",
            "figma_frames",
            frame_coverage["missing"] == [],
            "Figma packet includes all buyer-review frames",
            missing=frame_coverage["missing"],
        ),
        _check(
            "figma:required_tokens",
            "figma_tokens",
            token_coverage["missing"] == [],
            "Procurement frame artifact text includes required evidence tokens",
            missing=token_coverage["missing"],
        ),
        _check(
            "figma:metadata_tokens",
            "figma_metadata",
            metadata_check_ok,
            "Optional live Figma metadata snapshot includes required evidence tokens when provided",
            checked=snapshot_info["checked"],
            missing=snapshot_info["token_coverage"]["missing"],
        ),
    ]
    failed = [check for check in checks if not check["ok"]]
    manifest: dict[str, Any] = {
        "command": "build_figma_evidence_sync",
        "status": "ok" if not failed else "failed",
        "contract_value_krw": args.contract_value_krw,
        "generated_at": getattr(args, "generated_at", None)
        or datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": _source_commit(repo_root),
        "repo_root": str(repo_root),
        "packet": str(packet_path),
        "figma_url": figma_url,
        "code_connect": code_connect,
        "frame_coverage": frame_coverage,
        "required_token_coverage": token_coverage,
        "metadata_snapshot": snapshot_info,
        "checks": checks,
        "failed_checks": failed,
    }
    html_path = out_dir / "figma_evidence_sync_report.html"
    manifest_path = out_dir / "figma_evidence_sync_manifest.json"
    html_path.write_text(_render_report(manifest), encoding="utf-8")
    manifest["html_report_file"] = str(html_path)
    manifest["html_report_sha256"] = _sha256(html_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for Figma evidence synchronization."""
    parser = argparse.ArgumentParser(
        description="Build Figma evidence sync checks for fast-mlsirm."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--packet",
        default="examples/enterprise_demo/figma_design_packet.json",
        help="Path to figma_design_packet.json.",
    )
    parser.add_argument(
        "--metadata-snapshot",
        help="Optional exported live Figma metadata JSON snapshot.",
    )
    parser.add_argument(
        "--out", default="figma-evidence-sync", help="Output directory."
    )
    parser.add_argument(
        "--contract-value-krw",
        type=int,
        default=2_000_000_000,
        help="Target contract value.",
    )
    parser.add_argument(
        "--figma-url",
        default="https://www.figma.com/design/qD34PfMH8Kr41tFdqLCkem",
        help="Fallback Figma design URL when the packet does not include one.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run Figma evidence synchronization and print a machine-readable summary."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_figma_evidence_sync(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    repo_root = Path(args.repo_root).resolve()
    out_dir = _resolve_path(args.out, base=repo_root).resolve()
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "out": str(out_dir),
                "manifest": str(out_dir / "figma_evidence_sync_manifest.json"),
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
