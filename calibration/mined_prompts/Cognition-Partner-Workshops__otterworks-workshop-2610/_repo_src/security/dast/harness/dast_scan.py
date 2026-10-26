# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "pyyaml", "tabulate"]
# ///
"""
OtterWorks DAST harness.

Runs authenticated attack probes against a *running* OtterWorks deployment
(through the API gateway), optionally merges an OWASP ZAP report, compares the
result against an accepted-findings baseline, and gates on the difference.

This is the verification loop: a probe that reports `vulnerable` reproduces the
attack, and the same probe reporting `secure` after a fix is the proof the
finding is closed.

Usage:
    uv run security/dast/harness/dast_scan.py --target http://localhost:8080
    uv run security/dast/harness/dast_scan.py --only DAST-RATE-LIMIT-BYPASS
    uv run security/dast/harness/dast_scan.py --zap-report /tmp/zap.json
    uv run security/dast/harness/dast_scan.py --update-baseline

Environment overrides:
    OTTERWORKS_DAST_TARGET   default target base URL

Exit codes:
    0 = no findings outside the baseline
    1 = one or more findings at or above the fail-on severity
    2 = target unreachable / configuration error
    3 = nothing gating, but a selected probe could not reach a verdict
        (--only, i.e. `make dast-verify`, or --fail-on-inconclusive)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probes import REGISTRY, ScanContext, SeedError, Severity, Verdict
from probes.base import SEVERITY_ORDER, Result
from route_inventory import may_sweep_unsafely

DAST_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = DAST_DIR / "baseline.json"
DEFAULT_REPORT_DIR = DAST_DIR / "reports"
DEFAULT_TARGET = os.getenv("OTTERWORKS_DAST_TARGET", "http://localhost:8080")

VERDICT_MARK = {
    Verdict.VULNERABLE: "FAIL",
    Verdict.SECURE: "PASS",
    Verdict.INCONCLUSIVE: "SKIP",
    Verdict.SKIPPED: "SKIP",
}


# ── ZAP ingestion ─────────────────────────────────────────────────────────────

ZAP_RISK_TO_SEVERITY = {
    "3": Severity.HIGH,
    "2": Severity.MEDIUM,
    "1": Severity.LOW,
    "0": Severity.INFO,
}


DEFAULT_ZAP_RULES = DAST_DIR / "zap" / "zap-baseline.conf"


def load_zap_rule_levels(path: Path) -> dict[str, str]:
    """Plugin id -> WARN/IGNORE/FAIL from a zap-baseline.py rule file.

    ``zap-baseline.py -c`` applies these to its own console summary only; the
    JSON report still lists every alert, so the harness has to apply the tuning
    itself or a rule downgraded to WARN would gate anyway.
    """
    levels: dict[str, str] = {}
    if not path.exists():
        return levels
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) >= 2:
            levels[fields[0].strip()] = fields[1].strip().upper()
    return levels


def load_zap_findings(path: Path, rule_levels: dict[str, str] | None = None) -> list[Result]:
    """Convert an OWASP ZAP JSON report into harness results."""
    data = json.loads(path.read_text())
    rule_levels = rule_levels or {}
    results: list[Result] = []
    for site in data.get("site", []):
        for alert in site.get("alerts", []):
            plugin_id = str(alert.get("pluginid", "unknown"))
            level = rule_levels.get(plugin_id, "FAIL")
            if level == "IGNORE":
                continue
            severity = ZAP_RISK_TO_SEVERITY.get(str(alert.get("riskcode", "0")), Severity.INFO)
            instances = alert.get("instances", [])
            detail = f"{len(instances)} instance(s) reported by ZAP at {site.get('@name', '')}"
            if level == "WARN":
                # Reported, but not a second gate on something a probe already owns.
                severity = Severity.INFO
                detail = f"{detail} (WARN in {DEFAULT_ZAP_RULES.name}: reported, not gated)"
            results.append(
                Result(
                    finding_id=f"ZAP-{plugin_id}",
                    title=alert.get("alert", "ZAP alert"),
                    severity=severity,
                    owasp=alert.get("alertRef", ""),
                    cwe=f"CWE-{alert.get('cweid')}" if alert.get("cweid") else "",
                    service="api-gateway",
                    verdict=Verdict.VULNERABLE,
                    detail=detail,
                    remediation=alert.get("solution", "").strip(),
                )
            )
    return results


# ── baseline ──────────────────────────────────────────────────────────────────


def load_baseline(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or not path.read_text().strip():
        return {}
    data = json.loads(path.read_text())
    return {entry["finding_id"]: entry for entry in data.get("accepted", [])}


def write_baseline(path: Path, findings: list[Result], reason: str) -> None:
    existing = load_baseline(path)
    for finding in findings:
        existing.setdefault(
            finding.finding_id,
            {
                "finding_id": finding.finding_id,
                "title": finding.title,
                "severity": finding.severity.value,
                "reason": reason,
                "accepted_on": datetime.now(UTC).date().isoformat(),
            },
        )
    payload = {
        "_comment": (
            "Accepted DAST findings. An entry here suppresses the gate for that finding ID; "
            "every entry needs a reason and is expected to be temporary."
        ),
        "accepted": sorted(existing.values(), key=lambda e: e["finding_id"]),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ── reporting ─────────────────────────────────────────────────────────────────


def is_gating(result: Result, baseline: dict[str, Any], threshold: int) -> bool:
    """Whether this finding is what fails the run.

    The report and the exit code have to agree, so both go through here: a merged
    ZAP alert downgraded to `info` is reported but does not gate.
    """
    return (
        result.is_finding
        and result.finding_id not in baseline
        and SEVERITY_ORDER[result.severity] >= threshold
    )


def request_recorder(
    target: str, attribute_to: Callable[[], str] = lambda: ""
) -> tuple[list[dict[str, str | bool | int]], Callable[[httpx.Response], None]]:
    """A record of which routes a scan requested, and the hook that fills it in.

    Hooked on the response rather than the request: a route whose request timed
    out, or that only ever answered 5xx, was not covered by anything, and
    recording it as reached is the false assurance this gate exists to prevent.
    The status is kept so the gate decides what "delivered" means.

    A tenant can be served under a base path (`https://<nlb>/t-abc`), so the
    origin and that prefix are separated once here: requests are matched on the
    origin, and recorded against the route as the service declares it, which is
    what the inventory holds.

    Each request carries the probe that made it, because *which* probe reached a
    route is the whole difference between a route being enumerated and a route
    being attacked.
    """
    exercised: list[dict[str, str | bool | int]] = []
    seen: set[tuple[str, str, bool, str, int]] = set()
    # Parsed by httpx rather than urlparse, because the requests this is compared
    # against are httpx's: it lowercases the host and drops a default port, so a
    # target written `https://host:443` would otherwise match none of its own
    # traffic and the coverage gate would report the whole surface unattacked.
    parsed = httpx.URL(target)
    origin = (parsed.scheme, parsed.host, parsed.port)
    base_path = parsed.path.rstrip("/")

    def record(response: httpx.Response) -> None:
        request = response.request
        # Probes that deliberately leave the gateway (the direct-backend bypass) are
        # not edge coverage, so only requests to the target under test are recorded.
        if (request.url.scheme, request.url.host, request.url.port) != origin:
            return
        path = request.url.path
        if base_path:
            if not path.startswith(base_path):
                return
            path = path[len(base_path) :] or "/"
        # Reaching a route and attacking it as a logged-in caller are different
        # depths of coverage, so the report distinguishes them.
        key = (
            request.method,
            path,
            "authorization" in request.headers,
            attribute_to(),
            response.status_code,
        )
        if key not in seen:
            seen.add(key)
            exercised.append(
                {
                    "method": key[0],
                    "path": key[1],
                    "authenticated": key[2],
                    "probe": key[3],
                    "status": key[4],
                }
            )

    return exercised, record


def to_report(
    results: list[Result],
    target: str,
    baseline: dict[str, Any],
    threshold: int,
    exercised: list[dict[str, str | bool | int]] | None = None,
    partial: bool = False,
) -> dict[str, Any]:
    gating = [r for r in results if is_gating(r, baseline, threshold)]
    return {
        "target": target,
        "scanned_at": datetime.now(UTC).isoformat(),
        # A single-probe run (--only) requests a fraction of the surface, so its
        # record of what was attacked must not be read as a coverage measurement.
        "partial": partial,
        # Which routes were attacked, not just which attacks passed: dast_coverage.py
        # diffs this against the route inventory read from the services' source, so a
        # newly added endpoint cannot ride along untested behind a green gate.
        "exercised": exercised or [],
        # Whether the sweep was allowed to send write methods, recorded here rather
        # than re-read from the environment at grading time: the gate excuses a
        # withheld write route, and reading a different answer than the scan acted
        # on would excuse one that was swept and genuinely uncovered.
        "swept_unsafely": may_sweep_unsafely(),
        "summary": {
            "probes_run": len(results),
            "findings": sum(1 for r in results if r.is_finding),
            "gated": len(gating),
            "below_threshold": sum(
                1
                for r in results
                if r.is_finding and r.finding_id not in baseline and r not in gating
            ),
            "accepted": sum(1 for r in results if r.is_finding and r.finding_id in baseline),
            "inconclusive": sum(1 for r in results if r.verdict is Verdict.INCONCLUSIVE),
        },
        "gating": [r.finding_id for r in gating],
        "results": [r.to_dict() for r in results],
    }


def to_markdown(report: dict[str, Any], baseline: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# OtterWorks DAST report",
        "",
        f"- Target: `{report['target']}`",
        f"- Scanned: {report['scanned_at']}",
        f"- Probes run: {summary['probes_run']} | "
        f"gating findings: {summary['gated']} | "
        f"reported below threshold: {summary['below_threshold']} | "
        f"accepted: {summary['accepted']} | "
        f"inconclusive: {summary['inconclusive']}",
        "",
        "| Verdict | Finding | Severity | Service | OWASP |",
        "|---|---|---|---|---|",
    ]
    gating_ids = set(report["gating"])
    for result in report["results"]:
        verdict = Verdict(result["verdict"])
        state = VERDICT_MARK[verdict]
        if verdict is Verdict.VULNERABLE and result["finding_id"] in baseline:
            state = "ACCEPTED"
        elif verdict is Verdict.VULNERABLE and result["finding_id"] not in gating_ids:
            # Same distinction the console table draws: a finding below the threshold
            # is reported, but calling it FAIL would contradict the run that exited 0.
            state = "REPORTED"
        lines.append(
            f"| {state} | `{result['finding_id']}` — {result['title']} | "
            f"{result['severity']} | {result['service']} | {result['owasp']} |"
        )

    gating = [r for r in report["results"] if r["finding_id"] in gating_ids]
    if gating:
        lines += ["", "## Gating findings", ""]
        for result in gating:
            lines += [
                f"### `{result['finding_id']}` — {result['title']}",
                "",
                f"- Severity: **{result['severity']}** | {result['owasp']} | {result['cwe']}",
                f"- Service: {result['service']}",
                f"- Detail: {result['detail']}",
                f"- Remediation: {result['remediation']}",
            ]
            for evidence in result["evidence"]:
                excerpt = (evidence["response_excerpt"] or "").strip().replace("\n", " ")[:300]
                lines += [
                    "",
                    "```http",
                    evidence["request"],
                    f"-> {evidence['response_status']} {excerpt}",
                    "```",
                    f"{evidence['note']}" if evidence["note"] else "",
                ]
            lines.append("")
    return "\n".join(lines) + "\n"


def print_table(results: list[Result], baseline: dict[str, Any], threshold: int) -> None:
    rows = []
    for result in sorted(results, key=lambda r: (-SEVERITY_ORDER[r.severity], r.finding_id)):
        state = VERDICT_MARK[result.verdict]
        if result.is_finding and result.finding_id in baseline:
            state = "ACCEPTED"
        elif result.is_finding and not is_gating(result, baseline, threshold):
            state = "REPORTED"
        rows.append([state, result.finding_id, result.severity.value, result.detail[:70]])
    print(tabulate(rows, headers=["", "finding", "severity", "detail"], tablefmt="simple"))


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OtterWorks DAST probe suite.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="API gateway base URL")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="FINDING_ID",
        help="run only these probes (repeatable) — use to verify a single remediation",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="ignore accepted findings — used when verifying a specific remediation",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="record current findings as accepted instead of gating on them",
    )
    parser.add_argument("--reason", default="recorded by --update-baseline")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--zap-report", type=Path, help="merge an OWASP ZAP JSON report")
    parser.add_argument(
        "--zap-rules",
        type=Path,
        default=DEFAULT_ZAP_RULES,
        help="zap-baseline.py rule file whose WARN/IGNORE levels the gate honours",
    )
    parser.add_argument(
        "--fail-on",
        choices=[s.value for s in Severity],
        default=Severity.MEDIUM.value,
        help="minimum severity that fails the gate (default: medium)",
    )
    parser.add_argument(
        "--fail-on-inconclusive",
        action="store_true",
        help="exit 3 if any probe could not reach a verdict (implied by --only)",
    )
    parser.add_argument("--list", action="store_true", help="list registered probes and exit")
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        rows = [[p.finding_id, p.severity.value, p.service, p.owasp] for p in REGISTRY.values()]
        print(tabulate(rows, headers=["finding", "severity", "service", "owasp"]))
        return 0

    selected = list(REGISTRY.values())
    if args.only:
        unknown = [f for f in args.only if f not in REGISTRY]
        if unknown:
            print(f"unknown probe id(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        selected = [REGISTRY[f] for f in args.only]

    baseline = {} if args.no_baseline else load_baseline(args.baseline)
    target = args.target.rstrip("/")
    results: list[Result] = []
    running = [""]
    exercised, record = request_recorder(target, lambda: running[0])

    with httpx.Client(
        base_url=target,
        timeout=httpx.Timeout(args.timeout, connect=5.0),
        follow_redirects=False,
        event_hooks={"response": [record]},
    ) as client:
        ctx = ScanContext(base_url=target, client=client)
        try:
            ctx.wait_for_target()
        except SeedError as exc:
            print(f"DAST setup failed: {exc}", file=sys.stderr)
            return 2
        try:
            ctx.seed_identities()
        except SeedError as exc:
            # Unauthenticated probes (headers, CORS, telemetry, admin routes) are
            # still meaningful; probes that need identities report inconclusive.
            print(
                f"warning: could not seed identities ({exc}); "
                "authenticated probes will report inconclusive\n",
                file=sys.stderr,
            )

        print(f"Scanning {target} (run {ctx.run_id}) with {len(selected)} probe(s)\n")
        for entry in selected:
            if entry.requires_identity and not ctx.identities_ready:
                results.append(
                    entry.result(
                        Verdict.INCONCLUSIVE,
                        "the scan identities were not all seeded: this attack needs an "
                        "authenticated caller and a distinct victim, so its result would not "
                        "be evidence of a control",
                    )
                )
                continue
            running[0] = entry.finding_id
            try:
                results.append(entry.run(ctx))
            except Exception as exc:  # a broken probe must not mask the rest of the suite
                results.append(
                    entry.result(
                        Verdict.INCONCLUSIVE,
                        f"probe raised {type(exc).__name__}: {exc}",
                    )
                )

    # A scan whose identities never seeded attacked the target with a handful of
    # unauthenticated probes. That is a setup failure, not a clean bill of health: it
    # must not print PASSED, and it must not be recorded as a baseline either.
    unrun = sum(1 for entry in selected if entry.requires_identity)
    if unrun and not ctx.identities_ready:
        print(
            f"\nDAST setup failed: the scan identities were not seeded, so {unrun} "
            "authenticated probe(s) never attacked anything; this run proves nothing "
            "about them",
            file=sys.stderr,
        )
        return 2

    if args.zap_report:
        if not args.zap_report.exists():
            print(f"ZAP report not found: {args.zap_report}", file=sys.stderr)
            return 2
        results.extend(load_zap_findings(args.zap_report, load_zap_rule_levels(args.zap_rules)))

    if args.update_baseline:
        findings = [r for r in results if r.is_finding]
        write_baseline(args.baseline, findings, args.reason)
        print(f"Recorded {len(findings)} finding(s) as accepted in {args.baseline}")
        return 0

    threshold = SEVERITY_ORDER[Severity(args.fail_on)]

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = to_report(results, target, baseline, threshold, exercised, partial=bool(args.only))
    (args.report_dir / "dast-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.report_dir / "dast-report.md").write_text(to_markdown(report, baseline))

    print_table(results, baseline, threshold)
    print(f"\nReports written to {args.report_dir}/dast-report.{{json,md}}")

    gating = [r for r in results if is_gating(r, baseline, threshold)]
    if gating:
        print(f"\nDAST gate FAILED: {len(gating)} finding(s) at or above {args.fail_on}:")
        for result in gating:
            print(f"  - {result.finding_id} ({result.severity.value}): {result.detail}")
        return 1

    # A remediation is only proven by an attack that ran and failed. Verifying one
    # finding (--only) must therefore not accept "could not tell" as a pass.
    if args.only or args.fail_on_inconclusive:
        unverified = [r for r in results if r.verdict is Verdict.INCONCLUSIVE]
        if unverified:
            print(f"\nDAST gate UNVERIFIED: {len(unverified)} probe(s) reached no verdict:")
            for result in unverified:
                print(f"  - {result.finding_id}: {result.detail}")
            return 3

    print(f"\nDAST gate PASSED: no unaccepted findings at or above {args.fail_on}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
