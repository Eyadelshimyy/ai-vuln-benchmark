#!/usr/bin/env python3
"""Render container-hardening.json as the report-issue markdown body.

Reads REPORT_JSON and prints the markdown on stdout. Extracted from
container-audit.yml so tests can replay it against fixtures.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, cast


def _metric(table: dict[str, Any], key: str) -> Any:
    return table.get(key) or 0


def main() -> int:
    with open(os.environ["REPORT_JSON"], encoding="utf-8") as fh:
        data: Any = json.load(fh)
    r = cast(dict[str, Any], data)
    t = cast(dict[str, Any], r["trivy"])
    print("# Container hardening report\n")
    print(f"Image: `{r['image']}`\nDigest: `{r['digest']}`\n")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Trivy vulns C/H/M/L | {t['critical']}/{t['high']}/{t['medium']}/{t['low']} |")
    print(f"| Fixable HIGH+CRITICAL | {t['fixable_high_or_critical']} |")
    print(f"| Misconfig pass | {t['misconfig_pass']}/{t['misconfig_total']} |")
    print(f"| Secrets found | {_metric(t, 'secrets')} |")
    print(f"| CIS Docker pass/fail | {r['cis_docker']['passed']}/{r['cis_docker']['failed']} |")
    print(f"| Lynis hardening index (trend only) | {r['lynis']['hardening_index']} |")
    print(f"| Lynis warnings | {r['lynis']['warnings']} |")
    failures = cast(
        list[dict[str, Any]], cast(dict[str, Any], r["cis_docker"]).get("failures") or []
    )
    if failures:
        print("\n### Failed CIS Docker checks\n")
        print("| Check | Severity | Title |")
        print("|---|---|---|")
        for f in failures:
            print(f"| {f['id']} | {f['severity']} | {f['title']} |")
    fixable = cast(list[dict[str, Any]], t.get("top_fixable") or [])
    if fixable:
        print("\n### Top fixable vulnerabilities\n")
        print("| CVE | Package | Severity | Fixed in |")
        print("|---|---|---|---|")
        for v in fixable:
            print(f"| {v['id']} | {v['pkg']} | {v['severity']} | {v['fixed']} |")
    print(
        f"\nRun: {os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
        f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
