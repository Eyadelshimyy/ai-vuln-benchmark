#!/usr/bin/env python3
"""Aggregate the container-audit scans into container-hardening.json.

Reads the full Trivy image scan (TRIVY_JSON), the Dockerfile config scan
(TRIVY_CONFIG), the Docker CIS compliance report (TRIVY_CIS), and the
Lynis report under LYNIS_OUT; writes REPORT_JSON and echoes it.

Fails closed when the CIS report or the combined misconfig scans produce
zero results — an empty payload is a scanner anomaly, not coverage.
Extracted from container-audit.yml so tests can replay it against
fixtures (tests/test_container_audit.py).
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from collections.abc import Iterator
from typing import Any, cast


def _summaries(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield node
    for child in _list_of_dicts(node, "Results"):
        yield from _summaries(child)


def _list_of_dicts(node: dict[str, Any], key: str) -> list[dict[str, Any]]:
    items = node.get(key)
    if not isinstance(items, list):
        return []
    return [cast(dict[str, Any], item) for item in cast(list[Any], items) if isinstance(item, dict)]


def _load(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            payload: Any = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return cast(dict[str, Any], payload)


def main() -> int:
    trivy = _load(os.environ["TRIVY_JSON"])
    vulns: Counter[str] = Counter()
    secrets: Counter[str] = Counter()
    licenses = 0
    mis_pass = mis_total = 0
    fixable_high = 0
    top_fixable: list[dict[str, str]] = []
    for result in _list_of_dicts(trivy, "Results"):
        for v in _list_of_dicts(result, "Vulnerabilities"):
            sev = str(v.get("Severity") or "UNKNOWN").lower()
            fixed = str(v.get("FixedVersion") or "")
            vulns[sev] += 1
            if fixed:
                vulns[f"{sev}_fixed"] += 1
                if sev in ("critical", "high"):
                    fixable_high += 1
                    top_fixable.append(
                        {
                            "id": str(v.get("VulnerabilityID") or "?"),
                            "pkg": str(v.get("PkgName") or "?"),
                            "severity": str(v.get("Severity") or "UNKNOWN"),
                            "fixed": fixed,
                        }
                    )
        for s in _list_of_dicts(result, "Secrets"):
            secrets[str(s.get("Severity") or "UNKNOWN").lower()] += 1
        licenses += len(result.get("Licenses") or [])
    # `trivy image --scanners misconfig` emits no Misconfigurations for
    # this image class; the Dockerfile config scan (TRIVY_CONFIG) supplies
    # the real misconfig coverage. Trivy 0.75 omits passing checks from
    # Misconfigurations, so MisconfSummary is the authoritative count;
    # the entry walk is kept for scanners that emit PASS/FAIL rows.
    for key in ("TRIVY_JSON", "TRIVY_CONFIG"):
        config = _load(os.environ[key]) if os.environ.get(key) else {}
        for result in _list_of_dicts(config, "Results"):
            summary_node = result.get("MisconfSummary")
            if isinstance(summary_node, dict):
                summary = cast(dict[str, Any], summary_node)
                mis_pass += int(summary.get("Successes") or 0)
                mis_total += int(summary.get("Successes") or 0) + int(summary.get("Failures") or 0)
            else:
                for m in _list_of_dicts(result, "Misconfigurations"):
                    mis_total += 1
                    if m.get("Status") == "PASS":
                        mis_pass += 1
    if mis_total == 0:
        # dead telemetry must fail, not masquerade as coverage
        sys.exit("Misconfig scans produced no results")
    cis: dict[str, Any] = {"passed": 0, "failed": 0, "failures": []}
    cis_data = _load(os.environ["TRIVY_CIS"])
    for result in _summaries(cis_data):
        summary_node = result.get("MisconfSummary")
        if not isinstance(summary_node, dict):
            continue
        summary = cast(dict[str, Any], summary_node)
        cis["passed"] += int(summary.get("Successes") or 0)
        cis["failed"] += int(summary.get("Failures") or 0)
    if cis["passed"] + cis["failed"] == 0:
        # dead telemetry must fail, not masquerade as coverage
        sys.exit("Docker CIS scan produced no results")
    for result in _summaries(cis_data):
        for m in _list_of_dicts(result, "Misconfigurations"):
            if m.get("Status") == "FAIL":
                cast(list[Any], cis["failures"]).append(
                    {
                        "id": str(m.get("ID") or m.get("AVDID") or "?"),
                        "severity": str(m.get("Severity") or "UNKNOWN"),
                        "title": str(m.get("Title") or ""),
                    }
                )
    lynis: dict[str, Any] = {"hardening_index": "unknown", "warnings": 0}
    try:
        with open(
            os.environ["LYNIS_OUT"] + "/lynis-report.dat",
            encoding="utf-8",
            errors="replace",
        ) as fh:
            for line in fh:
                if line.startswith("hardening_index="):
                    lynis["hardening_index"] = line.split("=", 1)[1].strip()
                elif line.startswith("warning[]"):
                    lynis["warnings"] += 1
    except OSError:
        pass
    image, _, digest = os.environ["PINNED_IMAGE"].partition("@")
    top_fixable.sort(key=lambda v: (v["severity"] != "CRITICAL", v["id"]))
    report: dict[str, Any] = {
        "image": image,
        "digest": digest,
        "trivy": {
            "critical": vulns["critical"],
            "high": vulns["high"],
            "medium": vulns["medium"],
            "low": vulns["low"],
            "fixable_high_or_critical": fixable_high,
            "misconfig_pass": mis_pass,
            "misconfig_total": mis_total,
            "secrets": sum(secrets.values()),
            "license_findings": licenses,
            "top_fixable": top_fixable[:15],
        },
        "cis_docker": cis,
        "lynis": lynis,
        "gate": {"fixable_high_or_critical": fixable_high},
    }
    with open(os.environ["REPORT_JSON"], "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
