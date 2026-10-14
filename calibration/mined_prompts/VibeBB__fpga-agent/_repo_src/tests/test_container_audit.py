"""Run the container-audit scripts against captured trivy fixtures.

scripts/container_hardening_report.py, container_hardening_markdown.py
and trivy_gate_summary.py back the weekly audit and the publish gate;
these tests replay them against fixtures shaped like real scanner output
(the CIS fixture mirrors the DS-0002/DS-0026 failures a tools image
reports) so an edit that breaks the report JSON or the issue body fails
here instead of on the weekly run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SCRIPTS = REPO_ROOT / "scripts"


def _run_script(
    script: str, args: list[str], env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    merged = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=merged,
    )


def _compute_env(tmp_path: Path, *, write_config: bool = True) -> dict[str, str]:
    (tmp_path / "lynis-out").mkdir()
    (tmp_path / "lynis-out" / "lynis-report.dat").write_text(
        "hardening_index=72\nwarning[]=test-warning\n", encoding="utf-8"
    )
    (tmp_path / "trivy-cis.json").write_bytes((FIXTURES / "trivy-cis.json").read_bytes())
    (tmp_path / "trivy-image.json").write_bytes((FIXTURES / "trivy-image.json").read_bytes())
    # The image JSON fixture carries one PASS misconfiguration; the real
    # Dockerfile config scan result is a separate file in CI.
    env = {
        "LYNIS_OUT": str(tmp_path / "lynis-out"),
        "TRIVY_JSON": str(tmp_path / "trivy-image.json"),
        "TRIVY_CIS": str(tmp_path / "trivy-cis.json"),
        "REPORT_JSON": str(tmp_path / "container-hardening.json"),
        "PINNED_IMAGE": "ghcr.io/vibebb/fpga-tools@sha256:" + "0" * 64,
    }
    if write_config:
        (tmp_path / "trivy-config.json").write_bytes((FIXTURES / "trivy-image.json").read_bytes())
        env["TRIVY_CONFIG"] = str(tmp_path / "trivy-config.json")
    return env


def test_cis_aggregation_extracts_failures(tmp_path: Path) -> None:
    env = _compute_env(tmp_path)
    result = _run_script("container_hardening_report.py", [], env, tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(Path(env["REPORT_JSON"]).read_text(encoding="utf-8"))
    assert report["cis_docker"]["passed"] == 4
    assert report["cis_docker"]["failed"] == 2
    failures = report["cis_docker"]["failures"]
    assert [f["id"] for f in failures] == ["DS-0002", "DS-0026"]
    assert failures[0]["severity"] == "HIGH"
    assert "root" in failures[0]["title"]
    trivy = report["trivy"]
    assert trivy["fixable_high_or_critical"] == 2
    assert trivy["secrets"] == 1
    assert trivy["license_findings"] == 1
    # CRITICAL first in the fixable list
    assert [v["id"] for v in trivy["top_fixable"]] == ["CVE-2026-0001", "CVE-2026-0002"]
    assert trivy["top_fixable"][0]["fixed"] == "3.0.17-1"
    assert report["gate"]["fixable_high_or_critical"] == 2


def test_cis_empty_results_still_fail_closed(tmp_path: Path) -> None:
    env = _compute_env(tmp_path)
    (tmp_path / "trivy-cis.json").write_text('{"Results": []}', encoding="utf-8")
    result = _run_script("container_hardening_report.py", [], env, tmp_path)
    assert result.returncode != 0
    assert "Docker CIS scan produced no results" in result.stderr


def test_misconfig_summary_counts_when_no_entries(tmp_path: Path) -> None:
    # A clean `trivy config` run emits only MisconfSummary (trivy 0.75
    # omits passing checks from Misconfigurations): coverage exists and
    # the report must count it instead of failing closed.
    env = _compute_env(tmp_path)
    image = json.loads((tmp_path / "trivy-image.json").read_text(encoding="utf-8"))
    for result in image.get("Results", []):
        result.pop("Misconfigurations", None)
        result.pop("MisconfSummary", None)
    (tmp_path / "trivy-image.json").write_text(json.dumps(image), encoding="utf-8")
    (tmp_path / "trivy-config.json").write_text(
        '{"Results": [{"Target": "fpga-tools.Dockerfile", "Class": "config",'
        ' "MisconfSummary": {"Successes": 25, "Failures": 0}}]}',
        encoding="utf-8",
    )
    result = _run_script("container_hardening_report.py", [], env, tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "container-hardening.json").read_text(encoding="utf-8"))
    assert report["trivy"]["misconfig_pass"] == 25
    assert report["trivy"]["misconfig_total"] == 25


def test_empty_misconfig_results_fail_closed(tmp_path: Path) -> None:
    env = _compute_env(tmp_path, write_config=False)
    # Strip every Misconfigurations and MisconfSummary key so both scans
    # yield 0/0 — dead telemetry must fail, not masquerade as coverage.
    image = json.loads((tmp_path / "trivy-image.json").read_text(encoding="utf-8"))
    for result in image.get("Results", []):
        result.pop("Misconfigurations", None)
        result.pop("MisconfSummary", None)
    (tmp_path / "trivy-image.json").write_text(json.dumps(image), encoding="utf-8")
    result = _run_script("container_hardening_report.py", [], env, tmp_path)
    assert result.returncode != 0
    assert "Misconfig scans produced no results" in result.stderr


def test_report_markdown_lists_failed_checks_and_fixable_cves(tmp_path: Path) -> None:
    env = _compute_env(tmp_path)
    assert _run_script("container_hardening_report.py", [], env, tmp_path).returncode == 0
    md = _run_script(
        "container_hardening_markdown.py",
        [],
        {
            "REPORT_JSON": env["REPORT_JSON"],
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REPOSITORY": "VibeBB/fpga-agent",
            "GITHUB_RUN_ID": "1",
        },
        tmp_path,
    )
    assert md.returncode == 0, md.stderr
    out = md.stdout
    assert "### Failed CIS Docker checks" in out
    assert "| DS-0002 | HIGH |" in out
    assert "| DS-0026 | LOW |" in out
    assert "### Top fixable vulnerabilities" in out
    assert "| CVE-2026-0001 | libssl3 | CRITICAL | 3.0.17-1 |" in out
    assert "| CVE-2026-0002 | bash | HIGH | 5.2.15-2 |" in out


def test_publish_gate_summary_renders_offending_rules(tmp_path: Path) -> None:
    sarif = tmp_path / "trivy-image.sarif"
    sarif.write_bytes((FIXTURES / "trivy-image.sarif").read_bytes())
    result = _run_script("trivy_gate_summary.py", [str(sarif)], {}, tmp_path)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "### Trivy gate: fixable CRITICAL/HIGH findings" in out
    assert "| CVE-2026-0001 | 9.8 | error |" in out
    assert "| CVE-2026-0002 | 7.5 | error |" in out


def test_publish_gate_summary_reads_trivy_json(tmp_path: Path) -> None:
    report = tmp_path / "trivy-image.json"
    report.write_bytes((FIXTURES / "trivy-image.json").read_bytes())
    result = _run_script("trivy_gate_summary.py", [str(report)], {}, tmp_path)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "### Trivy gate: fixable CRITICAL/HIGH findings" in out
    assert "| CVE-2026-0001 | CRITICAL | critical | libssl3 fixed in 3.0.17-1 |" in out
    assert "| CVE-2026-0002 | HIGH | high | bash fixed in 5.2.15-2 |" in out


def test_publish_gate_summary_json_skips_unfixable(tmp_path: Path) -> None:
    report = tmp_path / "trivy-image.json"
    report.write_text(
        '{"Results": [{"Vulnerabilities": ['
        '{"Severity": "HIGH", "FixedVersion": "", "VulnerabilityID": "CVE-1"},'
        '{"Severity": "LOW", "FixedVersion": "1.0", "VulnerabilityID": "CVE-2"}'
        "]}]}",
        encoding="utf-8",
    )
    result = _run_script("trivy_gate_summary.py", [str(report)], {}, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "CVE-1" not in result.stdout
    assert "CVE-2" not in result.stdout


def test_publish_gate_summary_empty_results(tmp_path: Path) -> None:
    sarif = tmp_path / "trivy-image.sarif"
    sarif.write_text('{"runs": [{"results": []}]}', encoding="utf-8")
    result = _run_script("trivy_gate_summary.py", [str(sarif)], {}, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Trivy gate" not in result.stdout
