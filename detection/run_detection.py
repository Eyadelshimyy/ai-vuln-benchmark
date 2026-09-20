"""
Detection stage -- static analysis with Semgrep.

Runs the project's local, offline Semgrep rules (detection/semgrep_rules/)
against a directory of code samples and emits one JSON finding per rule
match, tagged with the CWE it corresponds to.

Local rule files (not the Semgrep Registry) are used deliberately: the
Registry requires network access to semgrep.dev, which may be blocked in
CI/sandboxed environments, and local rules keep the exact detection logic
under version control and reproducible.

This stage is deliberately allowed to be imprecise (over- or
under-inclusive) -- that gap between "flagged" and "confirmed" is the
thing exploit_confirmation/ exists to measure. See README.md.

Usage:
    python3 run_detection.py --dir ../calibration/python_handbuilt/sqli
    python3 run_detection.py --dir ../generation/samples/python --out ../results/raw/detection.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RULES_DIR = Path(__file__).parent / "semgrep_rules"


def run_semgrep(target_dir: Path) -> dict:
    rule_files = sorted(RULES_DIR.glob("*.yaml"))
    if not rule_files:
        raise SystemExit(f"no rule files found in {RULES_DIR}")

    cmd = ["semgrep", "--json", "--metrics=off", "--disable-version-check"]
    for rf in rule_files:
        cmd += ["--config", str(rf)]
    cmd.append(str(target_dir))

    import os

    env = {**os.environ, "SEMGREP_SEND_METRICS": "off", "SEMGREP_ENABLE_VERSION_CHECK": "0"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    if proc.returncode not in (0, 1):  # 1 = findings present, still success
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"semgrep failed with exit code {proc.returncode}")
    return json.loads(proc.stdout)


def summarize(raw: dict) -> list[dict]:
    findings = []
    for r in raw.get("results", []):
        findings.append(
            {
                "file": r["path"],
                "rule_id": r["check_id"],
                "cwe": r.get("extra", {}).get("metadata", {}).get("cwe", "UNKNOWN"),
                "line": r["start"]["line"],
                "message": r.get("extra", {}).get("message", "").strip(),
            }
        )
    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of code samples to scan")
    ap.add_argument("--out", help="write findings JSON to this path")
    args = ap.parse_args()

    raw = run_semgrep(Path(args.dir))
    findings = summarize(raw)

    print(json.dumps(findings, indent=2))
    print(f"\n{len(findings)} finding(s) across scanned files.", file=sys.stderr)

    if args.out:
        Path(args.out).write_text(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
