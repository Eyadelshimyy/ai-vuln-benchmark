"""
Runs the full pipeline (detection -> exploit confirmation) against the
hand-built Python calibration set and prints a combined comparison table.

This is the script to run first, before spending any API budget on real
LLM-generated code: it proves the detection rules and the exploit
harnesses behave correctly against known-vulnerable/known-safe ground
truth (i.e. it calibrates the pipeline itself). Re-run it after any change
to detection/semgrep_rules/*.yaml or exploit_confirmation/harness_*.py.

Usage:
    python3 scripts/run_calibration.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
CALIB = ROOT / "calibration" / "python_handbuilt"
HARNESS_BY_CWE = {
    "CWE-89": ROOT / "exploit_confirmation" / "harness_sqli.py",
    "CWE-78": ROOT / "exploit_confirmation" / "harness_cmdi.py",
    "CWE-22": ROOT / "exploit_confirmation" / "harness_path_traversal.py",
}
CWE_DIR = {
    "CWE-89": CALIB / "sqli",
    "CWE-78": CALIB / "cmdi",
    "CWE-22": CALIB / "path_traversal",
}


def run_json(cmd: list[str]) -> list[dict]:
    # Route the harness's structured results through a temp --out file instead
    # of parsing its stdout: the *code under test* (os.system, subprocess
    # shell=True, etc.) can itself print to the harness's inherited stdout
    # (e.g. a shell's "command not found" message when `ping` fires as part
    # of a payload), which would otherwise corrupt the JSON stream. --out is
    # written once, directly, after all samples are processed, so it can't
    # be polluted by anything the samples under test print.
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "results.json"
        proc = subprocess.run(cmd + ["--out", str(out_path)], capture_output=True, text=True)
        if not out_path.exists():
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"harness produced no output file (command: {' '.join(cmd)})")
        return json.loads(out_path.read_text())


def main():
    print(f"{'file':55} {'cwe':8} {'detected':10} {'confirmed':10}")
    print("-" * 95)

    ground_truth_correct = 0
    ground_truth_total = 0

    for cwe, sample_dir in CWE_DIR.items():
        with tempfile.TemporaryDirectory() as td:
            detect_out = Path(td) / "detection.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "detection" / "run_detection.py"),
                    "--dir",
                    str(sample_dir),
                    "--out",
                    str(detect_out),
                ],
                capture_output=True,
                text=True,
            )
            detections = json.loads(detect_out.read_text()) if detect_out.exists() else []
        detected_files = {Path(d["file"]).name for d in detections}

        confirm_results = run_json([sys.executable, str(HARNESS_BY_CWE[cwe]), "--dir", str(sample_dir)])
        confirm_by_file = {Path(r["file"]).name: r["confirmed_exploitable"] for r in confirm_results}

        for sample_file in sorted(sample_dir.glob("*.py")):
            name = sample_file.name
            is_detected = name in detected_files
            is_confirmed = confirm_by_file.get(name, False)
            expected_vulnerable = name.startswith("vulnerable")

            ground_truth_total += 1
            if is_confirmed == expected_vulnerable:
                ground_truth_correct += 1

            print(f"{name:55} {cwe:8} {str(is_detected):10} {str(is_confirmed):10}")

    print("-" * 95)
    accuracy = ground_truth_correct / ground_truth_total * 100
    print(
        f"\nExploit-confirmation harness accuracy against hand-built ground truth: "
        f"{ground_truth_correct}/{ground_truth_total} ({accuracy:.1f}%)"
    )
    print(
        "(Detection column shows Semgrep's raw flag; confirmed column shows the "
        "exploit-confirmation harness's verdict. Rows where detected=True but "
        "confirmed=False are exactly the false positives detection-only pipelines "
        "in prior work cannot distinguish from real vulnerabilities.)"
    )


if __name__ == "__main__":
    main()
