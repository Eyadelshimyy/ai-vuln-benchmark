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
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # each harness prints one JSON object per file, concatenated -- parse via loads-many
    text = proc.stdout.strip()
    decoder = json.JSONDecoder()
    objs, idx = [], 0
    while idx < len(text):
        text_from = text[idx:].lstrip()
        if not text_from:
            break
        skipped = len(text) - len(text_from) - (len(text) - len(text[idx:]))
        idx = len(text) - len(text_from)
        obj, end = decoder.raw_decode(text, idx)
        objs.append(obj)
        idx = end
    return objs


def main():
    print(f"{'file':55} {'cwe':8} {'detected':10} {'confirmed':10}")
    print("-" * 95)

    ground_truth_correct = 0
    ground_truth_total = 0

    for cwe, sample_dir in CWE_DIR.items():
        detect_proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "detection" / "run_detection.py"),
                "--dir",
                str(sample_dir),
            ],
            capture_output=True,
            text=True,
        )
        try:
            detections = json.loads(detect_proc.stdout)
        except json.JSONDecodeError:
            detections = []
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
