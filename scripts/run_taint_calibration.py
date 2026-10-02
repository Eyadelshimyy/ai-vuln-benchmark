"""
Calibrates analysis/taint_scan.py (the hand-built AST taint scanner)
against the same hand-built ground truth used for the exploit-confirmation
harnesses, in calibration/python_handbuilt/.

This is the detection-stage equivalent of scripts/run_calibration.py --
run it after any change to analysis/taint_scan.py to confirm the scanner
still correctly separates vulnerable_*.py from safe_*.py before trusting
it on real (uncontrolled) generated code.

Usage:
    python3 scripts/run_taint_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from taint_scan import scan_file  # noqa: E402

CALIB = ROOT / "calibration" / "python_handbuilt"
CWE_DIR = {
    "CWE-89": CALIB / "sqli",
    "CWE-78": CALIB / "cmdi",
    "CWE-22": CALIB / "path_traversal",
}


def main() -> None:
    print(f"{'file':30} {'cwe':8} {'expected':10} {'flagged':10} {'result'}")
    print("-" * 80)

    correct = 0
    total = 0

    for cwe, sample_dir in CWE_DIR.items():
        for sample_file in sorted(sample_dir.glob("*.py")):
            expected_vulnerable = sample_file.name.startswith("vulnerable")
            findings = scan_file(sample_file)
            flagged = any(f.cwe == cwe for f in findings)

            total += 1
            ok = flagged == expected_vulnerable
            if ok:
                correct += 1

            print(
                f"{sample_file.name:30} {cwe:8} {str(expected_vulnerable):10} "
                f"{str(flagged):10} {'OK' if ok else 'MISMATCH'}"
            )

    print("-" * 80)
    accuracy = correct / total * 100
    print(f"\nTaint scanner accuracy against hand-built ground truth: {correct}/{total} ({accuracy:.1f}%)")


if __name__ == "__main__":
    main()
