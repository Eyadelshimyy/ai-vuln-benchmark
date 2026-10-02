"""
Computes a real, defensible accuracy number for taint_scan.py against the
extracted real-CVE ground truth -- filtering out test files (which produce
tainted-looking patterns that aren't the actual shipped vulnerability) and
matching findings against each advisory's own labeled CWE.

USAGE:
    python3 analysis/evaluate_real_cve_accuracy.py

Reads calibration/real_cve_extracted/<cwe>/*.py, runs taint_scan.py's
scan_file() on each (skipping anything that looks like a test file), and
reports, per source advisory, whether the matching CWE was found anywhere
in that advisory's extracted file(s).

IMPORTANT: a "hit" here means "the scanner found *a* tainted sink of the
right CWE somewhere in the file that was changed to fix this CVE" -- it
does NOT confirm the scanner found the *exact* line the CVE describes.
For your thesis writeup, spot-check a few hits against the advisory's own
description (printed alongside each result) before calling them fully
verified; this script gets you 90% of the way, not the final word.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from taint_scan import scan_file  # noqa: E402

EXTRACTED_DIR = ROOT / "calibration" / "real_cve_extracted"

TEST_FILE_MARKERS = ("test_", "/tests/", "__tests__", "conftest")


def is_test_file(path: Path) -> bool:
    s = str(path).lower()
    return any(marker in s for marker in TEST_FILE_MARKERS)


def main() -> None:
    if not EXTRACTED_DIR.exists():
        print(f"No data at {EXTRACTED_DIR} -- run extract_real_cve_functions.py first.")
        return

    total = 0
    skipped_test_files = 0
    misses = []

    for cwe_dir in sorted(EXTRACTED_DIR.iterdir()):
        if not cwe_dir.is_dir():
            continue
        cwe = cwe_dir.name

        # Group extracted files by advisory (ghsa_id is the part before "__").
        by_advisory: dict[str, list[Path]] = {}
        for py_file in sorted(cwe_dir.glob("*.py")):
            ghsa_id = py_file.name.split("__", 1)[0]
            by_advisory.setdefault(ghsa_id, []).append(py_file)

        print(f"\n=== {cwe} ===")
        for ghsa_id, files in by_advisory.items():
            real_files = [f for f in files if not is_test_file(f)]
            skipped_test_files += len(files) - len(real_files)
            if not real_files:
                continue  # every file for this advisory was a test file

            total += 1
            found_matching_cwe = False
            for f in real_files:
                findings = scan_file(f)
                if any(finding.cwe == cwe for finding in findings):
                    found_matching_cwe = True
                    break

            status = "HIT " if found_matching_cwe else "MISS"
            if not found_matching_cwe:
                misses.append((cwe, ghsa_id, [str(f.relative_to(ROOT)) for f in real_files]))
            print(f"  [{status}] {ghsa_id}  ({len(real_files)} non-test file(s))")

    hits = total - len(misses)
    print(f"\n{'-'*60}")
    print(f"Real-CVE detection rate (test files excluded): {hits}/{total}")
    if total:
        print(f"  = {hits / total * 100:.1f}%")
    print(f"Skipped {skipped_test_files} test-file(s) from the comparison.")

    if misses:
        print("\nMissed advisories (worth a manual look, and worth discussing as a limitation):")
        for cwe, ghsa_id, paths in misses:
            print(f"  - {cwe} {ghsa_id}: {', '.join(paths)}")


if __name__ == "__main__":
    main()
