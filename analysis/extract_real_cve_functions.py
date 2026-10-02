"""
Pulls the actual pre-fix (vulnerable) source out of the top-ranked real CVE
candidates and saves each as a standalone, readable .py file -- so you can
look at real vulnerable code without digging through raw JSON, and so
taint_scan.py has something directly scannable.

USAGE:
    python3 analysis/extract_real_cve_functions.py            # top 10 by score
    python3 analysis/extract_real_cve_functions.py --top 20

OUTPUT:
    calibration/real_cve_extracted/<cwe>/<ghsa_id>__<filename>
    Each file starts with a comment block: CVE id, package, advisory URL,
    and the commit URL -- so the provenance travels with the code.

NOTE ON WHAT THIS DOES AND DOESN'T PROVE:
    Running taint_scan.py against these files tells you whether your SAST
    detector correctly flags a known-real vulnerable pattern -- that's a
    legitimate, real accuracy check. It does NOT by itself run the DAST
    exploit-confirmation harnesses, because those need the actual package
    installed and wired up the way its real advisory's PoC does (e.g.
    WsgiDAV's advisory includes a working curl-based PoC you could adapt
    directly). Treat SAST-against-real-CVEs as the first, fast validation
    pass, and pick 1-2 of these (WsgiDAV is the best candidate, since its
    advisory already has a full working PoC) to build a real DAST
    confirmation around as a deeper, more convincing validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
REAL_CVE_DIR = ROOT / "calibration" / "real_cve"
OUT_DIR = ROOT / "calibration" / "real_cve_extracted"

# Mirrors the scoring in triage_real_cves.py so this stays consistent with it.
from triage_real_cves import score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rows = []
    for cwe_dir in sorted(REAL_CVE_DIR.iterdir()):
        if not cwe_dir.is_dir():
            continue
        for path in sorted(cwe_dir.glob("*.json")):
            record = json.loads(path.read_text())
            s, reason = score(record)
            rows.append((s, reason, record))

    rows.sort(key=lambda r: r[0])

    written = 0
    for s, reason, record in rows:
        if written >= args.top:
            break
        commits = record.get("commits", [])
        if not commits:
            continue
        commit = commits[0]
        py_files = [f for f in commit.get("files", []) if f.get("pre_fix_source")]
        if not py_files:
            continue

        cwe = record["cwe"]
        out_subdir = OUT_DIR / cwe
        out_subdir.mkdir(parents=True, exist_ok=True)

        for f in py_files:
            safe_name = f["path"].replace("/", "__")
            out_path = out_subdir / f"{record['ghsa_id']}__{safe_name}"
            header = (
                f'"""\n'
                f"REAL CVE-confirmed vulnerable code -- extracted, not hand-written.\n\n"
                f"CVE: {record.get('cve_id')}\n"
                f"GHSA: {record.get('ghsa_id')}\n"
                f"Package(s): {', '.join(record.get('packages', []))}\n"
                f"CWE: {cwe}\n"
                f"Summary: {(record.get('summary') or '')[:300]}\n"
                f"Advisory: {record.get('advisory_url')}\n"
                f"Fix commit (this is the code BEFORE the fix): {commit.get('url')}\n"
                f'Original path in repo: {f["path"]}\n'
                f'"""\n\n'
            )
            out_path.write_text(header + f["pre_fix_source"])

        written += 1
        print(f"[{s:>4}] extracted {record['cve_id'] or record['ghsa_id']} -> {out_subdir.relative_to(ROOT)}/")

    print(f"\nExtracted {written} candidates' source into {OUT_DIR.relative_to(ROOT)}/<cwe>/")
    print("Next: skim these files, confirm the vulnerable function is actually in there")
    print("(sometimes the fix touches helper/test files, not the vulnerable function itself),")
    print("then run your scanner against them:")
    print(f"    python3 analysis/taint_scan.py --dir {OUT_DIR.relative_to(ROOT)}/CWE-89 --out /tmp/real_cve_sqli_findings.json")


if __name__ == "__main__":
    main()
