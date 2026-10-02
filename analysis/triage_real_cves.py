"""
Ranks the raw candidates saved by fetch_real_cves.py so you don't have to
open all ~50 JSON files by hand to find the usable ones.

A "good" candidate for your held-out ground-truth set is one where the fix
touched exactly one Python file, with a small patch (a targeted fix, not a
refactor), and where the pre-fix source actually resolved (so you have the
real vulnerable file content, not just a diff).

USAGE:
    python3 analysis/triage_real_cves.py

OUTPUT:
    Prints a ranked table to the terminal (best candidates first) and writes
    calibration/real_cve/shortlist.md -- open that file, and for each entry
    it lists, open the matching .json file to pull out the actual vulnerable
    function into your held-out test set.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
REAL_CVE_DIR = ROOT / "calibration" / "real_cve"


def score(record: dict) -> tuple[int, str]:
    """Lower score = better/simpler candidate. Returns (score, reason)."""
    commits = record.get("commits", [])
    if not commits:
        return (999, "no resolvable commit")

    # Use the first commit (script caps at 2 per advisory anyway).
    commit = commits[0]
    files = commit.get("files", [])
    py_files_with_source = [f for f in files if f.get("pre_fix_source")]

    if not py_files_with_source:
        return (999, "pre-fix source did not resolve (renamed/moved file?)")

    if len(files) > 3:
        # Category base 600 dominates all smaller categories regardless of size.
        return (600 + len(files), f"touches {len(files)} files -- likely a broad fix")

    total_patch_len = sum(len(f.get("patch") or "") for f in files)

    if len(files) == 1 and total_patch_len < 1500:
        # Best category: base 0, fine-grained by patch size (capped so it can
        # never spill into the next category's range).
        return (min(total_patch_len // 20, 99), "single file, small patch -- likely isolatable")

    if len(files) == 1:
        return (100 + min(total_patch_len // 200, 99), "single file, larger patch -- worth a look")

    # 2-3 files: always worse than any single-file case, better than >3 files.
    return (300 + len(files) * 10 + min(total_patch_len // 500, 9), f"{len(files)} files changed -- check carefully")


def main() -> None:
    if not REAL_CVE_DIR.exists():
        print(f"No data yet at {REAL_CVE_DIR} -- run fetch_real_cves.py first.")
        return

    rows = []
    for cwe_dir in sorted(REAL_CVE_DIR.iterdir()):
        if not cwe_dir.is_dir():
            continue
        for path in sorted(cwe_dir.glob("*.json")):
            record = json.loads(path.read_text())
            s, reason = score(record)
            rows.append(
                {
                    "score": s,
                    "reason": reason,
                    "cwe": record.get("cwe"),
                    "cve_id": record.get("cve_id"),
                    "ghsa_id": record.get("ghsa_id"),
                    "packages": ", ".join(record.get("packages", [])),
                    "path": path.relative_to(ROOT),
                }
            )

    rows.sort(key=lambda r: r["score"])

    print(f"{'score':>6}  {'cwe':8} {'cve_id':18} {'package':22} reason")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['score']:>6}  {r['cwe']:8} {str(r['cve_id']):18} "
            f"{r['packages'][:22]:22} {r['reason']}"
        )

    good = [r for r in rows if r["score"] < 300]
    out_path = REAL_CVE_DIR / "shortlist.md"
    lines = [
        "# Real-CVE shortlist (auto-ranked, still needs your eyes)\n",
        f"{len(good)} of {len(rows)} candidates scored as likely single-function fixes.\n",
        "Open each `path` below, find `commits[0].files[0].pre_fix_source` for the",
        "actual vulnerable file content, and pull out just the vulnerable function",
        "into your held-out test set.\n",
    ]
    for r in good:
        lines.append(f"- **{r['cve_id'] or r['ghsa_id']}** ({r['cwe']}, {r['packages']}) -- {r['reason']}")
        lines.append(f"  `{r['path']}`")
    out_path.write_text("\n".join(lines))

    print(f"\n{len(good)}/{len(rows)} candidates look promising -- see {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
