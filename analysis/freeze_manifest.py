#!/usr/bin/env python3
"""Freeze a REPRODUCIBILITY MANIFEST: every headline count tied to the exact
artifact, config, and verdict definition that produced it.

Collects the project's git commit, hashes of each result artifact, the census
configuration and verdict tallies, the baseline configs, and the funnel counts
WITH their definitions -- so any number in the thesis can be traced to a frozen
dataset version. Writes MANIFEST.json and a human-readable MANIFEST.md.

  python3 analysis/freeze_manifest.py --census corpus_census.json
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS = ["corpus_census.json", "gold_set.json", "reachable_worklist.json",
             "semgrep_fullrepo.json", "semgrep_default.json", "semgrep_owasp.json",
             "semgrep_union.json", "repo_archetypes.json", "provenance_records.json",
             "finalize.json", "consortium_triggered.json", "consortium_confirmed.json"]

VERDICT_DEFS = {
    "executable": "the function ran under the harness without an import/setup error",
    "triggered (Tier-1)": "static taint reached a security-relevant sink",
    "confirmed (Tier-2)": "the deterministic exploit bank made the sink fire",
    "reachable": "a real entry point (http/cli/api/main) reaches the sink in the static call graph",
    "REACHABLE_FROM_ENTRYPOINT": "reachable AND confirmed; a disclosure lead pending trust-boundary review",
    "real_net_confirmed": "driven to a live exploit by hand against operator-controlled infra",
    "COULD_NOT_EXECUTE": "inconclusive -- excluded from safe/vuln labels, never counted as safe",
}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", type=Path, default=Path("corpus_census.json"))
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--out", type=Path, default=Path("MANIFEST.json"))
    args = ap.parse_args()

    man = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_commit": _git("rev-parse", "HEAD"),
           "git_dirty": bool(_git("status", "--porcelain")),
           "verdict_definitions": VERDICT_DEFS}

    # artifact hashes
    arts = {}
    for name in ARTIFACTS:
        p = Path(name)
        if p.exists():
            arts[name] = {"sha256_16": _sha(p), "bytes": p.stat().st_size}
    man["artifacts"] = arts

    # census config + verdict tallies + funnel
    if args.census.exists():
        cen = json.loads(args.census.read_text())
        rows = cen["rows"] if isinstance(cen, dict) else cen
        vt = Counter(r.get("verdict") for r in rows)
        cwe_conf = Counter()
        for r in rows:
            if r.get("confirmed"):
                for c in (r.get("cwes") or []):
                    cwe_conf[c] += 1
        man["census"] = {
            "file": str(args.census), "tier2": cen.get("tier2") if isinstance(cen, dict) else None,
            "n_candidates": cen.get("n_candidates") if isinstance(cen, dict) else len(rows),
            "verdict_tally": dict(vt),
            "funnel": {
                "candidates": cen.get("n_candidates") if isinstance(cen, dict) else len(rows),
                "executable": sum(1 for r in rows if r.get("executable")),
                "triggered": sum(1 for r in rows if r.get("triggered")),
                "confirmed": sum(1 for r in rows if r.get("confirmed")),
            },
            "confirmed_by_cwe": dict(cwe_conf),
        }

    # mined corpus inventory
    if args.root.exists():
        repos = {}
        for d in sorted(p for p in args.root.iterdir() if p.is_dir()):
            n = len([p for p in d.glob("*.json") if "results" not in p.name
                     and "census" not in p.name and p.name != "SUMMARY.md"])
            repos[d.name] = n
        man["mined_corpus"] = {"n_repos": len(repos), "candidates_per_repo": repos}

    # baseline configs (read from the semgrep artifacts themselves)
    baselines = {}
    for name in ("semgrep_fullrepo.json", "semgrep_default.json", "semgrep_owasp.json", "semgrep_union.json"):
        p = Path(name)
        if p.exists():
            d = json.loads(p.read_text())
            baselines[name] = {"config": d.get("config") or d.get("configs"),
                               "recall": d.get("recall"), "precision": d.get("precision")}
    if baselines:
        man["baselines"] = baselines

    args.out.write_text(json.dumps(man, indent=2))

    # human-readable sidecar
    md = ["# VULCAN reproducibility manifest", "",
          f"- generated: {man['generated_utc']}",
          f"- git commit: `{man['git_commit']}`" + ("  **(working tree dirty)**" if man["git_dirty"] else ""),
          ""]
    if "census" in man:
        f = man["census"]["funnel"]
        md += ["## Funnel (this corpus version)", "",
               f"- candidates mined: {f['candidates']}",
               f"- executable: {f['executable']}",
               f"- triggered (Tier-1): {f['triggered']}",
               f"- confirmed (Tier-2): {f['confirmed']}",
               f"- census tier2 flag: {man['census']['tier2']}",
               f"- verdict tally: {man['census']['verdict_tally']}", ""]
    if "mined_corpus" in man:
        md += [f"## Corpus: {man['mined_corpus']['n_repos']} repos", ""]
    if arts:
        md += ["## Artifact hashes (sha256, first 16)", ""]
        md += [f"- `{k}`  {v['sha256_16']}" for k, v in arts.items()]
        md += [""]
    md += ["## Verdict definitions", ""]
    md += [f"- **{k}**: {v}" for k, v in VERDICT_DEFS.items()]
    Path("MANIFEST.md").write_text("\n".join(md))

    print("=" * 60)
    print("  REPRODUCIBILITY MANIFEST")
    print("=" * 60)
    print(f"  git commit ....... {man['git_commit'][:12]}{'  (DIRTY)' if man['git_dirty'] else ''}")
    if "census" in man:
        print(f"  funnel ........... {man['census']['funnel']}")
    print(f"  artifacts hashed . {len(arts)}")
    print(f"\n  Saved {args.out} + MANIFEST.md")


if __name__ == "__main__":
    main()
