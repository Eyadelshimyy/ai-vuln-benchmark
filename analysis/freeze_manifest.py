#!/usr/bin/env python3
"""Freeze a REPRODUCIBILITY MANIFEST: every headline count tied to the exact
artifact, config, and verdict definition that produced it.

Collects the project's git commit, hashes of each result artifact, the census
configuration and verdict tallies, the baseline configs, and the funnel counts
WITH their definitions -- so any number in the thesis can be traced to a frozen
dataset version. Writes MANIFEST.json and a human-readable MANIFEST.md.

  python3 analysis/freeze_manifest.py --census corpus_census_reproduced.json

The --census file is always hashed (whatever its name), every consortium_*.json
RQ2 artifact present is hashed, and the Markdown sidecar is written next to
--out (so a scratch run never overwrites the committed MANIFEST.md).
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS = ["corpus_census.json", "gold_set.json", "reachable_worklist.json",
             "semgrep_fullrepo.json", "semgrep_default.json", "semgrep_owasp.json",
             "semgrep_union.json", "repo_archetypes.json", "provenance_records.json",
             "finalize.json"]
# RQ2 detector-evaluation outputs; their names carry a run suffix (_P5, _3model, ...)
RQ2_GLOB = "consortium_*.json"
# Hand-curated MANIFEST.json sections this script does not compute; carried over
# from the existing --out file instead of being silently dropped on regeneration.
PRESERVED_KEYS = ("corpus_release",)

VERDICT_DEFS = {
    "executable": "the function ran under the harness without an import/setup error",
    "triggered (Tier-1)": "at execution, the injected marker value reached a hooked security-relevant sink",
    "confirmed (Tier-2)": "a deterministic exploit-bank payload produced the per-CWE observable failure "
                          "(see PREDICATES.md), not merely sink execution",
    "reachable": "a real entry point (http/cli/api/main) reaches the sink in the static call graph",
    "REACHABLE_FROM_ENTRYPOINT": "reachable AND confirmed; a disclosure lead pending trust-boundary review",
    "real_net_confirmed": "driven to a live exploit by hand against operator-controlled infra",
    "COULD_NOT_EXECUTE": "inconclusive -- excluded from safe/vuln labels, never counted as safe",
}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _is_substudy(name: str) -> bool:
    # Same exclusion as analysis/corpus_census.py::_candidate_files (kept in sync by hand).
    return name.startswith("ai-ssti-") or name == "llama-cpp-ssti"


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
    names = [str(args.census)] + [n for n in ARTIFACTS if n != str(args.census)]
    names += sorted(str(p) for p in Path(".").glob(RQ2_GLOB) if str(p) not in names)
    for name in names:
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
                cwe_conf[r.get("exec_cwe")] += 1   # by the class that FIRED (exec_cwe), not tags
        man["census"] = {
            "file": str(args.census),
            "sha256_16": _sha(args.census),
            "tier2": cen.get("tier2") if isinstance(cen, dict) else None,
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
        main_repos = [r for r in repos if not _is_substudy(r)]
        man["mined_corpus"] = {
            "n_repo_dirs": len(repos),
            "n_repos_main_census": len(main_repos),
            "n_repos_main_census_with_candidates": sum(1 for r in main_repos if repos[r] > 0),
            "excluded_substudy_dirs": sorted(r for r in repos if _is_substudy(r)),
            "candidates_per_repo": repos,
        }

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

    if args.out.exists():
        try:
            prev = json.loads(args.out.read_text())
        except (OSError, json.JSONDecodeError):
            prev = {}
        for k in PRESERVED_KEYS:
            if k in prev:
                man[k] = prev[k]

    args.out.write_text(json.dumps(man, indent=2))
    md_path = args.out.with_suffix(".md")

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
        mc = man["mined_corpus"]
        md += [f"## Corpus: {mc['n_repos_main_census']} repos in the main census "
               f"({mc['n_repos_main_census_with_candidates']} with >=1 candidate); "
               f"{len(mc['excluded_substudy_dirs'])} SSTI sub-study dirs excluded", ""]
    if arts:
        md += ["## Artifact hashes (sha256, first 16)", ""]
        md += [f"- `{k}`  {v['sha256_16']}" for k, v in arts.items()]
        md += [""]
    md += ["## Verdict definitions", ""]
    md += [f"- **{k}**: {v}" for k, v in VERDICT_DEFS.items()]
    md_path.write_text("\n".join(md))

    print("=" * 60)
    print("  REPRODUCIBILITY MANIFEST")
    print("=" * 60)
    print(f"  git commit ....... {man['git_commit'][:12]}{'  (DIRTY)' if man['git_dirty'] else ''}")
    if "census" in man:
        print(f"  funnel ........... {man['census']['funnel']}")
    print(f"  artifacts hashed . {len(arts)}")
    print(f"\n  Saved {args.out} + {md_path}")


if __name__ == "__main__":
    main()
