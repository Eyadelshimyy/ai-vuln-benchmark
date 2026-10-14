#!/usr/bin/env python3
"""Fast, LLM-free sink census for repo selection.

Before committing a repo to the expensive mine -> detect -> recycle pipeline,
run this to see whether it actually contains functions that call our target
sinks DIRECTLY (the only kind the name-based miner, and the live oracle, can
reach). A repo that hides its sinks behind backend/wrapper layers (e.g.
anyconfig's `to_method(pickle.loads)`, feedparser's `build_opener`) or only
uses SAFE variants (yaml.safe_load) will show near-zero here -- a poor target,
cheaply discovered.

Reuses mine_prompts.classify_sink so the census is defined identically to what
the miner will actually mine.

Usage:
  # already-cloned repos or installed package dirs (positional paths):
  python3 analysis/sink_census.py /path/to/repo1 /path/to/repo2
  # or clone-and-scan by URL:
  python3 analysis/sink_census.py --repo-url https://github.com/rq/rq.git
"""
import argparse
import ast
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mine_prompts import classify_sink, _is_skippable  # noqa: E402

CWES = ["CWE-89", "CWE-78", "CWE-22", "CWE-502", "CWE-611", "CWE-918"]


def census_dir(root: Path) -> dict:
    per_cwe = Counter()
    funcs_with_sink = 0
    total_funcs = 0
    examples: dict[str, list] = {c: [] for c in CWES}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith(".py"):
                continue
            p = Path(dp) / f
            if _is_skippable(p, root):
                continue
            try:
                tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:  # noqa: BLE001
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_funcs += 1
                    tags = classify_sink(node)
                    if tags:
                        funcs_with_sink += 1
                    for t in tags:
                        per_cwe[t] += 1
                        if len(examples[t]) < 4:
                            examples[t].append(f"{p.relative_to(root)}::{node.name}")
    return {"total_funcs": total_funcs, "funcs_with_sink": funcs_with_sink,
            "per_cwe": dict(per_cwe), "examples": examples}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="repo / package dirs to scan")
    ap.add_argument("--repo-url", action="append", default=[], help="git URL(s) to shallow-clone and scan")
    ap.add_argument("--examples", action="store_true", help="print example function names per CWE")
    args = ap.parse_args()

    targets: list[tuple[str, Path]] = [(p.name, p) for p in args.paths]
    clones: list[Path] = []
    for url in args.repo_url:
        d = Path(tempfile.mkdtemp(prefix="census_"))
        print(f"cloning {url} ...", file=sys.stderr)
        subprocess.run(["git", "clone", "--depth", "1", url, str(d)],
                       check=True, capture_output=True)
        clones.append(d)
        targets.append((url.rstrip("/").split("/")[-1].replace(".git", ""), d))

    if not targets:
        ap.error("give at least one path or --repo-url")

    hdr = f"{'repo':22} {'funcs':>6} {'wSink':>6} " + " ".join(f"{c.split('-')[1]:>5}" for c in CWES)
    print(hdr)
    print("-" * len(hdr))
    for name, path in targets:
        r = census_dir(path)
        cells = " ".join(f"{r['per_cwe'].get(c, 0):>5}" for c in CWES)
        print(f"{name[:22]:22} {r['total_funcs']:>6} {r['funcs_with_sink']:>6} {cells}")
        if args.examples:
            for c in CWES:
                if r["examples"][c]:
                    print(f"    {c}: {', '.join(r['examples'][c])}")

    print("\nColumns are DIRECT-call sink hits per CWE. A good target for an "
          "under-covered class shows several hits in that column; near-zero means "
          "the repo hides or sanitises that sink -- pick a different repo.", file=sys.stderr)
    for d in clones:
        print(f"(clone left at {d})", file=sys.stderr)


if __name__ == "__main__":
    main()
