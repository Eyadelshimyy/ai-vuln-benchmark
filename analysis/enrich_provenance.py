#!/usr/bin/env python3
"""Backfill auditable PROVENANCE onto every mined finding.

Records, per finding: source_sha256 (always), and via git blame the dominant
AI-attributed commit, the tool(s), the fraction of the function's current lines
that are AI-attributed, and the exact trailer/bot evidence. Reuses
provenance.function_provenance (the same blame logic the mining used).

Blame needs blobs locally, so repos are FULL-cloned (not blobless -- a blobless
clone makes blame fetch every blob over the network and crawl). Every git call is
hard-bounded by --git-timeout so one slow repo can never hang the run; a repo
whose clone or blame exceeds the limit is reported as not-blamed, never silently
skipped. Use --no-clone for source-hash only (instant, no network).

  python3 analysis/enrich_provenance.py --root calibration/mined_prompts \
      --census corpus_census.json --out provenance_records.json --write-back
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance  # noqa: E402
from provenance import function_provenance  # noqa: E402

# ---- hard-bound EVERY git call provenance.py makes (kills the child on timeout,
# returns "" so blame degrades to 'no attribution' instead of hanging) ----
_GIT_TIMEOUT = 30


def _git_bounded(repo_dir, *args):
    try:
        return subprocess.run(["git", "-C", str(repo_dir), *args],
                              capture_output=True, text=True, timeout=_GIT_TIMEOUT).stdout
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        return ""


provenance._git = _git_bounded   # monkeypatch the shared helper


def _clone_urls(repos_json: Path) -> dict:
    out = {}
    try:
        for r in json.loads(repos_json.read_text()):
            fn = r.get("full_name") or ""
            if fn:
                out[fn] = r.get("clone_url") or f"https://github.com/{fn}.git"
    except (OSError, json.JSONDecodeError):
        pass
    return out


def _clone(url: str, dest: Path, timeout: int) -> Path | None:
    if dest.exists():
        return dest
    try:
        r = subprocess.run(["git", "clone", "--no-tags", url, str(dest)],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"    clone TIMEOUT ({timeout}s) {url} -- repo not blamed", file=sys.stderr)
        return None
    if r.returncode != 0:
        print(f"    clone failed {url}: {r.stderr.strip()[:120]}", file=sys.stderr)
        return None
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--repos", type=Path, default=None)
    ap.add_argument("--census", type=Path, default=None,
                    help="restrict to the census's candidate files (exact corpus)")
    ap.add_argument("--out", type=Path, default=Path("provenance_records.json"))
    ap.add_argument("--write-back", action="store_true")
    ap.add_argument("--no-clone", action="store_true", help="source-hash only; no blame/network")
    ap.add_argument("--clone-timeout", type=int, default=300)
    ap.add_argument("--git-timeout", type=int, default=30)
    ap.add_argument("--skip", default="", help="comma-separated owner/repo to skip cloning (e.g. big monorepos)")
    args = ap.parse_args()

    global _GIT_TIMEOUT
    _GIT_TIMEOUT = args.git_timeout
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    urls = _clone_urls(args.repos) if args.repos else {}
    cache = Path(tempfile.mkdtemp(prefix="vulcan_prov_")) if not args.no_clone else None
    clones: dict = {}

    census_files = None
    if args.census and args.census.exists():
        cen = json.loads(args.census.read_text())
        rows = cen["rows"] if isinstance(cen, dict) else cen
        census_files = {r["file"] for r in rows}

    repo_dirs = sorted(p for p in args.root.iterdir() if p.is_dir())
    records, tool_tally = [], Counter()
    strong = not_strong = 0
    for ri, repo_dir in enumerate(repo_dirs, 1):
        full = repo_dir.name.replace("__", "/", 1)
        cand_files = [p for p in repo_dir.glob("*.json")
                      if "results" not in p.name and "census" not in p.name
                      and "provenance" not in p.name and p.name != "SUMMARY.md"]
        clone = None
        if not args.no_clone and full not in skip:
            url = urls.get(full) or f"https://github.com/{full}.git"
            if full not in clones:
                print(f"[{ri}/{len(repo_dirs)}] cloning {full} ...", file=sys.stderr)
                clones[full] = _clone(url, cache / repo_dir.name, args.clone_timeout)
            clone = clones[full]

        n_here = 0
        for cj_path in cand_files:
            try:
                cj = json.loads(cj_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            rel_json = str(cj_path.relative_to(args.root))
            if census_files is not None and rel_json not in census_files:
                continue
            src = cj.get("ground_truth_source") or ""
            if not src.strip():
                continue
            rec = {"repo": full, "file": cj.get("file"), "function": cj.get("function"),
                   "line": cj.get("line"),
                   "source_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest()}
            if clone and clone.exists() and cj.get("file") and cj.get("line"):
                try:
                    pv = function_provenance(clone, cj["file"], int(cj["line"]))
                    rec.update({k: pv.get(k) for k in
                                ("ai_commit", "ai_tools", "ai_line_fraction",
                                 "ai_attributed", "evidence", "span")})
                except Exception as e:  # noqa: BLE001
                    rec["provenance_error"] = str(e)[:120]
            records.append(rec); n_here += 1
            if rec.get("ai_attributed"):
                strong += 1
                for t in (rec.get("ai_tools") or {}):
                    tool_tally[t] += 1
            else:
                not_strong += 1
            if args.write_back:
                cj["provenance"] = {k: rec[k] for k in rec if k not in ("repo", "file", "function", "line")}
                cj_path.write_text(json.dumps(cj, indent=2))
        if n_here:
            print(f"    {full}: {n_here} findings blamed", file=sys.stderr)
        # free disk immediately: keep at most one clone on disk at a time
        if clone and cache and clone.exists():
            shutil.rmtree(clone, ignore_errors=True)
            clones.pop(full, None)

    args.out.write_text(json.dumps({"records": records, "strong_ai_attributed": strong,
                                    "not_strong": not_strong, "tools": dict(tool_tally)}, indent=2))
    print("=" * 60)
    print("  PROVENANCE BACKFILL")
    print("=" * 60)
    print(f"  findings recorded ........ {len(records)}")
    print(f"  source_sha256 on all ..... yes")
    if args.no_clone:
        print("  blame/commit ............. SKIPPED (--no-clone)")
    else:
        print(f"  strong AI-attributed ..... {strong}")
        print(f"  not strongly attributed .. {not_strong}")
        print(f"  tools (blame) ............ {dict(tool_tally)}")
    print(f"\n  Saved {args.out}" + ("  (+ provenance block in mined JSONs)" if args.write_back else ""))
    if cache:
        print(f"  (temp clones in {cache} -- delete when done)")


if __name__ == "__main__":
    main()
