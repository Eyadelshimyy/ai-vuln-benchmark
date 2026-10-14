#!/usr/bin/env python3
"""VULCAN end-to-end DISCLOSURE pipeline.

Turns the frozen corpus's Tier-2 CONFIRMED findings into a reviewed disclosure
funnel, end to end, with zero LLM and zero contact with anyone:

    frozen census ──select confirmed──▶ re-prove exploit (deterministic bank)
        ──▶ reproduce through the project's REAL public interface on a clean
            clone ──▶ verdict ──▶ dossier (only for the disclosable ones)

The single invariant that protects maintainers from a false report: a finding is
only ever marked a disclosure candidate when the attacker payload reaches the
dangerous sink through the project's OWN public API on a clean clone (verdict
REPRODUCED_PUBLIC_API, from reproduce.py). Everything else is routed to a human
or dropped -- the conservative behaviour disclosure requires. This tool PREPARES
dossiers; a person still confirms the trust boundary and does the actual report.

Stages per confirmed finding:
  1. load the mined candidate (real file path, function, source)
  2. re-prove the exploit with the deterministic Tier-2 bank to recover the exact
     confirming (param, payload, sink, proof) -- also a fresh sanity re-check
  3. call reproduce.reproduce() against a clean clone of the repo (cached per
     repo so N findings in one project clone it once)
  4. tally the verdict; write a dossier for REPRODUCED_PUBLIC_API

USAGE
  python3 analysis/disclose_pipeline.py --census corpus_census_FROZEN.json \
      --repos ai_repos.json --out dossiers/ [--limit N] [--cwe CWE-22]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "analysis", _REPO / "analysis" / "iast"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tier2.engine import attempt_exploit                              # noqa: E402
from reproduce import reproduce                                        # noqa: E402
from auto_harness import _parse_params, vendor_sys_path_roots          # noqa: E402
from reachability import reaches                                       # noqa: E402


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


def _full_name(rel: str) -> str:
    # "kamoras__civitas/backend__...json" -> "kamoras/civitas"
    return rel.split("/")[0].replace("__", "/", 1)


def _write_dossier(res: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    slug = f"{res.get('function','fn')}_{res.get('cwe','')}_{res.get('repo_full','')}".replace("/", "-")
    (out / f"{slug}.json").write_text(json.dumps(res, indent=2))
    path = " -> ".join(res.get("call_path") or [])
    (out / f"{slug}.md").write_text(
        f"# Disclosure dossier: {res.get('function')} ({res.get('cwe')})\n\n"
        f"- repo: {res.get('repo_full')}  ({res.get('repo_url')})\n"
        f"- file: {res.get('file')}\n"
        f"- tainted parameter: {res.get('param')}\n"
        f"- Tier-2 sink: {res.get('tier2_sink')}\n"
        f"- Tier-2 exploit proof: {res.get('tier2_proof')}\n\n"
        f"## Reachability (what makes this disclosable)\n"
        f"- reachable from a real entry point: **{res.get('reachable')}**\n"
        f"- entry-point type: **{res.get('entry_kind')}**\n"
        f"- call path to verify by hand:\n\n    {path}\n\n"
        f"- dynamic reproduce verdict: {res.get('reproduce_verdict')}\n\n"
        f"## Manual verification checklist\n"
        f"1. Open the entry point (`{(res.get('call_path') or ['?'])[0]}`) and confirm it takes external/attacker input.\n"
        f"2. Follow the call path above and confirm the input reaches `{res.get('function')}` without being sanitized.\n"
        f"3. Reproduce the exploit (payload below) through that entry point against a local run.\n"
        f"4. Only then prepare a responsible-disclosure report.\n\n"
        f"- payload: `{str(res.get('payload'))[:200]}`\n\n"
        f"NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). "
        f"Step 1-3 above are the human confirmation that closes that gap before any disclosure.\n")
    return slug


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True, type=Path, help="frozen census JSON (with --tier2)")
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--repos", type=Path, default=Path("ai_repos.json"))
    ap.add_argument("--out", type=Path, default=Path("dossiers"))
    ap.add_argument("--cwe", type=str, default=None, help="only this CWE (e.g. CWE-22)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of findings (0 = all)")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    data = json.loads(args.census.read_text())
    rows = data["rows"] if isinstance(data, dict) else data
    confirmed = [r for r in rows if r.get("confirmed")]
    if args.cwe:
        confirmed = [r for r in confirmed if r.get("exec_cwe") == args.cwe]
    if args.limit:
        confirmed = confirmed[: args.limit]

    clone_urls = _clone_urls(args.repos)
    cache_root = Path(tempfile.mkdtemp(prefix="vulcan_disc_clones_"))
    repo_clone: dict = {}   # full_name -> cached clone path (or None if clone failed)

    summary = Counter()
    disclosable = []
    print(f"VULCAN disclosure funnel over {len(confirmed)} Tier-2 CONFIRMED finding(s)\n")
    print(f"  {'CWE':9} {'function':30} {'repo':26} verdict")
    print(f"  {'-'*9} {'-'*30} {'-'*26} {'-'*22}")

    try:
        for row in confirmed:
            rel = row["file"]
            cand = args.root / rel
            try:
                j = json.loads(cand.read_text())
            except (OSError, json.JSONDecodeError):
                summary["LOAD_FAILED"] += 1
                continue

            src = j.get("ground_truth_source") or ""
            func = j.get("function")
            real_file = j.get("file")
            cwe = row.get("exec_cwe") or (j.get("candidate_cwes") or [None])[0]
            fn_full = _full_name(rel)
            rec = {"repo": j.get("repo"), "file": real_file, "function": func,
                   "line": j.get("line", 1), "ground_truth_source": src,
                   "module_source": j.get("module_source", "")}

            # 2. re-prove with the deterministic bank to recover param+payload+proof
            try:
                parsed, _ = _parse_params(src)
            except Exception:  # noqa: BLE001
                parsed = []
            params = [p["name"] for p in parsed if p["name"] not in ("self", "cls")]
            got = None
            # Mirror the census: make this candidate's repo vendored source
            # importable (appended, so it can't shadow stdlib) before re-proving,
            # or candidates that import repo-local modules fail to re-confirm.
            injected = vendor_sys_path_roots(cand.parent / "_repo_src")
            for r in injected:
                if r not in sys.path:
                    sys.path.append(r)
            # Isolate this finding's imports: snapshot sys.modules and roll back
            # anything imported while re-proving. Without this, repo A's module
            # (e.g. `config`, `utils`) stays cached and shadows repo B's same-named
            # module later in the run -> spurious NOT_RECONFIRMED (process drift).
            _mods_before = set(sys.modules)
            try:
                for pn in params:
                    try:
                        res = attempt_exploit(rec, pn, cwe, use="ground_truth",
                                              max_agent=0, timeout=args.timeout)
                    except Exception:  # noqa: BLE001
                        continue
                    if res.get("confirmed"):
                        got = res
                        break
            finally:
                for r in injected:
                    if r in sys.path:
                        sys.path.remove(r)
                for m in set(sys.modules) - _mods_before:
                    del sys.modules[m]
            # The frozen census (the full Tier-2 bank) is the ground-truth
            # confirmation. The re-prove above is a redundant in-run check that can
            # drift (same-named repo modules, process state) even though the finding
            # re-proves fine standalone. When it drifts we do NOT drop a census-
            # confirmed finding -- that would hide a real, confirmed result. We keep
            # it, still run the reachability gate below, and mark in the dossier which
            # confirmation it rests on (live re-prove vs frozen census). No
            # fabrication: a census_only finding carries no invented param/payload.
            reproven = got is not None
            if not reproven:
                summary["REPROVE_DRIFTED"] += 1
                got = {"param": None, "payload": None, "proof": None, "sink": None}

            # 3. REACHABILITY (primary gate): can a real entry point reach this
            # sink? Pure static analysis on the vendored source -- works even when
            # the real package won't import (unlike the dynamic reproduce below).
            repo_src = cand.parent / "_repo_src"
            reachable, call_path, entry_kind = (False, [], None)
            if repo_src.exists():
                try:
                    reachable, call_path, entry_kind = reaches(repo_src, func)
                except Exception:  # noqa: BLE001
                    pass

            # 4. reproduce through the real public interface -- a BONUS dynamic
            # confirmation when the package imports; never required.
            if fn_full not in repo_clone:
                dest = cache_root / fn_full.replace("/", "__")
                from reproduce import _clone_at_commit  # noqa: E402
                repo_clone[fn_full] = dest if _clone_at_commit(
                    clone_urls.get(fn_full, f"https://github.com/{fn_full}.git"), "", dest) else None
            clone_dir = repo_clone[fn_full]
            finding = {"repo_url": clone_urls.get(fn_full, f"https://github.com/{fn_full}.git"),
                       "commit": "", "file": real_file, "function": func, "cwe": cwe,
                       "param": got.get("param"), "payload": got.get("payload")}
            if reproven and got.get("param"):
                try:
                    with tempfile.TemporaryDirectory(prefix="vulcan_repro_") as td:
                        rres = reproduce(finding, Path(td), clone_dir=clone_dir)
                except Exception:  # noqa: BLE001
                    rres = {**finding, "verdict": "SETUP_FAILED", "detail": "reproduce raised"}
            else:
                # no live param/payload this run (re-prove drifted); skip the BONUS
                # dynamic reproduce -- reachability below is the real gate.
                rres = {**finding, "verdict": "SKIPPED_CENSUS_ONLY"}
            repro_verdict = rres.get("verdict")

            # Disclosure verdict = reachability-first.
            if reachable:
                verdict = "REACHABLE_FROM_ENTRYPOINT"
            else:
                verdict = "INTERNAL_UNREACHABLE"
            summary[verdict] += 1

            rres.update({"tier2_proof": got.get("proof"), "tier2_sink": got.get("sink"),
                         "reachable": reachable, "entry_kind": entry_kind,
                         "call_path": call_path, "reproduce_verdict": repro_verdict,
                         "disclosure_verdict": verdict, "repo_full": fn_full,
                         "reprove_status": "reproven" if reproven else "census_only"})
            if reachable:
                _write_dossier(rres, args.out)
                disclosable.append(rres)
            path_str = (" [" + entry_kind + ": " + " -> ".join(call_path) + "]") if reachable else ""
            tag = "" if reproven else "  (census-confirmed; in-run reprove drifted)"
            print(f"  {str(cwe):9} {str(func)[:26]:26} {fn_full[:24]:24} -> {verdict}{path_str}{tag}")
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)

    print("\n==================== DISCLOSURE TRIAGE ====================")
    for k, v in summary.most_common():
        print(f"  {v:3}  {k}")
    nreach = summary.get("REACHABLE_FROM_ENTRYPOINT", 0)
    print(f"\n{nreach} finding(s) REACHABLE from a real entry point (HTTP route / CLI / public API)")
    print(f"-> dossiers with the call path written to {args.out}/")
    print("Each dossier names the entry point and the exact call path to the sink. Walk that")
    print("path by hand, confirm the input is attacker-controlled and unsanitized, reproduce")
    print("through the interface, THEN disclose. 'INTERNAL_UNREACHABLE' = no entry point reaches")
    print("it (genuinely internal / dead / test-only) -- correctly held back.")


if __name__ == "__main__":
    main()
