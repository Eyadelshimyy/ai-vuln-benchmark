#!/usr/bin/env python3
"""VULCAN corpus-readiness census (NO LLM required).

Answers the only question that should gate "is the corpus ready per CWE":
not how many functions STATICALLY match a sink (that's the miner's guess), but
how many actually EXECUTE and how many the oracle TRIGGERS. Reports, per CWE,
the three numbers that matter -- distinguishing a thin class from an empty one:

  Discovered  -- mined candidates whose static sink-classifier tagged this CWE
                 (what calibration/mined_prompts/*/*.json already contains)
  Executable  -- ran in the sandbox without COULD_NOT_EXECUTE (construction,
                 import, and interface all worked) -> an ELIGIBLE sample
  Triggered   -- Tier-1 execution carried tainted input to the sink -> the
                 execution VULN ground truth detection is graded against
  Confirmed   -- (only with --tier2) the deterministic Tier-2 bank proved a
                 real observable exploit, NO LLM crafter involved

This is pure execution ground truth: it uses auto_harness.run_candidate and,
optionally, tier2.engine.attempt_exploit with max_agent=0 (bank payloads
only). Ollama / any detector model is NOT needed and NOT used.

Per-repo importability: each mined repo vendored its own source into
<repo>/_repo_src (mine_prompts.py). This injects the right one onto sys.path
per candidate so the REAL object constructs (same mechanism run_recycle uses),
then removes it again so repos can't cross-import.

USAGE
  python3 analysis/corpus_census.py                       # Tier-1 census (fast)
  python3 analysis/corpus_census.py --tier2               # also run the bank (slower)
  python3 analysis/corpus_census.py --limit 50            # sample, for a quick look
  python3 analysis/corpus_census.py --out corpus_census.json
"""
from __future__ import annotations

import argparse
import json
import sys
import collections
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "analysis" / "iast"))
sys.path.insert(0, str(_REPO))

from auto_harness import run_candidate, vendor_sys_path_roots  # noqa: E402
from run_benchmark import reduce_verdicts           # noqa: E402

_SKIP = {"auto_harness_results.json", "benchmark_results.json",
         "tier2_results.json", "detection_results.json", "provenance_results.json"}


def _candidate_files(root: Path):
    for p in sorted(root.glob("*/*.json")):
        if p.name in _SKIP:
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or "candidate_cwes" not in rec:
            continue
        if not (rec.get("ground_truth_source") or rec.get("prompt_source")):
            continue
        yield p, rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=_REPO / "calibration" / "mined_prompts")
    ap.add_argument("--tier2", action="store_true",
                    help="also run the deterministic Tier-2 bank (no LLM) to report Confirmed")
    ap.add_argument("--limit", type=int, default=0, help="census only the first N candidates (0 = all)")
    ap.add_argument("--cwe", default=None, help="census only candidates whose static tags include this CWE "
                    "(e.g. CWE-918) -- fast re-check of one class")
    ap.add_argument("--why", action="store_true",
                    help="for COULD_NOT_EXECUTE candidates, capture the error and print a root-cause "
                         "breakdown (missing module / construction / syntax / timeout / other) plus the "
                         "top missing modules -- tells you which fix recovers the most of the corpus")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--out", type=Path, default=_REPO / "corpus_census.json")
    args = ap.parse_args()

    if args.tier2:
        from tier2.engine import attempt_exploit  # noqa: E402

    # per-CWE tallies
    disc = collections.Counter()
    execd = collections.Counter()
    trig = collections.Counter()
    conf = collections.Counter()
    # --why root-cause tallies for COULD_NOT_EXECUTE
    cause = collections.Counter()
    missing_mod = collections.Counter()
    sig = collections.Counter()  # normalized error first-line -> count
    _MOD_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")

    def _signature(text: str) -> str:
        if not text:
            return "(no error text captured)"
        line = text.strip().splitlines()[0]
        line = re.sub(r"0x[0-9a-fA-F]+", "0x~", line)
        line = re.sub(r"'[^']*'", "'~'", line)
        line = re.sub(r'"[^"]*"', '"~"', line)
        line = re.sub(r"\b\d+\b", "N", line)
        return line[:160]

    def _classify_error(text: str) -> str:
        t = text or ""
        if "No module named" in t or t.startswith("ImportError") or "cannot import name" in t:
            m = _MOD_RE.search(t)
            if m:
                missing_mod[m.group(1).split(".")[0]] += 1
            return "missing_dependency"
        if "timed out" in t:
            return "timeout"
        if "SyntaxError" in t or "could not parse" in t:
            return "syntax/parse"
        if any(s in t for s in ("TypeError", "__init__", "positional argument",
                                "missing", "takes no", "AttributeError")):
            return "construction/interface"
        return "other"

    # executability is a property of a candidate; attribute to each of its CWEs
    rows = []

    files = list(_candidate_files(args.root))
    if args.cwe:
        files = [(p, r) for p, r in files if args.cwe in (r.get("candidate_cwes") or [])]
    if args.limit:
        files = files[: args.limit]
    total = len(files)
    print(f"Census over {total} mined candidates (tier2={args.tier2}) ...", file=sys.stderr)

    for i, (path, rec) in enumerate(files, 1):
        cwes = rec.get("candidate_cwes") or []
        for c in cwes:
            disc[c] += 1

        # Make this candidate's repo vendored source importable (all recorded
        # import roots, so nested/non-standard layouts resolve -- see
        # vendor_sys_path_roots / mine_prompts .pathdirs).
        # Append (not insert-at-front): the repo's vendored modules must NOT
        # shadow stdlib/installed packages (a repo-local http.py/types.py etc.
        # at the front of sys.path broke DB/SQL execution). Appended, stdlib
        # wins and the repo's own unique modules still resolve as a fallback.
        injected = vendor_sys_path_roots(path.parent / "_repo_src")
        for r in injected:
            if r not in sys.path:
                sys.path.append(r)
        err_text = ""
        try:
            verds = run_candidate(rec, use="ground_truth")
            red = reduce_verdicts(verds)
            v = red.get("verdict")
            exec_cwe = red.get("cwe")
            if v == "COULD_NOT_EXECUTE":
                err_text = next((getattr(vd, "error", None) or getattr(vd, "detail", "")
                                 for vd in verds if getattr(vd, "error", None) or getattr(vd, "detail", None)), "")
        except Exception as e:  # noqa: BLE001 -- one bad candidate must not abort the census
            v, exec_cwe, verds = "COULD_NOT_EXECUTE", None, []
            err_text = f"{type(e).__name__}: {e}"
        finally:
            for r in injected:
                if r in sys.path:
                    sys.path.remove(r)

        executable = v in ("TRIGGERED", "NOT_TRIGGERED")
        triggered = v == "TRIGGERED"
        for c in cwes:
            if executable:
                execd[c] += 1
            if triggered:
                trig[c] += 1
        if args.why and v == "COULD_NOT_EXECUTE":
            cause[_classify_error(err_text)] += 1
            sig[_signature(err_text)] += 1

        confirmed = False
        if args.tier2 and triggered:
            # deterministic bank only (max_agent=0) -> NO LLM. Try the candidate's
            # own CWE strategy on each plausible param until one confirms.
            for r in injected:
                if r not in sys.path:
                    sys.path.append(r)
            try:
                for param in {vd.injected_param for vd in verds if getattr(vd, "injected_param", None)}:
                    tgt_cwe = exec_cwe or (cwes[0] if cwes else None)
                    if not tgt_cwe:
                        continue
                    res = attempt_exploit(rec, param, tgt_cwe, use="ground_truth",
                                          max_agent=0, timeout=args.timeout)
                    if res.get("confirmed"):
                        confirmed = True
                        break
            except Exception:  # noqa: BLE001
                pass
            finally:
                for r in injected:
                    if r in sys.path:
                        sys.path.remove(r)
            if confirmed:
                for c in cwes:
                    conf[c] += 1

        rows.append({"file": str(path.relative_to(args.root)), "cwes": cwes,
                     "verdict": v, "exec_cwe": exec_cwe, "executable": executable,
                     "triggered": triggered, "confirmed": confirmed,
                     "error": (err_text or "").strip().splitlines()[0][:300] if (args.why and err_text) else None})
        if i % 25 == 0 or i == total:
            print(f"  {i}/{total}", file=sys.stderr, flush=True)

    # ---- report ----
    all_cwes = sorted(disc, key=lambda c: -disc[c])
    print("\n" + "=" * 72)
    print("VULCAN CORPUS-READINESS CENSUS (execution ground truth, no LLM)")
    print("=" * 72)
    hdr = f"{'CWE':10} {'Discovered':>10} {'Executable':>10} {'Triggered':>10}"
    if args.tier2:
        hdr += f" {'Confirmed':>10}"
    print(hdr)
    for c in all_cwes:
        line = f"{c:10} {disc[c]:>10} {execd[c]:>10} {trig[c]:>10}"
        if args.tier2:
            line += f" {conf[c]:>10}"
        print(line)
    print("-" * 72)
    tot = f"{'TOTAL':10} {sum(disc.values()):>10} {sum(execd.values()):>10} {sum(trig.values()):>10}"
    if args.tier2:
        tot += f" {sum(conf.values()):>10}"
    print(tot)
    print("\nDiscovered = static sink match | Executable = ran (eligible) | "
          "Triggered = taint reached sink (Tier-1 VULN ground truth)"
          + (" | Confirmed = Tier-2 bank proved exploit" if args.tier2 else ""))

    if args.why and sum(cause.values()):
        print("\n" + "=" * 72)
        print("WHY COULD_NOT_EXECUTE  (root cause of the excluded candidates)")
        print("=" * 72)
        tot_u = sum(cause.values())
        for c, n in cause.most_common():
            print(f"  {c:24} {n:>5}  ({round(100*n/tot_u)}%)")
        if missing_mod:
            print("\n  top missing modules (install these to recover the most):")
            for mod, n in missing_mod.most_common(20):
                print(f"     {mod:28} {n:>4}")
        print("\n  top error signatures (the ACTUAL failures -- this is what to fix):")
        for s, n in sig.most_common(25):
            print(f"     {n:>4}  {s}")

    out = {"per_cwe": {c: {"discovered": disc[c], "executable": execd[c],
                           "triggered": trig[c], "confirmed": conf[c]} for c in all_cwes},
           "n_candidates": total, "tier2": args.tier2, "rows": rows}
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nAuditable per-candidate census: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
