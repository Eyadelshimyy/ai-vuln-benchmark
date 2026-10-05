#!/usr/bin/env python3
"""
VULCAN -- per-candidate concordance analysis.

The benchmark table compares *rates* (AI 32% vs reference 31.8%). Matching
rates don't prove the AI is vulnerable on the SAME functions as the reference
implementation -- for that you need agreement candidate-by-candidate. This
reads the benchmark_results.json files the benchmark already produced (no new
runs) and, for each model, cross-tabulates the model's verdict against the real
reference implementation's verdict per function:

                      AI vulnerable      AI safe
  reference vuln      both vulnerable    reference-vuln, AI-safe  (AI missed it)
  reference safe      reference-safe,    both safe
                      AI-vuln (AI ADDED a vuln)

It reports the 2x2, observed agreement (concordance), and Cohen's kappa
(agreement beyond chance), per repo and pooled, and NAMES the off-diagonal
functions -- the AI-introduced and AI-missed cases, which are the interesting
discussion material.

Each candidate has k AI samples, so "the AI's verdict" needs a collapse. Two
are reported because they answer different questions:
  * vuln@k  -- VULN if ANY sample triggered (can the model produce a vulnerable
               implementation of this function?)
  * temp0   -- the single temperature-0 sample (what the model typically emits)

Only candidates where BOTH the reference executes AND the AI has an executable
verdict are counted (CNE on either side is excluded, not scored).

USAGE:
  python3 concordance.py --root calibration/mined_prompts/
  python3 concordance.py --results calibration/mined_prompts/sqlite-utils/benchmark_results.json ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _human(h: dict):
    v = h.get("verdict")
    return "VULN" if v == "TRIGGERED" else "SAFE" if v == "NOT_TRIGGERED" else None


def _ai_vulnk(samples: list):
    vs = [s.get("verdict") for s in samples]
    if "TRIGGERED" in vs:
        return "VULN"
    if "NOT_TRIGGERED" in vs:
        return "SAFE"
    return None  # all could-not-execute -> no verdict


def _ai_t0(samples: list):
    for s in samples:
        if s.get("sample") == "t0":
            v = s.get("verdict")
            return "VULN" if v == "TRIGGERED" else "SAFE" if v == "NOT_TRIGGERED" else None
    return None


def _kappa(bv: int, bs: int, hv_as: int, hs_av: int):
    """Cohen's kappa for the 2x2 (both-vuln, both-safe, ref-vuln/ai-safe, ref-safe/ai-vuln)."""
    n = bv + bs + hv_as + hs_av
    if n == 0:
        return None
    po = (bv + bs) / n
    # marginals
    ref_vuln = (bv + hv_as) / n
    ref_safe = (bs + hs_av) / n
    ai_vuln = (bv + hs_av) / n
    ai_safe = (bs + hv_as) / n
    pe = ref_vuln * ai_vuln + ref_safe * ai_safe
    if abs(1 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0  # degenerate (all one class)
    return (po - pe) / (1 - pe)


def _cells(candidates: list, model: str, collapse):
    """Return the 2x2 counts + off-diagonal function lists over candidates."""
    bv = bs = hv_as = hs_av = 0
    ai_added, ai_missed = [], []
    for r in candidates:
        h = _human(r.get("human", {}))
        if h is None:
            continue
        m = r.get("models", {}).get(model)
        if not m:
            continue
        a = collapse(m.get("samples", []))
        if a is None:
            continue
        fn = r.get("function", "?")
        if h == "VULN" and a == "VULN":
            bv += 1
        elif h == "SAFE" and a == "SAFE":
            bs += 1
        elif h == "VULN" and a == "SAFE":
            hv_as += 1; ai_missed.append(fn)
        else:  # h SAFE, a VULN
            hs_av += 1; ai_added.append(fn)
    return bv, bs, hv_as, hs_av, ai_added, ai_missed


def _fmt_row(label, bv, bs, hv_as, hs_av):
    n = bv + bs + hv_as + hs_av
    concord = (bv + bs) / n if n else 0.0
    k = _kappa(bv, bs, hv_as, hs_av)
    kstr = "n/a" if k is None else f"{k:+.2f}"
    return (f"  {label:<16} both_vuln={bv:<3} both_safe={bs:<3} "
            f"ref_vuln/AI_safe={hv_as:<3} ref_safe/AI_vuln={hs_av:<3} "
            f"| concord={concord*100:5.1f}%  kappa={kstr}  n={n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="dir containing <repo>/benchmark_results.json")
    ap.add_argument("--results", type=Path, nargs="*", help="explicit benchmark_results.json files")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = list(args.results or [])
    if args.root:
        files += sorted(args.root.glob("*/benchmark_results.json"))
    files = [f for f in files if f.exists()]
    if not files:
        raise SystemExit("no benchmark_results.json found (use --root or --results)")

    # repo -> per_candidate list ; collect model set
    per_repo: dict = {}
    models: set = set()
    for f in files:
        data = json.loads(f.read_text())
        cands = data.get("per_candidate", [])
        repo = (cands[0].get("repo") if cands else None) or f.parent.name
        per_repo.setdefault(repo, []).extend(cands)
        for r in cands:
            models.update(r.get("models", {}).keys())

    collapses = [("vuln@k", _ai_vulnk), ("temp0", _ai_t0)]
    report = {}
    for model in sorted(models):
        print("\n" + "=" * 78)
        print(f"MODEL: {model}")
        print("=" * 78)
        report[model] = {}
        for cname, cfn in collapses:
            print(f"\n  collapse = {cname}")
            tot = [0, 0, 0, 0]
            all_added, all_missed = [], []
            for repo in sorted(per_repo):
                bv, bs, hv_as, hs_av, added, missed = _cells(per_repo[repo], model, cfn)
                if bv + bs + hv_as + hs_av == 0:
                    continue
                print(_fmt_row(repo, bv, bs, hv_as, hs_av))
                tot[0] += bv; tot[1] += bs; tot[2] += hv_as; tot[3] += hs_av
                all_added += [f"{repo}:{x}" for x in added]
                all_missed += [f"{repo}:{x}" for x in missed]
            print(_fmt_row("POOLED", *tot))
            if all_added:
                print(f"    AI ADDED a vuln (ref safe, AI vuln): {', '.join(all_added)}")
            if all_missed:
                print(f"    AI MISSED a vuln (ref vuln, AI safe): {', '.join(all_missed)}")
            report[model][cname] = {"pooled": {"both_vuln": tot[0], "both_safe": tot[1],
                                               "ref_vuln_ai_safe": tot[2], "ref_safe_ai_vuln": tot[3],
                                               "concordance": (tot[0] + tot[1]) / sum(tot) if sum(tot) else 0.0,
                                               "kappa": _kappa(*tot)},
                                    "ai_added": all_added, "ai_missed": all_missed}

    print("\n" + "=" * 78)
    print("Reading: high concordance + near-empty 'ref safe / AI vuln' cell = the AI")
    print("reproduces the task's security posture rather than adding risk of its own.")
    print("=" * 78)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
