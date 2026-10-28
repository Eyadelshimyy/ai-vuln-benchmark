#!/usr/bin/env python3
"""RQ1 -- which execution-confirmed weaknesses does each AI code generator
produce? (descriptive, sink-conditioned, repo-clustered)

Reads the frozen census (no execution, no model calls) and reports, per
generating agent (claude-code / cursor / devin / github-copilot, from git-blame
commit signatures; see analysis/provenance.py) and -- where recoverable -- per
generating MODEL (from validation/rq1/commit_model_attribution.json):

  funnel        candidates -> executable -> triggered -> confirmed
  rate          confirmed / executable, with a REPO-CLUSTER bootstrap 95% CI
                (functions in one repo are not independent; the repo is the
                resampling unit)
  class mix     confirmed by exec_cwe (the class that fired, not candidate tags)
  CWE-22 kind   BASE_JOIN_OVERRIDE vs DIRECT_CALLER_PATH (analysis/cwe22_subtype.py)
  units         candidate rows and source-unique functions (repo, sha256(source))

What this can and cannot say (printed into the report too):
  * The population is SINK-SELECTED: candidates were mined because they contain
    a security-relevant sink. Rates are "among sink-bearing AI-authored functions",
    never "per line of AI code".
  * Generator is confounded with repository (domain, author, prompting). Each
    repo mostly uses one agent; a cross-generator difference is a difference
    between repo populations unless it also holds WITHIN the repos that use
    several agents (reported separately).
  * Agent product != model. The model is known only where the commit trailer
    names it.
  * COULD_NOT_EXECUTE rows are excluded from rate denominators, never counted safe.

  python3 analysis/rq1_by_generator.py --census corpus_census_reproduced.json \
      --models-attr validation/rq1/commit_model_attribution.json --out-dir validation/rq1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cwe22_subtype import classify_record  # noqa: E402

MIXED = "mixed"


def load_rows(census: Path, root: Path, models_attr: dict | None) -> list[dict]:
    out = []
    for r in json.loads(census.read_text())["rows"]:
        rec = json.loads((root / r["file"]).read_text())
        prov = rec.get("ai_provenance") or {}
        tools = sorted(prov.get("ai_tools") or {})
        gen = tools[0] if len(tools) == 1 else (MIXED if tools else "unattributed")
        src = rec.get("ground_truth_source") or rec.get("module_source") or ""
        cwe = r.get("confirmed_cwe") or r.get("exec_cwe")
        row = {"file": r["file"], "repo": rec.get("repo"), "generator": gen,
               "verdict": r.get("verdict"), "executable": bool(r.get("executable")),
               "triggered": bool(r.get("triggered")), "confirmed": bool(r.get("confirmed")),
               "cwe": cwe if r.get("confirmed") else None,
               "src_key": (rec.get("repo"), hashlib.sha256(src.strip().encode()).hexdigest()),
               "ai_line_fraction": prov.get("ai_line_fraction"),
               "model": None, "commit_date": None}
        if r.get("confirmed") and cwe == "CWE-22":
            row["cwe22_subtype"] = classify_record(rec)["subtype"]
        if models_attr is not None and prov.get("ai_commit"):
            m = models_attr.get(f"{rec.get('repo')}@{prov['ai_commit']}") or {}
            # "(1M context)" is a context-window variant of the same model
            row["model"] = re.sub(r"\s*\(1M context\)", "", m["model"]) if m.get("model") else None
            row["commit_date"] = (m.get("commit_date") or "")[:10] or None
        out.append(row)
    return out


def cluster_bootstrap_rate(rows: list[dict], n_boot=2000, seed=0):
    """confirmed/executable, resampling REPOS with replacement."""
    by_repo = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["executable"]:
            by_repo[r["repo"]][1] += 1
            by_repo[r["repo"]][0] += int(r["confirmed"])
    repos = list(by_repo)
    k = sum(v[0] for v in by_repo.values()); n = sum(v[1] for v in by_repo.values())
    if not n:
        return None, None, None
    if len(repos) < 2:
        return k / n, None, None
    rng = random.Random(seed); est = []
    for _ in range(n_boot):
        s = [by_repo[rng.choice(repos)] for _ in repos]
        kk = sum(a for a, _ in s); nn = sum(b for _, b in s)
        if nn:
            est.append(kk / nn)
    est.sort()
    return k / n, est[int(0.025 * len(est))], est[int(0.975 * len(est)) - 1]


def summarize(rows: list[dict]) -> dict:
    conf = [r for r in rows if r["confirmed"]]
    rate, lo, hi = cluster_bootstrap_rate(rows)
    return {
        "repos": len({r["repo"] for r in rows}),
        "repos_with_confirmed": len({r["repo"] for r in conf}),
        "candidates": len(rows),
        "executable": sum(r["executable"] for r in rows),
        "triggered": sum(r["triggered"] for r in rows),
        "confirmed": len(conf),
        "confirmed_source_unique": len({r["src_key"] for r in conf}),
        "confirm_rate_exec": rate, "confirm_rate_ci95_repo_cluster": [lo, hi],
        "confirmed_by_cwe": dict(Counter(r["cwe"] for r in conf)),
        "cwe22_subtypes": dict(Counter(r.get("cwe22_subtype") for r in conf if r["cwe"] == "CWE-22")),
        "max_repo_share_of_confirmed": (max(Counter(r["repo"] for r in conf).values()) / len(conf)) if conf else None,
    }


def within_repo(rows: list[dict]) -> list[dict]:
    """Repos where >=2 single-agent generators each have executable candidates."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["generator"] not in (MIXED, "unattributed"):
            by[r["repo"]][r["generator"]].append(r)
    out = []
    for repo, g in sorted(by.items()):
        gs = {k: v for k, v in g.items() if any(x["executable"] for x in v)}
        if len(gs) < 2:
            continue
        out.append({"repo": repo, "by_generator": {
            k: {"executable": sum(x["executable"] for x in v), "confirmed": sum(x["confirmed"] for x in v)}
            for k, v in sorted(gs.items())}})
    return out


def _fmt_rate(s):
    r, (lo, hi) = s["confirm_rate_exec"], s["confirm_rate_ci95_repo_cluster"]
    if r is None:
        return "-"
    if s["confirmed"] == 0:
        return f"0/{s['executable']} (no CI: zero events)"
    return f"{r:.1%}" + (f" [{lo:.1%}, {hi:.1%}]" if lo is not None else " [n/a: 1 repo]")


def render_md(res: dict) -> str:
    L = ["# RQ1 — execution-confirmed weaknesses by AI generator", "",
         f"Census: `{res['census']}` (sha256/16 `{res['census_sha256_16']}`). Generated by "
         "`analysis/rq1_by_generator.py`; no execution, no model calls.", "",
         "**Read first.** The population is *sink-selected* AI-authored Python functions (mined because they "
         "contain a security-relevant sink), so every rate is conditional on that selection — never per line "
         "of AI code. Generator is confounded with repository; see the within-repo table. CONFIRMED means "
         "*execution-confirmed conditional exploitability* (PREDICATES.md): it is not attacker reachability "
         "(validation/TAXONOMY.md Axis B).", "",
         "## By generating agent", "",
         "| Generator | Repos (w/ confirmed) | Candidates | Executable | Triggered | Confirmed (src-unique) | Confirmed / executable [95% repo-cluster CI] | Confirmed by CWE | CWE-22: direct / base-join / unknown | Largest single-repo share |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for g, s in res["by_generator"].items():
        st = s["cwe22_subtypes"]
        share = "-" if s["max_repo_share_of_confirmed"] is None else f"{s['max_repo_share_of_confirmed']:.0%}"
        L.append(f"| {g} | {s['repos']} ({s['repos_with_confirmed']}) | {s['candidates']} | {s['executable']} | "
                 f"{s['triggered']} | {s['confirmed']} ({s['confirmed_source_unique']}) | {_fmt_rate(s)} | "
                 f"{', '.join(f'{k}:{v}' for k, v in sorted(s['confirmed_by_cwe'].items())) or '-'} | "
                 f"{st.get('DIRECT_CALLER_PATH', 0)} / {st.get('BASE_JOIN_OVERRIDE', 0)} / {st.get('UNKNOWN', 0)} | "
                 f"{share} |")
    a = res["all"]
    L += ["", f"All generators: {a['confirmed']} confirmed candidates ({a['confirmed_source_unique']} source-unique) "
          f"from {a['executable']} executable; CWE-22 subtypes {a['cwe22_subtypes']}.", ""]
    if res.get("by_model"):
        L += ["## By generating model (only where a commit trailer names it)", "",
              f"Model string recovered for {res['model_coverage']['with_model']}/{res['model_coverage']['candidates']} "
              "candidates. Absence of a model string is not evidence about the model.", "",
              "| Model | Repos | Executable | Confirmed | Confirmed / executable [95% repo-cluster CI] | Confirmed by CWE |",
              "|---|---|---|---|---|---|"]
        for m, s in res["by_model"].items():
            L.append(f"| {m} | {s['repos']} | {s['executable']} | {s['confirmed']} | {_fmt_rate(s)} | "
                     f"{', '.join(f'{k}:{v}' for k, v in sorted(s['confirmed_by_cwe'].items())) or '-'} |")
        L.append("")
    L += ["## Within-repo comparison (repos using ≥2 agents)", "",
          "The only comparison not confounded by repository. Counts are small; descriptive only.", "",
          "| Repo | executable / confirmed per agent |", "|---|---|"]
    for w in res["within_repo"]:
        L.append(f"| {w['repo']} | " + "; ".join(f"{g}: {v['executable']}/{v['confirmed']}"
                                                  for g, v in w["by_generator"].items()) + " |")
    L += ["", "## Interpretation guardrails", "",
          "- Differences between generators in the first table are differences between *repository populations*. "
          "Overlapping repo-cluster CIs mean no detectable difference.",
          "- `DIRECT_CALLER_PATH` CWE-22 rows confirm that the function opens whatever path its caller passes. "
          "That is a vulnerability only when an untrusted caller controls the argument, which is an Axis-B "
          "(reachability) question.",
          "- The subtype is a static heuristic (analysis/cwe22_subtype.py). It was hand-checked on 10 random DIRECT rows (all correct). "
          "It stratifies the results; it never relabels them."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--models-attr", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    ma = json.loads(args.models_attr.read_text()) if args.models_attr and args.models_attr.exists() else None
    rows = load_rows(args.census, args.root, ma)
    gens = sorted({r["generator"] for r in rows}, key=lambda g: (g in (MIXED, "unattributed"), g))
    res = {"census": str(args.census),
           "census_sha256_16": hashlib.sha256(args.census.read_bytes()).hexdigest()[:16],
           "all": summarize(rows),
           "by_generator": {g: summarize([r for r in rows if r["generator"] == g]) for g in gens},
           "within_repo": within_repo(rows)}
    if ma is not None:
        with_m = [r for r in rows if r["model"]]
        res["model_coverage"] = {"candidates": len(rows), "with_model": len(with_m)}
        res["by_model"] = {m: summarize([r for r in with_m if r["model"] == m])
                           for m in sorted({r["model"] for r in with_m})}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rq1_by_generator.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    (args.out_dir / "RQ1_BY_GENERATOR.md").write_text(render_md(res))
    print(render_md(res))


if __name__ == "__main__":
    main()
