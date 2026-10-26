#!/usr/bin/env python3
"""VULCAN corpus-wide TWO-AXIS classification (Exploitability x Reachability).

See validation/TAXONOMY.md for canonical label definitions.

Axis A (Exploitability, function-level, execution oracle):
    CONFIRMED | TRIGGERED_ONLY | DETECTION_ONLY | SAFE | NOT_EXECUTABLE
Axis B (Reachability, application-level):
    APP_CONFIRMED | REFUTED | STATIC_LEAD | UNREACHABLE
    (manual labels: APP_CONFIRMED, REFUTED; auto labels: STATIC_LEAD, UNREACHABLE)

Honesty: only gold HTTP candidates carry MANUAL reachability verdicts; every other
row is auto. An unverified STATIC_LEAD is a lead, NOT a reachability claim.
"""
from __future__ import annotations
import json, collections, re
from pathlib import Path

CENSUS   = Path("corpus_census_pinned.json")
WORKLIST = Path("reachable_worklist.json")
OUT      = Path("corpus_classification.json")

DETECTION_ONLY_CWES = {"CWE-89", "CWE-502", "CWE-611"}

# MANUAL reachability verdicts established this session (gold HTTP candidates).
# key: (norm_repo, function) -> (reach_label, note)
MANUAL = {
 ("nthplusio__skills","send_webhook"): ("APP_CONFIRMED","stored SSRF, end-to-end canary + controls (fixture scope)"),
 ("2100preet__stockscanner","save_day_store"): ("REFUTED","route takes no request input; internal path"),
 ("2100preet__stockscanner","load_config"): ("REFUTED","defaults to DEFAULT_CONFIG; internal path"),
 ("kamoras__civitas","read_record"): ("REFUTED","GET /pipeline/status, no request-controlled input"),
 ("kamoras__civitas","read_json"): ("REFUTED","state allowlisted + taint->DB-filter not file path (full-source)"),
 ("kamoras__civitas","fetch_og_card"): ("REFUTED","admin-gated + internal-data taint"),
 ("kamoras__civitas","build_link_card"): ("REFUTED","admin-gated + internal-data taint"),
 ("kamoras__civitas","update_json_file"): ("REFUTED","auth-gated + internal-data taint"),
 ("kamoras__civitas","_get"): ("REFUTED","auth-gated + internal-data taint"),
 ("protossblaster__leapmotor-mate","_write_bytes"): ("REFUTED","refresh_now ignores request; internal cert paths (full-source)"),
 ("gunrunner444__bugforge","prepare_foundry_workspace"): ("REFUTED","operator-gated; repo from session; guarded copy (full-source)"),
}

def norm_repo(s: str) -> str:
    return s.replace("/", "__").lower()

def parse_census_file(fp: str):
    repo = fp.split("/", 1)[0]
    rest = re.sub(r"\.json$", "", fp.rsplit("/", 1)[-1])
    func = rest.split(".py__", 1)[1] if ".py__" in rest else rest.split("__")[-1]
    return norm_repo(repo), func

def main():
    census = json.loads(CENSUS.read_text())
    rows = census["rows"] if isinstance(census, dict) else census
    worklist = json.loads(WORKLIST.read_text())

    reach = {}
    for w in worklist:
        if not w.get("reachable"):
            continue
        k = (norm_repo(w.get("repo","")), w.get("function",""))
        prio = w.get("prio", 9)
        if k not in reach or prio < reach[k][0]:
            reach[k] = (prio, w.get("entry_kind"))

    EXPL_RANK  = {"CONFIRMED":0,"TRIGGERED_ONLY":1,"DETECTION_ONLY":2,"SAFE":3,"NOT_EXECUTABLE":4}
    REACH_RANK = {"APP_CONFIRMED":0,"REFUTED":1,"STATIC_LEAD":2,"UNREACHABLE":3}

    def expl_of(r, cwes):
        if not bool(r.get("executable")): return "NOT_EXECUTABLE"
        if bool(r.get("confirmed")):      return "CONFIRMED"
        if cwes and set(cwes) <= DETECTION_ONLY_CWES: return "DETECTION_ONLY"
        if bool(r.get("triggered")):      return "TRIGGERED_ONLY"
        return "SAFE"

    # ---- candidate-level pass (raw rows), then dedup to function-level ----
    cand_expl = collections.Counter()
    funcs = {}   # (repo,function) -> aggregated record
    for r in rows:
        k = parse_census_file(r["file"])
        cwes = r.get("cwes") or []
        e = expl_of(r, cwes)
        cand_expl[e] += 1
        rk = reach.get(k)
        f = funcs.get(k)
        if f is None:
            f = {"repo": k[0], "function": k[1], "file": r["file"],
                 "cwes": set(cwes), "expl": e,
                 "static_reachable": rk is not None,
                 "entry_kind": rk[1] if rk else None, "n_candidates": 0}
            funcs[k] = f
        else:
            f["cwes"].update(cwes)
            if EXPL_RANK[e] < EXPL_RANK[f["expl"]]:      # keep strongest exploitability
                f["expl"] = e
            if rk is not None:
                f["static_reachable"] = True
                f["entry_kind"] = f["entry_kind"] or rk[1]
        f["n_candidates"] += 1

    out = []
    expl_counts = collections.Counter()
    crosstab = collections.Counter()
    src_counts = collections.Counter()
    queue = []
    for k, f in funcs.items():
        manual = MANUAL.get(k)
        if manual:
            reach_label, source = manual[0], "manual"
        elif f["static_reachable"]:
            reach_label, source = "STATIC_LEAD", "auto"
        else:
            reach_label, source = "UNREACHABLE", "auto"
        expl = f["expl"]
        if expl == "CONFIRMED" and reach_label == "STATIC_LEAD" and f["entry_kind"] == "http":
            queue.append((k[0], k[1], ",".join(sorted(f["cwes"]))))
        expl_counts[expl] += 1
        crosstab[(expl, reach_label)] += 1
        src_counts[source] += 1
        out.append({
            "repo": k[0], "function": k[1], "file": f["file"],
            "cwes": sorted(f["cwes"]), "n_candidates": f["n_candidates"],
            "exploitability": expl, "reachability": reach_label,
            "entry_kind": f["entry_kind"], "source": source,
            "note": manual[1] if manual else None,
        })
    OUT.write_text(json.dumps(out, indent=2))

    EXPL = ["CONFIRMED","TRIGGERED_ONLY","DETECTION_ONLY","SAFE","NOT_EXECUTABLE"]
    REACH = ["APP_CONFIRMED","REFUTED","STATIC_LEAD","UNREACHABLE"]
    print("="*72)
    print(f"VULCAN TWO-AXIS CLASSIFICATION")
    print(f"  canonical unit: FUNCTION ({len(out)} distinct functions)")
    print(f"  secondary unit: CANDIDATE ({sum(cand_expl.values())} rows = function x injection-point)")
    print("="*72)
    print("\nExploitability (Axis A)        function | candidate")
    for e in EXPL:
        if expl_counts[e] or cand_expl[e]:
            print(f"  {e:16s} {expl_counts[e]:8d} | {cand_expl[e]:8d}")
    print("\nCrosstab  Exploitability x Reachability  (FUNCTION-level):")
    print(f"  {'':16s} " + "".join(f"{r:>14s}" for r in REACH))
    for e in EXPL:
        if not expl_counts[e]: continue
        cells = "".join(f"{crosstab[(e,r)]:>14d}" for r in REACH)
        print(f"  {e:16s} {cells}")
    print("-"*72)
    print(f"  reachability source: manual={src_counts['manual']}  auto={src_counts['auto']}")
    print("="*72)
    print("\nDerived sets:")
    gold = sum(crosstab[("CONFIRMED",r)] for r in ("APP_CONFIRMED","REFUTED","STATIC_LEAD"))
    print(f"  Gold set (CONFIRMED x reachability-signal): {gold}")
    print(f"  Headline reachable (CONFIRMED x APP_CONFIRMED): {crosstab[('CONFIRMED','APP_CONFIRMED')]}")
    print(f"  Attrition evidence (CONFIRMED x REFUTED): {crosstab[('CONFIRMED','REFUTED')]}")
    print("\nMANUAL-VERIFICATION QUEUE (CONFIRMED x STATIC_LEAD x http, auto, not yet hand-checked):")
    if not queue:
        print("  (empty — every confirmed HTTP static-lead already manually resolved)")
    else:
        for repo, fn, cwe in queue:
            print(f"   {repo:32s} {fn:28s} {cwe}")
    print(f"\nwrote per-function two-axis classification -> {OUT}")

if __name__ == "__main__":
    main()
