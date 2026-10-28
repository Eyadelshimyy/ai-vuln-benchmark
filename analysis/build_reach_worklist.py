#!/usr/bin/env python3
"""Phase 1 — unified reachability worklist across corpus v1 + v2.

Builds the canonical inventory of function-level CONFIRMED candidates that are
call-reachable from an HTTP route (any origin class), reconciled with the
existing manual/dynamic findings. Static evidence only: it runs the AST
call-graph screen (analysis/reachability.py) and reads frozen censuses and the
candidate records. It does NOT boot apps or send requests.

  python3 analysis/build_reach_worklist.py

Writes (canonical JSON first, Markdown generated from it):
  validation/reachability/REACH_WORKLIST.json
  validation/reachability/REACH_WORKLIST.md
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import reachability as R  # noqa: E402

CORPORA = [("v1", ROOT / "corpus_census_pinned.json", ROOT / "calibration" / "mined_prompts"),
           ("v2", ROOT / "corpus_census_v2.json", ROOT / "calibration" / "mined_prompts_v2")]

SINK_CAT = {"CWE-918": "ssrf-url", "CWE-78": "command", "CWE-89": "sql"}
# path sinks split by what the proof showed (write escape vs read); refined per record below
PRIORITY = {"ssrf-url": 0, "path-write": 1, "path-read": 2, "sql": 3, "command": 4, "other": 5}

# Existing findings to reconcile (from the in-repo artifacts, NOT from any external narrative).
PRIOR = {
    ("v1", "nthplusio__skills", "send_webhook"): {
        "prior_verdict": "APP_CONFIRMED (TAXONOMY.md) / REACHABILITY_SUPPORTED (REACHABILITY_MANUAL_VALIDATION.md)",
        "note": "label differs between the two v1 artifacts; the end-to-end run is described as 'pending/deferred' "
                "in REACHABILITY_MANUAL_VALIDATION.md. Reconcile before any 'demonstrated' claim. Fixture-scope code."},
    ("v1", "kamoras__civitas", None): {
        "prior_verdict": "REFUTED (manual full-source + faithful dynamic boot, DYNAMIC_PROBING.md)",
        "note": "114 routes probed live, canary-reachability proven, 0 SSRF/CWE-22/CWE-78 reachable."},
    ("v1", "ProtossBlaster__leapmotor-mate", None): {
        "prior_verdict": "REFUTED (stubbed boot + source: SSRF sink uses a hardcoded host)",
        "note": "attacker controls only lat/lon query data, never the target host."},
}


def path_cat(rec_src: str, proof: str) -> str:
    p = (proof or "").lower()
    if "write" in p or "escaped to" in p:
        return "path-write"
    if "read" in p or "exfiltr" in p or "opened our out-of-tree" in p:
        return "path-read"
    return "path-write"  # default for CWE-22 when proof text is unclassified


def main():
    graphs = {}
    items = []
    # param backfill: the pinned v1 census predates tier2_evidence; read params from the
    # reproduced v1 census + the v2 census, keyed by candidate file (unambiguous 1:1).
    param_map = {}
    for cf in (ROOT / "corpus_census_reproduced.json", ROOT / "corpus_census_v2.json"):
        if cf.exists():
            for r in json.loads(cf.read_text())["rows"]:
                ev = r.get("tier2_evidence") or {}
                if r.get("confirmed") and ev.get("param"):
                    param_map[r["file"]] = ev["param"]
    for tag, census, root in CORPORA:
        rows = [r for r in json.loads(census.read_text())["rows"] if r.get("confirmed")]
        for r in rows:
            rel = r["file"]; repo = rel.split("/")[0]
            rec = json.loads((root / rel).read_text())
            fn = rec.get("function")
            src_dir = root / repo / "_repo_src"
            src_sha = hashlib.sha256((rec.get("ground_truth_source") or "").strip().encode()).hexdigest()
            cwe = r.get("confirmed_cwe") or r.get("exec_cwe")
            ev = r.get("tier2_evidence") or {}
            cat = SINK_CAT.get(cwe) or (path_cat(rec.get("ground_truth_source", ""), ev.get("proof")) if cwe == "CWE-22" else "other")
            reachable = False; origin = None; chain = None; why = None
            skip = None
            if src_dir.is_dir():
                if repo not in graphs:
                    graphs[repo] = R.build_graph(str(src_dir))
                callers, handlers, funcs = graphs[repo]
                if fn not in funcs:
                    skip = "fn-not-found-in-callgraph"
                else:
                    why, chain = R.backward_reach(fn, callers, handlers, 4)
                    if why:
                        reachable = True
                        origin = R.source_origin(str(src_dir), fn)
            else:
                skip = "no-_repo_src"
            prior = PRIOR.get((tag, repo, fn)) or PRIOR.get((tag, repo, None))
            items.append({
                # stable, source-specific id: (tag,repo,source_file,function) is collision-free
                # (verified); never merges distinct source candidates that share repo+function name.
                "id": f"{tag}:{repo}:{rec.get('file')}:{fn}",
                "corpus": tag, "repo": repo, "function": fn,
                "candidate_file": rel, "source_file": rec.get("file"), "line": rec.get("line"),
                "source_sha256": src_sha,
                "pinned_commit": (rec.get("provenance") or {}).get("ai_commit") or (rec.get("ai_provenance") or {}).get("ai_commit"),
                "frozen_label": "CONFIRMED", "exec_cwe": cwe, "sink_category": cat,
                "confirmed_param": ev.get("param") or param_map.get(rel), "proof": ev.get("proof"),
                "route_reachable_static": reachable, "origin_class": origin,
                "call_chain": chain, "route_evidence": why, "screen_skip": skip,
                "generator": (lambda t: t[0] if len(t) == 1 else ("mixed" if t else "unattributed"))(
                    sorted((rec.get("ai_provenance") or {}).get("ai_tools") or {})),
                "auth_required": "unknown",
                "booted": False, "dynamically_tested": False,
                "prior_finding": prior,
                "reach_verdict": "INCONCLUSIVE",
                "adjudication_plan": None, "limitations": None,
            })

    # adjudication plan + priority
    for it in items:
        if not it["route_reachable_static"]:
            it["adjudication_plan"] = ("source-review only: no static HTTP entrypoint found"
                                       + (f" ({it['screen_skip']})" if it["screen_skip"] else ""))
            it["reach_verdict"] = "INCONCLUSIVE"
            it["limitations"] = "static screen is call-graph based and over/under-approximates; 'no route' is a screen negative, not demonstrated"
        else:
            it["adjudication_plan"] = ("Level-1 source review of route->sink dataflow; "
                                       "Level-2+ HTTP/dynamic test requires booting the app (not performed in this session)")
    items.sort(key=lambda it: (not it["route_reachable_static"], PRIORITY.get(it["sink_category"], 9),
                               it["origin_class"] != "attacker", it["corpus"], it["repo"], str(it["function"])))

    # counts reconciliation
    def uniq(tag):
        rows = [r for r in json.loads(dict((t, c) for t, c, _ in CORPORA)[tag].read_text())["rows"] if r.get("confirmed")]
        srcs = set()
        for r in rows:
            root = dict((t, rt) for t, _, rt in CORPORA)[tag]
            rec = json.loads((root / r["file"]).read_text())
            srcs.add((r["file"].split("/")[0], hashlib.sha256((rec.get("ground_truth_source") or "").strip().encode()).hexdigest()))
        return len(rows), len(srcs)
    v1n, v1s = uniq("v1"); v2n, v2s = uniq("v2")
    v1_repos = {it["repo"] for it in items if it["corpus"] == "v1"}
    v2_repos = {it["repo"] for it in items if it["corpus"] == "v2"}
    out = {
        "generated": "static reachability screen (analysis/reachability.py --depth 4) + frozen censuses; no app booted",
        "counts": {
            "v1_confirmed_candidate_rows": v1n, "v1_source_unique": v1s,
            "v2_confirmed_candidate_rows": v2n, "v2_source_unique": v2s,
            "repo_overlap_v1_v2": sorted(v1_repos & v2_repos),
            "combined_candidate_rows": v1n + v2n, "combined_source_unique": v1s + v2s,
            "unit_notes": "v1 pinned census = 160 candidate rows (156 source-unique, 149 name-dedup). "
                          "v2 = 37 rows (37 source-unique). v1/v2 repo sets are disjoint by construction "
                          "(v2 discovery excluded in-v1 repos), so combined = 197 rows / 193 source-unique. "
                          "The '34' elsewhere is RQ2 twin PAIRS (vuln/fixed), a different unit, not vulnerabilities.",
        },
        "route_reachable_static": {
            "v1": sum(1 for it in items if it["corpus"] == "v1" and it["route_reachable_static"]),
            "v2": sum(1 for it in items if it["corpus"] == "v2" and it["route_reachable_static"]),
        },
        "attacker_origin_flagged": [it["id"] for it in items if it["origin_class"] == "attacker"],
        "screen_limitations": [
            "call graph is name-based: generic names (execute, main, run) can create spurious or missed edges",
            "dynamic route registration and unresolved method dispatch are not followed",
            "origin classifier under-flags: it keys on request.* patterns; indirect request->value flows read as unclassified",
            "a 'not route-reachable' result is a STATIC negative, never a demonstrated one",
        ],
        "items": items,
    }
    wl = ROOT / "validation" / "reachability" / "REACH_WORKLIST.json"
    wl.write_text(json.dumps(out, indent=2))
    render_md(out, wl.with_suffix(".md"))
    print(f"worklist: {len(items)} confirmed candidates; route-reachable(static) "
          f"v1={out['route_reachable_static']['v1']} v2={out['route_reachable_static']['v2']}; "
          f"attacker-flagged {len(out['attacker_origin_flagged'])}")
    print(f"combined: {out['counts']['combined_candidate_rows']} rows / "
          f"{out['counts']['combined_source_unique']} source-unique; overlap {out['counts']['repo_overlap_v1_v2']}")
    print(f"wrote {wl} and {wl.with_suffix('.md')}")


def render_md(out, path):
    L = ["# VULCAN — unified reachability worklist (v1 + v2)", "",
         "Canonical source: `REACH_WORKLIST.json` (this table is generated from it). "
         "**Static evidence only** — the AST call-graph screen plus frozen censuses. No application was "
         "booted and no HTTP request was sent to build this. Route-reachability here is a *screen*, not a "
         "demonstration.", "",
         "## Counts", ""]
    c = out["counts"]
    L += [f"- v1: {c['v1_confirmed_candidate_rows']} confirmed rows ({c['v1_source_unique']} source-unique)",
          f"- v2: {c['v2_confirmed_candidate_rows']} confirmed rows ({c['v2_source_unique']} source-unique)",
          f"- repo overlap v1∩v2: {c['repo_overlap_v1_v2'] or 'none (disjoint by construction)'}",
          f"- combined: {c['combined_candidate_rows']} rows / {c['combined_source_unique']} source-unique",
          "", f"> {c['unit_notes']}", "",
          f"Route-reachable (static screen): v1 {out['route_reachable_static']['v1']}, "
          f"v2 {out['route_reachable_static']['v2']}. Attacker-origin flagged: "
          f"{len(out['attacker_origin_flagged'])} ({', '.join(out['attacker_origin_flagged']) or 'none'}).", "",
          "## Route-reachable candidates (adjudication worklist)", "",
          "Sorted by research priority: SSRF, path-write, path-read, SQL, command. "
          "`origin=attacker` first within each.", "",
          "| id | corpus | cwe | sink | param | origin | call chain | prior finding |",
          "|---|---|---|---|---|---|---|---|"]
    for it in out["items"]:
        if not it["route_reachable_static"]:
            continue
        chain = " → ".join(it["call_chain"] or []) if it["call_chain"] else (it["route_evidence"] or "")
        prior = (it["prior_finding"] or {}).get("prior_verdict", "—")
        L.append(f"| {it['id']} | {it['corpus']} | {it['exec_cwe']} | {it['sink_category']} | "
                 f"{it['confirmed_param'] or '?'} | {it['origin_class']} | {chain[:70]} | {prior} |")
    L += ["", "## Not route-reachable from any HTTP route (static screen)", "",
          f"{sum(1 for it in out['items'] if not it['route_reachable_static'])} confirmed candidates. "
          "Recorded as INCONCLUSIVE on app reachability (a screen negative is not a demonstrated one), "
          "not as safe. See JSON for per-item `screen_skip`.", "",
          "## Screen limitations", ""]
    L += [f"- {x}" for x in out["screen_limitations"]]
    L += ["", "## Adjudication plan", "",
          "- **Level-1 source review** (route→sink dataflow) is the next step for every route-reachable item "
          "and can be done statically.",
          "- **Level-2+ dynamic tests** (boot app, `curl --path-as-is`, in-process route/dataflow tracing) were "
          "**not performed in this session**. Candidates needing them stay INCONCLUSIVE on actual-application "
          "reachability. The Phase-2 tools named in the task brief (`reach_agent.py`, `dyn_prober_v4.py`) are "
          "**not present** in the repo.",
          "- Prior in-repo findings reconciled: nthplusio (label differs across two v1 docs — needs reconciling), "
          "civitas & leapmotor (REFUTED, with the specific hypothesis each refuted).",
          "", "## AutoClip note (integrity)", "",
          "The task brief describes an AutoClip `POST /api/v1/projects/../sync-data` test returning 200 with a "
          "DB row `id='..'`. **That route does not exist in the vendored AutoClip source, and no such evidence "
          "artifact (HTTP response, server log, DB-row dump) exists in this repo.** It was not run in this "
          "session. It is therefore NOT recorded as APP_CONFIRMED. AutoClip's confirmed `ai_cover` sink and its "
          "other `data/projects/{project_id}` path-construction sites are recorded as INCONCLUSIVE on "
          "application reachability (static trace in `../corpus_v2/REACHABILITY_V2.md` found the one traced "
          "route 404s on a non-existent project id)."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
