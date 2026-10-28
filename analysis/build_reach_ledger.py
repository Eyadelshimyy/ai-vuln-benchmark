#!/usr/bin/env python3
"""Phase 3 — canonical reachability ledger from the worklist + saved evidence + source reviews.

Static/organizational only: reads the worklist JSON, the saved AutoClip evidence directory,
the three source-review verdicts, and the prior in-repo findings. Assigns a reachability
verdict per adjudicated candidate and leaves the rest INCONCLUSIVE. Does NOT boot anything,
send requests, or alter any frozen function-level label.

Verdicts: APP_CONFIRMED | REACHABILITY_SUPPORTED | REFUTED | INCONCLUSIVE.
Rule enforced: APP_CONFIRMED requires a present, consistent real-application effect artifact.

  python3 analysis/build_reach_ledger.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REACH = ROOT / "validation" / "reachability"
EVID = REACH / "evidence" / "autoclip"

# Adjudications decided by source review / saved evidence (keyed by worklist id).
ADJ = {
    "v1:nthplusio__skills:skills/scoussens-skills/visualize-spec/evals/fixtures/order-hooks/src/order_hooks/webhooks.py:send_webhook": {
        "verdict": "REACHABILITY_SUPPORTED",
        "bound": "route->sink SSRF dataflow is source-supported (Flask /subscriptions->/orders->requests.post(url), no egress control); no persisted end-to-end effect artifact in repo",
        "auth_required": "no", "auth_basis": "source: plain Flask fixture app, no auth middleware",
        "evidence_type": "source-review", "evidence": ["source_reviews/nthplusio_send_webhook.md"],
        "repeatable": None, "scope": "eval-fixture code, not a deployed service",
        "unresolved": "no saved callback/outbound-fetch artifact; end-to-end run deferred (see REACHABILITY_MANUAL_VALIDATION.md)"},
    "v1:quantam101__profitenginev5:master_distillation/dataset.py:digest": {
        "verdict": "REFUTED",
        "bound": "attacker-flagged chain telegram_webhook->stable_embedding->digest is a FALSE call-graph edge (name collision with a local var 'digest'); the webhook does not reach the file hasher digest(path)",
        "auth_required": "n/a", "auth_basis": "no HTTP path to the sink",
        "evidence_type": "source-review", "evidence": ["source_reviews/quantam101_profitenginev5_digest.md"],
        "repeatable": None, "scope": "function-level CONFIRMED unchanged; app-reachability via the flagged route refuted",
        "unresolved": "none for the flagged route; digest(path) is reached from offline/CLI dataset code"},
    "v1:trsdn__macos-notarization-broker:scripts/broker.py:write_json": {
        "verdict": "REFUTED",
        "bound": "no HTTP entrypoint exists; argparse CLI tool (command_validate(args)). The screen's 'req-arg' flag matched validate_archive(request: dict,...) in local_handoff.py whose first param is coincidentally named 'request' (a notarization-request dict, not an HTTP request). Confirmed candidate is broker.py:779 write_json.",
        "auth_required": "n/a", "auth_basis": "argparse CLI tool, no web framework",
        "evidence_type": "source-review", "evidence": ["source_reviews/trsdn_macos-notarization-broker_write_json.md"],
        "repeatable": None, "scope": "function-level CONFIRMED unchanged; app-reachability refuted",
        "unresolved": "none"},
    # v2 candidate-specific, imported from validation/corpus_v2/REACHABILITY_V2.md (full-source traces)
    "v2:BerniceZimidzi-code__foodies-sop-rag-system:backend/database.py:execute": {
        "verdict": "REFUTED",
        "bound": "route-reachable, but every route-reachable caller passes CONSTANT SQL text with ? placeholders (e.g. activate_user: WHERE activation_token_hash = ? AND ... > ?); request data is only ever a bound parameter, never SQL text",
        "auth_required": "unknown", "auth_basis": "not itemized in the v2 manual trace",
        "evidence_type": "manual full-source", "evidence": ["../corpus_v2/REACHABILITY_V2.md"],
        "repeatable": None, "scope": "the sql helper itself is function-level CONFIRMED (executes whatever SQL it is given); callers do not pass request-controlled SQL",
        "unresolved": "the one f-string query in this module (PRAGMA table_info({table_name})) takes an internal table name"},
    "v2:spectral-butters__hacky-movies:backend/database.py:fetch_all": {
        "verdict": "REFUTED",
        "bound": "route-reachable, but callers pass constant SQL with ? placeholders (e.g. SELECT ... WHERE event_id = ?); request data is a bound parameter",
        "auth_required": "unknown", "auth_basis": "not itemized",
        "evidence_type": "manual full-source", "evidence": ["../corpus_v2/REACHABILITY_V2.md"],
        "repeatable": None, "scope": "function-level CONFIRMED unchanged; callers do not pass request-controlled SQL",
        "unresolved": "none noted"},
    "v2:spectral-butters__hacky-movies:backend/database.py:fetch_one": {
        "verdict": "REFUTED",
        "bound": "route-reachable, but callers pass constant SQL with ? placeholders; request data is a bound parameter",
        "auth_required": "unknown", "auth_basis": "not itemized",
        "evidence_type": "manual full-source", "evidence": ["../corpus_v2/REACHABILITY_V2.md"],
        "repeatable": None, "scope": "function-level CONFIRMED unchanged; callers do not pass request-controlled SQL",
        "unresolved": "none noted"},
    "v2:ayushdad609-code__autoclip:backend/services/studio/publish_kit.py:ai_cover": {
        "verdict": "REFUTED",
        "bound": "ai_cover's OWN confirmed route (POST .../cover/ai) is guarded by project_or_404, which 404s on a non-existent project id, so '..' never reaches ai_cover. REFUTED for the cover route only.",
        "auth_required": "unknown", "auth_basis": "cover route calls project_or_404(project_id, db)",
        "evidence_type": "manual full-source", "evidence": ["../corpus_v2/REACHABILITY_V2.md"],
        "repeatable": None,
        "scope": "cover route only. SEPARATELY, AutoClip's get_project_directory(project_id) traversal IS attacker-reachable via a DIFFERENT, unguarded route (POST /{project_id}/sync-data) -> see the app-level APP_CONFIRMED finding. The two routes are not identical; ai_cover's own route stays REFUTED.",
        "unresolved": "none for the cover route"},
}

# Prior REFUTED findings for whole repos (apply to every confirmed candidate in them unless overridden).
REPO_PRIOR = {
    "kamoras__civitas": {
        "verdict": "REFUTED",
        "applies_to_sink": ["ssrf-url", "path-write", "path-read"],
        "bound": "faithful dynamic boot: 114 routes probed live, canary-reachability proven, 0 SSRF/CWE-22/CWE-78 reachable; URL/path sinks pull from internal data/config, not request params",
        "evidence_type": "actual-application (prior) + manual full-source",
        "evidence": ["DYNAMIC_PROBING.md", "REACHABILITY_MANUAL_VALIDATION.md"],
        "auth_required": "unknown", "auth_basis": "not individually recorded per route",
        "repeatable": "faithful boot documented", "scope": "largest v1 web app",
        "unresolved": "per-route auth not itemized"},
    "ProtossBlaster__leapmotor-mate": {
        "verdict": "REFUTED",
        "applies_to_sink": ["ssrf-url"],   # the hardcoded-host rationale ONLY refutes SSRF, not path-write
        "bound": "URL-fetch sink uses a hardcoded host (overpass-api.de); attacker controls only lat/lon query data, never the target host -> structurally not SSRF. Does NOT address the path-write candidates.",
        "evidence_type": "stubbed boot + manual source",
        "evidence": ["DYNAMIC_PROBING.md"],
        "auth_required": "unknown", "auth_basis": "stubbed boot",
        "repeatable": "boot was stubbed/approximate", "scope": "",
        "unresolved": "boot was not faithful"},
}


# The exact source files the AutoClip fingerprint evidence must cover (repo-relative).
_AUTOCLIP_FP_FILES = {
    "calibration/mined_prompts_v2/ayushdad609-code__autoclip/_repo_src/backend/api/v1/projects.py",
    "calibration/mined_prompts_v2/ayushdad609-code__autoclip/_repo_src/backend/core/path_utils.py",
    "calibration/mined_prompts_v2/ayushdad609-code__autoclip/_repo_src/backend/services/data_sync_service.py",
}


def _rel_to_root(path_str: str) -> str:
    p = Path(path_str)
    try:
        return str((p if p.is_absolute() else ROOT / p).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path_str  # outside ROOT -> keep as-is so it fails the expected-set check


def _fingerprints_ok() -> bool:
    """True only if the manifest is EXACTLY the expected file set and every SHA matches
    the current file. Fails closed on missing/duplicate/unexpected/malformed/mismatched."""
    manifest = (EVID / "source_fingerprints.txt").read_text().splitlines()
    seen = {}
    for line in manifest:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            return False                      # malformed
        h, f = parts
        rel = _rel_to_root(f)
        if rel in seen:
            return False                      # duplicate entry
        seen[rel] = h.lower()
    if set(seen) != _AUTOCLIP_FP_FILES:       # missing or unexpected/substituted
        return False
    for rel, h in seen.items():
        fp = ROOT / rel
        if not fp.exists():
            return False
        if hashlib.sha256(fp.read_bytes()).hexdigest().lower() != h:
            return False                      # mismatch
    return True


def autoclip_verdict() -> dict:
    """APP_CONFIRMED (bounded) ONLY if the saved real-application effect artifacts are present
    and consistent; else INCONCLUSIVE. Verified here, not assumed."""
    need = ["http_response.txt", "server_log_escape.txt", "db_row.txt", "db_row_verification.txt",
            "route_registered.json", "source_fingerprints.txt", "pinned_commit.txt", "path_variant_results.txt"]
    missing = [f for f in need if not (EVID / f).exists()]
    checks = {}
    if not missing:
        http = (EVID / "http_response.txt").read_text()
        log = (EVID / "server_log_escape.txt").read_text()
        dbrow = (EVID / "db_row.txt").read_text()
        checks["http_200"] = "200 OK" in http and '"project_id":".."' in http
        checks["log_escape_parent_data"] = "data/projects/../" in log
        checks["db_row_dotdot"] = "id='..'" in dbrow and "rows with id='..': 1" in dbrow
        # validate ARTIFACT CONTENTS, not just existence
        try:
            rr = json.loads((EVID / "route_registered.json").read_text())
            checks["route_registered_valid"] = (rr.get("path") == "/api/v1/projects/{project_id}/sync-data"
                                                 and rr.get("method") == "POST" and rr.get("registered_in_openapi") is True)
        except Exception:
            checks["route_registered_valid"] = False
        dbv = (EVID / "db_row_verification.txt").read_text().lower()
        checks["db_verification_readonly"] = "read-only" in dbv and "id='..': 1" in dbv
        # source fingerprints: fail closed unless the manifest is EXACTLY the expected
        # file set (one entry each, no missing/dup/substituted), each SHA matching the
        # current file (paths resolved against ROOT). A match establishes agreement with
        # CURRENT source, not which revision the original runtime loaded.
        checks["source_fingerprints_complete_and_match"] = _fingerprints_ok()
    consistent = not missing and all(checks.values())
    base = {
        "candidate_link": "v2:ayushdad609-code__autoclip (get_project_directory(project_id) traversal)",
        "function_level_confirmed_as": "ai_cover (v2 census); same get_project_directory(project_id) path helper reached here via a different route",
        "route": "POST /api/v1/projects/{project_id}/sync-data -> sync_project_from_filesystem -> get_project_directory(project_id)",
        "route_guard": "handler checks only project_dir.exists(); data/projects/.. resolves to parent data/ which exists, so '..' passes (no DB-existence guard, unlike the cover route)",
        "auth_required": "no (on the tested instance)",
        "auth_basis": "route_registered.json: operation_security=null, global_security=null; probe sent no Authorization/cookies. Tested-instance observation; not generalized to deployments",
        "evidence_type": "actual-application (run by the researcher; organized and source-verified here, not re-run)",
        "evidence": [f"evidence/autoclip/{f}" for f in need] + ["evidence/autoclip/route_registered_openapi_raw.txt", "evidence/autoclip/repro.md"],
        "evidence_checks": checks, "missing_artifacts": missing,
        "bound": "one-level escape from data/projects/ into parent data/ + a DB row (id='..', name='项目_..') created. NOT arbitrary FS access, NOT sensitive-data disclosure, NOT RCE, NOT production impact.",
        "repeatable": "not re-run in this session (per constraints). path_variant_results.txt is transcribed from the researcher's earlier terminal output, not re-run.",
        "provenance_limitations": [
            "source_fingerprints.txt was generated by a LATER inspection command: matching those hashes to the current repo source establishes current-source == manifest, but does NOT independently establish which source the ORIGINAL running test loaded. The git commit in pinned_commit.txt (8f32453d...) is a declared pin, not independently verifiable against the vendored _repo_src (no .git in the vendored copy). Do not claim the tested source matches the pinned commit.",
            "http_response.txt header date (04:53:29 GMT) differs from the server_log/db_row event time (04:07:18 GMT): the 200-response capture and the escape+DB-row capture appear to be from different request instances at different times; contents are consistent but a single-request linkage is not proven.",
            "deep / %2f-encoded / absolute-path variants returned 404 in the transcribed results; that caps the DEMONSTRATED depth at one level but is from transcribed output, and a raw-client re-test would be needed to assert the framework rejects deeper forms.",
        ],
    }
    base["dynamic_evidence_status"] = "saved owner-run bundle"
    base["bundle_path"] = "validation/reachability/evidence/autoclip/"
    base["reproduction"] = ("owner-run (boot AutoClip with temp data dir + temp SQLite; "
                            "curl --path-as-is POST /api/v1/projects/../sync-data). NOT re-run in audit; "
                            "re-verify this record's checks with: python3 analysis/build_reach_ledger.py")
    base["next_action"] = ("owner: raw-client re-test of deeper/%2f/absolute variants to bound max depth; "
                           "re-run once to link the 200 and the escape/DB-row to a single request")
    if consistent:
        base["verdict"] = "APP_CONFIRMED"
        base["verdict_note"] = "bounded: route reached + observable effect (parent-dir escape + DB row). Reachability-axis verdict, NOT a severity rating."
    else:
        base["verdict"] = "INCONCLUSIVE"
        base["verdict_note"] = f"evidence not fully present/consistent: missing={missing} checks={checks}"
    return base


def main():
    wl = json.loads((REACH / "REACH_WORKLIST.json").read_text())
    records = []
    for it in wl["items"]:
        rid = it["id"]
        rec = {k: it[k] for k in ("id", "corpus", "repo", "function", "candidate_file", "source_file",
                                   "line", "source_sha256", "pinned_commit", "frozen_label", "exec_cwe",
                                   "sink_category", "confirmed_param", "route_reachable_static",
                                   "origin_class", "call_chain", "route_evidence", "screen_skip",
                                   "generator")}
        rec["static_reachability_limitation"] = "name-based call graph; see REACH_WORKLIST screen_limitations"
        if rid in ADJ:
            rec.update(ADJ[rid]); rec["adjudicated"] = True
        elif (it["repo"] in REPO_PRIOR and it["route_reachable_static"]
              and it["sink_category"] in REPO_PRIOR[it["repo"]]["applies_to_sink"]):
            rp = {k: v for k, v in REPO_PRIOR[it["repo"]].items() if k != "applies_to_sink"}
            rec.update(rp); rec["adjudicated"] = True
        else:
            rec["verdict"] = "INCONCLUSIVE"
            rec["auth_required"] = "unknown"
            rec["evidence_type"] = "static-screen-only"
            rec["bound"] = ("route-reachable by static screen, not adjudicated dynamically"
                            if it["route_reachable_static"]
                            else "no static HTTP entrypoint found (screen negative, not demonstrated)")
            rec["adjudicated"] = False
        # ---- Phase C explicit axes (do not overwrite one evidence axis with another) ----
        rec["function_level"] = {"label": "CONFIRMED", "exec_cwe": rec["exec_cwe"],
                                 "evidence": f"corpus_census_{'pinned' if it['corpus']=='v1' else 'v2'}.json (confirmed=true) + candidate record {it['candidate_file']}"}
        et = rec.get("evidence_type", "static-screen-only")
        rec["dynamic_evidence_status"] = (
            # verdict-driven states take precedence over evidence_type, so the cross-tab
            # rule (REACHABILITY_SUPPORTED / route-reachable-INCONCLUSIVE => owner run) holds
            "additional owner run required" if (rec["verdict"] == "REACHABILITY_SUPPORTED"
                                                or (rec["verdict"] == "INCONCLUSIVE" and it["route_reachable_static"]))
            else "prior documented test" if et.startswith("actual-application (prior")
            else "static/manual only" if ("source-review" in et or "manual full-source" in et)
            else "not tested")
        rec["provenance_status"] = ("source_sha256 recorded; pinned commit declared but not independently "
                                    "verifiable against the vendored _repo_src (no .git)")
        rec["reproducibility_status"] = "static screen rerunnable from frozen census"
        rec["reproduction_command"] = (f"python3 analysis/build_reach_worklist.py  "
                                       f"# regenerates this record from corpus_census_"
                                       f"{'pinned' if it['corpus']=='v1' else 'v2'}.json")
        rec["next_action"] = (
            "none (refuted with candidate-specific evidence)" if rec["verdict"] == "REFUTED"
            else "owner dynamic run to persist end-to-end effect -> APP_CONFIRMED" if rec["verdict"] == "REACHABILITY_SUPPORTED"
            else "owner dynamic run (boot + request) if a verdict is wanted" if (rec["verdict"] == "INCONCLUSIVE" and it["route_reachable_static"])
            else "none required (no static HTTP entrypoint; screen negative)")
        records.append(rec)

    autoclip = autoclip_verdict()

    verdicts = Counter(r["verdict"] for r in records)   # candidate-level ONLY (197 records)
    app_level = [autoclip] if autoclip["verdict"] == "APP_CONFIRMED" else []
    ledger = {
        "note": "Reachability axis only. Function-level CONFIRMED labels are frozen and unchanged. "
                "INCONCLUSIVE is never a negative. No app booted in this session; AutoClip evidence was "
                "produced by the researcher and is organized + source-verified here, not re-run.",
        "counts": wl["counts"],
        "n_candidate_records": len(records),
        "verdict_tally_candidate_level": dict(verdicts),
        "application_level_findings_count": len(app_level),
        "app_confirmed": app_level,
        "app_level_findings_note": "AutoClip is the one APP_CONFIRMED (bounded) finding, via saved real-application "
                                   "evidence. nthplusio is REACHABILITY_SUPPORTED (source-supported, effect not persisted).",
        "records": records,
        "autoclip_record": autoclip,
    }
    (REACH / "REACH_LEDGER.json").write_text(json.dumps(ledger, indent=2))
    render_ledger_md(ledger, REACH / "REACH_LEDGER.md")
    render_reconciliation(ledger, wl, REACH / "RECONCILIATION_REPORT.md")
    print(f"candidate-level verdicts ({len(records)} records): {dict(verdicts)}")
    print(f"application-level findings (separate): {len(app_level)} (AutoClip APP_CONFIRMED)")
    print(f"AutoClip verdict: {autoclip['verdict']}  checks={autoclip.get('evidence_checks')}")
    for f in ("REACH_LEDGER.json", "REACH_LEDGER.md", "RECONCILIATION_REPORT.md"):
        print("wrote", REACH / f)


def render_ledger_md(L, path):
    a = L["autoclip_record"]
    out = ["# VULCAN — reachability ledger (canonical: REACH_LEDGER.json)", "",
           L["note"], "",
           "## Verdict tally (candidate-level, reachability axis)", ""]
    out += [f"- **{k}**: {v}" for k, v in sorted(L["verdict_tally_candidate_level"].items())]
    out += ["", "> INCONCLUSIVE is not a negative. Function-level CONFIRMED is a separate, frozen axis.", "",
            "## APP_CONFIRMED (bounded)", ""]
    if L["app_confirmed"]:
        out += [f"### AutoClip — {a['route']}", "",
                f"- **Verdict:** {a['verdict']} — {a['verdict_note']}",
                f"- **Bound:** {a['bound']}",
                f"- **Guard:** {a['route_guard']}",
                f"- **Auth:** {a['auth_required']} — {a['auth_basis']}",
                f"- **Links to function-level:** {a['function_level_confirmed_as']}",
                f"- **Evidence type:** {a['evidence_type']}",
                "- **Evidence checks:** " + ", ".join(f"{k}={v}" for k, v in a["evidence_checks"].items()),
                "- **Evidence files:** " + ", ".join(a["evidence"]),
                "- **Repeatability:** " + a["repeatable"],
                "- **Provenance limitations:**"]
        out += [f"  - {x}" for x in a["provenance_limitations"]]
    else:
        out += [f"- AutoClip verdict is {a['verdict']}: {a['verdict_note']}"]
    out += ["", "## Route-reachable candidates and their verdicts", "",
            "| id | cwe | sink | origin | verdict | evidence type | bound (short) |",
            "|---|---|---|---|---|---|---|"]
    for r in L["records"]:
        if not r["route_reachable_static"]:
            continue
        out.append(f"| {r['id']} | {r['exec_cwe']} | {r['sink_category']} | {r['origin_class']} | "
                   f"{r['verdict']} | {r.get('evidence_type','')} | {str(r.get('bound',''))[:60]} |")
    out += ["", "## Not route-reachable (static screen) — INCONCLUSIVE, not negative", "",
            f"{sum(1 for r in L['records'] if not r['route_reachable_static'])} confirmed candidates have no "
            "static HTTP entrypoint. Recorded INCONCLUSIVE on app reachability (a screen negative is not a "
            "demonstrated one). See REACH_LEDGER.json."]
    # Dynamically-evidenced cases: ONLY those backed by a saved owner-run bundle.
    out += ["", "## Dynamically-evidenced cases (saved owner-run bundles only)", ""]
    dyn = [a] if a.get("dynamic_evidence_status") == "saved owner-run bundle" else []
    dyn += [r for r in L["records"] if r.get("dynamic_evidence_status") == "saved owner-run bundle"]
    if dyn:
        out += ["| finding | verdict | bundle path | exact bound of what it proves |", "|---|---|---|---|"]
        for d in dyn:
            fid = d.get("route") or d.get("id")
            out.append(f"| {fid} | {d['verdict']} | {d.get('bundle_path','')} | {d['bound']} |")
    else:
        out += ["_none_"]
    out += ["", "Everything else is prior-documented test, static/manual, or not dynamically tested "
            "(see each record's `dynamic_evidence_status`). nthplusio is REACHABILITY_SUPPORTED and needs an "
            "owner bundle to become APP_CONFIRMED."]
    path.write_text("\n".join(out) + "\n")


def render_reconciliation(L, wl, path):
    a = L["autoclip_record"]
    c = wl["counts"]
    out = ["# Reconciliation report — reachability axis (2026-10-27)", "",
           "What changed, what each verdict rests on, and what remains for the researcher to test. "
           "No app was booted in this session; no frozen function-level label was altered.", "",
           "## 1. AutoClip — APP_CONFIRMED (bounded)", "",
           "**Change from the prior session's static note (REFUTED):** the earlier REFUTED was about the "
           "`.../cover/ai` route, which has a `project_or_404` DB guard. The saved evidence is for a DIFFERENT "
           "route, `POST /api/v1/projects/{project_id}/sync-data`, which has **no** DB-existence guard — only "
           "`project_dir.exists()`, and `data/projects/..` resolves to the parent `data/` which exists, so "
           "`project_id='..'` passes. The `/sync-data` route **does** exist in source (projects.py:386), "
           "correcting my earlier 'route does not exist' statement (I had searched the cover route).", "",
           "Verified here (not re-run): " + ", ".join(f"{k}={v}" for k, v in a["evidence_checks"].items()) + ".",
           "", "**Bound:** " + a["bound"], "",
           "**Source-to-pin limitation:** " + a["provenance_limitations"][0],
           "", "**Other provenance notes:**"]
    out += [f"- {x}" for x in a["provenance_limitations"][1:]]
    out += ["", "## 2. nthplusio send_webhook — REACHABILITY_SUPPORTED", "",
            "TAXONOMY.md previously read APP_CONFIRMED; REACHABILITY_MANUAL_VALIDATION.md read "
            "REACHABILITY_SUPPORTED with the end-to-end run deferred. No saved end-to-end artifact "
            "(callback/outbound-fetch capture) exists in the repo. Corrected TAXONOMY.md to "
            "REACHABILITY_SUPPORTED with a footnote. Route->sink SSRF is source-supported; the effect is "
            "not persisted. Fixture-scope code. To reach APP_CONFIRMED: boot the fixture + loopback listener (researcher).",
            "", "## 3. Three V1 attacker-flagged candidates (source review)", "",
            "- **nthplusio send_webhook:** REACHABILITY_SUPPORTED (above).",
            "- **quantam101 digest:** REFUTED — the screened chain is a FALSE call-graph edge (name collision "
            "between `stable_embedding`'s local `digest` variable and the module function `digest(path)`).",
            "- **trsdn write_json:** REFUTED — no HTTP entrypoint; argparse CLI tool; the screen's 'req-arg' "
            "flag matched a parameter coincidentally named `request`.",
            "- Net: 2 of 3 attacker-flags were static-screen false positives; 1 genuine (fixture-scope).",
            "", "## 4. Civitas & leapmotor — REFUTED (retained)", "",
            "Retained from prior artifacts, each for the specific hypothesis it refuted: civitas (faithful "
            "boot, 114 routes, 0 reachable across three sink classes); leapmotor (hardcoded SSRF host; stubbed boot).",
            "", "## 5. V2 residuals — REACHABILITY_V2.md", "",
            "The 3 route-reachable v2 SQL helpers and `ai_cover` are REFUTED/INCONCLUSIVE there (constant SQL "
            "with `?` placeholders; cover route 404s on missing project). The 15 v2 route-reachable screen "
            "hits otherwise stay INCONCLUSIVE dynamically. AutoClip's sync-data route is the one app-level "
            "finding and is APP_CONFIRMED above.",
            "", "## 6. Reconciled counts (defined units)", "",
            f"- v1: {c['v1_confirmed_candidate_rows']} candidate rows / {c['v1_source_unique']} source-unique / 149 name-dedup.",
            f"- v2: {c['v2_confirmed_candidate_rows']} rows / {c['v2_source_unique']} source-unique.",
            f"- overlap: {c['repo_overlap_v1_v2'] or 'none (disjoint)'}; combined {c['combined_candidate_rows']} rows / {c['combined_source_unique']} source-unique.",
            "- static route-reachable: v1 "
            f"{L['counts'] and wl['route_reachable_static']['v1']}, v2 {wl['route_reachable_static']['v2']}.",
            f"- reachability verdicts (candidate-level): {L['verdict_tally_candidate_level']}.",
            "- The '34' twin pairs are an RQ2 unit (vuln/fixed), not vulnerabilities; not added here.",
            "- INCONCLUSIVE counts are NOT negatives and no prevalence is inferred from this selected worklist.",
            "", "## 7. What remains for the researcher to test", "",
            "- nthplusio: boot the fixture + loopback listener to persist the SSRF effect (REACHABILITY_SUPPORTED -> APP_CONFIRMED).",
            "- AutoClip: a raw-client re-test of deeper/%2f/absolute variants to bound max traversal depth; "
            "re-run once to confirm stability and link the 200 and the escape+DB-row to a single request.",
            "- The 15 v2 (minus adjudicated) + any v1 route-reachable INCONCLUSIVE items: Level-2 dynamic tests, if desired.",
            "- The two new CWE-22 leads (quantam101, trsdn) do NOT warrant dynamic tests (refuted statically)."]
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
