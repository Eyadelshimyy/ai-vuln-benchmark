# VULCAN — Phase A verification (claims → artifacts → status)

Generated 2026-10-27 by inspecting artifact **contents** (not just presence). Read-only; nothing
was fixed in this phase. Status legend:

- **COMMITTED_AND_VERIFIED** — committed, contents substantiate the claim.
- **PRESENT_MODIFIED_OR_UNTRACKED** — exists in the working tree, modified/untracked (noted); says what it shows.
- **MISSING** — no artifact.
- **PRESENT_BUT_INSUFFICIENT** — exists but does not fully substantiate the claim (says what is missing).

| # | Claim | Artifact | Status | What the contents show |
|---|---|---|---|---|
| 1 | v1 funnel 2032→1501→289→160 | `corpus_census_pinned.json` | **COMMITTED_AND_VERIFIED** | rows 2032, executable 1501, triggered 289, confirmed 160 — exact. |
| 2 | v1 160 confirmed by exec_cwe = CWE-22 130 / CWE-78 4 / CWE-918 26 | `corpus_census_pinned.json`, `MANIFEST.json` | **COMMITTED_AND_VERIFIED** | Both agree: 130/4/26 = 160. **Unit = candidate rows.** |
| 3 | v1 = 149 functions | derived from pinned census | **COMMITTED_AND_VERIFIED** | name-dedup (repo,function) = **149**; source-unique (repo,sha256) = **156**; candidate rows = 160. Three distinct units. |
| 3b | v1 function-level by exec_cwe = CWE-22 122 / CWE-918 24 / CWE-78 3 | `validation/TAXONOMY.md` | **PRESENT — UNIT MISMATCH TO STATE** | TAXONOMY's 122/24/3 (=149) is the *function-level* split; MANIFEST's 130/4/26 (=160) is the *candidate-level* split. Both correct for their unit; Phase B must state the unit beside each. |
| 4 | v2 37 confirmed by exec_cwe = CWE-22 30 / CWE-918 4 / CWE-89 3 | `corpus_census_v2.json` | **PRESENT_MODIFIED_OR_UNTRACKED** (untracked; new this session) | Contents: 37 confirmed = 30/4/3. Substantiates the claim; not yet committed. |
| 5 | Clean-clone reproduces 160 exactly (0 dropped / 0 added) | `corpus_census_reproduced.json`, `validation/REPRODUCE.md` | **COMMITTED_AND_VERIFIED** | Reproduced confirmed set = 160 and is set-equal to the pinned confirmed set (dropped 0, added 0). |
| 6 | Patch audit: 3 REVERSED / 1 BEHAVIOR_COUPLED | `validation/COUNTERFACTUAL_AUDIT_RESULTS.md` | **COMMITTED_AND_VERIFIED** | Table: fetch_json REVERSED, spdx_ids REVERSED, sha256_file REVERSED, execute_bash BEHAVIOR_COUPLED; reversal rate 3/3 among evaluable. |
| 6b | Patch-audit harness reproduces the pilot | `validation/counterfactual/audit_case.py`, `patch_*.py` | **PRESENT_MODIFIED_OR_UNTRACKED** (untracked; recovered this session) | Re-ran this session: fetch_json/spdx_ids/sha256_file REVERSED, execute_bash NOT_REVERSED — matches the results doc. Needs committing to be reproducible from a clean clone. |
| 7 | Anti-circularity control PASSES (independent /proc observable) | `validation/ANTICIRCULARITY_CONTROL_RESULTS.md` | **COMMITTED_AND_VERIFIED** | Arms A/B/C: honest-vuln CONFIRMED+LIVE, wrong-patch NOT_CONFIRMED+LIVE, real-fix NOT_CONFIRMED+GONE. "Control PASSES." |
| 8 | External control CVE-2024-34359 corroborates the oracle | `validation/external-control/CVE-2024-34359_cross_check.md` | **COMMITTED_AND_VERIFIED** (one gap, below) | Doc describes using CVE-Bench's unmodified container + grader (SSTI→RCE in llama-cpp-python). **Gap:** the raw grader transcript is not persisted as a separate artifact; the claim rests on the documented run. |
| 9 | Reachability: 1 APP_CONFIRMED, 1 REACHABILITY_SUPPORTED, 16 REFUTED, 180 INCONCLUSIVE | `validation/reachability/REACH_LEDGER.json` | **PRESENT_MODIFIED_OR_UNTRACKED** (untracked; new this session) | Candidate tally (197 records): INCONCLUSIVE 180, REFUTED 16, REACHABILITY_SUPPORTED 1; application-level findings = 1 (AutoClip APP_CONFIRMED). |
| 9a | AutoClip APP_CONFIRMED (bounded) | `validation/reachability/evidence/autoclip/` | **PRESENT_MODIFIED_OR_UNTRACKED** (untracked; owner-delivered bundle) | 8 evidence files present; 6/6 builder checks pass (http 200 + project_id="..", log escape into parent data/, DB row id="..", route_registered valid, db read-only, fingerprints complete+match). Bound: one-level escape + DB row. Provenance caveats recorded (original runtime revision unproven; 200 vs escape not single-request-linked; deep variants transcribed). |
| 9b | nthplusio = APP_CONFIRMED | `TAXONOMY.md` vs `REACHABILITY_MANUAL_VALIDATION.md` | **PRESENT_BUT_INSUFFICIENT for APP_CONFIRMED** | No persisted end-to-end artifact (callback/outbound-fetch capture) exists. Route→sink is source-supported only. TAXONOMY already reconciled this session to REACHABILITY_SUPPORTED with a footnote; manual log said "deferred". Correct label = REACHABILITY_SUPPORTED; needs an owner evidence bundle for APP_CONFIRMED. |
| 9c | Leapmotor path-write REFUTED | ledger + `DYNAMIC_PROBING.md` | **PRESENT_BUT_INSUFFICIENT → corrected to INCONCLUSIVE** | The repo prior only refutes the SSRF hypothesis (hardcoded host); it does not address the two CWE-22 `_write_bytes` path-write candidates. Ledger now holds them INCONCLUSIVE. |
| 9d | 3 v2 SQL helpers REFUTED | `validation/corpus_v2/REACHABILITY_V2.md` | **COMMITTED_AND_VERIFIED** | Candidate-specific: callers pass constant SQL with `?` placeholders; request data is only a bound parameter. |

## Counting-unit summary (to state consistently in Phase B)

| unit | v1 | v2 | combined |
|---|---|---|---|
| candidate rows | 160 | 37 | 197 |
| source-unique functions | 156 | 37 | 193 |
| name-dedup functions | 149 | — | — |
| RQ2 label-set positives | — | — | 193 (source-unique) + 386 neg + 34 twin pairs |

v1/v2 repo sets are disjoint (overlap 0). The "34" is RQ2 twin *pairs* (vuln/fixed), not vulnerabilities.

## Findings to carry into Phase B/C (recorded, not fixed here)

1. **exec_cwe unit mismatch (3b):** MANIFEST 130/4/26 (candidate) vs TAXONOMY 122/24/3 (function). Not an error — state the unit beside each wherever it appears.
2. **nthplusio (9b):** downgrade stands at REACHABILITY_SUPPORTED; flag that APP_CONFIRMED needs an owner evidence bundle.
3. **CVE-2024-34359 (8):** raw grader transcript not persisted separately; consider saving it, or label the claim as "documented external run".
4. **Untracked load-bearing artifacts:** `corpus_census_v2.json`, `REACH_LEDGER.*`, `validation/counterfactual/*.py`, the AutoClip evidence bundle, and this session's new scripts are untracked. Present and verified, but not yet reproducible from a clean clone until committed (owner's call; no commit in these phases).
5. **CWE-89 (for Phase B):** v2 shows 3 confirmed CWE-89 (whole-statement raw-SQL execute/fetch helpers); v1 shows 0. Verify against v2 execution evidence before updating TAXONOMY to explain CWE-89 confirmability + the behavior-coupled limitation.

## Function-level frozen labels and census rows: UNCHANGED in this phase.
