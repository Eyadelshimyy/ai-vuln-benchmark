# VULCAN evidence index (Phase E)

Maps each headline claim to its artifact, reproduction command, status, and limitations. Status
vocabulary: **COMMITTED+VERIFIED**, **UNTRACKED/MODIFIED** (present, not yet tracked), **MISSING**,
**INSUFFICIENT**, **REPRODUCIBLE-UNDER-CONDITIONS**. Canonical machine-readable form:
`ARTIFACT_MANIFEST.json`.

| # | Claim / result | Artifact(s) | Reproduce | Status | Limitations / next action |
|---|---|---|---|---|---|
| 1 | V1 funnel 2032→1501→289→**160** confirmed; 149 name-deduped | `corpus_census_pinned.json`, `MANIFEST.json` | `python3 analysis/corpus_census.py --tier2` (clean clone) | COMMITTED+VERIFIED | by-class unit stated in TAXONOMY COUNTING-UNIT KEY |
| 2 | V2 **37** confirmed (CWE-22 30 / CWE-918 4 / **CWE-89 3**) | `corpus_census_v2.json` | `corpus_census.py --tier2 --root calibration/mined_prompts_v2` | UNTRACKED (new); clean-checkout repro **BLOCKED** (inputs absent from pinned revision) | version v2 corpus + records, then tracked-only clean-checkout census to test self-containment vs `_repo_src` need |
| 2b | CWE-89 confirmable (3 v2, 0 v1) via raw-SQL execute/fetch sinks | `corpus_census_v2.json` tier2_evidence; `tier2/strategies.py::_SqlInjection` | inspect proofs (`vulcan_proof()` sentinel) | COMMITTED+VERIFIED | behavior-coupled (param IS the SQL); reachability REFUTED |
| 3 | Clean-clone reproduces 160 exactly (0 drop/0 add) | `corpus_census_reproduced.json`, `REPRODUCE.md` | `git clone --branch vulcan-freeze-149-rc` then census | COMMITTED+VERIFIED | reproduction floor; see execution-profile boundary |
| 4 | Tier-2 soundness: 0 FP / 0 missed | `tier2/test_soundness.py` | `python3 tier2/test_soundness.py` | COMMITTED+VERIFIED | — |
| 5 | Counterfactual patch audit: 3 REVERSED / 1 BEHAVIOR_COUPLED | `COUNTERFACTUAL_AUDIT_RESULTS.md`, `counterfactual/audit_case.py`+`patch_*.py` | `python3 validation/counterfactual/audit_case.py <cand> <patch>` | HARNESS UNTRACKED | commit harness for clean-clone reproducibility |
| 6 | Anti-circularity control PASSES | `ANTICIRCULARITY_CONTROL_RESULTS.md`, `anticirc_control.py` | `python3 validation/anticirc_control.py` | COMMITTED+VERIFIED | — |
| 7 | External CVE-2024-34359 corroborates oracle | `external-control/CVE-2024-34359_cross_check.md` | CVE-Bench container+grader (owner-run) | DOCUMENTED EXTERNAL RUN | raw grader transcript not persisted (INSUFFICIENT for re-run) |
| 8 | Canonical reachability ledger (197 records) | `reachability/REACH_LEDGER.json`/`.md`, `REACH_WORKLIST.json` | `python3 analysis/build_reach_worklist.py && python3 analysis/build_reach_ledger.py` | UNTRACKED (new) | commit builders + ledger |
| 8b | Source reviews (nthplusio/quantam101/trsdn) | `reachability/source_reviews/*.md`, `SOURCE_REVIEWS.md` | static | UNTRACKED (new) | — |
| 9 | AutoClip APP_CONFIRMED (bounded) via sync-data | `reachability/evidence/autoclip/` (owner bundle) | owner-run; re-verify `python3 analysis/build_reach_ledger.py` (6/6 checks) | OWNER-DELIVERED, VERIFIED | one app-level finding; `ai_cover` REFUTED (own route); provenance caveats recorded |
| 9b | nthplusio REACHABILITY_SUPPORTED | `reachability/source_reviews/nthplusio_send_webhook.md`, TAXONOMY | static | COMMITTED+VERIFIED | needs owner end-to-end bundle for APP_CONFIRMED |
| 10 | Supplemental: fret (CWE-918), gesh75 (CWE-22) | `supplemental/SUPPLEMENTAL_FINDINGS.md`, `supplemental/*/phaseD_census_record.json` | see supplemental doc | UNTRACKED (new) | outside frozen 160; env/source-dependent |
| 11 | RQ2 detector evaluation | `rq2/RQ2_PREREGISTRATION.md`, `rq2/rq2_labelset_v3.json`, `models_panelA.json` | NOT RUN | NOT AUTHORIZED/RUN | frozen label set ready; awaits owner key + approval |

## Remaining-work classification (Phase E §9)

**Reproducible from frozen inputs (clean clone):** V1 160 funnel/confirmations; Tier-2 soundness;
anti-circularity; reachability worklist/ledger regeneration (from committed censuses); TAXONOMY counts.

**Requires untracked or owner-delivered inputs:** V2 census (untracked v2 corpus + `_repo_src`);
patch-audit harness (untracked); reachability ledger builders (untracked); AutoClip evidence bundle
(owner-delivered, untracked); supplemental fret/gesh75; gesh75 `_repo_src` (untracked).

**Existing owner-run dynamic evidence:** AutoClip `sync-data` APP_CONFIRMED bundle; CVE-2024-34359
documented external run (no raw transcript).

**Further owner-run dynamic tests required:** nthplusio end-to-end (to reach APP_CONFIRMED); AutoClip
deeper/%2f/absolute path variants + single-request linkage; the 42 route-reachable INCONCLUSIVE
candidates if verdicts are wanted.

**RQ2 not authorized:** no detector model run has been authorized or executed; panel/protocol frozen,
budget capped, awaiting owner key + go.

**RQ3 / repair scope:** undecided / out of scope for this package.
