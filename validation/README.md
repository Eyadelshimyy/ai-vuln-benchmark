# validation/ — VULCAN evidence & validity artifacts

One home for everything produced in the validation pass. Read TAXONOMY.md first.

```
validation/
  TAXONOMY.md                         Canonical two-axis labels (Exploitability × Reachability)
                                      + Units reconciliation (candidate 2032 / function 149 confirmed / RQ2 label-set 51).
  reachability/
    REACHABILITY_MANUAL_VALIDATION.md Full gold-set hand-validation log (12 HTTP candidates traced;
                                      api class sampled; full-source verification at pinned commits).
    classify_corpus.py                Two-axis classifier (joins census + worklist + manual verdicts).
    corpus_classification.json        Per-function labels for the whole corpus (headline = function-level).
  external-control/
    CVE-2024-34359_cross_check.md     Independent corroboration via CVE-Bench's unmodified grader.
  coverage-gap/
    census_why.json                   Root-cause breakdown of the NOT_EXECUTABLE functions.
  rq1/                                RQ1: which weaknesses does each AI generator produce?
    RQ1_BY_GENERATOR.md / .json       Funnel + confirm rate per agent (repo-cluster CIs), per model
                                      where a commit trailer names it, within-repo comparison.
    cwe22_subtypes.json               DIRECT_CALLER_PATH vs BASE_JOIN_OVERRIDE per CWE-22 row.
    commit_model_attribution.json     Full trailer/date/model for 875/896 AI commits.
  rq2/                                RQ2: AI models as detectors (prepared; NOT yet run)
    RQ2_PREREGISTRATION.md            Frozen population, conditions, analysis plan, owner setup steps.
    rq2_labelset_v2.json              Frozen label set: 156 pos / 312 neg, source-unique, hashed.
```

Regenerate (no execution, no model calls):

    python3 analysis/rq1_by_generator.py --census corpus_census_reproduced.json \
        --models-attr validation/rq1/commit_model_attribution.json --out-dir validation/rq1
    python3 analysis/cwe22_subtype.py --census corpus_census_reproduced.json --out validation/rq1/cwe22_subtypes.json
    python3 analysis/test_rq2_pipeline.py && python3 analysis/test_consortium.py && python3 analysis/test_freeze_manifest.py

## Headline (function-level, canonical unit)
- 160 confirmed census rows = 156 source-unique functions = 149 name-deduplicated (see TAXONOMY "Correction").
  125/130 CWE-22 confirmations are DIRECT_CALLER_PATH (conditional on an untrusted caller).
- 149 CONFIRMED functions (name-deduplicated). Reachability split: 1 APP_CONFIRMED, 10 REFUTED, 66 STATIC_LEAD (unverified), 72 UNREACHABLE.
- Gold set (confirmed ∩ reachability-signal) = 77 distinct functions (the earlier "90" was duplicate-inflated).
- Exactly ONE demonstrated end-to-end attacker-reachable vuln (a stored SSRF, fixture-scope).
- One external-control APP_CONFIRMED (CVE-2024-34359) corroborating the oracle.

## What is NOT done yet (open review items)
- AI-corpus re-run with the improved lifecycle-aware SSTI oracle (can move primary numbers).
- Counterfactual patch audit (2×2 ablation) — defends what the CONFIRMED labels mean.
- CNE characterization + optional FIXTURE_CONFIRMED study (census_why.json is the input).
