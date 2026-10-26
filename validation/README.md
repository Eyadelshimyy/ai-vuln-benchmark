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
```

## Headline (function-level, canonical unit)
- 149 CONFIRMED functions. Reachability split: 1 APP_CONFIRMED, 10 REFUTED, 66 STATIC_LEAD (unverified), 72 UNREACHABLE.
- Gold set (confirmed ∩ reachability-signal) = 77 distinct functions (the earlier "90" was duplicate-inflated).
- Exactly ONE demonstrated end-to-end attacker-reachable vuln (a stored SSRF, fixture-scope).
- One external-control APP_CONFIRMED (CVE-2024-34359) corroborating the oracle.

## What is NOT done yet (open review items)
- AI-corpus re-run with the improved lifecycle-aware SSTI oracle (can move primary numbers).
- Counterfactual patch audit (2×2 ablation) — defends what the CONFIRMED labels mean.
- CNE characterization + optional FIXTURE_CONFIRMED study (census_why.json is the input).
