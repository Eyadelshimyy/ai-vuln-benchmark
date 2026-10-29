# VULCAN: Execution-Verified AI-Generated Vulnerabilities

**VULCAN** means **VULnerability Confirmation via Actual executioN**. It evaluates AI-attributed code using deterministic execution evidence rather than treating a static alert as proof of exploitability.

This repository supports the bachelor's thesis *An Empirical Evaluation and Benchmark of AI-Generated Vulnerabilities in Code* (GIU, Cybersecurity).

## Current research state

### RQ1 — Execution-confirmed vulnerability corpus

The benchmark uses an execution oracle: a candidate is confirmed only when a deterministic exploit produces an observable security-relevant effect. A candidate that cannot be executed is **inconclusive**, not safe.

- **V1 frozen baseline:** 160 confirmed injection-point records, corresponding to 149 name-deduplicated functions.
- **V2 frozen baseline:** 37 confirmed candidates among 560 records.
- **V2 by weakness:** CWE-22: 30; CWE-89: 3; CWE-918: 4.
- V1 and V2 use different corpus versions and counting units. Do not combine their counts.

### Reachability

Exploitability and application reachability are evaluated separately. A function-level confirmation does not by itself prove that an external attacker can reach the vulnerable sink through an application's entry point. Reachability outcomes distinguish supported, refuted, and inconclusive cases; inconclusive cases are not counted as negatives.

### RQ2 — Detector evaluation

The frozen RQ2 protocol and label sets are under `validation/rq2/`. The label policy compares detector predictions with execution-derived reference labels. Confirmed positives are separated from executed, non-triggering negatives; inconclusive cases are excluded from the binary score and reported separately.

Do not interpret a protocol, dry run, or mock output as a completed detector evaluation. Report model-specific metrics only when backed by the corresponding real run artifacts.

## Reproducing the frozen corpora

Run from the repository root with the dependencies required by the scripts installed.

**V1 census** (the execution environment affects candidates with optional dependencies):

```bash
python3 analysis/corpus_census.py --tier2 --out /tmp/v1.json
python3 tier2/test_soundness.py
```

The frozen V1 result is 160 confirmed records in the documented OpenCV-absent execution profile. With OpenCV available, an additional SSRF candidate may execute, so record the environment alongside any reproduction.

**V2 public-only census:**

```bash
python3 analysis/corpus_census.py --tier2 --root calibration/mined_prompts_v2 --out /tmp/v2.json
```

The published candidate records reproduce 33 of the 37 V2 confirmations (CWE-22: 26; CWE-89: 3; CWE-918: 4). Four confirmations depend on third-party source trees that are not bundled because their redistribution status is unclear. The canonical V2 baseline remains 37; the public-only run should not be described as reproducing all 37.

For the full procedure, execution-profile details, evidence links, and known reproduction differences, see `validation/REPRODUCE.md` and `validation/EVIDENCE_INDEX.md`.

## Evidence and protocol

- `corpus_census_pinned.json`, `corpus_census_reproduced.json` — frozen V1 census and reproduction.
- `corpus_census_v2.json` — frozen V2 census.
- `calibration/mined_prompts/`, `calibration/mined_prompts_v2/` — candidate records.
- `PREDICATES.md`, `validation/TAXONOMY.md` — confirmation predicates, labels, and counting units.
- `validation/reachability/` — reachability ledger and supporting evidence.
- `validation/rq2/RQ2_PREREGISTRATION.md` — frozen detector-evaluation protocol.
- `validation/rq2/rq2_labelset_v3.json` — current RQ2 label set.
- `validation/TIER2_COVERAGE_AUDIT.md` — confirmation strategy coverage.
- `validation/EVIDENCE_INDEX.md` — claim-to-artifact index.
- `validation/REPRODUCE.md` — reproduction instructions.

## Interpretation and limitations

- The corpus is **sink-selected**: candidates were mined because they contain relevant security sinks. It is not a random sample of all AI-generated code.
- **Function-level exploitability is conditional.** Real-world impact also depends on attacker-controlled input reaching the function.
- `COULD_NOT_EXECUTE` means the experiment could not establish a verdict; it does not mean safe.
- The public V2 reproduction has source-availability limits described above.
- Counts must always state their unit (candidate record, function, or injection point).

## License

Repository code is licensed under MIT unless a file states otherwise. Third-party source code and candidate repositories retain their original licenses; their presence in research metadata does not grant redistribution rights.
