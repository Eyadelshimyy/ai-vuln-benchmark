# Execution-verified vulnerable/fixed twin pairs (RQ2 pairwise evaluation)

These pairs test whether a detector tells a vulnerable function apart from its fixed
version, in the spirit of JitVul's pairwise accuracy. The difference: our "fixed" side is
**verified by execution** (the oracle no longer confirms it) rather than taken from a fix
commit.

- `../../analysis/make_twins.py` builds the pairs.
- `../../analysis/verify_twins.py` applies the validity rule: the original still CONFIRMS
  AND the twin is executable AND does NOT confirm, under the identical oracle and sandbox.

## Eligibility (and what was deliberately not twinned)

| class | twinned | fix |
|---|---|---|
| CWE-918 SSRF (30) | yes | generic guard text-inserted at the top of the body; every other character unchanged |
| CWE-22 COMPOSED (4, hand-reviewed: `../rq1/cwe22_composed_manual.json`) | yes | hand-written: name must be a single path component, or the resolved path must stay under its base (`patches/`) |
| CWE-22 DIRECT_CALLER_PATH (152) | no | opening the caller's path is the function's contract; the correct fix belongs at the caller |
| CWE-78 / CWE-89 helpers (7) | no | behaviour-coupled: the parameter IS the whole command or query, so no fix stops the exploit without removing the function's purpose (cf. `execute_bash` in `../COUNTERFACTUAL_AUDIT_RESULTS.md`) |

## Result (sandboxed census `census_twins.json`)

**34/34 pairs valid.** All originals still confirm, and all twins execute and do not
confirm. All 34 twins still TRIGGER: the input still reaches the sink, but the fix blocks
the exploit. This is why twins are scored only pairwise and never pooled with the
NOT_TRIGGERED negatives.

**Failed first attempt (kept, not hidden):** the first SSRF guard refused unresolvable hosts
and non-http(s) values. That changed the functions' behaviour on ordinary inputs, so 30/30
SSRF twins were not executable (`twins_verified_guard_v1_FAILED.json`). Guard v2 refuses
only `file://` and hosts that RESOLVE to non-public addresses, and falls through to the
original code otherwise.

**Sanity check:** a keyword-matching mock detector scores 0/34 pair accuracy. It flags
both sides, which is exactly what the metric exists to expose.

`fixed/` holds the 34 fixed twin records referenced by `../rq2/rq2_labelset_v3.json`.
