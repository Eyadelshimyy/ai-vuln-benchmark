# Tier-2 strategy coverage audit — all 8 registered strategies (Phase pre-F)

`STRATEGIES` registers: CWE-22, CWE-78, CWE-89, CWE-94, CWE-502, CWE-611, CWE-918, CWE-1336.
**Invocation** = a candidate triggered at Tier-1 with `exec_cwe == class` (the census runs the
`exec_cwe` strategy). A CWE **tag** does NOT imply invocation. Canonical data: `TIER2_COVERAGE_AUDIT.json`.
Census artifacts: V1 `corpus_census_pinned.json` (committed); V2 `corpus_census_v2.json` (untracked/new);
invocation detail for EXERCISED_ZERO classes from Phase D `census_v1.json` `tier2_evidence`.

## Four coverage states
1. **REGISTERED_AND_CONFIRMED** — invoked, >=1 qualifying confirmation.
2. **REGISTERED_AND_EXERCISED_ZERO_CONFIRMED** — invoked on N identified candidates, 0 confirmed (state N).
3. **REGISTERED_BUT_NOT_EXERCISED** — strategy exists, no candidate shown to invoke it.
4. **NO_STRATEGY** — none registered (not applicable here; all 8 exist).

## V1 — corpus_census_pinned.json (committed)

| CWE | tagged | executable | invoked (trig_as) | confirmed | CNE | state |
|---|---|---|---|---|---|---|
| CWE-22 | 747 | 566 | 167 | 130 | 181 | REGISTERED_AND_CONFIRMED |
| CWE-78 | 451 | 314 | 65 | 4 | 137 | REGISTERED_AND_CONFIRMED |
| CWE-89 | 598 | 449 | 3 | 0 | 149 | REGISTERED_AND_EXERCISED_ZERO_CONFIRMED |
| CWE-94 | 0 | 0 | 0 | 0 | 0 | REGISTERED_BUT_NOT_EXERCISED |
| CWE-502 | 6 | 5 | 0 | 0 | 1 | REGISTERED_BUT_NOT_EXERCISED |
| CWE-611 | 101 | 86 | 3 | 0 | 15 | REGISTERED_AND_EXERCISED_ZERO_CONFIRMED |
| CWE-918 | 197 | 128 | 51 | 26 | 69 | REGISTERED_AND_CONFIRMED |
| CWE-1336 | 0 | 0 | 0 | 0 | 0 | REGISTERED_BUT_NOT_EXERCISED |

## V2 — corpus_census_v2.json (UNTRACKED/new)

| CWE | tagged | executable | invoked (trig_as) | confirmed | CNE | state |
|---|---|---|---|---|---|---|
| CWE-22 | 217 | 146 | 34 | 30 | 71 | REGISTERED_AND_CONFIRMED |
| CWE-78 | 148 | 90 | 28 | 0 | 58 | REGISTERED_AND_EXERCISED_ZERO_CONFIRMED |
| CWE-89 | 152 | 117 | 5 | 3 | 35 | REGISTERED_AND_CONFIRMED |
| CWE-94 | 0 | 0 | 0 | 0 | 0 | REGISTERED_BUT_NOT_EXERCISED |
| CWE-502 | 2 | 2 | 0 | 0 | 0 | REGISTERED_BUT_NOT_EXERCISED |
| CWE-611 | 2 | 1 | 0 | 0 | 1 | REGISTERED_BUT_NOT_EXERCISED |
| CWE-918 | 56 | 39 | 15 | 4 | 17 | REGISTERED_AND_CONFIRMED |
| CWE-1336 | 0 | 0 | 0 | 0 | 0 | REGISTERED_BUT_NOT_EXERCISED |

## Notes
- **invocation_definition:** A Tier-2 strategy is INVOKED only when a candidate triggers at Tier-1 with exec_cwe == that class (census runs the exec_cwe strategy). Tag count != invocation.
- **CWE-502:** 0 invocations in v1 and v2 -> REGISTERED_BUT_NOT_EXERCISED. The 3 CWE-502-tagged confirmed rows confirm via CWE-22 (exec_cwe=CWE-22), NOT via the deserialization strategy.
- **CWE-611:** v1: 3 invocations, all diag EXECUTED_NO_SINK, 0 confirmed (tested denominator = 3, not the 101 tagged). v2: 0 invocations.
- **CWE-78:** v2: 28 invocations, 0 confirmed (EXERCISED_ZERO, denominator 28). v1: 65 invoked, 4 confirmed.
- **CWE-89:** v1: 3 invoked, 0 confirmed (EXERCISED_ZERO). v2: 5 invoked, 3 confirmed.
- **CWE-94:** 0 tagged candidates in the main corpus -> NOT_EXERCISED (no candidate). Strategy exists; exercised only as a soundness-battery control, not in the frozen census.
- **CWE-1336:** 0 tagged candidates in the MAIN census (the ai-ssti/llama-cpp-ssti SSTI sub-study is excluded by corpus_census.py). CWE-1336 is exercised+confirmed in that excluded sub-study and in tier2/test_soundness.py, but is UNKNOWN/out-of-scope for the frozen main V1/V2 counts.
- **confirmed_reconciliation:** v1 confirmed by class: CWE-22=130,CWE-78=4,CWE-918=26 (=160). v2: CWE-22=30,CWE-89=3,CWE-918=4 (=37). Matches frozen.
