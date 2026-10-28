# Pre-Phase-F verification (2026-10-28) — two read-only checks

No census rerun, no dynamic tests, no label changes, no commits. Check 1 uses existing census
records; Check 2 uses a `git archive` clean checkout of the pinned revision.

---

## Check 1 — CWE-502 / CWE-611 execution coverage

**Strategies exist.** `tier2/strategies.py` registers both:
`STRATEGIES = {... CWE502, CWE611 ...}` (`_Deserialization` for CWE-502, `_XXE` for CWE-611).

**How confirmation works (so "eligible" is read correctly):** the census runs Tier-2 only for a
candidate that **triggered** at Tier-1, and invokes the strategy for the class that fired
(`exec_cwe`). So a CWE-502/611 strategy is *invoked* only when a candidate **triggers as that class**
(`exec_cwe == CWE-502/611`), not merely when a candidate carries the tag.

### Coverage table (artifacts: `corpus_census_pinned.json` (v1), `corpus_census_v2.json` (v2); strategy-invocation detail from Phase D `census_v1.json` `tier2_evidence`)

| CWE | corpus | eligible (tag) | executable | triggered AS this class (strategy invoked) | confirmed | strategy actually exercised? |
|---|---|---|---|---|---|---|
| CWE-502 | v1 | 6 | 5 | **0** | 0 | **No — never invoked** |
| CWE-502 | v2 | 2 | 2 | **0** | 0 | **No — never invoked** |
| CWE-611 | v1 | 101 | 86 | **3** | 0 | **Yes, on 3** (diag `EXECUTED_NO_SINK`) |
| CWE-611 | v2 | 2 | 1 | 0 | 0 | No — never invoked |

Supporting detail:
- The 3 CWE-611 v1 triggered candidates (all `kamoras__civitas` pipeline funcs) show in the Phase D
  census: `params_tried=['xml_text']/['body']/['member_data']`, `diag=EXECUTED_NO_SINK`, confirmed=False.
  The XXE strategy ran; the payload did not reach/observe an XXE sink effect.
- The 3 CWE-502-**tagged** "confirmed" rows confirm via **CWE-22** (`exec_cwe=CWE-22`), not via the
  deserialization strategy (verified in `corpus_census_reproduced.json`). No candidate triggered as 502.

### Classification (per the three categories requested)
- **CWE-502 (v1 and v2): category 3-adjacent → "strategy exists, eligible candidates existed, but
  the strategy was never invoked"** (0 triggered as 502; the deserialization sink was never reached
  with taint). This is **NOT a tested negative.**
- **CWE-611 v1: category 1 (partially) →** strategy invoked on **3** triggered candidates, 0 confirmed
  → those 3 are genuine attempted/tested negatives (albeit with `EXECUTED_NO_SINK` diagnosis). The
  other ~98 tagged candidates never triggered as 611, so they are untested by the XXE strategy.
- **CWE-611 v2:** never invoked (0 triggered as 611).

### Discrepancy with current TAXONOMY.md wording (reported, NOT changed)
Line 52 currently reads: *"**CWE-502, CWE-611 (v1+v2). Their 0 is a genuine tested negative under
the bank's inputs.**"*

This **overstates CWE-502** and **partly overstates CWE-611**:
- **CWE-502:** 0 candidates ever triggered the deserialization sink, so the strategy was never
  exercised. Its 0 is "**not exercised / no candidate reached the sink**," not a tested negative.
- **CWE-611:** only **3** v1 candidates were actually tested by the XXE strategy (0 confirmed); the
  remaining 611-tagged candidates were never invoked, and v2 had 0 invoked. So "genuine tested
  negative" is true only for those 3, not for the class as a whole.

**Proposed wording (for a later phase, not applied now):**
> "(b) confirmable, strategy exists: **CWE-611** was invoked on 3 v1 candidates (0 confirmed —
> genuine attempted negatives, diag `EXECUTED_NO_SINK`); its other tagged candidates never triggered
> the XXE sink. **CWE-502** was never invoked in v1 or v2 (0 candidates triggered the deserialization
> sink), so its 0 is *not exercised*, not a tested negative. CWE-89 (v1) similarly 0."

Do not describe CWE-502's 0 as a tested negative.

---

## Check 2 — V2 clean-checkout reproducibility

### A. Pinned inputs
- **Pinned revision:** `96b68e0021bb25e2dc1a986a7364b7d2961166ab` (branch `vulcan-freeze-149-rc`, HEAD).
- Tracked at that revision:
  - `analysis/corpus_census.py` — **present** (supports `--root`).
  - `calibration/mined_prompts_v2/` — **0 tracked files**.
  - `corpus_census_v2.json` — **not tracked**.

### B. Clean checkout (git archive, tracked files only)
- Command: `git archive HEAD | tar -x -C /home/eyad/vulcan_v2work/archiveD`.
- Result: `calibration/mined_prompts_v2/` **absent** from the archive (0 candidate dirs);
  `corpus_census_v2.json` absent. `analysis/corpus_census.py` present.
- **The V2 census cannot be run faithfully from the pinned revision: the entire V2 corpus (candidate
  records + `_repo_src`) is missing from tracked inputs.** Test status: **BLOCKED — inputs absent.**
  Per instructions, the archive was **not** supplemented from the working tree.

Missing tracked inputs required for a faithful v2 run:
- `calibration/mined_prompts_v2/<repo>/*.json` (all 37 confirmation records + the rest of the v2 corpus).
- The v2 `_repo_src` for any candidate not self-contained.
- (Optionally) `corpus_census_v2.json` as the reference to compare against.

### C. Decision
- **Status: BLOCKED — required V2 inputs are absent from the pinned Git revision.** A faithful V2
  census did **not** run from the clean checkout because the inputs were missing. This is **not** a
  result that "0 of 37 reproduced" — no reproduction attempt could execute, so the 37 are
  **untested under a clean checkout**, neither reproduced nor failed.
- **Self-containment (from working-tree inspection, for planning only):** all **37/37** v2 confirmed
  candidate JSONs embed `ground_truth_source` + `module_source`. So tracking the candidate JSONs is
  **necessary**; it may be **sufficient** for self-contained candidates, but **not guaranteed** —
  `gesh75__netlog-ai` (v1) shows embedded source is not always enough (it needed `_repo_src`). All 13
  v2 repos have **untracked** `_repo_src`.
- **Next step (Phase F, not decided/implemented here):** version the required V2 corpus and
  candidate inputs, then perform a tracked-input-only clean-checkout census to determine whether the
  embedded candidate records suffice or any `_repo_src` content is required for the 37. No tracking
  change is made now.

---

## Summary
- **Verified:** CWE-502 strategy was never invoked (not a tested negative); CWE-611 invoked on 3 v1
  candidates only; V2 is entirely untracked, so its clean-checkout reproduction is BLOCKED (inputs absent from the
  pinned revision); the 37 are untested under a clean checkout, not failed.
- **Unresolved:** per-candidate v2 `_repo_src` necessity (needs a tracked-only run after tracking).
- **Proposed correction:** the TAXONOMY "502/611 genuine tested negatives" wording (above); not applied.
