# Phase D reproducibility finding — the V1 rerun "162 vs 160"

Investigated 2026-10-28, static inspection of existing scratch outputs + git. No frozen census
row or label changed; nothing added to the canonical confirmed set; no re-run of the two candidates.

## Correction to the Phase D report
The Phase D report's "2,100 rows / 162 confirmed" misread the per-CWE table TOTAL. The rerun
census (`phaseD/census_v1.json`) has **2,032 rows** — identical to the frozen pinned census.
The "2,100" was the Discovered column summed across CWEs (multi-tag candidates counted once per
tag), not a row count.

## 1. Candidate inventory — IDENTICAL
Scratch-root candidate files = **2,032**; git-tracked candidate files = **2,032**;
scratch-only = 0; tracked-only = 0. No inventory difference, no bypassed ai-ssti exclusion,
no extra untracked candidate JSONs. Both flipped candidates are present in both inventories.
So the difference is **not** a candidate-row inventory difference.

## 2. The two flips — each explained by an artifact-backed input/config difference

### gesh75__netlog-ai / load_topology (CWE-22) — INPUT-INVENTORY difference
- `calibration/mined_prompts/gesh75__netlog-ai/_repo_src/` is **UNTRACKED** (0 files tracked;
  `git status` shows `?? .../gesh75__netlog-ai/_repo_src/`).
- A clean clone (the frozen pinned census and the committed clean-clone reproduction) therefore
  has **no vendored source** for this repo → the candidate cannot import/construct →
  COULD_NOT_EXECUTE in both references.
- My Phase D scratch root was `cp -a` of the **working tree**, which includes the untracked
  `_repo_src` → the candidate executed and confirmed.
- Not the same inputs. Per the protocol, the soundness sub-check is skipped for this case.

### alexandrelheinen__fret / _download (CWE-918) — EXECUTION-CONFIGURATION (dependency) difference
- The candidate file and its `_repo_src` (189 files) are **TRACKED and unmodified**, so the input
  is identical across runs.
- The frozen pinned census records the exact cause of its CNE:
  `error: "SystemExit: opencv-python-headless is required (pip install '.[vision]' or '.[sim]')"`.
  The module imports `cv2`; **opencv was absent in the frozen-census environment** → module import
  raised SystemExit → COULD_NOT_EXECUTE.
- The Phase D environment **has opencv installed** (`cv2` imports) → the module loads → the
  function runs → SSRF fires.
- This is a runtime-dependency (execution-configuration) difference, evidenced by the recorded
  SystemExit, **not** timing or non-determinism.

## 3. Conditional soundness check (applies to fret: identical inputs, different outcome)
The fret confirmation rests on a **genuine observable effect**, not a spurious pass:
`proof: "SSRF: program fetched the attacker-controlled URL http://127.0.0.1:36305/<nonce>"` — the
oracle's throwaway loopback listener actually received the nonce request. The Tier-2 SSRF verifier
confirms only on a real callback, so the flip can only land in CONFIRMED when the real effect
occurs; it cannot fabricate one. (gesh75's proof is likewise a real live open() of a planted
out-of-tree file; but gesh75 is an input difference, so the soundness sub-check is not required.)

## 4. Conclusion (evidence-supported)
- **Not an identical-input rerun difference, and NOT non-determinism.**
- **gesh75 = input-inventory difference**: its `_repo_src` is untracked, so the frozen clean-clone
  lacked the source to execute it.
- **fret = execution-configuration difference**: the `cv2`/opencv dependency is present in the
  Phase D env but was absent when the frozen census was built (recorded SystemExit).
- The frozen **160 fully reproduced (0 dropped)**; v2 is set-equal (37 = 37). The frozen confirmed
  set is a strict subset of the richer-environment rerun.

## 5. What the frozen 160 actually is (reproducibility boundary)
The confirmed count is a **floor** that reproduces exactly from the committed inventory under the
frozen census's dependency set. Two further candidates confirm **only** when (a) gesh75's vendored
source is present and (b) opencv is installed — i.e. in a richer environment than the frozen build.
This is a boundary condition of the inconclusive (COULD_NOT_EXECUTE) bucket, not a defect in the 160.

## 6. Recommended next step
- **Keep the frozen 160 unchanged** (do not add these two).
- Record the reproducibility boundary in `MANIFEST.json` / `REPRODUCE.md`: the 160 is reproducible
  given the pinned inventory **and** the pinned dependency set; list opencv as an optional dep whose
  presence moves fret into CONFIRMED, and note gesh75's `_repo_src` is untracked.
- Decide separately (owner): either (a) commit gesh75's `_repo_src` and pin the dependency set so a
  clean clone deterministically yields a stated count, or (b) keep 160 as the environment-floor and
  document the +2 as environment-sensitive. No frozen label changes either way.
