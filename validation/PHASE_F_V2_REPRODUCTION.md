# Phase F — V2 clean-snapshot reproducibility audit (non-committing)

Goal: determine whether the 37 V2 confirmations reproduce from a clean, isolated source snapshot
(pinned revision + only the proposed V2 inputs), and propose the minimal files to version. No commit,
no staging, no label/census changes; all work in `/home/eyad/vulcan_v2work/phaseF/` (outside the repo).

## Integrity baselines (recorded before work)
- repo root `/home/eyad/ai-vuln-benchmark`; HEAD `96b68e0021bb25e2dc1a986a7364b7d2961166ab`;
  branch `vulcan-freeze-149-rc`; index staged = 0; tracked-modified = 13; untracked = 198.
- Preserved unchanged: no files moved/deleted/committed; real index/working tree untouched.

## Dependency closure (from `analysis/corpus_census.py` code path)
The census globs `root.glob("*/*.json")` (pathlib — **reads dotfile-named records**), reads each
candidate's embedded `ground_truth_source`/`module_source`, and injects each repo's `_repo_src` onto
`sys.path` (`vendor_sys_path_roots(path.parent/_repo_src)`). So embedded fields are **not** a
guarantee of self-containment — a candidate that imports sibling modules needs `_repo_src`.

- **Required inputs:** census scripts (TRACKED at HEAD: `analysis/corpus_census.py`, `analysis/iast/*`,
  `tier2/*`); V2 candidate records (`calibration/mined_prompts_v2/*/*.json`, **UNTRACKED**, 560 incl.
  5 dotfile-named); `_repo_src` for the non-self-contained candidates (**UNTRACKED**).
- **Reference output:** `corpus_census_v2.json` (UNTRACKED) — kept OUTSIDE the snapshot; comparison
  target only, never an input (no circular validation).
- **Generated outputs:** `phaseF/out_variantA.json`, `out_variantB.json`.

## Isolation
Snapshot = `git archive HEAD | tar -x` (tracked files only) + the proposed V2 inputs copied in.
Census run under bubblewrap with **the real repo masked** (`--tmpfs /home/eyad`, only `phaseF`
re-bound): verified inside the sandbox `ls /home/eyad/ai-vuln-benchmark` → "No such file or
directory". cwd = snapshot; `--root` → snapshot's v2 corpus; `--out` → outside the snapshot. No
working-tree fallback possible. Python 3.14.4; executed with `--unshare-net` (loopback only).

## Results

### Variant A — candidate JSONs only (no `_repo_src`)
- Snapshot: 555 candidate records (bash copy missed the 5 dotfile-named — a snapshot-construction
  bug, not a corpus issue; none of the 5 is confirmed).
- Census: **33 confirmed**. Candidate-level: 33/37 reproduced; **4 not reproduced**, all CWE-22, all
  `COULD_NOT_EXECUTE` (no vendored source to import). 0 extra.
- Conclusion: **33 of 37 are self-contained**; 4 are not.

### Variant B — candidate JSONs + `_repo_src` for the 4 affected repos
- Census: **37 confirmed**, candidate-level **set-equal to the reference (0 missing, 0 extra)**;
  CWE-22 30 / CWE-89 3 / CWE-918 4.
- The 4 (`Kopanov__EdgeGuard-Knowledge-Graph`, `artificial-hedge__dipcatcher`,
  `ayushdad609-code__autoclip`, `zorntyler540-byte__-estate-ez`) flip A=False → B=True once their
  `_repo_src` is present, matching the reference.

## Reproduction status: **PARTIALLY REPRODUCED → REPRODUCED under the identified minimal input set**
- From candidate records alone (variant A): PARTIAL (33/37; the 4 non-self-contained need source).
- With the minimal input set (variant B = records + 4 repos' `_repo_src`): **REPRODUCED** — exact
  37-candidate set agreement, no unexplained discrepancies, evidence generated in-run, compared to an
  independently preserved reference.
- V2 remains **BLOCKED from the pinned revision as-is** (those inputs are untracked); it is
  REPRODUCED only from the proposed snapshot. Tracking is a later, separately-approved phase.

## Minimal versioning proposal (checksummed: `validation/V2_VERSIONING_PROPOSAL.json`)
1. **All 560 V2 candidate records** `calibration/mined_prompts_v2/*/*.json` — **including the 5
   dotfile-named** `gyrinx-app__gyrinx/.github__…json` (a shell glob skips these; the census reads
   them). Sufficient for 33/37.
2. **`_repo_src` for exactly 4 repos** (the non-self-contained confirmations): EdgeGuard (8.7 MB),
   dipcatcher (82.6 MB), autoclip (7.3 MB), estate-ez (2.0 MB) ≈ **100 MB total**.
3. Census scripts already tracked at HEAD. Reference output stays a comparison artifact, not versioned as input.

SHA-256 of the 37 confirmed candidate records is recorded in the proposal JSON.

## Caveats / unresolved
- The full faithful run needs the dotfile-aware copy (use `cp -a`/pathlib, not a bash `*/*.json` glob).
- Whether the OTHER ~29 confirmed repos are *truly* self-contained is shown empirically by variant A
  (they confirmed without `_repo_src`); a future tracked-only run after versioning should re-confirm 37.
- This is a verification phase: no inputs were added to the repo, nothing committed.

---

## Phase F completion (2026-10-28): dotfile-complete 560 run + source-closure audit

Rebuilt the snapshot dotfile-aware (pathlib) with **all 560** candidate records (path-set exactly
equal to source; 5 dotfile candidates present and hash-verified) + the 4 repos' `_repo_src`. Ran the
committed census in the masked-repo sandbox (`/home/eyad/ai-vuln-benchmark` absent inside).

### Full 560 reconciliation vs frozen reference (`corpus_census_v2.json`)
- Candidate inventory: **560 expected = 560 observed**, IDs unique, 0 missing / 0 extra.
- **Confirmation-set reproduced:** confirmed **37 = 37, candidate-level set-equal**, CWE-22 30 / CWE-89 3 / CWE-918 4.
- **Full census: reproduced EXCEPT 3 non-confirming verdict differences** (none in the confirmed set):
  - `gyrinx .../screenshot.py __update_comparison_markdown`: ref COULD_NOT_EXECUTE → 560 NOT_TRIGGERED
  - `hacky-movies/imdb.py get_json`: ref NOT_TRIGGERED → 560 COULD_NOT_EXECUTE
  - `Lawsuit_agent/task_store.py _persist`: ref NOT_TRIGGERED → 560 COULD_NOT_EXECUTE
  - All three stay `confirmed=False` in both runs; they are CNE↔NOT_TRIGGERED flips in the
    inconclusive/not-triggered buckets — the same execution-environment sensitivity characterized in
    `REPRODUCIBILITY_FINDING.md`. Verdict dist: 560 = NT 301 / CNE 177 / TRIG 82; ref = NT 302 / CNE 176 / TRIG 82.
- **Status:** **Confirmation-set reproduced** (exact). **Full census reproduced with 3 explained
  non-confirming discrepancies** — NOT a clean full-census match; characterized, not reconciled by
  changing frozen data.

### Five dotfile records (individually reconciled)
All five `gyrinx …/.github__…json` reproduce their reference status exactly: 1 COULD_NOT_EXECUTE +
4 TRIGGERED, **all confirmed=False**. Their inclusion changed **none** of the other 555 records
(555-vs-560 shared diff = 0). So the earlier 555-file omission did not affect the 37 or any aggregate
beyond adding these 5 non-confirming rows.

### Source-closure audit (4 repos)
Full `_repo_src` footprint (logical bytes): EdgeGuard 8.7 MB/363f, dipcatcher 82.6 MB/19,695f,
autoclip 7.3 MB/648f (incl. 145 `.pyc`), estate-ez 2.0 MB/201f → **100.6 MB / 20,907 files total**.

Static import-closure trace (relative-import-aware) + **Variant C targeted validation** (committed
census over a root of just the 4 candidates with their minimal closures):

| repo | full | closure | files | Variant C result |
|---|---|---|---|---|
| EdgeGuard | 8.7 MB | 1.08 MB | 34 | **CONFIRMED ✓ (validated)** |
| estate-ez | 2.0 MB | 0.045 MB | 8 | **CONFIRMED ✓ (validated)** |
| dipcatcher | 82.6 MB | 0.42 MB | 40 | ✗ `ModuleNotFoundError: fx1` (closure incomplete) |
| autoclip | 7.3 MB | 0.49 MB | 46 | ✗ `ModuleNotFoundError: backend.services.base` (closure incomplete) |

**Targeted closure validation only** (not a full 560 run with Variant C). Minimal closure is
**validated for 2 of 4** repos; for dipcatcher and autoclip the static trace missed runtime-required
modules, so the subset is **not** sufficient as traced.

### Option A vs Option B (trade-off)
- **Option A — full trees:** 20,907 files / 100.6 MB. Verified by the full 560 run (37 set-equal).
- **Option B — minimal closure:** 128 files / 2.03 MB (≈98% smaller) **but validated only for EdgeGuard
  + estate-ez**. dipcatcher and autoclip closures are unverified (fail targeted validation).
- **Recommendation for the commit phase:** Option A is the currently-verified option for all 4.
  Option B is a validated leaner choice **only for EdgeGuard and estate-ez** (saves ≈10.6 MB there);
  dipcatcher/autoclip would need a completed, re-validated closure before Option B is safe for them.
  Do not adopt the unverified dipcatcher/autoclip subsets.

### Reproducibility boundary (unchanged)
V2 is **not reproducible from the current pinned Git revision alone** — the required V2 inputs
(candidate records + `_repo_src`) are untracked and absent from `96b68e0…`. The expanded snapshot
reproduces the confirmation set; tracking the inputs is a later, separately-authorized phase.
