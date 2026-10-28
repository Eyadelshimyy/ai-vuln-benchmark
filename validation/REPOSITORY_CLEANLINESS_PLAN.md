# Repository cleanliness plan (Phase E — PROPOSED, not applied)

Nothing here is executed in Phase E: no staging, commits, moves, deletes, or `.gitignore` edits.
This is the plan for a later, separately-approved Phase F. It preserves all existing work and
distinguishes files to track from files intentionally excluded.

## A. Proposed `.gitignore` additions (scratch / generated-outside-repo)

```
# Phase D / reproducibility scratch (never commit)
/home/eyad/vulcan_v2work/        # NOTE: outside the repo already; listed for operators, not a repo path
# session scratch copies
*.log.bak
# large regenerable census variants kept only as working notes (keep the frozen ones)
corpus_census_FULL*.json
corpus_census_why*.json
corpus_census_scaled.json
```
(Only add rules for genuinely regenerable/scratch outputs. Do NOT ignore the frozen censuses,
`corpus_census_pinned.json`, `corpus_census_reproduced.json`, or `MANIFEST.json`.)

## B. Scratch locations that must stay OUT of the repo
- `/home/eyad/vulcan_v2work/phaseD/` — Phase D rerun outputs (census_v1.json, census_v2.json, logs,
  copied corpora). Regenerable; never commit.
- `/home/eyad/vulcan_v2work/{clones,twins_root,boot,tmp}` — build scratch.

## C. Untracked source directories intentionally OUTSIDE the frozen corpus
- `calibration/mined_prompts/gesh75__netlog-ai/_repo_src/` — untracked vendored source. It is NOT
  part of the frozen V1 inventory (that is why gesh75 is supplemental, not in the 160). Decision for
  the owner: either (a) commit it + pin deps to make the +1 reproducible, or (b) leave it untracked
  and keep gesh75 supplemental. Do not silently add it to V1.
- The 18 other V1 repos + 13 V2 repos with untracked `_repo_src` (see `REPO_SRC_AUDIT.json`): these
  are supplemental working-tree material; the 160 reproduce without them (embedded candidate source).

## D. Load-bearing artifacts that SHOULD be tracked (for clean-clone reproducibility)
From `ARTIFACT_MANIFEST.json` (git_status = untracked, class ∈ generated/supplemental/owner-delivered):
- `corpus_census_v2.json` (+ the v2 corpus `calibration/mined_prompts_v2/` and, if v2 is to be
  frozen, its `_repo_src`).
- `validation/counterfactual/audit_case.py`, `patch_*.py` (patch-audit harness).
- `validation/reachability/REACH_WORKLIST.json`/`.md`, `REACH_LEDGER.json`/`.md`,
  `RECONCILIATION_REPORT.md`, `source_reviews/*`, `evidence/autoclip/*` (owner bundle).
- `validation/supplemental/*`, `validation/EVIDENCE_INDEX.md`, `validation/ARTIFACT_MANIFEST.json`,
  `validation/REPO_SRC_AUDIT.*`, `validation/VERIFICATION_REPORT.md`,
  `validation/REPRODUCIBILITY_FINDING.md`, `validation/corpus_v2/*`, `validation/twins/*`,
  `validation/rq2/*`.
- New analysis scripts: `analysis/build_reach_worklist.py`, `build_reach_ledger.py`,
  `cwe22_subtype.py`, `rq2_labelset.py`, `rq2_compare.py`, `make_twins.py`, `verify_twins.py`,
  `corpus_v2_*.py`, `app_screen.py`, `commit_model_attribution.py`, `rq1_by_generator.py`, and the
  new `test_*.py`.

## E. Stray / layout
- `analysis/iast/auto_harness.py.bak2` and other `*.bak*` — scratch backups; propose NOT tracking
  (add to `.gitignore`) or `git rm` only if the owner confirms they are obsolete. Do not move now.
- Secrets scan: **none found** among the artifacts to track. `~/.config/github_token` and
  `~/.config/openrouter_key` are OUTSIDE the repo and must never be tracked. No token/credential
  strings are present in any artifact proposed for tracking. (Re-scan before Phase F commit.)

## F. Missing deliverable to create in Phase F (not now)
- **Explicit input manifest** tying the frozen corpus to the pinned revision (file list + sha256),
  so reproduction never depends on `cp -a` of a working tree. Does not exist yet.
- **Versioned dependency reference profile** (e.g. `requirements-repro.txt`) EXCLUDING
  `opencv-python-headless`, validated to reproduce 160 before being claimed canonical.

## G. Expected git status after a future approved Phase F
- No untracked load-bearing artifacts (all of D tracked).
- No committed scratch (B ignored; C decided per owner).
- Frozen census rows and labels unchanged.
- Clean clone reproduces V1=160 and (if v2 corpus committed) V2=37 and regenerates the ledger
  identically; recorded in `REPRODUCE.md`.
