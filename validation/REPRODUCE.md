# Reproducing the VULCAN census (160 confirmed candidates / 149 functions)

    git clone --branch vulcan-freeze-149-rc <repo> vulcan && cd vulcan
    python3 analysis/corpus_census.py --tier2 --out census.json

Expected: **2032 candidates, 160 confirmed** = {CWE-22: 130, CWE-78: 4, CWE-918: 26}.
Dedup to the 149 distinct functions with analysis/classify_corpus.py.

Verified: a clean clone of this branch reproduces the 160-confirmed set EXACTLY
(DROPPED=0, ADDED=0 vs corpus_census_pinned.json).

Notes:
- ai-ssti-*/llama-cpp-ssti are excluded by corpus_census.py (selection-biased SSTI sub-study).
- 153/160 confirmations need only the in-tree candidate JSONs. The other 7 (in 3 repos:
  ExposureEclipse, jobbot, otterworks) need those repos' _repo_src, which IS committed in-tree.
- The full Tier-1 funnel (289 triggered) and reachability analysis additionally require the
  complete vendored environment (~540M); re-vendor it from validation/corpus_repo_pins.json.
- Tier-2 confirmation is execution-based; reproduction observed deterministic for the
  confirmed set on a clean clone.

---

## Execution profile & reproducibility boundary (Phase E, 2026-10-28)

The "2032 candidates / 160 confirmed" claim is reproducible **given the committed inputs AND the
execution profile under which the frozen census was built**. Confirmed-count is the floor that a
clean clone reproduces; it can rise at the margins in a richer environment (see below).

### Reference execution profile
Classification per the environment evidence available:

- **Historically evidenced** (from contemporaneous artifacts): under the frozen profile,
  `opencv-python-headless` was **absent** — the pinned census records
  `SystemExit: opencv-python-headless is required` for `alexandrelheinen__fret/_download`. That is
  the only directly-evidenced historical dependency fact; it does NOT establish the full dep set.
- **Reconstructed** (inferred, not a historical lock): the census needs Python 3 and the Tier-2
  strategy code in `tier2/` + `analysis/iast/` at the frozen commit; candidate execution pulls in
  whatever each candidate imports.
- **Proposed reference profile** (for future runs; NOT yet validated to reproduce 160): Python 3.14,
  and a declared dependency set EXCLUDING `opencv-python-headless` (so fret stays CNE as in the
  frozen baseline). Current Phase D env for comparison: Python 3.14.4, cv2 5.0.0, numpy 2.5.4,
  pyyaml 6.0.3, requests 2.32.5, httpx 0.28.1, jinja2 3.1.6, lxml 6.0.2, sqlalchemy 2.1.4,
  fastapi 0.143.0. **No complete historical lockfile exists** — this is a limitation; a versioned
  reference profile is a deliverable in REPOSITORY_CLEANLINESS_PLAN.md, not yet demonstrated.

### Phase D observation (specific cases, not a general guarantee)
A Phase D rerun reproduced all **160** frozen V1 confirmations (0 dropped) and v2 exactly (37=37).
Two *additional* candidates confirmed only under richer conditions and are kept OUTSIDE the baseline
(see supplemental/SUPPLEMENTAL_FINDINGS.md): `fret` (needs OpenCV present) and `gesh75` (needs its
untracked `_repo_src` present). This is an observation about these two cases; it is not a claim that
adding dependencies can never change other results.

### Clean-clone input acquisition (canonical)
Reproduce ONLY via one of:
1. a clean clone / archive of the exact pinned revision (`git clone --branch vulcan-freeze-149-rc`), or
2. a version-controlled input manifest tied to that revision.

Do NOT use `cp -a` of a working tree as the reproduction procedure: a working tree carries
**untracked** directories (e.g. `gesh75__netlog-ai/_repo_src/`) that do not exist in a clean
checkout by construction. A `.gitignore` rule does not remove an already-present untracked directory
from a whole-tree copy — which is exactly why the Phase D `cp -a` run executed gesh75 when a clean
clone cannot. An explicit input manifest does not yet exist; it is listed as a deliverable in
REPOSITORY_CLEANLINESS_PLAN.md. Frozen inputs/exclusions are unchanged in this phase.
