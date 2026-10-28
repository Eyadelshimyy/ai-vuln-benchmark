# Supplemental execution-confirmed findings (outside the frozen baseline)

Two candidates confirmed in the Phase D rerun that are **COULD_NOT_EXECUTE in the frozen census**.
They are kept **outside** the frozen V1 baseline of 160 and reported separately here. Neither is
added to any frozen set. See `../REPRODUCIBILITY_FINDING.md` for the full diagnosis.

## 2.1 alexandrelheinen__fret — `_download` (CWE-918 SSRF) — enhanced-dependency environment

- **Why outside the baseline:** the frozen census recorded
  `SystemExit: opencv-python-headless is required` — the module imports `cv2`, and OpenCV was
  **unavailable under the frozen execution profile**, so the function could not even be imported.
  In the Phase D environment OpenCV was present (`cv2 5.0.0`), the function executed, and the SSRF
  fired.
- **This is a genuine security-relevant effect, not a harmless discrepancy:** the oracle's
  throwaway loopback listener actually received the attacker nonce.
  - Proof: `SSRF: program fetched the attacker-controlled URL http://127.0.0.1:36305/<nonce>`.
- **Evidence artifact:** `fret/phaseD_census_record.json` (sha256 `17a5cca07156e6c2…`), copied
  unchanged from the Phase D rerun census (`/home/eyad/vulcan_v2work/phaseD/census_v1.json`).
- **Input provenance:** the candidate JSON `calibration/mined_prompts/alexandrelheinen__fret/
  scripts__vision_web_ball_gallery.py___download.json` is **tracked and unmodified**
  (sha256 `331ff29469c326ca…`); its `_repo_src` is tracked (189 files). So the input is identical to
  the frozen run — only the runtime dependency differed.
- **Limitation:** the saved proof is the census record's `tier2_evidence.proof` string plus verdict,
  not a separate packet-capture of the HTTP callback. The listener-hit is recorded by the oracle,
  not independently captured.

## 2.2 gesh75__netlog-ai — `load_topology` (CWE-22) — source-availability-dependent

- **Why outside the baseline:** the repo's vendored `_repo_src/` is **UNTRACKED** (0 files in git;
  `?? …/gesh75__netlog-ai/_repo_src/`). A clean clone / the frozen census therefore has **no
  vendored source** for this repo, so the candidate could not execute (COULD_NOT_EXECUTE). The
  Phase D scratch root was a `cp -a` of the working tree, which included the untracked source, so it
  executed and confirmed.
- **Evidence artifact:** `gesh75/phaseD_census_record.json` (sha256 `8683aebd7a854668…`), copied
  unchanged from the Phase D rerun census.
  - Proof: `attacker-controlled input opened our out-of-tree planted file /tmp/…/secret_<nonce>.txt
    (proven by live execution trace; read-traversal, no output echo required)`.
- **Input provenance:** the candidate JSON is **tracked and unmodified** (sha256
  `1555737b0c1d936a…`), but the `_repo_src` it needs to execute is **untracked** → not part of the
  frozen input inventory.
- **What it establishes (precisely):** under the working-tree source, a read-traversal to a planted
  out-of-tree file occurred. **What is unavailable:** the vendored source is not in version control,
  so this result is not reproducible from a clean clone or the pinned revision until the source is
  committed (an owner decision; see `../REPOSITORY_CLEANLINESS_PLAN.md`).

## 2.3 Counting units (kept separate — do not sum without a stated unit)

| population | count | condition |
|---|---|---|
| Frozen V1 baseline (confirmed candidate rows) | **160** | committed inputs, frozen execution profile |
| Frozen V2 baseline (confirmed candidate rows) | **37** | committed inputs |
| Supplemental: fret | 1 | identical input + OpenCV available |
| Supplemental: gesh75 | 1 | untracked `_repo_src` present |
| Application-level reachability findings | 1 | AutoClip `sync-data` APP_CONFIRMED (bounded) |
| Untested / inconclusive reachability candidates | see `../reachability/REACH_LEDGER.json` | — |

These are **distinct populations**. The supplemental 2 are **not** added to 160; the app-level
finding is a reachability-axis unit, not a candidate-row unit.
