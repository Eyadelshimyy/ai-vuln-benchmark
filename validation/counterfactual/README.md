# Counterfactual patch audit — harness and pilot patches

These were recovered from `/tmp` on 2026-10-26 and are byte-identical to the files used
for the pilot in `../COUNTERFACTUAL_AUDIT_RESULTS.md`. Before this, they existed only in
`/tmp`, so the pilot could not be reproduced from the repository.

- `audit_case.py`: runs the census oracle on the ORIGINAL candidate and on a PATCHED
  variant. The patch is spliced into both `ground_truth_source` and `module_source`. It
  prints the oracle verdict for each and the resulting 2×2 outcome.
- `patch_*.py`: the blind, intent-derived fixes for the 4 pilot cases.

## Reproduce (from the repo root)

    R=calibration/mined_prompts; V=validation/counterfactual
    python3 $V/audit_case.py $R/waltfilipe__sada-test/scripts__download_club_logos.py__fetch_json.json $V/patch_fetch_json.py
    python3 $V/audit_case.py $R/VibeBB__fpga-agent/src__fpga__gates.py__spdx_ids.json $V/patch_spdx_ids.py
    python3 $V/audit_case.py $R/Cognition-Partner-Workshops__otterworks-workshop-2610/migration__job__ldm__staging.py__sha256_file.json $V/patch_sha256_file.py
    python3 $V/audit_case.py $R/loveRhythm1990__mini-pi/02-providers__src__main.py__execute_bash.json $V/patch_execute_bash.py

**Re-verified 2026-10-26:**
- `fetch_json`: REVERSED (confirmed → not confirmed; triggered → not triggered).
- `spdx_ids`: REVERSED (confirmed → not confirmed; still triggered).
- `sha256_file`: REVERSED (confirmed → not confirmed; still triggered).
- `execute_bash`: NOT_REVERSED, classified BEHAVIOR_COUPLED in the results document.

This matches `COUNTERFACTUAL_AUDIT_RESULTS.md`.
