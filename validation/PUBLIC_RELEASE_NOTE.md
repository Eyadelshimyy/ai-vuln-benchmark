# VULCAN public-release scope & reproducibility note (prepared; NOT yet committed/pushed)

## Status: PREPARED, NOT PUBLISHED — blocked at the V1 acceptance gate
The licensing-conscious release design passed the publishing-safety gates (no secrets; no
unlicensed `_repo_src` bundled; V2 candidate records are AI-attributed research artifacts). It was
**not committed or pushed** because the mandatory **V1 clean-checkout acceptance gate (exactly 160
under the pinned OpenCV-absent profile) could not be validated in this environment** — see below.

## What a public release WOULD contain (proposed, not staged)
- The 560 V2 candidate JSON records (all AI-attributed; 5 dotfile-named included), + `corpus_census_v2.json` reference.
- The `validation/` audit + reproducibility docs, analysis scripts, `EXTERNAL_SOURCE_MANIFEST.json`.
- **Excluded:** the four full `_repo_src` trees (licensing/size). For MIT repos, their LICENSE text would be required if ever vendored.

## Reproducibility tiers (honest)
- **Public release alone: 33 of 37 V2 confirmations** (candidate records self-contained).
- **With locally-supplied source: up to 37/37**, only if the researcher supplies the 4 repos' source and is entitled to use it (EdgeGuard + autoclip MIT; dipcatcher + estate-ez NOASSERTION/unresolved).
- The canonical **37** remains the documented frozen baseline (supported by Phase F full-source evidence); the public artifact does NOT claim public-only 37/37.

## V1 acceptance gate — BLOCKED (not failed scientifically)
The frozen V1 baseline is **160**, reproduced once in `corpus_census_reproduced.json` under an
environment WITHOUT OpenCV. The current environment has OpenCV (cv2 5.0.0), under which a clean
checkout yields **161** (the `fret` SSRF candidate confirms instead of COULD_NOT_EXECUTE). The pinned
OpenCV-absent profile cannot be reconstructed non-destructively here: masking cv2 via tmpfs leaves an
empty namespace that still imports; a minimal venv lacks the other historical deps and under-counts;
and no historical dependency lockfile exists (Phase E finding). Uninstalling OpenCV from the user's
environment was not done.

**Therefore the V1 = 160 clean-checkout acceptance could not be validated, and per the phase's
mandatory gate the fast-forward, cleanup, and push were NOT performed.** The frozen 160 is unchanged
and remains the canonical result.

## To unblock (owner actions)
1. Provide/validate an OpenCV-absent dependency profile (e.g. `requirements-repro.txt` excluding
   opencv-python-headless) and demonstrate V1 = 160 in it; then this release can proceed.
2. Resolve dipcatcher/estate-ez redistribution rights (or keep them manifest-only).
