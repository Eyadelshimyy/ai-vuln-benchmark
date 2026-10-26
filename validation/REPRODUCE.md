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
