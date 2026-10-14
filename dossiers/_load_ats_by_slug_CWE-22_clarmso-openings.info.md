# Disclosure dossier: _load_ats_by_slug (CWE-22)

- repo: clarmso/openings.info  (https://github.com/clarmso/openings.info.git)
- file: scripts/generate_new_jobs.py
- tainted parameter: companies_csv
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_todte9jr/secret_19449380c44c.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **main**
- call path to verify by hand:

    main -> _build_manifest -> _load_ats_by_slug

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`main`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_load_ats_by_slug` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_todte9jr/secret_19449380c44c.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
