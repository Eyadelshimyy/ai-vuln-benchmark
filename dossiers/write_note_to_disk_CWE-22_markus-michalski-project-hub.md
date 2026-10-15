# Disclosure dossier: write_note_to_disk (CWE-22)

- repo: markus-michalski/project-hub  (https://github.com/markus-michalski/project-hub.git)
- file: servers/project-hub-server/tools/docs_writer.py
- tainted parameter: docs_path
- Tier-2 sink: Path.write_text
- Tier-2 exploit proof: traversal write escaped to /tmp/tier2_proof_hb2q5pom/pwned_779799dc8f70

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    write_note_to_disk

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`write_note_to_disk`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `write_note_to_disk` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_hb2q5pom/pwned_779799dc8f70`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
