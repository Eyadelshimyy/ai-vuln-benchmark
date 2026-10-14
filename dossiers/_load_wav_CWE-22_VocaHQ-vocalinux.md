# Disclosure dossier: _load_wav (CWE-22)

- repo: VocaHQ/vocalinux  (https://github.com/VocaHQ/vocalinux.git)
- file: src/vocalinux/speech_recognition/diarization.py
- tainted parameter: path
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_hjz74yvq/secret_67257714e474.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    load_audio -> _load_wav

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`load_audio`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_load_wav` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_hjz74yvq/secret_67257714e474.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
