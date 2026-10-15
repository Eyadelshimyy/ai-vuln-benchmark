# Disclosure dossier: execute_bash (CWE-78)

> **✅ REAL-NET CONFIRMED (hand-proven end-to-end).** CWE-78 RCE -- poc_minipi_rce.py: injected bash tool-call through mini-pi's real agent_loop; $(id)/$(hostname) executed and exfiltrated to an operator-controlled webhook.
>
> This finding was driven to a live exploit against operator-controlled infrastructure. It is the top rung of the ladder, not a lead.

- repo: loveRhythm1990/mini-pi  (https://github.com/loveRhythm1990/mini-pi.git)
- file: 02-providers/src/main.py
- tainted parameter: command
- Tier-2 sink: subprocess.Popen
- Tier-2 exploit proof: injected shell command created /tmp/tier2_proof_l1sxfv8w/pwned_b74641b99499

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    execute_bash

- dynamic reproduce verdict: REPRODUCED_PUBLIC_API

## Manual verification checklist
1. Open the entry point (`execute_bash`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `execute_bash` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `$(touch /tmp/tier2_proof_l1sxfv8w/pwned_b74641b99499)`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
