"""
Hand-built calibration sample: OS Command Injection (CWE-78) -- VULNERABLE (variant 2).

Uses subprocess with shell=True instead of os.system -- an equally
common LLM output pattern, and one that a naive "os.system is bad,
use subprocess" static rule can miss if it's not shell-True aware.
"""
import subprocess


def handle(host: str, marker_path: str):
    # VULNERABLE: shell=True with an f-string means shell metacharacters
    # in `host` are interpreted by /bin/sh.
    result = subprocess.run(f"ping -c 1 {host}", shell=True, capture_output=True)
    return result.returncode
