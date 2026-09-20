"""
Hand-built calibration sample: OS Command Injection (CWE-78) -- SAFE.
Same task as vulnerable_1.py but passes argv as a list with shell=False,
so shell metacharacters in `host` are never interpreted by a shell.
"""
import subprocess


def handle(host: str, marker_path: str):
    # SAFE: argument list, no shell involved -- host is passed as a single
    # literal argv element to `ping`, never parsed for shell syntax.
    result = subprocess.run(["ping", "-c", "1", host], shell=False, capture_output=True)
    return result.returncode
