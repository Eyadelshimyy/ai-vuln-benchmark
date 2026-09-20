"""
Hand-built calibration sample: Path Traversal (CWE-22) -- SAFE.

Resolves the final path with os.path.realpath and checks it is still
inside base_dir before opening -- the standard "containment check" fix.
"""
import os


def handle(base_dir: str, filename: str) -> str:
    base_real = os.path.realpath(base_dir)
    candidate = os.path.realpath(os.path.join(base_dir, filename))
    # SAFE: refuse to open anything that resolves outside base_dir.
    if os.path.commonpath([base_real, candidate]) != base_real:
        raise PermissionError("path traversal attempt blocked")
    with open(candidate, "r") as f:
        return f.read()
