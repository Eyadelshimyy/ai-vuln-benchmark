"""
Hand-built calibration sample: Path Traversal (CWE-22) -- SAFE (variant 2).

Uses an allowlist/basename-only strategy instead of realpath containment:
discards any directory component the caller supplied, so "../../etc/passwd"
collapses to just "passwd" and is looked up inside base_dir only.
"""
import os


def handle(base_dir: str, filename: str) -> str:
    # SAFE: os.path.basename strips every directory component, so the
    # caller can never name a path outside base_dir, traversal or not.
    safe_name = os.path.basename(filename)
    path = os.path.join(base_dir, safe_name)
    with open(path, "r") as f:
        return f.read()
