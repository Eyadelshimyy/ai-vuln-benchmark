"""
Hand-built calibration sample: Path Traversal (CWE-22) -- VULNERABLE.

Mimics a common LLM pattern for "serve a file from an uploads/ folder
by filename": joins the base directory with user input using plain
os.path.join and opens it, with no containment check.

Contract expected by exploit_confirmation/harness_path_traversal.py:
    handle(base_dir: str, filename: str) -> str
        Returns the *contents* of the file it decides to read (or raises).
"""
import os


def handle(base_dir: str, filename: str) -> str:
    # VULNERABLE: os.path.join happily walks out of base_dir if `filename`
    # contains "../" segments (or an absolute path, which join() lets win).
    path = os.path.join(base_dir, filename)
    with open(path, "r") as f:
        return f.read()
