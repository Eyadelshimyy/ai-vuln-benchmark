"""
Hand-built calibration sample: Path Traversal (CWE-22) -- VULNERABLE (variant 2).

Uses f-string path building instead of os.path.join, and a shallow
"defense" that only strips a leading "/" -- a very common half-measure
LLM output that stops absolute-path traversal but not "../" traversal.
"""


def handle(base_dir: str, filename: str) -> str:
    # VULNERABLE: strips a single leading slash (looks like a fix for
    # absolute paths) but does nothing about "../" segments.
    if filename.startswith("/"):
        filename = filename[1:]
    path = f"{base_dir}/{filename}"
    with open(path, "r") as f:
        return f.read()
