#!/usr/bin/env python3
"""VULCAN Tier-2 SOUNDNESS REGRESSION TEST.

The strongest claim VULCAN makes is that a function is counted vulnerable ONLY
when execution proves it -- never on a detector's word, and never as a side
effect of a diagnostic label. This test enforces that claim mechanically.

It runs two batteries through the *full* Tier-2 pipeline (attempt_exploit, the
same path real candidates take):

  SAFE  -- deliberately well-written functions that sanitise, confine, reject,
           or pass input as an argv element. NONE may ever come back confirmed.
           A single confirmation here is a FALSE POSITIVE and fails the suite.

  VULN  -- deliberately vulnerable functions. Each MUST confirm, proving the
           oracle still detects the real thing (a guard that never fires would
           trivially have zero false positives but be useless).

Run directly:   python3 tier2/test_soundness.py      (prints a report, exits 1 on any failure)
Run via pytest: pytest tier2/test_soundness.py

Keep this green. Any future change to the sinks, the harness, or a strategy's
verify() that lets a SAFE function reach `confirmed` must fail here first.

NOTE ON shlex.split + Popen: it is deliberately in the VULN set, not SAFE.
`shlex.split` only tokenises; `Popen(list)` on a fully attacker-controlled
string lets the attacker choose argv[0] (the program itself) -- genuine
arbitrary-command execution. The oracle confirming it is correct.
"""
import os
import sys

# Allow running from the repo root or from tier2/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "analysis", "iast")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tier2.engine import attempt_exploit  # noqa: E402

TIMEOUT = 10.0

# ---------------------------------------------------------------------------
# SAFE battery: correctly-written code. The attacker controls neither the
# program that runs nor a path that escapes the intended tree. EVERY entry
# MUST come back confirmed=False.
# ---------------------------------------------------------------------------
SAFE = [
    ("p_basename", "CWE-22", "name",
     "def p_basename(name):\n"
     "    import os\n"
     "    base = '/tmp/snd_root'; os.makedirs(base, exist_ok=True)\n"
     "    p = os.path.join(base, os.path.basename(name))\n"
     "    open(p, 'a').close()\n"
     "    return open(p).read()\n"),
    ("p_reject", "CWE-22", "name",
     "def p_reject(name):\n"
     "    if name.startswith('/') or '..' in name:\n"
     "        raise ValueError('bad path')\n"
     "    return open(name).read()\n"),
    ("p_fixed", "CWE-22", "name",
     "def p_fixed(name):\n"
     "    return open('/etc/hostname').read()\n"),
    ("p_confine", "CWE-22", "name",
     "def p_confine(name):\n"
     "    import os\n"
     "    base = os.path.realpath('/tmp/snd_c'); os.makedirs(base, exist_ok=True)\n"
     "    full = os.path.realpath(os.path.join(base, name))\n"
     "    if not full.startswith(base + os.sep):\n"
     "        raise ValueError('escape')\n"
     "    open(full, 'a').close()\n"
     "    return open(full).read()\n"),
    ("c_argv", "CWE-78", "name",
     "def c_argv(name):\n"
     "    import subprocess\n"
     "    p = subprocess.Popen(['echo', name]); p.wait()\n"
     "    return 0\n"),
    ("c_quote", "CWE-78", "name",
     "def c_quote(name):\n"
     "    import os, shlex\n"
     "    return os.system('echo ' + shlex.quote(name))\n"),
    ("c_fixedprog", "CWE-78", "name",
     "def c_fixedprog(name):\n"
     "    import subprocess\n"
     "    p = subprocess.Popen(['ls', name]); p.wait()\n"
     "    return 0\n"),
    # SSRF defended: a loopback/private/reserved-address guard rejects the
    # attacker URL before the request. The oracle server lives on loopback, so
    # a function that refuses loopback can NEVER be confirmed -- which is the
    # point: confirmation requires the program to ACTUALLY fetch the attacker
    # URL, not merely to contain requests.get(). Proves the SSRF oracle does
    # not "cheat" by rewarding SSRF-shaped code that is in fact defended.
    ("s_ssrf_block", "CWE-918", "url",
     "def s_ssrf_block(url):\n"
     "    import requests, urllib.parse, socket, ipaddress\n"
     "    host = urllib.parse.urlparse(url).hostname\n"
     "    ip = ipaddress.ip_address(socket.gethostbyname(host))\n"
     "    if ip.is_loopback or ip.is_private or ip.is_reserved or ip.is_link_local:\n"
     "        raise ValueError('blocked host')\n"
     "    return requests.get(url, timeout=3).status_code\n"),
]

# ---------------------------------------------------------------------------
# VULN battery: the input controls the sink -- the path opened, the path
# written, the shell command, or the program executed. EVERY entry MUST confirm.
# ---------------------------------------------------------------------------
VULN = [
    ("v_read", "CWE-22", "name",
     "def v_read(name):\n"
     "    with open(name) as f:\n"
     "        return len(f.read())\n"),
    ("v_write", "CWE-22", "name",
     "def v_write(name):\n"
     "    open(name, 'w').write('x')\n"
     "    return 1\n"),
    ("v_os", "CWE-78", "name",
     "def v_os(name):\n"
     "    import os\n"
     "    return os.system('echo ' + name)\n"),
    ("v_split", "CWE-78", "name",
     "def v_split(name):\n"
     "    import subprocess, shlex\n"
     "    p = subprocess.Popen(shlex.split(name)); p.wait()\n"  # attacker controls argv[0]
     "    return 0\n"),
    # SSRF: the request target is fully attacker-controlled. The sibling
    # params a real caller would omit (params=, headers=) must be left at
    # their real defaults, not fabricated placeholder strings -- the latter
    # made requests.get(url, params='<ph>') raise on params.items() AFTER the
    # sink hit was recorded, so Tier-1 said TRIGGERED but the oracle server
    # was never actually hit (a false NOT_CONFIRMED). This case guards that
    # live-path fix: it must confirm end-to-end (real fetch of the oracle URL).
    ("v_ssrf", "CWE-918", "url",
     "def v_ssrf(url, params=None, headers=None):\n"
     "    import requests\n"
     "    return requests.get(url, params=params, headers=headers, timeout=3).status_code\n"),
]


def _confirm(fn, cwe, param, src):
    rec = {"repo": "soundness", "file": "s.py", "function": fn, "line": 1,
           "ground_truth_source": src}
    res = attempt_exploit(rec, param, cwe, use="ground_truth", max_agent=0, timeout=TIMEOUT)
    return bool(res.get("confirmed")), res.get("diag_bucket")


# ---- pytest entry points (one assertion per case for readable failures) ----
def test_safe_never_confirmed():
    for fn, cwe, param, src in SAFE:
        confirmed, bucket = _confirm(fn, cwe, param, src)
        assert not confirmed, (
            f"FALSE POSITIVE: safe function {fn!r} ({cwe}) was confirmed vulnerable "
            f"(bucket={bucket}). The soundness guarantee is broken.")


def test_vuln_always_confirmed():
    for fn, cwe, param, src in VULN:
        confirmed, _ = _confirm(fn, cwe, param, src)
        assert confirmed, (
            f"MISSED: vulnerable function {fn!r} ({cwe}) was NOT confirmed. "
            f"The oracle has regressed and no longer detects a real vulnerability.")


def _main():
    fp = miss = 0
    print("=== SAFE (must all be NOT confirmed) ===")
    for fn, cwe, param, src in SAFE:
        c, b = _confirm(fn, cwe, param, src)
        fp += c
        print(f"  {'FALSE-POSITIVE!!' if c else 'ok':16} {fn:12} {cwe}  confirmed={c}  bucket={b}")
    print("=== VULN (must all confirm) ===")
    for fn, cwe, param, src in VULN:
        c, _ = _confirm(fn, cwe, param, src)
        miss += (not c)
        print(f"  {'MISS!!' if not c else 'ok':16} {fn:12} {cwe}  confirmed={c}")
    print(f"\nfalse positives = {fp} (MUST be 0) | missed vulns = {miss} (MUST be 0)")
    ok = (fp == 0 and miss == 0)
    print("VERDICT:", "SOUND" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
