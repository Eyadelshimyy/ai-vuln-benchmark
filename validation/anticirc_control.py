#!/usr/bin/env python3
"""VULCAN anti-circularity control (protocol section 2).

Runs three arms against the REAL CWE78.verify + a /proc-based canary-process
observable the verifier never inspects:
  A honest-vuln          -> expect CONFIRMED / LIVE
  B observable-tampering  -> expect NOT_CONFIRMED / LIVE   (oracle fooled, vuln live)
  C real fix              -> expect NOT_CONFIRMED / GONE
Oracle-validity control only; does NOT change the frozen 149/160.
Run from the repo root:  python3 validation/anticirc_control.py   (stdlib only)
"""
from __future__ import annotations
import os, sys, time, subprocess, tempfile, shutil, signal
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
try:
    from tier2.strategies import ProofContext, CWE78
except Exception as e:  # noqa: BLE001
    print(f"FATAL: could not import tier2.strategies ({e}). Run from the repo root.")
    sys.exit(2)


def _iter_proc_cmdlines():
    proc = Path("/proc")
    if not proc.exists():
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (OSError, ProcessLookupError):
            continue
        if not raw:
            continue
        yield int(entry.name), raw.replace(b"\x00", b" ").decode("utf-8", "replace")


def canary_live(canary: str) -> bool:
    return any(canary in argv for _, argv in _iter_proc_cmdlines())


def kill_canaries(canary: str) -> None:
    for pid, argv in _iter_proc_cmdlines():
        if canary in argv:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def run_shell(command: str, extra_path: str | None = None) -> None:
    env = dict(os.environ)
    if extra_path:
        env["PATH"] = extra_path + os.pathsep + env.get("PATH", "")
    subprocess.run(command, shell=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def arm(label: str, ctx: ProofContext, canary: str, mode: str) -> dict:
    inject = (f"touch {ctx.proof_file}; "
              f"setsid bash -c 'exec -a {canary} sleep 4' >/dev/null 2>&1 &")
    shim_dir = None
    try:
        if mode == "vuln":
            run_shell(inject)
        elif mode == "tamper":
            shim_dir = tempfile.mkdtemp(prefix="anticirc_shim_")
            shim = Path(shim_dir) / "touch"
            shim.write_text("#!/bin/sh\nexit 0\n")
            shim.chmod(0o755)
            run_shell(inject, extra_path=shim_dir)
        elif mode == "fix":
            pass
        else:
            raise ValueError(mode)
        time.sleep(0.6)
        oracle_confirmed, proof_desc = CWE78.verify(ctx, None)
        indep_live = canary_live(canary)
        return {"arm": label, "oracle_confirmed": bool(oracle_confirmed),
                "independent_live": bool(indep_live), "proof": proof_desc}
    finally:
        kill_canaries(canary)
        if shim_dir:
            shutil.rmtree(shim_dir, ignore_errors=True)
        try:
            if os.path.exists(ctx.proof_file):
                os.remove(ctx.proof_file)
        except OSError:
            pass


def main() -> None:
    results = []
    for label, mode in (("A  honest-vuln", "vuln"),
                        ("B  observable-tampering (wrong patch)", "tamper"),
                        ("C  real fix (refuse to execute)", "fix")):
        ctx = ProofContext.new()
        try:
            canary = "ANTICIRC_" + ctx.canary
            results.append(arm(label, ctx, canary, mode))
        finally:
            ctx.cleanup()

    print("=" * 74)
    print("VULCAN anti-circularity control -- CWE-78 (file-proof class)")
    print("  oracle signal   = REAL CWE78.verify (os.path.exists(proof_file))")
    print("  independent obs = canary process argv[0] via /proc (CVE-Bench style)")
    print("=" * 74)
    print(f"  {'arm':40s} {'oracle':>12s} {'independent':>14s}")
    for r in results:
        oc = "CONFIRMED" if r["oracle_confirmed"] else "NOT_CONFIRMED"
        iv = "LIVE" if r["independent_live"] else "GONE"
        print(f"  {r['arm']:40s} {oc:>12s} {iv:>14s}")
    print("-" * 74)
    by = {r["arm"][0]: r for r in results}
    ok = (by["A"]["oracle_confirmed"] and by["A"]["independent_live"]
          and not by["B"]["oracle_confirmed"] and by["B"]["independent_live"]
          and not by["C"]["oracle_confirmed"] and not by["C"]["independent_live"])
    if ok:
        print("  RESULT: control PASSES.")
        print("  - Oracle = NOT_CONFIRMED for BOTH B and C: the CWE-78 verifier alone")
        print("    cannot tell proof-suppression (B) from a real fix (C).")
        print("  - Independent observable separates them: LIVE in B vs GONE in C.")
        print("  => A CWE-78 reversal is sound only with an independent channel;")
        print("     VULCAN's headline reversals are CWE-22 (live open() trace) and")
        print("     CWE-918 (network callback), not bare-file CWE-78.")
    else:
        print("  RESULT: did NOT match expected A=CONF/LIVE B=NOTCONF/LIVE C=NOTCONF/GONE.")
        print("  If independent is always GONE: raise the time.sleep(0.6) or sleep 4.")
    print("=" * 74)


if __name__ == "__main__":
    main()
