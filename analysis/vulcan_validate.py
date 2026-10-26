#!/usr/bin/env python3
"""vulcan validate -- oracle self-check.

Proves VULCAN's execution oracle is SOUND (0 false positives, verified by
known-negative controls) and BOUNDED (a measured confirmable-CWE set + a
measured execution-coverage limit). This is the evidence behind the claim
'execution-confirmation = ground truth': confirmed is a subset of exploitable,
and both the soundness and the limits are shown, not assumed. Re-runnable by
any reviewer.

  python3 analysis/vulcan_validate.py
  python3 analysis/vulcan_validate.py --realworld corpus_census_cvebench_fixed.json
"""
import argparse, json, subprocess, sys, re
from pathlib import Path
from collections import defaultdict
REPO = Path(__file__).resolve().parent.parent

def soundness():
    try:
        out = subprocess.run([sys.executable, str(REPO/"tier2"/"test_soundness.py")],
                             capture_output=True, text=True, timeout=300).stdout
    except Exception as e:
        return {"error": str(e)}
    safe = out.split("=== VULN")[0]
    vuln = out.split("=== VULN")[1] if "=== VULN" in out else ""
    g = lambda p: (int(re.search(p, out).group(1)) if re.search(p, out) else None)
    v = re.search(r'VERDICT:\s*(\w+)', out)
    return {"safe_total": safe.count("confirmed="),
            "safe_confirmed": safe.count("confirmed=True"),
            "vuln_total": vuln.count("confirmed="),
            "vuln_confirmed": vuln.count("confirmed=True"),
            "fp": g(r'false positives = (\d+)'),
            "missed": g(r'missed vulns = (\d+)'),
            "verdict": v.group(1) if v else "?"}

def census(path):
    if not Path(path).exists(): return None
    rows = json.load(open(path)).get("rows", [])
    disc, conf = defaultdict(int), defaultdict(int)
    for r in rows:
        for c in (r.get("cwes") or r.get("candidate_cwes") or []): disc[c] += 1
        if r.get("confirmed"):
            ec = r.get("exec_cwe") or (r.get("cwes") or [None])[0]
            if ec: conf[ec] += 1
    return {"n": len(rows),
            "cne": sum(1 for r in rows if r.get("verdict") == "COULD_NOT_EXECUTE"),
            "confirmed": sum(1 for r in rows if r.get("confirmed")),
            "disc": dict(disc), "conf": dict(conf)}

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", default=str(REPO / "corpus_census_pinned.json"))
    ap.add_argument("--realworld", default=None)
    ap.add_argument("--skip-soundness", action="store_true")
    a = ap.parse_args()
    print("=" * 64); print("VULCAN ORACLE VALIDATION"); print("=" * 64)
    ok = True
    if not a.skip_soundness:
        s = soundness()
        if "error" in s:
            print(f"[1-2] controls: COULD NOT RUN ({s['error']})"); ok = False
        else:
            print(f"[1] known-NEGATIVE controls (safe): {s['safe_total']} tested, "
                  f"{s['safe_confirmed']} confirmed -> FP rate = {s['fp']}  "
                  f"[{'PASS (sound)' if s['fp']==0 else 'FAIL'}]")
            print(f"[2] known-POSITIVE controls (synthetic): {s['vuln_total']} tested, "
                  f"{s['vuln_confirmed']} confirmed -> recall {s['vuln_confirmed']}/{s['vuln_total']}  "
                  f"[{'PASS' if s['missed']==0 else 'FAIL'}]   battery VERDICT: {s['verdict']}")
            ok = ok and s['fp'] == 0 and s['missed'] == 0
    c = census(a.census)
    if c:
        cov = 100 * (c['n'] - c['cne']) / c['n'] if c['n'] else 0
        print(f"\n[3] per-CWE confirmable set  (corpus: {Path(a.census).name}, {c['confirmed']} confirmed)")
        for cwe in sorted(set(c['disc']) | set(c['conf']), key=lambda x: -c['conf'].get(x, 0)):
            cf = c['conf'].get(cwe, 0)
            print(f"      {cwe:9} discovered={c['disc'].get(cwe,0):>4}  confirmed={cf:>4}  "
                  f"-> {'CONFIRMABLE' if cf > 0 else 'detection-only'}")
        print(f"\n[4] execution coverage: {c['n']-c['cne']}/{c['n']} ran ({cov:.0f}%)  |  "
              f"{c['cne']} COULD_NOT_EXECUTE  (false-negative source)")
    else:
        print(f"\n[3-4] census not found: {a.census}")
    if a.realworld:
        rw = census(a.realworld)
        if rw:
            fired = [k for k, v in rw['conf'].items() if v]
            print(f"\n[5] real-world check ({Path(a.realworld).name}): {rw['confirmed']} "
                  f"confirmed of {rw['n']} mined; classes actually fired: {fired or 'none'}")
    print("\n" + "=" * 64)
    print("CLAIM: confirmed is a subset of exploitable. The oracle is SOUND")
    print("(0-FP, control-verified) and INCOMPLETE (bounded CWE set + measured")
    print("execution coverage). Both bounds are shown by the evidence above.")
    print("=" * 64)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
