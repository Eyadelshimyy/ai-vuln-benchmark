#!/usr/bin/env python3
"""VULCAN static-analyzer BASELINES.

Answers the obvious reviewer question -- "why not just run a static analyzer?"
-- by scoring Bandit and Semgrep on the SAME functions, against the SAME
execution ground truth VULCAN's LLM detectors are graded on. One comparable
table: per-tool TP/FP/FN/TN, precision, recall, and per-CWE recall.

The point is not that static tools are bad -- it is to MEASURE where they stand
relative to execution-confirmed vulnerabilities. In particular, pattern-based
tools have no taint tracking, so they tend to miss CWE-22 path traversal (the
dangerous value only looks dangerous once you follow attacker input to open()),
while catching shell/eval patterns. Execution adjudicates either way.

Ground truth comes from a corpus_census.json (execution: VULN = triggered,
SAFE = ran-but-not-triggered, UNRESOLVED excluded -- same rule as the detector
matrices). No LLM, no network beyond Semgrep's one-time rule fetch.

USAGE
  pip install bandit semgrep
  python3 analysis/baselines.py --root calibration/mined_prompts \
      --census corpus_census_why3.json --tools bandit,semgrep --out baselines.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

OUR_CWES = {"CWE-22", "CWE-78", "CWE-89", "CWE-94", "CWE-502", "CWE-611", "CWE-918"}

# Bandit test-id -> CWE (for versions whose JSON lacks issue_cwe). Note there is
# deliberately NO CWE-22 entry: Bandit has no taint-based path-traversal check,
# so it structurally cannot flag our largest class -- an informative baseline
# result, not a bug here.
_BANDIT_CWE = {
    "B602": "CWE-78", "B603": "CWE-78", "B604": "CWE-78", "B605": "CWE-78",
    "B606": "CWE-78", "B607": "CWE-78", "B609": "CWE-78",
    "B608": "CWE-89",
    "B301": "CWE-502", "B614": "CWE-502",
    "B102": "CWE-94", "B307": "CWE-94",
    "B313": "CWE-611", "B314": "CWE-611", "B405": "CWE-611", "B406": "CWE-611",
    "B407": "CWE-611", "B408": "CWE-611", "B409": "CWE-611", "B410": "CWE-611", "B319": "CWE-611",
    "B310": "CWE-918",
}

import re as _re
_CWE_RE = _re.compile(r"CWE-\d+")


def _matrix(pairs):
    tp = sum(1 for t, p in pairs if t == "VULN" and p == "vulnerable")
    fn = sum(1 for t, p in pairs if t == "VULN" and p == "safe")
    fp = sum(1 for t, p in pairs if t == "SAFE" and p == "vulnerable")
    tn = sum(1 for t, p in pairs if t == "SAFE" and p == "safe")
    n = tp + fn + fp + tn

    def r(a, b):
        return round(a / b, 3) if b else None
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, n=n, precision=r(tp, tp + fp),
                recall=r(tp, tp + fn), FPR=r(fp, fp + tn), accuracy=r(tp + tn, n))


def run_bandit(src_file: Path) -> tuple[bool, set]:
    """Return (flagged_in_our_cwes, set_of_claimed_cwes)."""
    try:
        out = subprocess.run(["bandit", "-f", "json", "-q", str(src_file)],
                             capture_output=True, text=True, timeout=60).stdout
        data = json.loads(out or "{}")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return False, set()
    cwes = set()
    for r in data.get("results", []):
        cwe = None
        ic = r.get("issue_cwe")
        if isinstance(ic, dict) and ic.get("id"):
            cwe = f"CWE-{ic['id']}"
        if cwe not in OUR_CWES:
            cwe = _BANDIT_CWE.get(r.get("test_id"))
        if cwe in OUR_CWES:
            cwes.add(cwe)
    return (len(cwes) > 0), cwes


def run_semgrep(src_file: Path, config: str) -> tuple[bool, set]:
    try:
        out = subprocess.run(["semgrep", "scan", "--config", config, "--json", "--quiet",
                              "--timeout", "30", str(src_file)],
                             capture_output=True, text=True, timeout=180).stdout
        data = json.loads(out or "{}")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return False, set()
    cwes = set()
    for r in data.get("results", []):
        meta = (r.get("extra", {}) or {}).get("metadata", {}) or {}
        raw = meta.get("cwe")
        items = raw if isinstance(raw, list) else [raw]
        for it in items:
            m = _CWE_RE.search(str(it or ""))
            if m and m.group(0) in OUR_CWES:
                cwes.add(m.group(0))
    return (len(cwes) > 0), cwes


def _tool_available(name: str) -> bool:
    try:
        subprocess.run([name, "--version"], capture_output=True, timeout=20)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--census", required=True, type=Path,
                    help="a corpus_census.json (the execution ground truth)")
    ap.add_argument("--tools", default="bandit,semgrep")
    ap.add_argument("--label", choices=["confirmed", "triggered"], default="confirmed",
                    help="ground-truth positive: execution-confirmed (default, defensible) "
                         "or static-triggered (candidate). Negatives are always executed-no-trigger; "
                         "triggered-unconfirmed and non-executable are excluded as inconclusive.")
    ap.add_argument("--semgrep-config", default="p/python",
                    help="Semgrep ruleset (needs a one-time network fetch); e.g. p/python, p/security-audit")
    ap.add_argument("--out", type=Path, default=Path("baselines.json"))
    args = ap.parse_args()

    census = json.loads(args.census.read_text())
    # ground truth per candidate FILE (relative to root), from execution
    truth = {}
    for row in census.get("rows", []):
        pos = row.get("confirmed") if args.label == "confirmed" else row.get("triggered")
        if pos:
            truth[row["file"]] = ("VULN", row.get("exec_cwe"))
        elif row.get("executable") and not row.get("triggered"):
            truth[row["file"]] = ("SAFE", None)
        # inconclusive (triggered-unconfirmed when label=confirmed, or non-executable) -> excluded
    print(f"ground truth: {sum(1 for v in truth.values() if v[0]=='VULN')} VULN, "
          f"{sum(1 for v in truth.values() if v[0]=='SAFE')} SAFE "
          f"(from {args.census.name})", file=sys.stderr)

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    avail = {t: _tool_available("semgrep" if t == "semgrep" else t) for t in tools}
    for t, ok in avail.items():
        if not ok:
            print(f"  !! {t} not installed -- skipping (pip install {t})", file=sys.stderr)
    tools = [t for t in tools if avail[t]]
    if not tools:
        raise SystemExit("no baseline tools available; pip install bandit semgrep")

    # one row per scored candidate per tool: {truth, exec_cwe, flagged}
    scored = {t: [] for t in tools}
    n = 0
    for p in sorted(args.root.glob("*/*.json")):
        rel = str(p.relative_to(args.root))
        if rel not in truth:
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        src = rec.get("ground_truth_source")
        if not src:
            continue
        n += 1
        t_truth, exec_cwe = truth[rel]
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as tf:
            tf.write(src)
            tf.flush()
            for t in tools:
                flagged, cwes = (run_bandit(Path(tf.name)) if t == "bandit"
                                 else run_semgrep(Path(tf.name), args.semgrep_config))
                scored[t].append({"truth": t_truth, "exec_cwe": exec_cwe,
                                  "verdict": "vulnerable" if flagged else "safe", "claimed": sorted(cwes)})
        if n % 50 == 0:
            print(f"  {n} scored ...", file=sys.stderr, flush=True)

    # ---- report ----
    print("\n" + "=" * 72)
    print("STATIC-ANALYZER BASELINES vs VULCAN execution ground truth")
    print("=" * 72)
    print(f"{n} functions scored (same set + same ground truth as the LLM detectors)\n")
    classes = sorted({r["exec_cwe"] for t in tools for r in scored[t]
                      if r["truth"] == "VULN" and r["exec_cwe"]})
    report = {"n": n, "tools": {}}
    for t in tools:
        rows = scored[t]
        m = _matrix([(r["truth"], r["verdict"]) for r in rows])
        print(f"[{t}]")
        print(f"  TP {m['TP']}  FN {m['FN']}  FP {m['FP']}  TN {m['TN']}  (n={m['n']})")
        print(f"  precision {m['precision']}  recall {m['recall']}  FPR {m['FPR']}  accuracy {m['accuracy']}")
        percwe = {}
        for X in classes:
            vulns_X = [r for r in rows if r["truth"] == "VULN" and r["exec_cwe"] == X]
            if not vulns_X:
                continue
            caught = sum(1 for r in vulns_X if r["verdict"] == "vulnerable")
            percwe[X] = {"recall": round(caught / len(vulns_X), 3), "caught": caught, "n": len(vulns_X)}
            print(f"     {X}: recall {percwe[X]['recall']} ({caught}/{len(vulns_X)})")
        report["tools"][t] = {"matrix": m, "per_cwe_recall": percwe}
        print()
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Saved {args.out}\n")
    print("Compare these recalls to the LLM detectors on the SAME ground truth. The expected\n"
          "headline: pattern analyzers miss taint-dependent classes (esp. CWE-22 path traversal)\n"
          "that execution confirms -- the empirical 'why not just static analysis?' answer.")


if __name__ == "__main__":
    main()
