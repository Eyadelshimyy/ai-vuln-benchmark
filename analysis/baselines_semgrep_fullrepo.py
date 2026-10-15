#!/usr/bin/env python3
"""Fair semgrep baseline: scan each repo's FULL source (not isolated mined
snippets) and map findings back to the mined candidate functions.

Why: semgrep's dataflow rules need whole-file / cross-function context to fire.
Running it per-mined-snippet (as the quick baseline does) starves it of that
context and it reports ~nothing -- which understates semgrep and reads as a
broken run. This scans `<repo>/_repo_src` (the vendored full tree) once per repo,
then marks a mined function "flagged" if any semgrep finding lands inside that
function's file+line span. Same census ground truth, same functions, so the
recall/precision are directly comparable to Bandit and the LLM detectors.

USAGE
  python3 analysis/baselines_semgrep_fullrepo.py \
      --census corpus_census_FROZEN.json \
      --root calibration/mined_prompts \
      --config p/security-audit \
      --out semgrep_fullrepo.json
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def _span(cand_json: dict) -> int:
    src = cand_json.get("ground_truth_source") or ""
    return max(1, len([ln for ln in src.splitlines()]))


def _run_semgrep(scan_dir: Path, config: str, timeout: int, semgrep_cmd: str = "semgrep") -> list[dict]:
    """Return semgrep results: list of {path, start, end, cwe}."""
    cmd = shlex.split(semgrep_cmd) + ["scan", "--config", config, "--json", "--quiet",
           "--timeout", "0", "--metrics", "off", str(scan_dir)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"    (semgrep timed out on {scan_dir.name})", file=sys.stderr)
        return []
    if not p.stdout.strip():
        if p.returncode != 0:
            print(f"    (semgrep error on {scan_dir.name}: {p.stderr.strip()[:160]})", file=sys.stderr)
        return []
    try:
        data = json.loads(p.stdout)
    except json.JSONDecodeError:
        return []
    out = []
    for r in data.get("results", []):
        meta = (r.get("extra") or {}).get("metadata") or {}
        cwe = meta.get("cwe")
        if isinstance(cwe, list):
            cwe = cwe[0] if cwe else None
        out.append({"path": r.get("path", ""),
                    "start": (r.get("start") or {}).get("line", 0),
                    "end": (r.get("end") or {}).get("line", 0),
                    "cwe": cwe})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--config", default="p/security-audit")
    ap.add_argument("--label", choices=["confirmed","triggered"], default="confirmed",
                    help="ground-truth positive: execution-confirmed (default, defensible) or static-triggered (candidate)")
    ap.add_argument("--semgrep-cmd", default="semgrep",
                    help='command that runs semgrep; e.g. for Docker: "docker run --rm -v $PWD:/src -w /src semgrep/semgrep semgrep"')
    ap.add_argument("--out", type=Path, default=Path("semgrep_fullrepo.json"))
    ap.add_argument("--timeout", type=int, default=600, help="per-repo semgrep timeout (s)")
    args = ap.parse_args()

    data = json.loads(args.census.read_text())
    rows = data["rows"] if isinstance(data, dict) else data

    # group candidates by repo dir
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["file"].split("/")[0]].append(r)

    TP = FP = FN = TN = 0
    per_cwe = defaultdict(lambda: [0, 0])   # cwe -> [caught_vuln, total_vuln]
    predictions = []
    repos = sorted(by_repo)
    for i, repo in enumerate(repos, 1):
        repo_src = args.root / repo / "_repo_src"
        print(f"[{i}/{len(repos)}] semgrep {repo} ...", file=sys.stderr)
        findings = _run_semgrep(repo_src, args.config, args.timeout, args.semgrep_cmd) if repo_src.exists() else []

        for r in by_repo[repo]:
            candp = args.root / r["file"]
            try:
                cj = json.loads(candp.read_text())
            except (OSError, json.JSONDecodeError):
                cj = {}
            cand_file = (cj.get("file") or "").replace("\\", "/")
            start = cj.get("line", 0) or 0
            end = start + _span(cj) if start else 0
            # predicted VULN if any finding overlaps this function's file+line span.
            # Path match: prefer the FULL relative-path suffix (unambiguous). Only
            # fall back to basename when the vendored path was restructured AND that
            # basename is unique among this repo's findings -- otherwise a hit in one
            # `main.py` could be mis-credited to a mined function in a *different*
            # `main.py` in the same repo (e.g. mini-pi's several main.py).
            cand_base = cand_file.split("/")[-1] if cand_file else ""
            matched = [f for f in findings
                       if cand_file and f["path"].replace("\\", "/").endswith(cand_file)]
            if not matched and cand_base:
                same_base = [f for f in findings
                             if f["path"].replace("\\", "/").split("/")[-1] == cand_base]
                if len({f["path"] for f in same_base}) == 1:   # unambiguous only
                    matched = same_base
            flagged = False
            for f in matched:
                if start == 0 or (f["start"] <= end and f["end"] >= start):
                    flagged = True
                    break
            # Ground-truth label. Reviewers' key point: static 'triggered' is a
            # CANDIDATE set, not ground truth. The defensible label set is
            # execution-CONFIRMED (positive) vs executed-but-no-trigger (negative);
            # 'triggered-unconfirmed' and 'non-executable' are INCONCLUSIVE and are
            # EXCLUDED (an unexecuted function is not a proven-safe negative).
            if args.label == "confirmed":
                if r.get("confirmed"):
                    is_vuln = True
                elif r.get("executable") and not r.get("triggered"):
                    is_vuln = False
                else:
                    continue   # inconclusive -> excluded from the detector metric
            else:
                is_vuln = bool(r.get("triggered"))
            if is_vuln:
                for c in (r.get("cwes") or []):
                    per_cwe[c][1] += 1
                    if flagged:
                        per_cwe[c][0] += 1
            if is_vuln and flagged:
                TP += 1
            elif is_vuln and not flagged:
                FN += 1
            elif (not is_vuln) and flagged:
                FP += 1
            else:
                TN += 1
            predictions.append({"file": r["file"], "vuln": is_vuln, "flagged": flagged})

    n = TP + FP + FN + TN
    prec = TP / (TP + FP) if (TP + FP) else None
    rec = TP / (TP + FN) if (TP + FN) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if (prec is not None and rec is not None) else None)
    fpr = FP / (FP + TN) if (FP + TN) else None
    acc = (TP + TN) / n if n else None
    args.out.write_text(json.dumps(
        {"tool": "semgrep-fullrepo", "config": args.config,
         "TP": TP, "FP": FP, "FN": FN, "TN": TN,
         "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "accuracy": acc,
         "per_cwe": {c: {"recall_num": v[0], "recall_den": v[1]} for c, v in per_cwe.items()},
         "predictions": predictions}, indent=2))

    print("\n" + "=" * 68)
    print("SEMGREP FULL-REPO BASELINE vs VULCAN execution ground truth")
    print("=" * 68)
    print(f"{n} functions scored  |  config={args.config}")
    print(f"[semgrep-fullrepo]")
    print(f"  TP {TP}  FN {FN}  FP {FP}  TN {TN}")
    print(f"  precision {prec if prec is None else round(prec,3)}  "
          f"recall {rec if rec is None else round(rec,3)}  "
          f"F1 {f1 if f1 is None else round(f1,3)}  "
          f"FPR {fpr if fpr is None else round(fpr,3)}  "
          f"accuracy {acc if acc is None else round(acc,3)}")
    for c in sorted(per_cwe):
        num, den = per_cwe[c]
        print(f"  {c}: recall {round(num/den,3) if den else 0.0} ({num}/{den})")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
