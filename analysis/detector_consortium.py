#!/usr/bin/env python3
"""VULCAN detector consortium: evaluate a PANEL of AI models as vulnerability
detectors against VULCAN's execution ground truth (RQ2).

The models are CONTESTANTS, never judges -- execution adjudicates, each model's
yes/no prediction is scored against it. We report precision / recall / F1 / FPR
per model and per CWE, plus a majority-vote "consortium" detector, and (for free)
the static baselines if their prediction files are passed in.

Ground-truth labels (execution-adjudicated):
  positive  = triggered  (Tier-1)  [or --label confirmed for Tier-2]
  negative  = executed but did NOT trigger (verdict NOT_TRIGGERED)
  excluded  = COULD_NOT_EXECUTE  -- inconclusive is NOT the same as safe

Every (model, finding) answer is cached under --cache, so a re-run costs nothing
and an interrupted run resumes. Use --limit / --cwe / --balance to control spend.

  python3 analysis/detector_consortium.py \
      --models models.json --census corpus_census.json \
      --root calibration/mined_prompts --balance 2 --cache .consortium_cache

models.json:
  [ {"name":"gpt-4o-mini","base_url":"https://api.openai.com/v1",
     "model":"gpt-4o-mini","api_key_env":"OPENAI_API_KEY"},
    {"name":"opencode-x","base_url":"https://.../v1","model":"x","api_key_env":"OPENCODE_KEY"} ]
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

PROMPT = """You are a security code reviewer. Decide whether the following Python \
function contains a real, exploitable security vulnerability in one of these \
classes: CWE-22 path traversal, CWE-78 OS command injection, CWE-89 SQL \
injection, CWE-611 XXE, CWE-918 SSRF, CWE-502 unsafe deserialization.

Judge only the function as written. Answer ONLY with a JSON object, no prose:
{"vulnerable": true|false, "cwe": "CWE-78"|null, "confidence": 0.0-1.0, "reason": "<=15 words"}

FUNCTION (%s):
```python
%s
```"""


def _call_openai(cfg: dict, source: str, fn: str, timeout: float) -> dict:
    """One detection call via an OpenAI-compatible endpoint. Returns parsed dict
    or {'error': ...}. Requires the `openai` package."""
    from openai import OpenAI
    key = cfg.get("api_key") or os.environ.get(cfg.get("api_key_env", ""), "")
    client = OpenAI(api_key=key or "none", base_url=cfg.get("base_url"))
    try:
        r = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": PROMPT % (fn, source[:6000])}],
            temperature=0, max_tokens=200, timeout=timeout,
        )
        txt = r.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}
    return _parse(txt)


def _parse(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {"vulnerable": None, "raw": txt[:200]}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"vulnerable": None, "raw": txt[:200]}
    cwe = d.get("cwe")
    if isinstance(cwe, str):
        mm = re.search(r"CWE-\d+", cwe.upper())
        cwe = mm.group(0) if mm else None
    return {"vulnerable": bool(d.get("vulnerable")), "cwe": cwe,
            "confidence": d.get("confidence"), "reason": str(d.get("reason", ""))[:120]}


def _mock(cfg, source, fn, timeout):
    """Deterministic stand-in so the SCORING can be validated with no API key:
    flags anything whose source names a risky sink. Not a real detector."""
    risky = any(s in source for s in ("subprocess", "os.system", "shell=True", "execute(",
                                      "urlopen", "requests.get", "httpx.get", "open(",
                                      "etree", "pickle.loads", "yaml.load"))
    h = int(hashlib.md5((cfg["name"] + fn).encode()).hexdigest(), 16)
    vuln = risky and (h % 10 < 7)   # 70% of risky-looking ones, to create FN/FP spread
    return {"vulnerable": vuln, "cwe": None, "confidence": 0.5, "reason": "mock"}


def build_eval_set(rows, root: Path, label: str):
    items = []
    for r in rows:
        v = r.get("verdict")
        if v == "COULD_NOT_EXECUTE":
            continue                       # inconclusive -> excluded, never "safe"
        pos = bool(r.get("confirmed")) if label == "confirmed" else bool(r.get("triggered"))
        # negatives: executed and did not trigger
        if not pos and r.get("triggered"):
            continue                       # triggered but using 'confirmed' label: skip ambiguous
        cand = root / r["file"]
        try:
            cj = json.loads(cand.read_text())
            src = cj.get("ground_truth_source") or cj.get("module_source") or ""
            fn = cj.get("function", "?")
        except (OSError, json.JSONDecodeError):
            continue
        if not src.strip():
            continue
        items.append({"file": r["file"], "function": fn, "source": src,
                      "cwes": r.get("cwes") or [], "label": pos})
    return items


def score(preds: dict, items: list):
    """preds: file -> bool prediction. Returns metrics dict."""
    TP = FP = FN = TN = 0
    per_cwe = defaultdict(lambda: [0, 0])
    for it in items:
        p = preds.get(it["file"])
        if p is None:
            continue
        y = it["label"]
        if y:
            for c in it["cwes"]:
                per_cwe[c][1] += 1
                if p:
                    per_cwe[c][0] += 1
        TP += y and p; FN += y and not p; FP += (not y) and p; TN += (not y) and not p
    prec = TP / (TP + FP) if (TP + FP) else None
    rec = TP / (TP + FN) if (TP + FN) else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if (prec is not None and rec is not None) else None)
    fpr = FP / (FP + TN) if (FP + TN) else None
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
            "per_cwe": {c: {"num": v[0], "den": v[1]} for c, v in per_cwe.items()}}


def fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, help="models.json (omit with --mock to dry-run)")
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--label", choices=["triggered", "confirmed"], default="triggered")
    ap.add_argument("--balance", type=float, default=0, help="negatives per positive (0 = use all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cwe", default=None)
    ap.add_argument("--cache", type=Path, default=Path(".consortium_cache"))
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--mock", action="store_true", help="no API; deterministic stand-in to test scoring")
    ap.add_argument("--out", type=Path, default=Path("consortium_results.json"))
    args = ap.parse_args()

    cen = json.loads(args.census.read_text())
    rows = cen["rows"] if isinstance(cen, dict) else cen
    items = build_eval_set(rows, args.root, args.label)
    if args.cwe:
        items = [it for it in items if args.cwe in it["cwes"] or not it["label"]]
    pos = [it for it in items if it["label"]]
    neg = [it for it in items if not it["label"]]
    if args.balance and neg:
        random.seed(0)
        neg = random.sample(neg, min(len(neg), int(len(pos) * args.balance)))
    items = pos + neg
    if args.limit:
        items = items[: args.limit]
    print(f"eval set: {len(pos)} positive ({args.label}) + {len(neg)} negative = {len(items)} "
          f"(COULD_NOT_EXECUTE excluded)", file=sys.stderr)

    models = [{"name": "mock", "mock": True}] if args.mock else json.loads(args.models.read_text())
    args.cache.mkdir(parents=True, exist_ok=True)

    model_preds = {}
    for cfg in models:
        name = cfg["name"]
        preds, raw = {}, {}
        for i, it in enumerate(items, 1):
            ck = args.cache / f"{name}__{hashlib.md5(it['file'].encode()).hexdigest()}.json"
            if ck.exists():
                ans = json.loads(ck.read_text())
            else:
                ans = _mock(cfg, it["source"], it["function"], args.timeout) if (args.mock or cfg.get("mock")) \
                    else _call_openai(cfg, it["source"], it["function"], args.timeout)
                ck.write_text(json.dumps(ans))
                if not (args.mock or cfg.get("mock")):
                    time.sleep(cfg.get("sleep", 0.2))
            raw[it["file"]] = ans
            if ans.get("vulnerable") is not None:
                preds[it["file"]] = bool(ans["vulnerable"])
            if i % 50 == 0:
                print(f"  [{name}] {i}/{len(items)}", file=sys.stderr)
        model_preds[name] = preds
        errs = sum(1 for a in raw.values() if a.get("error"))
        if errs:
            print(f"  [{name}] {errs} call errors (see cache)", file=sys.stderr)

    # majority-vote consortium
    if len(model_preds) > 1:
        consensus = {}
        for it in items:
            votes = [model_preds[m].get(it["file"]) for m in model_preds]
            votes = [v for v in votes if v is not None]
            if votes:
                consensus[it["file"]] = sum(votes) > len(votes) / 2
        model_preds["CONSORTIUM(majority)"] = consensus

    results = {name: score(p, items) for name, p in model_preds.items()}
    out = {"label": args.label, "n_pos": len(pos), "n_neg": len(neg),
           "models": results}
    args.out.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print(f"  DETECTOR CONSORTIUM vs VULCAN execution ground truth  (label={args.label})")
    print(f"  {len(pos)} positive / {len(neg)} negative   (inconclusive excluded)")
    print("=" * 78)
    print(f"  {'detector':26} {'recall':>7} {'prec':>7} {'F1':>7} {'FPR':>7}")
    print("  " + "-" * 60)
    for name, m in sorted(results.items(), key=lambda kv: (kv[1]['f1'] is None, -(kv[1]['f1'] or 0))):
        print(f"  {name[:26]:26} {fmt(m['recall']):>7} {fmt(m['precision']):>7} "
              f"{fmt(m['f1']):>7} {fmt(m['fpr']):>7}")
    print(f"\n  Saved {args.out}")
    print("  (add Bandit/semgrep rows from their own prediction files for one table.)")


if __name__ == "__main__":
    main()
