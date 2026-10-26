#!/usr/bin/env python3
"""VULCAN detector consortium: evaluate a PANEL of AI models as vulnerability
detectors against VULCAN's execution ground truth (RQ2).

The models are CONTESTANTS, never judges -- execution adjudicates, each model's
yes/no prediction is scored against it. Publication-grade analysis (all computed
from cached votes, NO extra API cost):

  * per-model + consortium precision / recall / F1 (bootstrap CI) / FPR, per-CWE
  * PREVALENCE-AWARE precision: reported at the benchmark balance AND projected
    to the real corpus prevalence (precision depends on prevalence; recall/FPR
    do not) -- benchmark precision is NOT deployment precision.
  * aggregation rules: majority / unanimous / any (precision-recall trade-off)
  * does the consortium beat the best individual?  McNemar's exact test +
    paired bootstrap CI on the F1 difference (tested, not eyeballed).
  * correlated-error analysis: pairwise Cohen's kappa + how often all models
    err together (so majority-vote gains are shown empirically, not assumed).

Ground-truth labels (execution-adjudicated):
  positive  = triggered (Tier-1)  [or --label confirmed for Tier-2]
  negative  = executed and verdict == NOT_TRIGGERED (STRICT; nothing else is "safe")
  excluded  = COULD_NOT_EXECUTE, triggered-unconfirmed, any other verdict

Integrity: experiment-aware cache key (model id + PROMPT_VERSION + source hash);
STRICT JSON parse (`vulnerable` must be a real boolean, else invalid/excluded);
headline metrics + consortium on the COMMON set every model answered; coverage
reported separately; prompt stays OPEN-ENDED over the 6 classes (no CWE hint ->
no answer leak). Every answer cached under --cache -> resumable, re-run is free.

  python3 analysis/detector_consortium.py --models models.json \
      --census corpus_census_P5_final.json --root calibration/mined_prompts \
      --label confirmed --balance 2 --cache .consortium_cache --out consortium_confirmed_P5.json
"""
from __future__ import annotations
import subprocess
import argparse, hashlib, json, math, os, random, re, sys, time
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

PROMPT_VERSION = "v1-openended-6cwe"   # bump to invalidate the cache after a prompt change

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


# --------------------------------------------------------------------------- #
# model adapters
# --------------------------------------------------------------------------- #
def _call_openai(cfg: dict, source: str, fn: str, timeout: float) -> dict:
    from openai import OpenAI
    key = cfg.get("api_key") or os.environ.get(cfg.get("api_key_env", ""), "")
    client = OpenAI(api_key=key or "none", base_url=cfg.get("base_url"))
    try:
        r = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": PROMPT % (fn, source[:6000])}],
            temperature=0, max_tokens=200, timeout=timeout,
        )
        ans = _parse(r.choices[0].message.content or "")
        try:
            ans["tokens"] = r.usage.total_tokens  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
        return ans
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200], "vulnerable": None}


def _call_opencode_cli(cfg, source, fn, timeout):
    prompt = PROMPT % (fn, source[:6000])
    cmd = ["opencode", "run", "-m", cfg["model"]]
    if cfg.get("agent"):
        cmd += ["--agent", cfg["agent"]]
    cmd.append(prompt)
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "cli timeout", "vulnerable": None}
    out = pr.stdout or ""
    blocks = re.findall(r"\{[^{}]*\}", out)
    if not blocks:
        return {"error": (pr.stderr or out).strip()[:200] or "refusal/no-json", "vulnerable": None}
    return _parse(blocks[-1])


def _parse(txt: str) -> dict:
    """STRICT: `vulnerable` must be a real JSON boolean, else INVALID."""
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {"vulnerable": None, "invalid": "no-json", "raw": txt[:200]}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"vulnerable": None, "invalid": "bad-json", "raw": txt[:200]}
    v = d.get("vulnerable")
    if not isinstance(v, bool):
        return {"vulnerable": None, "invalid": "vulnerable-not-bool", "raw": txt[:200]}
    cwe = d.get("cwe")
    if isinstance(cwe, str):
        mm = re.search(r"CWE-\d+", cwe.upper()); cwe = mm.group(0) if mm else None
    else:
        cwe = None
    conf = d.get("confidence")
    if not isinstance(conf, (int, float)) or isinstance(conf, bool):
        conf = None
    return {"vulnerable": v, "cwe": cwe, "confidence": conf, "reason": str(d.get("reason", ""))[:120]}


def _mock(cfg, source, fn, timeout):
    risky = any(s in source for s in ("subprocess", "os.system", "shell=True", "execute(",
                                      "urlopen", "requests.get", "httpx.get", "open(",
                                      "etree", "pickle.loads", "yaml.load"))
    h = int(hashlib.md5((cfg["name"] + fn).encode()).hexdigest(), 16)
    return {"vulnerable": bool(risky and (h % 10 < 7)), "cwe": None, "confidence": 0.5, "reason": "mock"}


# --------------------------------------------------------------------------- #
# eval set + scoring
# --------------------------------------------------------------------------- #
def build_eval_set(rows, root: Path, label: str):
    items = []
    for r in rows:
        v = r.get("verdict")
        is_pos = bool(r.get("confirmed")) if label == "confirmed" else bool(r.get("triggered"))
        if is_pos:
            lab = True
        elif v == "NOT_TRIGGERED":
            lab = False
        else:
            continue
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
                      "cwes": r.get("cwes") or [], "label": lab})
    return items


def _cache_key(cache: Path, name: str, model_id: str, it: dict) -> Path:
    src_h = hashlib.md5(it["source"].encode("utf-8", "replace")).hexdigest()
    sig = hashlib.md5("\x00".join([name, str(model_id), PROMPT_VERSION, it["file"], src_h]).encode()).hexdigest()
    return cache / f"{name}__{sig}.json"


def score(preds: dict, items: list):
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
        TP += int(y and p); FN += int(y and not p)
        FP += int((not y) and p); TN += int((not y) and not p)
    prec = TP / (TP + FP) if (TP + FP) else None
    rec = TP / (TP + FN) if (TP + FN) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else \
        (0.0 if (prec is not None and rec is not None) else None)
    fpr = FP / (FP + TN) if (FP + TN) else None
    den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = ((TP * TN - FP * FN) / den) if den else None   # Matthews corr: robust under imbalance
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "mcc": mcc,
            "per_cwe": {c: {"num": v[0], "den": v[1]} for c, v in per_cwe.items()}}


# ---- statistics ------------------------------------------------------------ #
def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _f1_of(pairs):
    TP = FP = FN = 0
    for y, p in pairs:
        if y and p: TP += 1
        elif (not y) and p: FP += 1
        elif y and (not p): FN += 1
    pr = TP / (TP + FP) if (TP + FP) else 0.0
    rc = TP / (TP + FN) if (TP + FN) else 0.0
    return (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0


def bootstrap_f1_ci(pairs, n_boot=2000, seed=0):
    if not pairs:
        return (None, None, None)
    rng = random.Random(seed); N = len(pairs)
    boots = sorted(_f1_of([pairs[rng.randrange(N)] for _ in range(N)]) for _ in range(n_boot))
    return (_f1_of(pairs), boots[int(0.025 * n_boot)], boots[min(n_boot - 1, int(0.975 * n_boot))])


def bootstrap_f1_diff_ci(pairs_a, pairs_b, n_boot=2000, seed=0):
    """Paired bootstrap of F1(a)-F1(b) over the SAME resampled cases."""
    if not pairs_a or len(pairs_a) != len(pairs_b):
        return (None, None, None)
    rng = random.Random(seed); N = len(pairs_a)
    point = _f1_of(pairs_a) - _f1_of(pairs_b)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(N) for _ in range(N)]
        diffs.append(_f1_of([pairs_a[i] for i in idx]) - _f1_of([pairs_b[i] for i in idx]))
    diffs.sort()
    return (point, diffs[int(0.025 * n_boot)], diffs[min(n_boot - 1, int(0.975 * n_boot))])


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar on paired correctness vectors. Returns dict."""
    b01 = sum(1 for x, y in zip(correct_a, correct_b) if (not x) and y)
    b10 = sum(1 for x, y in zip(correct_a, correct_b) if x and (not y))
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p": 1.0}
    k = min(b01, b10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"b01": b01, "b10": b10, "p": p}


def cohens_kappa(a, b, files):
    n = len(files)
    if not n:
        return None
    po = sum(1 for f in files if a[f] == b[f]) / n
    pa = sum(1 for f in files if a[f]) / n
    pb = sum(1 for f in files if b[f]) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


def fleiss_kappa_binary(vote_counts, n_raters):
    """Fleiss' kappa for binary ratings. vote_counts = #yes per item (all items rated by n_raters)."""
    N = len(vote_counts)
    if N == 0 or n_raters < 2:
        return None
    Pi = [((y * y + (n_raters - y) ** 2) - n_raters) / (n_raters * (n_raters - 1)) for y in vote_counts]
    Pbar = sum(Pi) / N
    p_yes = sum(vote_counts) / (N * n_raters)
    Pe = p_yes ** 2 + (1 - p_yes) ** 2
    return (Pbar - Pe) / (1 - Pe) if (1 - Pe) else 1.0


def precision_at(recall, fpr, pi):
    """Projected precision at prevalence pi, from prevalence-free recall & FPR."""
    if recall is None or fpr is None:
        return None
    den = recall * pi + fpr * (1 - pi)
    return (recall * pi / den) if den else None


# ---- formatting ------------------------------------------------------------ #
def fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def ci_str(k, n):
    lo, hi = wilson(k, n)
    return "   -        " if lo is None else f"{k/n:.3f} [{lo:.2f}-{hi:.2f}]"


def trip_str(t):
    return "   -        " if (not t or t[0] is None) else f"{t[0]:.3f} [{t[1]:.2f}-{t[2]:.2f}]"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, help="models.json (omit with --mock to dry-run)")
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--label", choices=["triggered", "confirmed"], default="triggered")
    ap.add_argument("--balance", type=float, default=0, help="negatives per positive (0 = use all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pilot", type=int, default=0, help="PILOT: N stratified cases, reliability only")
    ap.add_argument("--cwe", default=None)
    ap.add_argument("--cache", type=Path, default=Path(".consortium_cache"))
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--min-cwe-n", type=int, default=5)
    ap.add_argument("--prevalence", type=float, default=None,
                    help="deployment prevalence for projected precision (default: confirmed/executable from census)")
    ap.add_argument("--freeze", type=Path, default=None,
                    help="write the frozen eval manifest (file,label,cwes) here for reproducibility")
    ap.add_argument("--verify-only", action="store_true",
                    help="build + verify the frozen eval set and exit (no model calls)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("consortium_results.json"))
    args = ap.parse_args()

    cen = json.loads(args.census.read_text())
    rows = cen["rows"] if isinstance(cen, dict) else cen

    # deployment prevalence (confirmed among executable) for projected precision
    n_exec = sum(1 for r in rows if r.get("verdict") != "COULD_NOT_EXECUTE")
    n_conf = sum(1 for r in rows if r.get("confirmed"))
    pi = args.prevalence if args.prevalence is not None else (n_conf / n_exec if n_exec else 0.0)

    items = build_eval_set(rows, args.root, args.label)
    if args.cwe:
        items = [it for it in items if args.cwe in it["cwes"] or not it["label"]]
    pos = [it for it in items if it["label"]]
    neg = [it for it in items if not it["label"]]
    if args.balance and neg:
        neg = random.Random(0).sample(neg, min(len(neg), int(len(pos) * args.balance)))
    items = pos + neg

    if args.pilot:
        h = max(1, min(len(pos), args.pilot // 2))
        pil_pos = random.Random(1).sample(pos, min(h, len(pos))) if pos else []
        pil_neg = random.Random(2).sample(neg, min(args.pilot - len(pil_pos), len(neg))) if neg else []
        items = pil_pos + pil_neg
    elif args.limit and args.limit < len(items):
        frac = args.limit / len(items)
        lp = random.Random(3).sample(pos, min(len(pos), max(1, round(len(pos) * frac)))) if pos else []
        ln = random.Random(4).sample(neg, max(0, min(len(neg), args.limit - len(lp)))) if neg else []
        items = lp + ln

    # ---- frozen-set verification (publication: audit before spending) ----
    files = [it["file"] for it in items]
    dup = [f for f, c in Counter(files).items() if c > 1]
    npos = sum(1 for it in items if it["label"]); nneg = len(items) - npos
    print(f"eval set: {npos} positive ({args.label}) + {nneg} negative"
          + (f"  [PILOT {len(items)}]" if args.pilot else f" = {len(items)}")
          + "  (COULD_NOT_EXECUTE / triggered-unconfirmed excluded)", file=sys.stderr)
    print(f"  corpus prevalence (confirmed/executable) = {n_conf}/{n_exec} = {pi:.3%}", file=sys.stderr)
    assert not dup, f"FROZEN-SET ERROR: duplicate files: {dup[:5]}"
    # every negative must be a clean NOT_TRIGGERED row
    verdict_by_file = {r["file"]: r.get("verdict") for r in rows}
    bad_neg = [it["file"] for it in items if not it["label"] and verdict_by_file.get(it["file"]) != "NOT_TRIGGERED"]
    assert not bad_neg, f"FROZEN-SET ERROR: non-NOT_TRIGGERED negatives: {bad_neg[:5]}"
    print(f"  frozen-set checks OK: no duplicates; all {nneg} negatives verdict==NOT_TRIGGERED", file=sys.stderr)

    if args.freeze:
        args.freeze.write_text(json.dumps(
            {"label": args.label, "n_pos": npos, "n_neg": nneg, "prevalence_corpus": pi,
             "cases": [{"file": it["file"], "label": it["label"], "cwes": it["cwes"]} for it in items]},
            indent=2))
        print(f"  froze eval manifest -> {args.freeze}", file=sys.stderr)
    if args.verify_only:
        print("  --verify-only: eval set verified, no model calls made.", file=sys.stderr)
        return

    models_cfg = [{"name": "mock", "mock": True}] if args.mock else json.loads(args.models.read_text())
    args.cache.mkdir(parents=True, exist_ok=True)

    model_preds, model_stats, real_names = {}, {}, []
    for cfg in models_cfg:
        name = cfg["name"]; real_names.append(name)
        model_id = cfg.get("model", name)
        preds, raw = {}, {}
        for i, it in enumerate(items, 1):
            ck = _cache_key(args.cache, name, model_id, it)
            if ck.exists():
                ans = json.loads(ck.read_text())
            else:
                t0 = time.time()
                if args.mock or cfg.get("mock"):
                    ans = _mock(cfg, it["source"], it["function"], args.timeout)
                elif cfg.get("cli"):
                    ans = _call_opencode_cli(cfg, it["source"], it["function"], args.timeout)
                else:
                    ans = _call_openai(cfg, it["source"], it["function"], args.timeout)
                ans.setdefault("latency", round(time.time() - t0, 3))
                if "error" not in ans:
                    ck.write_text(json.dumps(ans))   # cache only real answers; transient errors retry on re-run
                if not (args.mock or cfg.get("mock")):
                    time.sleep(cfg.get("sleep", 0.2))
            raw[it["file"]] = ans
            if isinstance(ans.get("vulnerable"), bool):
                preds[it["file"]] = ans["vulnerable"]
            if i % 50 == 0:
                print(f"  [{name}] {i}/{len(items)}", file=sys.stderr)
        model_preds[name] = preds
        lat = [a["latency"] for a in raw.values() if isinstance(a.get("latency"), (int, float))]
        model_stats[name] = {
            "answered": len(preds), "total": len(items),
            "valid": sum(1 for a in raw.values() if isinstance(a.get("vulnerable"), bool)),
            "errors": sum(1 for a in raw.values() if a.get("error")),
            "median_latency": round(sorted(lat)[len(lat) // 2], 2) if lat else None,
            "tokens": sum(a.get("tokens", 0) for a in raw.values() if isinstance(a.get("tokens"), int)) or None}
        if model_stats[name]["errors"]:
            print(f"  [{name}] {model_stats[name]['errors']} call errors (see cache)", file=sys.stderr)

    if args.pilot:
        print("\n" + "=" * 70); print(f"  PILOT RUN  ({len(items)} cases)  --  NOT final results"); print("=" * 70)
        print(f"  {'model':26} {'answered':>9} {'valid-JSON':>11} {'errors':>7} {'med.lat':>8}")
        print("  " + "-" * 64)
        for name, s in model_stats.items():
            lat = "-" if s["median_latency"] is None else f"{s['median_latency']}s"
            print(f"  {name[:26]:26} {s['answered']:>4}/{s['total']:<4} {s['valid']:>6}/{s['total']:<4} {s['errors']:>7} {lat:>8}")
        print("\n  Check: answered≈total, valid-JSON high, errors 0. Fix ids before the full run.")
        print(f"  (cache: {args.cache} -- full run reuses these answers for free)")
        return

    # ---- COMMON set (fair comparison) ----
    common_files = set.intersection(*[set(model_preds[n]) for n in real_names]) if real_names else set()
    items_common = [it for it in items if it["file"] in common_files]
    dropped = len(items) - len(items_common)
    pos_c = sum(1 for it in items_common if it["label"]); neg_c = len(items_common) - pos_c

    # ---- aggregation strategies (free) ----
    if len(real_names) > 1:
        for rule in ("majority", "unanimous", "any"):
            agg = {}
            for it in items_common:
                votes = [model_preds[m][it["file"]] for m in real_names]
                if rule == "majority":
                    agg[it["file"]] = sum(votes) > len(votes) / 2
                elif rule == "unanimous":
                    agg[it["file"]] = all(votes)
                else:
                    agg[it["file"]] = any(votes)
            model_preds[f"CONSORTIUM({rule})"] = agg

    # ---- score everyone on the common set, with CIs + projected precision ----
    results = {}
    pairs_by = {}
    for name, p in model_preds.items():
        m = score(p, items_common)
        pr = [(it["label"], p[it["file"]]) for it in items_common if it["file"] in p]
        pairs_by[name] = pr
        m["f1_ci"] = bootstrap_f1_ci(pr)
        m["fpr_ci"] = wilson(m["FP"], m["FP"] + m["TN"])
        m["precision_bench"] = m["precision"]
        m["precision_at_corpus"] = precision_at(m["recall"], m["fpr"], pi)
        results[name] = m

    order = sorted(results.items(), key=lambda kv: (kv[1]["f1"] is None, -(kv[1]["f1"] or 0)))
    best_individual = next((n for n, _ in order if n in real_names), None)

    # ---- consortium vs best individual: McNemar + paired bootstrap F1 diff ----
    comparison = {}
    if best_individual and len(real_names) > 1:
        base_pairs = pairs_by[best_individual]
        base_correct = [y == p for y, p in base_pairs]
        for rule in ("majority", "unanimous", "any"):
            cname = f"CONSORTIUM({rule})"
            cpairs = pairs_by[cname]
            ccorrect = [y == p for y, p in cpairs]
            comparison[rule] = {
                "mcnemar": mcnemar_exact(base_correct, ccorrect),
                "f1_diff_ci": bootstrap_f1_diff_ci(cpairs, base_pairs),
                "f1_consortium": results[cname]["f1"], "f1_best_individual": results[best_individual]["f1"]}

    # ---- correlated-error analysis ----
    kappa = {}
    for i, a in enumerate(real_names):
        for b in real_names[i + 1:]:
            kappa[f"{a} vs {b}"] = cohens_kappa(model_preds[a], model_preds[b], common_files)
    corr = {}
    if len(real_names) > 1:
        pos_common = [it for it in items_common if it["label"]]
        neg_common = [it for it in items_common if not it["label"]]
        def n_correct(it):
            return sum(1 for m in real_names if model_preds[m][it["file"]] == it["label"])
        corr["pos_caught_by_k"] = dict(Counter(n_correct(it) for it in pos_common))
        corr["all_models_missed_pos"] = sum(1 for it in pos_common if n_correct(it) == 0)
        corr["neg_falseflag_by_k"] = dict(Counter(sum(1 for m in real_names if model_preds[m][it["file"]]) for it in neg_common))
        corr["all_models_falseflag_neg"] = sum(1 for it in neg_common
                                                if all(model_preds[m][it["file"]] for m in real_names))

    # ---- k-of-N agreement sweep + Fleiss' kappa (free ensemble analysis) ----
    N = len(real_names)
    sweep, fleiss = [], None
    if N > 1:
        vote_counts = [sum(model_preds[m][it["file"]] for m in real_names) for it in items_common]
        fleiss = fleiss_kappa_binary(vote_counts, N)
        for k in range(1, N + 1):
            pk = {it["file"]: (sum(model_preds[m][it["file"]] for m in real_names) >= k) for it in items_common}
            sk = score(pk, items_common)
            sweep.append({"k": k, "recall": sk["recall"], "precision": sk["precision"],
                          "f1": sk["f1"], "fpr": sk["fpr"], "mcc": sk["mcc"],
                          "precision_at_corpus": precision_at(sk["recall"], sk["fpr"], pi)})

    out = {"label": args.label, "prevalence_corpus": pi,
           "n_pos_full": len(pos), "n_neg_full": len(neg), "n_common": len(items_common),
           "n_pos_common": pos_c, "n_neg_common": neg_c, "dropped_for_coverage": dropped,
           "best_individual": best_individual, "stats": model_stats, "models": results,
           "consortium_vs_best": comparison, "kappa": kappa, "fleiss_kappa": fleiss,
           "k_of_n_sweep": sweep, "correlated_errors": corr}
    args.out.write_text(json.dumps(out, indent=2))

    cwes = sorted({c for m in results.values() for c in m["per_cwe"]})

    # ---------------- console ----------------
    print("\n" + "=" * 100)
    print("  DETECTOR CONSORTIUM  vs  VULCAN execution ground truth")
    print(f"  label = {args.label}  |  common set: {pos_c} pos / {neg_c} neg  ({dropped} dropped for coverage)"
          f"  |  corpus prevalence = {pi:.1%}")
    print("=" * 100)
    pcol = f"prec@{pi*100:.0f}%"
    print(f"  {'detector':22} {'recall (95% CI)':>20} {'F1 (95% CI)':>20} {'FPR':>6} {'MCC':>6}"
          f" {'prec@bench':>11} {pcol:>10}")
    print("  " + "-" * 102)
    for name, m in order:
        print(f"  {name[:22]:22} {ci_str(m['TP'], m['TP']+m['FN']):>20} {trip_str(m['f1_ci']):>20}"
              f" {fmt(m['fpr']):>6} {fmt(m['mcc']):>6} {fmt(m['precision_bench']):>11} {fmt(m['precision_at_corpus']):>10}")
    print("  (* = individual model; precision depends on prevalence -- prec@bench is the 2:1 set,")
    print(f"   prec@{pi:.0%} projects to the real corpus prevalence; recall & FPR are prevalence-free)")

    if comparison:
        print(f"\n  CONSORTIUM vs best individual ({best_individual}):")
        print(f"  {'rule':14} {'F1 cons':>8} {'F1 best':>8} {'ΔF1 (95% CI)':>24} {'McNemar p':>11}")
        print("  " + "-" * 70)
        for rule, c in comparison.items():
            d = c["f1_diff_ci"]
            dstr = trip_str(d) if d and d[0] is not None else "   -   "
            print(f"  {rule:14} {fmt(c['f1_consortium']):>8} {fmt(c['f1_best_individual']):>8} {dstr:>24} {c['mcnemar']['p']:>11.3f}")
        print("  (ΔF1>0 with CI excluding 0 AND McNemar p<0.05 => consortium significantly better)")

    if sweep:
        print("\n  k-of-N agreement sweep (≥k of %d models say vulnerable):" % N)
        print(f"  {'rule':14} {'recall':>8} {'precision':>10} {'F1':>8} {'FPR':>8} {'MCC':>8}")
        print("  " + "-" * 62)
        kmaj = (N // 2) + 1
        for s in sweep:
            lbl = f"≥{s['k']}/{N}"
            if s['k'] == 1: lbl += " (any)"
            elif s['k'] == N: lbl += " (unanim)"
            elif s['k'] == kmaj: lbl += " (major)"
            print(f"  {lbl:14} {fmt(s['recall']):>8} {fmt(s['precision']):>10} {fmt(s['f1']):>8} "
                  f"{fmt(s['fpr']):>8} {fmt(s['mcc']):>8}")
    if kappa:
        print("\n  Inter-model agreement (Cohen's κ pairwise; 1=identical, 0=chance):")
        for k, v in kappa.items():
            print(f"    {k:40} κ = {v:.3f}" if v is not None else f"    {k:40} κ = -")
        if fleiss is not None:
            print(f"    {'PANEL (Fleiss κ, all ' + str(N) + ')':40} κ = {fleiss:.3f}")
    if corr:
        print("\n  Correlated errors (on common set):")
        print(f"    positives caught by k models: {corr['pos_caught_by_k']}  "
              f"-> all models MISSED {corr['all_models_missed_pos']} positive(s)")
        print(f"    negatives false-flagged by k models: {corr['neg_falseflag_by_k']}  "
              f"-> all models FALSE-FLAGGED {corr['all_models_falseflag_neg']} negative(s)")

    print(f"\n  Per-CWE recall (caught / confirmed), 95% CI   (* = <{args.min_cwe_n} positives):")
    print("  " + f"{'detector':22}" + "".join(f"{c:>18}" for c in cwes))
    for name, m in order:
        cells = ""
        for c in cwes:
            d = m["per_cwe"].get(c, {"num": 0, "den": 0})
            cells += f"{'  -':>18}" if not d["den"] else \
                f"{(ci_str(d['num'], d['den']) + ('*' if d['den'] < args.min_cwe_n else '')):>18}"
        print(f"  {name[:22]:22}{cells}")

    print("\n  Coverage / reliability:")
    for name in real_names:
        s = model_stats[name]; lat = "-" if s["median_latency"] is None else f"{s['median_latency']}s"
        print(f"    {name[:22]:22} answered {s['answered']}/{s['total']}  valid {s['valid']}  errors {s['errors']}  med.lat {lat}")

    # ---------------- markdown ----------------
    md = [f"# Detector consortium — results (label = {args.label})", "",
          f"- Ground truth: VULCAN execution-adjudicated. Scored on the **common set every model "
          f"answered**: {pos_c} positive / {neg_c} negative ({dropped} dropped for incomplete coverage).",
          f"- **Prevalence caveat:** this set is balanced ~{npos}:{nneg}. **Recall and FPR are "
          f"prevalence-free; precision and F1 are not.** `prec@bench` is at this balanced set; "
          f"`prec@{pi:.1%}` projects to the real corpus prevalence ({n_conf}/{n_exec}).",
          f"- **Negative semantics:** `NOT_TRIGGERED` = not triggered **under the tested execution "
          f"conditions**, not a universal-safety proof.",
          f"- Diversity: three providers to **reduce, not eliminate**, correlated errors; consortium "
          f"gains are tested empirically (McNemar + bootstrap), not assumed.",
          f"- 95% CIs: Wilson (recall/precision), bootstrap (F1).", "",
          "## Leaderboard", "",
          f"| Detector | Recall (95% CI) | F1 (95% CI) | FPR | MCC | Precision @bench | Precision @{pi*100:.1f}% |",
          "|---|---|---|---|---|---|---|"]
    for name, m in order:
        md.append(f"| {name} | {ci_str(m['TP'], m['TP']+m['FN'])} | {trip_str(m['f1_ci'])} | "
                  f"{fmt(m['fpr'])} | {fmt(m['mcc'])} | {fmt(m['precision_bench'])} | {fmt(m['precision_at_corpus'])} |")
    if sweep:
        kmaj = (N // 2) + 1
        md += ["", f"## k-of-N agreement sweep (≥k of {N} models flag vulnerable)", "",
               "| Rule | Recall | Precision | F1 | FPR | MCC |", "|---|---|---|---|---|---|"]
        for s in sweep:
            lbl = f"≥{s['k']}/{N}" + (" (any)" if s['k'] == 1 else " (unanimous)" if s['k'] == N
                                      else " (majority)" if s['k'] == kmaj else "")
            md.append(f"| {lbl} | {fmt(s['recall'])} | {fmt(s['precision'])} | {fmt(s['f1'])} | "
                      f"{fmt(s['fpr'])} | {fmt(s['mcc'])} |")
        md.append("")
        md.append("> The sweep traces the ensemble precision–recall trade-off: ≥1 maximises recall, "
                  "≥N maximises precision; the majority rule is the headline consortium.")
    if comparison:
        md += ["", f"## Consortium vs best individual ({best_individual})", "",
               "| Rule | F1 consortium | F1 best | ΔF1 (95% CI) | McNemar p |", "|---|---|---|---|---|"]
        for rule, c in comparison.items():
            d = c["f1_diff_ci"]; dstr = trip_str(d) if d and d[0] is not None else "-"
            md.append(f"| {rule} | {fmt(c['f1_consortium'])} | {fmt(c['f1_best_individual'])} | {dstr} | {c['mcnemar']['p']:.3f} |")
        md.append("")
        md.append("> Consortium is significantly better only if ΔF1>0 with a CI excluding 0 **and** McNemar p<0.05.")
    if kappa:
        md += ["", "## Inter-model agreement", "", "| Pair | κ |", "|---|---|"]
        for k, v in kappa.items():
            md.append(f"| {k} | {v:.3f} |" if v is not None else f"| {k} | - |")
        if fleiss is not None:
            md.append(f"| **PANEL (Fleiss κ, all {N})** | **{fleiss:.3f}** |")
    if corr:
        md += ["", "## Correlated errors (common set)", "",
               f"- Positives caught by k models: `{corr['pos_caught_by_k']}` — all models missed "
               f"**{corr['all_models_missed_pos']}** positive(s).",
               f"- Negatives false-flagged by k models: `{corr['neg_falseflag_by_k']}` — all models "
               f"false-flagged **{corr['all_models_falseflag_neg']}** negative(s)."]
    md += ["", f"## Per-CWE recall (caught / confirmed, 95% CI; * = <{args.min_cwe_n} positives)", "",
           "| Detector | " + " | ".join(cwes) + " |", "|" + "---|" * (len(cwes) + 1)]
    for name, m in order:
        row = [name]
        for c in cwes:
            d = m["per_cwe"].get(c, {"num": 0, "den": 0})
            row.append("-" if not d["den"] else ci_str(d["num"], d["den"]) + ("*" if d["den"] < args.min_cwe_n else ""))
        md.append("| " + " | ".join(row) + " |")
    md += ["", "## Coverage / reliability", "",
           "| Model | answered | valid | errors | median latency | tokens |", "|---|---|---|---|---|---|"]
    for name in real_names:
        s = model_stats[name]
        md.append(f"| {name} | {s['answered']}/{s['total']} | {s['valid']} | {s['errors']} | "
                  f"{s['median_latency'] or '-'} | {s['tokens'] or '-'} |")
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"\n  Saved {args.out}  and thesis-ready report -> {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
