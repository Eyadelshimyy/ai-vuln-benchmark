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
import argparse, hashlib, json, math, os, random, re, shutil, sys, tempfile, time
from datetime import datetime, timezone
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

# Threat-model-explicit variant. VULCAN's positive label is CONDITIONAL
# exploitability: the oracle drives each PARAMETER with attacker input. Under
# PROMPT (v1) a model that reasons "open(path) is fine if the caller is trusted"
# is marked a miss although it answered a different question; v2 states the
# oracle's threat model so detector and ground truth measure the same construct.
# Still open-ended over the six classes (no CWE hint). See
# validation/rq2/RQ2_PREREGISTRATION.md.
PROMPT_V2_VERSION = "v2-untrusted-params-6cwe"
PROMPT_V2 = """You are a security code reviewer. Threat model: every parameter of \
the function below is fully controlled by an attacker, and no caller validates \
it. Decide whether, under that threat model, calling this function can produce a \
security vulnerability in one of these classes: CWE-22 path traversal / arbitrary \
file access, CWE-78 OS command injection, CWE-89 SQL injection, CWE-611 XXE, \
CWE-918 SSRF, CWE-502 unsafe deserialization.

Judge only the function as written. Answer ONLY with a JSON object, no prose:
{"vulnerable": true|false, "cwe": "CWE-78"|null, "confidence": 0.0-1.0, "reason": "<=15 words"}

FUNCTION (%s):
```python
%s
```"""

# Run protocols. "v1" reproduces the P5 run exactly (legacy cache keys, legacy
# CLI invocation from the current directory). "v2" is the protocol for the frozen
# RQ2 run: the CLI runs in a fresh EMPTY directory (an agentic CLI started in the
# repo can read files there, including ground-truth labels), the full raw
# output is kept in the cache for audit, and cache keys are namespaced so v1 and
# v2 answers can never mix.
# Prompt text and version are named, not copied, and looked up at call time
# (_prompt / _prompt_version), so bumping PROMPT_VERSION still invalidates the cache.
PROTOCOLS = {
    "v1": {"prompt": "PROMPT", "prompt_version": "PROMPT_VERSION", "isolate_cwd": False},
    "v2": {"prompt": "PROMPT", "prompt_version": "PROMPT_VERSION", "isolate_cwd": True},
    "v2-threat": {"prompt": "PROMPT_V2", "prompt_version": "PROMPT_V2_VERSION", "isolate_cwd": True},
}


def _prompt(protocol: str) -> str:
    return globals()[PROTOCOLS[protocol]["prompt"]]


def _prompt_version(protocol: str) -> str:
    return globals()[PROTOCOLS[protocol]["prompt_version"]]


# --------------------------------------------------------------------------- #
# model adapters
# --------------------------------------------------------------------------- #
def call_settings(cfg: dict, protocol: str) -> dict:
    """The generation settings sent for this model under this protocol. v1 keeps
    the legacy fixed settings. v2 protocols take them from the model entry,
    because reasoning models (e.g. on OpenRouter) reject `temperature` and spend
    output tokens on reasoning, so a 200-token cap returns an empty answer.
      temperature      number, or null = do not send (model does not support it)
      max_tokens       output cap INCLUDING reasoning tokens (default 4000)
      reasoning_effort "low"|"medium"|"high", or null = provider default
      seed             int, or null
      provider         OpenRouter routing object, e.g. {"order": ["deepinfra"],
                       "allow_fallbacks": false}, or null = OpenRouter chooses
    The settings are part of the v2 cache key, so changing one never reuses
    answers produced under another."""
    if protocol == "v1":
        return {"temperature": 0, "max_tokens": 200}
    return {"temperature": cfg.get("temperature", 0), "max_tokens": cfg.get("max_tokens", 4000),
            "reasoning_effort": cfg.get("reasoning_effort"), "seed": cfg.get("seed"),
            "provider": cfg.get("provider")}


def _call_openai(cfg: dict, source: str, fn: str, timeout: float, protocol: str = "v1") -> dict:
    from openai import OpenAI
    key = cfg.get("api_key") or os.environ.get(cfg.get("api_key_env", ""), "")
    if protocol != "v1" and not key:
        # never send an unauthenticated request; a missing key is a setup error, not an answer
        return {"error": f"no API key: set ${cfg.get('api_key_env')}", "vulnerable": None}
    client = OpenAI(api_key=key or "none", base_url=cfg.get("base_url"))
    st = call_settings(cfg, protocol)
    kw = {"model": cfg["model"], "timeout": timeout, "max_tokens": st["max_tokens"],
          "messages": [{"role": "user", "content": _prompt(protocol) % (fn, source[:6000])}]}
    if st.get("temperature") is not None:
        kw["temperature"] = st["temperature"]
    if st.get("seed") is not None:
        kw["seed"] = st["seed"]
    extra = {}
    if protocol != "v1" and "openrouter.ai" in (cfg.get("base_url") or ""):
        extra["usage"] = {"include": True}            # per-call cost in the response
        if st.get("reasoning_effort"):
            extra["reasoning"] = {"effort": st["reasoning_effort"]}
        if st.get("provider"):
            extra["provider"] = st["provider"]
    if extra:
        kw["extra_body"] = extra
    try:
        r = client.chat.completions.create(**kw)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:200]}", "vulnerable": None}
    choice = r.choices[0] if r.choices else None
    content = (choice.message.content if choice and choice.message else None) or ""
    ans = _parse(content)
    try:
        ans["tokens"] = r.usage.total_tokens  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    if protocol != "v1":
        # audit record: what was actually served, at what cost, and why it stopped
        dump = r.model_dump() if hasattr(r, "model_dump") else {}
        usage = dump.get("usage") or {}
        ans.update({
            "raw_content": content[-8000:],
            "served_model": dump.get("model"), "served_provider": dump.get("provider"),
            "generation_id": dump.get("id"),
            "finish_reason": choice.finish_reason if choice else None,
            "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")},
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "settings": st,
            "called_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        if not content and choice and choice.finish_reason == "length":
            # the cap was spent on reasoning: a setup problem, so retry/raise the cap rather than score it invalid
            return {**ans, "error": "empty answer: max_tokens exhausted (raise max_tokens)", "vulnerable": None}
    return ans


def _call_opencode_cli(cfg, source, fn, timeout, protocol: str = "v1"):
    prompt = _prompt(protocol) % (fn, source[:6000])
    cmd = ["opencode", "run", "-m", cfg["model"]]
    if cfg.get("agent"):
        cmd += ["--agent", cfg["agent"]]
    cmd.append(prompt)
    isolate = PROTOCOLS[protocol]["isolate_cwd"]
    # v2: run in a fresh empty dir so an agentic CLI has nothing to read (the
    # repo holds the ground-truth labels). v1 keeps the legacy behaviour.
    workdir = tempfile.mkdtemp(prefix="vulcan_rq2_") if isolate else None
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=workdir)
    except subprocess.TimeoutExpired:
        return {"error": "cli timeout", "vulnerable": None}
    finally:
        if workdir:
            leftovers = os.listdir(workdir)
            shutil.rmtree(workdir, ignore_errors=True)
    out = pr.stdout or ""
    meta = {}
    if isolate:
        # full raw record for audit: every answer can be re-parsed or inspected
        # for tool use later without re-spending a call
        meta = {"raw_stdout": out[-20000:], "raw_stderr": (pr.stderr or "")[-4000:],
                "returncode": pr.returncode, "cwd_isolated": True,
                "workdir_leftovers": leftovers[:20],
                "called_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    blocks = re.findall(r"\{[^{}]*\}", out)
    if not blocks:
        return {"error": (pr.stderr or out).strip()[:200] or "refusal/no-json", "vulnerable": None, **meta}
    return {**_parse(blocks[-1]), **meta}


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


def _cache_key(cache: Path, name: str, model_id: str, it: dict, protocol: str = "v1",
               cfg: dict | None = None) -> Path:
    src_h = hashlib.md5(it["source"].encode("utf-8", "replace")).hexdigest()
    parts = [name, str(model_id), _prompt_version(protocol), it["file"], src_h]
    if protocol != "v1":          # v1 keys stay byte-identical so the P5 cache still resolves
        parts.append(f"protocol={protocol}")
        if cfg is not None:
            parts.append("settings=" + json.dumps(call_settings(cfg, protocol), sort_keys=True))
    sig = hashlib.md5("\x00".join(parts).encode()).hexdigest()
    return cache / f"{name}__{sig}.json"


def _cached_answer(ck: Path) -> dict | None:
    """A cached answer, or None if the call still has to be made. Entries that
    hold only a transient error (written by an earlier version of this script)
    count as missing, so they are retried instead of silently dropping the item."""
    if not ck.exists():
        return None
    ans = json.loads(ck.read_text())
    if ans.get("error") and not isinstance(ans.get("vulnerable"), bool):
        return None
    return ans


def plan_calls(models_cfg: list, items: list, cache: Path, protocol: str) -> dict:
    """Uncached (model, item) calls a run would make. Pure: no model is called."""
    return {cfg["name"]: sum(1 for it in items
                             if _cached_answer(_cache_key(cache, cfg["name"], cfg.get("model", cfg["name"]),
                                                          it, protocol, cfg)) is None)
            for cfg in models_cfg}


def estimate_cost(cfgs: list, items: list, plan: dict, pricing: dict, out_tokens, protocol: str) -> list:
    """Cost range for the planned calls: input tokens from the actual prompt
    length (~3.5 chars/token), output tokens assumed (LOW..HIGH, incl.
    reasoning). An ESTIMATE for budgeting; the run records the real cost."""
    prices = {m["id"]: m.get("pricing") or {} for m in (pricing.get("data") or pricing.get("models") or [])}
    avg_in = sum(len(_prompt(protocol)) + len(it["source"]) for it in items) / max(1, len(items)) / 3.5
    lines, lo_t, hi_t = [], 0.0, 0.0
    for cfg in cfgs:
        pr = prices.get(cfg.get("model"))
        n = plan.get(cfg["name"], 0)
        if not pr or not n:
            lines.append(f"cost est. {cfg['name']}: " + ("no pricing for this model id" if n else "0 calls"))
            continue
        pin, pout = float(pr.get("prompt") or 0), float(pr.get("completion") or 0)
        lo = n * (avg_in * pin + out_tokens[0] * pout); hi = n * (avg_in * pin + out_tokens[1] * pout)
        lo_t += lo; hi_t += hi
        lines.append(f"cost est. {cfg['name']}: {n} calls x ~{avg_in:.0f} in-tok -> ${lo:.2f}-${hi:.2f}")
    lines.append(f"cost est. TOTAL ${lo_t:.2f}-${hi_t:.2f} (assumes {out_tokens[0]}-{out_tokens[1]} output tokens/call)")
    return lines


def load_labelset(path: Path) -> tuple[list, dict]:
    """Items from a frozen label set (analysis/rq2_labelset.py), with the
    source read back and checked against the frozen hash."""
    ls = json.loads(path.read_text())
    default_root = Path(ls["meta"].get("root", "calibration/mined_prompts"))
    items = []
    for it in ls["items"]:
        cj = json.loads((Path(it["root"]) if it.get("root") else default_root).joinpath(it["file"]).read_text())
        src = cj.get("ground_truth_source") or cj.get("module_source") or ""
        if hashlib.sha256(src.encode()).hexdigest() != it["source_sha256"]:
            raise SystemExit(f"LABEL-SET ERROR: source of {it['file']} changed since the label set was frozen")
        items.append({**it, "source": src, "function": cj.get("function", "?")})
    # length terciles over the whole label set (pre-registered confound check:
    # negatives are longer than positives on average)
    lens = sorted(it["src_chars"] for it in items)
    if lens:
        t1, t2 = lens[len(lens) // 3], lens[2 * len(lens) // 3]
        for it in items:
            it["len_tercile"] = "short" if it["src_chars"] < t1 else ("medium" if it["src_chars"] < t2 else "long")
    return items, ls["meta"]


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


STRATA = ("generator", "exec_cwe", "cwe22_subtype", "len_tercile")


def pairwise_metrics(main_preds: dict, pair_preds: dict, pair_items: list) -> dict:
    """JitVul-style pairwise accuracy on execution-verified vulnerable/fixed twins.
    A pair counts only if the model answered both versions. Outcomes:
      correct    original flagged AND fixed cleared
      both_vuln  both flagged (cannot tell the fix apart)
      both_safe  neither flagged (misses the original)
      reversed   original cleared AND fixed flagged"""
    out = {}
    for m, pp in pair_preds.items():
        c = Counter()
        for it in pair_items:
            o, f = main_preds.get(m, {}).get(it["pair_of"]), pp.get(it["file"])
            if o is None or f is None:
                c["unanswered"] += 1
                continue
            c["correct" if (o and not f) else "both_vuln" if (o and f) else "both_safe" if (not o and not f) else "reversed"] += 1
        n = sum(v for k, v in c.items() if k != "unanswered")
        out[m] = {**{k: c.get(k, 0) for k in ("correct", "both_vuln", "both_safe", "reversed", "unanswered")},
                  "n_pairs": n, "pair_accuracy": c["correct"] / n if n else None,
                  "pair_accuracy_ci": wilson(c["correct"], n) if n else None}
    return out


def per_stratum_metrics(model_preds: dict, items: list) -> dict:
    """Recall (over positives) and FPR (over negatives) per stratum value, with
    Wilson CIs. exec_cwe / cwe22_subtype exist only on positives, so they get
    recall only. Small cells are reported with their n, never as bare rates."""
    out = {}
    for key in STRATA:
        vals = sorted({it.get(key) for it in items if it.get(key) is not None})
        if not vals:
            continue
        out[key] = {}
        for v in vals:
            cell_pos = [it for it in items if it["label"] and it.get(key) == v]
            cell_neg = [it for it in items if not it["label"] and it.get(key) == v]
            out[key][v] = {}
            for m, p in model_preds.items():
                tp = sum(1 for it in cell_pos if p.get(it["file"]) is True)
                fp = sum(1 for it in cell_neg if p.get(it["file"]) is True)
                out[key][v][m] = {"n_pos": len(cell_pos), "tp": tp,
                                  "recall": tp / len(cell_pos) if cell_pos else None,
                                  "recall_ci": wilson(tp, len(cell_pos)) if cell_pos else None,
                                  "n_neg": len(cell_neg), "fp": fp,
                                  "fpr": fp / len(cell_neg) if cell_neg else None,
                                  "fpr_ci": wilson(fp, len(cell_neg)) if cell_neg else None}
    return out


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
    ap.add_argument("--census", type=Path, default=None,
                    help="census JSON (required unless --labelset, which names its own census)")
    ap.add_argument("--labelset", type=Path, default=None,
                    help="frozen label set from analysis/rq2_labelset.py (the RQ2 input; replaces "
                         "--census/--label/--balance sampling)")
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="v1",
                    help="v1 = reproduce P5 from its cache; v2 = isolated-cwd CLI + raw capture; "
                         "v2-threat = v2 with the threat-model-explicit prompt")
    ap.add_argument("--live", action="store_true",
                    help="ALLOW real (paid) model calls. Without it, any uncached call aborts the run "
                         "after printing the call plan.")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="hard cap on NEW model calls in this run (with --live); the run stops at the cap")
    ap.add_argument("--smoke", type=int, default=0,
                    help="setup check: N items per model (1 pos + rest neg), prints the raw answers; needs --live")
    ap.add_argument("--retries", type=int, default=2, help="retries for transient call errors")
    ap.add_argument("--stop-after-errors", type=int, default=5,
                    help="stop calling a model after N consecutive failed items (e.g. a free-model daily "
                         "429 limit); the cache makes resuming later free")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="hard cap in USD on this run, from the per-call cost OpenRouter reports")
    ap.add_argument("--pricing", type=Path, default=None,
                    help="pricing snapshot (validation/rq2/openrouter_models_snapshot.json) for the cost estimate")
    ap.add_argument("--est-out-tokens", type=int, nargs=2, default=[300, 2000], metavar=("LOW", "HIGH"),
                    help="assumed output tokens per call (incl. reasoning) for the estimate range")
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

    ls_meta = None
    if args.labelset:
        items, ls_meta = load_labelset(args.labelset)
        args.label = "confirmed"
        corpora = ls_meta.get("corpora") or [{"census": str(args.census or ls_meta["census"]),
                                               "census_sha256": ls_meta["census_sha256"]}]
        rows = []
        for c in corpora:
            cp = Path(c["census"])
            if hashlib.sha256(cp.read_bytes()).hexdigest() != c["census_sha256"]:
                raise SystemExit(f"LABEL-SET ERROR: {cp} is not the census the label set was frozen from")
            cj = json.loads(cp.read_text())
            rows += cj["rows"] if isinstance(cj, dict) else cj
    elif args.census is None:
        raise SystemExit("--census or --labelset is required")
    if ls_meta is None:
        cen = json.loads(args.census.read_text())
        rows = cen["rows"] if isinstance(cen, dict) else cen

    # deployment prevalence (confirmed among executable) for projected precision
    n_exec = sum(1 for r in rows if r.get("verdict") != "COULD_NOT_EXECUTE")
    n_conf = sum(1 for r in rows if r.get("confirmed"))
    if args.prevalence is not None:
        pi = args.prevalence
    elif ls_meta is not None:
        pi = ls_meta["prevalence_corpus"]          # same (source-unique) unit as the label set
        n_conf, n_exec = ls_meta["population"]["pos_source_unique"], (
            ls_meta["population"]["pos_source_unique"] + ls_meta["population"]["neg_source_unique_pool"])
    else:
        pi = n_conf / n_exec if n_exec else 0.0

    if ls_meta is None:
        items = build_eval_set(rows, args.root, args.label)
    # verified FIXED twins are scored only pairwise (validation/twins), never as
    # ordinary negatives: they still TRIGGER, which the main labels treat as inconclusive
    pair_items = [it for it in items if it.get("role") == "pair_fixed"]
    items = [it for it in items if it.get("role") != "pair_fixed"]
    if args.cwe:
        items = [it for it in items if args.cwe in it["cwes"] or not it["label"]]
    pos = [it for it in items if it["label"]]
    neg = [it for it in items if not it["label"]]
    if args.balance and neg and ls_meta is None:
        neg = random.Random(0).sample(neg, min(len(neg), int(len(pos) * args.balance)))
    items = pos + neg
    if args.smoke:
        if not args.live:
            raise SystemExit("--smoke makes real model calls; add --live to authorize them")
        sp = random.Random(5).sample(pos, 1) if pos else []
        items = sp + random.Random(6).sample(neg, max(0, min(len(neg), args.smoke - len(sp))))

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

    # ---- call plan + spend guard (nothing below calls a model without --live) ----
    real_cfgs = [c for c in models_cfg if not (args.mock or c.get("mock"))]
    if args.smoke or args.pilot or args.limit:
        pair_items = []
    call_items = items + pair_items
    plan = plan_calls(real_cfgs, call_items, args.cache, args.protocol)
    n_new = sum(plan.values())
    print(f"  call plan [protocol={args.protocol}]: {n_new} uncached real calls "
          + ", ".join(f"{k}={v}" for k, v in plan.items()), file=sys.stderr)
    if n_new and args.pricing:
        for line in estimate_cost(real_cfgs, items, plan, json.loads(args.pricing.read_text()),
                                  args.est_out_tokens, args.protocol):
            print("  " + line, file=sys.stderr)
    if n_new and not args.live:
        print("  NOT CALLING: re-run with --live (and ideally --max-calls) to authorize these calls.",
              file=sys.stderr)
        return
    if args.live and args.max_calls is not None and n_new > args.max_calls:
        print(f"  NOTE: plan ({n_new}) exceeds --max-calls {args.max_calls}; the run will stop at the cap "
              "and report partial coverage.", file=sys.stderr)
    calls_made = 0
    spent = 0.0

    model_preds, model_stats, real_names = {}, {}, []
    for cfg in models_cfg:
        name = cfg["name"]; real_names.append(name)
        model_id = cfg.get("model", name)
        preds, raw = {}, {}
        consecutive_err, halted = 0, False
        for i, it in enumerate(call_items, 1):
            ck = _cache_key(args.cache, name, model_id, it, args.protocol, cfg)
            ans = _cached_answer(ck)
            if ans is None:
                is_mock = args.mock or cfg.get("mock")
                if not is_mock and args.max_calls is not None and calls_made >= args.max_calls:
                    raw[it["file"]] = {"error": "not called: --max-calls reached", "vulnerable": None}
                    continue
                if not is_mock and args.max_cost is not None and spent >= args.max_cost:
                    raw[it["file"]] = {"error": "not called: --max-cost reached", "vulnerable": None}
                    continue
                if halted:
                    raw[it["file"]] = {"error": "not called: model halted after repeated errors", "vulnerable": None}
                    continue
                t0 = time.time()
                for attempt in range(1 + (0 if is_mock else args.retries)):
                    if is_mock:
                        ans = _mock(cfg, it["source"], it["function"], args.timeout)
                    elif cfg.get("cli"):
                        ans = _call_opencode_cli(cfg, it["source"], it["function"], args.timeout, args.protocol)
                    else:
                        ans = _call_openai(cfg, it["source"], it["function"], args.timeout, args.protocol)
                    if not is_mock:
                        calls_made += 1
                        spent += float((ans.get("usage") or {}).get("cost") or 0)
                    if "error" not in ans:
                        break
                    time.sleep(min(30, 2 ** (attempt + 1)))
                ans.setdefault("latency", round(time.time() - t0, 3))
                ans.update({"model_id": model_id, "protocol": args.protocol,
                            "prompt_version": _prompt_version(args.protocol)})
                if "error" not in ans:
                    ck.write_text(json.dumps(ans))   # cache only real answers; transient errors retry on re-run
                    consecutive_err = 0
                elif not is_mock:
                    consecutive_err += 1
                    if consecutive_err >= args.stop_after_errors:
                        halted = True
                        print(f"  [{name}] halted after {consecutive_err} consecutive errors; last: "
                              f"{str(ans.get('error'))[:160]}  (re-run later to resume from cache)", file=sys.stderr)
                if args.smoke:
                    print(f"\n----- SMOKE [{name}] {it['file']} (label={it['label']}) -----\n"
                          f"parsed: { {k: ans.get(k) for k in ('vulnerable', 'cwe', 'confidence', 'reason', 'invalid', 'error')} }\n"
                          f"returncode={ans.get('returncode')} workdir_leftovers={ans.get('workdir_leftovers')}\n"
                          f"raw stdout (tail):\n{(ans.get('raw_stdout') or '')[-3000:]}", file=sys.stderr)
                if not is_mock:
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

    if args.smoke:
        print(f"\n  SMOKE done: {calls_made} real calls. Inspect the raw outputs above (JSON answer present? "
              "any sign of tool use / file reads?) before authorizing the full run.", file=sys.stderr)
        return

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
    pair_preds = {n: {it["file"]: model_preds[n].get(it["file"]) for it in pair_items} for n in real_names}
    main_files = {it["file"] for it in items}
    for n in real_names:
        model_preds[n] = {f: v for f, v in model_preds[n].items() if f in main_files}
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
           "k_of_n_sweep": sweep, "correlated_errors": corr,
           "protocol": args.protocol, "prompt_version": _prompt_version(args.protocol),
           "labelset": ({"path": str(args.labelset), **{k: ls_meta.get(k) for k in
                         ("labelset_version", "census_sha256", "corpora", "items_sha256", "seed", "neg_per_pos")}}
                        if ls_meta else None),
           "per_stratum": per_stratum_metrics(model_preds, items_common),
           "pairwise": pairwise_metrics(model_preds, pair_preds, pair_items),
           # per-item predictions on the common set: lets every table above be
           # recomputed (or re-stratified) without another model call
           "predictions": {m: {it["file"]: p[it["file"]] for it in items_common if it["file"] in p}
                           for m, p in model_preds.items()}}
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
    ps = out["per_stratum"]

    def _cell(c, which):
        k, n, ci = (c["tp"], c["n_pos"], c["recall_ci"]) if which == "recall" else (c["fp"], c["n_neg"], c["fpr_ci"])
        return "-" if not n else f"{k}/{n} [{ci[0]:.2f}-{ci[1]:.2f}]"

    if "generator" in ps:
        md += ["", "## By generating agent (who wrote the code)", "",
               "Recall = caught/positives; FPR = flagged/negatives (Wilson 95% CI). Generator is confounded "
               "with repository; small cells are descriptive only.", "",
               "| Detector | " + " | ".join(f"{g} recall | {g} FPR" for g in ps["generator"]) + " |",
               "|---|" + "---|---|" * len(ps["generator"])]
        for name in real_names:
            md.append(f"| {name} | " + " | ".join(f"{_cell(c[name], 'recall')} | {_cell(c[name], 'fpr')}"
                                                  for c in ps["generator"].values()) + " |")
    if "cwe22_subtype" in ps:
        md += ["", "## CWE-22 recall by subtype", "",
               "DIRECT_CALLER_PATH = the function opens the path its caller passes (vulnerable only if the "
               "caller is untrusted); BASE_JOIN_OVERRIDE = intended base directory escaped. Under prompt "
               "v1 a 'no' on DIRECT rows can be a defensible trust-boundary judgement rather than a miss.", "",
               "| Detector | " + " | ".join(ps["cwe22_subtype"]) + " |",
               "|---|" + "---|" * len(ps["cwe22_subtype"])]
        for name in real_names:
            md.append(f"| {name} | " + " | ".join(_cell(c[name], "recall") for c in ps["cwe22_subtype"].values()) + " |")
    pw = out.get("pairwise") or {}
    if any(v["n_pairs"] for v in pw.values()):
        md += ["", "## Pairwise: execution-verified vulnerable vs fixed twins", "",
               "Correct = original flagged AND fixed version cleared. Both-vuln = cannot tell the fix "
               "apart. Twins differ only by the fix, and the oracle verified that each fix removes the exploit.", "",
               "| Detector | pairs | pair accuracy [95% CI] | correct | both vuln | both safe | reversed |",
               "|---|---|---|---|---|---|---|"]
        for name in real_names:
            v = pw.get(name) or {}
            if not v.get("n_pairs"):
                continue
            lo, hi = v["pair_accuracy_ci"]
            md.append(f"| {name} | {v['n_pairs']} | {v['pair_accuracy']:.2f} [{lo:.2f}-{hi:.2f}] | {v['correct']} | "
                      f"{v['both_vuln']} | {v['both_safe']} | {v['reversed']} |")
    md += ["", f"Protocol `{args.protocol}` / prompt `{_prompt_version(args.protocol)}`"
           + (f" / label set `{ls_meta['labelset_version']}` items-sha256 `{ls_meta['items_sha256'][:16]}`"
              if ls_meta else "")]
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"\n  Saved {args.out}  and thesis-ready report -> {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
