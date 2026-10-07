#!/usr/bin/env python3
"""
VULCAN Tier-2: execution-PROVEN exploitation (curated).

Pipeline:
  1. Find every (candidate, sample) that Tier 1 flags TRIGGERED -- a tainted
     value reached a dangerous sink -- recording its CWE and injected param.
  2. For a curated subset (first triggering sample per candidate+CWE, capped
     per CWE), dispatch to the matching Tier-2 strategy and attempt to PROVE
     exploitation: run real attacker payloads through the real code in live
     mode and check for a real observable effect.
  3. Report CONFIRMED_EXPLOITABLE only when the observable fired, with the
     exact payload and proof artifact -- fully auditable.

The detection oracle is untouched and never agentic. --use-agent only lets an
LLM *craft* payloads; the deterministic observable check still decides.

USAGE (local, free):
  export OPENSOURCE_BASE_URL=http://localhost:11434/v1
  python3 run_tier2.py --candidates calibration/mined_prompts/invoke/ --model qwen2.5-coder:7b
  # add the agentic crafter on top of the deterministic bank:
  python3 run_tier2.py --candidates calibration/mined_prompts/sqlite-utils/ --model qwen2.5-coder:7b --use-agent --max-agent 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "analysis" / "iast"))
sys.path.insert(0, str(_REPO))

from auto_harness import run_candidate            # noqa: E402
from run_benchmark import reduce_verdicts          # noqa: E402
from tier2.engine import attempt_exploit           # noqa: E402
from tier2.strategies import STRATEGIES            # noqa: E402


import re as _re


def injection_context(t1_sql: str) -> str:
    """Classify WHERE in the SQL grammar the attacker-controlled value lands,
    from the Tier-1 sink SQL (which embeds the benign IAST taint token -- no SQL
    metacharacters, so a quote-state lex around it is reliable). This is the
    sound 'genuine negative vs fixable' signal, computed from data we already
    capture -- unlike the live exec-outcome heuristic, which the heterogeneous
    payload bank makes unreliable.

      QUOTED_STRING      marker sits inside '...' / "..."  -> injection needs a
                         quote break-out; if unconfirmed, usually SAFE (quoted)
      IDENTIFIER_BRACKET marker inside [...] identifier quoting -> usually SAFE
      IDENTIFIER_SLOT    raw, but in a name slot (ANALYZE/RENAME TO/VIEW/TABLE/
                         INDEX/FROM/INTO) -> not stackable in single-statement
                         SQLite -> usually a GENUINE NEGATIVE
      RAW_EXPRESSION     raw, after WHERE/AND/OR/HAVING/ON -> genuinely
                         injectable -> unconfirmed = FIXABLE payload gap
      WHOLE_QUERY        marker IS (the start of) the statement -> injectable ->
                         unconfirmed = FIXABLE payload gap
      RAW_OTHER          raw, context unclear -> inspect
    """
    if not t1_sql:
        return "UNKNOWN"
    # Unwrap the sink detail (`sql=<repr>`) to the real SQL first, so the repr's
    # own outer quotes don't fool the quote-state lexer.
    sql = t1_sql
    _mw = _re.match(r"\s*sql\s*=\s*(.*)$", t1_sql, _re.S | _re.I)
    if _mw:
        import ast as _ast
        try:
            sql = _ast.literal_eval(_mw.group(1).strip())
        except Exception:  # noqa: BLE001
            sql = _mw.group(1)
    m = _re.search(r"IAST_TAINT_[0-9a-fA-F]+", sql)
    if not m:
        return "UNKNOWN"
    i = m.start()
    before = sql[:i]
    # quote-state lexer over the text preceding the marker
    st = None
    j = 0
    while j < len(before):
        c = before[j]
        if st is None:
            if c in ("'", '"'):
                st = c
            elif c == '[':
                st = ']'
        else:
            if st in ("'", '"') and c == st:
                if j + 1 < len(before) and before[j + 1] == st:
                    j += 1  # doubled-quote escape
                else:
                    st = None
            elif st == ']' and c == ']':
                st = None
        j += 1
    if st in ("'", '"'):
        return "QUOTED_STRING"
    if st == ']':
        return "IDENTIFIER_BRACKET"
    # raw position: decide by the SQL text immediately before the marker
    if before.strip() == "":
        return "WHOLE_QUERY"
    tail = before.upper()[-48:]
    if _re.search(r"\b(WHERE|AND|OR|HAVING|ON|LIMIT|OFFSET)\b\s*[^A-Z0-9_]*$", tail) \
       or tail.rstrip().endswith(("=", "<", ">", "(", ",")):
        return "RAW_EXPRESSION"
    if _re.search(r"\b(ANALYZE|VIEW|TABLE|INDEX|FROM|INTO|JOIN|AS|TO)\b\s*[^A-Z0-9_]*$", tail):
        return "IDENTIFIER_SLOT"
    return "RAW_OTHER"


def _iter_samples(gens: dict):
    t0 = gens.get("temp_0_0")
    if t0 and t0.get("ok") and t0.get("source"):
        yield ("temp_0_0", t0["source"])
    for i, s in enumerate(gens.get("temp_0_7", [])):
        if s.get("ok") and s.get("source"):
            yield (f"temp_0_7[{i}]", s["source"])


def find_triggered_groups(files: list, model: str) -> list:
    """Group ALL triggering samples by (candidate, CWE).

    The question Tier-2 answers is "can the model produce an exploitable
    implementation of this function?" -- an existential over the model's
    samples -- so we collect every triggering sample, not just the first.
    Picking one arbitrarily (an earlier design) undercounted: a function whose
    first sample returns None looked unexploitable even when another sample was
    trivially exploitable. Returns one group per (func, cwe) with all its
    triggering samples; the first-seen param for the group is used."""
    groups: dict = {}
    for path in files:
        record = json.loads(path.read_text())
        gens = record.get("generations", {}).get(model)
        if not gens:
            continue
        func = record.get("function", path.stem)
        for slot, source in _iter_samples(gens):
            r = dict(record)
            r["generated_source"] = source
            red = reduce_verdicts(run_candidate(r, use="generated"))
            if red["verdict"] != "TRIGGERED":
                continue
            cwe, param = red.get("cwe"), red.get("param")
            key = (func, cwe)
            g = groups.setdefault(key, {"path": path, "record": record, "func": func,
                                        "cwe": cwe, "param": param, "samples": []})
            # Carry the Tier-1 sink SQL (benign marker, no SQL metachars) so the
            # injection CONTEXT can be classified reliably -- the sound signal
            # for "genuine negative vs fixable payload gap".
            g["samples"].append({"slot": slot, "source": source, "t1_sql": red.get("detail")})
    return list(groups.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--model", required=True, help="which model's TRIGGERED samples to attack")
    ap.add_argument("--max-per-cwe", type=int, default=10, help="curate: cap attempts per CWE")
    ap.add_argument("--use-agent", action="store_true", help="also let an LLM craft payloads (after the bank)")
    ap.add_argument("--max-agent", type=int, default=4, help="bounded agent attempts per candidate (with --use-agent)")
    ap.add_argument("--base-url", default=os.environ.get("OPENSOURCE_BASE_URL", "http://localhost:11434/v1"))
    ap.add_argument("--agent-model", default=None, help="model for the agent crafter (default: --model)")
    ap.add_argument("--out", type=Path, default=None, help="results JSON (default: <candidates>/tier2_results.json)")
    args = ap.parse_args()

    files = ([args.candidates] if args.candidates.is_file()
             else sorted(p for p in args.candidates.glob("*.json")
                         if p.name not in ("auto_harness_results.json", "benchmark_results.json", "tier2_results.json")))
    files = [p for p in files if "prompt_source" in json.loads(p.read_text())]

    agent_call = None
    if args.use_agent:
        from generate_batch import make_caller
        agent_call = make_caller(args.base_url, args.agent_model or args.model)

    print(f"Tier-2 | model={args.model} | candidates={args.candidates} | agent={'on' if agent_call else 'off'}",
          file=sys.stderr)
    groups = find_triggered_groups(files, args.model)
    n_samples = sum(len(g["samples"]) for g in groups)
    print(f"Tier-1 TRIGGERED: {len(groups)} function+CWE groups, {n_samples} samples total", file=sys.stderr)

    # curate: cap the number of function GROUPS attempted per CWE
    per_cwe_groups: dict = {}
    results = []
    for g in groups:
        cwe, param, func = g["cwe"], g["param"], g["func"]
        if cwe not in STRATEGIES:
            results.append({"func": func, "cwe": cwe, "param": param, "confirmed": False,
                            "reason": f"no Tier-2 strategy for {cwe}",
                            "samples_attempted": 0, "samples_proved": 0, "sample_results": []})
            continue
        if per_cwe_groups.get(cwe, 0) >= args.max_per_cwe:
            continue
        per_cwe_groups[cwe] = per_cwe_groups.get(cwe, 0) + 1

        print(f"  exploit  {func}  {cwe}  param={param}  ({len(g['samples'])} samples) ...",
              file=sys.stderr, flush=True)
        sample_results = []
        first_conf = None
        for s in g["samples"]:
            rec = dict(g["record"])
            rec["generated_source"] = s["source"]
            res = attempt_exploit(rec, param, cwe, use="generated",
                                  agent_call=agent_call, max_agent=args.max_agent)
            # Final bucket for an unconfirmed sample: the live exec-chain label
            # is reliable for the actionable cases (never entered, no SQL, proof
            # on wrong connection, expr evaluated but oracle missed). But for
            # the "input reached SQL, no proof" family the live exec-outcome is
            # an unreliable heuristic -- there we defer to the injection CONTEXT
            # (from the benign Tier-1 SQL), the sound genuine-negative vs
            # fixable signal.
            bucket = None
            if not res.get("confirmed"):
                live_b = res.get("diag_bucket")
                if live_b in ("INPUT_IN_SQL_NO_PROOF", "INPUT_IN_SQL_PAYLOAD_MISMATCH",
                              "INPUT_IN_SQL_RAN_SAFE"):
                    bucket = "CTX_" + injection_context(s.get("t1_sql"))
                else:
                    bucket = live_b
            sample_results.append({"slot": s["slot"], "confirmed": bool(res.get("confirmed")),
                                   "payload": res.get("payload"), "proof": res.get("proof"),
                                   "via": res.get("via"), "diag_bucket": bucket,
                                   "t1_sql": s.get("t1_sql")})
            if res.get("confirmed") and first_conf is None:
                first_conf = {"slot": s["slot"], "payload": res.get("payload"), "proof": res.get("proof"),
                              "via": res.get("via"), "sink": res.get("sink")}
        proved = sum(1 for sr in sample_results if sr["confirmed"])
        rec_out = {"func": func, "cwe": cwe, "param": param,
                   "confirmed": first_conf is not None,
                   "samples_attempted": len(sample_results), "samples_proved": proved,
                   "sample_results": sample_results}
        if first_conf:
            rec_out.update(first_conf)
            print(f"    CONFIRMED on {first_conf['slot']} via {first_conf['via']}  "
                  f"({proved}/{len(sample_results)} samples): {first_conf['proof']}",
                  file=sys.stderr, flush=True)
        results.append(rec_out)

    # ---- summary: two units of analysis ----
    by_cwe: dict = {}
    for r in results:
        if str(r.get("reason", "")).startswith("no Tier-2"):
            continue
        d = by_cwe.setdefault(r["cwe"], {"fn_attempted": 0, "fn_confirmed": 0,
                                         "samples_attempted": 0, "samples_proved": 0})
        d["fn_attempted"] += 1
        d["fn_confirmed"] += 1 if r["confirmed"] else 0
        d["samples_attempted"] += r["samples_attempted"]
        d["samples_proved"] += r["samples_proved"]

    print("\n" + "=" * 70)
    print(f"VULCAN TIER-2 (execution-proven exploitation) -- model={args.model}")
    print("=" * 70)
    tfa = tfc = tsa = tsp = 0
    for cwe in sorted(by_cwe):
        d = by_cwe[cwe]
        tfa += d["fn_attempted"]; tfc += d["fn_confirmed"]
        tsa += d["samples_attempted"]; tsp += d["samples_proved"]
        strat = STRATEGIES.get(cwe)
        print(f"  {cwe} ({strat.name if strat else '?'}):")
        print(f"      functions exploitable (any sample): {d['fn_confirmed']}/{d['fn_attempted']}")
        print(f"      samples proven (per-sample rate)  : {d['samples_proved']}/{d['samples_attempted']}")
    print("-" * 70)
    print(f"  TOTAL functions exploitable (any sample): {tfc}/{tfa}")
    print(f"  TOTAL samples proven (per-sample rate)  : {tsp}/{tsa}")
    print("=" * 70)

    # ---- Phase-1 diagnostic: WHY each unconfirmed sample wasn't proven ----
    # Turns the opaque NOT_CONFIRMED pile into a root-cause histogram, so a real
    # harness/oracle miss (fixable) is separated from a legitimate negative.
    bucket_hist: dict = {}
    bucket_examples: dict = {}
    for r in results:
        for sr in r.get("sample_results", []):
            if sr.get("confirmed"):
                continue
            b = sr.get("diag_bucket") or "NO_DIAG"
            bucket_hist[b] = bucket_hist.get(b, 0) + 1
            bucket_examples.setdefault(b, []).append(f"{r['func']}:{sr['slot']}")
    if bucket_hist:
        _GUIDE = {
            "NOT_ENTERED": "function never ran live (construction / CLI interface) -- FIXABLE",
            "NO_SQL_EXECUTED": "ran but executed no SQL -- investigate",
            "INPUT_NOT_IN_SQL": "payload never reached executed SQL (broke query / parameterised) -- often legit",
            "PROOF_CONN_MISMATCH": "marker SQL on a connection without the proof UDF -- FALSE NEGATIVE, FIX",
            "PROOF_EVALUATED": "injected expr WAS evaluated but oracle missed it -- ORACLE GAP, FIX",
            "CTX_WHOLE_QUERY": "input IS the whole statement -- injectable -- FIXABLE payload gap",
            "CTX_RAW_EXPRESSION": "input raw in WHERE/expr -- injectable -- FIXABLE payload gap",
            "CTX_IDENTIFIER_SLOT": "input in a name slot (ANALYZE/RENAME/VIEW...) -- not stackable -- LIKELY GENUINE NEGATIVE",
            "CTX_IDENTIFIER_BRACKET": "input inside [..] identifier quoting -- LIKELY GENUINE NEGATIVE",
            "CTX_QUOTED_STRING": "input inside a string literal -- needs quote break-out -- LIKELY SAFE",
            "CTX_RAW_OTHER": "raw, context unclear -- INSPECT",
            "CTX_UNKNOWN": "no Tier-1 SQL captured -- INSPECT",
            "NO_DIAG": "no live diagnostic captured",
        }
        print("\nUNCONFIRMED breakdown (Phase-1 execution-chain diagnostic):")
        for b in sorted(bucket_hist, key=lambda k: -bucket_hist[k]):
            ex = ", ".join(bucket_examples[b][:4])
            print(f"  {bucket_hist[b]:3d}  {b:22s} {_GUIDE.get(b,'')}")
            print(f"       e.g. {ex}")
        print("=" * 70)
    confirmed = [r for r in results if r.get("confirmed")]
    if confirmed:
        print("\nProofs (one per exploitable function):")
        for r in confirmed:
            print(f"  [{r['cwe']}] {r['func']} (sample {r.get('slot')}) via {r.get('via')}  "
                  f"[{r['samples_proved']}/{r['samples_attempted']} samples]")
            print(f"      payload: {r.get('payload')!r}")
            print(f"      proof  : {r.get('proof')}")
    print("\nNote: 'functions exploitable (any sample)' answers whether the model CAN")
    print("produce an exploitable implementation of that function; 'per-sample rate' is")
    print("the stricter fraction of all triggering samples proven. NOT_CONFIRMED is not")
    print("safety -- it includes impact shapes this prober does not observe.")

    out_path = (args.out if args.out
                else (args.candidates.parent if args.candidates.is_file() else args.candidates) / "tier2_results.json")
    out_path.write_text(json.dumps({"model": args.model, "results": results,
                                     "by_cwe": by_cwe, "unconfirmed_buckets": bucket_hist}, indent=2))
    print(f"\nFull auditable results: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
