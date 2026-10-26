#!/usr/bin/env python3
"""Replay the recorded legacy-portal corpus against a deployed target and diff it with the Java capture.

Adapted from run_parity.py on the legacy-portal-dotnet branch (PR #1761). The Java side is always
java-reference.json, the response the Java monolith gave when the corpus was recorded; the target
side is a live base URL (for example the API Gateway HTTP API of a legacy-portal-serverless run).

Compared per request: HTTP status, media type (Content-Type without parameters) and the body.
JSON bodies are compared as serialized JSON with key order and number formatting preserved, and
wall-clock timestamps are replaced by a placeholder naming their format.

Stages:
  first  run each context in corpus order and stop that context at its first divergence,
         printing the diff (the pause point before any adapter fix)
  full   run every case of each selected context

  python3 replay.py --base https://abc.execute-api.us-east-1.amazonaws.com --context all --stage first
  python3 replay.py --base http://localhost:8095 --context feedback --stage full --out /tmp/replay

The corpus is ordered and stateful, so reset the target's schemas before every replay.
"""
import argparse
import hashlib
import http.client
import json
import re
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTEXTS = ["common", "announcements", "preferences", "feedback"]
ALIASES = {"user_preferences": "preferences", "user-preferences": "preferences", "ann": "announcements",
           "pref": "preferences", "fb": "feedback"}
ISO_INSTANT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.(\d{3}|\d{6}|\d{9}))?Z$")
SPRING_ERROR_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00$")
ANY_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str):
        if ISO_INSTANT.match(value):
            return "<instant:ISO_INSTANT>"
        if SPRING_ERROR_TS.match(value):
            return "<timestamp:yyyy-MM-ddTHH:mm:ss.SSS+00:00>"
        if ANY_DATETIME.match(value):
            return "<datetime:unrecognized-format:" + value + ">"
    return value


def verify_checksums(directory):
    sums = directory / "SHA256SUMS"
    failures = []
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")
    return failures


def send(base, case, timeout):
    url = urllib.parse.urlsplit(base)
    if url.scheme not in ("http", "https") or not url.hostname:
        raise SystemExit(f"base URL must be http(s)://host[:port], got {base!r}")
    conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(url.hostname, url.port, timeout=timeout)
    headers = {"Accept": case.get("accept", "application/json")}
    data = case.get("body")
    payload = None
    if data is not None:
        payload = data.encode(case.get("bodyEncoding", "utf-8"))
        headers["Content-Type"] = case.get("contentType", "application/json")
    started = time.monotonic()
    try:
        conn.request(case["method"], url.path.rstrip("/") + case["path"], body=payload, headers=headers)
        resp = conn.getresponse()
        status, raw = resp.status, resp.read()
        content_type = resp.getheader("Content-Type") or ""
    finally:
        conn.close()
    elapsed_ms = round((time.monotonic() - started) * 1000)
    media = content_type.split(";")[0].strip().lower()
    return {"status": status, "mediaType": media, "body": raw.decode("utf-8", errors="replace")}, elapsed_ms


def canonical(resp):
    text = resp["body"]
    if not text.strip():
        return {"status": resp["status"], "mediaType": resp["mediaType"] or None, "json": None, "raw": ""}
    try:
        parsed = json.loads(text)
    except ValueError:
        return {"status": resp["status"], "mediaType": resp["mediaType"], "json": None, "raw": text}
    return {"status": resp["status"], "mediaType": resp["mediaType"],
            "json": json.dumps(normalize(parsed), ensure_ascii=False), "raw": None}


def media_equivalent(a, b, strict):
    if strict:
        return a == b

    def family(m):
        return "json" if m and (m == "application/json" or m.endswith("+json")) else m
    return family(a) == family(b)


def compare(java_resp, target_resp, strict):
    jc, tc = canonical(java_resp), canonical(target_resp)
    problems = []
    if jc["status"] != tc["status"]:
        problems.append(f"status {jc['status']} != {tc['status']}")
    if not media_equivalent(jc["mediaType"], tc["mediaType"], strict):
        problems.append(f"media type {jc['mediaType']} != {tc['mediaType']}")
    if jc["json"] != tc["json"] or jc["raw"] != tc["raw"]:
        problems.append("body differs")
    return jc, tc, problems


def parse_contexts(values):
    selected = []
    for value in values or ["all"]:
        for item in value.split(","):
            item = ALIASES.get(item.strip(), item.strip())
            if item == "all":
                selected.extend(CONTEXTS)
            elif item in CONTEXTS:
                selected.append(item)
            else:
                raise SystemExit(f"unknown context {item!r}; use one of {', '.join(CONTEXTS)}, user_preferences or all")
    return [c for c in CONTEXTS if c in selected]


def render_body(c):
    return c["json"] if c["json"] is not None else (c["raw"] or "")


def run_context(ctx, cases, java, base, stage, strict, timeout):
    results, first_div = [], None
    for case in cases:
        target_resp, elapsed_ms = send(base, case, timeout)
        java_resp = java.get(case["id"])
        if java_resp is None:
            jc, tc, problems = None, canonical(target_resp), ["missing Java capture"]
        else:
            jc, tc, problems = compare(java_resp, target_resp, strict)
        result = {"id": case["id"], "method": case["method"], "path": case["path"], "match": not problems,
                  "problems": problems, "elapsedMs": elapsed_ms, "java": jc, "target": tc}
        results.append(result)
        if problems and first_div is None:
            first_div = result
            if stage == "first":
                break
    corpus = len(cases)
    identical = sum(1 for r in results if r["match"])
    return {"context": ctx, "stage": stage, "casesInCorpus": corpus, "casesRun": len(results),
            "identical": identical, "different": len(results) - identical,
            "firstDivergence": first_div["id"] if first_div else None,
            "divergentIds": [r["id"] for r in results if not r["match"]],
            "status": ("stopped at " + first_div["id"]) if (first_div and stage == "first")
            else ("all identical" if not first_div else f"{len(results) - identical} different"),
            "cases": results}


def summary_lines(base, stage, evidence, started):
    lines = ["# legacy-portal replay against the recorded Java responses", "",
             f"Target: `{base}`", "", "Java: `java-reference.json` (recorded capture)", "",
             f"Stage: `{stage}`", "", f"Started (UTC): {started}", "",
             "| Context | Cases in corpus | Cases run | Identical | Different | First divergence | Result |",
             "|---|---|---|---|---|---|---|"]
    for e in evidence:
        lines.append(f"| {e['context']} | {e['casesInCorpus']} | {e['casesRun']} | {e['identical']} | "
                     f"{e['different']} | {e['firstDivergence'] or '-'} | {e['status']} |")
    total_run = sum(e["casesRun"] for e in evidence)
    total_same = sum(e["identical"] for e in evidence)
    lines += ["", f"{total_same}/{total_run} replayed cases identical."]
    for e in evidence:
        for r in e["cases"]:
            if r["match"]:
                continue
            lines += ["", f"## {r['id']} ({e['context']})", "", f"`{r['method']} {r['path'][:120]}`: "
                      + "; ".join(r["problems"]), "", "Java (recorded):", "```",
                      f"{r['java']['status']} {r['java']['mediaType']}" if r["java"] else "missing",
                      render_body(r["java"]) if r["java"] else "", "```", "Target:", "```",
                      f"{r['target']['status']} {r['target']['mediaType']}", render_body(r["target"]), "```"]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="target base URL, for example the HTTP API invoke URL")
    ap.add_argument("--context", action="append",
                    help="common|announcements|preferences|feedback|all (repeatable or comma separated)")
    ap.add_argument("--stage", choices=["first", "full"], default="full")
    ap.add_argument("--requests", default=str(HERE / "requests.json"))
    ap.add_argument("--java-capture", default=str(HERE / "java-reference.json"))
    ap.add_argument("--out", default=str(HERE / "out"), help="directory for <context>.json and SUMMARY.md")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds")
    ap.add_argument("--strict-media-type", action="store_true",
                    help="treat application/json vs application/*+json as a difference")
    ap.add_argument("--skip-checksum", action="store_true", help="skip the SHA256SUMS check of the corpus")
    args = ap.parse_args()

    if not args.skip_checksum:
        failures = verify_checksums(Path(args.requests).resolve().parent)
        if failures:
            print("corpus checksum mismatch, refusing to replay:", *failures, sep="\n  ", file=sys.stderr)
            return 3
        print("corpus checksums: OK (SHA256SUMS)")

    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cases = json.loads(Path(args.requests).read_text())["cases"]
    java = {c["id"]: c["response"] for c in json.loads(Path(args.java_capture).read_text())["cases"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    evidence = []
    for ctx in parse_contexts(args.context):
        ctx_cases = [c for c in cases if c["context"] == ctx]
        result = run_context(ctx, ctx_cases, java, args.base, args.stage, args.strict_media_type, args.timeout)
        result.update({"base": args.base, "startedUtc": started})
        (out / f"{ctx}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        evidence.append(result)
        print(f"[{ctx}] {result['identical']}/{result['casesRun']} identical of {result['casesInCorpus']} "
              f"in corpus; {result['status']}")
        if args.stage == "first" and result["firstDivergence"]:
            r = result["cases"][-1]
            print(f"  first divergence {r['id']}: {r['method']} {r['path'][:100]}: {'; '.join(r['problems'])}")
            if r["java"]:
                print(f"  - java   {r['java']['status']} {r['java']['mediaType']} {render_body(r['java'])[:300]}")
            print(f"  + target {r['target']['status']} {r['target']['mediaType']} {render_body(r['target'])[:300]}")

    lines = summary_lines(args.base, args.stage, evidence, started)
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines[7:7 + len(evidence) + 3]))
    print(f"\nevidence: {out}/<context>.json, summary: {out / 'SUMMARY.md'}")
    return 0 if all(e["different"] == 0 for e in evidence) else 1


if __name__ == "__main__":
    sys.exit(main())
