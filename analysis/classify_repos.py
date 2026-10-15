#!/usr/bin/env python3
"""Classify each mined repo by SOFTWARE ARCHETYPE, so findings can be read in the
only honest frame: a confirmed sink means different things in a deployed web
service, a library, an agent tool, and a tutorial/fixture.

We do NOT claim to know what is "live on the internet" -- that is unknowable from
a repo. We classify the archetype (what KIND of software it is), which is what an
exploitability claim actually rests on:

  web_service   network-facing app with real routes (remote attack surface IF deployed)
  agent_tool    reachable via a model tool-call / MCP server
  library_cli   installed & called; attack surface only via a caller / local use
  build_script  offline data/build pipeline (runs in dev/CI, not a served surface)
  tutorial_demo not real software: numbered lessons, examples, evals/fixtures

Signals (per repo's vendored _repo_src):
  - web: imports of fastapi/flask/django/starlette/aiohttp.web or route decorators
  - agent/mcp: imports of mcp / an llm sdk, or a tool-dispatch shape
  - packaging: setup.py / pyproject with console entry points
  - tutorial/fixture: path markers (lesson/example/fixture/demo/tutorial/skills) and
    README keywords
The label is a heuristic PROPOSAL -- eyeball the table and override where wrong
(a thesis author owns the final call). An optional --overrides JSON ({repo: label})
is applied on top.

  python3 analysis/classify_repos.py --root calibration/mined_prompts \
      --repos mid_ai_repos.json [--gold gold_set.json] [--overrides overrides.json]
"""
from __future__ import annotations
import argparse, ast, json, re
from collections import Counter
from pathlib import Path

WEB_IMPORTS = ("fastapi", "flask", "django", "starlette", "aiohttp", "sanic", "bottle",
               "tornado", "quart", "falcon")
WEB_DECOS = ("route", "get", "post", "put", "delete", "patch", "api_route", "endpoint",
             "websocket", "add_url_rule")
AGENT_HINTS = ("mcp", "anthropic", "openai", "langchain", "llama_index", "tool_call",
               "tool_use", "execute_tool", "FastMCP", "@mcp", "@tool")
# NOTE: tests/ are NOT a tutorial signal -- real software has tests. Only
# fixtures/examples/demos/lessons/samples indicate example-not-product code.
FIXTURE_PATH = re.compile(r"(^|/)(fixtures?|examples?|example|demos?|samples?|"
                          r"lessons?|tutorials?|\d\d-[a-z])", re.I)
FIXTURE_NAME = re.compile(r"(tutorial|example|lesson|teaching|learn|demo|sample|fixture|"
                          r"skills?)\b", re.I)


def _py_files(root: Path):
    return [p for p in root.rglob("*.py")]


def classify(repo_src: Path, repo_name: str = "") -> dict:
    files = _py_files(repo_src)
    n = len(files)
    web = agent = 0
    for p in files:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        low = txt.lower()
        if any(f"import {m}" in low or f"from {m}" in low for m in WEB_IMPORTS):
            web += 1
        if any(h.lower() in low for h in AGENT_HINTS):
            agent += 1
        # route decorators
        if re.search(r"@\w+\.(route|get|post|put|delete|patch|websocket)\b", txt):
            web += 1

    rels = [str(p.relative_to(repo_src)) for p in files]
    fixture_files = sum(1 for r in rels if FIXTURE_PATH.search(r))
    fixture_ratio = fixture_files / n if n else 0.0
    name_fixture = bool(FIXTURE_NAME.search(repo_name))
    has_pkg = any((repo_src / f).exists() for f in ("setup.py", "pyproject.toml", "setup.cfg"))

    # numbered-lesson layout (01-x/ 02-y/) is a dead giveaway of a tutorial
    top = {r.split("/")[0] for r in rels}
    numbered = sum(1 for t in top if re.match(r"\d\d[-_]", t))

    # ---- decide primary archetype (ordered; a strong web signal beats a fixture
    # ratio so a real service with lots of tests isn't mislabeled a demo) ----
    if numbered >= 2 and web < 3:
        label = "tutorial_demo"            # numbered lessons (01-x/ 02-y/)
    elif web >= 3:
        label = "web_service"
    elif name_fixture or fixture_ratio > 0.6:
        label = "tutorial_demo"
    elif agent >= 2:
        label = "agent_tool"
    elif has_pkg:
        label = "library_cli"
    else:
        label = "build_script"

    return {"repo": repo_name, "py_files": n, "web_files": web, "agent_files": agent,
            "fixture_ratio": round(fixture_ratio, 2), "numbered_dirs": numbered,
            "has_packaging": has_pkg, "archetype": label}


def _vuln_file_kind(path: str) -> str:
    """Where the vulnerable function LIVES -- a per-finding modifier."""
    p = path.replace("\\", "/").lower()
    if re.search(r"(^|/)(scripts?|build|tools?|bin)/", p):
        return "build/script-time"
    if re.search(r"(^|/)(tests?|fixtures?|examples?)/", p):
        return "test/fixture"
    return "runtime"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--repos", type=Path, default=None, help="discovery json for star counts")
    ap.add_argument("--gold", type=Path, default=None, help="gold_set.json to slice by archetype")
    ap.add_argument("--overrides", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("repo_archetypes.json"))
    args = ap.parse_args()

    stars = {}
    if args.repos and args.repos.exists():
        for r in json.loads(args.repos.read_text()):
            stars[r.get("full_name", "")] = r.get("stars") or r.get("stargazers_count")
    overrides = json.loads(args.overrides.read_text()) if args.overrides and args.overrides.exists() else {}

    rows = []
    for d in sorted(args.root.iterdir()):
        src = d / "_repo_src"
        if not src.is_dir():
            continue
        full = d.name.replace("__", "/", 1)
        info = classify(src, full)
        if full in overrides:
            info["archetype"] = overrides[full]
            info["overridden"] = True
        info["stars"] = stars.get(full)
        rows.append(info)

    by_arch = Counter(r["archetype"] for r in rows)
    print("=" * 70)
    print("  REPO ARCHETYPES  (heuristic -- eyeball & override where wrong)")
    print("=" * 70)
    print(f"  {'repo':34} {'archetype':14} web ag fx★")
    for r in sorted(rows, key=lambda r: r["archetype"]):
        st = "" if r.get("stars") is None else f" {r['stars']}★"
        ov = " (override)" if r.get("overridden") else ""
        print(f"  {r['repo'][:34]:34} {r['archetype']:14} "
              f"{r['web_files']:>3}{r['agent_files']:>3} {r['fixture_ratio']:>4}{st}{ov}")
    print(f"\n  archetype totals: {dict(by_arch)}")

    out = {"repos": rows, "by_archetype": dict(by_arch)}

    # ---- slice confirmed findings by archetype (+ where the sink lives) ----
    if args.gold and args.gold.exists():
        gold = json.loads(args.gold.read_text())
        arch_of = {r["repo"]: r["archetype"] for r in rows}
        seen = set()
        sliced = []
        for g in gold:
            key = g.get("census_file") or (g["repo"], g["function"])
            if key in seen:
                continue
            seen.add(key)
            sliced.append({"repo": g["repo"], "function": g["function"],
                           "cwe": g.get("exec_cwe"), "entry_kind": g.get("entry_kind"),
                           "archetype": arch_of.get(g["repo"], "?"),
                           "sink_location": _vuln_file_kind(g.get("file", ""))})
        out["findings_by_archetype"] = sliced
        tally = Counter(s["archetype"] for s in sliced)
        print("\n  CONFIRMED+REACHABLE findings by archetype:")
        for a, c in tally.most_common():
            print(f"    {a:14} {c}")
        print("\n  (archetype × sink location):")
        for s in sorted(sliced, key=lambda s: (s["archetype"], s["sink_location"])):
            print(f"    {s['archetype']:14} {s['sink_location']:16} "
                  f"{s['repo']:26} {s['function']}  [{s['entry_kind']}]")

    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
