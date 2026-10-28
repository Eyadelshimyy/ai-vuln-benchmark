#!/usr/bin/env python3
"""Corpus v2 application-level screen (CORPUS_V2_PROTOCOL.md §3).

For each discovered lead: shallow clone (no history, large blobs skipped, no
code executed), then a static decision recorded for EVERY lead:

  APP_LEVEL  iff  (1) >= 10 non-test .py files
                  (2) a web app object is constructed in non-test code
                      (Flask( / FastAPI( / Starlette( / web.Application( / Sanic(,
                       or a Django project: ROOT_URLCONF in a settings module
                       and `urlpatterns` in a urls.py)
                  (3) >= 3 HTTP route handlers in non-test code
  otherwise  REJECT with the first failing criterion as the reason.

  python3 analysis/app_screen.py --leads validation/corpus_v2/leads.json \
      --out validation/corpus_v2/app_screen.json --work /tmp/v2_screen
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SKIP_DIRS = {".git", ".venv", "venv", "env", "site-packages", "node_modules", "vendor",
              "third_party", "__pycache__", "build", "dist", ".tox", "migrations"}
_APP_OBJ = {
    "flask": re.compile(r"\bFlask\s*\("),
    "fastapi": re.compile(r"\bFastAPI\s*\("),
    "starlette": re.compile(r"\bStarlette\s*\("),
    "aiohttp": re.compile(r"\bweb\.Application\s*\("),
    "sanic": re.compile(r"\bSanic\s*\("),
}
_ROUTE_DECOR = re.compile(r"^\s*@\s*[A-Za-z_][\w.]*\.(?:route|get|post|put|patch|delete|api_route|websocket)\s*\(",
                          re.M)
_ROUTE_CALL = re.compile(r"\.(?:add_url_rule|add_api_route|add_route)\s*\(")
_DJ_PATH = re.compile(r"\b(?:re_)?path\s*\(")


def is_test(rel: Path) -> bool:
    parts = [p.lower() for p in rel.parts]
    return any(p in ("tests", "test", "testing") for p in parts[:-1]) or \
        parts[-1].startswith("test_") or parts[-1].endswith("_test.py") or parts[-1] == "conftest.py"


def py_files(root: Path):
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts[:-1]):
            continue
        yield p, rel


def screen_tree(root: Path) -> dict:
    n_py = 0; frameworks = set(); routes = 0; dj_settings = False; dj_urls = False
    for p, rel in py_files(root):
        if is_test(rel):
            continue
        n_py += 1
        try:
            t = p.read_text(errors="ignore")
        except OSError:
            continue
        if len(t) > 2_000_000:
            continue
        for fw, rx in _APP_OBJ.items():
            if rx.search(t):
                frameworks.add(fw)
        routes += len(_ROUTE_DECOR.findall(t)) + len(_ROUTE_CALL.findall(t))
        if "ROOT_URLCONF" in t:
            dj_settings = True
        if rel.name == "urls.py" and "urlpatterns" in t:
            dj_urls = True
            routes += len(_DJ_PATH.findall(t))
    if dj_settings and dj_urls:
        frameworks.add("django")
    if n_py < 10:
        decision, reason = "REJECT", f"only {n_py} non-test .py files (<10)"
    elif not frameworks:
        decision, reason = "REJECT", "no web application object constructed"
    elif routes < 3:
        decision, reason = "REJECT", f"only {routes} route handlers (<3)"
    else:
        decision, reason = "APP_LEVEL", None
    return {"decision": decision, "reason": reason, "n_py_nontest": n_py,
            "frameworks": sorted(frameworks), "route_handlers": routes}


def screen_repo(full_name: str, work: Path) -> dict:
    dest = work / full_name.replace("/", "__")
    shutil.rmtree(dest, ignore_errors=True)
    cmd = ["git", "-c", "http.version=HTTP/1.1", "clone", "--quiet", "--depth", "1",
           "--filter=blob:limit=2m", "--no-tags", f"https://github.com/{full_name}.git", str(dest)]
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                            env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"})
    except subprocess.TimeoutExpired:
        shutil.rmtree(dest, ignore_errors=True)
        return {"decision": "REJECT", "reason": "clone timed out (300s)"}
    if pr.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        return {"decision": "REJECT", "reason": f"clone failed: {(pr.stderr or '').strip()[-150:]}"}
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = screen_tree(dest)
    res["head"] = head
    shutil.rmtree(dest, ignore_errors=True)   # mining re-clones with full history
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leads", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    leads = json.loads(args.leads.read_text())["leads"]
    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    todo = [r for r in leads if r not in out]
    args.work.mkdir(parents=True, exist_ok=True)
    print(f"screening {len(todo)} leads ({len(out)} already done)", file=sys.stderr)
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(screen_repo, r, args.work): r for r in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                out[r] = {**f.result(), "lead": leads[r]}
            except Exception as e:  # noqa: BLE001
                out[r] = {"decision": "REJECT", "reason": f"screen error: {type(e).__name__}: {e}", "lead": leads[r]}
            if i % 10 == 0 or i == len(todo):
                args.out.write_text(json.dumps(out, indent=1, sort_keys=True))
                print(f"  {i}/{len(todo)}", file=sys.stderr, flush=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    from collections import Counter
    print(Counter(v["decision"] for v in out.values()))


if __name__ == "__main__":
    main()
