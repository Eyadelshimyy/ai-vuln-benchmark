#!/usr/bin/env python3
"""Corpus v2 discovery: shortlist AI-authored repos likely to be web apps
(validation/corpus_v2/CORPUS_V2_PROTOCOL.md §2).

Commit search for the strong AI signatures of analysis/discover_repos.py, each
ANDed with one web keyword. Uses only the search API (works unauthenticated at
~10 req/min; set GITHUB_TOKEN for 30/min). No per-repo metadata calls -- the
application-level decision is made later from the code itself
(analysis/app_screen.py). Forks and v1 repos are dropped here; everything else
is a LEAD, not an inclusion.

Resumable: per-query results are kept in --out and skipped on re-run.

  python3 analysis/corpus_v2_discover.py --out validation/corpus_v2/leads.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discover_repos import API, _QUERIES  # noqa: E402

WEB_KEYWORDS = ["fastapi", "flask", "django", "endpoint", "route"]
# authenticated search allows 30 req/min; unauthenticated 10 (and trips abuse limits)
PAUSE = 2.5 if (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")) else 8


def _get_search(url: str, tries: int = 8) -> tuple[dict | None, str | None]:
    """GET one search page. Unauthenticated search trips GitHub's secondary
    (abuse) limit easily, so back off hard; return (data, None) or (None, why)
    -- never an empty result that would look like "no hits"."""
    import urllib.error
    import urllib.request
    from discover_repos import _headers
    wait = 30
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()), timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "replace")), None
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                ra = e.headers.get("Retry-After")
                time.sleep(int(ra) if ra and ra.isdigit() else wait)
                wait = min(wait * 2, 300)
                continue
            return None, f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(10)
    return None, "rate-limited: gave up after retries"


def search(query: str, pages: int, per_page: int = 30) -> tuple[list[dict], int | None, str | None, bool]:
    hits, total, incomplete = [], None, False
    for page in range(1, pages + 1):
        url = (f"{API}/search/commits?q={urllib.parse.quote(query)}&per_page={per_page}"
               f"&page={page}&sort=committer-date&order=desc")
        data, err = _get_search(url)
        if err:
            return hits, total, err, incomplete
        total = data.get("total_count", total)
        incomplete = incomplete or bool(data.get("incomplete_results"))
        for it in data.get("items", []):
            repo = it.get("repository") or {}
            if repo.get("full_name"):
                hits.append({"repo": repo["full_name"], "fork": bool(repo.get("fork")),
                             "private": bool(repo.get("private")), "sha": it.get("sha"),
                             "date": ((it.get("commit") or {}).get("committer") or {}).get("date")})
        if len(data.get("items", [])) < per_page:
            break
        time.sleep(PAUSE)
    return hits, total, None, incomplete


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pages", type=int, default=10, help="pages per query (GitHub caps a query at 1000 results)")
    ap.add_argument("--per-page", type=int, default=100)
    ap.add_argument("--v1-root", type=Path, default=Path("calibration/mined_prompts"))
    args = ap.parse_args()
    state = json.loads(args.out.read_text()) if args.out.exists() else {"queries": {}}
    for tool, qs in _QUERIES.items():
        for base in qs:
            for kw in WEB_KEYWORDS:
                q = f"{base} {kw}"
                if q in state["queries"] and not state["queries"][q].get("error"):
                    continue
                print(f"[{tool}] {q}", file=sys.stderr, flush=True)
                hits, total, err, incomplete = search(q, args.pages, args.per_page)
                state["queries"][q] = {"tool": tool, "keyword": kw, "total_count": total, "hits": hits,
                                       "error": err, "incomplete_results": incomplete,
                                       "searched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                print(f"   -> {len(hits)} hits (total {total}){'  ERROR ' + err if err else ''}",
                      file=sys.stderr, flush=True)
                args.out.write_text(json.dumps(state, indent=1))
                time.sleep(PAUSE)
    # aggregate to leads
    v1 = {p.name.replace("__", "/", 1).lower() for p in args.v1_root.iterdir() if p.is_dir()}
    leads, dropped = {}, {"fork": set(), "private": set(), "in_v1": set()}
    failed = [q for q, rec in state["queries"].items() if rec.get("error")]
    if failed:
        print(f"WARNING: {len(failed)} queries failed (re-run to retry): {failed[:3]}", file=sys.stderr)
    state["failed_queries"] = failed
    for q, rec in state["queries"].items():
        for h in rec["hits"]:
            r = h["repo"]
            if h["fork"]:
                dropped["fork"].add(r); continue
            if h["private"]:
                dropped["private"].add(r); continue
            if r.lower() in v1:
                dropped["in_v1"].add(r); continue
            e = leads.setdefault(r, {"tools": set(), "keywords": set(), "hits": 0, "latest": ""})
            e["tools"].add(rec["tool"]); e["keywords"].add(rec["keyword"]); e["hits"] += 1
            e["latest"] = max(e["latest"], h.get("date") or "")
    state["leads"] = {r: {**e, "tools": sorted(e["tools"]), "keywords": sorted(e["keywords"])}
                      for r, e in sorted(leads.items(), key=lambda kv: (-kv[1]["hits"], kv[0]))}
    state["dropped"] = {k: sorted(v) for k, v in dropped.items()}
    args.out.write_text(json.dumps(state, indent=1))
    print(f"queries {len(state['queries'])}; leads {len(state['leads'])}; dropped "
          + ", ".join(f"{k}={len(v)}" for k, v in state["dropped"].items()))


if __name__ == "__main__":
    main()
