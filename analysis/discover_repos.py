#!/usr/bin/env python3
"""VULCAN real-AI-repo DISCOVERY.

Seed the real-AI-repo arm by finding repositories whose git history carries
STRONG AI-authorship evidence -- the same machine-emitted signatures
analysis/provenance.py verifies per function. This is the "find AI repos"
front end: it queries GitHub's commit search for AI trailers / bot authors,
dedupes to candidate repos, filters to real Python projects, ranks them, and
(optionally) drives the full-clone -> provenance-gate -> mine pipeline so the
whole thing is one command.

WHY COMMIT SEARCH, NOT README/topic SEARCH: a commit trailer like
`Co-authored-by: Claude <noreply@anthropic.com>` or an author of
`devin-ai-integration[bot]` is emitted by the tool itself -- it is evidence,
not a self-applied label. A repo's README saying "built with AI" is marketing;
its history saying so is provenance. Discovery only SHORTLISTS repos; the
per-function gate in provenance.py still decides, at mine time, which exact
functions are AI-written. A repo surfacing here is a lead, not a guarantee.

AUTH / NETWORK: uses the GitHub REST API over HTTPS (api.github.com). Set a
token in GITHUB_TOKEN or GH_TOKEN for a sane rate limit (authenticated search
= 30 req/min; unauthenticated = 10/min and will crawl). No token still works
for a small run. Nothing here needs `gh`.

USAGE
  # shortlist Python repos with AI commit history, write a ranked list
  GITHUB_TOKEN=ghp_... python3 analysis/discover_repos.py --out ai_repos.json --limit-repos 40

  # one command: discover AND mine each (full clone + per-function AI gate)
  GITHUB_TOKEN=ghp_... python3 analysis/discover_repos.py --limit-repos 15 --mine --mine-limit 25

OUTPUT: a JSON list of {full_name, clone_url, tools, ai_commit_hits, stars,
language, pushed_at, archived, fork} ranked best-first, plus a printed table.
Every entry is a lead backed by at least one strong AI-commit match.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"

# Strong AI-authorship search queries -> tool label. These mirror the STRONG
# tier in analysis/provenance.py (machine trailers + known bot accounts).
_QUERIES = {
    "claude-code": [
        '"Co-authored-by: Claude" "noreply@anthropic.com"',
        '"Generated with Claude Code"',
    ],
    "github-copilot": [
        '"Co-authored-by: Copilot"',
        "author:copilot-swe-agent[bot]",
    ],
    "devin": ["author:devin-ai-integration[bot]"],
    "cursor": ["author:cursoragent"],
}


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json",
         "User-Agent": "vulcan-discover"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _get(url: str, retries: int = 3) -> dict | None:
    """GET a JSON endpoint, honoring GitHub's secondary/primary rate limits
    (sleep until X-RateLimit-Reset on a 403 rate-limit, then retry)."""
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=_headers())
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                reset = e.headers.get("X-RateLimit-Reset")
                now = int(time.time())
                wait = max(2, (int(reset) - now) if reset and reset.isdigit() else 15)
                wait = min(wait, 70)  # don't sleep forever in one call
                print(f"  rate-limited ({e.code}); waiting {wait}s ...", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  HTTP {e.code} for {url[:80]}", file=sys.stderr)
            return None
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"  request error: {e}", file=sys.stderr)
            time.sleep(3)
    return None


def search_commits(query: str, pages: int, per_page: int = 30) -> list[str]:
    """Return repo full_names from commit-search hits for one query."""
    repos: list[str] = []
    for page in range(1, pages + 1):
        q = urllib.parse.quote(query)
        url = f"{API}/search/commits?q={q}&per_page={per_page}&page={page}&sort=committer-date&order=desc"
        data = _get(url)
        if not data or "items" not in data:
            break
        for it in data["items"]:
            full = (it.get("repository") or {}).get("full_name")
            if full:
                repos.append(full)
        if len(data["items"]) < per_page:
            break
        time.sleep(2.2)  # stay under the search rate limit
    return repos


def repo_meta(full_name: str) -> dict | None:
    return _get(f"{API}/repos/{full_name}")


def discover(tools: list[str], pages: int) -> dict[str, dict]:
    """Aggregate repos across all queries -> {full_name: {tools:set, hits:int}}."""
    agg: dict[str, dict] = {}
    for tool in tools:
        for query in _QUERIES.get(tool, []):
            print(f"searching [{tool}]: {query}", file=sys.stderr)
            for full in search_commits(query, pages):
                e = agg.setdefault(full, {"tools": set(), "hits": 0})
                e["tools"].add(tool)
                e["hits"] += 1
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", default="claude-code,github-copilot,devin,cursor",
                    help="comma-separated subset of: " + ",".join(_QUERIES))
    ap.add_argument("--pages", type=int, default=2, help="commit-search pages per query (30/page)")
    ap.add_argument("--limit-repos", type=int, default=40, help="max repos in the final ranked list")
    ap.add_argument("--language", default="Python", help="required primary language ('any' to disable)")
    ap.add_argument("--min-stars", type=int, default=0)
    ap.add_argument("--include-forks", action="store_true")
    ap.add_argument("--include-archived", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("ai_repos.json"))
    ap.add_argument("--mine", action="store_true",
                    help="after ranking, full-clone + gate-mine each repo (mine_prompts.py "
                         "--require-ai-provenance)")
    ap.add_argument("--mine-limit", type=int, default=25, help="--mine: functions per repo")
    ap.add_argument("--from-json", type=Path, default=None,
                    help="skip GitHub search entirely and load the ranked repo list from this JSON "
                         "(a prior --out file). Use with --mine to clone+gate-mine the saved "
                         "shortlist without spending rate limit or needing a token.")
    args = ap.parse_args()

    # --from-json: reuse a saved shortlist, no search, no token needed.
    if args.from_json:
        rows = json.loads(args.from_json.read_text())
        print(f"Loaded {len(rows)} repos from {args.from_json} (no GitHub search).", file=sys.stderr)
        _mine_rows(rows, args) if args.mine else _print_rows(rows, args.language)
        return

    if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")):
        print("NOTE: no GITHUB_TOKEN/GH_TOKEN set -- search is limited to ~10 req/min and will "
              "be slow. A token raises it to ~30/min.", file=sys.stderr)

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    agg = discover(tools, args.pages)
    print(f"\n{len(agg)} distinct candidate repos from commit search; fetching metadata ...",
          file=sys.stderr)

    rows = []
    want_lang = None if args.language.lower() == "any" else args.language.lower()
    for full, e in agg.items():
        meta = repo_meta(full)
        time.sleep(0.6)
        if not meta or meta.get("message"):
            continue
        lang = (meta.get("language") or "")
        if want_lang and lang.lower() != want_lang:
            continue
        if meta.get("archived") and not args.include_archived:
            continue
        if meta.get("fork") and not args.include_forks:
            continue
        if (meta.get("stargazers_count") or 0) < args.min_stars:
            continue
        rows.append({
            "full_name": full,
            "clone_url": meta.get("clone_url"),
            "tools": sorted(e["tools"]),
            "ai_commit_hits": e["hits"],
            "stars": meta.get("stargazers_count", 0),
            "language": lang,
            "pushed_at": meta.get("pushed_at"),
            "archived": bool(meta.get("archived")),
            "fork": bool(meta.get("fork")),
            "size_kb": meta.get("size", 0),
        })

    # Rank: more AI-commit evidence first, then stars (as a proxy for a real,
    # non-toy project), then most recently pushed.
    rows.sort(key=lambda r: (r["ai_commit_hits"], r["stars"], r["pushed_at"] or ""), reverse=True)
    rows = rows[: args.limit_repos]

    _print_rows(rows, args.language)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {args.out} ({len(rows)} repos). Each is a LEAD -- the per-function "
          f"provenance gate decides which exact functions are AI-written at mine time.", file=sys.stderr)

    if args.mine:
        _mine_rows(rows, args)


def _print_rows(rows: list[dict], language: str) -> None:
    print("\n" + "=" * 78)
    print(f"RANKED AI-AUTHORED REPO SHORTLIST ({len(rows)})  [lang={language}]")
    print("=" * 78)
    print(f"{'repo':42} {'hits':>4} {'stars':>6}  tools")
    for r in rows:
        print(f"{r['full_name'][:42]:42} {r['ai_commit_hits']:>4} {r['stars']:>6}  {','.join(r['tools'])}")


def _mine_rows(rows: list[dict], args) -> None:
    mp = Path(__file__).resolve().parent / "mine_prompts.py"
    print("\n" + "=" * 78 + "\nMINING (full clone + per-function AI gate)\n" + "=" * 78)
    for i, r in enumerate(rows, 1):
        name = r["full_name"].replace("/", "__")
        print(f"\n--- [{i}/{len(rows)}] {r['full_name']} ---", flush=True)
        try:
            subprocess.run(
                [sys.executable, str(mp), "--repo-url", r["clone_url"],
                 "--repo-name", name, "--require-ai-provenance",
                 "--limit", str(args.mine_limit)],
                check=False)
        except Exception as ex:  # noqa: BLE001 -- one repo failing must not abort the batch
            print(f"  mine failed for {r['full_name']}: {ex}", file=sys.stderr)


if __name__ == "__main__":
    main()
