#!/usr/bin/env python3
"""Recover the generating MODEL (not just the agent product) behind each
AI-attributed commit, from the commit's own metadata.

provenance.py attributes a function to an agent PRODUCT (claude-code, cursor,
devin, github-copilot) from a matched signature fragment, and stores only that
fragment (e.g. "Co-Authored-By: Claude"). Several agents write the model into
the full trailer ("Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"),
and every commit carries a date. This script fetches the full commit message
and dates for the recorded `ai_commit` SHAs and extracts any model string.

Method: one commit-only bare clone per repo (`--filter=tree:0`: commits but no
trees/blobs, nothing checked out, no hooks run), then `git log -1` per SHA.
Read-only, public data; no API quota. Clones go under --work (default: a temp
dir), which is safe to delete afterwards.

Output (an ENRICHMENT artifact; the frozen corpus files are not modified):
  {"<repo>@<sha>": {"model": "Claude Opus 4.5" | null, "model_family": ...,
                    "trailer": "<matched line>", "commit_date": iso,
                    "resolved": bool, "error": str|null}}

  python3 analysis/commit_model_attribution.py --census corpus_census_reproduced.json \
      --out validation/rq1/commit_model_attribution.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

# Ordered: most specific first. Each yields (family, display string).
_MODEL_PATTERNS = [
    ("claude", re.compile(r"\bClaude\s+(?:(?:Opus|Sonnet|Haiku)\s+\d+(?:\.\d+)?|\d+(?:\.\d+)?\s+(?:Opus|Sonnet|Haiku))(?:\s*\(1M context\))?", re.I)),
    ("claude", re.compile(r"\bclaude-(?:opus|sonnet|haiku)-[\w.-]+", re.I)),
    ("gpt", re.compile(r"\bGPT-?\d+(?:\.\d+)?(?:-?(?:codex|mini|turbo|o))*\b", re.I)),
    ("openai-o", re.compile(r"\bo[134](?:-mini|-pro)?\b(?=[^\n]*(?:openai|codex))", re.I)),
    ("gemini", re.compile(r"\bGemini\s*\d+(?:\.\d+)?(?:\s*(?:Pro|Flash))?", re.I)),
]
_TRAILER_HINT = re.compile(r"co-authored-by|generated with|model:|assisted-by", re.I)


def extract_model(message: str) -> tuple[str | None, str | None, str | None]:
    """Return (model, family, line) from the first trailer-ish line naming a model."""
    # Trailer-ish lines ONLY: commit bodies routinely name models as the SUBJECT
    # of the change (e.g. litellm "add claude-sonnet-4 support"), which must not
    # be read as the model that wrote the commit.
    for line in message.splitlines():
        if not _TRAILER_HINT.search(line):
            continue
        for fam, rx in _MODEL_PATTERNS:
            m = rx.search(line)
            if m:
                return re.sub(r"\s+", " ", m.group(0)).strip(), fam, line.strip()[:200]
    return None, None, None


def _git(*args, cwd=None, timeout=600) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--pins", type=Path, default=Path("validation/corpus_repo_pins.json"))
    ap.add_argument("--work", type=Path, default=None, help="clone dir (default: fresh temp dir)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = json.loads(args.census.read_text())["rows"]
    pins = json.loads(args.pins.read_text())
    want = defaultdict(set)
    for r in rows:
        rec = json.loads((args.root / r["file"]).read_text())
        sha = (rec.get("ai_provenance") or {}).get("ai_commit")
        if sha:
            want[rec["repo"]].add(sha)
    work = args.work or Path(tempfile.mkdtemp(prefix="vulcan_commitmeta_"))
    work.mkdir(parents=True, exist_ok=True)
    out = json.loads(args.out.read_text()) if args.out.exists() else {}

    for i, (repo, shas) in enumerate(sorted(want.items()), 1):
        todo = [s for s in sorted(shas) if not out.get(f"{repo}@{s}", {}).get("resolved")]
        if not todo:
            continue
        url = (pins.get(repo) or {}).get("url")
        dest = work / f"{repo}.git"
        err = None
        if not url:
            err = "no url in pins"
        elif not dest.exists():
            pr = _git("clone", "--bare", "--quiet", "--filter=tree:0", url, str(dest))
            if pr.returncode != 0:
                err = f"clone failed: {pr.stderr.strip()[:200]}"
        for sha in todo:
            key = f"{repo}@{sha}"
            if err:
                out[key] = {"resolved": False, "error": err}
                continue
            pr = _git("log", "-1", "--format=%cI%x00%B", sha, cwd=dest)
            if pr.returncode != 0:
                out[key] = {"resolved": False, "error": f"sha not found: {pr.stderr.strip()[:120]}"}
                continue
            date, _, msg = pr.stdout.partition("\x00")
            model, fam, line = extract_model(msg)
            out[key] = {"resolved": True, "error": None, "commit_date": date.strip(),
                        "model": model, "model_family": fam, "trailer": line}
        print(f"  [{i}/{len(want)}] {repo}: {len(todo)} commits"
              + (f"  ERROR {err}" if err else ""), file=sys.stderr, flush=True)
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True))   # resumable

    res = [v for v in out.values() if v.get("resolved")]
    print(f"resolved {len(res)}/{len(out)} commits; with a model string: "
          f"{sum(1 for v in res if v.get('model'))}")
    for m, n in Counter(v.get("model") for v in res).most_common(15):
        print(f"  {n:5}  {m}")
    print(f"clones in {work} (safe to delete)", file=sys.stderr)


if __name__ == "__main__":
    main()
