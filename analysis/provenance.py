#!/usr/bin/env python3
"""VULCAN AI-authorship provenance verifier.

For the REAL-AI-REPO arm of VULCAN (run_detection.py --use ground_truth), we
test a repository's actual shipped code as-is. That is only honest if the code
we test was genuinely written by an AI -- and "this repo is AI-generated" is a
claim that must be backed by auditable evidence, not a vibe. This module
produces that evidence from git history.

TWO GRANULARITIES
-----------------
1. Repo-level: scan every commit for AI-authorship signals and report what
   fraction of history is AI-attributed, which tools, with example commit SHAs.
2. Function-level (the one that matters): `git blame` the exact line span of a
   mined function and attribute each line to the commit that last wrote it, then
   classify that commit. This ties the unit we actually TEST to the commit that
   actually WROTE it -- the strongest provenance claim available, and the right
   granularity because real "AI-assisted" repos are almost always MIXED
   (human scaffolds, AI writes a module, human edits). We never claim a whole
   repo is AI-written when only part of it is; we attribute the function.

EVIDENCE TIERS (precision over recall -- a false "AI-written" is worse here
than a missed one, because it poisons the benchmark's ground truth)
  strong : an explicit machine trailer or a known AI bot account. Examples:
             Co-authored-by: Claude <noreply@anthropic.com>
             Co-authored-by: Copilot <...>
             "Generated with Claude Code"
             author = devin-ai-integration[bot], copilot-swe-agent[bot], ...
           These are emitted by the tool itself, not typed by a human, so a
           match is near-certain. Only `strong` evidence counts toward the gate.
  weak   : a human-typed hint (author/committer DISPLAY NAME loosely contains
           "aider"/"gpt"/..., or the message mentions an AI by name). Recorded
           for transparency, NEVER counted as proof -- a real person can be
           named "Claude", and "fixed the bug GPT suggested" is not AI
           authorship.

Nothing here infers AI authorship from the *code itself* (style, comments):
that is exactly the kind of unfalsifiable guess the thesis must avoid. Evidence
is provenance metadata only.

USAGE
  # repo-level report
  python3 analysis/provenance.py --repo-dir /path/to/full/clone
  # attribute every mined function in a mined_prompts dir
  python3 analysis/provenance.py --repo-dir /path/to/full/clone \
      --candidates calibration/mined_prompts/<name>/
IMPORTANT: the repo must be a FULL clone (git history intact). A --depth 1
shallow clone has no history to attribute -- this tool detects and refuses it.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# AI-authorship signatures. Each entry: (tool, tier, compiled pattern).
# Patterns are matched case-insensitively against the relevant text.
# ---------------------------------------------------------------------------

# STRONG: machine-emitted trailers in the commit message body. These are
# written by the tool, not the human, so a match is near-certain provenance.
_TRAILER_STRONG = [
    ("claude-code", re.compile(r"co-authored-by:\s*claude\s*<[^>]*anthropic\.com>", re.I)),
    ("claude-code", re.compile(r"generated with \[?claude code\]?", re.I)),
    ("claude-code", re.compile(r"co-authored-by:\s*claude\b", re.I)),
    ("github-copilot", re.compile(r"co-authored-by:\s*copilot\s*<", re.I)),
    ("github-copilot", re.compile(r"co-authored-by:[^\n]*@users\.noreply\.github\.com>[^\n]*copilot", re.I)),
    ("aider", re.compile(r"\baider\b.*\(aider\)", re.I)),
    ("gpt-engineer", re.compile(r"co-authored-by:\s*gpt-engineer", re.I)),
    ("sweep", re.compile(r"co-authored-by:\s*sweep", re.I)),
]

# STRONG: known AI bot/agent accounts appearing as the commit AUTHOR or
# COMMITTER (name or email). A dedicated bot account is unambiguous.
_AUTHOR_STRONG = [
    ("devin", re.compile(r"devin-ai-integration(\[bot\])?", re.I)),
    ("github-copilot", re.compile(r"copilot-swe-agent(\[bot\])?", re.I)),
    ("github-copilot", re.compile(r"\bcopilot\[bot\]", re.I)),
    ("cursor", re.compile(r"cursoragent|cursor\[bot\]", re.I)),
    ("devin", re.compile(r"\bdevin\[bot\]", re.I)),
    ("sweep", re.compile(r"sweep-ai(\[bot\])?", re.I)),
    ("codegen", re.compile(r"codegen-sh(\[bot\])?", re.I)),
    ("claude-code", re.compile(r"claude\s*<noreply@anthropic\.com>", re.I)),
]

# WEAK: human-typed hints. Recorded, never counted toward the gate. A person
# can be named "Claude"; a message can mention GPT without the code being AI's.
_WEAK_MESSAGE = [
    ("mentions-ai", re.compile(r"\b(chatgpt|gpt-4|gpt4|gpt-3\.5|copilot|claude|aider|cursor|llm-generated|ai-generated)\b", re.I)),
]
_WEAK_AUTHOR = [
    ("name-hint", re.compile(r"\b(aider|gpt|copilot|llm|ai-bot)\b", re.I)),
]


def classify_commit(message: str, author: str, committer: str) -> dict | None:
    """Classify one commit. `author`/`committer` are "Name <email>" strings.
    Returns {tool, tier, evidence} for the STRONGEST match, or (for a
    weak-only match) {tool, tier:'weak', evidence}, or None if no signal.
    Strong always wins over weak."""
    # strong trailers in the message
    for tool, pat in _TRAILER_STRONG:
        m = pat.search(message or "")
        if m:
            return {"tool": tool, "tier": "strong", "evidence": m.group(0).strip()[:200]}
    # strong bot author/committer identity
    for who in (author, committer):
        for tool, pat in _AUTHOR_STRONG:
            m = pat.search(who or "")
            if m:
                return {"tool": tool, "tier": "strong",
                        "evidence": f"author/committer={who.strip()[:120]!r}"}
    # weak signals (recorded, not proof)
    for tool, pat in _WEAK_AUTHOR:
        for who in (author, committer):
            m = pat.search(who or "")
            if m:
                return {"tool": tool, "tier": "weak",
                        "evidence": f"author/committer={who.strip()[:120]!r}"}
    for tool, pat in _WEAK_MESSAGE:
        m = pat.search(message or "")
        if m:
            return {"tool": tool, "tier": "weak", "evidence": m.group(0).strip()[:120]}
    return None


def _git(repo_dir: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo_dir), *args],
                          capture_output=True, text=True, check=True).stdout


def is_shallow(repo_dir: Path) -> bool:
    try:
        out = _git(repo_dir, "rev-parse", "--is-shallow-repository").strip()
        return out == "true"
    except subprocess.CalledProcessError:
        return False


def repo_commit_provenance(repo_dir: Path) -> dict:
    """Scan ALL commits. Returns counts, AI fraction (strong only), the set of
    tools detected, and example SHAs -- the auditable repo-level evidence."""
    shallow = is_shallow(repo_dir)
    # One record per commit: SHA, author "N <e>", committer "N <e>", raw body.
    # %x00 field sep, %x01 record sep -- safe against newlines in messages.
    fmt = "%H%x00%an <%ae>%x00%cn <%ce>%x00%B%x01"
    raw = _git(repo_dir, "log", "--no-merges", f"--format={fmt}")
    commits = [c for c in raw.split("\x01") if c.strip()]
    total = 0
    strong = weak = 0
    tools: dict[str, int] = {}
    examples: list[dict] = []
    for c in commits:
        parts = c.lstrip("\n").split("\x00")
        if len(parts) < 4:
            continue
        sha, author, committer, body = parts[0], parts[1], parts[2], parts[3]
        total += 1
        verdict = classify_commit(body, author, committer)
        if not verdict:
            continue
        if verdict["tier"] == "strong":
            strong += 1
            tools[verdict["tool"]] = tools.get(verdict["tool"], 0) + 1
            if len(examples) < 10:
                examples.append({"sha": sha[:12], "tool": verdict["tool"],
                                 "evidence": verdict["evidence"]})
        else:
            weak += 1
    return {
        "repo_dir": str(repo_dir),
        "shallow": shallow,
        "total_commits": total,
        "ai_commits_strong": strong,
        "ai_commits_weak": weak,
        "ai_fraction_strong": round(strong / total, 4) if total else 0.0,
        "tools_detected": tools,
        "examples": examples,
    }


def _func_line_span(file_path: Path, start_line: int) -> tuple[int, int]:
    """Return (start, end) line numbers of the function whose def is at
    start_line, by re-parsing the file. Falls back to a small window if the
    exact node can't be matched."""
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return start_line, start_line + 30
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # match on the def line; AST lineno points at the `def`/decorator-free line
            if node.lineno == start_line or (node.lineno <= start_line <= (node.end_lineno or node.lineno)):
                span = (node.lineno, node.end_lineno or node.lineno)
                if best is None or (span[1] - span[0]) < (best[1] - best[0]):
                    best = span
    return best or (start_line, start_line + 30)


def _blame_commits(repo_dir: Path, rel_file: str, start: int, end: int) -> list[str]:
    """Return the commit SHA responsible for each line in [start, end], via
    `git blame --porcelain`.

    HARDENED: `-w` ignores whitespace-only changes and `-C` follows lines moved
    or copied within the file/commit back to their ORIGINAL commit. This fixes
    the main attribution risk: a human running a formatter/linter (black, etc.)
    over AI-written code would otherwise reassign every reformatted line to the
    human's commit, under-counting AI authorship (and vice-versa). With -w -C,
    a pure reformat no longer steals authorship from the commit that wrote the
    logic. Lines may still map to different commits (genuinely mixed code)."""
    try:
        out = _git(repo_dir, "blame", "--porcelain", "-w", "-C", "-L", f"{start},{end}", "--", rel_file)
    except subprocess.CalledProcessError:
        return []
    shas = []
    for line in out.splitlines():
        # porcelain: a header line "<40-hex sha> <orig> <final> <count>"
        m = re.match(r"^([0-9a-f]{40})\s+\d+\s+\d+", line)
        if m:
            shas.append(m.group(1))
    return shas


def _commit_meta(repo_dir: Path, sha: str) -> tuple[str, str, str]:
    """(author 'N <e>', committer 'N <e>', body) for one commit."""
    out = _git(repo_dir, "show", "-s", "--format=%an <%ae>%x00%cn <%ce>%x00%B", sha)
    parts = out.split("\x00")
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def function_provenance(repo_dir: Path, rel_file: str, start_line: int) -> dict:
    """Attribute ONE mined function to the commit(s) that wrote its lines.
    `ai_attributed` is True iff at least one line is strong-AI-attributed; we
    also report the FRACTION of lines that are, and the dominant AI commit."""
    f = (repo_dir / rel_file)
    start, end = _func_line_span(f, start_line) if f.is_file() else (start_line, start_line + 30)
    shas = _blame_commits(repo_dir, rel_file, start, end)
    if not shas:
        return {"file": rel_file, "function_line": start_line, "span": [start, end],
                "ai_attributed": False, "reason": "no blame (shallow clone / file moved / not tracked)",
                "lines_total": 0, "lines_ai": 0}
    meta_cache: dict[str, dict | None] = {}
    lines_ai = 0
    tool_counts: dict[str, int] = {}
    evidence_example = None
    dominant_sha = None
    for sha in shas:
        if sha not in meta_cache:
            try:
                author, committer, body = _commit_meta(repo_dir, sha)
                meta_cache[sha] = classify_commit(body, author, committer)
            except subprocess.CalledProcessError:
                meta_cache[sha] = None
        v = meta_cache[sha]
        if v and v["tier"] == "strong":
            lines_ai += 1
            tool_counts[v["tool"]] = tool_counts.get(v["tool"], 0) + 1
            if evidence_example is None:
                evidence_example = v["evidence"]
                dominant_sha = sha[:12]
    n = len(shas)
    return {
        "file": rel_file, "function_line": start_line, "span": [start, end],
        "lines_total": n, "lines_ai": lines_ai,
        "ai_line_fraction": round(lines_ai / n, 3) if n else 0.0,
        "ai_attributed": lines_ai > 0,
        "ai_tools": tool_counts,
        "ai_commit": dominant_sha,
        "evidence": evidence_example,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-dir", required=True, type=Path,
                    help="Path to a FULL clone (git history intact; NOT --depth 1)")
    ap.add_argument("--candidates", type=Path, default=None,
                    help="A mined_prompts/<name>/ dir: attribute every mined function in it")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not (args.repo_dir / ".git").exists():
        raise SystemExit(f"{args.repo_dir} is not a git repo (provenance needs git history)")

    report = repo_commit_provenance(args.repo_dir)
    print("=" * 70)
    print(f"REPO PROVENANCE: {args.repo_dir}")
    print("=" * 70)
    if report["shallow"]:
        print("!! SHALLOW CLONE -- history is truncated; attribution is unreliable.")
        print("   Re-clone WITHOUT --depth, or run: git -C <repo> fetch --unshallow")
    print(f"commits: {report['total_commits']} | AI-attributed (strong): "
          f"{report['ai_commits_strong']} ({report['ai_fraction_strong']*100:.1f}%) | "
          f"weak-only hints: {report['ai_commits_weak']}")
    print(f"tools detected: {report['tools_detected'] or '(none)'}")
    for ex in report["examples"]:
        print(f"   {ex['sha']}  {ex['tool']:16} {ex['evidence']}")

    func_reports = []
    if args.candidates:
        files = sorted(p for p in args.candidates.glob("*.json")
                       if p.name not in ("auto_harness_results.json", "benchmark_results.json",
                                         "tier2_results.json", "detection_results.json",
                                         "provenance_results.json"))
        print("\n" + "-" * 70)
        print(f"PER-FUNCTION ATTRIBUTION ({len(files)} mined candidates)")
        print("-" * 70)
        n_ai = 0
        for p in files:
            try:
                rec = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            rel = rec.get("file")
            line = rec.get("line", 0)
            if not rel:
                continue
            fp = function_provenance(args.repo_dir, rel, line)
            fp["candidate"] = p.name
            fp["function"] = rec.get("function")
            func_reports.append(fp)
            n_ai += fp["ai_attributed"]
            flag = "AI" if fp["ai_attributed"] else "--"
            print(f"  [{flag}] {rec.get('function','?'):24} {rel}:{line}  "
                  f"lines_ai={fp['lines_ai']}/{fp['lines_total']} "
                  f"{('('+fp['ai_commit']+' '+str(fp.get('ai_tools')))+')' if fp['ai_attributed'] else ''}")
        print(f"\n{n_ai}/{len(func_reports)} mined functions are AI-attributed (strong evidence)")

    out = {"repo": report, "functions": func_reports}
    dest = args.out or (args.candidates / "provenance_results.json" if args.candidates
                        else args.repo_dir / "provenance_results.json")
    try:
        Path(dest).write_text(json.dumps(out, indent=2))
        print(f"\nAuditable provenance written: {dest}", file=sys.stderr)
    except OSError:
        pass


if __name__ == "__main__":
    main()
