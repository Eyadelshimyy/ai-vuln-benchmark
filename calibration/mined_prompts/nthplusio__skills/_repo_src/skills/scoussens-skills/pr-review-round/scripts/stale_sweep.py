#!/usr/bin/env python3
"""stale_sweep.py <base-branch>

Classify every blocking review in the queue in one pass, oldest PR first.

Two mechanics make this reliable, and both are easy to get wrong by hand:

* **The head SHA comes from the review object.** Every GitHub review carries
  ``commit_id`` -- the commit it was submitted against -- including human
  reviews. Reading the body for a ``reviewed=<sha>`` marker only works for bots
  that print one, and misclassifies every human review as unknown.
* **Base-branch history is excluded from the since-range.** ``--no-merges``
  drops the merge commit itself but keeps every base commit reachable through
  its second parent, so a branch that merely merged the base looks like it
  received dozens of commits. ``--not origin/<base>`` is what separates work the
  author did from history that arrived with a merge.

Author work and base merges mean opposite things. Author commits since a review
may have fixed its findings, so read them first. A branch carrying only base
merges has addressed nothing -- the findings are still live no matter how long
the range looks.
"""
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# Fallback only: some bots post through a token that reports a stale commit_id
# and print the SHA they actually read into the body instead.
SHA_RE = re.compile(r"reviewed=([0-9a-f]{7,40})")


def gh(args, tries=3):
    for attempt in range(tries):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode == 0:
            try:
                return json.loads(r.stdout)
            except json.JSONDecodeError:
                return r.stdout
        if attempt < tries - 1:
            time.sleep(2 * (attempt + 1))
    return None


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def have(sha):
    return git("cat-file", "-e", f"{sha}^{{commit}}") is not None


def count_since(rsha, head, base_ref, num):
    """Return (author_commits, base_commits, error).

    ``author_commits`` excludes base history; ``base_commits`` is what arrived
    through merges. ``error`` is a string when the range cannot be computed --
    never silently zero, because zero is the answer that means "addressed
    nothing" and must be trustworthy.
    """
    if not have(rsha) or not have(head):
        subprocess.run(["git", "fetch", "-q", "origin", f"pull/{num}/head"],
                       capture_output=True)
    if not have(rsha):
        return None, None, f"commit {rsha[:9]} not fetchable (branch rebased?)"
    if not have(head):
        return None, None, f"head {head[:9]} not fetchable"

    mine = git("log", "--oneline", "--no-merges", f"{rsha}..{head}", "--not", base_ref)
    allc = git("log", "--oneline", "--no-merges", f"{rsha}..{head}")
    if mine is None or allc is None:
        return None, None, "git log failed on the range"
    a = [l for l in mine.splitlines() if l]
    t = [l for l in allc.splitlines() if l]
    return a, max(len(t) - len(a), 0), None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: stale_sweep.py <base-branch>")
    base = sys.argv[1]
    repo = (gh(["gh", "repo", "view", "--json", "nameWithOwner",
                "-q", ".nameWithOwner"]) or "").strip()

    # The since-range is meaningless against a stale base ref.
    subprocess.run(["git", "fetch", "-q", "origin", base], capture_output=True)
    base_ref = f"origin/{base}" if have(f"origin/{base}") else None
    if base_ref is None:
        print(f"!! origin/{base} is not available locally -- base-merge commits "
              f"cannot be excluded, so 'author commits' will be overcounted.\n")
        base_ref = "HEAD"  # degrades loudly above, never silently

    listed = gh(["gh", "pr", "list", "--base", base, "--state", "open", "--limit", "200",
                 "--json", "number,isDraft,createdAt,headRefOid,reviewDecision"])
    if listed is None:
        sys.exit("could not list PRs")
    prs = sorted((p for p in listed if not p["isDraft"]), key=lambda p: p["createdAt"])

    print(f"staleness sweep over {len(prs)} open non-draft PR(s) on {base}\n")
    hdr = (f'{"PR":<7}{"BY":<26}{"VERDICT":<12}{"AUTHOR":<8}{"BASE":<7}{"AGE":<7}SINCE THE REVIEW')
    print(hdr)
    print("-" * (len(hdr) + 18))

    now = datetime.now(timezone.utc)
    stale, live, current, broken, clean = [], [], [], [], []

    for p in prs:
        num, head = p["number"], p["headRefOid"]
        reviews = gh(["gh", "api", f"repos/{repo}/pulls/{num}/reviews", "--paginate"])
        if not isinstance(reviews, list):
            print(f'#{num:<6}{"(fetch failed -- re-check by hand)":<60}')
            continue
        blocking = [r for r in reviews if r.get("state") == "CHANGES_REQUESTED"]
        if not blocking:
            clean.append(num)
            continue

        for r in blocking:
            who = (r.get("user") or {}).get("login", "?")[:25]
            m = SHA_RE.search(r.get("body") or "")
            rsha = (m.group(1) if m else None) or r.get("commit_id") or ""
            submitted = r.get("submitted_at") or ""
            try:
                age = (now - datetime.fromisoformat(submitted.replace("Z", "+00:00"))).days
                age_s = f"{age}d"
            except ValueError:
                age_s = "?"

            if not rsha:
                broken.append(num)
                print(f'#{num:<6}{who:<26}{"NO SHA":<12}{"-":<8}{"-":<7}{age_s:<7}'
                      f'review carries no commit_id -- re-read at head')
                continue
            if head.startswith(rsha) or rsha.startswith(head[:9]):
                current.append(num)
                print(f'#{num:<6}{who:<26}{"CURRENT":<12}{"0":<8}{"0":<7}{age_s:<7}'
                      f'reviewed at head -- verify each finding yourself')
                continue

            authored, base_n, err = count_since(rsha, head, base_ref, num)
            if err:
                broken.append(num)
                print(f'#{num:<6}{who:<26}{"UNKNOWN":<12}{"?":<8}{"?":<7}{age_s:<7}{err}')
                continue

            if authored:
                stale.append(num)
                print(f'#{num:<6}{who:<26}{"STALE?":<12}{len(authored):<8}{base_n:<7}'
                      f'{age_s:<7}author pushed work -- read it first, the finding may be fixed')
                for l in authored[:3]:
                    print(f'{"":<66}{l[:64]}')
            else:
                live.append(num)
                print(f'#{num:<6}{who:<26}{"LIVE":<12}{"0":<8}{base_n:<7}{age_s:<7}'
                      f'base merges only -- nothing addressed, findings stand')

    print(f"\n  STALE?  : {len(stale):>3}  -> author commits landed; read them FIRST, the finding may be fixed")
    print(f"  LIVE    : {len(live):>3}  -> only base history merged in; the findings are still live")
    print(f"  CURRENT : {len(current):>3}  -> reviewed at head; verify each finding yourself")
    if broken:
        print(f"  UNKNOWN : {len(broken):>3}  -> range not computable; check these by hand")
    print(f"  no blocker: {len(clean):>3}  -> these need a verdict from you")
    print("\n  AUTHOR = commits the author pushed.  BASE = commits that arrived via a base merge.")
    print("  A stale review is not automatically resolved -- verify the fix at head, then")
    print("  dismiss naming the sha and what you confirmed.")


if __name__ == "__main__":
    main()
