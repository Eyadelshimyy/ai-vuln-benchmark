#!/usr/bin/env python3
"""prstate.py <pr-number>

Everything worth knowing before you read a line of the diff -- and in
particular the one check that saves the most wasted effort: is the blocking
review still about the code that is actually there?

Automated reviewers stamp the commit they reviewed into their body as
its ``commit_id``. Authors very often push a fix minutes after a review lands,
so a blocking review is often about code that no longer exists.
Skip this check and you spend your time re-litigating findings that are already
fixed, and you ask the author for work they have already done -- the fastest way
to make a review worthless to the person receiving it.
"""
import json
import re
import subprocess
import sys
import time

SHA_RE = re.compile(r"reviewed=([0-9a-f]{7,40})")


def gh(args, tries=3):
    """gh, retried. The API 504s under load; one blip should not abort a round."""
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
    return r.stdout.strip() if r.returncode == 0 else ""


def have(sha):
    """True when the commit is present locally. git() returns "" on failure, so
    callers must not test it with `is None`."""
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                       capture_output=True)
    return r.returncode == 0


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: prstate.py <pr-number>")
    num = sys.argv[1]
    repo = gh(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
    repo = (repo or "").strip()

    pr = gh(["gh", "pr", "view", num, "--json",
             "headRefOid,headRefName,baseRefName,author,isDraft,mergeable,"
             "mergeStateStatus,reviewDecision,additions,deletions,title,url"])
    if not pr:
        sys.exit(f"could not read PR #{num}")
    head = pr["headRefOid"]

    print(f'#{num}  {pr["title"]}')
    print(f'  author {pr["author"]["login"]}   {pr["baseRefName"]} <- {pr["headRefName"]}')
    print(f'  +{pr["additions"]}/-{pr["deletions"]}   {pr["mergeable"]} / '
          f'{pr["mergeStateStatus"]}   review={pr["reviewDecision"]}')
    print(f"  head {head}")
    print(f'  {pr["url"]}')

    reviews = gh(["gh", "api", f"repos/{repo}/pulls/{num}/reviews", "--paginate"])
    if not isinstance(reviews, list):
        # A failed fetch and a PR with no reviews are the same empty list. The
        # message below would then read as "nothing is blocking this" -- a clean
        # bill of health manufactured from an error. Refuse instead.
        sys.exit(f"could not read reviews for #{num} -- re-check by hand rather "
                 f"than treating this as 'no blocking reviews'")

    # Only these states gate a merge. COMMENTED/DISMISSED reviews are history.
    active = [r for r in reviews if r.get("state") in ("CHANGES_REQUESTED", "APPROVED")]

    print(f"\n  review history ({len(reviews)} total, {len(active)} still in force):")
    for r in reviews:
        state = r.get("state", "?")
        if state == "COMMENTED":
            continue
        mark = "*" if state in ("CHANGES_REQUESTED", "APPROVED") else " "
        who = (r.get("user") or {}).get("login", "?")
        print(f'   {mark} {r["id"]:<12} {who:<28} {state:<18} {(r.get("submitted_at") or "")[:16]}')
    print("     (* = still in force; anything else is superseded history)")

    blocking = [r for r in active if r["state"] == "CHANGES_REQUESTED"]
    if not blocking:
        print("\n  no blocking reviews -- this PR needs a verdict from you.")
        return

    base = pr["baseRefName"]
    subprocess.run(["git", "fetch", "-q", "origin", base], capture_output=True)
    base_ref = f"origin/{base}"
    if not have(base_ref):
        print(f"\n  !! origin/{base} unavailable -- base-merge commits cannot be")
        print(f"     excluded, so the author-commit count below is overcounted.")
        base_ref = None

    print(f"\n  {len(blocking)} BLOCKING review(s) -- staleness check:")
    for r in blocking:
        who = (r.get("user") or {}).get("login", "?")
        # commit_id is the SHA the review was submitted against, and every
        # review carries it -- human reviews included. The body marker is a
        # fallback for bots that report a different SHA than they read.
        m = SHA_RE.search(r.get("body") or "")
        rsha = (m.group(1) if m else None) or r.get("commit_id") or ""
        submitted = (r.get("submitted_at") or "")[:10]
        print(f'\n   review {r["id"]} by {who}  (submitted {submitted})')
        if not rsha:
            print("     no commit_id on the review -- re-read it against head yourself.")
            continue
        if head.startswith(rsha) or rsha.startswith(head[:9]):
            print(f"     reviewed {rsha[:12]} == head -> CURRENT. Verify each finding at head,")
            print("     then decide. Do not assume the findings are right; confirm them.")
            continue
        subprocess.run(["git", "fetch", "-q", "origin", f"pull/{num}/head"],
                       capture_output=True)
        if not have(rsha):
            print(f"     reviewed {rsha[:12]}, which is not fetchable -- the branch was")
            print("     probably rebased. Compare the review against head by hand.")
            continue
        allc = git("log", "--oneline", "--no-merges", f"{rsha}..{head}") or ""
        if base_ref:
            mine = git("log", "--oneline", "--no-merges", f"{rsha}..{head}",
                       "--not", base_ref) or ""
        else:
            mine = allc
        authored = [l for l in mine.splitlines() if l]
        base_n = max(len([l for l in allc.splitlines() if l]) - len(authored), 0)

        print(f"     reviewed {rsha[:12]} != head {head[:12]}")
        print(f"     {len(authored)} author commit(s), {base_n} arrived via a base merge")
        if authored:
            print("     -> POSSIBLY STALE. Read the author's commits first:")
            for line in authored[:12]:
                print(f"       {line}")
            print("     If they fix the findings, verify that at head and dismiss the review")
            print("     naming the sha and what you confirmed -- otherwise the author cannot")
            print("     tell you actually looked.")
        else:
            print("     -> LIVE. The author has addressed nothing; only base history merged")
            print("     in. The findings stand however long the raw range looks.")


if __name__ == "__main__":
    main()
