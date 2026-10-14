#!/usr/bin/env python3
"""round_report.py <base-branch> [--since ISO8601] [--me LOGIN]

The report pass. Emits one JSON record per PR touched this round, carrying the
three things a report needs and a review round reliably forgets:

  * ``url``          -- the PR link, so every row in the artifact is clickable
  * ``reviewed_at``  -- when the review was actually submitted, per PR
  * ``reviews`` /
    ``comments``     -- the full body text you posted, so the artifact can
                        expand into what was said instead of paraphrasing it

Why a script rather than recalling it: by report time the merged PRs have
dropped out of ``queue.py``'s open list, and the bodies you posted are hundreds
of lines back in the transcript. Both are exactly the things that get quietly
approximated. This reads them back from the API.

``--since`` bounds "this round". It decides which *merged* PRs are included --
open PRs against <base> are always listed -- and which reviews count as this
round's rather than a previous one's.

The default is the **previous round's end time**, read from the ledger. A fixed
clock window is wrong in both directions: too wide and it reports PRs that
landed before the round opened (and then warns you never reviewed them), too
narrow and it drops a long round's earliest merges. With no ledger history yet
it falls back to 12h and says so on stderr.

Usage:
    python3 round_report.py <base> > /tmp/round.json
    python3 round_report.py <base> --since 2026-08-21T15:00:00Z

Piping the result into the artifact build is the point; read it with jq to spot
check, e.g. `jq -r '.[] | "\\(.number) \\(.verdict) \\(.reviewed_at)"' /tmp/round.json`
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

LIST_FIELDS = ("number,title,url,author,createdAt,isDraft,reviewDecision,"
               "mergeable,mergeStateStatus,additions,deletions,state,mergedAt")


def gh(args, tries=3):
    """gh with a retry -- the API 504s under load often enough that one
    transient failure should not lose the report."""
    for attempt in range(tries):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode == 0:
            try:
                return json.loads(r.stdout)
            except json.JSONDecodeError:
                pass
        if attempt < tries - 1:
            time.sleep(2 * (attempt + 1))
    return None


def repo_slug():
    r = gh(["gh", "repo", "view", "--json", "nameWithOwner"])
    return r["nameWithOwner"] if r else None


def me():
    r = gh(["gh", "api", "user", "--jq", "{login: .login}"])
    return r["login"] if r else None


def verdict_of(pr):
    """The line the report groups by. Deliberately distinguishes MERGED from
    everything else and never calls a queued PR merged -- reporting a PR that
    is sitting in a merge queue as landed is the one inaccuracy that bites."""
    if pr.get("state") == "MERGED":
        return "MERGED"
    ms = pr.get("mergeStateStatus")
    rd = pr.get("reviewDecision")
    if ms in ("DIRTY", "CONFLICTING"):
        return "APPROVED_CONFLICTING" if rd == "APPROVED" else "CHANGES_REQUESTED_CONFLICTING"
    if rd == "CHANGES_REQUESTED":
        return "CHANGES_REQUESTED"
    if rd == "APPROVED":
        return "APPROVED_QUEUED" if ms == "CLEAN" else "APPROVED"
    return "NEEDS_REVIEW"


def main():
    argv = sys.argv[1:]
    if not argv:
        sys.exit("usage: round_report.py <base-branch> [--since ISO8601] [--me LOGIN]")
    base = argv[0]
    since = None
    who = None
    for i, a in enumerate(argv):
        if a == "--since" and i + 1 < len(argv):
            since = argv[i + 1]
        if a == "--me" and i + 1 < len(argv):
            who = argv[i + 1]
    if since is None:
        prev = None
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import ledger
            prev = ledger.last_round(base)
        except Exception:
            prev = None
        if prev and prev.get("ended_at"):
            since = prev["ended_at"]
            print(f"[since] previous round ended {since} -- using that as the "
                  f"round boundary", file=sys.stderr)
        else:
            since = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"[since] no completed round in the ledger; falling back to a "
                  f"12h window ({since}). Merged PRs listed below may predate "
                  f"this round.", file=sys.stderr)
    who = who or me()
    slug = repo_slug()
    if not slug:
        sys.exit("could not resolve repo (gh failed) -- check auth and network")

    openq = gh(["gh", "pr", "list", "--base", base, "--state", "open",
                "--limit", "200", "--json", LIST_FIELDS]) or []
    # Merged PRs have already dropped out of the open list by report time. They
    # are the rows a report most wants and most often loses.
    #
    # `--search "merged:>=..."` rather than fetching a page and filtering here:
    # `gh pr list --state merged --limit N` pages by CREATION date, so a
    # long-lived PR opened months ago and merged in this round falls outside the
    # newest-N window and silently vanishes from the report. That happened on a
    # real round -- a PR open since July merged first and was the one row
    # missing. Let the server filter on merge date instead.
    mergedq = gh(["gh", "pr", "list", "--base", base, "--state", "merged",
                  "--limit", "100", "--search", f"merged:>={since}",
                  "--json", LIST_FIELDS]) or []

    prs = [p for p in openq if not p["isDraft"]]
    prs += mergedq
    prs.sort(key=lambda p: p["createdAt"])

    out = []
    for p in prs:
        n = p["number"]
        # Review bodies carry the reasoning; the inline comments carry the
        # specifics. Both belong in an expandable section, so fetch both.
        reviews = gh(["gh", "api", f"repos/{slug}/pulls/{n}/reviews",
                      "--paginate"]) or []
        issue_comments = gh(["gh", "api", f"repos/{slug}/issues/{n}/comments",
                             "--paginate"]) or []

        mine = [r for r in reviews
                if r.get("user", {}).get("login") == who
                and (r.get("submitted_at") or "") >= since
                and r.get("state") != "PENDING"]
        # Latest first: the report wants this round's verdict, not the history.
        mine.sort(key=lambda r: r.get("submitted_at") or "", reverse=True)

        rec = {
            "number": n,
            "url": p["url"],
            "title": p["title"],
            "author": (p.get("author") or {}).get("login"),
            "created_at": p["createdAt"],
            "additions": p.get("additions"),
            "deletions": p.get("deletions"),
            "state": p.get("state"),
            "merged_at": p.get("mergedAt"),
            "mergeable": p.get("mergeable"),
            "merge_state": p.get("mergeStateStatus"),
            "review_decision": p.get("reviewDecision"),
            "verdict": verdict_of(p),
            # The per-PR review timestamp the report asks for. None means you
            # left a comment but no formal review -- say so rather than
            # implying a review happened.
            "reviewed_at": mine[0]["submitted_at"] if mine else None,
            "reviews": [
                {
                    "id": r["id"],
                    "state": r["state"],
                    "submitted_at": r["submitted_at"],
                    "url": r.get("html_url"),
                    "body": r.get("body") or "",
                }
                for r in mine
            ],
            "comments": [
                {
                    "created_at": c["created_at"],
                    "url": c.get("html_url"),
                    "body": c.get("body") or "",
                }
                for c in issue_comments
                if c.get("user", {}).get("login") == who
                and (c.get("created_at") or "") >= since
            ],
        }
        out.append(rec)

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    counts = {}
    for r in out:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"\n{len(out)} PR(s) in this round ({summary})", file=sys.stderr)
    unreviewed = [r["number"] for r in out if not r["reviews"] and not r["comments"]]
    if unreviewed:
        # A PR reviewed in an earlier round is a different situation from one
        # never looked at. Saying so keeps the warning worth reading -- a check
        # that fires on every round trains you to skip it.
        seen = {}
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import ledger
            seen = ledger.heads()
        except Exception:
            seen = {}
        fresh = [n for n in unreviewed if n not in seen]
        prior = [n for n in unreviewed if n in seen]
        if fresh:
            print(f"NO REVIEW FROM {who} IN ANY ROUND: "
                  + ", ".join(f"#{n}" for n in fresh)
                  + "\n  -> reconcile before reporting; a dropped PR is worse than an "
                    "unreviewed one, because the summary asserts the round is complete.",
                  file=sys.stderr)
        if prior:
            print("REVIEWED IN AN EARLIER ROUND, not this one: "
                  + ", ".join(f"#{n} (last head {seen[n][:9]})" for n in prior)
                  + "\n  -> run `ledger.py open` for the findings still recorded "
                    "against these, and `prstate.py <n>` to see whether the head moved.",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
