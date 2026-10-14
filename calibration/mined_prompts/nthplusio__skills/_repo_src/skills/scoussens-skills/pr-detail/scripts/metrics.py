#!/usr/bin/env python3
"""Derive the mechanical counts /pr-detail scores against, from gather_pr.sh output.

Usage: gather_pr.sh <owner/repo> <n> | metrics.py
       metrics.py <gathered.json>

Everything here is arithmetic on observed data — no judgment. The point of
computing it in a script rather than by eye is that these numbers end up in a
document about a named person's work, where a miscount is not a rounding error
but an unfair claim. Three of these were mis-derived by hand while this skill
was being built:

  * Commit churn. `gh pr view --json commits` includes commits authored by
    OTHER people when the branch absorbed upstream work. On one PR that was
    8 of 11 commits — attributing all 11 to the author would have tripled the
    churn figure for someone who wrote 3.
  * Finding categories. An AI reviewer's `_(source: ...)_` tag can be
    MULTI-valued ("bugs, style"). Grouping on the raw string
    invents categories and undercounts the real ones.
  * CI history. `statusCheckRollup` only describes the newest commit, so a PR
    that was red for a day reads as clean. Failures are counted per commit.

Emits JSON. Every field is a count or a timestamp you can point at in the
timeline; none of it is a score.
"""

import json
import os
import re
import sys
from datetime import datetime

# Bot detection must not be a name list. This artifact publishes a scored
# judgement about a named person, and a bot counted as a human inflates the
# "a teammate blocked this" signal — the strongest one in the whole report.
#
# The authority is GitHub's own `user.type == "Bot"`, which `gather_pr.sh`
# collects into `botLogins`. Two traps make the naive check wrong:
#   * `gh pr view --json reviews` exposes only `author.login` — no type field.
#   * It also STRIPS the `[bot]` suffix, so matching on `[bot]` never fires on
#     that path. `copilot-pull-request-reviewer` arrives bare.
# The name list below is a last-resort fallback for a login that never appeared
# in the REST payload; anything it cannot classify is reported as UNKNOWN rather
# than silently counted as a human.
BOT_HINT = re.compile(r"claude|\[bot\]|dependabot|renovate|linear-code|copilot|"
                      r"coderabbit|sourcery|greptile|codecov|sonar", re.I)
BOT_LOGINS = set()      # populated from GitHub's user.type == "Bot"
UNCLASSIFIED = set()    # logins we could not decide; the artifact must disclose these
SOURCE_TAG = re.compile(r"_\(source:\s*([^)]+)\)_", re.I)
FAIL = {"FAILURE", "TIMED_OUT", "ERROR", "CANCELLED", "STARTUP_FAILURE"}
_SLA = float(os.environ["PR_DETAIL_SLA_HOURS"]) if os.environ.get("PR_DETAIL_SLA_HOURS") else None
FIXUP = re.compile(r"^(fix(up)?!|squash!|wip\b|fix ci\b|lint\b|typo\b|address (review|comments|feedback))", re.I)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def hours(a, b):
    return round((b - a).total_seconds() / 3600, 1) if a and b else None


def _bare(login):
    """REST reports `acme-reviewer[bot]`; `gh pr view --json reviews` reports the
    same account as `acme-reviewer`. Compare on the bare form or the
    authoritative lookup misses exactly the bots no name list knows about."""
    return (login or "").removesuffix("[bot]")


def is_bot(login):
    """True for a bot, False for a human. A login GitHub did not label and no
    hint matches is recorded in UNCLASSIFIED and treated as human -- the caller
    must surface that list rather than present the count as certain."""
    if not login:
        return False
    bare = _bare(login)
    if bare in BOT_LOGINS or login.endswith("[bot]"):
        return True
    if BOT_HINT.search(login):
        return True
    if bare not in _KNOWN_HUMANS:
        UNCLASSIFIED.add(login)
    return False


_KNOWN_HUMANS = set()   # logins GitHub labelled type == "User"


def main():
    raw = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    d = json.loads(raw)
    pr, threads, timeline, checks = d["pr"], d["threads"], d["timeline"], d["checkRuns"]

    # GitHub's own account types, when gather_pr.sh could fetch them.
    for login, kind in (d.get("accountTypes") or {}).items():
        (BOT_LOGINS if kind == "Bot" else _KNOWN_HUMANS).add(_bare(login))

    author = (pr.get("author") or {}).get("login")
    opened, merged = ts(pr.get("createdAt")), ts(pr.get("mergedAt") or pr.get("closedAt"))

    # --- reviews -------------------------------------------------------
    # Three-way split, not two. The author's own replies land in the reviews
    # array as COMMENTED submissions, and counting them as reviews of the PR
    # is nonsense — on one PR that reported 40+ "human reviews" that were
    # the author answering feedback on their own change.
    reviews = sorted(pr.get("reviews") or [], key=lambda r: r.get("submittedAt") or "")
    bot_reviews = [r for r in reviews if is_bot((r.get("author") or {}).get("login"))]
    author_replies = [r for r in reviews
                      if (r.get("author") or {}).get("login") == author]
    human_reviews = [r for r in reviews
                     if not is_bot((r.get("author") or {}).get("login"))
                     and (r.get("author") or {}).get("login") != author]
    # Each bot submission is one round. Counting timeline `review_dismissed`
    # events instead under-counts: the newest round is never dismissed, and
    # pagination can clip the tail.
    # Repos may run several review bots; one combined "rounds" number
    # hides which. Break out by login.
    rounds_by_bot = {}
    for r in bot_reviews:
        who = (r.get("author") or {}).get("login")
        rounds_by_bot[who] = rounds_by_bot.get(who, 0) + 1
    rounds = len(bot_reviews)

    first_human = next((ts(r["submittedAt"]) for r in human_reviews), None)
    first_any = next((ts(r["submittedAt"]) for r in reviews
                      if (r.get("author") or {}).get("login") != author), None)

    # The review-SLA clock starts when review is ASKED FOR, not when the branch
    # first got a PR. One PR sat as a draft for 21 days; measuring from
    # createdAt reported a 355h "SLA breach" against reviewers who had not yet
    # been asked for anything.
    rfr = next((ts(e["at"]) for e in timeline if e.get("event") == "ready_for_review"), None)
    sla_start = rfr or opened

    # A review can land while the PR is still a draft (a bot commented on
    # one PR six days before it was marked ready), which makes a naive
    # "first review minus ready_for_review" go NEGATIVE. Zero is the honest
    # reading: the feedback was already there when review was requested.
    def response_hours(first):
        if not first:
            return None
        h = hours(sla_start, first)
        return 0.0 if h is not None and h < 0 else h

    # --- threads and their fate ----------------------------------------
    cats, thread_rows = {}, []
    for t in threads:
        nodes = (t.get("comments") or {}).get("nodes") or []
        if not nodes:
            continue
        head = nodes[0]
        body = head.get("body") or ""
        m = SOURCE_TAG.search(body)
        # Multi-valued tags ("bugs, style") must be split, or the
        # category histogram invents combined categories and undercounts real ones.
        tags = [s.strip().lower() for s in m.group(1).split(",")] if m else ["untagged"]
        for tag in tags:
            cats[tag] = cats.get(tag, 0) + 1
        title = re.match(r"\*\*\[(.+?)\]\*\*", body)
        thread_rows.append({
            "title": title.group(1) if title else body[:80],
            "raisedBy": (head.get("author") or {}).get("login"),
            "byBot": is_bot((head.get("author") or {}).get("login")),
            "path": head.get("path"),
            "line": head.get("line"),
            "categories": tags,
            "resolved": t.get("isResolved"),
            "outdated": t.get("isOutdated"),
            "replies": len(nodes) - 1,
            "authorReplied": any(
                (c.get("author") or {}).get("login") == author for c in nodes[1:]
            ),
        })

    # --- commits: the author's own work vs. absorbed upstream work ------
    commits = pr.get("commits") or []
    own, foreign = [], []
    for c in commits:
        who = ((c.get("authors") or [{}])[0].get("login")
               or (c.get("authors") or [{}])[0].get("name"))
        (own if who == author else foreign).append(c)
    after_first_review = [
        c for c in own if first_any and ts(c.get("committedDate")) and ts(c["committedDate"]) > first_any
    ]

    # --- CI over the whole life of the PR, not just at merge ------------
    ci_fail_commits, ci_failed_checks = 0, {}
    for entry in checks:
        bad = [r for r in entry.get("runs") or [] if (r.get("conclusion") or "").upper() in FAIL]
        if bad:
            ci_fail_commits += 1
            for r in bad:
                ci_failed_checks[r["name"]] = ci_failed_checks.get(r["name"], 0) + 1

    ev = lambda name: [e for e in timeline if e.get("event") == name]

    print(json.dumps({
        "pr": {
            "number": pr.get("number"), "title": pr.get("title"), "author": author,
            "state": pr.get("state"), "baseRef": pr.get("baseRefName"),
            "openedAt": pr.get("createdAt"), "mergedAt": pr.get("mergedAt"),
            "mergedBy": (pr.get("mergedBy") or {}).get("login"),
            "openHours": hours(opened, merged),
            "additions": pr.get("additions"), "deletions": pr.get("deletions"),
            "changedFiles": pr.get("changedFiles"),
            "labels": [l["name"] for l in pr.get("labels") or []],
        },
        "review": {
            "botRounds": rounds,
            "botReviewStates": [r.get("state") for r in bot_reviews],
            "humanReviews": [
                {"who": (r.get("author") or {}).get("login"), "state": r.get("state"),
                 "at": r.get("submittedAt")} for r in human_reviews
            ],
            "changesRequestedBy": sorted({
                (r.get("author") or {}).get("login") for r in reviews
                if r.get("state") == "CHANGES_REQUESTED"
            }),
            "authorReplyCount": len(author_replies),
            "openedAsDraft": rfr is not None,
            "readyForReviewAt": rfr.isoformat() if rfr else None,
            "draftHours": hours(opened, rfr),
            "roundsByBot": rounds_by_bot,
            "hoursToFirstReview": response_hours(first_any),
            "hoursToFirstHumanReview": response_hours(first_human),
            "reviewedWhileDraft": bool(rfr and first_any and first_any < rfr),
            # No SLA is assumed. Set PR_DETAIL_SLA_HOURS only where the repo
            # actually publishes one; otherwise this stays null rather than
            # implying a commitment that does not exist.
            "unclassifiedLogins": sorted(UNCLASSIFIED),
            "slaThresholdHours": _SLA,
            "slaBreach": (None if (_SLA is None or not first_any)
                          else (response_hours(first_any) or 0) > _SLA),
            "slaClockFrom": "ready_for_review" if rfr else "createdAt",
            "mergedWithNoReview": len(reviews) == 0,
        },
        "findings": {
            "total": len(thread_rows),
            "byCategory": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
            "resolved": sum(1 for t in thread_rows if t["resolved"]),
            # Labelled by the PR's actual state: "unresolved at merge" is a
            # finding, "unresolved on an open PR" is just work in progress.
            ("unresolvedAtMerge" if pr.get("mergedAt") else "unresolvedStillOpen"):
                sum(1 for t in thread_rows if not t["resolved"]),
            "raisedByHuman": sum(1 for t in thread_rows
                                 if not t["byBot"] and t["raisedBy"] != author),
            "authorEngaged": sum(1 for t in thread_rows if t["authorReplied"]),
            "perKLoc": round(len(thread_rows) / max((pr.get("additions") or 0) / 1000, 0.001), 1),
            "threads": thread_rows,
        },
        "churn": {
            "commitsTotal": len(commits),
            "commitsByAuthor": len(own),
            "commitsByOthers": len(foreign),
            "otherAuthors": sorted({
                ((c.get("authors") or [{}])[0].get("login")
                 or (c.get("authors") or [{}])[0].get("name")) for c in foreign
            }),
            "authorCommitsAfterFirstReview": len(after_first_review),
            "fixupStyleCommits": sum(1 for c in own if FIXUP.match(c.get("messageHeadline") or "")),
            "forcePushes": len(ev("head_ref_force_pushed")),
            "convertedToDraft": len(ev("convert_to_draft")),
        },
        "ci": {
            "commitsWithFailingChecks": ci_fail_commits,
            "commitsChecked": len(checks),
            "failedCheckCounts": dict(sorted(ci_failed_checks.items(), key=lambda kv: -kv[1])),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
