#!/usr/bin/env python3
"""queue.py <base-branch>

The triage pass. Every open non-draft PR against <base>, oldest first, with the
three facts that decide what you do next: who is actually blocking (you or the
author), whether CI is even meaningful yet, and how risky the changed paths are.

Run this once at the start of a round. Reviewing in creation order matters --
the oldest PRs have the most accumulated review history to reconcile, and
merging them first is what stops the queue growing.
"""
import json, subprocess, sys, time

# Two phases on purpose. Asking for `files` + `statusCheckRollup` across a
# whole PR list makes GitHub's GraphQL endpoint 504 once the queue gets big --
# so the list call stays cheap and the heavy per-PR fields are fetched one at a
# time, where a single failure degrades one row instead of the whole run.
LIST_FIELDS = ("number,title,author,createdAt,isDraft,reviewDecision,"
               "mergeable,mergeStateStatus,additions,deletions,changedFiles,"
               "autoMergeRequest")
DETAIL_FIELDS = "files,statusCheckRollup"

# Risk tiers drive review depth. Classify by the *riskiest* path in the diff:
# a PR touching a migration and a stylesheet is a DATA PR, not a WEB one.
#
# These patterns are deliberately generic — they are the vocabulary most stacks
# share, not one repo's layout. Add your repo's own names (a migration prefix,
# an ORM class, a package dir) in `.pr-review-tiers.json` at the repo root:
#
#     {"DATA": ["V001__", "db/specs/"], "API": ["BaseCrud"]}
#
# The point is the OTHER count printed at the end. Tier-driven depth is this
# script's whole purpose, so a queue where most PRs land in OTHER means the
# patterns do not fit this repo and the depth column should not be trusted --
# far better to say so than to let every PR quietly read as low risk.
# Anchor every pattern that could appear inside a longer word: a bare "entity"
# matches "identity" and files an auth change as DATA. Prefer "/entities/" to
# "entity", "/seed/" to "seed".
#
# Order is load-bearing: the first match wins, so the stricter tier must come
# first. DATA before AUTH is deliberate — auth vocabulary ("secret", "token")
# appears incidentally across a large diff, while a migration path does not.
TIERS = [
    ("DATA",  ("migration", "/seed/", "schema.sql", "/store/", "store.py",
               "/mongo/", "/models/", "/entities/", "/repositories/")),
    ("AUTH",  ("auth", "jwt", "principal", "licens", "billing", "entitlement",
               "secret", "middleware", "token", "permission", "session",
               "credential")),
    ("API",   ("routers/", "/services/", "/handlers/", "/controllers/",
               "sql_quer", "/queries/", "query_builder", "/api/", "packages/",
               "/jobs/", "/dags/", "/cron")),
    ("INFRA", ("/terraform/", ".github/workflows/", "Dockerfile",
               "docker-compose", "/infra/", "/deploy", "/helm/", ".tf")),
    ("WEB",   ("/web/", "/frontend/", "/ui/", ".tsx", ".jsx", ".vue",
               ".svelte", ".css", ".scss")),
]


def load_overrides(path=".pr-review-tiers.json"):
    """Repo-supplied patterns, merged onto the generic ones. Absent is normal."""
    try:
        with open(path) as f:
            extra = json.load(f)
    except (OSError, ValueError):
        return TIERS
    merged = []
    for name, pats in TIERS:
        merged.append((name, tuple(pats) + tuple(extra.get(name) or ())))
    return merged


def tier(paths, tiers=None):
    joined = " ".join(paths).lower()
    for name, pats in (tiers or TIERS):
        if any(p.lower() in joined for p in pats):
            return name
    if paths and all(p.endswith((".md", ".json", ".html")) or "docs/" in p for p in paths):
        return "DOCS"
    return "OTHER"


def auto_of(pr):
    """AUTO means the author enabled auto-merge, so your approval is the merge
    action -- GitHub enqueues the moment the review gate clears, with no merge
    command from you. Read this column before approving, not after."""
    return "AUTO" if pr.get("autoMergeRequest") else "-"


def blocker_of(pr, red):
    """Who has to act. This is the most useful column: a conflict or a stale
    branch is the author's problem and approving does not clear it, so those
    must never be answered with 'request changes'."""
    ms, rd = pr["mergeStateStatus"], pr.get("reviewDecision")
    if ms in ("DIRTY", "CONFLICTING"):
        return "AUTHOR: conflicts"
    if ms == "BEHIND":
        return "AUTHOR: behind base"
    if red:
        return "AUTHOR: red CI"
    if rd == "CHANGES_REQUESTED":
        return "REVIEW: changes-req"
    if rd == "APPROVED" and ms == "CLEAN":
        return "-> MERGEABLE NOW"
    return "REVIEW: needs review"


def gh_json(args, tries=3):
    """gh with a retry. The API 504s under load often enough that a single
    transient failure should not abort a review round."""
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


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: queue.py <base-branch>")
    base = sys.argv[1]
    listed = gh_json(["gh", "pr", "list", "--base", base, "--state", "open",
                      "--limit", "200", "--json", LIST_FIELDS])
    if listed is None:
        sys.exit("could not list PRs (gh failed after retries) -- check auth and network")
    prs = [p for p in listed if not p["isDraft"]]
    prs.sort(key=lambda p: p["createdAt"])
    for p in prs:
        detail = gh_json(["gh", "pr", "view", str(p["number"]), "--json", DETAIL_FIELDS])
        p.update(detail or {"files": [], "statusCheckRollup": []})
        p["_degraded"] = detail is None

    tiers = load_overrides()
    print(f"{len(prs)} open non-draft PR(s) targeting {base}, oldest first\n")
    hdr = (f'{"PR":<7}{"OPENED":<12}{"AUTHOR":<18}{"RISK":<7}'
           f'{"SIZE":<13}{"BLOCKER":<22}{"CI":<9}{"AUTO":<6}TITLE')
    print(hdr)
    print("-" * len(hdr))
    for p in prs:
        paths = [f["path"] for f in (p.get("files") or [])]
        checks = p.get("statusCheckRollup") or []
        state = lambda c: c.get("conclusion") or c.get("state")
        red = [c for c in checks if state(c) in ("FAILURE", "ERROR", "CANCELLED", "TIMED_OUT")]
        pend = [c for c in checks if state(c) in ("PENDING", "IN_PROGRESS", "QUEUED", None, "")]
        if p.get("_degraded"):
            ci = "?"          # detail fetch failed; re-check this row by hand
        else:
            ci = f"{len(red)} RED" if red else (f"{len(pend)} pend" if pend else "green")
        size = f'+{p["additions"]}/-{p["deletions"]}'
        print(f'#{p["number"]:<6}{p["createdAt"][:10]:<12}{p["author"]["login"][:17]:<18}'
              f'{tier(paths, tiers):<7}{size:<13}{blocker_of(p, red):<22}{ci:<9}'
              f'{auto_of(p):<6}{p["title"][:46]}')

    others = [p["number"] for p in prs
              if tier([f["path"] for f in (p.get("files") or [])], tiers) == "OTHER"]
    if others and len(others) >= max(2, len(prs) // 3):
        print(f"\n!! {len(others)} of {len(prs)} PR(s) classified OTHER: "
              f"{', '.join('#%d' % n for n in others)}")
        print("   The tier patterns do not fit this repo, so the RISK column is not a")
        print("   reliable depth signal. Add this repo's own path names to")
        print("   .pr-review-tiers.json, or choose depth by reading the diff instead.")

    print("""
RISK -> depth   DATA/AUTH  full independent trace; check migrations and reverts
                API/INFRA  verify open findings, then trace the changed logic
                WEB        verify findings, check key logic in both directions
                DOCS       check the claims actually match the diff

BLOCKER         AUTHOR:*   approve anyway when the CODE is ready -- that clears
                           the review gate -- then comment about the conflict.
                           Requesting changes for a conflict just adds noise.
AUTO            The author enabled auto-merge, so APPROVING IS LANDING: GitHub
                enqueues as soon as the review gate clears and no merge command
                is needed. Check this column before you approve. Confirm the
                landing with `land.py --watch <n>`, which only polls.

CI              'N RED' is not automatically the author's fault: a failing
                review/lint job is sometimes the reviewer tooling erroring out,
                not a finding. Read the job before you attribute it.""")


if __name__ == "__main__":
    main()
