#!/usr/bin/env python3
"""findings.py <pr-number> [review-id ...]

Print the full text of blocking reviews plus their inline comments.

Inline comments are where the substance lives -- a review body usually just
says "see inline comments for 3 issues". They live on a different endpoint than
the review itself and have to be joined on `pull_request_review_id`, which is
why this is a script and not a one-liner you retype every time.

With no review-id, prints every review still in CHANGES_REQUESTED state.
"""
import json
import subprocess
import sys
import time


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


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: findings.py <pr-number> [review-id ...]")
    num = sys.argv[1]
    wanted = {int(a) for a in sys.argv[2:]} if len(sys.argv) > 2 else None
    repo = (gh(["gh", "repo", "view", "--json", "nameWithOwner",
                "-q", ".nameWithOwner"]) or "").strip()

    reviews = gh(["gh", "api", f"repos/{repo}/pulls/{num}/reviews", "--paginate"])
    if not isinstance(reviews, list):
        # A failed fetch and a PR with no reviews produce the same empty list.
        # Saying "no blocking reviews" on a fetch failure invents a clean bill
        # of health, so refuse instead.
        sys.exit(f"could not read reviews for #{num} -- re-check by hand rather "
                 f"than treating this as 'no blocking reviews'")
    if isinstance(reviews, str):
        reviews = []
    if wanted is None:
        targets = [r for r in reviews if r.get("state") == "CHANGES_REQUESTED"]
    else:
        targets = [r for r in reviews if r.get("id") in wanted]
    if not targets:
        print("no blocking reviews on this PR")
        return

    comments = gh(["gh", "api", f"repos/{repo}/pulls/{num}/comments?per_page=100",
                   "--paginate"]) or []
    if isinstance(comments, str):
        comments = []

    for r in targets:
        who = (r.get("user") or {}).get("login", "?")
        print("=" * 78)
        print(f'REVIEW {r["id"]}  by {who}  {r.get("state")}  '
              f'{(r.get("submitted_at") or "")[:16]}')
        print("=" * 78)
        print((r.get("body") or "").strip() or "(no body)")

        mine = [c for c in comments if c.get("pull_request_review_id") == r["id"]]
        print(f"\n--- {len(mine)} inline comment(s) on this review ---")
        for c in mine:
            line = c.get("line") or c.get("original_line") or "?"
            resolved = " [reply-thread]" if c.get("in_reply_to_id") else ""
            print(f'\n### {c["path"]}:{line}{resolved}')
            print((c.get("body") or "").strip())
        print()


if __name__ == "__main__":
    main()
