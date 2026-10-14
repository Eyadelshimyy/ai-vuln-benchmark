#!/usr/bin/env python3
"""test_reach.py <pr-number>

List the symbols a PR adds that **no test mentions anywhere**.

A test file in the diff is not coverage. A PR can add a test that exercises no
production code at all -- builds its own fixture, reimplements the logic inside
the test body, and asserts on its own result. It passes with the entire fix
reverted, and it passes CI, so nothing downstream objects. The only cheap way to
catch it is to check reachability in the other direction: take the symbols the
PR introduced and ask whether the test tree names any of them.

Zero hits on a symbol is not automatically a defect -- a private helper covered
through its caller is fine. It is a question worth asking, and the answer takes
seconds:

    "_normalize_rows: 0 test references"

Read the flagged symbol's caller and decide whether the behaviour is genuinely
exercised. What you must not do is infer coverage from a changed test file.
"""
import json
import re
import subprocess
import sys

# Definitions worth tracking across the languages a PR usually touches. Names
# are captured from added lines only, so an untouched definition never appears.
DEF_PATTERNS = (
    r"^\+\s*(?:async\s+)?def\s+([A-Za-z_]\w*)",                    # python
    r"^\+\s*class\s+([A-Za-z_]\w*)",                               # python / ts
    r"^\+\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$]\w*)",  # js / ts
    r"^\+\s*(?:export\s+)?(?:const|let)\s+([A-Za-z_$]\w*)\s*[:=]\s*(?:async\s*)?\(",
    r"^\+\s*(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$]\w*)",
)

TEST_GLOBS = ("*test*", "*spec*", "*__tests__*", "*/tests/*", "*_test.go")

# A one- or two-character name generates noise, and dunders are never the
# subject of a coverage question.
def interesting(name):
    return len(name) > 3 and not (name.startswith("__") and name.endswith("__"))


def run(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def pr_diff(num, base, head, merge_commit=None):
    """Local git first. ``gh pr diff`` is capped by the API at 20,000 lines and
    returns HTTP 406 above it -- which is exactly the large PR most worth
    checking. A local diff has no cap and needs no extra request.

    Two cases, and conflating them yields an empty diff on merged PRs: while a
    PR is open its head is not an ancestor of the base, so
    ``merge-base(base, head)`` is the fork point. Once it merges, the head *is*
    an ancestor and merge-base returns the head itself. For a merged PR the
    change is the merge commit against its first parent instead."""
    subprocess.run(["git", "fetch", "-q", "origin", base], capture_output=True)
    subprocess.run(["git", "fetch", "-q", "origin", f"pull/{num}/head"],
                   capture_output=True)

    mb = run(["git", "merge-base", f"origin/{base}", head]).strip()
    if mb and not mb.startswith(head[:12]):
        d = run(["git", "diff", mb, head])
        if d:
            return d, "local git (fork point)"

    if merge_commit:
        subprocess.run(["git", "fetch", "-q", "origin", merge_commit],
                       capture_output=True)
        # ^1 for a real merge; a squash/rebase merge has a single parent.
        for parent in (f"{merge_commit}^1", f"{merge_commit}^"):
            d = run(["git", "diff", parent, merge_commit])
            if d:
                return d, "local git (merge commit)"
    r = subprocess.run(["gh", "pr", "diff", num], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout:
        return r.stdout, "gh pr diff"
    err = (r.stderr or "").strip().splitlines()
    hint = err[0] if err else "no output"
    if "too_large" in (r.stderr or "") or "406" in (r.stderr or ""):
        hint += ("\n  The API caps diffs at 20,000 lines. Fetch the branch and use "
                 "`git diff $(git merge-base origin/<base> <head>) <head>` instead.")
    sys.exit(f"could not read the diff for #{num}: {hint}")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: test_reach.py <pr-number>")
    num = sys.argv[1]

    pr = run(["gh", "pr", "view", num, "--json",
              "headRefOid,baseRefName,mergeCommit"])
    if not pr:
        sys.exit(f"could not read PR #{num}")
    meta = json.loads(pr)
    head, base = meta["headRefOid"], meta["baseRefName"]
    mc = (meta.get("mergeCommit") or {}).get("oid")

    diff, source = pr_diff(num, base, head, mc)

    # Only added lines outside test files -- a symbol defined in a test is not
    # production code and has nothing to answer for.
    symbols, in_test = {}, False
    current = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            low = current.lower()
            in_test = any(t in low for t in ("test", "spec", "__tests__", "fixture"))
            continue
        if in_test or not line.startswith("+"):
            continue
        for pat in DEF_PATTERNS:
            m = re.match(pat, line)
            if m and interesting(m.group(1)):
                symbols.setdefault(m.group(1), current)

    if not symbols:
        print(f"#{num}: no new named definitions in non-test files (diff via {source}).")
        print("Nothing to check here -- this is a pure edit, deletion, or config change.")
        return

    if subprocess.run(["git", "cat-file", "-e", f"{head}^{{commit}}"],
                      capture_output=True).returncode != 0:
        sys.exit(f"head {head[:9]} is not available locally -- cannot grep the tree")

    unreached, reached = [], []
    for sym, path in sorted(symbols.items()):
        hits = run(["git", "grep", "-l", "-w", sym, head, "--", *TEST_GLOBS])
        n = len([l for l in hits.splitlines() if l])
        (reached if n else unreached).append((sym, path, n))

    print(f"#{num} at {head[:9]}: {len(symbols)} new symbol(s) in non-test files "
          f"(diff via {source})\n")
    if unreached:
        print("NO TEST REFERENCES THESE -- read each one's caller and decide whether")
        print("the behaviour is genuinely exercised. Do not infer coverage from a")
        print("changed test file.\n")
        for sym, path, _ in unreached:
            print(f"  {sym:<44} {path}")
    if reached:
        print(f"\nreferenced by at least one test ({len(reached)}):")
        for sym, _, n in reached:
            print(f"  {sym:<44} {n} test file(s)")
    if not unreached:
        print("\nEvery new symbol is named by some test. That is necessary, not")
        print("sufficient -- a test that names a symbol can still assert nothing")
        print("about it.")


if __name__ == "__main__":
    main()
