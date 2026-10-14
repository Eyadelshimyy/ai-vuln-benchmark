#!/usr/bin/env python3
"""land.py [--watch] <pr-number> [pr-number ...]

Enqueue approved PRs and watch them until they actually land.

``--watch`` polls only: it never enqueues, re-enqueues, or merges anything. Use
it when the PRs are already on their way -- an author with auto-merge enabled
has GitHub enqueue on approval, so there is nothing to enqueue and the only
open question is whether the queue landed them. It is also the form to reach for
when a permission gate refuses the merge command: the watch is read-only, so it
still answers "did it land?" without asking for write access.

Four things make this worth scripting rather than eyeballing:

0. With auto-merge on, APPROVING IS LANDING -- GitHub enqueues the moment the
   review gate clears. `queue.py`'s AUTO column reports this per PR. Calling
   this script to "merge" such a PR is redundant at best; the honest command is
   `--watch`.

1. With a merge queue, `gh pr merge` must be called with NO strategy flag --
   the queue owns the strategy and passing --merge/--squash errors out.
2. "already queued to merge" is SUCCESS, not a failure. It reads like an error.
3. The queue evicts entries when it rebuilds after another PR lands. A PR can
   sit CLEAN + APPROVED and simply not be in the queue any more, silently. The
   fix is to notice and re-enqueue, which is exactly what a human forgets.

Refuses to touch anything that is not APPROVED and CLEAN. That guard is the
point: a review round should never merge something on the strength of a stale
green, and re-checking immediately before enqueueing is the only way to know.
"""
import json
import subprocess
import sys
import time

POLL_SECONDS = 30
MAX_POLLS = 40  # ~20 minutes; queue CI runs are slow


def gh(args, tries=2):
    for attempt in range(tries):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode == 0:
            try:
                return json.loads(r.stdout)
            except json.JSONDecodeError:
                return r.stdout
        if attempt < tries - 1:
            time.sleep(2)
    return None


def state_of(num):
    d = gh(["gh", "pr", "view", str(num), "--json",
            "state,mergeStateStatus,reviewDecision,mergedAt"])
    return d or {}


def enqueue(num):
    """No strategy flag: with a queue the repo decides, and without one gh
    falls back to the repo's only enabled method."""
    r = subprocess.run(["gh", "pr", "merge", str(num)],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if "already queued" in out.lower() or r.returncode == 0:
        return True, out
    return False, out


def main():
    argv = sys.argv[1:]
    watch_only = "--watch" in argv
    nums = [int(a) for a in argv if a != "--watch"]
    if not nums:
        sys.exit("usage: land.py [--watch] <pr-number> [pr-number ...]")

    pending = []
    for n in nums:
        s = state_of(n)
        if s.get("state") == "MERGED":
            print(f"#{n} already merged ({s.get('mergedAt', '')[:16]})")
            continue
        if watch_only:
            print(f"#{n} watching -- {s.get('mergeStateStatus')} / "
                  f"review={s.get('reviewDecision')}")
            pending.append(n)
            continue
        if s.get("reviewDecision") != "APPROVED":
            print(f"#{n} SKIP -- review={s.get('reviewDecision')} (needs APPROVED)")
            continue
        if s.get("mergeStateStatus") != "CLEAN":
            print(f"#{n} SKIP -- {s.get('mergeStateStatus')} "
                  f"(author must resolve; approving does not clear this)")
            continue
        ok, out = enqueue(n)
        print(f"#{n} {'enqueued' if ok else 'ENQUEUE FAILED'}"
              + (f" -- {out}" if out and not ok else ""))
        if ok:
            pending.append(n)

    if not pending:
        print("\nnothing to wait on")
        return

    print(f"\nwatching {len(pending)} PR(s); polling every {POLL_SECONDS}s")
    for poll in range(MAX_POLLS):
        time.sleep(POLL_SECONDS)
        still = []
        for n in pending:
            s = state_of(n)
            if s.get("state") == "MERGED":
                print(f"  [{poll+1}] #{n} MERGED {s.get('mergedAt', '')[:16]}")
                continue
            mss = s.get("mergeStateStatus")
            if mss in ("DIRTY", "CONFLICTING"):
                print(f"  [{poll+1}] #{n} went {mss} -- another PR landed and "
                      f"conflicted it. Author action; dropping from the watch.")
                continue
            # Silent eviction: clean, approved, but no longer queued.
            # --watch never writes, so it reports instead of re-enqueueing.
            if mss == "CLEAN" and watch_only:
                print(f"  [{poll+1}] #{n} is CLEAN and not landed -- if it was "
                      f"evicted from the queue it needs re-enqueueing (not done "
                      f"in --watch mode)")
            elif mss == "CLEAN":
                ok, out = enqueue(n)
                if ok and "already queued" not in out.lower():
                    print(f"  [{poll+1}] #{n} had been evicted from the queue -- re-enqueued")
            still.append(n)
        pending = still
        if not pending:
            print("\nall watched PRs landed")
            return
        print(f"  [{poll+1}] still pending: {', '.join('#%d' % n for n in pending)}")

    print(f"\ntimed out with {len(pending)} still queued: "
          f"{', '.join('#%d' % n for n in pending)}")
    print("They are in the queue and will land on their own -- report them as "
          "queued rather than claiming they merged.")


if __name__ == "__main__":
    main()
