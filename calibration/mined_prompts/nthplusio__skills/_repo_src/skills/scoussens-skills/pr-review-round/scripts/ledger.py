#!/usr/bin/env python3
"""ledger.py -- what one round remembers for the next one.

Read a round's history:

    python3 ledger.py open <base>        # findings still open, per PR
    python3 ledger.py heads <base>       # head SHA each PR carried last round
    python3 ledger.py precision          # how often my findings were overstated
    python3 ledger.py rounds             # round history, newest first

Every field here exists because it changes what the *next* round does. A field
that does not name the step whose behaviour it changes does not belong in the
ledger, and per-author or per-week aggregates deliberately are not recorded.

    head      -> unchanged since last round means the author pushed nothing:
                 skip the diff, escalate on age, answer with one comment.
    status    -> `open` findings are what step 3 verifies first, instead of
                 re-deriving them from the review body. `fixed` drafts the
                 dismissal evidence.
    overstated-> the only field that audits the reviewer. After a few rounds it
                 gives a per-tier precision rate; a tier that runs consistently
                 overstated has the wrong depth in the risk table.
    unblock   -> the smallest change that makes the finding false, carried
                 forward so a blocked PR's next round opens with the path off it.

State lives outside the repository, so git worktrees share one ledger and
nothing is ever committed:

    ~/.local/state/pr-review-round/<owner>__<repo>/{findings,rounds}.jsonl
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

STATUSES = ("open", "fixed", "overstated", "pre-existing", "withdrawn")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_slug():
    r = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner",
                        "-q", ".nameWithOwner"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def state_dir(repo=None):
    repo = repo or repo_slug()
    if not repo:
        raise SystemExit("could not determine the repo (is `gh` authenticated?)")
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    d = os.path.join(base, "pr-review-round", repo.replace("/", "__"))
    os.makedirs(d, exist_ok=True)
    return d


def _path(name, repo=None):
    return os.path.join(state_dir(repo), f"{name}.jsonl")


def _read(name, repo=None):
    p = _path(name, repo)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn line must not take down the round
    return out


def _append(name, rec, repo=None):
    with open(_path(name, repo), "a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------- rounds

def round_start(base, repo=None):
    """Stamp the start of a round and return its id. The id is what findings
    are tagged with, so `--since` can be the round itself rather than a
    guessed clock window."""
    rid = _now()
    _append("rounds", {"round": rid, "base": base, "started_at": rid,
                       "ended_at": None, "prs": []}, repo)
    return rid


def round_end(rid, prs, repo=None):
    rounds = _read("rounds", repo)
    for r in rounds:
        if r.get("round") == rid:
            r["ended_at"] = _now()
            r["prs"] = prs
    with open(_path("rounds", repo), "w") as f:
        for r in rounds:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def last_round(base=None, repo=None):
    """The most recent completed round. Its `ended_at` is the honest lower
    bound for "what merged during this round" -- a fixed clock window catches
    PRs that landed before the round opened."""
    done = [r for r in _read("rounds", repo)
            if r.get("ended_at") and (base is None or r.get("base") == base)]
    return sorted(done, key=lambda r: r["ended_at"])[-1] if done else None


# -------------------------------------------------------------- findings

def record(rid, findings, repo=None):
    """findings: list of dicts with at least {pr, head, summary, status}.
    Optional: file, line, tier, severity, unblock."""
    n = 0
    for f in findings:
        if f.get("status") not in STATUSES:
            raise SystemExit(f"bad status {f.get('status')!r}; use one of {STATUSES}")
        rec = {"round": rid, "recorded_at": _now()}
        rec.update(f)
        _append("findings", rec, repo)
        n += 1
    return n


def open_findings(repo=None):
    """Latest status per (pr, file, line, summary), keeping only `open`.
    This is what step 3 checks at head first."""
    latest = {}
    for f in _read("findings", repo):
        key = (f.get("pr"), f.get("file"), f.get("line"), f.get("summary"))
        latest[key] = f
    out = {}
    for f in latest.values():
        if f.get("status") == "open":
            out.setdefault(f["pr"], []).append(f)
    return out


def heads(repo=None):
    """Head SHA each PR carried the last time it was reviewed."""
    out = {}
    for f in _read("findings", repo):
        if f.get("pr") and f.get("head"):
            out[f["pr"]] = f["head"]
    return out


def precision(repo=None):
    """Per-tier counts of resolved findings. A tier with a high overstated rate
    is being reviewed at the wrong depth -- that is the risk table's problem,
    not the author's."""
    agg = {}
    for f in _read("findings", repo):
        st = f.get("status")
        if st in ("open", "withdrawn"):
            continue
        t = agg.setdefault(f.get("tier") or "?", {s: 0 for s in STATUSES})
        t[st] += 1
    return agg


# ------------------------------------------------------------------ cli

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]

    if cmd == "open":
        o = open_findings()
        if not o:
            print("no open findings carried forward -- this is a first round, or "
                  "every prior finding was resolved")
            return
        print("Open findings from previous rounds. Verify each at the CURRENT head")
        print("before re-stating it; a finding is not live because it is recorded.\n")
        for pr in sorted(o):
            print(f"#{pr}")
            for f in o[pr]:
                loc = f.get("file") or ""
                if loc and f.get("line"):
                    loc += f":{f['line']}"
                print(f"   [{f.get('tier','?')}/{f.get('severity','?')}] {f.get('summary','')}")
                if loc:
                    print(f"      at {loc} (as of {f.get('head','?')[:9]})")
                if f.get("unblock"):
                    print(f"      unblock: {f['unblock']}")
        return

    if cmd == "heads":
        h = heads()
        if not h:
            print("no recorded heads yet")
            return
        print("Head SHA per PR at its last review. Unchanged means the author has")
        print("pushed nothing since -- skip the diff and escalate on age.\n")
        for pr in sorted(h):
            print(f"  #{pr:<7} {h[pr]}")
        return

    if cmd == "precision":
        agg = precision()
        if not agg:
            print("no resolved findings yet -- precision needs a few rounds of "
                  "outcomes before it means anything")
            return
        print(f'{"TIER":<8}{"FIXED":<8}{"OVERSTATED":<12}{"PRE-EXIST":<11}RATE')
        for t in sorted(agg):
            c = agg[t]
            tot = sum(c.values())
            rate = f"{100 * c['overstated'] / tot:.0f}% overstated" if tot else "-"
            print(f'{t:<8}{c["fixed"]:<8}{c["overstated"]:<12}'
                  f'{c["pre-existing"]:<11}{rate}')
        print("\nA tier running consistently overstated is being reviewed at the")
        print("wrong depth. Adjust the risk table, not the authors.")
        return

    if cmd == "rounds":
        rs = sorted(_read("rounds"), key=lambda r: r.get("started_at") or "", reverse=True)
        if not rs:
            print("no rounds recorded yet")
            return
        for r in rs[:20]:
            end = r.get("ended_at") or "(unfinished)"
            print(f'  {r.get("started_at")}  ->  {end:<22} base={r.get("base")} '
                  f'prs={len(r.get("prs") or [])}')
        return

    if cmd == "where":
        print(state_dir())
        return

    sys.exit(f"unknown command {cmd!r}\n\n{__doc__}")


if __name__ == "__main__":
    main()
