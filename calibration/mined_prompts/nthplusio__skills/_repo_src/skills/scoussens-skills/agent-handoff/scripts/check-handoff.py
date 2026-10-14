#!/usr/bin/env python3
"""Check the action table in a handoff document.

A dependency-mapped action table is a directed graph written by hand, and by
hand is where graphs acquire defects that reading will not catch: a cycle
deadlocks the receiver, a dependency naming a row that does not exist never
unblocks, a blank completion criterion is a row that never ends.

Fails on duplicate IDs, dangling dependencies, cycles, and blank actions or
completion criteria. Then prints back what the table actually says -- the rows
ready now, the rows waiting on somebody outside the room, the longest chain
through the dependencies, and ready rows touching the same file, which
therefore cannot run at once. Read that as a description of the plan you wrote;
it disagrees with the intended plan more often than you would expect.

Recognises any Markdown table carrying an `ID` column and a `Blocked by`
column, so it can be pointed at the whole handoff rather than an extract.

Usage: python3 check-handoff.py <handoff-file>
"""

import re
import sys
from collections import defaultdict

# Cells meaning "nothing here". An em dash is the convention the skill uses for
# an unblocked row; the rest are what people write instead.
EMPTY = {"", "-", "--", "—", "–", "n/a", "na", "none", "tbd", "‑"}

ID_RE = re.compile(r"^[A-Za-z]+[0-9]+$")
EXT_RE = re.compile(r"^ext\b[:\s]*", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"^:?-{2,}:?$")

DEP_HEADERS = ["blockedby", "blocked", "dependson", "depends", "waitson"]

errors = []
warnings = []


def norm(text):
    """Header names compare on letters alone, so `Blocked by` == `blocked-by`."""
    return re.sub(r"[^a-z]", "", text.lower())


def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def parse_tables(text):
    """Every Markdown pipe table in the file, as (headers, body_rows)."""
    blocks, block = [], []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            block.append(line)
        elif block:
            blocks.append(block)
            block = []
    if block:
        blocks.append(block)

    tables = []
    for b in blocks:
        if len(b) < 2:
            continue
        rows = [split_row(line) for line in b]
        if not rows[1] or not all(SEPARATOR_RE.match(c) for c in rows[1]):
            continue
        tables.append((rows[0], rows[2:]))
    return tables


def index_of(keys, candidates, skip=()):
    """First unclaimed column whose normalised header is, or starts with, a
    candidate. `skip` holds indices another role has already taken."""
    for i, k in enumerate(keys):
        if i in skip:
            continue
        if any(k == c or k.startswith(c) for c in candidates):
            return i
    return None


def cell(row, i):
    return row[i].strip() if i is not None and i < len(row) else ""


def is_empty(text):
    return text.strip().strip("`").strip().lower() in EMPTY


def split_list(text):
    if is_empty(text):
        return []
    parts = re.split(r"[,;]|\band\b", text)
    return [p.strip().strip("`").strip() for p in parts if not is_empty(p)]


def parse_blockers(text):
    """Split a `Blocked by` cell into internal IDs, external owners, and junk."""
    internal, external, unknown = [], [], []
    for tok in split_list(text):
        if EXT_RE.match(tok):
            external.append(EXT_RE.sub("", tok).strip() or "unnamed")
        elif ID_RE.match(tok):
            internal.append(tok)
        else:
            unknown.append(tok)
    return internal, external, unknown


def find_cycle(deps):
    """A dependency cycle as a list of IDs, or None. Ignores dangling edges,
    which are reported separately and would otherwise mask a real loop."""
    WHITE, GREY = 0, 1
    colour = defaultdict(int)
    path = []

    def visit(node):
        colour[node] = GREY
        path.append(node)
        for nxt in deps.get(node, []):
            if nxt not in deps:
                continue
            if colour[nxt] == GREY:
                return path[path.index(nxt):] + [nxt]
            if colour[nxt] == WHITE:
                found = visit(nxt)
                if found:
                    return found
        path.pop()
        colour[node] = 2
        return None

    for node in deps:
        if colour[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return None


def longest_chain(deps):
    """The longest dependency chain, ending at the row that unblocks last."""
    memo = {}

    def chain(node):
        if node in memo:
            return memo[node]
        memo[node] = [node]  # provisional, guards against re-entry
        best = []
        for nxt in deps.get(node, []):
            if nxt not in deps:
                continue
            candidate = chain(nxt)
            if len(candidate) > len(best):
                best = candidate
        memo[node] = best + [node]
        return memo[node]

    longest = []
    for node in deps:
        candidate = chain(node)
        if len(candidate) > len(longest):
            longest = candidate
    return longest


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2

    path = sys.argv[1]
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"✗ cannot read {path}: {exc}", file=sys.stderr)
        return 2

    # The dependency column is what makes a table an action table; every other
    # column is recognised by name with a sensible fallback, so a table headed
    # `Step | Depends on | …` is still found.
    table = None
    for headers, rows in parse_tables(text):
        keys = [norm(h) for h in headers]
        if index_of(keys, DEP_HEADERS) is not None:
            table = (keys, rows)
            break

    if table is None:
        print(
            f"✗ {path}: no action table found "
            "(needs a Markdown table with a `Blocked by` column)",
            file=sys.stderr,
        )
        return 1

    keys, rows = table
    i_dep = index_of(keys, DEP_HEADERS)
    # ID defaults to the first column, which is where it sits in every table
    # that does not label it. Claimed columns are then excluded, so a table
    # headed `Step | …` does not read its IDs as its actions too.
    i_id = index_of(keys, ["id", "ref", "key", "num", "no"])
    if i_id is None:
        i_id = 0
    i_action = index_of(keys, ["action", "task", "work", "step"], skip={i_id, i_dep})
    i_done = index_of(keys, ["donewhen", "done", "completion", "criterion"], skip={i_id, i_dep})
    i_touch = index_of(keys, ["touches", "files", "where", "surface"], skip={i_id, i_dep})

    for label, idx in (("Action", i_action), ("Done when", i_done), ("Touches", i_touch)):
        if idx is None:
            warnings.append(f"the table has no `{label}` column")

    order, deps, external, actions, touches = [], {}, {}, {}, {}

    for n, row in enumerate(rows, start=1):
        rid = cell(row, i_id).strip("`")
        if not rid:
            errors.append(f"row {n} has no ID")
            continue
        if rid in deps:
            errors.append(f"duplicate ID `{rid}`")
            continue

        internal, ext, unknown = parse_blockers(cell(row, i_dep))
        for junk in unknown:
            warnings.append(
                f"`{rid}` is blocked by \"{junk}\", which is neither a row ID "
                "nor an `EXT <owner>` external blocker"
            )

        order.append(rid)
        deps[rid] = internal
        external[rid] = ext
        actions[rid] = cell(row, i_action)
        touches[rid] = split_list(cell(row, i_touch))

        if i_action is not None and is_empty(actions[rid]):
            errors.append(f"`{rid}` has no action")
        if i_done is not None and is_empty(cell(row, i_done)):
            errors.append(f"`{rid}` has no completion criterion; it can never be finished")

    for rid in order:
        for dep in deps[rid]:
            if dep not in deps:
                errors.append(f"`{rid}` is blocked by `{dep}`, which is not a row in the table")
            elif dep == rid:
                errors.append(f"`{rid}` is blocked by itself")

    cycle = find_cycle(deps)
    if cycle:
        errors.append("dependency cycle, so nothing in it can ever start: " + " → ".join(cycle))

    for msg in warnings:
        print(f"! {msg}", file=sys.stderr)
    for msg in errors:
        print(f"✗ {msg}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} error(s) in the action table of {path}.", file=sys.stderr)
        return 1

    ready = [r for r in order if not deps[r] and not external[r]]
    blocked_ext = [r for r in order if external[r]]

    print(f"✓ {len(order)} action(s), no cycles, every dependency resolves\n")

    print(f"Ready now ({len(ready)}):")
    for rid in ready:
        print(f"  {rid}  {actions[rid]}")
    if not ready:
        print("  none — every row waits on something, so the receiver cannot start")

    if blocked_ext:
        print(f"\nWaiting on someone outside the room ({len(blocked_ext)}):")
        for rid in blocked_ext:
            print(f"  {rid}  {', '.join(external[rid])}  —  {actions[rid]}")

    chain = longest_chain(deps)
    if len(chain) > 1:
        print(f"\nLongest chain ({len(chain)} deep): " + " → ".join(chain))

    conflicts = []
    for i, a in enumerate(ready):
        for b in ready[i + 1:]:
            shared = sorted(set(touches[a]) & set(touches[b]))
            if shared:
                conflicts.append((a, b, shared))
    if conflicts:
        print("\nReady rows touching the same files, so not parallel work:")
        for a, b, shared in conflicts:
            print(f"  {a} and {b} both touch {', '.join(shared)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
