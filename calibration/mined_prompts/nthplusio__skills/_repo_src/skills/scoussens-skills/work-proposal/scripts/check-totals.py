#!/usr/bin/env python3
"""Check that the cost tables in a proposal actually add up.

A column that does not sum destroys the credibility of every other number on
the page, and hand-editing ranges is exactly how it happens. This reads HTML
and Markdown tables, sums each numeric column, compares the sum to any total
row, and — given a rate — checks money equals hours times rate on every row.

Real proposals carry discount and credit rows, footers that hold two unrelated
lines, and cells spanning several columns. Rather than guess, this reports a
table it cannot read unambiguously as NEEDS EYE and shows the numbers, and
fails only where a genuine mismatch is unambiguous.

Usage:
    check-totals.py proposal.html
    check-totals.py proposal.html --rate 100
    check-totals.py proposal.md --rate 100 --strict

Exit status is 1 if any check fails, so it can gate a send. --strict also
fails on anything reported as NEEDS EYE.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

DASHES = "-‐‑‒–—"
MINUS = "−–—-"
NUM = r"\$?\s*([\d,]+(?:\.\d+)?)"
RANGE_RE = re.compile(rf"{NUM}\s*(?:&ndash;|&mdash;|[{DASHES}])\s*{NUM}")
SINGLE_RE = re.compile(rf"^\s*[{MINUS}]?\s*{NUM}\s*$")
CELL_RE = re.compile(r"<t([hd])\b([^>]*)>(.*?)</t\1>", re.S | re.I)
COLSPAN_RE = re.compile(r"colspan\s*=\s*[\"']?(\d+)", re.I)


def clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    for a, b in (("&ndash;", "–"), ("&mdash;", "—"), ("&minus;", "−"),
                 ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def parse_cell(cell: str) -> tuple[float, float] | None:
    """Return (low, high) for a numeric cell, or None if it isn't numeric."""
    text = clean(cell)
    if not text:
        return None
    negative = bool(re.match(rf"^[{MINUS}]\s*\$?\s*[\d]", text))
    m = RANGE_RE.search(text)
    if m:
        lo, hi = (float(m.group(i).replace(",", "")) for i in (1, 2))
    else:
        m = SINGLE_RE.match(text)
        if not m:
            return None
        lo = hi = float(m.group(1).replace(",", ""))
    return (-lo, -hi) if negative else (lo, hi)


def expand(row_html: str) -> list[str]:
    """Cells of one HTML row, with colspan expanded so columns line up."""
    cells: list[str] = []
    for _, attrs, inner in CELL_RE.findall(row_html):
        span = COLSPAN_RE.search(attrs)
        cells.append(inner)
        cells.extend([""] * (int(span.group(1)) - 1 if span else 0))
    return cells


@dataclass
class Table:
    label: str
    headers: list[str] = field(default_factory=list)
    body: list[list[str]] = field(default_factory=list)
    footer: list[list[str]] = field(default_factory=list)
    has_negative: bool = False


def html_tables(src: str) -> list[Table]:
    out = []
    for i, tm in enumerate(re.finditer(r"<table\b.*?</table>", src, re.S | re.I), 1):
        block = tm.group(0)
        cap = re.search(r"<caption\b[^>]*>(.*?)</caption>", block, re.S | re.I)
        label = clean(cap.group(1)) if cap else ""
        label = (label.split(".")[0] or f"table {i}")[:52]
        t = Table(label=label)

        head = re.search(r"<thead\b.*?</thead>", block, re.S | re.I)
        rest = block
        if head:
            rows = re.findall(r"<tr\b.*?</tr>", head.group(0), re.S | re.I)
            if rows:
                t.headers = [clean(c) for c in expand(rows[-1])]
            rest = rest.replace(head.group(0), "")

        foot = re.search(r"<tfoot\b.*?</tfoot>", block, re.S | re.I)
        if foot:
            rest = rest.replace(foot.group(0), "")
            t.footer = [expand(r) for r in re.findall(r"<tr\b.*?</tr>", foot.group(0), re.S | re.I)]

        for row in re.findall(r"<tr\b.*?</tr>", rest, re.S | re.I):
            cells = expand(row)
            if any(clean(c) for c in cells):
                t.body.append(cells)
        if t.body:
            out.append(t)
    return out


def markdown_tables(src: str) -> list[Table]:
    out, cur, n = [], None, 0
    total_row = re.compile(r"total|subtotal|\bsum\b", re.I)
    for line in src.splitlines():
        if line.strip().startswith("|") and line.count("|") >= 3:
            cells = line.strip().strip("|").split("|")
            if all(re.fullmatch(r"\s*:?-{2,}:?\s*", c) for c in cells):
                continue
            if cur is None:
                n += 1
                cur = Table(label=f"table {n}", headers=[clean(c) for c in cells])
                continue
            (cur.footer if total_row.search(clean(cells[0])) else cur.body).append(cells)
        else:
            if cur and cur.body:
                out.append(cur)
            cur = None
    if cur and cur.body:
        out.append(cur)
    return out


def fmt(v: tuple[float, float], money: bool) -> str:
    f = (lambda x: f"${x:,.0f}") if money else (lambda x: f"{x:,.0f}")
    return f(v[0]) if abs(v[0] - v[1]) < 0.005 else f"{f(v[0])}–{f(v[1])}"


def analyse(t: Table, rate: float | None, tol: float):
    """Return (failures, notes, summary_line)."""
    fails: list[str] = []
    notes: list[str] = []
    width = max(len(r) for r in t.body)

    money_cols = {
        c for c in range(width)
        if sum(1 for r in t.body if c < len(r) and "$" in r[c]) > len(t.body) / 2
    }

    sums: dict[int, tuple[float, float]] = {}
    for c in range(width):
        vals = [v for v in (parse_cell(r[c]) for r in t.body if c < len(r)) if v]
        if len(vals) < max(2, len(t.body) / 2):
            continue
        sums[c] = (sum(v[0] for v in vals), sum(v[1] for v in vals))
        if any(v[0] < 0 for v in vals):
            t.has_negative = True

    head = lambda c: (t.headers[c].strip() if c < len(t.headers) and t.headers[c].strip()
                      else f"column {c + 1}")

    # A footer is checkable as a column total only when it is unambiguous: one
    # footer row, and no negative rows in the body. Anything else — a credit
    # line, a footer holding two separate lines — is a legitimate shape this
    # cannot verify, and guessing would produce false alarms.
    ambiguous = len(t.footer) > 1 or t.has_negative
    for frow in t.footer:
        for c, got in sums.items():
            if c >= len(frow):
                continue
            stated = parse_cell(frow[c])
            if not stated:
                continue
            ok = abs(stated[0] - got[0]) <= tol and abs(stated[1] - got[1]) <= tol
            if ok:
                notes.append(f"'{head(c)}' total {fmt(stated, c in money_cols)} matches the rows")
            elif ambiguous:
                notes.append(
                    f"NEEDS EYE '{head(c)}' footer says {fmt(stated, c in money_cols)}, "
                    f"rows sum to {fmt(got, c in money_cols)} "
                    f"({'credit or discount row present' if t.has_negative else 'multi-line footer'})"
                )
            else:
                fails.append(
                    f"{t.label}: '{head(c)}' total says {fmt(stated, c in money_cols)} "
                    f"but the rows sum to {fmt(got, c in money_cols)}"
                )

    if rate:
        for mc in sorted(money_cols & set(sums)):
            for hc in sorted(set(sums) - money_cols):
                bad = []
                for i, r in enumerate(t.body, 1):
                    if mc >= len(r) or hc >= len(r):
                        continue
                    money, hours = parse_cell(r[mc]), parse_cell(r[hc])
                    if not money or not hours:
                        continue
                    if abs(hours[0] * rate - money[0]) > tol or abs(hours[1] * rate - money[1]) > tol:
                        bad.append(i)
                if bad:
                    fails.append(
                        f"{t.label}: row(s) {', '.join(map(str, bad))} — '{head(mc)}' "
                        f"does not equal '{head(hc)}' × {rate:g}"
                    )
                else:
                    notes.append(f"'{head(mc)}' = '{head(hc)}' × {rate:g} on every row")

    cols = ", ".join(f"{head(c)} {fmt(v, c in money_cols)}" for c, v in sorted(sums.items()))
    summary = f"{t.label}: {len(t.body)} rows | {cols or 'no numeric columns'}"
    return fails, notes, summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--rate", type=float, help="hourly rate; checks money == hours * rate")
    ap.add_argument("--tolerance", type=float, default=0.51, help="rounding slack (default 0.51)")
    ap.add_argument("--strict", action="store_true", help="also fail on NEEDS EYE")
    args = ap.parse_args()

    try:
        src = open(args.file, encoding="utf-8").read()
    except OSError as e:
        print(f"cannot read {args.file}: {e}", file=sys.stderr)
        return 2

    tables = html_tables(src) if "<table" in src.lower() else markdown_tables(src)
    if not tables:
        print("no tables found — nothing to check")
        return 0

    fails, eyes = [], []
    for t in tables:
        f, n, summary = analyse(t, args.rate, args.tolerance)
        print(summary)
        for line in n:
            print(f"    {'· ' if not line.startswith('NEEDS EYE') else ''}{line}")
            if line.startswith("NEEDS EYE"):
                eyes.append(f"{t.label}: {line[10:]}")
        fails += f
        print()

    for p in fails:
        print(f"FAIL  {p}")
    if fails:
        print(f"\n{len(fails)} problem(s) found")
        return 1
    if eyes:
        print(f"{len(eyes)} table(s) this cannot verify — check those by eye")
        return 1 if args.strict else 0
    print("every column sums and every stated total agrees"
          + (", and money matches hours × rate" if args.rate else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
