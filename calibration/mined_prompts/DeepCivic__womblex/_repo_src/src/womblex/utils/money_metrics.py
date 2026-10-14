"""Money accuracy metrics against typed transcript tags.

The corpus labels a document's financial figures by wrapping each one, in
place, in a typed tag — `<DOLLARS>`, `<THOUSANDS>`, `<MILLIONS>`,
`<BILLIONS>`, `<SHARES>`, `<PERCENT>` — inside a copy of the document's
transcript (`MONEY-LABELLING.md` in the benchmark corpus). Stripping the tags
must reproduce the transcript byte for byte, so the tags carry offsets into the
real text and nothing here needs a second artefact to align against.

Three numbers come out, and the third is the one a bare count hides:

* **recall** — tagged figures the detector also found
* **precision** — detector spans that land on a tagged figure
* **scale accuracy** — of the figures found, how many were read at the
  magnitude the document stated. Reading `$10 000` as ten dollars scores as a
  recall success and is wrong by 10**3; the tag's type catches it.

`<PERCENT>` is money-adjacent, not money: it is tagged so that an amount read
out of a rate is measurable, and it is scored as its own error class rather
than folded into precision.

Nothing here reads parquet or calls a detector — callers pass the spans in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

MONEY_TAGS: tuple[str, ...] = (
    "DOLLARS", "THOUSANDS", "MILLIONS", "BILLIONS", "SHARES", "PERCENT",
)

#: The magnitude each tag asserts. `SHARES` (a per-share amount) and `PERCENT`
#: are kinds, not scales, so they are absent — scale accuracy skips them.
TAG_SCALE: dict[str, Decimal] = {
    "DOLLARS": Decimal(1),
    "THOUSANDS": Decimal(10) ** 3,
    "MILLIONS": Decimal(10) ** 6,
    "BILLIONS": Decimal(10) ** 9,
}

#: `MoneySpan.multiplier` is one canonical name per magnitude.
MULTIPLIER_SCALE: dict[str | None, Decimal] = {
    None: Decimal(1),
    "thousand": Decimal(10) ** 3,
    "million": Decimal(10) ** 6,
    "billion": Decimal(10) ** 9,
    "trillion": Decimal(10) ** 12,
}

_TAG_RE = re.compile(r"</?(" + "|".join(MONEY_TAGS) + r")>")


@dataclass(slots=True)
class TaggedFigure:
    """One ground-truth figure. Offsets index the *stripped* transcript."""

    start: int
    end: int
    kind: str
    text: str
    #: `table` when the figure sits on a flattened markdown table row — the
    #: transcript renders tables inline, and those figures are the column
    #: path's to find, not the self-evidencing detector's.
    locus: str = "narrative"

    @property
    def is_money(self) -> bool:
        return self.kind != "PERCENT"


@dataclass(slots=True)
class MoneyScore:
    """Counts plus the figures behind them, so a report can name each failure."""

    gt_money: int = 0
    gt_percent: int = 0
    predicted: int = 0
    matched: int = 0
    misses: list[TaggedFigure] = field(default_factory=list)
    #: Predicted spans overlapping no tagged figure, as `(span, why)` where
    #: *why* is `"percent"` (landed on a `<PERCENT>` tag) or `"untagged"`.
    false_positives: list[tuple[object, str]] = field(default_factory=list)
    #: `(figure, span)` for figures found at the wrong magnitude.
    scale_errors: list[tuple[TaggedFigure, object]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.matched / self.gt_money if self.gt_money else 0.0

    @property
    def precision(self) -> float:
        hit = self.predicted - len(self.false_positives)
        return hit / self.predicted if self.predicted else 0.0

    @property
    def scale_accuracy(self) -> float:
        return 1.0 - (len(self.scale_errors) / self.matched) if self.matched else 0.0


def parse_tagged_transcript(tagged: str) -> tuple[str, list[TaggedFigure]]:
    """Strip the money tags, returning the plain text and the figures.

    Raises `ValueError` on an unbalanced or nested tag — both are corruption in
    the ground truth, and a silently mis-parsed fixture scores the detector
    against fiction.
    """
    out: list[str] = []
    stack: list[tuple[int, str]] = []
    figures: list[TaggedFigure] = []
    pos = 0
    length = 0

    for m in _TAG_RE.finditer(tagged):
        out.append(tagged[pos:m.start()])
        length += m.start() - pos
        pos = m.end()
        if m.group(0).startswith("</"):
            if not stack:
                raise ValueError(f"closing {m.group(0)} with no opening tag at {m.start()}")
            start, kind = stack.pop()
            if kind != m.group(1):
                raise ValueError(f"<{kind}> closed by {m.group(0)} at {m.start()}")
            figures.append(TaggedFigure(start, length, kind, "".join(out)[start:length]))
        else:
            if stack:
                raise ValueError(f"{m.group(0)} nested inside <{stack[-1][1]}> at {m.start()}")
            stack.append((length, m.group(1)))

    if stack:
        raise ValueError(f"unclosed <{stack[-1][1]}> at offset {stack[-1][0]}")

    out.append(tagged[pos:])
    text = "".join(out)
    for fig in figures:
        fig.locus = "table" if _on_table_row(text, fig.start) else "narrative"
    figures.sort(key=lambda f: f.start)
    return text, figures


def _on_table_row(text: str, offset: int) -> bool:
    return text[text.rfind("\n", 0, offset) + 1:].lstrip().startswith("|")


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def score_money(figures: list[TaggedFigure], spans: list) -> MoneyScore:
    """Score detector *spans* against ground-truth *figures*.

    Spans need `start` / `end` / `multiplier`; any `MoneySpan` satisfies that.
    A figure counts as found when a span overlaps it — the detector's span
    boundaries and the labeller's need not agree to the character.
    """
    money = [f for f in figures if f.is_money]
    percent = [f for f in figures if not f.is_money]
    score = MoneyScore(gt_money=len(money), gt_percent=len(percent), predicted=len(spans))

    claimed: set[int] = set()
    for fig in money:
        hits = [i for i, s in enumerate(spans) if _overlaps(fig.start, fig.end, s.start, s.end)]
        if not hits:
            score.misses.append(fig)
            continue
        score.matched += 1
        claimed.update(hits)
        best = max((spans[i] for i in hits), key=lambda s: s.end - s.start)
        want = TAG_SCALE.get(fig.kind)
        got = MULTIPLIER_SCALE.get(best.multiplier)
        if want is not None and got is not None and want != got:
            score.scale_errors.append((fig, best))

    for i, span in enumerate(spans):
        if i in claimed:
            continue
        on_pct = any(_overlaps(span.start, span.end, p.start, p.end) for p in percent)
        score.false_positives.append((span, "percent" if on_pct else "untagged"))

    return score
