"""Generic record-linkage matcher.

Resolves document mentions (candidates) to canonical reference entities.
The match strategy is intentionally minimal — no rules DSL; the
``match_spec`` is implicit in the :class:`ReferenceTable` (which columns
were declared exact vs fuzzy at load time) plus a similarity threshold:

1. **alias** — curated normalised-name override (handles entities the
   register doesn't carry, e.g. a prior trustee) → confidence 1.0.
2. **address_exact** — for address-kind candidates, normalised equality
   against the entity's ``exact_key`` → 1.0 (OCR-robust primary key).
3. **name_fuzzy** — for name-kind candidates, best ``difflib`` ratio over
   the entity's ``fuzzy_keys`` ≥ threshold → ratio.
4. otherwise **unmatched** (still emitted, ``matched=False``).

Fuzzy matching uses the stdlib :class:`difflib.SequenceMatcher` — no new
dependency (``rapidfuzz`` is not in the dependency set).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from womblex.link.normalise import normalise_address, normalise_name
from womblex.link.reference import ReferenceEntity, ReferenceTable


@dataclass
class Candidate:
    """A document mention to resolve."""

    text: str
    kind: str            # enrichment entity_type: corporate | address | ...
    source_hash: str
    mention_start: int = -1
    mention_end: int = -1
    # Where the mention offsets point (see store/evidence.py): the text layer, the
    # table (``elem_order`` or ``sheet``) for a table mention, and the enricher's
    # own text for the mention (``text`` is the entity's name).
    text_layer: str | None = None
    elem_order: int | None = None
    sheet: str | None = None
    mention_text: str | None = None


@dataclass
class Link:
    """Result of resolving one candidate."""

    candidate: Candidate
    entity: ReferenceEntity | None
    confidence: float
    method: str          # alias | address_exact | name_fuzzy | unmatched

    @property
    def matched(self) -> bool:
        return self.entity is not None


def resolve(
    candidates: list[Candidate],
    reference: ReferenceTable,
    *,
    name_threshold: float = 0.85,
    address_kinds: tuple[str, ...] = ("address",),
) -> list[Link]:
    """Resolve every candidate to at most one reference entity."""
    exact_index: dict[str, ReferenceEntity] = {}
    for e in reference.entities:
        if e.exact_key:
            exact_index.setdefault(e.exact_key, e)
    return [
        _match_one(c, reference, exact_index, name_threshold, address_kinds)
        for c in candidates
    ]


def _match_one(
    c: Candidate,
    reference: ReferenceTable,
    exact_index: dict[str, ReferenceEntity],
    threshold: float,
    address_kinds: tuple[str, ...],
) -> Link:
    # 1. alias override (on the normalised name form)
    alias_id = reference.aliases.get(normalise_name(c.text))
    if alias_id:
        ent = reference.entity_by_id(alias_id)
        if ent is not None:
            return Link(c, ent, 1.0, "alias")

    # 2. address-kind candidate → exact normalised-address equality
    if c.kind in address_kinds:
        ent = exact_index.get(normalise_address(c.text))
        if ent is not None:
            return Link(c, ent, 1.0, "address_exact")
        return Link(c, None, 0.0, "unmatched")

    # 3. name-kind candidate → fuzzy over fuzzy_keys
    query = normalise_name(c.text)
    best_ent: ReferenceEntity | None = None
    best_score = 0.0
    if query:
        for e in reference.entities:
            for fk in e.fuzzy_keys:
                score = _name_similarity(query, fk)
                if score > best_score:
                    best_score, best_ent = score, e
    if best_ent is not None and best_score >= threshold:
        return Link(c, best_ent, best_score, "name_fuzzy")
    return Link(c, None, 0.0, "unmatched")


# Per-token character-similarity floor for treating an (OCR-noisy) query token
# as covering a reference token. Below this a reference token counts as absent,
# which is what preserves brand discrimination (e.g. "urambi" never covers
# "artemis").
_TOKEN_SIM_FLOOR = 0.72


def _name_similarity(a: str, b: str) -> float:
    """Best of two stdlib name scorers — recall + precision + OCR tolerance.

    ``_token_set_ratio`` handles token reordering / token-merge noise
    (``"pty ltd"`` vs ``"ptyltd"``); ``_fuzzy_coverage`` handles intra-token
    OCR typos (``"earty"`` ~ ``"early"``) and extra query tokens (suburb
    suffix) while still rejecting a different brand whose distinctive token
    is absent. Taking the max gives every real Artemis surface form a high
    score without letting generic shared tokens pull in another brand.
    """
    return max(_token_set_ratio(a, b), _fuzzy_coverage(a, b))


def _token_set_ratio(a: str, b: str) -> float:
    """Token-set similarity over whitespace tokens (rapidfuzz-style, stdlib only).

    Scores the sorted token *intersection* against each side's
    intersection+remainder and takes the best ratio. A token superset
    (``"artemis early learning fyshwick"`` vs ``"artemis early learning"``)
    scores 1.0; the distinctive brand token still dominates over generic
    shared tokens. Falls back to a plain ratio when there is no overlap.
    """
    ta, tb = set(a.split()), set(b.split())
    inter = sorted(ta & tb)
    if not inter:
        return SequenceMatcher(None, a, b).ratio()
    s_inter = " ".join(inter)
    s_a = " ".join(inter + sorted(ta - tb)).strip()
    s_b = " ".join(inter + sorted(tb - ta)).strip()
    return max(
        SequenceMatcher(None, s_inter, s_a).ratio(),
        SequenceMatcher(None, s_inter, s_b).ratio(),
        SequenceMatcher(None, s_a, s_b).ratio(),
    )


def _fuzzy_coverage(query: str, ref: str) -> float:
    """Mean best per-token similarity, requiring *every* ref token covered.

    Each reference token must have a query token within ``_TOKEN_SIM_FLOOR``
    char-similarity (tolerating OCR typos like ``earty`` → ``early``); extra
    query tokens (a suburb suffix) don't penalise. If any reference token is
    uncovered (a distinct brand token absent) the score is 0 — this is the
    precision guard that stops one brand matching another.
    """
    qa, rb = query.split(), ref.split()
    if not qa or not rb:
        return 0.0
    total = 0.0
    for rt in rb:
        best = max((SequenceMatcher(None, rt, qt).ratio() for qt in qa), default=0.0)
        if best < _TOKEN_SIM_FLOOR:
            return 0.0
        total += best
    return total / len(rb)
