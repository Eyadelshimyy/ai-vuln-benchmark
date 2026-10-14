"""PII detection and replacement using regex pattern recognition and Sentence Transformers
context validation.

Detection is two-stage:
1. Regex patterns (Presidio-style PatternRecogniser approach) find candidate spans.
   - High confidence: honorific + name (e.g. "Dr. Jane Smith") — kept unconditionally.
   - Low confidence: adjacent title-case words (e.g. "Jane Smith") — validated by context.
2. Low-confidence candidates are scored against reference person-in-context sentences
   using cosine similarity from the all-MiniLM-L6-v2 Sentence Transformers model.
   Candidates below ``context_similarity_threshold`` are discarded.

Replacement is handled by presidio-anonymizer; emitted tags use angle
brackets (``<ENTITY_TYPE>``) inline per detected span. This aligns the
PII marker style with the ``<REDACTED>`` convention used for source
redactions and matches the labelling-packet ground-truth fixtures.
The Sentence Transformers model is loaded lazily on first use.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from womblex.utils.model_registry import SLOT_PII_CONTEXT, record_use, register, resolve

logger = logging.getLogger(__name__)

# Enrichment-derived span: (start, end, entity_type), optionally + entity_id.
_KnownSpan = tuple[int, int, str] | tuple[int, int, str, str]


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# High confidence: recognised honorific immediately precedes a capitalised name.
_HONORIFIC_RE = re.compile(
    r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|Sir|Miss)[^\S\n]+"
    r"[A-Z][a-zA-Z'\-]+(?:[^\S\n]+[A-Z][a-zA-Z'\-]+)*\b"
)

# Lower confidence: two or more adjacent title-case words that could be a full name.
# Uses [^\S\n]+ (non-newline whitespace) so multi-line text blocks are not captured
# as a single name span.
_TITLE_CASE_RE = re.compile(r"\b[A-Z][a-zA-Z'\-]+(?:[^\S\n]+[A-Z][a-zA-Z'\-]+)+\b")

# High-confidence Australian street address pattern.
# Matches: number + street name + street type (+ optional suburb/state/postcode).
# Deliberately conservative — requires a recognisable street type suffix to avoid
# false positives on generic number+word sequences.
_STREET_TYPES = (
    r"Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Place|Pl|Crescent|Cres|"
    r"Court|Ct|Lane|Ln|Terrace|Tce|Way|Boulevard|Blvd|Circuit|Cct|"
    r"Parade|Pde|Highway|Hwy|Close|Cl"
)
_ADDRESS_RE = re.compile(
    # Optional level/unit prefix
    r"(?:(?:Level|Unit|Suite|Lot)[^\S\n]+\d+[,;]?[^\S\n]+)?"
    # Street number (possibly with range like 100-102)
    r"\d+(?:\s*[-–]\s*\d+)?[^\S\n]+"
    # Street name (one or more title-case words)
    r"[A-Z][a-zA-Z'\-]+(?:[^\S\n]+[A-Z][a-zA-Z'\-]+)*[^\S\n]+"
    # Street type (required — this is the high-confidence anchor)
    rf"(?:{_STREET_TYPES})\b"
    # Optional comma + suburb/locality (suburb words must not consume state+postcode)
    r"(?:[,][^\S\n]*[A-Z][a-zA-Z'\-]+"
    r"(?:[^\S\n]+(?!(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)\b[^\S\n]+\d{4})[A-Z][a-zA-Z'\-]+)*)?"
    # Optional state abbreviation + postcode
    r"(?:[,]?[^\S\n]+(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)[^\S\n]+\d{4})?"
)

# Words that form common title-case non-name sequences (e.g. department names,
# place names, calendar terms).  Matches whose every word is in this set are
# skipped without context scoring.
_COMMON_WORDS: frozenset[str] = frozenset({
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "New", "South", "Wales", "Victoria", "Queensland", "Western", "Northern",
    "Australian", "Australia", "Capital", "Territory", "Federal",
    "Government", "Commonwealth", "Parliament", "Senate", "House",
    "Department", "Ministry", "Minister", "Secretary", "Director", "Manager",
    "Officer", "Officers", "Commissioner", "Ombudsman", "Tribunal", "Committee",
    "Act", "Bill", "Policy", "Report", "Review", "Inquiry",
    "The", "This", "That", "These", "Those", "Its", "Their",
    "North", "East", "West", "Central",
    # Regulatory/legal/organisational terms
    "Authorised", "Regulatory", "Authority", "Regulation", "Regulations",
    "Community", "Services", "Service", "Assurance", "Education", "Care",
    "National", "Law", "Laws", "Legal", "Administrative", "Action",
    "Decision", "Attachment", "Refer", "Issue", "Notice", "Direction",
    "School", "Hours", "Centre", "Program", "Programs",
    "Assessment", "Application", "Compliance", "Incorporated", "Association",
    "Council", "Board", "Agency", "Office", "Division", "Unit",
    "Plan", "Plans", "Strategy", "Framework", "Guidelines", "Procedure",
    "Section", "Part", "Clause", "Schedule", "Appendix", "Annexure",
    "Annual", "Corporate", "Regional", "Local", "State", "General", "Senior", "Principal", "Assistant", "Deputy", "Acting",
})

# Reference sentences used to calibrate person-in-context similarity scoring.
# Covers social-work, hearing, and formal-correspondence contexts common in
# Australian government documents.
_REFERENCE_CONTEXTS: list[str] = [
    "The person attended the meeting and provided a statement.",
    "She spoke with the client about the incident and took notes.",
    "He was present at the hearing and answered questions.",
    "The child was referred to the service by their carer.",
    "The worker assessed the family and made recommendations.",
    "The victim reported the incident to the authorities.",
    "The witness described what they saw during the event.",
    "The applicant submitted their claim through the portal.",
    # Formal correspondence and government-official contexts
    "Yours sincerely, the officer signed the letter on behalf of the department.",
    "The director authorised the decision and signed the official correspondence.",
    "The authorised officer's name appeared at the bottom of the regulatory notice.",
    "The assistant director issued the direction under the relevant legislation.",
]


# ---------------------------------------------------------------------------
# Internal match model
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    start: int
    end: int
    entity_type: str
    score: float
    detector: str = ""  # enrichment | regex_high | regex_context
    entity_id: str = ""  # graph entity id for enrichment spans ("" for regex)


# ---------------------------------------------------------------------------
# Context model slot
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_MODEL = "all-MiniLM-L6-v2"


def _unit(x: np.ndarray) -> np.ndarray:
    """Row-normalise *x* so a dot product is cosine similarity."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)  # type: ignore[no-any-return]


class _SentenceTransformerEncoder:
    """Built-in context encoder: ``encode(texts) -> (n, dim)`` embeddings."""

    def __init__(self, name: str = DEFAULT_CONTEXT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        from womblex.utils.models import resolve_local_model_path

        path = resolve_local_model_path(name)
        logger.info("Loading PII context model: %s", path)
        self._model = SentenceTransformer(str(path))

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(  # type: ignore[no-any-return]
            texts, convert_to_numpy=True, show_progress_bar=False
        )


register(
    SLOT_PII_CONTEXT, DEFAULT_CONTEXT_MODEL, _SentenceTransformerEncoder,
    aliases=("sentence-transformers/all-MiniLM-L6-v2",),
)


# ---------------------------------------------------------------------------
# PIICleaner
# ---------------------------------------------------------------------------


class PIICleaner:
    """Detect and replace PII spans with ``<ENTITY_TYPE>`` tags.

    Args:
        entities: Entity types to detect. Currently only ``PERSON`` is supported.
        model: Registered context model (``encode(texts) -> embeddings``).
        model_options: Passed unchanged to the model's factory.
        context_similarity_threshold: Cosine similarity cutoff for low-confidence
            candidates.  Candidates below this score are discarded.
    """

    _HIGH_CONFIDENCE = 0.9

    def __init__(
        self,
        entities: list[str] | None = None,
        model: str = DEFAULT_CONTEXT_MODEL,
        context_similarity_threshold: float = 0.35,
        model_options: dict | None = None,
    ) -> None:
        self._entities: set[str] = set(entities or ["PERSON"])
        # Resolved eagerly so an unknown name fails here, not on first use.
        self._model_entry = resolve(SLOT_PII_CONTEXT, model)
        self._model_options = dict(model_options or {})
        self._threshold = context_similarity_threshold
        self._model: Any = None
        self._ref_embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Build the registered context model on first use."""
        if self._model is None:
            self._model = self._model_entry.factory(**self._model_options)
            record_use(self._model_entry)
            self._ref_embeddings = np.asarray(self._model.encode(_REFERENCE_CONTEXTS))
        return self._model

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _is_all_common(self, text: str) -> bool:
        """Return True if every word in the span is a known non-name word."""
        return all(w in _COMMON_WORDS for w in text.split())

    def _score_context_batch(
        self, text: str, candidates: list[re.Match]  # type: ignore[type-arg]
    ) -> list[float]:
        """Return context similarity scores for a batch of regex matches."""
        contexts = []
        for m in candidates:
            ctx_start = max(0, m.start() - 100)
            ctx_end = min(len(text), m.end() + 100)
            contexts.append(text[ctx_start:ctx_end])

        model = self._load_model()
        embeddings = np.asarray(model.encode(contexts))
        sims = _unit(embeddings) @ _unit(self._ref_embeddings).T  # type: ignore[arg-type]
        return [float(sims[i].max()) for i in range(len(candidates))]

    def _find_candidates(self, text: str) -> list[_Candidate]:
        """Find and validate PII candidate spans in ``text``."""
        candidates: list[_Candidate] = []
        high_spans: list[tuple[int, int]] = []

        # Address detection (high confidence — requires street type anchor)
        if "ADDRESS" in self._entities:
            for m in _ADDRESS_RE.finditer(text):
                candidates.append(
                    _Candidate(m.start(), m.end(), "ADDRESS", self._HIGH_CONFIDENCE, "regex_high")
                )
                high_spans.append((m.start(), m.end()))

        if "PERSON" in self._entities:
            # Stage 1: high-confidence honorific matches
            for m in _HONORIFIC_RE.finditer(text):
                candidates.append(
                    _Candidate(m.start(), m.end(), "PERSON", self._HIGH_CONFIDENCE, "regex_high")
                )
                high_spans.append((m.start(), m.end()))

            # Stage 2: low-confidence title-case candidates
            low: list[re.Match] = []  # type: ignore[type-arg]
            for m in _TITLE_CASE_RE.finditer(text):
                # Skip if already covered by a high-confidence span
                if any(s <= m.start() and m.end() <= e for s, e in high_spans):
                    continue
                # Skip obvious non-name sequences
                if self._is_all_common(m.group()):
                    continue
                low.append(m)

            if low:
                scores = self._score_context_batch(text, low)
                for m, score in zip(low, scores):
                    if score >= self._threshold:
                        candidates.append(
                            _Candidate(m.start(), m.end(), "PERSON", score, "regex_context")
                        )

        # Sort by position; resolve overlaps (keep first)
        candidates.sort(key=lambda c: c.start)
        deduped: list[_Candidate] = []
        last_end = -1
        for cand in candidates:
            if cand.start >= last_end:
                deduped.append(cand)
                last_end = cand.end

        return deduped

    # ------------------------------------------------------------------
    # Enrichment-derived candidates
    # ------------------------------------------------------------------

    @staticmethod
    def _enrichment_candidates(
        text_len: int,
        text_offset: int,
        known_spans: Sequence[_KnownSpan],
    ) -> list[_Candidate]:
        """Convert enrichment-derived spans to local-offset candidates.

        ``known_spans`` are ``(start, end, entity_type)`` or
        ``(start, end, entity_type, entity_id)`` tuples in full-document
        coordinates.  ``text_offset`` is where the current text block begins
        in the full document, used to compute local positions within the block.

        Spans that do not overlap the current text block are discarded.
        Spans are clipped to text boundaries.
        """
        candidates: list[_Candidate] = []
        text_end = text_offset + text_len
        for span in known_spans:
            start, end, entity_type, *rest = span
            entity_id = rest[0] if rest else ""
            if start >= text_end or end <= text_offset:
                continue
            local_start = max(0, start - text_offset)
            local_end = min(text_len, end - text_offset)
            if local_end > local_start:
                candidates.append(
                    _Candidate(local_start, local_end, entity_type, 0.95, "enrichment", entity_id)
                )
        return candidates

    # ------------------------------------------------------------------
    # Span detection (no rewrite) — used by the per-stage pii sidecar
    # ------------------------------------------------------------------

    def detect_spans(
        self,
        text: str,
        known_spans: Sequence[_KnownSpan] | None = None,
        text_offset: int = 0,
        *,
        use_regex: bool = True,
    ) -> list[_Candidate]:
        """Return merged, overlap-resolved PII candidates for ``text``.

        Combines enrichment-derived spans (high confidence, pre-validated by
        Isaacus — passed via ``known_spans`` in full-document coordinates) with
        the local regex/context detector. Set ``use_regex=False`` to rely on
        the graph alone (isolates graph-only recall for measurement). Returns
        candidates with chunk-local offsets; does NOT rewrite the text.
        """
        if not text:
            return []
        candidates: list[_Candidate] = []
        if use_regex:
            candidates.extend(self._find_candidates(text))
        if known_spans:
            candidates.extend(
                self._enrichment_candidates(len(text), text_offset, known_spans)
            )
        # Sort by position, prefer higher score on overlap, keep first
        candidates.sort(key=lambda c: (c.start, -c.score))
        deduped: list[_Candidate] = []
        last_end = -1
        for cand in candidates:
            if cand.start >= last_end:
                deduped.append(cand)
                last_end = cand.end
        return deduped

    def _anonymize(self, text: str, candidates: list[_Candidate]) -> tuple[str, int]:
        """Replace ``candidates`` in ``text`` with ``<ENTITY_TYPE>`` tags."""
        if not candidates:
            return text, 0
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

        engine = AnonymizerEngine()
        analyzer_results = [
            RecognizerResult(
                entity_type=c.entity_type, start=c.start, end=c.end, score=c.score
            )
            for c in candidates
        ]
        operators = {
            c.entity_type: OperatorConfig("replace", {"new_value": f"<{c.entity_type}>"})
            for c in candidates
        }
        result = engine.anonymize(
            text=text, analyzer_results=analyzer_results, operators=operators
        )
        return result.text, len(candidates)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> tuple[str, int]:
        """Replace PII spans with ``<ENTITY_TYPE>`` tags.

        Args:
            text: Input text.

        Returns:
            Tuple of ``(cleaned_text, replacement_count)``.
        """
        if not text:
            return text, 0
        return self._anonymize(text, self.detect_spans(text))

    def clean_with_known_spans(
        self,
        text: str,
        known_spans: Sequence[_KnownSpan],
        text_offset: int = 0,
    ) -> tuple[str, int]:
        """Replace PII spans using both regex detection and enrichment-derived spans.

        Enrichment spans (from the Isaacus graph) are treated as high-confidence
        candidates and bypass context validation.  Regex-based detection still
        runs to catch anything the enrichment missed.

        Args:
            text: Input text block.
            known_spans: ``(start, end, entity_type)`` tuples in full-document
                coordinates.
            text_offset: Start position of ``text`` within the full document.

        Returns:
            Tuple of ``(cleaned_text, replacement_count)``.
        """
        if not text:
            return text, 0
        candidates = self.detect_spans(text, known_spans, text_offset)
        return self._anonymize(text, candidates)
