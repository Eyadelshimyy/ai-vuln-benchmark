"""Interpretation limits for governed reference-free RAG evidence regimes.

The adapter makes negative identification claims explicit for reporting and
policy layers. It performs no scoring, retrieval, truth adjudication, or
statistical arithmetic, and it does not upgrade any evidence regime into a
ground-truth authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from types import MappingProxyType
from typing import Any

from ._contract_safety import artifact_digest, enum_value
from ._validation import CanonicalContract, assessment_error
from .rag import RAGEvidenceRegime

_RAG_EVIDENCE_LIMITATIONS_TOKEN = object()

_REGIME_LIMITATIONS = MappingProxyType(
    {
        RAGEvidenceRegime.PROMPT_ONLY: (
            "absolute_retrieval_recall_not_identified",
            "no_external_factual_authority",
            "world_correctness_not_identified",
        ),
        RAGEvidenceRegime.RETRIEVED_CONTEXT: (
            "absolute_retrieval_recall_not_identified",
            "target_context_only",
            "world_correctness_not_identified",
        ),
        RAGEvidenceRegime.POOLED_CORPUS: (
            "corpus_authority_not_implied",
            "corpus_completeness_not_implied",
            "world_correctness_not_identified",
        ),
        RAGEvidenceRegime.AUTHORITATIVE_CORPUS: (
            "authority_scope_requires_external_validation",
            "corpus_completeness_not_implied",
            "world_correctness_not_identified",
        ),
        RAGEvidenceRegime.HUMAN_ANCHOR: (
            "absolute_retrieval_recall_not_identified_without_corpus_design",
            "human_anchor_is_fallible_observation",
            "world_correctness_not_identified",
        ),
    }
)


@dataclass(frozen=True)
class RAGEvidenceRegimeLimitations(CanonicalContract):
    """Factory-sealed negative identification claims for one evidence regime."""

    regime: RAGEvidenceRegime
    limitation_codes: tuple[str, ...]
    _limitations_token: InitVar[object | None] = None

    def __post_init__(self, _limitations_token: object | None) -> None:
        """Reject caller-authored semantics and normalize the evidence regime."""
        if _limitations_token is not _RAG_EVIDENCE_LIMITATIONS_TOKEN:
            raise assessment_error(
                "unverified_rag_evidence_limitations",
                "$",
                "use rag_evidence_regime_limitations",
            )
        normalized_regime = enum_value(
            self.regime,
            RAGEvidenceRegime,
            "rag_evidence_regime",
            "$.regime",
        )
        expected = _REGIME_LIMITATIONS[normalized_regime]
        object.__setattr__(self, "regime", normalized_regime)
        object.__setattr__(self, "limitation_codes", expected)

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical limitation content without the derived digest."""
        return {
            "regime": self.regime.value,
            "limitation_codes": list(self.limitation_codes),
        }

    @property
    def limitations_fingerprint(self) -> str:
        """Return SHA-256 over the replay-validated interpretation-limit contract."""
        replayed = _replay_limitations(self)
        return artifact_digest(replayed)

    def to_dict(self) -> dict[str, Any]:
        """Return replay-validated limitation content and deterministic digest."""
        replayed = _replay_limitations(self)
        return {
            **RAGEvidenceRegimeLimitations._content_dict(replayed),
            "limitations_fingerprint": artifact_digest(replayed),
        }


def _replay_limitations(
    value: RAGEvidenceRegimeLimitations,
) -> RAGEvidenceRegimeLimitations:
    """Re-establish exact factory-derived state before manifest projection."""
    if (
        type(value) is not RAGEvidenceRegimeLimitations
        or type(value.regime) is not RAGEvidenceRegime
        or type(value.limitation_codes) is not tuple
        or any(type(code) is not str for code in value.limitation_codes)
        or value.limitation_codes != _REGIME_LIMITATIONS[value.regime]
    ):
        raise assessment_error(
            "invalid_rag_evidence_limitations",
            "$",
            "RAG evidence limitations must retain factory-derived state",
        )
    return RAGEvidenceRegimeLimitations(
        regime=value.regime,
        limitation_codes=_REGIME_LIMITATIONS[value.regime],
        _limitations_token=_RAG_EVIDENCE_LIMITATIONS_TOKEN,
    )


def rag_evidence_regime_limitations(
    regime: RAGEvidenceRegime | str,
) -> RAGEvidenceRegimeLimitations:
    """Return package-owned reporting limitations for one RAG evidence regime.

    The returned codes state what the evidence label alone cannot establish.
    No regime label by itself identifies world correctness. Target-retriever
    context also cannot identify absolute retrieval recall. Authoritative-corpus
    and human-anchor labels remain bounded claims: the caller must validate
    authority scope, completeness, and human measurement quality outside this
    adapter.
    """
    normalized_regime = enum_value(
        regime,
        RAGEvidenceRegime,
        "rag_evidence_regime",
        "$.regime",
    )
    return RAGEvidenceRegimeLimitations(
        regime=normalized_regime,
        limitation_codes=_REGIME_LIMITATIONS[normalized_regime],
        _limitations_token=_RAG_EVIDENCE_LIMITATIONS_TOKEN,
    )


__all__ = [
    "RAGEvidenceRegimeLimitations",
    "rag_evidence_regime_limitations",
]
