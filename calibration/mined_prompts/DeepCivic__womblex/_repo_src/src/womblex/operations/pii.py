"""run_pii_cleaning — replace PII spans with ``<ENTITY_TYPE>`` tags."""

from __future__ import annotations

import logging

from womblex.config import WomblexConfig
from womblex.operations.models import DocumentResult, PreconditionError

logger = logging.getLogger(__name__)


def run_pii_cleaning(
    results: list[DocumentResult], config: WomblexConfig
) -> list[DocumentResult]:
    """Replace PII spans with ``<ENTITY_TYPE>`` tags.

    Uses regex pattern recognisers (Presidio-style) for candidate detection
    and a Sentence Transformers context model (all-MiniLM-L6-v2) for
    low-confidence validation. Both are core dependencies — no extra required.

    The pipeline point is configurable via ``config.pii.pipeline_point``:

    - ``post_extraction``: Cleans page texts before chunking.
    - ``post_chunk``:      Cleans individual chunk texts after chunking.

    Args:
        results: DocumentResults from extraction (and optionally chunking).
        config: Pipeline configuration (uses ``config.pii``).

    Returns:
        The same list with PII replaced in-place.
    """
    if not config.pii.enabled:
        return results

    # Precondition (pragmatic): graph-driven PII needs an enrichment graph. If
    # post_enrichment is requested but no completed document carries enrichment,
    # the enrichment stage never ran — a composition error, not a per-doc gap.
    # A partially-enriched batch (some docs have enrichment) is tolerated: the
    # per-doc path below handles the un-enriched ones.
    if config.pii.pipeline_point == "post_enrichment" and not any(
        dr.enrichment for dr in results if dr.status == "completed"
    ):
        raise PreconditionError(
            "run_pii_cleaning(pipeline_point='post_enrichment') needs the enrichment "
            "graph, but no completed document carries enrichment — run run_enrichment "
            "first, or use pipeline_point='post_chunk'/'post_extraction'."
        )

    from womblex.pii.cleaner import PIICleaner
    from womblex.pii.stage import clean_chunks, clean_enriched_chunks, clean_extraction

    cleaner = PIICleaner(
        entities=config.pii.entities,
        model=config.pii.model,
        model_options=config.pii.model_options or None,
        context_similarity_threshold=config.pii.context_similarity_threshold,
    )
    point = config.pii.pipeline_point

    for dr in results:
        if dr.status != "completed" or not dr.extraction:
            continue
        try:
            if point == "post_extraction":
                count = clean_extraction(dr.extraction, cleaner)
            elif point == "post_chunk":
                count = clean_chunks(dr.chunks, cleaner) if dr.chunks else 0
            elif point == "post_enrichment":
                if not dr.enrichment:
                    logger.debug(
                        "PII [post_enrichment]: doc=%s has no enrichment — falling back to regex",
                        dr.doc_id,
                    )
                    count = clean_chunks(dr.chunks, cleaner) if dr.chunks else 0
                elif not dr.chunks:
                    logger.warning(
                        "PII [post_enrichment]: doc=%s has no chunks — skipping", dr.doc_id,
                    )
                    continue
                else:
                    count = clean_enriched_chunks(
                        dr.chunks,
                        dr.enrichment,
                        cleaner,
                        entities=set(config.pii.entities),
                        person_types=set(config.pii.person_types),
                    )
            else:
                logger.warning("Unknown PII pipeline_point %r for %s — skipping", point, dr.doc_id)
                continue

            if count:
                logger.info("PII [%s]: doc=%s replacements=%d", point, dr.doc_id, count)
        except Exception as exc:
            logger.error("PII cleaning failed: doc=%s error=%s", dr.doc_id, exc)

    return results
