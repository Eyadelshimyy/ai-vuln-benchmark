"""Isaacus embeddings API wrapper (kanon-2-embedder).

Thin adapter over ``client.embeddings.create``. Batches into the API's
128-text-per-request limit, handles 429 rate-limit retries with
exponential backoff, and returns vectors in input order. The embedding
*task* matters: documents being indexed use ``retrieval/document``;
search queries use ``retrieval/query`` (see CLAUDE.md "Isaacus task
types matter").
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "kanon-2-embedder"
DEFAULT_TASK = "retrieval/document"
MAX_TEXTS_PER_REQUEST = 128
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 2.0


def embed_texts(
    texts: list[str],
    client: object,
    *,
    model: str = DEFAULT_MODEL,
    task: str | None = DEFAULT_TASK,
    dimensions: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> list[list[float]]:
    """Embed ``texts`` via the Isaacus embedding API, preserving order.

    Splits into ``MAX_TEXTS_PER_REQUEST`` chunks. Each text must contain a
    non-whitespace character (the caller filters empties). Raises
    ``RuntimeError`` if a request fails after all retries.
    """
    out: list[list[float] | None] = [None] * len(texts)
    for start in range(0, len(texts), MAX_TEXTS_PER_REQUEST):
        batch = texts[start : start + MAX_TEXTS_PER_REQUEST]
        resp = _embed_batch(
            batch, client, model=model, task=task, dimensions=dimensions,
            max_retries=max_retries, retry_base_delay=retry_base_delay,
        )
        for emb in resp.embeddings:  # type: ignore[attr-defined]
            out[start + emb.index] = emb.embedding
    if any(v is None for v in out):
        raise RuntimeError("embedding API returned fewer vectors than inputs")
    return out  # type: ignore[return-value]


def _embed_batch(
    texts: list[str], client: object, *, model: str, task: str | None,
    dimensions: int | None, max_retries: int, retry_base_delay: float,
) -> object:
    kwargs: dict = {"model": model, "texts": texts}
    if task is not None:
        kwargs["task"] = task
    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.embeddings.create(**kwargs)  # type: ignore[attr-defined]
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                delay = retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Rate limited on embeddings (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, max_retries + 1, delay,
                )
                time.sleep(delay)
                continue
            logger.error("Embedding failed: %s", e)
            raise RuntimeError(f"Embedding failed: {e}") from e
    raise RuntimeError(
        f"Embedding failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error
