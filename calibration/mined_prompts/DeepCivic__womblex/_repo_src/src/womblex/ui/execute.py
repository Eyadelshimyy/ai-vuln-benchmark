"""The Execution Controls' read model (docs/ui-plan.md merge 11).

The console's dispatch actions — enqueue an extraction run, dispatch a run's
downstream stages — live in :mod:`womblex.cloud.dispatch`, shared with the
service API, and are re-exported here. What stays is the console-only read: the
ingest preflight behind the composer's "N documents ready" line.

"Log streaming" is the queue's own job-status transitions
(`RunBoard.list_jobs`) plus the per-stage checkpoints
:mod:`womblex.ui.dashboard` already reads — a batch-granular feed, labelled
as such, not a fabricated line-by-line log the pipeline does not emit.
"""
from __future__ import annotations

import logging

from womblex.cli._shared import NestedCorpusError, normalise_prefix
from womblex.cloud.dispatch import (
    EnqueueResult,
    ExecutionCapability,
    ExecutionDisabled,
    StageDispatchResult,
    _supported_under,
    enqueue_downstream_stages,
    enqueue_extraction,
    execution_status,
)
from womblex.ui.deps import UISettings

__all__ = [
    "EnqueueResult",
    "ExecutionCapability",
    "ExecutionDisabled",
    "StageDispatchResult",
    "enqueue_downstream_stages",
    "enqueue_extraction",
    "execution_status",
    "ingest_preflight",
]

logger = logging.getLogger(__name__)


def ingest_preflight(settings: UISettings, *, input_prefix: str | None = None) -> dict:
    """Reachability + document count of the ingest location *input_prefix* names.

    Feeds the composer's "N documents ready" line through
    :func:`_supported_under` — the listing the enqueue itself uses — so the
    count shown is the count that press would enqueue, for the *same*
    ingest-relative prefix the composer then posts.

    A nested layout reports the refusal rather than a number, and names the
    immediate subdirectories holding documents with a count and a ready-to-send
    prefix each: the recovery from that refusal is to point at one of them, so
    the operator picks one here rather than reaching for the CLI or having the
    deployment's ingest location repointed. Each count is the documents
    anywhere beneath that subdirectory — enough to tell an intended corpus from
    a stray directory, which is what it is for — so one that is itself nested
    previews as a refusal of its own, naming the level below.

    Raises ``ValueError`` on a prefix that escapes the ingest root (→ 400),
    which is where the enqueue refuses it too.
    """
    prefix = normalise_prefix(input_prefix)
    empty: dict[str, object] = {
        "uri": settings.ingest_uri, "input_prefix": prefix, "kind": None,
        "reachable": False, "document_count": 0, "sample": [],
        "subdirectories": [], "error": None,
    }
    if not settings.ingest_uri:
        return {**empty, "error": "no ingest location configured"}
    from womblex.store.remote import is_remote_uri

    kind = "remote" if is_remote_uri(settings.ingest_uri) else "local"
    try:
        _, keys = _supported_under(settings, prefix)
    except NestedCorpusError as e:
        # The count must be the count that would be enqueued, so a layout the
        # enqueue will refuse reports the refusal here rather than a number.
        return {
            **empty, "kind": kind, "reachable": True, "error": str(e),
            "subdirectories": [
                {
                    "name": name,
                    "input_prefix": f"{prefix}/{name}" if prefix else name,
                    "document_count": count,
                }
                for name, count in sorted(e.nested.items())
            ],
        }
    except Exception as e:
        logger.warning("execute: ingest unreachable: %s", e)
        return {**empty, "kind": kind, "error": str(e)}
    return {
        **empty, "kind": kind, "reachable": True,
        "document_count": len(keys), "sample": keys[:5],
    }
