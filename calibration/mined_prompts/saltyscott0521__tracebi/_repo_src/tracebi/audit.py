"""
Who is doing this — the actor behind a run.

TraceBi's pitch is that every number can be traced back to how it was made.
Until this module existed the audit trail recorded *what* happened and never
*who*: ``tracebi_runs`` knew a layer ran, at what time, over how many rows,
and nothing about the person or process that asked for it. "Who ran this?"
was unanswerable, which is an odd hole in a product sold on traceability and
the first question an auditor asks.

The identity lives in the web layer (``request.state.user``) and the writer
lives in the library (``PipelineRunner``), and the library must not import the
web layer. A context variable carries it across that boundary without one
depending on the other, and works the same for the paths that have no HTTP
request at all — the CLI, a notebook, a scheduled job.

Nothing here is required: an unattributed run records ``None`` and behaves
exactly as before, so this cannot break an existing deployment or a script
that has never heard of actors.

    from tracebi.audit import actor

    with actor("alice", role="admin"):
        runner.run("orders_bronze")     # recorded against alice
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

#: (user, role). A ContextVar rather than a module global so concurrent
#: requests in one process cannot read each other's actor — the exact bug a
#: global would introduce under a threaded or async server.
_actor: ContextVar[tuple[Optional[str], Optional[str]]] = ContextVar(
    "tracebi_actor", default=(None, None)
)


def set_actor(user: Optional[str], role: Optional[str] = None):
    """Set the current actor. Returns a token for :func:`reset_actor`."""
    return _actor.set((user, role))


def reset_actor(token) -> None:
    """Restore whatever actor was current before the matching ``set_actor``."""
    _actor.reset(token)


def get_actor() -> tuple[Optional[str], Optional[str]]:
    """Current ``(user, role)``; ``(None, None)`` when unattributed."""
    return _actor.get()


@contextmanager
def actor(user: Optional[str], role: Optional[str] = None):
    """Scope an actor to a block, restoring the previous one on exit."""
    token = set_actor(user, role)
    try:
        yield
    finally:
        reset_actor(token)
