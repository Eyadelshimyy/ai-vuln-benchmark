"""
Backward-compatible re-export of the TraceBi registry.

The registry moved to :mod:`tracebi.registry` — it has no web dependencies
and is consumed by the CLI, request scripts, and notebooks as well as by
this API. Import it from there in new code::

    from tracebi.registry import registry

Both names are re-exported here so existing routers, app modules, and tests
keep working against the *same* singleton object.

Note for maintainers: a test isolates state by rebinding this module's
attribute (``tracebi.web.api.registry.registry = local``) before the router
under test is first imported. Routers bind the object at import time, so
repointing them directly at ``tracebi.registry`` would silently break that
isolation. If you ever do repoint them, convert that test in the same
change.
"""

from tracebi.registry import Registry, registry  # noqa: F401

__all__ = ["Registry", "registry"]
