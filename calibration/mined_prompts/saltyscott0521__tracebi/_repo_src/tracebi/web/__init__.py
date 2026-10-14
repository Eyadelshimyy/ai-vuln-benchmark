"""
``tracebi.web`` — auto-discovery helpers and the registration facade.

Kept for backward compatibility: ``register`` now lives in the library
proper and is better imported as ``from tracebi import register``, since
the registry is no longer part of the web layer. This spelling still
works and appears throughout the scaffolds and docs.

Notebook usage::

    from tracebi import register          # preferred
    from tracebi.web import register      # equivalent, original spelling

    register.connector(my_connector)
    register.model(my_model, default=True)

    # Reports are PACKAGES (reports/<name>/ or a <name>.json spec) and are
    # registered for you by discovery. ``register.report`` remains the
    # underlying seam it uses; a factory with no package cannot be served.
    # A recurring report declares a ``"schedule"`` block in ``report.json``
    # and runs with ``tracebi schedule``.

    register.auto_discover("reports/")

These calls populate the process-global singleton in
:mod:`tracebi.registry`. No web install is required — the same code works
in a bare notebook and under the running server, and lookups fall back to
the project-root ``models/`` and ``pipelines/`` directories.
"""

from __future__ import annotations

from tracebi.web.discovery import (
    auto_discover,
    discovery_report,
    reload_modules,
)
from tracebi.web.register import register

__all__ = ["register", "auto_discover", "reload_modules", "discovery_report"]
