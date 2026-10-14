"""Live transports for real candidate runs.

Kept out of :mod:`field_notes.adapters` so the adapter layer stays free of HTTP
clients and credential handling. A transport is only ever constructed by an
explicit, opt-in caller (the CLI's ``--live`` flag or a script), never by loading
a manifest.
"""

from field_notes.transports.openai_compatible import OpenAICompatibleTransport

__all__ = ["OpenAICompatibleTransport"]
