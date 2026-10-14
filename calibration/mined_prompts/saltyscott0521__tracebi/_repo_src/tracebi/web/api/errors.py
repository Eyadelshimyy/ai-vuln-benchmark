"""Shared structured-error payload for API routers."""

from __future__ import annotations

import os
import traceback


def error_detail(message: str, exc: Exception) -> dict:
    """HTTPException detail: message, exception type, and a dev-only traceback.

    The traceback exposes absolute paths (and so the server username),
    dependency versions, and internal structure — an unauthenticated caller
    could fingerprint the deployment from an error. So it is included only
    under ``TRACEBI_DEV_MODE=1`` (the same gate as the reload endpoint). The
    key is always present so the UI keeps rendering the shape; in production it
    is empty.
    """
    include_trace = os.environ.get("TRACEBI_DEV_MODE") == "1"
    return {
        "message": f"{message}: {exc}",
        "exception_type": type(exc).__name__,
        "traceback": traceback.format_exc() if include_trace else "",
    }
