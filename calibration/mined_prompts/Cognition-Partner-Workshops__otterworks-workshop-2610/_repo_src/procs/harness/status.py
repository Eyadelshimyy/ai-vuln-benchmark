from __future__ import annotations

PENDING = "pending"
EXTRACTED = "extracted"
VALID_STATUSES = frozenset({PENDING, EXTRACTED})


def status_for(module: dict | None) -> str | None:
    if module is None:
        return PENDING
    return module.get("status")
