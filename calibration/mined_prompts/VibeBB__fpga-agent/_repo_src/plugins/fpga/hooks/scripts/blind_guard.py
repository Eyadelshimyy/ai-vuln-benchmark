#!/usr/bin/env python3
"""Keep a blind reviewer blind: deny reads of earlier reviews and impressions.

Declared only by the ``wire-blind-review`` sub-agent. Anchoring research
shows a judge drifts toward any earlier verdict it can see, even when told
to ignore it, so the independent second review must never open the VRP
logs, the typed advisory records or liaison answers. Paths in tool
arguments and terminal commands are matched textually; anything that
mentions a hidden location is denied (exit 2 + reason).
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, cast

HIDDEN = re.compile(
    r"(^|[\s/'\"=:])("
    r"observations(/|$|[\s'\"])"
    r"|[^\s/'\"]*\.advisory\.json"
    r"|review-visual-[^\s/'\"]*"
    r"|[^\s/'\"]*\.ux-response\.json"
    r"|[^\s/'\"]*\.bard-song\.json"
    r"|design-report\.md"
    r")",
)
GUARDED_TOOLS = {"file_editor", "terminal", "grep", "glob", "apply_patch"}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in cast(dict[str, Any], value).values() for s in _strings(item)]
    if isinstance(value, list):
        return [s for item in cast(list[Any], value) for s in _strings(item)]
    return []


def hidden_target(payload: dict[str, Any]) -> str | None:
    if payload.get("tool_name") not in GUARDED_TOOLS:
        return None
    for text in _strings(payload.get("tool_input")):
        match = HIDDEN.search(text.replace("\\", "/"))
        if match is not None:
            return match.group(2)
    return None


def main() -> int:
    try:
        payload: Any = json.load(sys.stdin)
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    target = hidden_target(cast(dict[str, Any], payload))
    if target is None:
        return 0
    print(
        f"blind review: {target} holds earlier reviews or impressions; judge the image"
        " and the contract only",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
