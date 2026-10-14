"""Record successful inspect_image_with_vision observations without blocking.

Vision-derived observations are L2-only audit data: they are hashed and
appended to a JSONL log, never promoted to gate verdicts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _provenance import (
    actor,
    event_id,
    events_path,
    next_sequence,
)

EVENTS_ENV = "FPGA_VISION_TOOL_EVENTS"
EVENTS_RELATIVE_PATH = Path("observations/fpga/vision-tool-events.jsonl")
VISION_TOOL_NAME = "inspect_image_with_vision"


def _response_sha256(response: str) -> str:
    return f"sha256:{hashlib.sha256(response.encode('utf-8')).hexdigest()}"


def _record(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("tool_name") != VISION_TOOL_NAME:
        return None
    response = payload.get("tool_response")
    if not isinstance(response, dict):
        return None
    response = cast(dict[str, Any], response)
    if "error" in response or response.get("is_error"):
        return None
    answer = response.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    tool_input = payload.get("tool_input")
    tool_input = {} if not isinstance(tool_input, dict) else cast(dict[str, Any], tool_input)
    profile_name = response.get("profile_name")
    model = response.get("model")
    if (
        not isinstance(profile_name, str)
        or not profile_name.strip()
        or not isinstance(model, str)
        or not model.strip()
    ):
        return None
    identity = {
        "sequence": 0,
        "tool_name": VISION_TOOL_NAME,
        "tool_input": tool_input,
        "profile_name": profile_name,
        "model": model,
        "response_sha256": _response_sha256(answer),
    }
    return {
        "tool_name": VISION_TOOL_NAME,
        "profile_name": profile_name,
        "model": model,
        "image_index": tool_input.get("image_index"),
        "question": tool_input.get("question"),
        "identity": identity,
        "session_id": payload.get("session_id"),
        "actor": actor(payload),
        "tool_call_id": payload.get("tool_call_id") or payload.get("action_id"),
    }


def main() -> int:
    try:
        payload: Any = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return 0
    if not isinstance(payload, dict):
        return 0
    candidate = _record(cast(dict[str, Any], payload))
    if candidate is None:
        return 0
    path = events_path(cast(dict[str, Any], payload), EVENTS_ENV, EVENTS_RELATIVE_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sequence = next_sequence(path)
        identity = cast(dict[str, Any], candidate.pop("identity"))
        identity["sequence"] = sequence
        record = {
            "sequence": sequence,
            "event_id": event_id(identity),
            "tool_name": candidate["tool_name"],
            "profile_name": candidate["profile_name"],
            "model": candidate["model"],
            "image_index": candidate["image_index"],
            "question": candidate["question"],
            "response_sha256": identity["response_sha256"],
            "recorded_at": datetime.now(UTC).isoformat(),
            "session_id": candidate["session_id"],
            "actor": candidate["actor"],
            "tool_call_id": candidate["tool_call_id"],
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"vision tool event log unavailable: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
