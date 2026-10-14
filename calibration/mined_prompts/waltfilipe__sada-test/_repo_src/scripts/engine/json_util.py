from __future__ import annotations

import json
import math
from typing import Any


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def dumps_json(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(sanitize_for_json(payload), ensure_ascii=False, indent=indent, allow_nan=False)
