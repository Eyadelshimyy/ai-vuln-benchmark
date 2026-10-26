"""Bounded baseline tune from one job posting (#54).

One vacancy → a few density/clarity suggestions → HITL → profile.yaml.
Never invents; never deletes facts. The tailored PDF stays derived.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Decision = Literal["yes", "no", "skip_all"]

_JOB_ID_RE = re.compile(r"^J\d+$", re.IGNORECASE)
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def resolve_tune_ref(ref: str) -> tuple[Literal["url", "id"], str]:
    """Parse `cv tune-for` argument into a hard-link URL or a stored job id."""
    text = (ref or "").strip()
    if not text:
        msg = "pass a job id (Jxxxx) or a hard-link URL"
        raise ValueError(msg)
    if _URL_RE.match(text):
        return "url", text
    if _JOB_ID_RE.match(text):
        return "id", text.upper()
    msg = f"not a job id or URL: {ref!r}"
    raise ValueError(msg)


def char_delta_label(before: str, after: str) -> str:
    """Human label for how much tighter (or longer) a rewrite is."""
    delta = len(after) - len(before)
    if delta == 0:
        return "same length"
    if delta < 0:
        return f"{delta} chars"
    return f"+{delta} chars"


def prompt_advice_decision(
    prompt: Callable[[str], str] = input,
) -> Decision:
    """HITL gate: yes / no (default) / stop the rest of the run."""
    raw = prompt("Take this wording? [y/N/skip-all] ").strip().casefold()
    if raw in {"y", "yes"}:
        return "yes"
    if raw in {"skip-all", "skipall", "all"}:
        return "skip_all"
    return "no"


def backup_profile(path: Path) -> Path:
    """Copy profile.yaml beside itself as profile.yaml.bak.<UTC stamp>."""
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{stamp}")
    shutil.copy2(path, backup)
    return backup
