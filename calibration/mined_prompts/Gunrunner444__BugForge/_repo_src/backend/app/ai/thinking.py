"""Extract thinking/reasoning traces from model output.

Thinking is a provider capability, not a Qwen-only setting. Reasoning text
is never treated as the final answer.
"""

from __future__ import annotations

import re

_TAG_PAIRS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
    ("<thought>", "</thought>"),
)

_FENCE_THINK = re.compile(
    r"```(?:think|thinking|reasoning)\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def split_thinking(
    content: str | None,
    *,
    reasoning_field: str | None = None,
) -> tuple[str | None, str]:
    """Return ``(thinking, final_content)``.

    ``reasoning_field`` is used when the API provides a dedicated reasoning
    attribute (e.g. ``message.reasoning_content``). Tagged blocks inside
    ``content`` are stripped from the final answer.
    """
    text = content or ""
    thoughts: list[str] = []
    if reasoning_field and reasoning_field.strip():
        thoughts.append(reasoning_field.strip())

    remaining = text
    for start, end in _TAG_PAIRS:
        extracted, remaining = _extract_tag(remaining, start, end)
        thoughts.extend(extracted)

    for match in _FENCE_THINK.finditer(remaining):
        thoughts.append(match.group(1).strip())
    remaining = _FENCE_THINK.sub("", remaining)

    thinking = "\n\n".join(part for part in thoughts if part) or None
    return thinking, remaining.strip()


def _extract_tag(text: str, start: str, end: str) -> tuple[list[str], str]:
    found: list[str] = []
    remaining = text
    lower = remaining.lower()
    start_l = start.lower()
    end_l = end.lower()
    while True:
        i = lower.find(start_l)
        if i < 0:
            break
        j = lower.find(end_l, i + len(start))
        if j < 0:
            found.append(remaining[i + len(start) :].strip())
            remaining = remaining[:i]
            break
        found.append(remaining[i + len(start) : j].strip())
        remaining = remaining[:i] + remaining[j + len(end) :]
        lower = remaining.lower()
    return found, remaining
