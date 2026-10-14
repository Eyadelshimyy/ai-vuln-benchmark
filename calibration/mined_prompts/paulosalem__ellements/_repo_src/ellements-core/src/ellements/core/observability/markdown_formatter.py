"""Render an LLM call log as human-readable Markdown.

This is a pure pretty-printer with no I/O dependencies beyond reading
the input ``.jsonl`` file. It is decoupled from
:class:`JsonlPromptLogger` so it works against any
JSON-lines log that follows the :class:`LoggedCall` schema, including
files merged from multiple sources.

Output format:

- One ``## 🤖`` heading per call.
- A parameter table (model, temperature, max tokens, duration).
- A ``### 📨 Messages`` block, each message in a code fence.
- A ``### 🔧 Available Tools`` line listing tool names, if present.
- A ``### ⚙️ Tool Calls`` list of invocations with arguments and
  truncated results.
- Either a ``### 💬 Response`` block (on success) or a ``### ❌ Error``
  block (on failure).
- A ``### 📋 Metadata`` bullet list, if any.

Truncation:
    Long fields are clipped with a footer that records the total
    length. The defaults (3 KB response, 2 KB messages, 300 B tool
    results) keep the document scannable even for very chatty calls.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _truncate(text: str, limit: int) -> str:
    """Clip *text* at *limit* chars and append a length footer."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n… *(truncated, {len(text):,} chars total)*"


def _render_messages_markdown(messages: Iterable[dict[str, Any]]) -> str:
    """Render a message list as a sequence of fenced code blocks."""
    parts: list[str] = []
    for i, message in enumerate(messages, 1):
        role = str(message.get("role", "unknown")).upper()
        content = message.get("content", "")
        if isinstance(content, list):
            rendered = json.dumps(content, indent=2)
        else:
            rendered = str(content)
        parts.append(
            f"**Message {i} — {role}**\n\n```\n{_truncate(rendered, 2000)}\n```"
        )
    return "\n\n".join(parts)


def _render_call_header(record: dict[str, Any]) -> list[str]:
    """Render the call heading + parameter table."""
    lines: list[str] = []
    lines.append(f"## 🤖 {record.get('method', '?')} — {record.get('timestamp', '?')}")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| **Status** | `{record.get('status', '?')}` |")
    lines.append(f"| **Model** | `{record.get('model', '?')}` |")
    if record.get("temperature") is not None:
        lines.append(f"| **Temperature** | {record['temperature']} |")
    if record.get("max_tokens") is not None:
        lines.append(f"| **Max Tokens** | {record['max_tokens']:,} |")
    duration = record.get("duration_ms") or 0
    lines.append(f"| **Duration** | {duration/1000:.2f}s ({duration:,}ms) |")
    if record.get("extra_params"):
        for key, value in record["extra_params"].items():
            lines.append(f"| **{key}** | `{value}` |")
    lines.append("")
    return lines


def _render_tools_section(record: dict[str, Any]) -> list[str]:
    """Render the available-tools line + the executed tool-call list."""
    lines: list[str] = []

    tools = record.get("tools") or []
    if tools:
        names = [
            (tool.get("function") or {}).get("name", "?")
            for tool in tools
        ]
        lines.append(f"### 🔧 Available Tools: {', '.join(names)}")
        lines.append("")

    tool_calls = record.get("tool_calls") or []
    if tool_calls:
        lines.append("### ⚙️ Tool Calls")
        lines.append("")
        for i, call in enumerate(tool_calls, 1):
            name = call.get("name", "?")
            args = call.get("arguments", {})
            result = call.get("result", "")
            lines.append(f"**{i}. `{name}`**")
            lines.append(f"- Args: `{json.dumps(args)}`")
            lines.append(f"- Result: {_truncate(str(result), 300)}")
            lines.append("")
    return lines


def _render_outcome(record: dict[str, Any]) -> list[str]:
    """Render either the success response or the error block."""
    lines: list[str] = []
    if record.get("status") == "ok":
        lines.append("### 💬 Response")
        lines.append("")
        lines.append(f"```\n{_truncate(str(record.get('response', '')), 3000)}\n```")
        lines.append("")
    else:
        lines.append("### ❌ Error")
        lines.append("")
        lines.append(f"`{record.get('error_type', 'UnknownError')}`")
        lines.append("")
        lines.append(f"```\n{_truncate(str(record.get('error_message', '')), 2000)}\n```")
        lines.append("")
    return lines


def _render_metadata(record: dict[str, Any]) -> list[str]:
    """Render the optional metadata bullet list."""
    lines: list[str] = []
    metadata = record.get("metadata") or {}
    if metadata:
        lines.append("### 📋 Metadata")
        lines.append("")
        for key, value in metadata.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
    return lines


def _render_call_markdown(record: dict[str, Any]) -> str:
    """Render a single :class:`LoggedCall`-shaped dict as Markdown.

    Composed of four section helpers — header, messages, tools, and
    outcome + metadata — so each piece stays small and individually
    testable.
    """
    lines: list[str] = []
    lines.extend(_render_call_header(record))

    messages = record.get("messages") or []
    if messages:
        lines.append("### 📨 Messages")
        lines.append("")
        lines.append(_render_messages_markdown(messages))
        lines.append("")

    lines.extend(_render_tools_section(record))
    lines.extend(_render_outcome(record))
    lines.extend(_render_metadata(record))

    lines.append("---")
    return "\n".join(lines)


def format_log_markdown(jsonl_path: str | Path) -> str:
    """Render every record in a ``.jsonl`` log file as Markdown.

    Args:
        jsonl_path: Path to a file produced by
            :class:`~ellements.core.observability.JsonlPromptLogger`,
            or any JSON-lines file following the same schema.

    Returns:
        A Markdown document with one section per logged LLM call,
        separated by ``---`` rules.

    Raises:
        FileNotFoundError: If ``jsonl_path`` does not exist.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    sections: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            sections.append(f"<!-- Skipping malformed line: {exc} -->")
            continue
        sections.append(_render_call_markdown(record))
    return "\n\n".join(sections)


__all__ = ["format_log_markdown"]
