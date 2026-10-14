"""Edit callbacks for human-in-the-loop strategy execution.

Defines the :class:`EditCallback` Protocol and three concrete
implementations:

- :class:`PassthroughEditCallback` — auto-approves (testing/non-interactive)
- :class:`FileEditCallback` — opens the user's ``$EDITOR`` with a temp file
- :class:`CallableEditCallback` — wraps a plain sync or async callable

:class:`FileEditCallback` uses ``shlex.split`` to interpret the editor
command and ``subprocess.run`` with ``shell=False`` to avoid shell
injection. Filesystem I/O runs through :func:`asyncio.to_thread` so
the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import shlex
import subprocess
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class EditCallback(Protocol):
    """Protocol for presenting content to a user and receiving their edits."""

    async def request_edit(self, content: str, context: str = "") -> str:
        """Present *content* and return the user's edited version.

        Args:
            content: Text to be reviewed/edited.
            context: Human-readable description of what *content*
                represents (e.g. ``"Draft 1 of 3"``).

        Returns:
            The edited text. May equal *content* if the user approves
            without changes. Empty string signals abort.
        """
        ...


class PassthroughEditCallback:
    """Auto-approve callback — returns *content* unchanged."""

    async def request_edit(self, content: str, context: str = "") -> str:
        return content


class FileEditCallback:
    """Opens the user's ``$EDITOR`` with a temp file for editing.

    The file includes a brief instruction header (commented out) so the
    user knows how to signal completion. The header is stripped from
    the returned content.

    Editor commands are interpreted via :func:`shlex.split`, so options
    like ``"code --wait"`` work without invoking a shell.

    Args:
        editor: Editor command to use. Defaults to ``$VISUAL``, then
            ``$EDITOR``, then ``nano``.
    """

    HEADER_MARKER = "# --- EDIT BELOW THIS LINE ---"
    DONE_INSTRUCTION = (
        "# To finish: save and close this file.\n"
        "# To abort:  delete all content below the marker, then save and close.\n"
        "# Context: {context}\n"
    )

    def __init__(self, editor: str | None = None) -> None:
        self._editor = editor

    def _resolve_editor(self) -> str:
        if self._editor:
            return self._editor
        return os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"

    async def request_edit(self, content: str, context: str = "") -> str:
        editor = self._resolve_editor()

        header = (
            self.DONE_INSTRUCTION.format(context=context or "collaborative editing")
            + self.HEADER_MARKER
            + "\n\n"
        )
        file_content = header + content

        suffix = ".md" if self._looks_like_markdown(content) else ".txt"
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            prefix="ellements_edit_",
            delete=False,
        ) as handle:
            handle.write(file_content)
            tmp_path = Path(handle.name)

        try:
            editor_cmd = shlex.split(editor) + [str(tmp_path)]
            await asyncio.to_thread(subprocess.run, editor_cmd, check=False)
            edited = await asyncio.to_thread(tmp_path.read_text, "utf-8")
            if self.HEADER_MARKER in edited:
                _, _, after = edited.partition(self.HEADER_MARKER)
                return after.lstrip("\n")
            return edited.strip()
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink()

    @staticmethod
    def _looks_like_markdown(text: str) -> bool:
        md_signals = ("# ", "## ", "**", "```", "- ", "* ")
        return any(signal in text for signal in md_signals)


class CallableEditCallback:
    """Wrap a plain (sync or async) callable as an :class:`EditCallback`.

    The callable receives ``(content, context)`` and returns the edited
    string.
    """

    def __init__(
        self,
        fn: Callable[[str, str], str] | Callable[[str, str], Awaitable[str]],
    ) -> None:
        self._fn = fn

    async def request_edit(self, content: str, context: str = "") -> str:
        result = self._fn(content, context)
        if inspect.isawaitable(result):
            return await result
        return result


__all__ = [
    "CallableEditCallback",
    "EditCallback",
    "FileEditCallback",
    "PassthroughEditCallback",
]
