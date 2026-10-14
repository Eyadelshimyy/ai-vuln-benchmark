"""Write note content as markdown files to docs_path subfolders."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

_TYPE_TO_SUBFOLDER: dict[str, str] = {
    "meeting-notes": "meeting-notes",
    "email": "emails",
    "action-item": "misc",
    "decision": "misc",
    "note": "misc",
}

_TODO_FILENAME = "todo.md"


def write_note_to_disk(docs_path: str, title: str, content: str, note_type: str) -> str:
    """Write a note to the appropriate subfolder under docs_path.

    Returns the absolute path of the written file.
    """
    base = Path(docs_path)
    subfolder = _TYPE_TO_SUBFOLDER.get(note_type, "misc")
    folder = base / subfolder
    folder.mkdir(parents=True, exist_ok=True)

    if note_type == "action-item":
        file_path = folder / _TODO_FILENAME
        _append_action_item(file_path, title)
    else:
        today = date.today().isoformat()
        slug = _slugify(title)
        filename = f"{today}-{slug}.md"
        file_path = folder / filename
        file_path.write_text(_render_note(title, content, note_type), encoding="utf-8")

    return str(file_path)


def _append_action_item(todo_path: Path, title: str) -> None:
    today = date.today().isoformat()
    item_line = f"- [ ] {title} ({today})"

    if not todo_path.exists():
        lines = ["# Action Items", "", "## Open", "", item_line, "", "## Done", ""]
        todo_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines = todo_path.read_text(encoding="utf-8").splitlines()

    insert_pos: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == "## Open":
            insert_pos = i + 1
            while insert_pos < len(lines) and lines[insert_pos].strip() == "":
                insert_pos += 1
            break

    if insert_pos is not None:
        lines.insert(insert_pos, item_line)
    else:
        lines.append(item_line)

    todo_path.write_text("\n".join(lines), encoding="utf-8")


def _render_note(title: str, content: str, note_type: str) -> str:
    today = date.today().isoformat()
    return f"# {title}\n\n**Date:** {today}\n**Type:** {note_type}\n\n{content}\n"


def _slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug.strip("-")[:50]
