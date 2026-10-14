"""Knowledge management — read/write/list knowledge files per project type.

Knowledge files live at ~/.project-hub/knowledge/<project-type>/<topic>.md
They are Markdown files the user maintains locally and Claude reads at session start.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from .config import get_knowledge_root

# Bundled templates ship alongside this plugin
_PLUGIN_TEMPLATES = Path(__file__).parent.parent.parent.parent / "knowledge"


def _knowledge_dir(project_type: str) -> Path:
    d = get_knowledge_root() / project_type
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_topic_path(knowledge_dir: Path, topic: str) -> Path:
    """Return the resolved .md path for topic, raising ValueError on path traversal."""
    candidate = (knowledge_dir / f"{topic}.md").resolve()
    if not candidate.is_relative_to(knowledge_dir.resolve()):
        raise ValueError(f"Invalid topic: {topic!r}")
    return candidate


def list_knowledge(project_type: str) -> list[dict]:
    """List all knowledge topics available for a project type."""
    d = _knowledge_dir(project_type)
    files = sorted(d.glob("*.md"))
    result = []
    for f in files:
        # Extract first H1 as title, fall back to filename stem
        content = f.read_text(encoding="utf-8")
        title = f.stem
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        result.append({
            "topic": f.stem,
            "title": title,
            "path": str(f),
            "size_bytes": f.stat().st_size,
        })
    return result


def get_knowledge(project_type: str, topic: str) -> Optional[dict]:
    """Read a knowledge file. Returns None if not found."""
    d = _knowledge_dir(project_type)
    f = _resolve_topic_path(d, topic)
    if not f.exists():
        return None
    content = f.read_text(encoding="utf-8")
    title = topic
    for line in content.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return {
        "topic": topic,
        "title": title,
        "path": str(f),
        "content": content,
    }


def save_knowledge(project_type: str, topic: str, content: str) -> dict:
    """Write or overwrite a knowledge file."""
    d = _knowledge_dir(project_type)
    f = _resolve_topic_path(d, topic)
    f.write_text(content, encoding="utf-8")
    result = get_knowledge(project_type, topic)
    assert result is not None
    return result


def delete_knowledge(project_type: str, topic: str) -> bool:
    """Delete a knowledge file. Returns True if deleted, False if not found."""
    d = _knowledge_dir(project_type)
    f = _resolve_topic_path(d, topic)
    if f.exists():
        f.unlink()
        return True
    return False


def get_all_knowledge(project_type: str) -> list[dict]:
    """Load full content of all knowledge files for a project type (used on resume)."""
    items = list_knowledge(project_type)
    result = []
    for item in items:
        full = get_knowledge(project_type, item["topic"])
        if full:
            result.append(full)
    return result


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def sync_knowledge_templates(
    force: bool = False,
    plugin_templates_dir: Optional[Path] = None,
) -> dict:
    """Compare bundled plugin templates with the user's installed knowledge files.

    Returns a report dict:
      - items: list of {"path", "status": "up-to-date"|"newer-version-available"|"new",
                        "plugin_bytes", "local_bytes"?}
      - synced: list of paths actually written (only when force=True)
    """
    templates_root = plugin_templates_dir or _PLUGIN_TEMPLATES
    user_root = get_knowledge_root()

    items = []
    synced = []

    for plugin_file in sorted(templates_root.rglob("*.md")):
        rel = plugin_file.relative_to(templates_root)
        user_file = user_root / rel

        entry: dict = {
            "path": rel.as_posix(),
            "plugin_bytes": plugin_file.stat().st_size,
        }

        if not user_file.exists():
            entry["status"] = "new"
        elif _md5(plugin_file) == _md5(user_file):
            entry["status"] = "up-to-date"
            entry["local_bytes"] = user_file.stat().st_size
        else:
            entry["status"] = "newer-version-available"
            entry["local_bytes"] = user_file.stat().st_size

        if force and entry["status"] != "up-to-date":
            user_file.parent.mkdir(parents=True, exist_ok=True)
            user_file.write_bytes(plugin_file.read_bytes())
            synced.append(rel.as_posix())

        items.append(entry)

    return {"items": items, "synced": synced}
