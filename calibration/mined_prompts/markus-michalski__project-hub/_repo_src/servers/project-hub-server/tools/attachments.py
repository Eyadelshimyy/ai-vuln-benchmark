"""File attachment operations for notes.

Every file copy also generates a best-effort `<name>.md` sibling (via MarkItDown)
so the text content of PDFs/DOCX/XLSX etc. is available without re-parsing the
binary on every read. Images are skipped (no real OCR). See _convert_to_markdown_sibling().
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from .db import db_connection
from .notes import get_note
from .projects import get_project_by_id

_SIZE_WARN_BYTES = 10 * 1024 * 1024  # 10 MB

# MarkItDown only extracts EXIF metadata for images (no llm_client wired in),
# not real OCR text — a sibling .md would just be metadata noise. Claude reads
# images natively, so skipping them is strictly better.
_MARKDOWN_SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".tiff", ".tif", ".svg", ".heic", ".md", ".markdown",
}

# Lazily instantiated on first real use — importing markitdown pulls in a heavy
# transitive chain (pandas, numpy, onnxruntime, ~215 MB) that would otherwise
# slow down every MCP server cold start, even sessions that never touch attachments.
# Typed as Any (not markitdown.MarkItDown) so mypy never has to resolve markitdown's
# own type stubs, which transitively pull in a numpy stub requiring Python 3.12+
# syntax — incompatible with this project's `python_version = "3.11"` mypy target.
_markitdown: Optional[Any] = None
_markitdown_load_attempted = False


def _get_markitdown() -> Optional[Any]:
    global _markitdown, _markitdown_load_attempted
    if _markitdown is not None or _markitdown_load_attempted:
        return _markitdown
    _markitdown_load_attempted = True
    try:
        from markitdown import MarkItDown

        _markitdown = MarkItDown()
    except Exception as exc:
        # Markdown-sibling generation is best-effort — a missing markitdown
        # install, or a broken native dependency (e.g. onnxruntime failing to
        # load its platform binary), must not disable attach_file/remove_attachment.
        print(f"[project-hub] WARNING: markitdown unavailable: {exc}", file=sys.stderr)
        _markitdown = None
    return _markitdown


def _convert_to_markdown_sibling(dest: Path) -> None:
    """Best-effort: write `dest.name + '.md'` next to a copied attachment.

    Never raises — conversion failures only warn to stderr and must never abort
    the attachment copy itself. Never overwrites an existing file at the sibling
    path either: that path could already be occupied by an unrelated attachment
    (e.g. a real file literally named "notes.md"), so a collision just means no
    sibling gets written this time, rather than silently destroying user data.
    """
    if dest.suffix.lower() in _MARKDOWN_SKIP_EXTENSIONS:
        return
    sibling = dest.with_name(dest.name + ".md")
    if sibling.exists():
        return
    markitdown = _get_markitdown()
    if markitdown is None:
        return
    try:
        result = markitdown.convert(str(dest))
        markdown = result.markdown.strip()
        if markdown:
            sibling.write_text(markdown, encoding="utf-8")
    except Exception as exc:
        print(
            f"[project-hub] WARNING: Markdown conversion failed for {dest.name}: {exc}",
            file=sys.stderr,
        )


def _get_attachments_dir(note: dict) -> Path:
    project = get_project_by_id(note["project_id"])
    if not project or not project.get("docs_path"):
        raise ValueError(f"Project for note {note['id']} has no docs_path")
    return Path(project["docs_path"]) / "attachments"


def _resolve_source(file_path: str, home: Path) -> Path:
    """Resolve `file_path` and confirm it exists, is a regular file, lives under
    `home`, and isn't hidden.

    Shared validation for attach_file() and attach_files(). The existence/traversal
    checks are unchanged from attach_file()'s pre-extraction behavior. Two checks
    were added when attach_files() made batch/folder-shaped imports mandatory
    (project-hub#134 review):
    - is_file(): exists() is also true for directories and FIFOs — a directory makes
      shutil.copy2() raise immediately, but a FIFO makes it block forever waiting for
      a writer. Reject anything that isn't a regular file up front.
    - hidden-path guard: with preservation now mandatory and often folder-shaped,
      silently sweeping up a dotfile like ~/.ssh/id_rsa or ~/.aws/credentials is a
      real risk a single "under $HOME" boundary doesn't cover. Any dotfile path
      component between `home` and the file is refused.
    """
    source = Path(file_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        rel = source.relative_to(home)
    except ValueError:
        raise ValueError(f"Path traversal blocked: {file_path} is not under {home}") from None
    if not source.is_file():
        raise ValueError(f"Not a regular file: {file_path}")
    if any(part.startswith(".") for part in rel.parts):
        raise ValueError(f"Refusing to attach a hidden/dotfile path: {file_path}")
    return source


def _copy_with_disambiguation(source: Path, dest_dir: Path, dest_name: str) -> Path:
    """Copy `source` into `dest_dir` as `dest_name`, never silently overwriting an
    existing file that happens to share that name — disambiguates with a counter
    suffix instead (`invoice-2.pdf`, `invoice-3.pdf`, ...). Returns the actual
    destination path used.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    name_path = Path(dest_name)
    stem, suffix = name_path.stem, name_path.suffix
    dest = dest_dir / dest_name
    counter = 2
    while dest.exists():
        dest = dest_dir / f"{stem}-{counter}{suffix}"
        counter += 1
    shutil.copy2(source, dest)
    _convert_to_markdown_sibling(dest)
    return dest


def attach_file(
    note_id: int,
    file_path: str,
    *,
    home_override: Optional[Path] = None,
) -> dict:
    """Copy a local file to the note's attachments folder and store the reference.

    home_override: used in tests to simulate a different home directory for
    path-traversal checking.
    """
    note = get_note(note_id)
    if note is None:
        raise ValueError(f"Note not found: {note_id}")

    # .resolve() matters on Windows: home_override/Path.home() can be an 8.3
    # short path (e.g. RUNNER~1) that won't compare equal to the resolved
    # long-form `source` below even for genuinely identical directories.
    home = (home_override or Path.home()).resolve()
    source = _resolve_source(file_path, home)

    size = source.stat().st_size
    if size > _SIZE_WARN_BYTES:
        print(
            f"[project-hub] WARNING: Attachment {source.name} is {size / 1024 / 1024:.1f} MB (> 10 MB)",
            file=sys.stderr,
        )

    dest_dir = _get_attachments_dir(note)
    dest = _copy_with_disambiguation(source, dest_dir, source.name)

    attachment = {"name": dest.name, "path": str(dest), "size": size}

    current = json.loads(note["attachments"])
    current.append(attachment)

    with db_connection() as conn:
        with conn:
            conn.execute(
                "UPDATE notes SET attachments = ? WHERE id = ?",
                (json.dumps(current), note_id),
            )

    return attachment


def attach_files(
    note_id: int,
    file_paths: list[str],
    *,
    home_override: Optional[Path] = None,
) -> dict:
    """Copy multiple local files into the note's attachments folder in one call.

    Unlike attach_file(), a bad individual path (missing file, not a regular file,
    hidden/dotfile, path-traversal violation) is collected instead of raised, so
    one bad entry in a batch (e.g. someone pointed Claude at a whole folder) never
    loses the rest — see project-hub#134. Note-level problems still raise ValueError
    immediately, since there is nothing per-file to isolate them from: the note not
    existing, or (via _get_attachments_dir) the project having no docs_path. Callers
    that create a note and attach in the same step (see add_note()'s source_paths)
    must catch this — the note row may already be committed by the time it's raised.

    Returns {"attached": [{"name", "path", "size"}, ...], "failed": [{"path", "error"}, ...]}.
    """
    note = get_note(note_id)
    if note is None:
        raise ValueError(f"Note not found: {note_id}")

    home = (home_override or Path.home()).resolve()
    dest_dir = _get_attachments_dir(note)

    attached: list[dict] = []
    failed: list[dict] = []
    seen: set[Path] = set()

    for file_path in file_paths:
        try:
            source = _resolve_source(file_path, home)
            if source in seen:
                continue  # same resolved file passed twice in this batch — skip silently
            seen.add(source)
            size = source.stat().st_size
            if size > _SIZE_WARN_BYTES:
                print(
                    f"[project-hub] WARNING: Attachment {source.name} is "
                    f"{size / 1024 / 1024:.1f} MB (> 10 MB)",
                    file=sys.stderr,
                )
            dest = _copy_with_disambiguation(source, dest_dir, source.name)
            attached.append({"name": dest.name, "path": str(dest), "size": size})
        except (ValueError, OSError) as exc:  # FileNotFoundError is an OSError subclass
            failed.append({"path": file_path, "error": str(exc)})

    if attached:
        current = json.loads(note["attachments"])
        current.extend(attached)
        with db_connection() as conn:
            with conn:
                conn.execute(
                    "UPDATE notes SET attachments = ? WHERE id = ?",
                    (json.dumps(current), note_id),
                )

    return {"attached": attached, "failed": failed}


def list_attachments(note_id: int) -> list[dict]:
    """Return all attachments for a note."""
    note = get_note(note_id)
    if note is None:
        raise ValueError(f"Note not found: {note_id}")
    return json.loads(note["attachments"])


def remove_attachment(note_id: int, file_name: str) -> None:
    """Remove an attachment by file name. Deletes the file (and its markdown
    sibling, if one was generated) and updates the DB.
    """
    note = get_note(note_id)
    if note is None:
        raise ValueError(f"Note not found: {note_id}")

    current: list[dict] = json.loads(note["attachments"])
    match = next((a for a in current if a["name"] == file_name), None)
    if match is None:
        raise ValueError(f"Attachment not found: {file_name}")

    file_path = Path(match["path"])
    if file_path.exists():
        file_path.unlink()

    sibling = file_path.with_name(file_path.name + ".md")
    if sibling.exists():
        sibling.unlink()

    updated = [a for a in current if a["name"] != file_name]

    with db_connection() as conn:
        with conn:
            conn.execute(
                "UPDATE notes SET attachments = ? WHERE id = ?",
                (json.dumps(updated), note_id),
            )
