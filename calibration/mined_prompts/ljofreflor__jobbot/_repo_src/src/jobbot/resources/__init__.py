"""Packaged seeds and templates for cold install (`jobbot init`)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

_PKG = "jobbot.resources"


def resource_text(name: str) -> str:
    """Read a UTF-8 text file shipped inside the wheel."""
    return resources.files(_PKG).joinpath(name).read_text(encoding="utf-8")


def resource_bytes(name: str) -> bytes:
    return resources.files(_PKG).joinpath(name).read_bytes()


def iter_template_names() -> list[str]:
    """Relative paths under ``templates/`` (e.g. ``cv_ats.txt.j2``)."""
    root = resources.files(_PKG).joinpath("templates")
    names: list[str] = []
    for entry in root.iterdir():
        if entry.is_file() and entry.name.endswith((".j2", ".gitkeep")):
            names.append(entry.name)
    return sorted(names)


def copy_templates(dest: Path) -> list[Path]:
    """Copy packaged Jinja templates into ``dest``; return written paths."""
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    root = resources.files(_PKG).joinpath("templates")
    for entry in root.iterdir():
        if not entry.is_file():
            continue
        target = dest / entry.name
        target.write_bytes(entry.read_bytes())
        written.append(target)
    app_dir = root.joinpath("application")
    if app_dir.is_dir():
        (dest / "application").mkdir(parents=True, exist_ok=True)
        for entry in app_dir.iterdir():
            if entry.is_file():
                out = dest / "application" / entry.name
                out.write_bytes(entry.read_bytes())
                written.append(out)
    return written
