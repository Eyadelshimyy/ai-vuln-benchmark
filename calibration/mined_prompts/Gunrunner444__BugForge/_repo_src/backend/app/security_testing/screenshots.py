"""Controlled project evidence directory for screenshots."""

from __future__ import annotations

from pathlib import Path

_MAX_STEM = 80


class ScreenshotStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = 2_000_000,
        max_files: int = 50,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_files = max_files

    @classmethod
    def for_project(cls, project_id: str, *, base: Path | None = None) -> ScreenshotStore:
        safe = _safe_stem(project_id) or "project"
        root = (base or Path.cwd() / "var" / "bugforge" / "evidence") / safe / "screenshots"
        return cls(root)

    def save(self, data: bytes, *, stem: str) -> Path:
        if not data:
            raise ValueError("Screenshot is empty")
        if len(data) > self.max_bytes:
            raise ValueError(f"Screenshot exceeds size limit ({self.max_bytes} bytes)")
        if any(part in stem for part in ("..", "/", "\\")):
            raise ValueError("Invalid screenshot name")
        name = _safe_stem(stem)
        if not name:
            raise ValueError("Invalid screenshot name")
        path = (self.root / f"{name}.png").resolve()
        if not _is_within(self.root, path):
            raise ValueError("Screenshot path escapes the evidence directory")
        path.write_bytes(data)
        self.cleanup()
        return path

    def cleanup(self) -> None:
        files = sorted(
            (item for item in self.root.glob("*.png") if item.is_file()),
            key=lambda item: item.stat().st_mtime,
        )
        while len(files) > self.max_files:
            stale = files.pop(0)
            stale.unlink(missing_ok=True)


def _safe_stem(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        elif char in {".", " ", ":"}:
            cleaned.append("_")
    return "".join(cleaned)[:_MAX_STEM]


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False
