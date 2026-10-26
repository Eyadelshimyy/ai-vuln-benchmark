from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "services" / "legacy-billing" / "db" / "procs"
FIXTURE_FILES = (
    ROOT / "services" / "legacy-billing" / "db" / "schema.sql",
    ROOT / "services" / "legacy-billing" / "db" / "seed.sql",
)


def _digest(paths: list[Path] | tuple[Path, ...], relative_to: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(relative_to)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_sha() -> str:
    return _digest(list(PROC_DIR.glob("*.sql")), PROC_DIR)


def fixture_sha() -> str:
    return _digest(FIXTURE_FILES, ROOT)
