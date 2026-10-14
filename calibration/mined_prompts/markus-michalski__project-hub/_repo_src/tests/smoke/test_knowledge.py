"""Smoke: knowledge files in knowledge/ are parseable, valid Markdown structure."""
from pathlib import Path

import yaml

KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge"


def _all_knowledge_files() -> list[Path]:
    return sorted(KNOWLEDGE_DIR.rglob("*.md"))


def test_knowledge_directory_exists():
    assert KNOWLEDGE_DIR.exists(), f"knowledge/ directory not found: {KNOWLEDGE_DIR}"


def test_at_least_one_knowledge_file():
    files = _all_knowledge_files()
    assert files, "No .md files found under knowledge/"


def test_all_knowledge_files_are_readable():
    failures = []
    for md in _all_knowledge_files():
        try:
            text = md.read_text(encoding="utf-8")
            if not text.strip():
                failures.append(f"{md.relative_to(KNOWLEDGE_DIR)}: empty file")
        except Exception as e:
            failures.append(f"{md.relative_to(KNOWLEDGE_DIR)}: read error — {e}")
    assert not failures, "Unreadable knowledge files:\n" + "\n".join(failures)


def test_all_knowledge_files_have_h1_title():
    """Every knowledge file must start with a top-level heading."""
    failures = []
    for md in _all_knowledge_files():
        text = md.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        has_h1 = any(line.startswith("# ") for line in lines[:10])
        if not has_h1:
            failures.append(str(md.relative_to(KNOWLEDGE_DIR)))
    assert not failures, "Knowledge files missing H1 heading:\n" + "\n".join(failures)


def test_knowledge_subdirs_match_project_types():
    """knowledge/ subdirectories should correspond to known project types."""
    known_types = {
        "consulting",
        "event",
        "generic",
        "it-project",
        "marketing",
        "merchant-onboarding",
    }
    actual_dirs = {p.name for p in KNOWLEDGE_DIR.iterdir() if p.is_dir()}
    unknown = actual_dirs - known_types
    assert not unknown, f"Unknown knowledge subdirectories: {sorted(unknown)}"
