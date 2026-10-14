from __future__ import annotations

from pathlib import Path

from app.analyzers.language_detector import detect_languages, language_for_path


def test_python_extension() -> None:
    assert language_for_path(Path("foo.py")) == "python"


def test_typescript_extension() -> None:
    assert language_for_path(Path("foo.ts")) == "typescript"


def test_tsx_extension() -> None:
    assert language_for_path(Path("foo.tsx")) == "typescript"


def test_unknown_extension() -> None:
    assert language_for_path(Path("foo.xyz")) is None


def test_detect_languages_basic() -> None:
    paths = [
        Path("a.py"),
        Path("b.py"),
        Path("c.py"),
        Path("d.ts"),
        Path("e.js"),
    ]
    stats = detect_languages(paths)
    lang_map = {s.language: s for s in stats}
    assert lang_map["python"].file_count == 3
    assert lang_map["python"].percentage == 60.0
    assert lang_map["typescript"].file_count == 1
    assert lang_map["javascript"].file_count == 1


def test_detect_languages_empty() -> None:
    stats = detect_languages([])
    assert stats == []


def test_detect_languages_ignores_unknown() -> None:
    paths = [Path("a.py"), Path("b.unknown")]
    stats = detect_languages(paths)
    assert len(stats) == 1
    assert stats[0].language == "python"
    assert stats[0].percentage == 100.0


def test_sorted_by_count() -> None:
    paths = [Path("a.py"), Path("b.ts"), Path("c.ts"), Path("d.ts")]
    stats = detect_languages(paths)
    assert stats[0].language == "typescript"
    assert stats[1].language == "python"
