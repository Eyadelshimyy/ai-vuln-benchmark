from pathlib import Path

from test_architecture_boundaries import find_violations


def test_new_field_note_domain_and_application_modules_respect_boundaries() -> None:
    root = Path(__file__).parents[1] / "src" / "gvas"
    for layer, names in {
        "domain": ("field_notes.py", "field_note_repositories.py"),
        "application": ("field_notes.py", "field_note_transcription.py"),
    }.items():
        for name in names:
            path = root / layer / name
            assert not find_violations(layer, path, path.read_text())
