"""Require the living architecture baseline document at repository root."""

import re
from pathlib import Path
from urllib.parse import urlsplit


_ROOT = Path(__file__).resolve().parents[1]
_ARCHITECTURE = _ROOT / "ARCHITECTURE.md"


def test_architecture_md_exists_with_required_sections() -> None:
    """Buyers and agents must find architecture, recovery, and citation sections."""
    text = _ARCHITECTURE.read_text(encoding="utf-8")
    required = [
        "# Architecture — fast-mlsirm",
        "Rust numeric core",
        "Recovery evidence path",
        "multilevel",
        "APA 7th",
        "python/fast_mlsirm",
        "crates/mlsirm-core",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"ARCHITECTURE.md missing sections: {missing}"


def test_architecture_doctoring_note_exists() -> None:
    """Doctoring note must cite multilevel / LSIRM literature in APA 7th form."""
    note = (_ROOT / "docs" / "doctoring" / "architecture_baseline.md").read_text(
        encoding="utf-8"
    )
    assert "Fox" in note
    assert "Jeon" in note
    assert "Kang" in note
    citations = [urlsplit(url) for url in re.findall(r"https://[^\s)>\]]+", note)]
    assert any(url.scheme == "https" and url.hostname == "doi.org"
               and url.path.startswith("/10.") for url in citations)
