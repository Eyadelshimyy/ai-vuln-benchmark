"""Require the living governance index and its doctoring note."""

import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_INDEX = _ROOT / "docs" / "GOVERNANCE_INDEX.md"
_DOC = _ROOT / "docs" / "doctoring" / "governance_index.md"


def test_governance_index_has_required_sections() -> None:
    """Buyers must find ADR, threat model, test strategy, and multilevel links."""
    text = _INDEX.read_text(encoding="utf-8")
    required = [
        "# Governance index",
        "ADR index",
        "Threat model",
        "Test strategy",
        "Operability",
        "Traceability",
        "multilevel",
        "APA 7th",
        "ARCHITECTURE.md",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing sections: {missing}"


def test_governance_doctoring_cites_multilevel_literature() -> None:
    """Doctoring note must cite multilevel / LSIRM literature."""
    note = _DOC.read_text(encoding="utf-8")
    assert "Fox" in note and "Jeon" in note and "Kang" in note
    citations = [urlsplit(url) for url in re.findall(r"https://[^\s)>\]]+", note)]
    assert any(url.scheme == "https" and url.hostname == "doi.org"
               and url.path.startswith("/10.") for url in citations)


@pytest.mark.parametrize("url", [
    "https://example.invalid/https://doi.org/10.1007/example",
    "https://doi.org.example.invalid/10.1007/example",
    "https://doi.org@example.invalid/10.1007/example",
])
def test_governance_doctoring_rejects_misleading_doi_hosts(tmp_path, monkeypatch, url):
    document = tmp_path / "citation.md"
    document.write_text(f"Fox Jeon Kang {url}", encoding="utf-8")
    monkeypatch.setitem(globals(), "_DOC", document)
    with pytest.raises(AssertionError):
        test_governance_doctoring_cites_multilevel_literature()
