"""Require APA 7th doctoring for the crossed ``u_h`` estimator."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DOC = _ROOT / "docs" / "doctoring" / "multilevel_crossed_person_effects.md"


def test_crossed_uh_doctoring_cites_fox_glas_and_browne_mmmc() -> None:
    """The estimator doctoring note must cite both primary papers in APA 7th."""
    note = _DOC.read_text(encoding="utf-8")
    assert "Fox, J.-P., & Glas, C. A. W. (2001)" in note
    assert "Browne, W. J., Goldstein, H., & Rasbash, J. (2001)" in note
    assert "https://doi.org/10.1007/BF02294839" in note
    assert "https://doi.org/10.1177/1471082X0100100202" in note
    assert "Multiple membership" in note
    assert "multiple classification (MMMC) models" in note
    assert "does not estimate OLS" in note
