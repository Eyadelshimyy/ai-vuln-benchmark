"""A subset font that never explains one of its glyphs.

Word embeds one subset font per style and sometimes leaves the ligature out of
/ToUnicode, so pypdf falls back to the raw byte and 'instituciones' reads
'insVtuciones'. The same font still writes a genuine 'V', so the fix has to happen on
the font, not on the text. The PDFs are built here; the data is invented.
"""

from __future__ import annotations

from pathlib import Path

from jobbot.profile.importer_pdf import extract_pdf_text
from tests.fixtures.cv_pdf import SubsetFont, write_subset_font_cv

# The byte pypdf shows when a glyph has no Unicode, and a code that means a real 'V'.
LIGATURE_BYTE = "V"
GENUINE_V_BYTE = "~"

# Calibri's own advances: 't' plus 'i' is 564, 'f' plus 'i' and 'f' plus 'l' are 534.
WIDTHS = {"t": 335, "i": 229, "f": 305, "l": 229}
TI_LIGATURE_WIDTH = 557
FI_OR_FL_WIDTH = 531


def subset(name: str, base_font: str, drawn: str, hole_width: int) -> SubsetFont:
    """A subset covering exactly the bytes drawn, minus the glyph left unexplained."""
    letters = {ch for ch in drawn if ch not in {LIGATURE_BYTE, GENUINE_V_BYTE}}
    widths = {ord(ch): WIDTHS.get(ch, 500) for ch in letters}
    widths[ord(LIGATURE_BYTE)] = hole_width
    widths[ord(GENUINE_V_BYTE)] = 500
    return SubsetFont(
        name=name,
        base_font=base_font,
        widths=widths,
        to_unicode={ord(ch): ch for ch in letters} | {ord(GENUINE_V_BYTE): "V"},
    )


def test_a_ligature_missing_from_the_font_map_is_read_from_its_width(tmp_path: Path) -> None:
    """No Unicode for the 'ti' glyph, but it is as wide as 't' plus 'i' and nothing else."""
    drawn = "Experiencia en insVtuciones de salud y ~igilancia de brotes."
    pdf = write_subset_font_cv(
        tmp_path / "hole.pdf",
        [subset("F1", "AAAAAA+Fixture", drawn, TI_LIGATURE_WIDTH)],
        [("F1", drawn)],
    )

    text, warnings = extract_pdf_text(pdf)

    assert "instituciones" in text
    # The genuine 'V' of the same font must survive the repair.
    assert "Vigilancia" in text
    assert warnings == []


def test_a_subset_too_small_to_prove_the_width_follows_its_sibling(tmp_path: Path) -> None:
    """A bold run of three words carries no 't', so only the other subset can name it."""
    body = "Experiencia en insVtuciones de salud."
    heading = "Universidad del Sur, SanVago."
    pdf = write_subset_font_cv(
        tmp_path / "sibling.pdf",
        [
            subset("F1", "AAAAAA+Fixture", body, TI_LIGATURE_WIDTH),
            subset("F2", "AAAAAB+Fixture", heading, TI_LIGATURE_WIDTH),
        ],
        [("F1", body), ("F2", heading)],
    )

    text, warnings = extract_pdf_text(pdf)

    assert "instituciones" in text
    assert "Santiago" in text
    assert warnings == []


def test_a_width_that_fits_two_ligatures_is_reported_instead_of_guessed(tmp_path: Path) -> None:
    """'fi' and 'fl' are the same width, so the glyph stays unknown and is flagged."""
    drawn = "Perfil: insVtuciones de salud."
    pdf = write_subset_font_cv(
        tmp_path / "ambiguous.pdf",
        [subset("F1", "AAAAAA+Fixture", drawn, FI_OR_FL_WIDTH)],
        [("F1", drawn)],
    )

    text, warnings = extract_pdf_text(pdf)

    assert "insVtuciones" in text
    assert any("ligature belongs" in warning for warning in warnings)
