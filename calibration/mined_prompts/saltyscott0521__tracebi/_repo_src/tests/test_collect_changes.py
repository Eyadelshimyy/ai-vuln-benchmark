"""Changelog fragments fold into [Unreleased] without a shared-file conflict."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_PATH = Path(__file__).parent.parent / "scripts" / "collect_changes.py"
_spec = spec_from_file_location("collect_changes", _PATH)
collect_changes = module_from_spec(_spec)
_spec.loader.exec_module(collect_changes)


def test_two_fragments_land_in_issue_order(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Added — already here\n\n- old\n",
        encoding="utf-8",
    )
    changes = tmp_path / "changes"
    changes.mkdir()
    (changes / "12-later.md").write_text(
        "### Fixed — later\n\n- second\n", encoding="utf-8")
    (changes / "3-earlier.md").write_text(
        "### Added — earlier\n\n- first\n", encoding="utf-8")
    (changes / "README.md").write_text("leave me\n", encoding="utf-8")

    folded = collect_changes.collect(tmp_path)

    text = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.index("## [Unreleased]") < text.index("### Added — earlier")
    assert text.index("### Added — earlier") < text.index("### Fixed — later")
    assert text.index("### Fixed — later") < text.index("### Added — already here")
    assert [path.name for path in folded] == ["3-earlier.md", "12-later.md"]
    assert not (changes / "3-earlier.md").exists()
    assert not (changes / "12-later.md").exists()
    assert (changes / "README.md").read_text(encoding="utf-8") == "leave me\n"


def test_a_fragment_with_no_issue_number_is_folded_too(tmp_path):
    """An unnumbered fragment used to be skipped silently, so its change never
    reached a release. It folds after the numbered ones."""
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n", encoding="utf-8")
    changes = tmp_path / "changes"
    changes.mkdir()
    (changes / "7-numbered.md").write_text("### Added — numbered\n", encoding="utf-8")
    (changes / "no-issue.md").write_text("### Added — no issue\n", encoding="utf-8")
    (changes / "README.md").write_text("leave me\n", encoding="utf-8")

    folded = collect_changes.collect(tmp_path)

    text = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert [p.name for p in folded] == ["7-numbered.md", "no-issue.md"]
    assert text.index("— numbered") < text.index("— no issue")
    assert (changes / "README.md").exists()
