"""Contract tests for authoritative unreleased changelog fragments."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "render_changelog_fragments.py"
RELEASED_CHANGELOG = ROOT / "CHANGELOG.md"


def _module():
    """Load the repository script without making ``scripts`` a package."""
    spec = importlib.util.spec_from_file_location("render_changelog_fragments", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fragment(path: Path, note: str = "Alpha.") -> Path:
    """Write one valid deterministic test fragment."""
    path.write_text(f"# First feature\n\n## Added\n\n- {note}\n", encoding="utf-8")
    return path


def _changelog() -> str:
    """Return a changelog with manual Unreleased notes and immutable history."""
    return (
        "# Changelog\n\n"
        "## Unreleased\n\n"
        "### Changed\n\n"
        "- Manual note.\n\n"
        "## [1.0.0] - 2026-08-01\n\n"
        "### Added\n\n"
        "- Historical note.\n"
    )


def test_fragment_renderer_is_deterministic_and_preserves_release_sections(tmp_path):
    """Stable rendering groups fragments under the canonical Unreleased sections."""
    first = _fragment(tmp_path / "100-first.md")
    second = tmp_path / "200-second.md"
    second.write_text("# Second fix\n\n## Fixed\n\n- Beta.\n", encoding="utf-8")
    module = _module()
    rendered = module.render_unreleased((first, second))
    assert rendered == module.render_unreleased((first, second))
    assert rendered.startswith("## Unreleased\n")
    assert "### Added\n\n#### First feature\n\n- Alpha." in rendered
    assert "### Fixed\n\n#### Second fix\n\n- Beta." in rendered


def test_rubric_slice_is_release_noted_in_the_published_changelog():
    """The public rubric slice is release-noted now rather than deferred silently."""
    changelog = RELEASED_CHANGELOG.read_text(encoding="utf-8")
    release_start = changelog.index("## [0.2.0] - 2026-08-03")
    release_section = changelog[release_start : changelog.index("## [0.1.2]")]
    assert "#### Rubric blueprint compiler" in release_section
    assert "SHA-256" in release_section
    assert "JSON Schema 2020-12" in release_section
    assert "will be folded" not in release_section


def test_every_repository_fragment_matches_the_authoritative_format():
    """A release cannot silently omit a malformed or unclassified fragment."""
    module = _module()
    paths = module.fragment_paths()
    assert paths
    rendered = module.render_unreleased(paths)
    for path in paths:
        title, _ = module.parse_fragment(path)
        assert f"#### {title}" in rendered


def test_fragment_parser_allows_issue_reference_text_starting_with_hash(tmp_path):
    """An issue reference at line start is body text, not an ATX heading."""
    module = _module()
    fragment = tmp_path / "100-issue-reference.md"
    fragment.write_text(
        "# First feature\n\n## Fixed\n\n#394 fixed the contract.\n",
        encoding="utf-8",
    )
    _, sections = module.parse_fragment(fragment)
    assert sections["Fixed"] == ("#394 fixed the contract.",)


def test_changelog_check_update_round_trip_preserves_manual_notes_and_history(
    tmp_path,
):
    """Update only the managed block; check fails closed before and after drift."""
    module = _module()
    changelog = tmp_path / "CHANGELOG.md"
    fragment = _fragment(tmp_path / "100-first.md")
    changelog.write_text(_changelog(), encoding="utf-8")
    historical = _changelog().split("## [1.0.0]", 1)[1]

    with pytest.raises(ValueError, match="stale"):
        module.check_changelog(changelog, (fragment,))
    module.update_changelog(changelog, (fragment,))
    module.check_changelog(changelog, (fragment,))

    updated = changelog.read_text(encoding="utf-8")
    assert "- Manual note." in updated
    assert module.BEGIN_MARKER in updated
    assert "#### First feature\n\n- Alpha." in updated
    assert f"{module.END_MARKER}\n\n## [1.0.0]" in updated
    assert updated.split("## [1.0.0]", 1)[1] == historical

    _fragment(fragment, "Changed fragment.")
    with pytest.raises(ValueError, match="stale"):
        module.check_changelog(changelog, (fragment,))
    module.update_changelog(changelog, (fragment,))
    module.check_changelog(changelog, (fragment,))
    rewritten = changelog.read_text(encoding="utf-8")
    assert "Changed fragment." in rewritten
    assert "- Alpha." not in rewritten
    assert f"{module.END_MARKER}\n\n## [1.0.0]" in rewritten


def test_changelog_sync_rejects_ambiguous_headings_and_markers(tmp_path):
    """Duplicate headings, incomplete markers, and out-of-section markers fail."""
    module = _module()
    fragment = _fragment(tmp_path / "100-first.md")
    rendered = module.render_unreleased((fragment,))

    with pytest.raises(ValueError, match="exactly one"):
        module.synchronize_text("# Changelog\n", rendered)
    with pytest.raises(ValueError, match="exactly one"):
        module.synchronize_text(
            "# Changelog\n\n## Unreleased\n\n## Unreleased\n", rendered
        )
    with pytest.raises(ValueError, match="marker pair"):
        module.synchronize_text(
            f"# Changelog\n\n## Unreleased\n\n{module.BEGIN_MARKER}\n", rendered
        )
    with pytest.raises(ValueError, match="only inside"):
        module.synchronize_text(
            f"# Changelog\n{module.BEGIN_MARKER}\n\n## Unreleased\n", rendered
        )


@pytest.mark.parametrize(
    "fragment_text",
    [
        "# <!-- BEGIN AUTHORITATIVE CHANGELOG FRAGMENTS -->\n\n## Fixed\n\n- Note.\n",
        "# First feature\n\n## Fixed\n\n- <!-- END AUTHORITATIVE CHANGELOG FRAGMENTS -->\n",
    ],
)
def test_fragment_markers_are_rejected_before_changelog_update(
    tmp_path, fragment_text: str
) -> None:
    """Reserved marker content must fail closed before the target file changes."""
    module = _module()
    changelog = tmp_path / "CHANGELOG.md"
    fragment = tmp_path / "100-marker.md"
    changelog.write_text(_changelog(), encoding="utf-8")
    fragment.write_text(fragment_text, encoding="utf-8")
    before = changelog.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="reserved fragment marker"):
        module.update_changelog(changelog, (fragment,))

    assert changelog.read_text(encoding="utf-8") == before


def test_cli_check_and_update_modes_are_fail_closed(tmp_path, capsys):
    """The release command returns failure on drift and success after update."""
    module = _module()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(_changelog(), encoding="utf-8")

    with pytest.raises(SystemExit) as failure:
        module.main(["--check", str(changelog)])
    assert failure.value.code == 1
    assert "stale" in capsys.readouterr().err

    assert module.main(["--update", str(changelog)]) == 0
    assert module.main(["--check", str(changelog)]) == 0


def test_render_contract_rejects_empty_and_malformed_fragments(tmp_path):
    """Malformed or empty fragment inventories cannot produce release evidence."""
    module = _module()
    with pytest.raises(ValueError, match="at least one"):
        module.render_unreleased(())

    malformed = tmp_path / "bad.md"
    malformed.write_text("not a title\n", encoding="utf-8")
    with pytest.raises(ValueError, match="level-one title"):
        module.parse_fragment(malformed)

    unsupported = tmp_path / "unsupported.md"
    unsupported.write_text("# Bad\n\n## Other\n\n- Note.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported section"):
        module.parse_fragment(unsupported)
