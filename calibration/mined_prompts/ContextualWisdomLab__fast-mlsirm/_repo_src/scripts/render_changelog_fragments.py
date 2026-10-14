#!/usr/bin/env python3
"""Render and synchronize authoritative unreleased changelog fragments."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_DIR = ROOT / "docs" / "changelog.d"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
BEGIN_MARKER = "<!-- BEGIN AUTHORITATIVE CHANGELOG FRAGMENTS -->"
END_MARKER = "<!-- END AUTHORITATIVE CHANGELOG FRAGMENTS -->"
_ALLOWED_SECTIONS = (
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
)
_UNRELEASED_HEADING = re.compile(r"^## Unreleased\s*$", re.MULTILINE)
_NEXT_RELEASE_HEADING = re.compile(r"^## (?!Unreleased\s*$).+$", re.MULTILINE)
_ATX_HEADING = re.compile(r"^#{1,6}(?:\s|$)")


def fragment_paths(directory: Path = FRAGMENT_DIR) -> tuple[Path, ...]:
    """Return every authoritative Markdown fragment in stable path order."""
    return tuple(
        path
        for path in sorted(directory.glob("*.md"))
        if path.name.casefold() != "readme.md"
    )


def parse_fragment(path: Path) -> tuple[str, dict[str, tuple[str, ...]]]:
    """Parse one fragment into a title and validated release-note sections."""
    content = path.read_text(encoding="utf-8")
    if BEGIN_MARKER in content or END_MARKER in content:
        raise ValueError(f"{path}: fragment contains reserved fragment marker")
    lines = content.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{path}: first line must be a level-one title")
    title = lines[0][2:].strip()
    if not title:
        raise ValueError(f"{path}: title must not be empty")

    sections: dict[str, list[str]] = defaultdict(list)
    active: str | None = None
    for line in lines[1:]:
        if line.startswith("## "):
            active = line[3:].strip()
            if active not in _ALLOWED_SECTIONS:
                raise ValueError(f"{path}: unsupported section {active!r}")
            continue
        if _ATX_HEADING.match(line):
            raise ValueError(f"{path}: unsupported heading depth")
        if line.strip():
            if active is None:
                raise ValueError(f"{path}: content must follow a release section")
            sections[active].append(line.rstrip())

    if not sections:
        raise ValueError(f"{path}: at least one release section is required")
    return title, {name: tuple(values) for name, values in sections.items()}


def render_unreleased(paths: Iterable[Path] | None = None) -> str:
    """Render all fragments as one deterministic ``Unreleased`` Markdown block."""
    selected = fragment_paths() if paths is None else tuple(paths)
    if not selected:
        raise ValueError("at least one changelog fragment is required")

    grouped: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    for path in selected:
        title, sections = parse_fragment(path)
        for section, lines in sections.items():
            grouped[section].append((title, lines))

    output = ["## Unreleased", ""]
    for section in _ALLOWED_SECTIONS:
        entries = grouped.get(section)
        if not entries:
            continue
        output.extend((f"### {section}", ""))
        for title, lines in entries:
            output.extend((f"#### {title}", "", *lines, ""))
    return "\n".join(output).rstrip() + "\n"


def _managed_block(rendered: str) -> str:
    """Return the marker-delimited fragment body embedded under Unreleased."""
    heading, separator, body = rendered.partition("\n\n")
    if heading != "## Unreleased" or not separator or not body.strip():
        raise ValueError("rendered changelog must contain one non-empty Unreleased block")
    return f"{BEGIN_MARKER}\n{body.rstrip()}\n{END_MARKER}\n"


def synchronize_text(changelog: str, rendered: str) -> str:
    """Return changelog text with exactly one managed Unreleased fragment block.

    Existing manually maintained Unreleased notes and every historical release
    remain byte-for-byte unchanged. The marker-delimited fragment block is
    inserted or replaced only inside the single ``## Unreleased`` section.
    """
    headings = tuple(_UNRELEASED_HEADING.finditer(changelog))
    if len(headings) != 1:
        raise ValueError("CHANGELOG.md must contain exactly one ## Unreleased heading")
    heading = headings[0]
    next_release = _NEXT_RELEASE_HEADING.search(changelog, heading.end())
    section_end = next_release.start() if next_release is not None else len(changelog)
    section = changelog[heading.end() : section_end]

    begin_count = section.count(BEGIN_MARKER)
    end_count = section.count(END_MARKER)
    if begin_count != end_count or begin_count > 1:
        raise ValueError("Unreleased must contain zero or one complete fragment marker pair")
    if changelog[: heading.end()].count(BEGIN_MARKER) or changelog[section_end:].count(
        BEGIN_MARKER
    ):
        raise ValueError("fragment markers must occur only inside Unreleased")
    if changelog[: heading.end()].count(END_MARKER) or changelog[section_end:].count(
        END_MARKER
    ):
        raise ValueError("fragment markers must occur only inside Unreleased")

    managed = _managed_block(rendered)
    if begin_count == 1:
        begin = section.index(BEGIN_MARKER)
        end = section.index(END_MARKER, begin) + len(END_MARKER)
        replacement = section[:begin] + managed.rstrip("\n") + section[end:]
    else:
        prefix = section.rstrip()
        trailing = section[len(prefix) :]
        if not trailing:
            trailing = "\n\n" if next_release is not None else "\n"
        body = managed.rstrip("\n") + trailing
        replacement = f"{prefix}\n\n{body}" if prefix else f"\n\n{body}"

    return changelog[: heading.end()] + replacement + changelog[section_end:]


def check_changelog(
    path: Path = CHANGELOG_PATH, paths: Iterable[Path] | None = None
) -> None:
    """Fail when the managed fragment block is absent or differs from fragments."""
    current = path.read_text(encoding="utf-8")
    expected = synchronize_text(current, render_unreleased(paths))
    if current != expected:
        raise ValueError(
            f"{path}: authoritative changelog fragments are stale; run "
            f"{Path(__file__).name} --update {path}"
        )


def update_changelog(
    path: Path = CHANGELOG_PATH, paths: Iterable[Path] | None = None
) -> None:
    """Write the deterministic managed fragment block into ``path``."""
    current = path.read_text(encoding="utf-8")
    updated = synchronize_text(current, render_unreleased(paths))
    path.write_text(updated, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Render fragments or check/update a changelog file deterministically."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--output", type=Path, help="write rendered Markdown here")
    actions.add_argument(
        "--check",
        type=Path,
        metavar="CHANGELOG",
        help="fail when CHANGELOG does not match authoritative fragments",
    )
    actions.add_argument(
        "--update",
        type=Path,
        metavar="CHANGELOG",
        help="synchronize authoritative fragments into CHANGELOG",
    )
    args = parser.parse_args(argv)
    try:
        if args.check is not None:
            check_changelog(args.check)
        elif args.update is not None:
            update_changelog(args.update)
        else:
            rendered = render_unreleased()
            if args.output is None:
                sys.stdout.write(rendered)
            else:
                args.output.write_text(rendered, encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
