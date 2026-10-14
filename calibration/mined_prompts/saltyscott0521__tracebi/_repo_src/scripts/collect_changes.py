"""Fold ``changes/<issue>-<name>.md`` into ``CHANGELOG.md``.

Run this when cutting a release, then commit the result. The release
workflow does not run it: that job cannot push, and folding only on the
runner would ship notes ``main`` does not have.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _fragments(changes: Path) -> list[Path]:
    if not changes.is_dir():
        return []
    # Numbered fragments (<issue>-<name>.md) first, in issue order; then the
    # ones with no issue, by name. Every fragment is folded: an unnumbered one
    # used to be skipped silently, and its change left out of the release.
    numbered, named = [], []
    for path in changes.glob("*.md"):
        if path.name == "README.md":
            continue
        issue, _, _ = path.name.partition("-")
        (numbered if issue.isdigit() else named).append(path)
    numbered.sort(key=lambda path: int(path.name.partition("-")[0]))
    named.sort(key=lambda path: path.name)
    return numbered + named


def collect(root: Path) -> list[Path]:
    """Insert every fragment under ``## [Unreleased]``, then delete them.

    Order is issue number, lowest first. ``changes/README.md`` is left
    in place. Returns the fragments that were folded.
    """
    changelog = root / "CHANGELOG.md"
    frags = _fragments(root / "changes")
    if not frags:
        return []
    text = changelog.read_text(encoding="utf-8")
    heading = "## [Unreleased]"
    at = text.find(heading)
    if at < 0:
        raise SystemExit("CHANGELOG.md has no ## [Unreleased] section")
    at += len(heading)
    if text[at:at + 1] == "\r":
        at += 1
    if text[at:at + 1] == "\n":
        at += 1
    body = "\n\n".join(path.read_text(encoding="utf-8").strip() for path in frags)
    rest = text[at:].lstrip("\n")
    changelog.write_text(
        text[:at] + "\n" + body + "\n\n" + rest, encoding="utf-8")
    for path in frags:
        path.unlink()
    return frags


def main() -> None:
    folded = collect(Path(__file__).resolve().parents[1])
    if not folded:
        print("no changelog fragments")
        return
    print(f"folded {len(folded)} fragment(s) into CHANGELOG.md")


if __name__ == "__main__":
    main()
    sys.exit(0)
