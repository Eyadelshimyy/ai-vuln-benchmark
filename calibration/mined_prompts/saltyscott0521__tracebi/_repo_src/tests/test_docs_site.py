"""
The static docs site under ``site/docs/`` is generated, and stays generated.

``site/`` deploys to any static host with no build step — that is what makes
the marketing site trivial to host — so the rendered HTML is committed rather
than built at deploy time. The cost of committing generated output is drift:
someone edits ``docs/*.md``, the markdown and the published page disagree, and
nothing says so.

These tests are what says so. The first re-runs the generator into a temp
directory and compares it byte-for-byte with what is committed; editing the
vault without running ``python site/build_docs.py`` fails here. The rest pin
the properties the generator exists to provide.
"""

from __future__ import annotations

import filecmp
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SITE_DOCS = REPO / "site" / "docs"
BUILDER = REPO / "site" / "build_docs.py"

pytestmark = pytest.mark.skipif(
    not BUILDER.is_file(), reason="the docs-site generator is not present"
)


def _built(tmp_path: Path) -> Path:
    """Run the generator against a copy of site/, return its docs output."""
    pytest.importorskip("markdown")
    import shutil

    site = tmp_path / "site"
    site.mkdir()
    shutil.copy(REPO / "site" / "index.html", site / "index.html")
    shutil.copy(BUILDER, site / "build_docs.py")
    # The generator resolves docs/ as a sibling of site/, so link the real one.
    (tmp_path / "docs").symlink_to(REPO / "docs", target_is_directory=True)

    r = subprocess.run([sys.executable, str(site / "build_docs.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return site / "docs"


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


class TestGeneratedSiteIsCurrent:
    def test_regenerating_produces_no_diff(self, tmp_path):
        """Edit docs/*.md without re-running the generator and this fails.

        The fix is always the same: ``python site/build_docs.py``.
        """
        fresh = _built(tmp_path)
        committed, generated = _tree(SITE_DOCS), _tree(fresh)

        missing = sorted(generated - committed)
        extra = sorted(committed - generated)
        assert not missing, (
            f"site/docs is missing {missing} — run: python site/build_docs.py")
        assert not extra, (
            f"site/docs has stale pages {extra} — run: python site/build_docs.py")

        differing = [
            rel for rel in sorted(generated)
            if not filecmp.cmp(fresh / rel, SITE_DOCS / rel, shallow=False)
        ]
        assert not differing, (
            f"site/docs is out of date with docs/*.md in {differing} — "
            f"run: python site/build_docs.py"
        )


class TestSiteProperties:
    def test_every_vault_page_is_published_except_internal_ones(self):
        """A page in the vault that never reaches the site is a page nobody
        outside the repo can read."""
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location("build_docs", BUILDER)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)

        published = {p.stem for p in SITE_DOCS.rglob("*.html")}
        for md in (REPO / "docs").rglob("*.md"):
            if md.stem in mod.HIDDEN:
                assert md.stem not in published, (
                    f"{md.stem} is an internal document and must not publish")
                continue
            rel = md.relative_to(REPO / "docs")
            if len(rel.parts) > 1 and rel.parts[0] not in {k for k, _, _ in mod.SECTIONS}:
                continue                      # a directory the site does not carry
            assert md.stem in published, (
                f"docs/{rel} is not published — add its section to "
                f"build_docs.SECTIONS, or run the generator")

    def test_the_positioning_doc_never_ships(self):
        """north-star.md is gitignored and internal. If it ever reaches the
        public site it is a leak, not a formatting problem."""
        assert not (SITE_DOCS / "north-star.html").exists()
        for page in SITE_DOCS.rglob("*.html"):
            assert "north-star" not in page.read_text(encoding="utf-8")

    def test_pages_carry_no_unrendered_wiki_links(self):
        """A literal [[link]] on the published page means the rewrite missed
        it — the reader gets markup instead of a link."""
        offenders = [
            str(p.relative_to(SITE_DOCS)) for p in SITE_DOCS.rglob("*.html")
            if "[[" in p.read_text(encoding="utf-8")
        ]
        assert not offenders, f"unrendered wiki links in {offenders}"
