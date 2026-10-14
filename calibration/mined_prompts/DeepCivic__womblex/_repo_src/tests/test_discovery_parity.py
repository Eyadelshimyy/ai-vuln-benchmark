"""Local and cloud enumerate a corpus by one rule (P11).

A run ingests the documents directly under the location it is given. A corpus
with documents in subdirectories is refused by every entry point, so the two
paths cannot disagree about what a location yields.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from womblex.cli._shared import (
    NestedCorpusError,
    discover_files,
    normalise_prefix,
    select_supported,
)


def _make(root: Path, *relpaths: str) -> None:
    for rel in relpaths:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"%PDF-1.4\n")


class TestSelectSupported:
    def test_flat_corpus_returns_documents_sorted(self):
        got = select_supported(["b.pdf", "a.pdf", "c.docx"], location="/corpus")
        assert got == ["a.pdf", "b.pdf", "c.docx"]

    def test_unsupported_extensions_ignored(self):
        assert select_supported(["a.pdf", "notes.txt", "x.json"], location="/c") == ["a.pdf"]

    def test_nested_documents_refused(self):
        with pytest.raises(NestedCorpusError) as e:
            select_supported(["2026-08/a.pdf", "2026-08/b.pdf"], location="/corpus")
        assert e.value.nested == {"2026-08": 2}
        assert "2026-08/ (2 documents)" in str(e.value)

    def test_mixed_layout_refused_not_partially_ingested(self):
        # A top level that yields documents does not excuse the nested ones:
        # ingesting only the top level is a partial ingest.
        with pytest.raises(NestedCorpusError):
            select_supported(["top.pdf", "agency/a.pdf"], location="/corpus")

    def test_subdirectory_without_documents_is_not_a_refusal(self):
        got = select_supported(
            ["a.pdf", ".git/config", "notes/readme.txt", "data/rows.json"],
            location="/corpus",
        )
        assert got == ["a.pdf"]

    @pytest.mark.parametrize(
        "name", ["scan.png", "scan.JPG", "scan.jpeg", "scan.tif", "scan.tiff", "scan.bmp", "scan.gif", "scan.jp2"],
    )
    def test_standalone_images_are_supported(self, name):
        assert select_supported([name, "notes.txt"], location="/c") == [name]

    def test_nested_image_is_refused_like_any_document(self):
        with pytest.raises(NestedCorpusError):
            select_supported(["images/logo.png"], location="/corpus")

    def test_message_names_a_subdirectory_and_its_count(self):
        with pytest.raises(NestedCorpusError) as e:
            select_supported(["agency-a/one.pdf"], location="/corpus")
        msg = str(e.value)
        assert "/corpus" in msg
        assert "agency-a/ (1 document)" in msg

    def test_error_is_a_value_error_for_the_console_route(self):
        # ui/execute raises ValueError for bad input, which the route maps to
        # 400; the refusal travels the same way without a route change.
        assert issubclass(NestedCorpusError, ValueError)


class TestNormalisePrefix:
    """One key scope per location, for the CLI enqueue and the console alike.

    A spelling that survives normalisation is published provenance: the keys
    under the prefix are what the queue carries and what is recorded as each
    document's `source_relpath`.
    """

    @pytest.mark.parametrize(
        "spelling", ["2026-08", "./2026-08", "2026-08/", "/2026-08", "2026-08//", " 2026-08 "],
    )
    def test_every_spelling_of_one_location_yields_one_scope(self, spelling):
        assert normalise_prefix(spelling) == "2026-08"

    def test_an_inner_dot_segment_is_collapsed(self):
        assert normalise_prefix("2026-08/./health") == "2026-08/health"

    def test_omitted_or_empty_means_the_whole_root(self):
        assert normalise_prefix(None) == normalise_prefix("") == normalise_prefix("./") == ""

    @pytest.mark.parametrize(
        "escape", ["../elsewhere", "2026-08/../../etc", "..", "s3://other-bucket", "a\\b"],
    )
    def test_a_prefix_that_leaves_the_root_is_refused_not_sanitised(self, escape):
        with pytest.raises(ValueError, match="unsafe input_prefix"):
            normalise_prefix(escape)

    def test_a_literal_dotted_name_is_not_an_escape(self):
        # `....` and `..foo` are directory names, not traversal.
        assert normalise_prefix("..../elsewhere") == "..../elsewhere"


class TestDiscoverFiles:
    def test_flat_corpus_ingests_in_full(self, tmp_path):
        _make(tmp_path, "a.pdf", "b.pdf", "c.docx")
        assert [p.name for p in discover_files(tmp_path)] == ["a.pdf", "b.pdf", "c.docx"]

    def test_nested_corpus_refused_rather_than_reported_empty(self, tmp_path):
        _make(tmp_path, "2026-08/a.pdf")
        with pytest.raises(NestedCorpusError):
            discover_files(tmp_path)

    def test_limit_and_skip_still_apply(self, tmp_path):
        _make(tmp_path, "a.pdf", "b.pdf", "c.pdf", "d.pdf")
        assert [p.name for p in discover_files(tmp_path, limit=2)] == ["a.pdf", "b.pdf"]
        assert [p.name for p in discover_files(tmp_path, skip=2)] == ["c.pdf", "d.pdf"]

    def test_incidental_directories_do_not_refuse(self, tmp_path):
        _make(tmp_path, "a.pdf")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("x")
        assert [p.name for p in discover_files(tmp_path)] == ["a.pdf"]


class TestLocalAndCloudAgree:
    """The two paths reach the same verdict on the same layout."""

    @pytest.mark.parametrize(
        "layout,nested",
        [
            (["a.pdf", "b.pdf"], False),
            (["2026-08/a.pdf"], True),
            (["top.pdf", "agency/a.pdf"], True),
            (["a.pdf", "notes/readme.txt"], False),
        ],
    )
    def test_same_verdict_from_a_path_walk_and_a_key_listing(self, tmp_path, layout, nested):
        _make(tmp_path, *layout)

        def local() -> list[str] | str:
            try:
                return [p.name for p in discover_files(tmp_path)]
            except NestedCorpusError:
                return "refused"

        def cloud() -> list[str] | str:
            # What the enqueue path holds: a recursive, store-relative listing.
            keys = sorted(
                p.relative_to(tmp_path).as_posix()
                for p in tmp_path.rglob("*")
                if p.is_file()
            )
            try:
                return select_supported(keys, location=str(tmp_path))
            except NestedCorpusError:
                return "refused"

        assert local() == cloud()
        assert (local() == "refused") is nested
