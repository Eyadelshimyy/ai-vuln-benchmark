"""Tests for Markdown file extraction."""

from pathlib import Path

from womblex.ingest.detect import DetectionConfig, DocumentType, detect_file_type
from womblex.ingest.extract import extract_text


class TestMarkdownDetection:
    def test_md_detected_as_markdown(self, tmp_path: Path) -> None:
        md = tmp_path / "sample.md"
        md.write_text("# Title\n\nBody text.\n", encoding="utf-8")
        profile = detect_file_type(md, DetectionConfig())
        assert profile.doc_type == DocumentType.MARKDOWN
        assert profile.confidence == 1.0
        assert profile.page_count == 1

    def test_markdown_extension_also_detected(self, tmp_path: Path) -> None:
        md = tmp_path / "sample.markdown"
        md.write_text("# Title\n", encoding="utf-8")
        profile = detect_file_type(md, DetectionConfig())
        assert profile.doc_type == DocumentType.MARKDOWN

    def test_table_signal_detected(self, tmp_path: Path) -> None:
        md = tmp_path / "table.md"
        md.write_text("| A | B |\n| --- | --- |\n| 1 | 2 |\n", encoding="utf-8")
        profile = detect_file_type(md, DetectionConfig())
        assert profile.has_tables is True

    def test_non_md_not_affected(self, tmp_path: Path) -> None:
        txt = tmp_path / "data.txt"
        txt.write_text("hello", encoding="utf-8")
        profile = detect_file_type(txt, DetectionConfig())
        assert profile.doc_type != DocumentType.MARKDOWN


class TestMarkdownExtraction:
    def test_headings_and_paragraphs_in_order(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text(
            "# Title Heading\n\n"
            "Intro paragraph with **bold** text.\n\n"
            "## Section Two\n\n"
            "Final paragraph.\n",
            encoding="utf-8",
        )
        profile = detect_file_type(md)
        results = extract_text(md, profile)

        assert len(results) == 1
        r = results[0]
        assert r.error is None
        assert r.method == "markdown"
        kinds = [(e.kind, e.text) for e in r.elements]
        assert kinds == [
            ("heading", "Title Heading"),
            ("paragraph", "Intro paragraph with **bold** text."),
            ("heading", "Section Two"),
            ("paragraph", "Final paragraph."),
        ]

    def test_list_items_get_list_item_kind(self, tmp_path: Path) -> None:
        md = tmp_path / "list.md"
        md.write_text("- first item\n- second item\n", encoding="utf-8")
        profile = detect_file_type(md)
        results = extract_text(md, profile)

        r = results[0]
        assert [e.kind for e in r.elements] == ["list_item", "list_item"]
        assert [e.text for e in r.elements] == ["first item", "second item"]

    def test_code_fence_preserved_verbatim(self, tmp_path: Path) -> None:
        md = tmp_path / "code.md"
        md.write_text("```python\ndef f():\n    return 1\n```\n", encoding="utf-8")
        profile = detect_file_type(md)
        results = extract_text(md, profile)

        r = results[0]
        assert len(r.elements) == 1
        assert r.elements[0].kind == "paragraph"
        assert r.elements[0].text == "def f():\n    return 1"

    def test_gfm_table_extracted_with_header_row(self, tmp_path: Path) -> None:
        md = tmp_path / "table.md"
        md.write_text(
            "| Name | Amount |\n"
            "| --- | --- |\n"
            "| Alice | $10 |\n"
            "| Bob | $20 |\n",
            encoding="utf-8",
        )
        profile = detect_file_type(md)
        results = extract_text(md, profile)

        r = results[0]
        table_elements = [e for e in r.elements if e.kind == "table"]
        assert len(table_elements) == 1
        table = table_elements[0]
        assert table.header_rows == [0]
        assert table.cells is not None
        values = {(c.row, c.col): c.value for c in table.cells}
        assert values == {
            (0, 0): "Name", (0, 1): "Amount",
            (1, 0): "Alice", (1, 1): "$10",
            (2, 0): "Bob", (2, 1): "$20",
        }

    def test_table_interleaved_with_prose_in_document_order(self, tmp_path: Path) -> None:
        md = tmp_path / "mixed.md"
        md.write_text(
            "Before the table.\n\n"
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
            "After the table.\n",
            encoding="utf-8",
        )
        profile = detect_file_type(md)
        results = extract_text(md, profile)

        r = results[0]
        kinds = [e.kind for e in r.elements]
        assert kinds == ["paragraph", "table", "paragraph"]
        assert [e.order for e in r.elements] == [0, 1, 2]

    def test_empty_markdown_file(self, tmp_path: Path) -> None:
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")
        profile = detect_file_type(md)
        results = extract_text(md, profile)

        assert len(results) == 1
        assert results[0].error is None
        assert results[0].elements == []

    def test_utf8_bom_does_not_hide_first_heading(self, tmp_path: Path) -> None:
        md = tmp_path / "bom.md"
        md.write_bytes(b"\xef\xbb\xbf# Title\r\n\r\nBody\r\n")
        results = extract_text(md, detect_file_type(md))

        assert [(e.kind, e.text) for e in results[0].elements] == [
            ("heading", "Title"), ("paragraph", "Body"),
        ]

    def test_front_matter_kept_verbatim_not_read_as_heading(self, tmp_path: Path) -> None:
        md = tmp_path / "fm.md"
        md.write_text("---\ntitle: Report\n\nauthor: X\n---\n\n# Real Heading\n", encoding="utf-8")
        results = extract_text(md, detect_file_type(md))

        assert [(e.kind, e.text) for e in results[0].elements] == [
            ("paragraph", "---\ntitle: Report\n\nauthor: X\n---"),
            ("heading", "Real Heading"),
        ]

    def test_front_matter_at_end_of_file_without_newline(self, tmp_path: Path) -> None:
        md = tmp_path / "fm_only.md"
        md.write_text("---\ntitle: Only\n---", encoding="utf-8")
        results = extract_text(md, detect_file_type(md))

        assert [e.text for e in results[0].elements] == ["---\ntitle: Only\n---"]

    def test_thematic_break_mid_document_is_not_front_matter(self, tmp_path: Path) -> None:
        md = tmp_path / "rule.md"
        md.write_text("Intro\n\n---\n\nAfter\n", encoding="utf-8")
        results = extract_text(md, detect_file_type(md))

        assert [e.text for e in results[0].elements] == ["Intro", "After"]

    def test_link_reference_definitions_kept_in_place(self, tmp_path: Path) -> None:
        md = tmp_path / "refs.md"
        md.write_text("See [x].\n\n[x]: http://example.com\n\nTail.\n", encoding="utf-8")
        results = extract_text(md, detect_file_type(md))

        r = results[0]
        assert [e.text for e in r.elements] == ["See [x].", "[x]: http://example.com", "Tail."]
        assert [e.order for e in r.elements] == [0, 1, 2]

    def test_latin1_fallback(self, tmp_path: Path) -> None:
        md = tmp_path / "latin.md"
        md.write_bytes("# caf\xe9".encode("latin-1"))
        profile = detect_file_type(md)
        results = extract_text(md, profile)

        assert len(results) == 1
        assert "caf" in results[0].full_text
