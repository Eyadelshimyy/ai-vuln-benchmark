"""The evidence reference: building it at each level and checking it at its own precision."""

from __future__ import annotations

import logging

import pyarrow as pa
import pytest

from tests._shard import DOC, NARRATIVE, contact_shard, table_markdown
from womblex.process.evidence import EvidenceIndexes, EvidenceReport, check_rows, stamp_evidence
from womblex.store.evidence import EVIDENCE_COLUMNS, backfill_evidence, evidence


@pytest.fixture
def index(tmp_path):
    idx = EvidenceIndexes(contact_shard(tmp_path)).get(DOC)
    assert idx is not None
    return idx


def test_a_narrative_span_names_the_element_it_starts_in(index):
    start = NARRATIVE.index("delegate")
    ref = index.narrative_ref(start, start + len("delegate"))

    assert ref is not None
    assert (ref["anchor_level"], ref["elem_order"], ref["page"], ref["text_layer"]) == (
        "span", 2, 1, "elements")
    assert index.holds(ref, "delegate") is True
    assert index.holds(ref, "Delegate") is False
    assert index.narrative_ref(0, len(NARRATIVE) + 1) is None


def test_table_and_cell_spans_resolve_to_their_text(tmp_path):
    base = contact_shard(tmp_path)
    index = EvidenceIndexes(base).get(DOC)
    markdown = table_markdown(base)
    assert index is not None

    table = index.resolve_table(elem_order=1)
    assert table is not None
    start = markdown.index("John Citizen")
    ref = index.table_ref(table, start, start + len("John Citizen"))
    assert ref is not None and (ref["elem_order"], ref["text_layer"]) == (1, "table_markdown")
    assert index.holds(ref, "John Citizen") is True

    cell = index.cell_ref(1, 0, 5, cell=(1, 1))
    assert cell is not None and (cell["cell_row"], cell["cell_col"], cell["text_layer"]) == (
        1, 1, "cell")
    assert index.holds(cell, "1,200") is True and index.holds(cell, "1,201") is False
    assert index.cell_ref(1, 0, 99, cell=(1, 1)) is None   # past the cell's value
    assert index.cell_ref(1, 0, 1, cell=(9, 9)) is None    # no such cell


def test_an_element_receipt_holds_when_the_text_lies_in_that_element_or_cell(index):
    paragraph = index.element_ref(2)
    cell = index.element_ref(1, cell=(2, 0))
    table = index.element_ref(1)

    assert paragraph is not None and cell is not None and table is not None
    assert paragraph["anchor_level"] == "element" and paragraph["char_start"] is None
    assert index.holds(paragraph, "the delegate") is True
    assert index.holds(paragraph, "Jane Doe") is False  # in another element
    assert index.holds(cell, "Mary Major") is True and index.holds(cell, "John Citizen") is False
    assert index.holds(table, "John Citizen") is True
    assert index.element_ref(1, cell=(9, 9)) is None


def test_a_document_receipt_holds_when_the_text_lies_anywhere_in_the_document(index):
    ref = index.document_ref()

    assert (ref["anchor_level"], ref["elem_order"], ref["char_start"]) == ("document", None, None)
    assert index.holds(ref, "Jane Doe") is True      # narrative
    assert index.holds(ref, "Ann Roe") is True       # table markdown
    assert index.holds(ref, "Nobody Here") is False


def test_a_chunk_receipt_cannot_be_checked_without_the_chunk(index):
    assert index.holds(evidence("chunk", page=1), "Jane Doe") is None


def test_a_receipt_under_another_text_layer_is_not_checked_against_this_one(index):
    ref = {**index.narrative_ref(0, 4), "text_layer": "spellfix"}

    assert index.holds(ref, index.narrative[:4]) is None
    assert index.holds({**index.document_ref(), "text_layer": "spellfix"}, "Jane Doe") is None


def test_a_document_without_elements_has_no_index(tmp_path):
    base = contact_shard(tmp_path)
    (tmp_path / "batch-0001.elements.parquet").unlink()

    assert EvidenceIndexes(base).get(DOC) is None


def test_a_reference_is_built_from_known_columns_and_levels():
    ref = evidence("span", char_start=1)
    assert set(ref) == set(EVIDENCE_COLUMNS) and ref["anchor_level"] == "span"
    with pytest.raises(ValueError):
        evidence("exact")
    with pytest.raises(ValueError):
        evidence("span", row=1)


def test_backfill_adds_the_columns_a_file_lacks_with_one_level():
    table = pa.table({"cell": pa.array([3], pa.int32())})

    out = backfill_evidence(table, {"cell_col": "cell"}, level="chunk")

    assert set(EVIDENCE_COLUMNS) <= set(out.schema.names)
    row = out.to_pylist()[0]
    assert (row["cell_col"], row["anchor_level"], row["char_start"]) == (3, "chunk", None)
    with pytest.raises(ValueError):
        backfill_evidence(table, level="exact")


# ---------------------------------------------------------------------------
# Checking rows at each receipt's own precision
# ---------------------------------------------------------------------------


def _row(text: str, ref: dict | None) -> dict:
    return {"source_hash": DOC, "text": text, **(ref or {})}


def _rows(tmp_path):
    base = contact_shard(tmp_path)
    indexes = EvidenceIndexes(base)
    index = indexes.get(DOC)
    assert index is not None
    start = NARRATIVE.index("Jane Doe")
    span = index.narrative_ref(start, start + 8)
    rows = [
        _row("Jane Doe", span),
        _row("the delegate", index.element_ref(2)),
        _row("Ann Roe", index.document_ref()),
    ]
    return base, indexes, rows


def test_receipts_at_every_level_but_chunk_hold_against_the_source(tmp_path):
    base, indexes, rows = _rows(tmp_path)

    report = stamp_evidence(rows, indexes.get, text_key="text", label="pii_spans", base=base)

    assert (report.checked, report.unchecked, report.ok) == (3, 0, True)
    assert {r["evidence_check"] for r in rows} == {"holds"}


def test_a_receipt_that_fails_its_own_check_is_written_labelled_fails(tmp_path, caplog):
    base, indexes, rows = _rows(tmp_path)
    rows[0]["text"] = "Jane Roe"          # span: slice is "Jane Doe"
    rows[1]["text"] = "Jane Doe"          # element 2 does not hold it
    rows[2]["text"] = "Nobody Here"       # not in the document

    with caplog.at_level(logging.WARNING):
        report = stamp_evidence(rows, indexes.get, text_key="text", label="pii_spans", base=base)

    assert (report.n_mismatched, report.ok) == (3, False)
    assert {r["evidence_check"] for r in rows} == {"fails"}
    assert "3 of 3 receipt(s) that fail" in caplog.text


def test_a_chunk_receipt_is_checked_against_the_chunk_text(tmp_path):
    _, indexes, _ = _rows(tmp_path)
    holds = _row("Jane Doe", evidence("chunk", page=1))
    fails = _row("John Citizen", evidence("chunk", page=1))

    report = check_rows(
        [holds, fails], indexes.get, text_key="text", label="pii_spans",
        chunk_text=lambda _row: "Contact Jane Doe about the grant.")

    assert (report.checked, report.n_mismatched) == (2, 1)


def test_a_receipt_whose_source_is_absent_is_written_unchecked_and_warned(tmp_path, caplog):
    base, _, rows = _rows(tmp_path)
    rows.append(_row("Jane Doe", evidence("chunk", page=1)))  # no chunk text to hand

    with caplog.at_level(logging.WARNING):
        report = stamp_evidence(rows, lambda *_: None, text_key="text", label="pii_spans", base=base)

    assert (report.checked, report.unchecked, report.ok) == (0, 4, True)
    assert {r["evidence_check"] for r in rows} == {"unchecked"}
    assert "written as unchecked" in caplog.text


def test_a_row_without_text_is_unchecked_not_passed(tmp_path):
    _, indexes, rows = _rows(tmp_path)
    for row in rows:
        row["text"] = None
    rows.append(_row("", evidence("chunk", page=1)))

    report = check_rows(rows, indexes.get, text_key="text", label="entity_links",
                        chunk_text=lambda _row: "Contact Jane Doe about the grant.")

    assert (report.checked, report.unchecked, report.ok) == (0, 4, True)


def test_merged_reports_keep_the_named_rows_capped():
    a = EvidenceReport(checked=4, n_mismatched=4, mismatches=["a"] * 4)
    a.merge(EvidenceReport(checked=4, n_mismatched=4, mismatches=["b"] * 4))

    assert (a.checked, a.n_mismatched, len(a.mismatches)) == (8, 8, 5)


def test_a_null_receipt_fails(tmp_path):
    base, indexes, _ = _rows(tmp_path)
    rows = [_row("Jane Doe", None)]

    report = stamp_evidence(rows, indexes.get, text_key="text", label="money_spans", base=base)

    assert (report.n_mismatched, rows[0]["evidence_check"]) == (1, "fails")


def test_pii_and_link_stage_the_elements_cells_and_declared_text_layer(tmp_path):
    from tests.test_cloud import _minimal_config
    from womblex.cloud.stage_contracts import (
        ELEMENTS_SUFFIX,
        NORMALISED_TEXT_SUFFIX,
        STAGE_CONTRACTS,
        TABLE_CELLS_SUFFIX,
    )

    config = _minimal_config(tmp_path)
    config.processing.text_source = "normalised"

    for stage in ("pii", "link"):
        needed = {c.suffix: c.strict for c in STAGE_CONTRACTS[stage].conditional_inputs(config)}
        assert needed[NORMALISED_TEXT_SUFFIX] is True
        assert needed[ELEMENTS_SUFFIX] is False and needed[TABLE_CELLS_SUFFIX] is False
