"""Tests for the per-stage money op over a shard directory.

Builds minimal shards (elements + table_cells + manifest) directly — no heavy
extraction — and asserts the two sidecars, the three loci and their anchors,
checkpoint resume, and that the narrative coordinate space follows
``processing.text_source``.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from womblex.config import MoneyConfig
from womblex.process.money_stage import money_shards
from womblex.process.text_overlay import MissingOverlayError
from womblex.store.checkpoint import CheckpointManager
from womblex.store.money_output import (
    MONEY_SPANS_SCHEMA,
    money_spans_path_for,
    quantise,
    read_money_columns,
    read_money_spans,
)
from womblex.store.normalise_output import NORMALISED_TEXT_SCHEMA
from womblex.store.output import ELEMENT_SCHEMA, MANIFEST_SCHEMA, TABLE_CELLS_SCHEMA

DOC = "doc1"


# ---------------------------------------------------------------------------
# Shard construction
# ---------------------------------------------------------------------------


def _element(order: int, kind: str, **kwargs) -> dict:
    row = {f.name: None for f in ELEMENT_SCHEMA}
    row.update({
        "source_hash": DOC, "collection_id": "c", "elem_order": order,
        "kind": kind, "extractor": "test", "confidence": 1.0, "page": 1,
    })
    row.update(kwargs)
    return row


def _cell(parent: int, row_i: int, col: int, value: str) -> dict:
    return {
        "source_hash": DOC, "parent_elem_order": parent, "row": row_i, "col": col,
        "value": value, "rowspan": 1, "colspan": 1, "value_type": "text",
    }


def _build_shard(d: Path, elements: list[dict], cells: list[dict] | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    base = d / "batch-0001.parquet"
    man = {f.name: None for f in MANIFEST_SCHEMA}
    man.update({"source_hash": DOC, "doc_id": DOC, "filename": "doc1.pdf", "status": "ok"})
    pq.write_table(pa.Table.from_pylist([man], schema=MANIFEST_SCHEMA),
                   str(d / "batch-0001._manifest.parquet"))
    pq.write_table(pa.Table.from_pylist(elements, schema=ELEMENT_SCHEMA),
                   str(d / "batch-0001.elements.parquet"))
    pq.write_table(pa.Table.from_pylist(cells or [], schema=TABLE_CELLS_SCHEMA),
                   str(d / "batch-0001.table_cells.parquet"))
    return base


# ---------------------------------------------------------------------------
# Narrative locus
# ---------------------------------------------------------------------------


def test_narrative_spans_anchor_to_reassembled_offsets(tmp_path: Path):
    text = "The appropriation was $33.1 million for the program."
    base = _build_shard(tmp_path, [_element(0, "paragraph", text=text)])

    result = money_shards(tmp_path, MoneyConfig())

    assert result.batches_written == 1
    rows = read_money_spans(base).to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert row["locus"] == "narrative"
    assert row["text_layer"] == "elements"
    assert row["value"] == Decimal("33100000.0000")
    assert row["currency"] == "AUD"
    assert row["evidence"] == "p6"
    assert row["page"] == 1 and row["elem_order"] == 0
    assert text[row["char_start"]:row["char_end"]] == row["text"] == "$33.1 million"
    assert "appropriation" in row["context"]
    # Cell anchors stay null on a narrative row.
    assert row["sheet"] is None and row["cell_row"] is None and row["cell_col"] is None


def test_narrative_offsets_follow_text_source(tmp_path: Path):
    """Offsets index the selected element-text layer, not the raw elements."""
    raw = "The  appropriation  was  $33.1 million."
    cleaned = "The appropriation was $33.1 million."
    base = _build_shard(tmp_path, [_element(0, "paragraph", text=raw)])
    pq.write_table(
        pa.Table.from_pylist(
            [{"source_hash": DOC, "elem_order": 0, "kind": "paragraph",
              "page": 1, "text": cleaned, "n_changes": 3}],
            schema=NORMALISED_TEXT_SCHEMA),
        str(tmp_path / "batch-0001.normalised_text.parquet"),
    )

    money_shards(tmp_path, MoneyConfig(), text_source="normalised")

    row = read_money_spans(base).to_pylist()[0]
    assert row["text_layer"] == "normalised"
    assert cleaned[row["char_start"]:row["char_end"]] == "$33.1 million"


def test_missing_declared_overlay_refuses_before_writing(tmp_path: Path):
    """A declared layer that is absent is never silently replaced by verbatim text."""
    base = _build_shard(tmp_path, [_element(0, "paragraph", text="It cost $5.")])

    with pytest.raises(MissingOverlayError, match="batch-0001"):
        money_shards(tmp_path, MoneyConfig(), text_source="normalised")

    assert not money_spans_path_for(base).exists()


def test_narrative_can_be_disabled(tmp_path: Path):
    base = _build_shard(tmp_path, [_element(0, "paragraph", text="paid $100 today")])
    money_shards(tmp_path, MoneyConfig(narrative=False))
    assert read_money_spans(base).num_rows == 0


# ---------------------------------------------------------------------------
# table_cell locus
# ---------------------------------------------------------------------------


def _table_shard(d: Path) -> Path:
    elements = [
        _element(0, "paragraph", text="Financial summary follows."),
        _element(1, "table", header_rows=[0]),
    ]
    cells = [
        _cell(1, 0, 0, "Program"), _cell(1, 0, 1, "Expenditure $'000"),
        _cell(1, 0, 2, "Staff (FTE)"),
        _cell(1, 1, 0, "Alpha"), _cell(1, 1, 1, "1,500"), _cell(1, 1, 2, "12"),
        _cell(1, 2, 0, "Beta"), _cell(1, 2, 1, "(300)"), _cell(1, 2, 2, "8"),
        _cell(1, 3, 0, "Gamma"), _cell(1, 3, 1, "—"), _cell(1, 3, 2, "15"),
        _cell(1, 4, 0, "Delta"), _cell(1, 4, 1, "2,700"), _cell(1, 4, 2, "9"),
    ]
    return _build_shard(d, elements, cells)


def test_table_column_evidence_and_anchors(tmp_path: Path):
    base = _table_shard(tmp_path)

    money_shards(tmp_path, MoneyConfig())

    rows = [r for r in read_money_spans(base).to_pylist() if r["locus"] == "table_cell"]
    assert {r["cell_col"] for r in rows} == {1}, "only the expenditure column is money"
    by_row = {r["cell_row"]: r for r in rows}
    assert by_row[1]["value"] == Decimal("1500000.0000")   # $'000 header scale
    assert by_row[2]["value"] == Decimal("-300000.0000")   # brackets = negative
    assert 3 not in by_row                                  # em-dash is absent
    assert by_row[1]["elem_order"] == 1
    assert by_row[1]["column_id"] == "elem1:col1"
    assert by_row[1]["text_layer"] == "cell"                # offsets index the cell value
    assert (by_row[1]["char_start"], by_row[1]["char_end"]) == (0, len(by_row[1]["text"]))

    columns = read_money_columns(base).to_pylist()
    verdicts = {c["col"]: c for c in columns if c["locus"] == "table_cell"}
    assert verdicts[1]["verdict"] == "money"
    assert verdicts[1]["scale"] == "thousand"
    assert verdicts[2]["verdict"] == "vetoed"
    assert verdicts[2]["veto_term"] in {"staff", "fte"}
    assert verdicts[1]["cells_extracted"] == 3


def test_wrapped_header_row_is_recovered(tmp_path: Path):
    """Modelled on the ANAO Major Projects Report: the header wraps across two
    rows (`Approved` / `Budget $m`) and only the first is declared. The folded
    row must supply the scale and must not itself be extracted as an amount."""
    elements = [_element(0, "table", header_rows=[0])]
    cells = [_cell(0, 0, 0, "Project"), _cell(0, 0, 1, "Approved")]
    cells += [_cell(0, 1, 0, ""), _cell(0, 1, 1, "Budget $m")]
    for i, v in enumerate(["16,631.3", "9108.9", "6291.8", "78,699.2"], start=2):
        cells += [_cell(0, i, 0, f"Project {i}"), _cell(0, i, 1, v)]
    base = _build_shard(tmp_path, elements, cells)

    money_shards(tmp_path, MoneyConfig())

    rows = read_money_spans(base).to_pylist()
    assert [r["value"] for r in rows] == [
        Decimal("16631300000.0000"), Decimal("9108900000.0000"),
        Decimal("6291800000.0000"), Decimal("78699200000.0000")]
    assert all(r["multiplier"] == "million" for r in rows)
    assert 1 not in {r["cell_row"] for r in rows}, "the folded header row is not an amount"

    verdict = next(c for c in read_money_columns(base).to_pylist() if c["col"] == 1)
    assert verdict["header_text"] == "Approved Budget $m"
    assert (verdict["verdict"], verdict["scale"]) == ("money", "million")


def test_table_without_header_rows_leaves_bare_cells_alone(tmp_path: Path):
    elements = [_element(0, "table")]
    cells = [_cell(0, r, 0, v) for r, v in enumerate(["1,500", "2,700", "300", "900"])]
    base = _build_shard(tmp_path, elements, cells)

    money_shards(tmp_path, MoneyConfig())

    assert read_money_spans(base).num_rows == 0
    assert read_money_columns(base).num_rows == 0


def test_self_evidencing_cell_survives_an_unclassified_column(tmp_path: Path):
    elements = [_element(0, "table", header_rows=[0])]
    cells = [
        _cell(0, 0, 0, "Notes"),
        _cell(0, 1, 0, "paid $1,200.50 on invoice"),
        _cell(0, 2, 0, "no payment"),
        _cell(0, 3, 0, "pending"),
    ]
    base = _build_shard(tmp_path, elements, cells)

    money_shards(tmp_path, MoneyConfig())

    rows = read_money_spans(base).to_pylist()
    assert [r["value"] for r in rows] == [Decimal("1200.5000")]
    assert rows[0]["evidence"] == "p1"
    assert rows[0]["column_id"] is None  # inline evidence, not inherited


# ---------------------------------------------------------------------------
# sheet_cell locus
# ---------------------------------------------------------------------------


def test_sheet_column_classified_from_number_format(tmp_path: Path):
    """The register case: bare digits, no symbol anywhere but the cell format."""
    elements = [_element(0, "sheet_meta", sheet="Awards")]
    order = 1
    for row_i, (name, value) in enumerate([
        ("Recipient", "Value"), ("Alpha", "50000"), ("Beta", "125000"),
        ("Gamma", "7500"), ("Delta", "0"),
    ]):
        elements.append(_element(order, "sheet_cell", sheet="Awards", row=row_i,
                                 col=0, value=name, value_type="text", page=None))
        elements.append(_element(order + 1, "sheet_cell", sheet="Awards", row=row_i,
                                 col=1, value=value, page=None,
                                 value_type="text" if row_i == 0 else "number",
                                 number_format=None if row_i == 0 else "$#,##0.00"))
        order += 2
    base = _build_shard(tmp_path, elements)

    money_shards(tmp_path, MoneyConfig())

    rows = [r for r in read_money_spans(base).to_pylist() if r["locus"] == "sheet_cell"]
    assert [r["value"] for r in rows] == [
        Decimal("50000.0000"), Decimal("125000.0000"),
        Decimal("7500.0000"), Decimal("0.0000")]
    assert {r["sheet"] for r in rows} == {"Awards"}
    assert {r["cell_col"] for r in rows} == {1}
    assert rows[0]["currency_source"] == "number_format"
    assert rows[0]["elem_order"] is not None

    verdict = next(c for c in read_money_columns(base).to_pylist() if c["col"] == 1)
    assert (verdict["verdict"], verdict["evidence"]) == ("money", "number_format")
    assert verdict["header_text"] == "Value"
    assert verdict["cells_extracted"] == 4


def test_merge_covered_blank_cells_shift_column_denominators(tmp_path: Path):
    """a2 cascade: the blank cells a merge now emits enter classify_column.

    Before a2, a merge's covered cells were absent from the element stream, so
    the money column saw only its populated cells. a2 emits those covered
    blanks (empty ``value``); they are null markers, so they are excluded from
    the numeric denominator but *counted* in ``cells_total`` and
    ``null_fraction``. This test stands in for that emitted shape by placing an
    empty-valued ``sheet_cell`` in the money column and asserting the verdict's
    denominators move while the money verdict itself survives.
    """
    elements = [_element(0, "sheet_meta", sheet="Awards")]
    order = 1
    # Row 2's value cell is a merge-covered blank (empty value), exactly what
    # a2 now emits; every other value cell is a formatted amount.
    grid = [
        ("Recipient", "Value", None),
        ("Alpha", "50000", "$#,##0.00"),
        ("Beta", "", "$#,##0.00"),          # a2-emitted covered blank
        ("Gamma", "7500", "$#,##0.00"),
        ("Delta", "125000", "$#,##0.00"),
    ]
    for row_i, (name, value, fmt) in enumerate(grid):
        elements.append(_element(order, "sheet_cell", sheet="Awards", row=row_i,
                                 col=0, value=name, value_type="text", page=None))
        elements.append(_element(order + 1, "sheet_cell", sheet="Awards", row=row_i,
                                 col=1, value=value, page=None,
                                 value_type="text" if row_i == 0 else "number",
                                 number_format=fmt))
        order += 2
    base = _build_shard(tmp_path, elements)

    money_shards(tmp_path, MoneyConfig())

    verdict = next(c for c in read_money_columns(base).to_pylist() if c["col"] == 1)
    # The verdict survives — a blank is an absent value, not a non-numeric one.
    assert verdict["verdict"] == "money"
    # cells_total counts the four data rows including the emitted blank; the
    # blank lifts null_fraction and is excluded from the extracted amounts.
    assert verdict["cells_total"] == 4
    assert verdict["null_fraction"] == 0.25          # 1 of 4 data cells blank
    assert verdict["numeric_fraction"] == 1.0        # blank not in the denominator
    assert verdict["cells_extracted"] == 3

    rows = [r for r in read_money_spans(base).to_pylist() if r["locus"] == "sheet_cell"]
    assert [r["value"] for r in rows] == [
        Decimal("50000.0000"), Decimal("7500.0000"), Decimal("125000.0000")]


# ---------------------------------------------------------------------------
# Stage mechanics
# ---------------------------------------------------------------------------


def test_checkpoint_skips_second_run(tmp_path: Path):
    _build_shard(tmp_path, [_element(0, "paragraph", text="paid $100")])
    ckpt = CheckpointManager(tmp_path / ".money-checkpoint", "ds_money")

    first = money_shards(tmp_path, MoneyConfig(), checkpoint_mgr=ckpt)
    second = money_shards(tmp_path, MoneyConfig(), checkpoint_mgr=ckpt)

    assert first.batches_written == 1
    assert second.batches_written == 0


def test_empty_batch_writes_schema_correct_sidecars(tmp_path: Path):
    base = _build_shard(tmp_path, [_element(0, "paragraph", text="no amounts here")])

    money_shards(tmp_path, MoneyConfig())

    spans = read_money_spans(base)
    assert spans.num_rows == 0
    assert spans.schema.names == MONEY_SPANS_SCHEMA.names


def test_missing_shard_dir_raises(tmp_path: Path):
    try:
        money_shards(tmp_path / "nope", MoneyConfig())
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_resume_scan_drops_corrupt_sidecars(tmp_path: Path):
    """The shared resume self-heal must recognise the money sidecar suffix."""
    from womblex.store.money_output import MONEY_SPANS_SUFFIX
    from womblex.store.shard_audit import reconcile_stage_checkpoint_with_shards

    _build_shard(tmp_path, [_element(0, "paragraph", text="paid $100")])
    ckpt = CheckpointManager(tmp_path / ".money-checkpoint", "ds_money")
    money_shards(tmp_path, MoneyConfig(), checkpoint_mgr=ckpt)
    assert DOC in ckpt.state.processed_ids

    (tmp_path / f"batch-0001{MONEY_SPANS_SUFFIX}").write_bytes(b"")  # corrupt
    dropped = reconcile_stage_checkpoint_with_shards(
        ckpt, tmp_path, suffix=MONEY_SPANS_SUFFIX)

    assert dropped == [DOC]
    assert money_shards(tmp_path, MoneyConfig(), checkpoint_mgr=ckpt).batches_written == 1


def test_prose_cells_without_digits_are_skipped(tmp_path: Path):
    elements = [_element(0, "table", header_rows=[0])]
    cells = [
        _cell(0, 0, 0, "Notes"), _cell(0, 1, 0, "no amount here"),
        _cell(0, 2, 0, "also nothing"), _cell(0, 3, 0, "paid $50"),
    ]
    base = _build_shard(tmp_path, elements, cells)
    money_shards(tmp_path, MoneyConfig())
    assert [r["value"] for r in read_money_spans(base).to_pylist()] == [Decimal("50.0000")]


def test_a_worded_amount_in_a_cell_is_still_scanned(tmp_path: Path):
    """The digit fast-path skipped prose cells outright, so an amount written
    in words in a contract table was unreachable. The pre-filter now admits a
    currency word too."""
    elements = [_element(0, "table", header_rows=[0])]
    cells = [
        _cell(0, 0, 0, "Consideration"), _cell(0, 1, 0, "Two million dollars"),
        _cell(0, 2, 0, "no amount here"),
    ]
    base = _build_shard(tmp_path, elements, cells)
    money_shards(tmp_path, MoneyConfig())
    rows = read_money_spans(base).to_pylist()
    assert [(r["locus"], r["value"], r["evidence"]) for r in rows] == [
        ("table_cell", Decimal("2000000.0000"), "p11")]


def test_every_row_is_located_to_the_span_in_its_text(tmp_path: Path):
    """The documented schema contract: `locus` says which anchor group is
    populated, and the others stay null. Mixing coordinate spaces on one row
    would make the sidecar unjoinable."""
    elements = [
        _element(0, "paragraph", text="The department paid $33.1 million."),
        _element(1, "table", header_rows=[0]),
        _element(2, "sheet_cell", sheet="S", row=0, col=0, value="Value", page=None),
        _element(3, "sheet_cell", sheet="S", row=1, col=0, value="1000",
                 page=None, number_format="$#,##0.00"),
        _element(4, "sheet_cell", sheet="S", row=2, col=0, value="2000",
                 page=None, number_format="$#,##0.00"),
        _element(5, "sheet_cell", sheet="S", row=3, col=0, value="3000",
                 page=None, number_format="$#,##0.00"),
    ]
    cells = [_cell(1, 0, 0, "Amount"), _cell(1, 1, 0, "500"),
             _cell(1, 2, 0, "600"), _cell(1, 3, 0, "700")]
    base = _build_shard(tmp_path, elements, cells)

    money_shards(tmp_path, MoneyConfig())

    rows = read_money_spans(base).to_pylist()
    assert {r["locus"] for r in rows} == {"narrative", "table_cell", "sheet_cell"}
    for r in rows:
        assert r["anchor_level"] == "span"
        assert r["char_start"] is not None and r["elem_order"] is not None
        if r["locus"] == "narrative":
            assert r["text_layer"] == "elements"
            assert r["sheet"] is None and r["cell_row"] is None
        else:
            assert r["text_layer"] == "cell" and r["cell_row"] is not None
        if r["locus"] == "table_cell":
            assert r["sheet"] is None
        if r["locus"] == "sheet_cell":
            assert r["sheet"] is not None


def test_quantise_drops_unstorable_values():
    assert quantise(Decimal("1.23456")) == Decimal("1.2346")
    assert quantise(Decimal(10) ** 40) is None


def test_cli_reports_a_missing_overlay_as_a_refusal(tmp_path: Path):
    import argparse

    from womblex.cli.money import _register_money, cmd_money

    _build_shard(tmp_path, [_element(0, "paragraph", text="It cost $5.")])
    p = argparse.ArgumentParser()
    _register_money(p)
    args = p.parse_args(["--shards", str(tmp_path), "--text-source", "normalised"])
    assert cmd_money(args) == 1
