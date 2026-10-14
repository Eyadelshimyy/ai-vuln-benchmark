"""Tests for the downstream text-cleaning op.

Unit tests for the pure transforms in :mod:`womblex.process.normalise`, plus a
per-stage test that builds a minimal shard (elements + manifest) directly and
asserts the ``*.normalised_text.parquet`` sidecar — no heavy extraction.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.config import NormaliseConfig
from womblex.process.normalise import (
    NormaliseTransforms,
    apply_substitutions,
    collapse_whitespace,
    despace_page_marker,
    normalise_text,
    normalise_unicode,
)
from womblex.process.normalise_stage import normalise_shards
from womblex.store.checkpoint import CheckpointManager
from womblex.store.normalise_output import (
    NORMALISED_TEXT_SUFFIX,
    normalised_text_path_for,
    read_normalised_text,
)
from womblex.store.output import ELEMENT_SCHEMA, MANIFEST_SCHEMA, TABLE_CELLS_SCHEMA

DOC = "doc1"


# ---------------------------------------------------------------------------
# Pure transforms
# ---------------------------------------------------------------------------


def test_collapse_whitespace_runs_and_trailing():
    text, n = collapse_whitespace("foo   bar\tbaz   \nnext  line   ")
    assert text == "foo bar baz\nnext line"
    assert n > 0


def test_collapse_whitespace_preserves_newlines():
    text, _ = collapse_whitespace("line one\nline two\n\npara")
    assert text == "line one\nline two\n\npara"


def test_collapse_whitespace_noop():
    text, n = collapse_whitespace("already clean text")
    assert text == "already clean text"
    assert n == 0


def test_normalise_unicode_folds_spaces_and_separators():
    text, n = normalise_unicode("a b c d")  # NBSP, em-space, line sep
    assert text == "a b c\nd"
    assert n == 3


def test_normalise_unicode_removes_zero_width_and_bom():
    text, n = normalise_unicode("foo\u200b﻿bar")
    assert text == "foobar"
    assert n == 2


def test_normalise_unicode_preserves_typography():
    s = "‘q’ “d” – — •"
    text, n = normalise_unicode(s)
    assert text == s  # smart quotes, en/em dash, bullet untouched
    assert n == 0


def test_normalise_unicode_noop_on_ascii():
    text, n = normalise_unicode("plain ascii\ntext\twith tab")
    assert text == "plain ascii\ntext\twith tab"
    assert n == 0


def test_despace_page_marker_variants():
    assert despace_page_marker("3|P age")[0] == "3|Page"
    assert despace_page_marker("Pa ge 4")[0] == "Page 4"
    assert despace_page_marker("P a g e 1")[0] == "Page 1"


def test_despace_page_marker_preserves_case():
    assert despace_page_marker("see p age")[0] == "see page"


def test_apply_substitutions_longest_first():
    subs = {"http:lL": "http://", "lL": "//"}
    text, n = apply_substitutions("visit http:lLsite.gov.au", subs)
    assert text == "visit http://site.gov.au"
    assert n == 1


def test_normalise_text_gates_page_marker_to_footer():
    t = NormaliseTransforms()
    # In body prose, "P age" must NOT be despaced (only footer/header kinds).
    out, _ = normalise_text("turn the P age slowly", "paragraph", t)
    assert "P age" in out
    out_footer, _ = normalise_text("3|P age", "footer", t)
    assert out_footer == "3|Page"


def test_normalise_text_empty():
    assert normalise_text("", "paragraph", NormaliseTransforms()) == ("", 0)


# ---------------------------------------------------------------------------
# Per-stage driver
# ---------------------------------------------------------------------------


def _element_row(order: int, kind: str, text: str, page: int = 1) -> dict:
    row = {f.name: None for f in ELEMENT_SCHEMA}
    row.update({
        "source_hash": DOC, "elem_order": order, "kind": kind,
        "extractor": "test", "confidence": 1.0, "page": page, "text": text,
    })
    return row


def _build_shard(d: Path, element_rows: list[dict]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    base = d / "batch-0001.parquet"
    man = {f.name: None for f in MANIFEST_SCHEMA}
    man.update({"source_hash": DOC, "doc_id": DOC, "filename": "doc1.pdf", "status": "ok"})
    pq.write_table(
        pa.Table.from_pylist([man], schema=MANIFEST_SCHEMA),
        str(d / "batch-0001._manifest.parquet"),
    )
    pq.write_table(
        pa.Table.from_pylist(element_rows, schema=ELEMENT_SCHEMA),
        str(d / "batch-0001.elements.parquet"),
    )
    # Empty table_cells sidecar — extraction always writes all four siblings;
    # _load_elements (reused from chunk_stage) reads this for cell stitching.
    pq.write_table(
        pa.table({f.name: pa.array([], type=f.type) for f in TABLE_CELLS_SCHEMA},
                 schema=TABLE_CELLS_SCHEMA),
        str(d / "batch-0001.table_cells.parquet"),
    )
    return base


def test_normalise_shards_writes_sidecar(tmp_path: Path):
    base = _build_shard(tmp_path, [
        _element_row(0, "paragraph", "The  quick   brown fox"),
        _element_row(1, "footer", "3|P age"),
        _element_row(2, "table", "should be ignored   here"),
    ])

    result = normalise_shards(tmp_path, NormaliseConfig())

    assert result.batches_written == 1
    assert result.elements_normalised == 2  # paragraph + footer; table excluded
    assert result.elements_changed == 2

    table = read_normalised_text(base)
    by_order = {r["elem_order"]: r for r in table.to_pylist()}
    assert set(by_order) == {0, 1}
    assert by_order[0]["text"] == "The quick brown fox"
    assert by_order[1]["text"] == "3|Page"
    assert by_order[1]["kind"] == "footer"
    assert all(r["n_changes"] > 0 for r in by_order.values())


def test_normalise_shards_passthrough_unchanged(tmp_path: Path):
    base = _build_shard(tmp_path, [_element_row(0, "paragraph", "already clean")])
    normalise_shards(tmp_path, NormaliseConfig())
    rows = read_normalised_text(base).to_pylist()
    assert rows[0]["text"] == "already clean"
    assert rows[0]["n_changes"] == 0


def test_normalise_shards_checkpoint_skips_second_run(tmp_path: Path):
    _build_shard(tmp_path, [_element_row(0, "paragraph", "foo  bar")])
    ckpt = CheckpointManager(tmp_path / ".normalise-checkpoint", "ds_normalise")

    first = normalise_shards(tmp_path, NormaliseConfig(), checkpoint_mgr=ckpt)
    assert first.batches_written == 1

    second = normalise_shards(tmp_path, NormaliseConfig(), checkpoint_mgr=ckpt)
    assert second.batches_written == 0  # all docs checkpointed → skipped


def test_normalise_shards_substitutions(tmp_path: Path):
    base = _build_shard(tmp_path, [_element_row(0, "paragraph", "see http:lLx.gov.au")])
    cfg = NormaliseConfig(substitutions={"http:lL": "http://"})
    normalise_shards(tmp_path, cfg)
    rows = read_normalised_text(base).to_pylist()
    assert rows[0]["text"] == "see http://x.gov.au"


def test_normalised_text_path_for():
    base = Path("/tmp/run/batch-0001.parquet")
    assert normalised_text_path_for(base).name == f"batch-0001{NORMALISED_TEXT_SUFFIX}"
