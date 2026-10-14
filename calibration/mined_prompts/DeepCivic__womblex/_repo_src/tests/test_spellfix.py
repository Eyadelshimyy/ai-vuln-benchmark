"""Tests for the dictionary-gated OCR character-confusion repair op.

Unit tests for the pure :func:`womblex.process.spellfix.repair_text` corrector,
a per-stage test that builds a minimal elements shard and asserts the element-
grain ``*.spellfix_text.parquet`` overlay + ``*.spellfix_corrections.parquet``
audit, and a composition test proving the overlay flows into both reassembly
sites (chunk + enrich) through ``process.text_overlay``. Runs offline against the
bundled en_AU Hunspell dictionary (needs ``spylls``).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("spylls")

from womblex.config import SpellfixConfig
from womblex.ingest.elements import Element
from womblex.process.chunker import reassemble_narrative
from womblex.process.spellfix import repair_text
from womblex.process.spellfix_stage import spellfix_shards
from womblex.process.text_overlay import apply_overlay, load_overlay
from womblex.store.normalise_output import write_normalised_text
from womblex.store.output import ELEMENT_SCHEMA, MANIFEST_SCHEMA, TABLE_CELLS_SCHEMA
from womblex.store.spellfix_output import (
    read_spellfix_corrections,
    read_spellfix_text,
    spellfix_text_path_for,
)

DOC = "doc1"


# ---------------------------------------------------------------------------
# Corrector (Tier A homoglyph, default)
# ---------------------------------------------------------------------------


def test_homoglyph_fixes_digit_for_letter():
    fixed, corr = repair_text("The chi1d went home.")
    assert fixed == "The child went home."
    assert len(corr) == 1
    assert (corr[0].original, corr[0].corrected, corr[0].method) == ("chi1d", "child", "homoglyph")


def test_homoglyph_is_length_preserving():
    text = "A good p1an indeed."
    fixed, _ = repair_text(text)
    assert fixed == "A good plan indeed."
    assert len(fixed) == len(text)


def test_valid_word_is_never_touched():
    fixed, corr = repair_text("The colour of our neighbourhood.")
    assert fixed == "The colour of our neighbourhood."
    assert corr == []


def test_codes_and_ids_left_alone():
    for text in ("Section3 of the Act.", "FY2024 budget B2B."):
        fixed, corr = repair_text(text)
        assert fixed == text
        assert corr == []


def test_all_caps_acronym_skipped():
    fixed, corr = repair_text("The CH1LD record.")
    assert fixed == "The CH1LD record."
    assert corr == []


def test_no_digit_no_homoglyph_change():
    fixed, corr = repair_text("teh cat sat.")
    assert fixed == "teh cat sat."
    assert corr == []


def test_ambiguous_token_left_verbatim_even_in_tier_b():
    fixed, corr = repair_text("teh cat sat.", general_edits=True)
    assert fixed == "teh cat sat."
    assert corr == []


def test_in_dictionary_proper_noun_untouched_under_tier_b():
    fixed, _ = repair_text("Standish attended.", general_edits=True)
    assert fixed == "Standish attended."


# ---------------------------------------------------------------------------
# Per-stage driver (element grain)
# ---------------------------------------------------------------------------


def _elem_row(order: int, text: str, kind: str = "paragraph") -> dict:
    row = {f.name: None for f in ELEMENT_SCHEMA}
    row.update({
        "source_hash": DOC, "elem_order": order, "kind": kind,
        "extractor": "native", "confidence": 1.0, "page": 1, "text": text,
    })
    return row


def _build_elements_shard(d: Path, element_rows: list[dict]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    base = d / "batch-0001.parquet"
    man = {f.name: None for f in MANIFEST_SCHEMA}
    man.update({"source_hash": DOC, "doc_id": DOC, "filename": "doc1.pdf", "status": "ok"})
    pq.write_table(pa.Table.from_pylist([man], schema=MANIFEST_SCHEMA),
                   str(d / "batch-0001._manifest.parquet"))
    pq.write_table(pa.Table.from_pylist(element_rows, schema=ELEMENT_SCHEMA),
                   str(d / "batch-0001.elements.parquet"))
    pq.write_table(
        pa.table({f.name: pa.array([], type=f.type) for f in TABLE_CELLS_SCHEMA},
                 schema=TABLE_CELLS_SCHEMA),
        str(d / "batch-0001.table_cells.parquet"),
    )
    return base


def test_spellfix_shards_writes_overlay_and_audit(tmp_path: Path):
    base = _build_elements_shard(tmp_path, [
        _elem_row(0, "The chi1d is fine."),
        _elem_row(1, "All good here."),
    ])
    result = spellfix_shards(tmp_path, SpellfixConfig(enabled=True))

    assert result.batches_written == 1
    assert result.elements_repaired == 1          # only the first element changed
    assert result.corrections_applied == 1

    overlay = read_spellfix_text(spellfix_text_path_for(base))
    by_order = dict(zip(overlay.column("elem_order").to_pylist(),
                        overlay.column("text").to_pylist()))
    assert by_order[0] == "The child is fine."
    assert by_order[1] == "All good here."          # passthrough element retained

    audit = read_spellfix_corrections(spellfix_text_path_for(base).parent)
    assert audit.column("original").to_pylist() == ["chi1d"]
    assert audit.column("corrected").to_pylist() == ["child"]


def test_spellfix_shards_leaves_raw_elements_untouched(tmp_path: Path):
    _build_elements_shard(tmp_path, [_elem_row(0, "The chi1d is fine.")])
    spellfix_shards(tmp_path, SpellfixConfig(enabled=True))

    raw = pq.read_table(str(tmp_path / "batch-0001.elements.parquet"))
    assert raw.column("text").to_pylist() == ["The chi1d is fine."]


def test_spellfix_chains_off_normalised_layer(tmp_path: Path):
    # Element has both a normalise-target (double space) and an OCR error.
    base = _build_elements_shard(tmp_path, [_elem_row(0, "The  chi1d is fine.")])
    # Pre-write a normalise overlay that collapsed the double space.
    write_normalised_text(
        [{"source_hash": DOC, "elem_order": 0, "kind": "paragraph",
          "page": 1, "text": "The chi1d is fine.", "n_changes": 1}],
        base,
    )
    spellfix_shards(tmp_path, SpellfixConfig(enabled=True))

    overlay = read_spellfix_text(spellfix_text_path_for(base))
    # spellfix_text is the terminal layer: carries normalisation AND repair.
    assert overlay.column("text").to_pylist() == ["The child is fine."]


# ---------------------------------------------------------------------------
# Composition: the overlay flows into reassembly (chunk + enrich share it)
# ---------------------------------------------------------------------------


def test_overlay_flows_into_reassembled_narrative(tmp_path: Path):
    base = _build_elements_shard(tmp_path, [
        _elem_row(0, "The chi1d is fine."),
        _elem_row(1, "Second p1an line."),
    ])
    spellfix_shards(tmp_path, SpellfixConfig(enabled=True))

    elements = [Element(order=0, kind="paragraph", extractor="native", text="The chi1d is fine."),
                Element(order=1, kind="paragraph", extractor="native", text="Second p1an line.")]
    overrides = load_overlay(base, "spellfix")
    apply_overlay(DOC, elements, overrides)
    narrative, _ = reassemble_narrative(elements)

    # This is the exact text both the chunker and the enricher would consume.
    assert narrative == "The child is fine.\n\nSecond plan line."
