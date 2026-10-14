"""Tests for the chunk-quality stage.

Unit tests for the pure metrics + duplicate clustering in
:mod:`womblex.process.quality`, plus a per-stage test that writes minimal chunk
shards across two batches and asserts the ``*.chunk_quality.parquet`` sidecars
(cross-batch dedup, flags).
"""

from __future__ import annotations

from pathlib import Path

from womblex.config import QualityConfig
from womblex.process.quality import (
    alpha_frac,
    boilerplate_flag,
    char_len,
    compile_patterns,
    exact_dup_ids,
    near_dup_ids,
)
from womblex.process.quality_stage import quality_shards
from womblex.store.output import write_chunks
from womblex.store.quality_output import read_chunk_quality


# ---- pure metrics -----------------------------------------------------------
def test_char_len_and_alpha_frac():
    assert char_len("abcd") == 4
    assert alpha_frac("ab12") == 0.5
    assert alpha_frac("") == 0.0


def test_boilerplate_flag():
    pats = compile_patterns([r"GPO Box 158"])
    assert boilerplate_flag("... GPO Box 158 Canberra", pats) is True
    assert boilerplate_flag("ordinary text", pats) is False


def test_exact_dups_are_also_near_dups():
    # identical text must land in the same near-dup cluster too (consistency)
    texts = ["the same long sentence about records and children", "x",
             "the same long sentence about records and children"]
    near = near_dup_ids(texts, perms=64, bands=4, k=5)
    assert near[0] == near[2] is not None
    assert near[1] is None


def test_near_dup_ids_deterministic_regardless_of_hash_seed():
    # stable hash (not Python's salted hash) -> identical ids across calls
    texts = [f"w{i}" for i in range(40)]
    texts = [" ".join(texts), " ".join(texts[:-1]), "unrelated words entirely here"]
    a = near_dup_ids(texts, perms=64, bands=8, k=5)
    b = near_dup_ids(texts, perms=64, bands=8, k=5)
    assert a == b


def test_quality_config_rejects_bands_not_dividing_permutations():
    import pytest
    with pytest.raises(ValueError, match="must divide"):
        QualityConfig(minhash_permutations=64, minhash_bands=5)


def test_exact_dup_ids_normalises_and_clusters():
    ids = exact_dup_ids(["Hello, world", "hello   world", "different"])
    assert ids[0] == ids[1] is not None       # same after normalisation
    assert ids[2] is None                       # singleton


def test_near_dup_ids_clusters_near_copies_only():
    words = [f"w{i}" for i in range(40)]
    base = " ".join(words)
    near = " ".join(words[:-1])    # ~0.97 shingle Jaccard — clearly a near-copy
    far = " ".join(f"z{i}" for i in range(40))   # disjoint vocabulary
    # bands=8 keeps the assertion off the probabilistic ~0.92 boundary while
    # still proving the mechanism: near-copies cluster, unrelated do not.
    ids = near_dup_ids([base, near, far], perms=64, bands=8, k=5)
    assert ids[0] == ids[1] is not None
    assert ids[2] is None


# ---- stage (cross-batch) ----------------------------------------------------
def _chunk_row(h, idx, text):
    return {"source_hash": h, "chunk_index": idx, "text": text, "start_char": 0,
            "end_char": len(text), "content_type": "narrative",
            "has_redaction": False, "page_start": 1, "page_end": 1}


def test_quality_stage_cross_batch_dedup(tmp_path: Path):
    shared = "The approved provider must keep records of every child enrolled."
    write_chunks([_chunk_row("a", 0, shared), _chunk_row("a", 1, "short")],
                 tmp_path / "batch-0001.parquet")
    # identical text in a *different* batch -> exact dup must cross batches
    write_chunks([_chunk_row("b", 0, shared)], tmp_path / "batch-0002.parquet")

    res = quality_shards(tmp_path, QualityConfig(short_chars=10))
    assert res.batches_written == 2
    assert res.chunks_annotated == 3

    q = read_chunk_quality(tmp_path).to_pylist()
    by = {(r["source_hash"], r["chunk_index"]): r for r in q}
    a0, b0, a1 = by[("a", 0)], by[("b", 0)], by[("a", 1)]
    assert a0["exact_dup_id"] == b0["exact_dup_id"] is not None   # cross-batch
    assert a1["is_short"] is True and a0["is_short"] is False
    assert a0["char_len"] == len(shared)
