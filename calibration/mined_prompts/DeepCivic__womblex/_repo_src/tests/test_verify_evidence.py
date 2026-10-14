"""``womblex verify-evidence`` re-checks a finished run; older link files re-anchor exactly."""

from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tests._shard import DOC, contact_shard, table_markdown, write_contact_chunks
from tests.test_enrich_packing import _FakeClient, _FakeCounter
from womblex.analyse.enrich_stage import enrich_shards
from womblex.cli.verify import cmd_verify_evidence
from womblex.config import EnrichmentConfig, MoneyConfig, PIIConfig
from womblex.pii.pii_stage import pii_shards
from womblex.process.evidence import verify_shards
from womblex.process.money_stage import money_shards
from womblex.store.entity_links_output import read_entity_links
from womblex.store.money_output import money_spans_path_for


def _run(d):
    base = contact_shard(d)
    money_shards(d, MoneyConfig())
    enrich_shards(d, EnrichmentConfig(), client=_FakeClient(), token_counter=_FakeCounter())
    write_contact_chunks(base, table_markdown(base))
    pii_shards(d, PIIConfig(use_regex_backstop=False))
    return base


def test_a_clean_run_verifies_and_a_shifted_span_is_reported(tmp_path, capsys):
    base = _run(tmp_path)
    clean = verify_shards(tmp_path)
    assert clean.ok and clean.checked > 0 and clean.unchecked == 0
    assert cmd_verify_evidence(argparse.Namespace(run_dir=tmp_path)) == 0

    path = money_spans_path_for(base)
    table = pq.read_table(str(path))
    shifted = table.set_column(
        table.schema.get_field_index("char_start"), "char_start",
        pc.add(table["char_start"], pa.scalar(1, pa.int32())))
    pq.write_table(shifted, str(path))

    report = verify_shards(tmp_path)
    assert not report.ok and report.n_mismatched >= 1
    assert cmd_verify_evidence(argparse.Namespace(run_dir=tmp_path)) == 2
    assert "fails: money_spans" in capsys.readouterr().out


def test_receipts_whose_source_is_gone_are_unchecked_not_failed(tmp_path):
    base = _run(tmp_path)
    # Without the elements only the chunk-level PII receipts can still be checked.
    (tmp_path / f"{base.stem}.elements.parquet").unlink()

    report = verify_shards(tmp_path)

    assert report.ok and report.unchecked > 0


def test_a_directory_that_is_not_a_run_is_refused(tmp_path):
    assert cmd_verify_evidence(argparse.Namespace(run_dir=tmp_path / "nowhere")) == 1


# ---------------------------------------------------------------------------
# Older link files
# ---------------------------------------------------------------------------


def _legacy_links(d) -> None:
    pq.write_table(pa.table({
        "source_hash": [DOC], "candidate_text": ["Jane Doe"], "candidate_kind": ["natural"],
        "mention_start": pa.array([8], pa.int32()), "mention_end": pa.array([16], pa.int32()),
        "entity_id": ["PR-1"], "entity_type": ["provider"], "canonical_name": ["Jane"],
        "parent_entity_id": [""], "confidence": pa.array([1.0], pa.float32()),
        "match_method": ["alias"], "matched": [True],
    }), str(d / "batch-0001.entity_links.parquet"))


def test_an_older_link_is_re_anchored_exactly_when_the_elements_are_present(tmp_path):
    contact_shard(tmp_path)
    _legacy_links(tmp_path)

    link = read_entity_links(tmp_path).to_pylist()[0]

    assert (link["anchor_level"], link["elem_order"], link["char_start"], link["char_end"]) == (
        "span", 0, 8, 16)
    assert (link["text_layer"], link["evidence_check"], link["mention_text"]) == (
        "elements", "unchecked", None)


def test_an_older_link_stays_document_located_with_an_overlay_or_without_elements(tmp_path):
    contact_shard(tmp_path / "overlay")
    _legacy_links(tmp_path / "overlay")
    (tmp_path / "overlay" / "batch-0001.normalised_text.parquet").write_bytes(b"")
    (tmp_path / "bare").mkdir()
    _legacy_links(tmp_path / "bare")

    for d in ("overlay", "bare"):
        link = read_entity_links(tmp_path / d).to_pylist()[0]
        assert (link["anchor_level"], link["char_start"]) == ("document", None)
