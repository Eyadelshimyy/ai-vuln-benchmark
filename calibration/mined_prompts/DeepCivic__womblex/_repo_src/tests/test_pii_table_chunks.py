"""A table chunk is masked from its own table's mentions (offline fake enricher)."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests._shard import contact_shard, table_markdown, write_contact_chunks
from tests.test_enrich_packing import _FakeClient, _FakeCounter
from womblex.analyse.enrich_stage import enrich_shards
from womblex.config import EnrichmentConfig, PIIConfig
from womblex.pii.pii_stage import pii_shards
from womblex.store.enrichment_output import (
    ENTITY_SCHEMA,
    enrichment_entities_path_for,
    read_enrichment_entities,
)
from womblex.store.pii_output import read_clean_text


def _enriched(d, **enrich):
    base = contact_shard(d)
    enrich_shards(d, EnrichmentConfig(**enrich), client=_FakeClient(), token_counter=_FakeCounter())
    return base


def _table_mentions(base, mention: tuple[int, int] | None) -> None:
    """Move the table's mention to ``mention`` (offsets into its markdown), or drop it."""
    rows = []
    for r in read_enrichment_entities(base).to_pylist():
        if r["text_layer"] != "table_markdown":
            rows.append(r)
        elif mention is not None:
            rows.append({**r, "mention_start": mention[0], "mention_end": mention[1]})
    pq.write_table(pa.Table.from_pylist(rows, schema=ENTITY_SCHEMA),
                   str(enrichment_entities_path_for(base)))


def _pii(d) -> dict[str, dict]:
    pii_shards(d, PIIConfig(use_regex_backstop=False))
    return {r["content_type"]: r for r in read_clean_text(d).to_pylist()}


def test_a_table_chunk_is_masked_from_its_tables_mentions(tmp_path):
    base = _enriched(tmp_path)
    write_contact_chunks(base, table_markdown(base))

    clean = _pii(tmp_path)

    assert clean["table"]["mask_status"] == "masked"
    assert clean["table"]["text"].startswith("<PERSON_")


def test_a_chunk_starting_part_way_into_its_table_is_masked_in_place(tmp_path):
    base = _enriched(tmp_path)
    markdown = table_markdown(base)
    name = markdown.index("John Citizen")
    _table_mentions(base, (name, name + len("John Citizen")))
    cut = name - 3
    write_contact_chunks(base, markdown, table_start=cut)

    clean = _pii(tmp_path)

    # The narrative's person took <PERSON_1>; the table's is a different entity id.
    assert clean["table"]["text"] == (
        markdown[cut:name] + "<PERSON_2>" + markdown[name + len("John Citizen"):])


def test_a_table_chunk_is_uncovered_when_its_tables_were_not_enriched(tmp_path):
    base = _enriched(tmp_path, include_tables=False)
    write_contact_chunks(base, table_markdown(base))

    clean = _pii(tmp_path)

    assert (clean["narrative"]["mask_status"], clean["table"]["mask_status"]) == (
        "masked", "not_masked")


def test_an_enriched_table_with_no_pii_reads_no_entity(tmp_path):
    base = _enriched(tmp_path)
    write_contact_chunks(base, table_markdown(base))
    _table_mentions(base, None)  # the meta row still says the table was enriched

    assert _pii(tmp_path)["table"]["mask_status"] == "no_entity"


def test_a_table_chunk_that_does_not_line_up_takes_no_table_spans(tmp_path):
    base = _enriched(tmp_path)
    write_contact_chunks(base, "| Name |\n| --- |\n| a repaired chunk |")

    table = _pii(tmp_path)["table"]

    assert table["mask_status"] == "not_masked" and table["n_masked"] == 0


def test_a_skipped_narrative_stays_uncovered_beside_enriched_tables(tmp_path):
    base = _enriched(tmp_path, skip_short_documents=10_000)
    write_contact_chunks(base, table_markdown(base))

    clean = _pii(tmp_path)

    assert (clean["narrative"]["mask_status"], clean["table"]["mask_status"]) == (
        "not_masked", "masked")


def test_a_declared_text_layer_missing_from_the_batch_refuses_the_stage(tmp_path):
    from womblex.config import LinkingConfig, ReferenceConfig
    from womblex.link.stage import link_shards
    from womblex.process.text_overlay import MissingOverlayError

    _enriched(tmp_path)
    reference = ReferenceConfig(path=tmp_path / "register.csv", id_col="id", name_col="name")

    with pytest.raises(MissingOverlayError):
        pii_shards(tmp_path, PIIConfig(use_regex_backstop=False), text_source="normalised")
    with pytest.raises(MissingOverlayError):
        link_shards(tmp_path, LinkingConfig(enabled=True, reference=reference),
                    text_source="normalised")
