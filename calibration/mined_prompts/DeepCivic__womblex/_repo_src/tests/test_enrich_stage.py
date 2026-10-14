"""Tests for the per-stage enrich + link wiring over a shard directory.

Builds an extraction shard from the synthetic quokka care decision notice
(small, native), runs ``enrich_shards`` against the **live** Isaacus
Kanon-2 enricher (no mocks — real for local validation per CLAUDE.md; skips
cleanly without ``ISAACUS_API_KEY``), then ``link_shards`` against the
notice's register row to confirm the two stages compose: the
provider legal name resolves to the canonical SE-/PR- ids. Also covers
checkpoint skip-on-resume and no-checkpoint-on-failure (the latter via a real
invalid-key client, not a stubbed exception).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._synthetic import NOTICE_PDF
from womblex.analyse.enrich_stage import enrich_shards
from womblex.config import EnrichmentConfig, LinkingConfig, ReferenceConfig
from womblex.ingest.detect import DetectionConfig, detect_file_type
from womblex.ingest.extract import extract_text
from womblex.link.stage import link_shards
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_output import (
    enrichment_entities_path_for,
    graph_edges_path_for,
    read_enrichment_entities,
    read_graph_edges,
)
from womblex.store.entity_links_output import read_entity_links
from womblex.store.output import write_results

# The notice's service, as a register row. Enrichment extracts the provider
# legal name ("Rottnest Community Services Incorporated"), which fuzzy-resolves
# to this SE-/PR- pair.
_REGISTER_CSV = (
    "ServiceApprovalNumber,Provider Approval Number,ServiceName,ProviderLegalName,"
    "ServiceAddress,Suburb,Postcode\n"
    "SE-40099001,PR-00099017,Quokka Cove Out of School Hours Care,"
    "Rottnest Community Services Incorporated,1 Thomson Bay Road,ROTTNEST,6161\n"
)


@pytest.fixture
def shard_dir(tmp_path) -> Path:
    d = tmp_path / "documents"
    d.mkdir()
    extraction = extract_text(NOTICE_PDF, detect_file_type(NOTICE_PDF, DetectionConfig()))[0]
    write_results([("notice", str(NOTICE_PDF), extraction)], d / "batch-0001.parquet",
                  collection_id="test")
    return d


@pytest.fixture
def reference_config(tmp_path) -> ReferenceConfig:
    csv_path = tmp_path / "services.csv"
    csv_path.write_text(_REGISTER_CSV, encoding="utf-8")
    return ReferenceConfig(
        path=csv_path, id_col="ServiceApprovalNumber", name_col="ServiceName",
        entity_type="service", parent_id_col="Provider Approval Number",
        match_exact_cols=["ServiceAddress", "Suburb", "Postcode"],
        match_fuzzy_cols=["ServiceName", "ProviderLegalName"],
    )


class TestEnrichShards:
    def test_writes_entities_sidecar(self, shard_dir, isaacus_client):
        result = enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client)
        assert result.docs_enriched == 1
        base = shard_dir / "batch-0001.parquet"
        assert enrichment_entities_path_for(base).exists()
        rows = read_enrichment_entities(base).to_pylist()
        assert rows, "real enrichment produced no entities"
        kinds = {r["entity_type"] for r in rows}
        # The notice names a corporate provider and gives a postal address.
        assert "corporate" in kinds and "address" in kinds

    def test_writes_graph_edges_sidecar(self, shard_dir, isaacus_client):
        result = enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client)
        assert result.docs_enriched == 1
        base = shard_dir / "batch-0001.parquet"
        assert graph_edges_path_for(base).exists()
        edges = read_graph_edges(base)
        assert edges.num_rows > 0, "real enrichment produced no graph edges"
        # entities and edges join the other sidecars on source_hash
        hashes = set(read_enrichment_entities(base).column("source_hash").to_pylist())
        assert set(edges.column("source_hash").to_pylist()) <= hashes

    def test_checkpoint_skips_on_resume(self, shard_dir, isaacus_client, tmp_path):
        ckpt = CheckpointManager(tmp_path / ".enrich-ckpt", "t_enrich")
        ckpt.load()
        first = enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client,
                              checkpoint_mgr=ckpt)
        assert first.docs_enriched == 1
        # Resume: doc checkpointed → no new enrich calls.
        second = enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client,
                               checkpoint_mgr=ckpt)
        assert second.docs_enriched == 0

    def test_transient_failure_not_checkpointed(self, shard_dir, bad_isaacus_client, tmp_path):
        # A real API failure (invalid key) must leave the doc unprocessed so a
        # resume retries it rather than skipping it forever.
        ckpt = CheckpointManager(tmp_path / ".enrich-ckpt", "t_enrich")
        ckpt.load()
        enrich_shards(shard_dir, EnrichmentConfig(), client=bad_isaacus_client,
                      checkpoint_mgr=ckpt)
        assert "notice" not in ckpt.state.processed_ids


class TestPersistDocumentReuse:
    """Live: the persisted Document round-trips and byte-matches the narrative.

    This is the runtime form of verification gate 1 (docs/decisions.md): the
    chunk stage's reuse guard accepts a Document only when ``document.text``
    equals the reassembled narrative, so the persisted-then-rehydrated text
    must be byte-identical to what enrich/chunk reassemble.
    """

    def test_doc_sidecar_round_trips_and_matches_narrative(self, shard_dir, isaacus_client):
        from womblex.analyse.enrich_stage import _load_narratives
        from womblex.store.enrichment_doc import (
            enrichment_doc_path_for,
            read_enrichment_docs,
        )

        enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client,
                      persist_document=True)
        base = shard_dir / "batch-0001.parquet"
        assert enrichment_doc_path_for(base).exists()

        stored = read_enrichment_docs(base)
        assert stored, "persist_document=True produced no doc sidecar rows"

        from isaacus.types.ilgs.v1.document import Document

        narratives, _ = _load_narratives(base, "elements")
        for source_hash, (stamp, doc_json) in stored.items():
            assert stamp == "elements"
            doc = Document.model_validate_json(doc_json)
            assert doc.text == narratives[source_hash], (
                "rehydrated Document.text must byte-match the narrative the "
                "chunk-stage reuse guard reassembles"
            )

    def test_default_writes_no_doc_sidecar(self, shard_dir, isaacus_client):
        from womblex.store.enrichment_doc import enrichment_doc_path_for

        enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client)
        assert not enrichment_doc_path_for(shard_dir / "batch-0001.parquet").exists()


class TestEnrichThenLink:
    def test_full_chain_resolves_the_notice_provider(self, shard_dir, reference_config, isaacus_client):
        enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client)

        cfg = LinkingConfig(enabled=True, reference=reference_config)
        result = link_shards(shard_dir, cfg)
        assert result.docs_linked == 1
        assert result.matched_links >= 1

        doc = read_entity_links(shard_dir, grain="doc").to_pylist()
        assert len(doc) == 1
        # provider legal name resolved to the notice's canonical service/provider ids
        assert doc[0]["entity_id"] == "SE-40099001"
        assert doc[0]["parent_entity_id"] == "PR-00099017"
