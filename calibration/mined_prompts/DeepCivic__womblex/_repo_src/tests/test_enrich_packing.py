"""Offline tests for enrich_stage token-budget packing + long-doc split.

A fake Isaacus client records the ``texts`` lists it receives so packing
behaviour is asserted without the real API; a whitespace token counter makes
budgets deterministic. Shards are built with the real records ingest.
"""

from __future__ import annotations

from types import SimpleNamespace

from womblex.analyse.enrich_stage import enrich_shards
from womblex.config import EnrichmentConfig
from womblex.ingest.records import RecordFieldMapping, ingest_records
from womblex.store.enrichment_output import read_enrichment_entities

_MAPPING = RecordFieldMapping(id_field="version_id", text_field="text", collection_id="t")


class _FakeCounter:
    """Duck-typed TokenCounter: token count == whitespace word count."""

    def count_batch(self, texts: list[str]) -> list[int]:
        return [len(t.split()) for t in texts]


def _fake_document(text: str) -> SimpleNamespace:
    """Minimal ILGS-shaped doc: one person mention so entities are produced."""
    end = min(4, len(text))
    return SimpleNamespace(
        text=text, type="decision", jurisdiction="new_south_wales",
        title=None, subtitle=None,
        segments=[SimpleNamespace(
            id="s1", kind="unit", type="paragraph", category="main",
            span=SimpleNamespace(start=0, end=len(text)), parent=None, children=[],
            level=0, type_name=None, code=None, title=None,
        )],
        crossreferences=[],
        locations=[],
        persons=[SimpleNamespace(
            id="p1", name=SimpleNamespace(start=0, end=end), type="natural", role="other",
            mentions=[SimpleNamespace(start=0, end=end)], parent=None, children=[], residence=None,
        )],
        emails=[], websites=[], phone_numbers=[], id_numbers=[],
        terms=[], external_documents=[], quotes=[], dates=[], headings=[], junk=[],
    )


class _FakeEnrichments:
    def __init__(self, calls: list[list[str]]):
        self._calls = calls

    def create(self, *, model, texts, overflow_strategy):
        self._calls.append(list(texts))
        results = [SimpleNamespace(document=_fake_document(t)) for t in texts]
        return SimpleNamespace(
            results=results, usage=SimpleNamespace(input_tokens=sum(len(t.split()) for t in texts)),
        )


class _FakeClient:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.enrichments = _FakeEnrichments(self.calls)


def _make_shard(tmp_path, records: list[dict]):
    out = tmp_path / "documents"
    ingest_records(records, out, _MAPPING, batch_size=500)
    return out / "batch-0001.parquet"


def test_small_docs_pack_into_one_request(tmp_path):
    records = [{"version_id": f"d{i}", "text": "a b c"} for i in range(3)]  # 3 tokens each
    shard = _make_shard(tmp_path, records)
    client = _FakeClient()
    cfg = EnrichmentConfig(max_texts_per_request=8, token_budget=100, split_ceiling=10_000)
    result = enrich_shards(shard.parent, cfg, client=client, token_counter=_FakeCounter())
    assert result.docs_enriched == 3
    # all three fit one request (9 tokens < 100, 3 docs < 8)
    assert len(client.calls) == 1
    assert len(client.calls[0]) == 3


def test_max_texts_per_request_caps_group(tmp_path):
    records = [{"version_id": f"d{i}", "text": "a"} for i in range(5)]
    shard = _make_shard(tmp_path, records)
    client = _FakeClient()
    cfg = EnrichmentConfig(max_texts_per_request=2, token_budget=1000, split_ceiling=10_000)
    enrich_shards(shard.parent, cfg, client=client, token_counter=_FakeCounter())
    assert [len(c) for c in client.calls] == [2, 2, 1]


def test_over_budget_doc_sent_solo(tmp_path):
    records = [
        {"version_id": "small1", "text": "a"},
        {"version_id": "big", "text": "a b c d e"},  # 5 tokens == budget
        {"version_id": "small2", "text": "a"},
    ]
    shard = _make_shard(tmp_path, records)
    client = _FakeClient()
    cfg = EnrichmentConfig(max_texts_per_request=8, token_budget=5, split_ceiling=10_000)
    enrich_shards(shard.parent, cfg, client=client, token_counter=_FakeCounter())
    # buffered small1 flushes, big goes solo, small2 trails
    assert [len(c) for c in client.calls] == [1, 1, 1]
    assert client.calls[1] == ["a b c d e"]


def test_long_doc_split_and_offset_merged(tmp_path):
    # 3 blocks of 4 tokens (12 total). budget 5 → doc is solo; ceiling 5 →
    # over-ceiling, split on blank lines to <=5-token segments = one per block.
    text = "aa bb cc dd\n\nee ff gg hh\n\nii jj kk ll"
    shard = _make_shard(tmp_path, [{"version_id": "long", "text": text}])
    client = _FakeClient()
    cfg = EnrichmentConfig(max_texts_per_request=8, token_budget=5, split_ceiling=5)
    result = enrich_shards(shard.parent, cfg, client=client, token_counter=_FakeCounter())
    assert result.docs_enriched == 1
    # doc split into 3 segment requests (each a solo segment enrich call)
    assert len(client.calls) == 3
    # merged entities carry offsets into the FULL narrative
    rows = read_enrichment_entities(shard).to_pylist()
    assert rows, "split doc produced no merged entities"
    # every mention offset is valid against the full text (max end <= len)
    assert max(r["mention_end"] for r in rows) <= len(text)
    # the second segment's person starts at its block offset, not 0
    starts = sorted(r["mention_start"] for r in rows)
    assert starts[0] == 0 and any(s > 0 for s in starts)


def test_group_failure_isolates_docs(tmp_path):
    records = [{"version_id": f"d{i}", "text": "a b"} for i in range(2)]
    shard = _make_shard(tmp_path, records)

    class _Boom(_FakeClient):
        def __init__(self):
            super().__init__()

            def _raise(**kwargs):
                raise RuntimeError("boom")

            self.enrichments.create = _raise  # type: ignore[assignment]

    client = _Boom()
    from womblex.store.checkpoint import CheckpointManager

    ckpt = CheckpointManager(tmp_path / ".ckpt", "t_enrich")
    ckpt.load()
    cfg = EnrichmentConfig(token_budget=1000, split_ceiling=10_000)
    result = enrich_shards(shard.parent, cfg, client=client, token_counter=_FakeCounter(),
                           checkpoint_mgr=ckpt)
    assert result.docs_enriched == 0
    # failed docs stay unprocessed so a resume retries them
    assert "d0" not in ckpt.state.processed_ids
    assert "d1" not in ckpt.state.processed_ids
