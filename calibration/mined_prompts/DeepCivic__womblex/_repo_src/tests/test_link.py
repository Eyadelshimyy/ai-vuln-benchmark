"""Tests for the entity-link stage (normalise / reference / matcher / IO).

Grounded in the *real* ACT childcare register values for the Artemis
service (public data: SE-40002132 / PR-40030037 @ 11 Cessnock St,
Fyshwick) so the matcher is exercised against genuine surface forms — the
same name variants + address Kanon-2 produced in the I6 spike — not
invented entities. Pure-function tests need no document fixture; the
end-to-end link_shards test builds a shard with a canned (mocked)
enrichment sidecar.
"""

from __future__ import annotations

import pytest

from womblex.config import ReferenceConfig
from womblex.link.matcher import Candidate, resolve
from womblex.link.normalise import normalise_address, normalise_name
from womblex.link.reference import load_reference
from womblex.store.entity_links_output import (
    ENTITY_LINKS_SCHEMA,
    entity_links_path_for,
    read_entity_links,
    write_entity_links,
)

# --- real register row (ACT public data) ------------------------------------

_REGISTER_CSV = (
    "ServiceApprovalNumber,Provider Approval Number,ServiceName,ProviderLegalName,"
    "ServiceAddress,Suburb,Postcode\n"
    "SE-40002132,PR-40030037,Artemis Early Learning,ARTEMIS EDUCATION PTY LTD,"
    "11 Cessnock St,FYSHWICK,2609\n"
    # Real cross-brand decoy that shares the generic tokens "Early Learning
    # Centre" — exercises that the distinctive brand token wins.
    "SE-40007473,PR-00005847,Urambi Early Learning Centre,Majura Park Childcare Centre Pty Ltd,"
    "1 Crozier Cct,KAMBAH,2902\n"
    "SE-99999999,PR-99999999,Decoy Childcare,DECOY PTY LTD,1 Nowhere Rd,SOMETOWN,9999\n"
)


@pytest.fixture
def reference_config(tmp_path) -> ReferenceConfig:
    csv_path = tmp_path / "services.csv"
    csv_path.write_text(_REGISTER_CSV, encoding="utf-8")
    return ReferenceConfig(
        path=csv_path,
        id_col="ServiceApprovalNumber",
        name_col="ServiceName",
        entity_type="service",
        parent_id_col="Provider Approval Number",
        match_exact_cols=["ServiceAddress", "Suburb", "Postcode"],
        match_fuzzy_cols=["ServiceName", "ProviderLegalName"],
    )


# --- normalise ---------------------------------------------------------------


class TestNormalise:
    def test_address_collapses_state_and_abbrev(self):
        # K2 candidate form and register concat form collapse to the same key.
        cand = normalise_address("11 Cessnock st, Fyshwick ACT 2609")
        ref = normalise_address("11 Cessnock St FYSHWICK 2609")
        assert cand == ref == "11 cessnock street fyshwick 2609"

    def test_address_drops_po_box(self):
        assert normalise_address("PO Box 6270, O'Connor ACT 2602") == "o connor 2602"

    def test_name_casefold_punct(self):
        assert normalise_name("ARTEMIS EDUCATION PTY LTD") == "artemis education pty ltd"

    def test_name_newline_collapsed(self):
        assert normalise_name("Artemis Early\nLearning Fyshwick") == "artemis early learning fyshwick"

    def test_empty_safe(self):
        assert normalise_name(None) == "" and normalise_address(None) == ""


# --- reference loading -------------------------------------------------------


class TestReferenceLoad:
    def test_entities_and_roles(self, reference_config):
        ref = load_reference(reference_config)
        assert len(ref.entities) == 3
        artemis = ref.entity_by_id("SE-40002132")
        assert artemis is not None
        assert artemis.parent_id == "PR-40030037"
        assert artemis.entity_type == "service"
        assert artemis.name == "Artemis Early Learning"
        assert artemis.exact_key == "11 cessnock street fyshwick 2609"
        assert "artemis education pty ltd" in artemis.fuzzy_keys

    def test_unimplemented_format_raises(self, tmp_path):
        cfg = ReferenceConfig(
            path=tmp_path / "x.shp", format="shapefile",
            id_col="id", name_col="name",
        )
        with pytest.raises(NotImplementedError):
            load_reference(cfg)


# --- matcher -----------------------------------------------------------------


class TestMatcher:
    def _ref(self, reference_config):
        return load_reference(reference_config)

    def test_address_exact_beats_ocr_names(self, reference_config):
        ref = self._ref(reference_config)
        c = Candidate("11 Cessnock st, Fyshwick ACT 2609", "address", "h1")
        [link] = resolve([c], ref)
        assert link.matched and link.method == "address_exact"
        assert link.entity.entity_id == "SE-40002132"
        assert link.confidence == 1.0

    def test_name_fuzzy_legalname_variant(self, reference_config):
        ref = self._ref(reference_config)
        # K2's spaced-out "PTYLTD" form resolves to the legal-name row.
        c = Candidate("ARTEMIS EDUCATION PTYLTD", "corporate", "h1")
        [link] = resolve([c], ref)
        assert link.matched and link.method == "name_fuzzy"
        assert link.entity.entity_id == "SE-40002132"
        assert link.confidence >= 0.85

    def test_name_fuzzy_picks_correct_of_two(self, reference_config):
        ref = self._ref(reference_config)
        c = Candidate("Decoy Childcare", "corporate", "h1")
        [link] = resolve([c], ref)
        assert link.matched and link.entity.entity_id == "SE-99999999"

    def test_token_superset_suburb_suffix(self, reference_config):
        # "Artemis Early Learning Fyshwick" scores only ~0.83 on raw difflib
        # (the suburb suffix), but the token-set ratio recovers it as a
        # superset of the register's "Artemis Early Learning".
        ref = self._ref(reference_config)
        c = Candidate("Artemis Early Learning Fyshwick", "corporate", "h1")
        [link] = resolve([c], ref)
        assert link.matched and link.method == "name_fuzzy"
        assert link.entity.entity_id == "SE-40002132"

    def test_ocr_typo_token_tolerated(self, reference_config):
        # OCR misread "Early" -> "Earty"; intra-token char similarity covers it.
        ref = self._ref(reference_config)
        c = Candidate("Artemis Earty Learning Fyshwick", "corporate", "h1")
        [link] = resolve([c], ref)
        assert link.matched and link.entity.entity_id == "SE-40002132"

    def test_ocr_tolerance_does_not_cross_brand(self, reference_config):
        # OCR tolerance must NOT let a different brand through: "urambi" is not
        # a fuzzy variant of "artemis", so this stays on Urambi only.
        ref = self._ref(reference_config)
        c = Candidate("Urambi Early Learning Centre", "corporate", "h1")
        [link] = resolve([c], ref)
        assert link.matched and link.entity.entity_id == "SE-40007473"

    def test_generic_tokens_pick_distinctive_brand(self, reference_config):
        # Shares "Early Learning Centre" with Urambi, but the brand token
        # "Artemis" must win — no cross-brand false match.
        ref = self._ref(reference_config)
        c = Candidate("Artemis Early Learning Centre", "corporate", "h1")
        [link] = resolve([c], ref)
        assert link.matched and link.entity.entity_id == "SE-40002132"

    def test_below_threshold_unmatched(self, reference_config):
        ref = self._ref(reference_config)
        c = Candidate("Totally Unrelated Community Centre", "corporate", "h1")
        [link] = resolve([c], ref)
        assert not link.matched and link.method == "unmatched"

    def test_address_no_hit_unmatched(self, reference_config):
        ref = self._ref(reference_config)
        c = Candidate("42 Imaginary Way, Nowhere 0000", "address", "h1")
        [link] = resolve([c], ref)
        assert not link.matched

    def test_alias_override(self, reference_config):
        ref = load_reference(reference_config)
        # Prior-trustee name the register doesn't carry; alias resolves it.
        ref.aliases[normalise_name("Canberra Childcare Pty Ltd ATF The Fyshwick Child Care Trust")] = "SE-40002132"
        c = Candidate("Canberra Childcare Pty Ltd ATF The Fyshwick Child Care Trust", "corporate", "h1")
        [link] = resolve([c], ref)
        assert link.matched and link.method == "alias"
        assert link.entity.entity_id == "SE-40002132"

    def test_multi_candidate_doc(self, reference_config):
        ref = self._ref(reference_config)
        cands = [
            Candidate("Artemis Early Learning Centre", "corporate", "h1"),
            Candidate("11 Cessnock st, Fyshwick ACT 2609", "address", "h1"),
        ]
        links = resolve(cands, ref)
        assert all(lk.matched for lk in links)
        assert {lk.entity.entity_id for lk in links} == {"SE-40002132"}


# --- entity_links IO + derived doc view --------------------------------------


class TestEntityLinksIO:
    def _rows(self):
        return [
            {"source_hash": "h1", "candidate_text": "Artemis Early Learning",
             "candidate_kind": "corporate", "mention_start": 0, "mention_end": 5,
             "entity_id": "SE-40002132", "entity_type": "service",
             "canonical_name": "Artemis Early Learning", "parent_entity_id": "PR-40030037",
             "confidence": 0.95, "match_method": "name_fuzzy", "matched": True},
            {"source_hash": "h1", "candidate_text": "11 Cessnock St",
             "candidate_kind": "address", "mention_start": 10, "mention_end": 24,
             "entity_id": "SE-40002132", "entity_type": "service",
             "canonical_name": "Artemis Early Learning", "parent_entity_id": "PR-40030037",
             "confidence": 1.0, "match_method": "address_exact", "matched": True},
            {"source_hash": "h1", "candidate_text": "Some Council",
             "candidate_kind": "corporate", "mention_start": 30, "mention_end": 42,
             "entity_id": "", "entity_type": "", "canonical_name": "",
             "parent_entity_id": "", "confidence": 0.0, "match_method": "unmatched",
             "matched": False},
        ]

    def test_span_roundtrip(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        write_entity_links(self._rows(), base)
        assert entity_links_path_for(base).exists()
        t = read_entity_links(base, grain="span")
        assert t.schema.equals(ENTITY_LINKS_SCHEMA)
        assert t.num_rows == 3

    def test_doc_view_groups_and_drops_unmatched(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        write_entity_links(self._rows(), base)
        doc = read_entity_links(base, grain="doc").to_pylist()
        assert len(doc) == 1  # two matched mentions of one entity -> one row; unmatched dropped
        row = doc[0]
        assert row["source_hash"] == "h1" and row["entity_id"] == "SE-40002132"
        assert row["mention_count"] == 2
        assert row["max_confidence"] == pytest.approx(1.0)

    def test_empty_dir_safe(self, tmp_path):
        assert read_entity_links(tmp_path, grain="span").num_rows == 0
        assert read_entity_links(tmp_path, grain="doc").num_rows == 0

    def test_bad_grain_raises(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        write_entity_links([], base)
        with pytest.raises(ValueError):
            read_entity_links(base, grain="nonsense")
