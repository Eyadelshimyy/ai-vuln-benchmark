"""One declared pipeline order, and the consumers that must follow it."""

from __future__ import annotations

import pytest

from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.pipeline_order import (
    DOWNSTREAM_STAGES,
    PIPELINE_ORDER,
    enabled_downstream_stages,
    in_pipeline_order,
    sort_by_pipeline,
    stage_rank,
)
from womblex.store.retention import STAGE_SUFFIXES


def _checkpoint_dirnames() -> dict[str, str]:
    """`ui.dashboard` pulls in FastAPI (the `[ui]` extra), so it is imported
    inside the tests that need it rather than at module scope."""
    pytest.importorskip("fastapi")
    from womblex.ui.dashboard import CHECKPOINT_DIRNAMES

    return CHECKPOINT_DIRNAMES


class TestPipelineOrder:
    def test_every_contract_has_a_place(self) -> None:
        """A new stage must be given a position, not silently sort last."""
        assert set(STAGE_CONTRACTS) <= set(PIPELINE_ORDER)

    def test_names_are_unique(self) -> None:
        assert len(set(PIPELINE_ORDER)) == len(PIPELINE_ORDER)

    def test_extraction_leads(self) -> None:
        assert PIPELINE_ORDER[0] == "extract"

    def test_enrich_precedes_chunk(self) -> None:
        """README: AI chunking reuses the enrichment, so enrich runs first.

        The edge this module exists for — it is config-derived (`chunk` reads
        `.enrichment_doc` only under `chunking.chunking_model`), so no sort
        over `required_inputs` recovers it.
        """
        assert stage_rank("enrich") < stage_rank("chunk")

    def test_text_overlays_precede_their_readers(self) -> None:
        """`processing.text_source` makes enrich *and* chunk read the overlay."""
        for reader in ("enrich", "chunk"):
            assert stage_rank("normalise") < stage_rank(reader)
            assert stage_rank("spellfix") < stage_rank(reader)
        assert stage_rank("normalise") < stage_rank("spellfix")

    def test_dependent_stages_follow_their_producers(self) -> None:
        assert stage_rank("chunk") < stage_rank("embed")
        assert stage_rank("chunk") < stage_rank("graph-refresh")
        assert stage_rank("enrich") < stage_rank("link")

    def test_pii_is_terminal_over_the_stages_needing_raw_text(self) -> None:
        """Masking is irreversible; enrich and embed must see the raw chunks."""
        for earlier in ("enrich", "embed", "chunk"):
            assert stage_rank(earlier) < stage_rank("pii")

    def test_quality_is_last(self) -> None:
        """Run-scoped: its duplicate clusters are corpus-wide."""
        assert PIPELINE_ORDER[-1] == "quality"

    def test_unknown_stage_sorts_last_without_raising(self) -> None:
        assert stage_rank("not-a-stage") == len(PIPELINE_ORDER)
        assert in_pipeline_order(["quality", "zzz", "extract"]) == (
            "extract", "quality", "zzz",
        )

    def test_sort_by_pipeline_preserves_values(self) -> None:
        mapping = {"chunk": 1, "extract": 2, "enrich": 3}
        assert sort_by_pipeline(mapping) == {"extract": 2, "enrich": 3, "chunk": 1}


class TestConsumersFollowIt:
    """The three places that had drifted into their own order."""

    def test_stage_suffixes_is_ordered(self) -> None:
        assert list(STAGE_SUFFIXES) == list(in_pipeline_order(STAGE_SUFFIXES))

    def test_checkpoint_dirnames_is_ordered(self) -> None:
        dirnames = _checkpoint_dirnames()
        assert list(dirnames) == list(in_pipeline_order(dirnames))

    def test_both_now_run_enrich_before_chunk(self) -> None:
        """The specific regression: both listed chunk first."""
        for mapping in (STAGE_SUFFIXES, _checkpoint_dirnames()):
            names = list(mapping)
            assert names.index("enrich") < names.index("chunk")


def _config(**sections):
    """A minimal `WomblexConfig` with the named sections overridden."""
    from womblex.config import WomblexConfig

    return WomblexConfig.model_validate({
        "dataset": {"name": "t"},
        "paths": {"input_root": ".", "output_root": ".", "checkpoint_dir": "."},
        **sections,
    })


class TestDispatchableStages:
    """What a dispatcher may enqueue, and what it must never enqueue."""

    def test_dispatchable_is_a_subset_of_the_pipeline(self) -> None:
        assert set(DOWNSTREAM_STAGES) <= set(PIPELINE_ORDER)

    def test_it_is_declared_in_pipeline_order(self) -> None:
        assert DOWNSTREAM_STAGES == in_pipeline_order(DOWNSTREAM_STAGES)

    def test_pii_and_quality_are_never_dispatched(self) -> None:
        """PII masking is irreversible and quality is run-scoped: neither may
        run because a flag was left on in a copied config. Both stay reachable
        through `womblex run-stage --stage …`."""
        assert "pii" not in DOWNSTREAM_STAGES
        assert "quality" not in DOWNSTREAM_STAGES
        assert {"pii", "quality"} <= set(STAGE_CONTRACTS)

    def test_extraction_is_not_a_downstream_stage(self) -> None:
        """It is the batch queue itself, dispatched per batch by `enqueue`."""
        assert "extract" not in DOWNSTREAM_STAGES

    def test_cli_choices_do_not_drift(self) -> None:
        from womblex.cli.cloud import DISPATCHABLE_STAGES

        assert DISPATCHABLE_STAGES == DOWNSTREAM_STAGES


class TestEnabledDownstreamStages:
    def test_defaults_dispatch_only_the_stages_defaulted_on(self) -> None:
        """chunk and money default on; enrich/embed/link/spellfix default off."""
        assert enabled_downstream_stages(_config()) == ("chunk", "money")

    def test_each_stage_answers_to_its_own_section(self) -> None:
        stages = enabled_downstream_stages(_config(
            enrichment={"enabled": True},
            embedding={"enabled": True},
            spellfix={"enabled": True},
            money={"enabled": False},
        ))
        assert stages == ("spellfix", "enrich", "chunk", "embed")

    def test_nothing_enabled_dispatches_nothing(self) -> None:
        assert enabled_downstream_stages(_config(
            chunking={"enabled": False}, money={"enabled": False},
        )) == ()

    def test_normalise_follows_the_text_source_that_reads_it(self) -> None:
        """It has no `enabled` flag — the gate is whether anything opens the
        overlay it writes."""
        assert "normalise" not in enabled_downstream_stages(_config())
        assert "normalise" in enabled_downstream_stages(
            _config(processing={"text_source": "normalised"})
        )
        assert "normalise" in enabled_downstream_stages(
            _config(processing={"text_source": "spellfix"})
        )
        # money's own override is a reader too.
        assert "normalise" in enabled_downstream_stages(
            _config(money={"text_source": "normalised"})
        )

    def test_graph_refresh_only_after_ai_chunking(self) -> None:
        """It re-links mentions onto chunks AI chunking rewrote; after plain
        semchunk the offsets enrich recorded still hold."""
        enriching = {"enrichment": {"enabled": True}}
        assert "graph-refresh" not in enabled_downstream_stages(_config(**enriching))
        assert "graph-refresh" in enabled_downstream_stages(_config(
            chunking={"chunking_model": "kanon-2-enricher"}, **enriching,
        ))
        # No enrichment to re-link.
        assert "graph-refresh" not in enabled_downstream_stages(_config(
            chunking={"chunking_model": "kanon-2-enricher"},
        ))

    def test_the_result_is_always_ordered(self) -> None:
        stages = enabled_downstream_stages(_config(
            enrichment={"enabled": True}, embedding={"enabled": True},
            linking={"enabled": True}, spellfix={"enabled": True},
            processing={"text_source": "spellfix"},
        ))
        assert stages == in_pipeline_order(stages)
        assert stages.index("normalise") < stages.index("spellfix")
        assert stages.index("enrich") < stages.index("chunk")
