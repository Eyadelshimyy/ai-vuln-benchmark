"""Declarative contracts for the remote shard-stage runner.

Each downstream `*_shards()` stage is described here as data: what it reads,
what it writes, whether those sets overlap, whether it needs the Isaacus API,
and how much of a run it needs in one pass. The runner
(:mod:`womblex.cloud.stage_runner`) executes against these declarations; no
stage function is modified and no filesystem abstraction is threaded through
them — the same stage-in/stage-out shape ``womblex finalize`` already uses.

Two things are deliberately *functions of config*, not of stage name:

- :attr:`StageContract.conditional_inputs` — ``chunk`` reads
  ``*.enrichment_doc.parquet`` only when ``chunking.chunking_model`` is set;
  ``chunk`` / ``enrich`` / ``money`` read a text-overlay sidecar chosen by
  ``processing.text_source`` (and ``money.text_source`` outranks it).
- :attr:`StageContract.outputs` — ``pii`` writes ``*.clean_text.parquet`` only
  when ``write_clean_text``; ``enrich`` writes ``*.enrichment_doc.parquet``
  only when ``persist_document``.

Resolving either from the stage name alone would download the wrong files and,
worse, make the runner's output-exists skip fire on an incomplete set.

``manifest`` is absent by design: ``womblex finalize`` already downloads every
``*._manifest.parquet``, consolidates, and uploads ``manifest.parquet``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

# `_SUFFIX` / `_SHARD_SUFFIX` are imported rather than re-spelled so a schema
# change in `store/` or `process/text_overlay.py` reaches the runner
# automatically — the same cross-module private-import idiom the stages already
# use for `chunk_stage._batch_bases`.
from womblex.process.text_overlay import _SUFFIX as _OVERLAY_SUFFIX
from womblex.store.embed_output import EMBEDDINGS_SUFFIX
from womblex.store.enrichment_doc import ENRICHMENT_DOC_SUFFIX
from womblex.store.enrichment_output import (
    ENRICHMENT_ENTITIES_SUFFIX,
    ENRICHMENT_META_SUFFIX,
    GRAPH_EDGES_SUFFIX,
)
from womblex.store.entity_links_output import ENTITY_LINKS_SUFFIX
from womblex.store.layout_output import LAYOUT_REGIONS_SUFFIX
from womblex.store.money_output import MONEY_COLUMNS_SUFFIX, MONEY_SPANS_SUFFIX
from womblex.store.normalise_output import NORMALISED_TEXT_SUFFIX
from womblex.store.output import _SHARD_ROLES, _SHARD_SUFFIX, CHUNKS_SUFFIX
from womblex.store.pii_output import CLEAN_TEXT_SUFFIX, PII_SPANS_SUFFIX
from womblex.store.quality_output import CHUNK_QUALITY_SUFFIX
from womblex.store.spellfix_output import SPELLFIX_CORRECTIONS_SUFFIX, SPELLFIX_TEXT_SUFFIX

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import WomblexConfig
    from womblex.store.checkpoint import CheckpointManager

ELEMENTS_SUFFIX = _SHARD_SUFFIX["elements"]
TABLE_CELLS_SUFFIX = _SHARD_SUFFIX["table_cells"]
FORM_FIELDS_SUFFIX = _SHARD_SUFFIX["form_fields"]
MANIFEST_SUFFIX = _SHARD_SUFFIX["manifest"]

#: Suffixes that make a batch base discoverable. Downstream sidecars are never
#: in this set — a `*.chunks.parquet` with no extraction sibling is not a batch.
DISCOVERY_SUFFIXES: tuple[str, ...] = tuple(_SHARD_SUFFIX[r] for r in _SHARD_ROLES)


class StageScope(Enum):
    """How much of a run the stage needs staged in one pass."""

    PER_BATCH = "per_batch"
    """One batch base at a time. No whole-run staging."""

    WHOLE_RUN = "whole_run"
    """Every base at once. `quality` only — its dedup clusters are corpus-wide."""


class MutationMode(Enum):
    """Whether the stage's outputs are disjoint from its inputs."""

    SIDECAR = "sidecar"
    """New siblings. Output-exists skip applies."""

    IN_PLACE = "in_place"
    """Outputs are a subset of inputs. Never skipped; idempotency carries re-runs."""


@dataclass(frozen=True)
class ConditionalInput:
    """An input the stage reads only under some configurations.

    ``strict`` marks the case where absence is wrong rather than merely
    degraded: running anyway would build a sidecar from a text layer the config
    did not select. The runner refuses the stage before any base runs.
    """

    suffix: str
    strict: bool
    reason: str


def _no_models(_config: WomblexConfig) -> tuple[str, ...]:
    """Stage calls no Isaacus model."""
    return ()


def _chunk_needs_api(config: WomblexConfig) -> bool:
    """``chunk`` needs the Isaacus API only when AI chunking is enabled.

    Plain token chunking sizes chunks with the vendored kanon-2 tokeniser and
    runs fully offline, so a keyless local run must still chunk. AI chunking
    (``chunking.chunking_model`` set) calls the enricher API per document at
    chunk time, so it needs a deployment. The static ``needs_isaacus_api=True``
    on the contract is the *display* answer (chunk can use the API); this is
    the *runtime-gate* answer, resolved from config.
    """
    return bool(config.chunking.chunking_model)


@dataclass
class RunContext:
    """Runtime dependencies the runner constructs once, before any base."""

    client: object | None = None
    checkpoint_mgr: CheckpointManager | None = None
    # Source-document stages only (`layout`): where the corpus is, if not where
    # the manifest recorded it; a fetcher that stages one document from the
    # ingest store (the distributed runner's); and whether to rerun an
    # unchanged fingerprint.
    source_root: str | None = None
    source_fetcher: Callable[[dict], Path] | None = None
    force: bool = False


@dataclass(frozen=True)
class StageContract:
    """Everything the runner needs to execute one stage against a store."""

    name: str
    scope: StageScope
    mutation: MutationMode
    required_inputs: tuple[str, ...]
    conditional_inputs: Callable[[WomblexConfig], tuple[ConditionalInput, ...]]
    outputs: Callable[[WomblexConfig], tuple[str, ...]]
    run: Callable[[Path, WomblexConfig, RunContext], None]
    needs_isaacus_api: bool = False
    needs_client: bool = False
    # Isaacus model ids the stage calls. Checked against the deployed endpoints
    # when the run targets a SageMaker deployment (subscriptions are per model,
    # so an endpoint need not serve all of them); ignored on the hosted API.
    models: Callable[[WomblexConfig], tuple[str, ...]] = _no_models
    checkpoint_dirname: str | None = None
    preflight: Callable[[WomblexConfig], None] | None = None
    # Config-aware override of the runtime Isaacus gate. When set, the runner's
    # pre-flight consults this instead of the static ``needs_isaacus_api`` flag
    # — the flag stays the coarse "this stage can call the API" answer used for
    # display (the composer graph), while this resolves whether *this config*
    # actually will (e.g. ``chunk`` calls the API only under AI chunking).
    needs_isaacus_api_for: Callable[[WomblexConfig], bool] | None = None
    # The stage re-reads the source documents, not just the shards: the runner
    # hands it a fetcher for the ingest store (`RunContext.source_fetcher`).
    needs_sources: bool = False

    def requires_isaacus_api(self, config: WomblexConfig) -> bool:
        """Whether *config* makes this stage call the Isaacus API at runtime."""
        if self.needs_isaacus_api_for is not None:
            return self.needs_isaacus_api_for(config)
        return self.needs_isaacus_api

    def input_suffixes(self, config: WomblexConfig) -> tuple[str, ...]:
        """Required + conditional inputs for *config*, de-duplicated, in order."""
        seen: list[str] = list(self.required_inputs)
        for c in self.conditional_inputs(config):
            if c.suffix not in seen:
                seen.append(c.suffix)
        return tuple(seen)


# ---------------------------------------------------------------------------
# Conditional-input resolvers
# ---------------------------------------------------------------------------


def _overlay_input(text_source: str) -> tuple[ConditionalInput, ...]:
    """The element-text overlay sidecar selected by *text_source*, if any."""
    suffix = _OVERLAY_SUFFIX.get(text_source)
    if suffix is None:  # "elements" — verbatim, no overlay
        return ()
    return (
        ConditionalInput(
            suffix,
            strict=True,
            reason=f"processing.text_source={text_source!r}",
        ),
    )


def _chunk_conditional(config: WomblexConfig) -> tuple[ConditionalInput, ...]:
    inputs = _overlay_input(config.processing.text_source)
    if config.chunking.chunking_model:
        # Ordering requirement, not a hard dependency: without the sidecar the
        # chunker self-enriches (double cost, same output). Warn, don't fail.
        inputs += (
            ConditionalInput(
                ENRICHMENT_DOC_SUFFIX,
                strict=False,
                reason=f"chunking.chunking_model={config.chunking.chunking_model!r}",
            ),
        )
    return inputs


def _money_conditional(config: WomblexConfig) -> tuple[ConditionalInput, ...]:
    # money's own text_source outranks the pipeline-level one (money_stage.py).
    return _overlay_input(config.money.text_source or config.processing.text_source)


def _enrich_conditional(config: WomblexConfig) -> tuple[ConditionalInput, ...]:
    # Chunks are optional: present, they add mention->chunk edges to the graph.
    return _overlay_input(config.processing.text_source) + (
        ConditionalInput(CHUNKS_SUFFIX, strict=False, reason="adds mention->chunk edges"),
    )


def _spellfix_conditional(_config: WomblexConfig) -> tuple[ConditionalInput, ...]:
    # Hardcoded chain off the normalise layer with warn_if_missing=False, so
    # absence is legitimate — spellfix then repairs verbatim element text.
    return (
        ConditionalInput(NORMALISED_TEXT_SUFFIX, strict=False, reason="normalise->spellfix chain"),
    )


def _evidence_inputs(config: WomblexConfig) -> tuple[ConditionalInput, ...]:
    """What locating a span in its source needs: the elements, cells and text layer.

    Absent elements leave a span located only as far as its chunk or document
    allows (warned); a declared text layer that is missing refuses the stage, as
    it does for chunk and enrich.
    """
    return _overlay_input(config.processing.text_source) + tuple(
        ConditionalInput(suffix, strict=False, reason="locates spans in the source (evidence)")
        for suffix in (ELEMENTS_SUFFIX, TABLE_CELLS_SUFFIX)
    )


def _pii_conditional(config: WomblexConfig) -> tuple[ConditionalInput, ...]:
    # Optional in code, load-bearing in practice — the Kanon-2 graph is the
    # primary candidate source; without it only the opt-in backstop fires.
    return (
        ConditionalInput(
            ENRICHMENT_ENTITIES_SUFFIX, strict=False, reason="graph is the primary PII source",
        ),
    ) + _evidence_inputs(config)


def _no_conditional(_config: WomblexConfig) -> tuple[ConditionalInput, ...]:
    return ()


def _layout_conditional(_config: WomblexConfig) -> tuple[ConditionalInput, ...]:
    # The sidecar being replaced: its footer fingerprint drives the skip rule.
    return (
        ConditionalInput(LAYOUT_REGIONS_SUFFIX, strict=False, reason="fingerprint-aware skip"),
    )


# ---------------------------------------------------------------------------
# Output resolvers
# ---------------------------------------------------------------------------


def _enrich_outputs(config: WomblexConfig) -> tuple[str, ...]:
    outputs: tuple[str, ...] = (
        ENRICHMENT_ENTITIES_SUFFIX, ENRICHMENT_META_SUFFIX, GRAPH_EDGES_SUFFIX,
    )
    if _persist_document(config):
        outputs += (ENRICHMENT_DOC_SUFFIX,)
    return outputs


def _pii_outputs(config: WomblexConfig) -> tuple[str, ...]:
    outputs: tuple[str, ...] = (PII_SPANS_SUFFIX,)
    if config.pii.write_clean_text:
        outputs += (CLEAN_TEXT_SUFFIX,)
    return outputs


def _persist_document(config: WomblexConfig) -> bool:
    """Mirror ``cli/link.py``: persist whenever this config also enables AI chunking."""
    return config.enrichment.persist_document or bool(config.chunking.chunking_model)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _link_preflight(config: WomblexConfig) -> None:
    """The reference register is a worker-local file, not a store object."""
    reference = config.linking.reference
    if reference is None:
        raise ValueError(
            "link requires linking.reference in the config (the reference register "
            "declaration); none is set."
        )
    if not Path(reference.path).exists():
        raise FileNotFoundError(
            f"linking.reference.path does not resolve on this host: {reference.path}. "
            "The reference register is read from the local filesystem, not the store."
        )
    if reference.alias_table is not None and not Path(reference.alias_table).exists():
        raise FileNotFoundError(
            f"linking.reference.alias_table does not resolve on this host: {reference.alias_table}"
        )


# ---------------------------------------------------------------------------
# Stage bodies — each calls the unchanged `*_shards()` function
# ---------------------------------------------------------------------------


def _run_normalise(shard_dir: Path, config: WomblexConfig, ctx: RunContext) -> None:
    from womblex.process.normalise_stage import normalise_shards

    normalise_shards(shard_dir, config.normalise, checkpoint_mgr=ctx.checkpoint_mgr)


def _run_spellfix(shard_dir: Path, config: WomblexConfig, ctx: RunContext) -> None:
    from womblex.process.spellfix_stage import spellfix_shards

    spellfix_shards(shard_dir, config.spellfix, checkpoint_mgr=ctx.checkpoint_mgr)


def _run_chunk(shard_dir: Path, config: WomblexConfig, ctx: RunContext) -> None:
    from womblex.process.chunk_stage import chunk_shards

    chunk_shards(
        shard_dir, config.chunking,
        text_source=config.processing.text_source,
        checkpoint_mgr=ctx.checkpoint_mgr,
    )


def _run_money(shard_dir: Path, config: WomblexConfig, ctx: RunContext) -> None:
    from womblex.process.money_stage import money_shards

    money_shards(
        shard_dir, config.money,
        text_source=config.processing.text_source,
        checkpoint_mgr=ctx.checkpoint_mgr,
    )


def _run_enrich(shard_dir: Path, config: WomblexConfig, ctx: RunContext) -> None:
    from womblex.analyse.enrich_stage import enrich_shards

    enrich_shards(
        shard_dir, config.enrichment,
        client=ctx.client,
        text_source=config.processing.text_source,
        persist_document=_persist_document(config),
        checkpoint_mgr=ctx.checkpoint_mgr,
    )


def _run_embed(shard_dir: Path, config: WomblexConfig, ctx: RunContext) -> None:
    from womblex.analyse.embed_stage import embed_shards

    embed_shards(shard_dir, config.embedding, client=ctx.client,
                 checkpoint_mgr=ctx.checkpoint_mgr)


def _run_link(shard_dir: Path, config: WomblexConfig, ctx: RunContext) -> None:
    from womblex.link.stage import link_shards

    link_shards(
        shard_dir, config.linking,
        text_source=config.processing.text_source, checkpoint_mgr=ctx.checkpoint_mgr,
    )


def _run_pii(shard_dir: Path, config: WomblexConfig, ctx: RunContext) -> None:
    from womblex.pii.pii_stage import pii_shards

    pii_shards(
        shard_dir, config.pii,
        text_source=config.processing.text_source, checkpoint_mgr=ctx.checkpoint_mgr,
    )


def _run_layout(shard_dir: Path, config: WomblexConfig, ctx: RunContext) -> None:
    from womblex.process.layout_stage import layout_shards

    result = layout_shards(
        shard_dir, config, source_root=ctx.source_root,
        source_fetcher=ctx.source_fetcher, force=ctx.force,
    )
    if result.batches_failed:
        raise RuntimeError(
            f"layout: {result.batches_failed} batch(es) not analysed; they are left as they "
            "were (the log names each document)"
        )


def _run_graph_refresh(shard_dir: Path, _config: WomblexConfig, ctx: RunContext) -> None:
    from womblex.analyse.graph_refresh import refresh_graph_edges

    refresh_graph_edges(shard_dir, checkpoint_mgr=ctx.checkpoint_mgr)


def _run_quality(shard_dir: Path, config: WomblexConfig, _ctx: RunContext) -> None:
    from womblex.process.quality_stage import quality_shards

    quality_shards(shard_dir, config.quality)


# ---------------------------------------------------------------------------
# The inventory
# ---------------------------------------------------------------------------

_ELEMENT_INPUTS = (ELEMENTS_SUFFIX, TABLE_CELLS_SUFFIX, MANIFEST_SUFFIX)

STAGE_CONTRACTS: dict[str, StageContract] = {
    # Replaces the extraction batch's own layout sidecar, so it never takes the
    # output-exists skip (IN_PLACE); its fingerprint check is the skip rule.
    "layout": StageContract(
        name="layout",
        scope=StageScope.PER_BATCH,
        mutation=MutationMode.IN_PLACE,
        required_inputs=(ELEMENTS_SUFFIX, MANIFEST_SUFFIX),
        conditional_inputs=_layout_conditional,
        outputs=lambda _c: (LAYOUT_REGIONS_SUFFIX,),
        run=_run_layout,
        needs_sources=True,
    ),
    "normalise": StageContract(
        name="normalise",
        scope=StageScope.PER_BATCH,
        mutation=MutationMode.SIDECAR,
        required_inputs=_ELEMENT_INPUTS,
        conditional_inputs=_no_conditional,
        outputs=lambda _c: (NORMALISED_TEXT_SUFFIX,),
        run=_run_normalise,
        checkpoint_dirname=".normalise-checkpoint",
    ),
    "spellfix": StageContract(
        name="spellfix",
        scope=StageScope.PER_BATCH,
        mutation=MutationMode.SIDECAR,
        required_inputs=_ELEMENT_INPUTS,
        conditional_inputs=_spellfix_conditional,
        outputs=lambda _c: (SPELLFIX_TEXT_SUFFIX, SPELLFIX_CORRECTIONS_SUFFIX),
        run=_run_spellfix,
        checkpoint_dirname=".spellfix-checkpoint",
    ),
    "chunk": StageContract(
        name="chunk",
        scope=StageScope.PER_BATCH,
        mutation=MutationMode.SIDECAR,
        required_inputs=_ELEMENT_INPUTS,
        conditional_inputs=_chunk_conditional,
        outputs=lambda _c: (CHUNKS_SUFFIX,),
        run=_run_chunk,
        # `needs_isaacus_api=True` is the display answer (chunk *can* call the
        # enricher API, under AI chunking). The runtime gate is config-aware:
        # plain token chunking uses the vendored tokeniser and runs offline, so
        # only AI chunking (`chunking_model`) actually needs a deployment.
        needs_isaacus_api=True,
        needs_isaacus_api_for=_chunk_needs_api,
        checkpoint_dirname=".chunk-checkpoint",
    ),
    "money": StageContract(
        name="money",
        scope=StageScope.PER_BATCH,
        mutation=MutationMode.SIDECAR,
        required_inputs=_ELEMENT_INPUTS,
        conditional_inputs=_money_conditional,
        outputs=lambda _c: (MONEY_SPANS_SUFFIX, MONEY_COLUMNS_SUFFIX),
        run=_run_money,
        checkpoint_dirname=".money-checkpoint",
    ),
    "enrich": StageContract(
        name="enrich",
        scope=StageScope.PER_BATCH,
        mutation=MutationMode.SIDECAR,
        required_inputs=(ELEMENTS_SUFFIX, MANIFEST_SUFFIX),
        conditional_inputs=_enrich_conditional,
        outputs=_enrich_outputs,
        run=_run_enrich,
        needs_isaacus_api=True,
        needs_client=True,
        models=lambda c: (c.enrichment.model,),
        checkpoint_dirname=".enrich-checkpoint",
    ),
    "embed": StageContract(
        name="embed",
        scope=StageScope.PER_BATCH,
        mutation=MutationMode.SIDECAR,
        required_inputs=(CHUNKS_SUFFIX, MANIFEST_SUFFIX),
        conditional_inputs=_no_conditional,
        outputs=lambda _c: (EMBEDDINGS_SUFFIX,),
        run=_run_embed,
        needs_isaacus_api=True,
        needs_client=True,
        models=lambda c: (c.embedding.model,),
        checkpoint_dirname=".embed-checkpoint",
    ),
    "link": StageContract(
        name="link",
        scope=StageScope.PER_BATCH,
        mutation=MutationMode.SIDECAR,
        required_inputs=(ENRICHMENT_ENTITIES_SUFFIX, MANIFEST_SUFFIX),
        conditional_inputs=_evidence_inputs,
        outputs=lambda _c: (ENTITY_LINKS_SUFFIX,),
        run=_run_link,
        checkpoint_dirname=".link-checkpoint",
        preflight=_link_preflight,
    ),
    "pii": StageContract(
        name="pii",
        scope=StageScope.PER_BATCH,
        mutation=MutationMode.SIDECAR,
        required_inputs=(CHUNKS_SUFFIX, MANIFEST_SUFFIX),
        conditional_inputs=_pii_conditional,
        outputs=_pii_outputs,
        run=_run_pii,
        checkpoint_dirname=".pii-checkpoint",
    ),
    "graph-refresh": StageContract(
        name="graph-refresh",
        scope=StageScope.PER_BATCH,
        mutation=MutationMode.IN_PLACE,
        required_inputs=(
            ENRICHMENT_ENTITIES_SUFFIX, GRAPH_EDGES_SUFFIX, CHUNKS_SUFFIX, MANIFEST_SUFFIX,
        ),
        conditional_inputs=_no_conditional,
        # A subset of the inputs: the stage rewrites both sidecars in place.
        outputs=lambda _c: (ENRICHMENT_ENTITIES_SUFFIX, GRAPH_EDGES_SUFFIX),
        run=_run_graph_refresh,
        checkpoint_dirname=".graph-refresh-checkpoint",
    ),
    "quality": StageContract(
        name="quality",
        scope=StageScope.WHOLE_RUN,
        mutation=MutationMode.SIDECAR,
        required_inputs=(CHUNKS_SUFFIX,),
        conditional_inputs=_no_conditional,
        outputs=lambda _c: (CHUNK_QUALITY_SUFFIX,),
        run=_run_quality,
        # No CheckpointManager parameter — a partial corpus changes the clusters.
        checkpoint_dirname=None,
    ),
}

#: Stage that most plausibly produces each required input, for the
#: "you ran these out of order" error message.
PRODUCER_OF: dict[str, str] = {
    CHUNKS_SUFFIX: "chunk",
    ENRICHMENT_ENTITIES_SUFFIX: "enrich",
    GRAPH_EDGES_SUFFIX: "enrich",
    NORMALISED_TEXT_SUFFIX: "normalise",
    SPELLFIX_TEXT_SUFFIX: "spellfix",
    ENRICHMENT_DOC_SUFFIX: "enrich",
    LAYOUT_REGIONS_SUFFIX: "layout",
}

STAGE_NAMES: tuple[str, ...] = tuple(STAGE_CONTRACTS)


__all__ = [
    "DISCOVERY_SUFFIXES",
    "PRODUCER_OF",
    "STAGE_CONTRACTS",
    "STAGE_NAMES",
    "ConditionalInput",
    "MutationMode",
    "RunContext",
    "StageContract",
    "StageScope",
]
