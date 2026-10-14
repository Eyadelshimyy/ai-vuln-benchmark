"""The order the pipeline's stages run in — declared once.

`StageContract` knows each stage's *inputs*, not its *place*. That is
deliberate (`run-stage` documents "ordering is yours to pick"), and it holds
because the one edge that matters most is not expressible as an input at all:
`chunk` reads `.enrichment_doc` only when `chunking.chunking_model` is set, so
enrich→chunk is config-derived and a topological sort over `required_inputs`
alone will happily put chunk first.

So the order has to be written down. It was — three times, in three places,
disagreeing: `STAGE_CONTRACTS`'s dict order, `store.retention.STAGE_SUFFIXES`
(commented "in pipeline order") and the README's command sequence. The first
two both ran chunk before enrich, and the console rendered its lifecycle
checkpoints in that order. This module is the README's sequence — the correct
one — as the single declaration the others now derive from.

`sort_by_pipeline` rather than an exported list of pairs, because every
consumer so far holds its own stage→something mapping (a sidecar suffix, a
checkpoint dirname) and wants that mapping ordered, not replaced.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import WomblexConfig

#: Every pipeline stage, in the order a full run executes them. Extraction
#: leads (it produces what all the rest read); the two non-obvious edges are:
#:
#: - **normalise and spellfix precede enrich and chunk**, because they write
#:   the element-text overlays `processing.text_source` makes *both* of those
#:   stages reassemble from. Running them later leaves the overlay on disk and
#:   unread — the silent no-op that makes a config setting look broken.
#:   spellfix chains on top of normalised, so it follows it.
#: - **enrich precedes chunk**, so AI chunking reuses the persisted ILGS
#:   Document rather than paying for a second enrichment pass (README: "enrich
#:   must precede chunk so AI chunking reuses the enrichment (no double
#:   cost)").
#:
#: The rest follow their hard input edges: `graph-refresh` re-links mentions
#: onto the chunks `chunk` just wrote, `embed` vectorises those chunks, `link`
#: matches the entities `enrich` wrote, and `pii` is terminal — its masking is
#: irreversible and must not precede the stages that need the raw text.
#: `money` reads only extraction output, so its position is free; it sits with
#: the other offline stages. `quality` is last because it is the one
#: run-scoped stage: its duplicate-cluster ids are corpus-wide, so it is
#: meaningful only over a drained run.
PIPELINE_ORDER: tuple[str, ...] = (
    "extract",
    "layout",
    "normalise",
    "spellfix",
    "enrich",
    "chunk",
    "graph-refresh",
    "embed",
    "money",
    "link",
    "pii",
    "quality",
)

#: Rank per stage, for sorting. Unknown names sort last (rather than raising)
#: so a caller's mapping that carries an extra key still round-trips — this is
#: a presentation order, and dropping or exploding on an unrecognised stage
#: would be a worse failure than putting it at the end.
_RANK: dict[str, int] = {name: i for i, name in enumerate(PIPELINE_ORDER)}


def sort_by_pipeline[V](mapping: Mapping[str, V]) -> dict[str, V]:
    """*mapping*, re-keyed in `PIPELINE_ORDER`. Values are untouched."""
    return {k: mapping[k] for k in sorted(mapping, key=stage_rank)}


def stage_rank(stage: str) -> int:
    """Position of *stage* in the pipeline; unknown stages sort last."""
    return _RANK.get(stage, len(PIPELINE_ORDER))


def in_pipeline_order(stages: Iterable[str]) -> tuple[str, ...]:
    """*stages*, ordered. For a bare collection with no mapping to carry."""
    return tuple(sorted(stages, key=stage_rank))


#: The stages a dispatcher may enqueue on the operator's behalf, in order.
#:
#: Four of `PIPELINE_ORDER`'s entries are deliberately absent:
#:
#: - `extract` is the batch queue itself — it is already dispatched, per batch,
#:   and downstream stages run over what it published.
#: - `pii` masks irreversibly and writes a terminal `*.clean_text.parquet`. It
#:   must be a deliberate act, never a consequence of a flag left on in a config
#:   copied from another run.
#: - `quality` is run-scoped and undercooked; its cluster ids are meaningful
#:   only over a fully drained run.
#:
#: A fourth, `layout`, is not listed either: extraction already ran it, so a
#: rerun is a deliberate act (a new model to measure), never automatic.
#:
#: These remain reachable through `womblex run-stage --stage …`, which is
#: unchanged: this tuple bounds *automatic* dispatch, not the stage runner.
DOWNSTREAM_STAGES: tuple[str, ...] = (
    "normalise",
    "spellfix",
    "enrich",
    "chunk",
    "graph-refresh",
    "embed",
    "money",
    "link",
)


def _normalise_wanted(config: WomblexConfig) -> bool:
    """Whether anything downstream reads the normalised overlay.

    `NormaliseConfig` has no `enabled` flag — its toggles are transforms, all
    on by default — so the real gate is whether a consumer selects the layer it
    writes. `processing.text_source` is that selector for chunk and enrich;
    `money.text_source` overrides it for the money stage alone. Dispatching
    normalise with no reader writes an overlay nothing opens; skipping it with
    one silently falls back to verbatim text, which is the failure that reads
    as a broken config setting.
    """
    overlays = ("normalised", "spellfix")
    return config.processing.text_source in overlays or config.money.text_source in overlays


def enabled_downstream_stages(config: WomblexConfig) -> tuple[str, ...]:
    """The `DOWNSTREAM_STAGES` *config* actually asks for, in pipeline order.

    Each stage answers to its own config section, so a run that enriches but
    does not embed dispatches enrich and not embed. `graph-refresh` is the one
    stage with no section of its own: it re-links mentions onto chunks that AI
    chunking rewrote, so it is meaningful only when chunk consumed the
    enrichment (`chunking.chunking_model`) — after plain semchunk the offsets
    enrich recorded still hold and the refresh is a no-op.
    """
    ai_chunking = bool(config.chunking.enabled and config.chunking.chunking_model)
    wanted = {
        "normalise": _normalise_wanted(config),
        "spellfix": config.spellfix.enabled,
        "enrich": config.enrichment.enabled,
        "chunk": config.chunking.enabled,
        "graph-refresh": ai_chunking and config.enrichment.enabled,
        "embed": config.embedding.enabled,
        "money": config.money.enabled,
        "link": config.linking.enabled,
    }
    return tuple(s for s in DOWNSTREAM_STAGES if wanted[s])


__all__ = [
    "DOWNSTREAM_STAGES",
    "PIPELINE_ORDER",
    "enabled_downstream_stages",
    "in_pipeline_order",
    "sort_by_pipeline",
    "stage_rank",
]
