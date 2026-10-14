"""`womblex plan`: what a workflow would run over a shard directory, and what it would spend.

Offline: it stops at `check_workflow` (no model request, no client); `validate` probes
models. Enrich and embed counts come from the stages' own request builders over the offline
kanon-2 tokeniser. A free earlier step (`normalise`, `spellfix`, `chunk` without AI chunking)
is produced in a scratch directory of symlinks so the count is over the layer the step would
read; the shard directory is untouched. A price is the caller's assumed rate, labelled.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from womblex.cloud.stage_contracts import STAGE_CONTRACTS, MutationMode, RunContext
from womblex.cloud.workflow_order import present_from_names
from womblex.cloud.workflow_runner import check_workflow, not_acted_on
from womblex.cloud.workflow_store import StoredWorkflow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import WomblexConfig
    from womblex.ui.presets import Preset

#: Free, local, deterministic steps whose output a count may need, per counted stage.
_FEEDS: dict[str, tuple[str, ...]] = {
    "enrich": ("normalise", "spellfix"),
    "embed": ("normalise", "spellfix", "chunk"),
}


@dataclass
class TokenEstimate:
    """Counts for a paid step; `basis` says what they were taken over."""

    basis: str  # on_disk | simulated | pending | unavailable
    note: str = ""
    tokens: int = 0
    requests: int = 0
    texts: int = 0
    detail: dict[str, int] = field(default_factory=dict)
    assumed_price_per_million: float | None = None
    assumed_cost: float | None = None


@dataclass
class StepPlan:
    stage: str
    action: str  # run | skip
    reason: str = ""
    bases: int = 0
    rewrites: int = 0
    tokens: TokenEstimate | None = None


@dataclass
class WorkflowPlan:
    workflow_id: str
    digest: str
    steps: list[StepPlan]
    warnings: list[str]
    not_acted_on: tuple[str, ...]
    shard_dir: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _layout_skips(shard_dir: Path, config: WomblexConfig) -> tuple[int, int]:
    """`(bases whose layout sidecar already carries this config's fingerprint, bases)`."""
    import pyarrow.parquet as pq

    from womblex.process.chunk_stage import _batch_bases
    from womblex.store.layout_output import layout_fingerprint, layout_regions_path_for, read_footer_layout_fingerprint

    want, bases = layout_fingerprint(config), _batch_bases(shard_dir)
    sidecars = [p for p in map(layout_regions_path_for, bases) if p.exists()]
    same = sum(read_footer_layout_fingerprint(pq.read_metadata(str(p)).metadata) == want for p in sidecars)
    return same, len(bases)


def _estimate(stage: str, index: int, stages: list[str], config: WomblexConfig, shard_dir: Path) -> TokenEstimate:
    contract = STAGE_CONTRACTS[stage]
    # Optional inputs (enrich reading chunks for graph edges) never change a count.
    wanted = set(contract.required_inputs) | {c.suffix for c in contract.conditional_inputs(config) if c.strict}
    for earlier in stages[:index]:
        c = STAGE_CONTRACTS[earlier]
        simulable = earlier in _FEEDS[stage] and not c.requires_isaacus_api(config)
        if not simulable and c.mutation is MutationMode.SIDECAR and wanted & set(c.outputs(config)):
            return TokenEstimate("pending", f"{earlier!r} produces an input of {stage!r} and cannot be run "
                                 "offline; counted once it has run")
    sim = [s for s in stages[:index] if s in _FEEDS[stage]]
    held_back = {sfx for s in sim for sfx in STAGE_CONTRACTS[s].outputs(config)}
    try:
        with tempfile.TemporaryDirectory(prefix="womblex-plan-") as tmp:
            scratch = Path(tmp)
            for f in shard_dir.glob("*.parquet"):
                if not any(f.name.endswith(sfx) for sfx in held_back):
                    (scratch / f.name).symlink_to(f.resolve())
            for s in sim:
                STAGE_CONTRACTS[s].run(scratch, config, RunContext())
            basis = "simulated" if sim else "on_disk"
            if stage == "enrich":
                from womblex.analyse.enrich_stage import plan_enrichment

                p = plan_enrichment(scratch, config.enrichment, text_source=config.processing.text_source)
                return TokenEstimate(
                    basis, tokens=p.tokens, requests=p.requests, texts=p.documents + p.tables,
                    detail={"documents": p.documents, "tables": p.tables, "narrative_tokens": p.narrative_tokens,
                            "table_tokens": p.table_tokens, "split_documents": p.split_documents},
                )
            from womblex.analyse.embed_stage import plan_embedding

            e = plan_embedding(scratch)
            return TokenEstimate(basis, tokens=e.tokens, requests=e.requests, texts=e.chunks)
    except Exception as exc:  # a layer that is not there: the plan says so, never raises
        return TokenEstimate("unavailable", f"{type(exc).__name__}: {exc}")


def plan_workflow(
    stored: StoredWorkflow,
    shard_dir: Path,
    *,
    resolve_preset: Callable[[str], Preset | None],
    prices: dict[str, float] | None = None,
) -> WorkflowPlan:
    """Plan *stored* over *shard_dir*. *prices* maps `enrich` / `embed` to an assumed USD per million tokens.

    Raises `WorkflowRefusedError` for what the runner would refuse before any step
    (spec, preset, config, order); the model and Isaacus checks are `validate`'s.
    """
    from womblex.cloud.workflow_order import optional_order_warnings

    present = present_from_names(p.name for p in shard_dir.glob("*.parquet"))
    config, stages = check_workflow(
        stored, resolve_preset=resolve_preset, present=present, shard_dir=shard_dir,
    )
    steps: list[StepPlan] = []
    for i, stage in enumerate(stages):
        contract = STAGE_CONTRACTS[stage]
        plan = StepPlan(stage, "run", bases=len(present))
        outs = set(contract.outputs(config)) if contract.mutation is MutationMode.SIDECAR else set()
        if outs:
            plan.rewrites = sum(1 for held in present.values() if outs <= held)
            if plan.rewrites:
                plan.reason = f"rewrites {plan.rewrites} of {plan.bases} existing output set(s)"
        if stage == "layout":
            same, total = _layout_skips(shard_dir, config)
            if total and same == total:
                plan.action, plan.reason = "skip", "every layout sidecar already carries this config's fingerprint"
            elif same:
                plan.reason = f"{same} of {total} base(s) skipped (fingerprint unchanged)"
        if stage in _FEEDS:
            plan.tokens = _estimate(stage, i, stages, config, shard_dir)
            price = (prices or {}).get(stage)
            if price is not None and plan.tokens.basis in ("on_disk", "simulated"):  # no count, no cost
                plan.tokens.assumed_price_per_million = price
                plan.tokens.assumed_cost = plan.tokens.tokens / 1e6 * price
        steps.append(plan)
    return WorkflowPlan(
        stored.id, stored.digest, steps, optional_order_warnings(stages, config),
        not_acted_on(stored), str(shard_dir),
    )


__all__ = ["StepPlan", "TokenEstimate", "WorkflowPlan", "plan_workflow"]
