"""Up-front ordering check: does each step's input exist, or get produced earlier?

Pure over a step list, a config and what is on disk, so a local run, a store run and
a dispatch share one answer. Checked before any step spends anything.

An input is *produced* by an earlier step that writes it as a new sidecar. `IN_PLACE`
stages (`layout`, `graph-refresh`) only rewrite what they read, so they never satisfy
or invalidate an input. A copy already on disk is **stale** when a later step will
rewrite it and no earlier step does: the consumer would read last run's layer while the
run ends with a different one. Only required and strict inputs are checked: an optional
one falls back by design, and the canonical order puts its producer later (`enrich`
reads `chunk`'s output only when it already exists). An optional input whose producer runs
later against `PIPELINE_ORDER` (`spellfix` before `normalise`) is a warning, not a refusal.
Extraction's own siblings are provided by extraction; `extraction_pending` says they
may still be arriving (a dispatch), so their absence is not an error.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from womblex.cloud.stage_contracts import (
    DISCOVERY_SUFFIXES,
    PRODUCER_OF,
    STAGE_CONTRACTS,
    MutationMode,
)
from womblex.cloud.stage_runner import remote_bases
from womblex.pipeline_order import stage_rank

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import WomblexConfig
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

#: Suffixes held per batch stem (callers pass `None` where the disk state is not known).
Present = dict[str, frozenset[str]]


class OrderCode(str, Enum):
    """Stable codes for an ordering refusal; the CLI prints them and the API returns them."""

    INPUT_MISSING = "input_missing"
    """No step before this one produces the input and it is not on disk."""

    INPUT_OUT_OF_ORDER = "input_out_of_order"
    """A later step produces the input and nothing is on disk yet."""

    INPUT_STALE = "input_stale"
    """A copy is on disk but a later step will rewrite it."""


@dataclass(frozen=True)
class OrderIssue:
    """One input a step cannot rely on."""

    code: OrderCode
    stage: str
    suffix: str
    reason: str
    producer: str | None
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "code": self.code.value}


class StageOrderError(ValueError):
    """The step sequence cannot complete as ordered. Nothing has run."""

    def __init__(self, issues: Sequence[OrderIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"[{i.code.value}] {i.message}" for i in self.issues))


def present_from_names(names: Iterable[str]) -> Present:
    """Suffixes held per batch stem, from sibling file names (a directory or store listing).

    Stems come from extraction-role siblings only, as every stage discovers batches.
    """
    bare = [n.rsplit("/", 1)[-1] for n in names if n.endswith(".parquet")]
    stems = remote_bases(bare)
    return {
        stem: frozenset(n[len(stem):] for n in bare if n.startswith(f"{stem}."))
        for stem in stems
    }


def present_from_store(store: RemoteStore, shard_prefix: str) -> Present:
    return present_from_names(store.list_files(shard_prefix, "*.parquet"))


def _produced(stages: Sequence[str], config: WomblexConfig) -> list[frozenset[str]]:
    """What each step writes as a new sidecar; `IN_PLACE` steps produce nothing."""
    return [
        frozenset(c.outputs(config)) if c.mutation is MutationMode.SIDECAR else frozenset()
        for c in (STAGE_CONTRACTS[s] for s in stages)
    ]


def optional_order_warnings(stages: Sequence[str], config: WomblexConfig) -> list[str]:
    """Optional inputs a later step produces where `PIPELINE_ORDER` puts that step earlier.

    The consumer falls back (or reads last run's copy), so this warns rather than refuses.
    `enrich` before `chunk` is the canonical order and stays quiet.
    """
    produced = _produced(stages, config)
    warnings: list[str] = []
    for i, stage in enumerate(stages):
        for c in STAGE_CONTRACTS[stage].conditional_inputs(config):
            if c.strict or any(c.suffix in p for p in produced[:i]):
                continue
            later = next((stages[j] for j in range(i + 1, len(stages))
                          if c.suffix in produced[j]), None)
            if later is not None and stage_rank(later) < stage_rank(stage):
                warnings.append(f"{stage}: {c.suffix} ({c.reason}): {later!r} produces it but "
                                f"runs later, so {stage!r} falls back or reads a stale copy")
    return warnings


def check_step_order(
    stages: Sequence[str],
    config: WomblexConfig,
    *,
    present: Present | None,
    extraction_pending: bool = False,
) -> list[OrderIssue]:
    """Every input a step in *stages* (in the order given) cannot rely on.

    *present* is `None` when the disk state is unknown (only ordering among the steps
    is checked). An empty list is a clean bill.
    """
    contracts = [STAGE_CONTRACTS[s] for s in stages]
    produced = _produced(stages, config)
    held = frozenset().union(*present.values()) if present else frozenset()
    extraction = frozenset(DISCOVERY_SUFFIXES)
    issues: list[OrderIssue] = []

    for i, contract in enumerate(contracts):
        wanted = [(s, "required input") for s in contract.required_inputs] + [
            (c.suffix, c.reason) for c in contract.conditional_inputs(config) if c.strict
        ]
        for suffix, reason in wanted:
            if any(suffix in p for p in produced[:i]):
                continue
            if suffix in extraction and (extraction_pending or present is None):
                continue
            later = next((stages[j] for j in range(i + 1, len(stages)) if suffix in produced[j]), None)
            on_disk = present is not None and suffix in held
            lacking = (
                sorted(s for s, held_s in present.items() if suffix not in held_s)
                if present is not None else []
            )
            if later is not None:
                code = OrderCode.INPUT_STALE if on_disk else OrderCode.INPUT_OUT_OF_ORDER
                what = (
                    f"a copy is on disk, but {later!r} runs later and will rewrite it"
                    if on_disk else f"{later!r} produces it but runs later"
                )
                fix = f"move {later!r} before {contract.name!r}"
            elif present is None or on_disk and not lacking:
                continue
            else:
                code = OrderCode.INPUT_MISSING
                producer = PRODUCER_OF.get(suffix)
                what = (
                    f"absent for {len(lacking)} base(s) (e.g. {lacking[0]})" if on_disk
                    else "not on disk and no earlier step produces it"
                )
                fix = f"add {producer!r} before {contract.name!r}" if producer else "produce it first"
                later = producer
            issues.append(OrderIssue(
                code=code,
                stage=contract.name, suffix=suffix, reason=reason, producer=later,
                message=f"{contract.name}: {suffix} ({reason}): {what}; {fix}",
            ))
    return issues


def check_dispatch_order(
    stages: Sequence[str], config: WomblexConfig, store: RemoteStore | None, shard_prefix: str,
) -> None:
    """The object-store form: raise `StageOrderError` if any input cannot be relied on.

    Extraction may still be draining at dispatch, so its siblings are not required. A
    store that cannot be listed (an I/O, access or connection error) leaves the disk
    state unknown, warned, rather than blocking a dispatch that would not have read it;
    anything else propagates. Lists are in `PIPELINE_ORDER`, so only `input_missing` fires.
    """
    from botocore.exceptions import BotoCoreError

    present: Present | None = None
    if store is not None:
        try:
            present = present_from_store(store, shard_prefix)
        except (OSError, BotoCoreError) as e:
            logger.warning("cannot list %s (%s: %s): stored layers not checked, only the "
                           "order among the stages", shard_prefix, type(e).__name__, e)
    issues = check_step_order(stages, config, present=present, extraction_pending=True)
    if issues:
        raise StageOrderError(issues)


__all__ = [
    "OrderCode", "OrderIssue", "StageOrderError", "check_dispatch_order", "check_step_order",
    "optional_order_warnings", "present_from_names", "present_from_store",
]
