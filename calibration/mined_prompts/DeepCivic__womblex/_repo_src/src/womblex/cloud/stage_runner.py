"""Remote per-batch shard-stage runner — the generalisation of ``finalize``.

``womblex finalize`` is already store-aware in exactly the shape a downstream
stage needs: list one sidecar class under the run's shard prefix, download it to
a temp dir, call the unchanged library function against that ``Path``, upload
the result back. This module extends that shape from a single consolidation step
to the per-batch stages declared in :mod:`womblex.cloud.stage_contracts`.

Invariants:

- **Discovery is separate from required inputs.** Bases come from
  extraction-role siblings only; a downstream sidecar can never introduce one.
  Discovery-only files (``*.form_fields.parquet``) are never downloaded.
- **One unit of work at a time.** ``PER_BATCH`` stages stage exactly one base
  into a temp dir. Only ``quality`` is ``WHOLE_RUN``, because its dedup cluster
  ids are corpus-wide and per-batch execution would emit colliding ids.
- **All declared outputs publish, or none do.** A partial sidecar set can never
  read as complete on a later run.
- **Idempotent.** Completed bases skip on their published outputs, so re-running
  as more batches land processes only the new ones — the property ``finalize``
  already has, extended to every stage.

Stage *ordering* is the caller's: this runs one stage, it does not sequence them.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from womblex.cloud.stage_contracts import (
    DISCOVERY_SUFFIXES,
    PRODUCER_OF,
    MutationMode,
    RunContext,
    StageContract,
    StageScope,
)
from womblex.process.text_overlay import MissingOverlayError
from womblex.utils.checksum import sha256_file

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import WomblexConfig
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)


@dataclass
class StageRunSummary:
    """Outcome of one runner invocation."""

    stage: str
    processed: int = 0
    skipped: int = 0
    not_ready: int = 0
    failed: int = 0
    published: int = 0
    bases: int = 0
    not_ready_missing: set[str] = field(default_factory=set)
    discovery_failed: bool = False
    error: str | None = None

    @property
    def exit_code(self) -> int:
        if self.failed:
            return 1
        # Nothing to discover means a bad prefix, a typo'd --run-id or an empty
        # run. Exiting 0 there would read as success in `run-stage … && next`.
        if self.discovery_failed:
            return 1
        # Every base blocked on an absent upstream sidecar is a stage-ordering
        # error, not a still-draining fleet.
        if self.bases and self.not_ready == self.bases:
            return 1
        return 0

    def log(self) -> None:
        logger.info(
            "%s: %d base(s) — %d processed, %d skipped, %d not-ready, %d failed "
            "(%d file(s) published)",
            self.stage, self.bases, self.processed, self.skipped,
            self.not_ready, self.failed, self.published,
        )


class NotReady(Exception):
    """A required input is absent for this base — upstream has not produced it."""

    def __init__(self, suffix: str, stem: str):
        self.suffix = suffix
        self.stem = stem
        super().__init__(f"{stem}{suffix}")


class InputContractError(Exception):
    """Config selects an input that is absent, and running anyway would be wrong.

    Distinct from :class:`NotReady`: that one is a fine, expected state on a
    draining fleet. This one means the stage would build its output from the
    wrong text layer — a failure, but an actionable one rather than a crash,
    logged without a traceback.
    """


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def remote_bases(keys: list[str]) -> list[str]:
    """Batch base stems in *keys*, discovered from extraction-role siblings only.

    Mirrors ``chunk_stage._batch_bases`` over store-relative keys: the same four
    roles, the same ``.corrupt`` exclusion, the same name sort. A store holding
    only ``*.chunks.parquet`` yields nothing — downstream sidecars do not define
    batches.
    """
    seen: set[str] = set()
    for key in keys:
        name = key.rsplit("/", 1)[-1]
        for suffix in DISCOVERY_SUFFIXES:
            if not name.endswith(suffix):
                continue
            stem = name[: -len(suffix)]
            # A bare `.elements.parquet` has no stem to key a base on.
            if not stem or stem.endswith(".corrupt"):
                break
            seen.add(stem)
            break
    return sorted(seen)


def _key(shard_prefix: str, stem: str, suffix: str) -> str:
    return f"{shard_prefix}/{stem}{suffix}"


def _skippable(
    contract: StageContract, config: WomblexConfig, shard_prefix: str,
    unit: list[str], present: set[str], force: bool,
) -> bool:
    # Skip is only sound for sidecar producers: graph-refresh's outputs are a
    # subset of its inputs, so output-exists can never fire.
    return (
        not force
        and contract.mutation is MutationMode.SIDECAR
        and _outputs_present(contract, config, shard_prefix, unit, present)
    )


def _strict_input_gaps(
    contract: StageContract, config: WomblexConfig, shard_prefix: str,
    units: list[list[str]], present: set[str], force: bool,
) -> list[tuple[str, InputContractError]]:
    """Every unit that would be refused for a missing strict input, checked up front.

    A unit still waiting on a required input is left to the loop's ``NotReady``
    handling — an upstream that is still draining is not a contract error.
    """
    gaps: list[tuple[str, InputContractError]] = []
    for unit in units:
        if _skippable(contract, config, shard_prefix, unit, present, force):
            continue
        try:
            _resolve_inputs(contract, config, shard_prefix, unit, present)
        except NotReady:
            continue
        except InputContractError as e:
            gaps.append((unit[0] if len(unit) == 1 else f"{len(unit)} base(s)", e))
    return gaps


# ---------------------------------------------------------------------------
# Per-unit execution
# ---------------------------------------------------------------------------


def _resolve_inputs(
    contract: StageContract, config: WomblexConfig, shard_prefix: str,
    stems: list[str], present: set[str],
) -> list[str]:
    """Input keys to download for *stems*. Raises ``NotReady`` / ``InputContractError``.

    ``NotReady`` means an upstream stage has not written a required sidecar yet.
    ``InputContractError`` means config selected a *strict* conditional input that
    is absent — running anyway would build a sidecar from the wrong text layer.
    """
    keys: list[str] = []
    for suffix in contract.required_inputs:
        for stem in stems:
            key = _key(shard_prefix, stem, suffix)
            if key not in present:
                raise NotReady(suffix, stem)
            keys.append(key)

    for cond in contract.conditional_inputs(config):
        for stem in stems:
            key = _key(shard_prefix, stem, cond.suffix)
            if key in present:
                keys.append(key)
            elif cond.strict:
                raise InputContractError(
                    f"{cond.reason} selects {cond.suffix} but {key} is absent. "
                    f"Run `womblex run-stage --stage "
                    f"{PRODUCER_OF.get(cond.suffix, '<upstream>')}` first — "
                    "the stage never falls back to verbatim element text."
                )
            else:
                logger.debug("%s: %s absent for %s (%s) — stage falls back",
                             contract.name, cond.suffix, stem, cond.reason)
    return keys


def _outputs_present(
    contract: StageContract, config: WomblexConfig, shard_prefix: str,
    stems: list[str], present: set[str],
) -> bool:
    """True when every declared output already exists for every stem."""
    return all(
        _key(shard_prefix, stem, suffix) in present
        for stem in stems
        for suffix in contract.outputs(config)
    )


#: Store-relative subdir under a shard prefix where an IN_PLACE commit stages
#: its outputs before moving them onto the live keys. Kept out of the batch
#: namespace (a leading dot, its own segment) so discovery — which globs
#: `<shard_prefix>/*.parquet` at the immediate level — never sees a staged temp
#: object as a shard.
_STAGING_DIRNAME = ".staging"


def _publish(
    contract: StageContract, config: WomblexConfig, store: RemoteStore,
    shard_prefix: str, stems: list[str], local_dir: Path,
) -> dict[str, str]:
    """Verify every declared output exists locally, then publish them all.

    Returns each published key with the SHA-256 of the bytes uploaded.

    SIDECAR stages publish straight to their live keys: their outputs are new
    siblings disjoint from their inputs, so a transport failure part-way through
    leaves only an incomplete *new* set, which the skip rule refuses to treat as
    complete — the next run redoes the base and overwrites.

    An IN_PLACE stage (only ``graph-refresh``) rewrites keys that already hold
    good bytes and is never skipped, so a straight overwrite is destructive: a
    crash between the two uploads could leave one sidecar rewritten against the
    other's stale bytes, or a half-written object, with no clean copy to fall
    back to. It commits atomically instead — upload every output to a temp key
    under ``.staging`` first, and only once *all* have landed, move each temp
    key onto its live key (:meth:`RemoteStore.move`, a server-side rename on S3,
    a rename locally). The live keys are touched only in the fast move phase,
    after the slow uploads have all succeeded, so the destructive window is
    closed; a crash before the moves leaves the live keys untouched and orphans
    only temp objects.
    """
    pending: list[tuple[Path, str]] = []
    for stem in stems:
        for suffix in contract.outputs(config):
            local = local_dir / f"{stem}{suffix}"
            if not local.exists():
                raise FileNotFoundError(
                    f"{contract.name} declared {suffix} for {stem} but did not write it; "
                    "publishing nothing for this unit."
                )
            pending.append((local, _key(shard_prefix, stem, suffix)))

    checksums = {key: sha256_file(local) for local, key in pending}
    if contract.mutation is MutationMode.IN_PLACE:
        _publish_atomic(store, shard_prefix, pending)
    else:
        for local, key in pending:
            store.upload_file(local, key)
    return checksums


def _publish_atomic(
    store: RemoteStore, shard_prefix: str, pending: list[tuple[Path, str]],
) -> None:
    """Upload every output to a temp key, then move each onto its live key.

    The commit step of an in-place rewrite. Uploads (the slow, failure-prone
    part) happen entirely on temp keys; the live keys change only in the move
    phase, one backend rename each, after every upload has succeeded. On any
    upload failure the live keys are still their old, good bytes; the staged
    temp objects are cleaned up best-effort and the raised error re-runs the
    base later.
    """
    staged: list[tuple[str, str]] = []  # (temp_key, live_key)
    staging = f"{shard_prefix}/{_STAGING_DIRNAME}"
    try:
        for local, live_key in pending:
            name = live_key.rsplit("/", 1)[-1]
            temp_key = f"{staging}/{name}"
            store.upload_file(local, temp_key)
            staged.append((temp_key, live_key))
    except Exception:
        _cleanup_staged(store, [tk for tk, _ in staged])
        raise

    # All uploads landed — commit. A failure here can leave a partial set on the
    # live keys, but graph-refresh is never skipped and recomputes both sidecars
    # from the same offsets, so the next run overwrites cleanly.
    for temp_key, live_key in staged:
        store.move(temp_key, live_key)


def _cleanup_staged(store: RemoteStore, temp_keys: list[str]) -> None:
    for temp_key in temp_keys:
        try:
            store.delete(temp_key)
        except Exception:  # a leaked temp object is harmless — never mask the real error
            logger.debug("Could not remove staged temp object %s", temp_key)


def _run_unit(
    contract: StageContract, config: WomblexConfig, ctx: RunContext,
    store: RemoteStore, shard_prefix: str, stems: list[str], input_keys: list[str],
    ingest: RemoteStore,
) -> dict[str, str]:
    """Stage in, run the unchanged ``*_shards()``, publish. Returns key -> SHA-256 published."""
    with tempfile.TemporaryDirectory(prefix="womblex-stage-") as tmp:
        documents = Path(tmp) / "documents"
        store.download_to_dir(input_keys, documents)
        if contract.needs_sources:
            ctx = replace(ctx, source_fetcher=_source_fetcher(ingest, Path(tmp) / "sources"))
        contract.run(documents, config, ctx)
        return _publish(contract, config, store, shard_prefix, stems, documents)


class StageDiscoveryError(Exception):
    """No batch bases under the shard prefix: a bad prefix, run id or an empty run."""


def plan_units(
    contract: StageContract, store: RemoteStore, shard_prefix: str, config: WomblexConfig,
) -> list[list[str]]:
    """The units *contract* runs over: one per base, or one run-wide unit.

    Refuses before any unit is attempted when the config selects a strict input
    some base lacks (:class:`InputContractError`), or when there is nothing to
    run over (:class:`StageDiscoveryError`). Nothing is skipped here — a
    workflow records each finished unit itself, so a re-run does not repeat it.
    """
    present = set(store.list_files(shard_prefix, "*.parquet"))
    stems = remote_bases(sorted(present))
    if not stems:
        raise StageDiscoveryError(
            f"No batch bases under {shard_prefix} — expected extraction siblings "
            f"({', '.join(DISCOVERY_SUFFIXES)}). Downstream sidecars alone cannot drive discovery."
        )
    units = [stems] if contract.scope is StageScope.WHOLE_RUN else [[s] for s in stems]
    gaps = _strict_input_gaps(contract, config, shard_prefix, units, present, True)
    if gaps:
        raise InputContractError(
            f"{contract.name}: refusing before any base is processed — "
            + "; ".join(f"{label}: {err}" for label, err in gaps)
        )
    return units


def run_unit(
    contract: StageContract, config: WomblexConfig, ctx: RunContext, store: RemoteStore,
    shard_prefix: str, unit: list[str], ingest: RemoteStore,
) -> dict[str, str]:
    """Run one unit and return the keys it published, each with its SHA-256.

    Raises :class:`NotReady` when an upstream sidecar is absent and
    :class:`InputContractError` when a strict input the config selects is.
    """
    present = set(store.list_files(shard_prefix, "*.parquet"))
    input_keys = _resolve_inputs(contract, config, shard_prefix, unit, present)
    return _run_unit(contract, config, ctx, store, shard_prefix, unit, input_keys, ingest)


def _source_fetcher(ingest: RemoteStore, scratch: Path) -> Callable[[dict], Path]:
    """A per-document fetcher for a source-document stage.

    Maps a manifest row to its ingest key from the recorded provenance
    (``source_relpath``), stages it flat under *scratch* by hash, and checks the
    bytes against ``source_hash``. A missing or moved document, or one whose
    bytes differ, is an error for that document alone; the stage decides what a
    failed document means for its batch.
    """
    from womblex.store.output import _source_hash

    def fetch(row: dict) -> Path:
        key = (row.get("source_relpath") or "").strip("/")
        if not key:
            raise FileNotFoundError("the manifest row records no source_relpath")
        local = scratch / f"{row['source_hash']}{row.get('ext') or ''}"
        try:
            ingest.download_file(key, local)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{key} is not in the ingest store (missing or moved)") from e
        if _source_hash(str(local)) != row["source_hash"]:
            raise ValueError(f"{key} does not match the hash the manifest records")
        return local

    return fetch


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def checkpoint_prefix_for(contract: StageContract, output_prefix: str) -> str | None:
    """Store-relative dir holding this stage's checkpoint, or ``None`` if it keeps none."""
    if contract.checkpoint_dirname is None:
        return None
    return f"{output_prefix.strip('/')}/{contract.checkpoint_dirname}"


def _stage_checkpoint_in(
    contract: StageContract, store: RemoteStore, prefix: str, dataset: str, local_dir: Path,
) -> CheckpointManager:
    """Pull the stage's checkpoint dir down and open a manager over it.

    Checkpoint state is a *directory*, not a shard suffix, so it cannot ride
    along with the per-base staging. Opt-in only, and single-invocation per run:
    concurrent runners would clobber each other's JSON — which is precisely why
    the job queue, not a checkpoint file, is the distributed checkpoint.
    """
    from womblex.store.checkpoint import CheckpointManager

    local_dir.mkdir(parents=True, exist_ok=True)
    existing = store.list_files(prefix, "*.json")
    if existing:
        store.download_to_dir(existing, local_dir)
    mgr = CheckpointManager(local_dir, f"{dataset}_{contract.name.replace('-', '_')}")
    mgr.load()
    return mgr


def _stage_checkpoint_out(store: RemoteStore, prefix: str, mgr: CheckpointManager) -> None:
    if mgr.checkpoint_file.exists():
        store.upload_file(mgr.checkpoint_file, f"{prefix}/{mgr.checkpoint_file.name}")


def run_stage_remote(
    contract: StageContract,
    store: RemoteStore,
    shard_prefix: str,
    config: WomblexConfig,
    *,
    ctx: RunContext | None = None,
    force: bool = False,
    checkpoint_prefix: str | None = None,
    checkpoint_dataset: str = "runner",
    ingest: RemoteStore | None = None,
) -> StageRunSummary:
    """Execute *contract* against a store's shard prefix, one unit at a time.

    ``ingest`` is where a source-document stage reads the documents from (the
    store itself when ``None``, as the worker does); other stages ignore it.
    """
    ctx = ctx or RunContext()
    summary = StageRunSummary(stage=contract.name)

    present = set(store.list_files(shard_prefix, "*.parquet"))
    stems = remote_bases(sorted(present))
    if not stems:
        logger.error(
            "No batch bases under %s — expected extraction siblings (%s). "
            "Downstream sidecars alone cannot drive discovery.",
            shard_prefix, ", ".join(DISCOVERY_SUFFIXES),
        )
        summary.discovery_failed = True
        return summary

    units = [stems] if contract.scope is StageScope.WHOLE_RUN else [[s] for s in stems]
    summary.bases = len(units)
    if contract.scope is StageScope.WHOLE_RUN:
        logger.info(
            "%s is run-scoped: staging %d base(s) in one pass (its clusters are "
            "corpus-wide; per-batch execution would emit colliding ids).",
            contract.name, len(stems),
        )

    with contextlib.ExitStack() as stack:
        if checkpoint_prefix and contract.checkpoint_dirname:
            ckpt_dir = Path(stack.enter_context(
                tempfile.TemporaryDirectory(prefix="womblex-stage-ckpt-")))
            ctx.checkpoint_mgr = _stage_checkpoint_in(
                contract, store, checkpoint_prefix, checkpoint_dataset, ckpt_dir)
            stack.callback(_stage_checkpoint_out, store, checkpoint_prefix, ctx.checkpoint_mgr)

        refused = _strict_input_gaps(contract, config, shard_prefix, units, present, force)
        if refused:
            for label, err in refused:
                logger.error("%s: %s refused: %s", contract.name, label, err)
            logger.error(
                "%s: refusing the stage before any base is processed — %d base(s) "
                "lack a strict input the config selects.", contract.name, len(refused),
            )
            summary.failed = len(refused)
            return summary

        for unit in units:
            label = unit[0] if len(unit) == 1 else f"{len(unit)} base(s)"
            try:
                if _skippable(contract, config, shard_prefix, unit, present, force):
                    logger.info("%s: %s already complete — skipping", contract.name, label)
                    summary.skipped += 1
                    continue

                input_keys = _resolve_inputs(contract, config, shard_prefix, unit, present)
                summary.published += len(_run_unit(
                    contract, config, ctx, store, shard_prefix, unit, input_keys,
                    ingest or store,
                ))
                summary.processed += 1
            except NotReady as nr:
                producer = PRODUCER_OF.get(nr.suffix, "the upstream stage")
                # Name the offending base, not the unit — on a run-scoped stage
                # one un-chunked batch blocks the whole pass, and "3 base(s)
                # missing" would send you looking in the wrong place.
                logger.warning(
                    "%s: %s missing %s (blocking %s) — %s has not produced it yet",
                    contract.name, nr.stem, nr.suffix, label, producer,
                )
                summary.not_ready += 1
                summary.not_ready_missing.add(nr.suffix)
            except InputContractError as e:
                # Actionable, not a crash — no traceback.
                logger.error("%s: %s refused: %s", contract.name, label, e)
                summary.failed += 1
            except Exception:  # one bad base must not stop the run
                logger.exception("%s: %s failed", contract.name, label)
                summary.failed += 1

    if summary.not_ready:
        if summary.not_ready == summary.bases:
            missing = ", ".join(sorted(summary.not_ready_missing))
            producers = sorted(
                {PRODUCER_OF.get(s, "<upstream>") for s in summary.not_ready_missing}
            )
            logger.error(
                "%s: every base is missing %s. Run `womblex run-stage --stage %s` first.",
                contract.name, missing, "` / `".join(producers),
            )
        else:
            logger.warning(
                "%s: %d base(s) skipped — required inputs absent; the upstream "
                "stage may still be draining. Re-run when it has.",
                contract.name, summary.not_ready,
            )
    return summary


def run_stage_local(
    contract: StageContract,
    shard_dir: Path,
    config: WomblexConfig,
    *,
    ctx: RunContext | None = None,
) -> StageRunSummary:
    """Run *contract* over a local shard dir — one call, whole directory.

    The parity oracle for the remote path: the stages write per-base sidecars
    independently, so processing the directory in one call and processing it a
    base at a time produce the same bytes.
    """
    ctx = ctx or RunContext()
    summary = StageRunSummary(stage=contract.name)
    try:
        contract.run(shard_dir, config, ctx)
    except MissingOverlayError as e:
        # Actionable, not a crash — no traceback.
        logger.error("%s: refused: %s", contract.name, e)
        summary.failed = 1
        summary.error = str(e)
        return summary
    summary.processed = 1
    summary.bases = 1
    return summary


class StagePreconditionError(Exception):
    """A stage cannot start: preflight failed, or Isaacus is not resolvable.

    Distinct from a base failing mid-run — nothing has been attempted yet, so
    the message is the whole story and needs no traceback. Both dispatchers
    (the CLI and the worker) refuse on it rather than running a stage that
    would write nothing and read as success.
    """


def prepare_stage_context(contract: StageContract, config: WomblexConfig) -> RunContext:
    """Check *contract*'s preconditions and build the runtime it needs.

    Raises :class:`StagePreconditionError` when the stage must not run. Shared
    by every dispatcher so a stage run from the queue applies exactly the
    checks `womblex run-stage` does — the Isaacus one especially: without it
    `chunk_shards` warns, writes nothing and returns cleanly, which a queue
    would record as a completed job.
    """
    if contract.preflight is not None:
        try:
            contract.preflight(config)
        except Exception as e:
            raise StagePreconditionError(f"{contract.name} preflight failed: {e}") from e

    if contract.requires_isaacus_api(config):
        from womblex.utils.availability import isaacus_available

        # Cheap and network-free, so it runs before the model check's own
        # Isaacus probe — a deployment that is not configured at all should
        # fail with this message, not with the probe's raw SDK error.
        if not isaacus_available():
            raise StagePreconditionError(
                f"{contract.name} needs Isaacus (isaacus SDK + ISAACUS_API_KEY, or "
                "ISAACUS_SAGEMAKER_ENDPOINTS for a private deployment); none is "
                "resolvable. Refusing to run rather than publishing nothing."
            )

    from womblex.utils.model_check import check_models

    models = check_models(config, scopes=(contract.name,))
    if models.failures:
        raise StagePreconditionError(
            f"{contract.name} model check failed: {models.message()}"
        )

    ctx = RunContext()
    if contract.needs_client:
        from womblex.utils.isaacus_client import make_isaacus_client

        try:
            ctx.client = make_isaacus_client(models=contract.models(config))
        except ImportError as e:
            raise StagePreconditionError(f"Isaacus SDK not usable (reinstall womblex): {e}") from e
        except Exception as e:
            raise StagePreconditionError(
                "Could not construct Isaacus client (check ISAACUS_API_KEY, or "
                f"ISAACUS_SAGEMAKER_ENDPOINTS for a private deployment): {e}"
            ) from e
    return ctx


__all__ = [
    "InputContractError",
    "NotReady",
    "StageDiscoveryError",
    "StagePreconditionError",
    "StageRunSummary",
    "checkpoint_prefix_for",
    "plan_units",
    "prepare_stage_context",
    "remote_bases",
    "run_stage_local",
    "run_stage_remote",
    "run_unit",
]
