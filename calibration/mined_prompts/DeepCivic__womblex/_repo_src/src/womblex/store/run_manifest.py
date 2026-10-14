"""Run-level document manifest consolidation.

The per-batch ``batch-NNNN._manifest.parquet`` sidecars are the only
mapping from ``source_hash`` (the join key on every other sidecar) back
to the source document (``doc_id`` / ``filename``). Consumers of a
shipped run need that mapping in one place, so the end of ``womblex run``
— and the standalone ``womblex manifest`` command — consolidate them
into a single ``manifest.parquet`` at the run root, one row per document
(``MANIFEST_SCHEMA``).

**It also carries the run record** — what produced the run, in the file's own
footer key-value metadata. Footer rather than columns because the record is run
grain and the manifest is document grain, which is the rule ``decisions.md``
already states for run- and corpus-shaped provenance; this file rather than a
new one because consolidation is already the moment both finalisation paths
pass through and is already regenerable on demand, which is exactly the
lifecycle the record needs.

Everything in it is **observed, not declared**. The stages are read from the
run stamps the files themselves carry, so a stage that was configured and never
ran is absent while one that ran and produced nothing is present with no rows —
a distinction a list taken from configuration flags cannot make. The local
models are the union of what each file recorded loading, by artefact digest;
the slot models are the union of what each file recorded building, by
swappable slot, name, distribution and version (O3) — the two are distinct
because an API-backed engine (Mistral, Ollama) has no artefact to digest but
still used a slot. The documents, extraction methods and statuses are counted
off the consolidated rows. Nothing is taken from a configuration file, because
``womblex manifest`` may be re-run long after the run and against a
configuration that has since moved on.

What cannot be established is named rather than omitted or invented: a run
finalised before the stamps existed produces a record whose ``partial`` list
says so, instead of a record that quietly claims a run had no stages.

Attribution, not reproduction. Outputs are not bit-reproducible — OCR and model
variability mean two runs over one corpus differ — so the record inventories
what a run did and never promises the run can be recomputed from it.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.pipeline_order import stage_rank
from womblex.store.build_info import image_info
from womblex.store.contract import contract_footer
from womblex.store.output import read_manifest
from womblex.store.run_stamp import (
    read_footer_model_check,
    read_footer_models,
    read_footer_slot_models,
    read_footer_stamp,
    stamp_from_footers,
)
from womblex.store.source_provenance import NAMESPACE, IngestProvenance
from womblex.utils.isaacus_client import endpoints_from_env, sagemaker_configured

logger = logging.getLogger(__name__)

RUN_MANIFEST_FILENAME = "manifest.parquet"

#: The run record's footer key, and the version of its shape. The version is
#: carried so a later reader can tell a record it understands from one it does
#: not, rather than inferring the shape from which fields happen to be present.
#: Version 2 adds ``image`` — which container image, if any, the run executed
#: inside. Bumped rather than added silently, since telling a record that has
#: no image block from one whose run was not containerised is exactly the
#: distinction a reader inferring from presence would get wrong. Version 3 adds
#: ``model_check``, on the same reasoning: a record with no check block is one
#: that predates it, and must not read as a run whose models were fine.
#: Version 4 adds ``slot_models`` — which swappable-slot model each stage
#: actually built, by distribution and version (O3) — for the same reason.
#: Version 5 adds ``files`` — the SHA-256 of each published output file, taken
#: at write time — and a record without it names the gap in ``partial``.
#: Version 6 adds ``workflows`` — each ``womblex.workflow_digest`` the files
#: carry with the stages that wrote under it — so an empty list means no file
#: was written by ``run-workflow``, not a record that predates the key.
RUN_RECORD_KEY = f"{NAMESPACE}.run_record"
RUN_RECORD_VERSION = 6

#: What one observed parquet contributes: its footer key-value metadata and its
#: row count. A caller that reads files where they live supplies these rather
#: than staging the run in (`womblex finalize`).
FooterList = list[tuple[dict[bytes, bytes] | None, int]]

#: The sidecar whose rows name the Isaacus model that was actually called.
#: Every other stage's use of the API is inferable at best, so this is the one
#: served-model fact the shard directory can supply.
_EMBEDDINGS_SUFFIX = ".embeddings.parquet"


def _footer_from_columns(table: pa.Table) -> dict[bytes, bytes] | None:
    """Re-stamp the consolidated manifest's footer from the rows it consolidates.

    The per-batch shards were stamped at write time; consolidation reads them
    as columns, so the run manifest restates the same pair rather than needing
    the run's configuration handed to it a second time. A run whose shards
    predate the columns, or that mixes roots, leaves the footer off — a footer
    that named one of several roots would be worse than none.
    """
    if table.num_rows == 0:
        return None
    roots = set(table.column("ingest_root").to_pylist())
    collections = set(table.column("collection_id").to_pylist())
    if len(roots) != 1 or len(collections) != 1:
        return None
    root, collection = roots.pop(), collections.pop()
    if not root or not collection:
        return None
    prov = IngestProvenance(ingest_root=root, collection_id=collection)
    return prov.footer_metadata(
        r for r in table.column("source_relpath").to_pylist() if r
    )


def _merged_footer(
    shard_dir: Path, table: pa.Table, footers: FooterList | None = None,
    checksums: dict[str, str] | None = None, undigested: list[str] | None = None,
) -> dict[bytes, bytes] | None:
    """Where the corpus came from, which run, and the record of what that run did.

    The run is read back from the per-batch manifests' own stamps rather than
    recomputed, for the reason the batch sidecars inherit theirs — consolidation
    is handed a shard directory, not the run's configuration, and `womblex
    manifest` may be re-run long after. Batches naming more than one run leave
    the run keys off, the rule the ingest-root block above already follows.

    The run record is written unconditionally, because a record that cannot
    name its run still names the documents, the stages and the models, and says
    in its own ``partial`` list what it could not establish. Withholding it
    would lose all of that to the one fact it is missing.
    """
    stamp = stamp_from_footers(sorted(shard_dir.glob("*._manifest.parquet")), "manifest")
    record = {
        RUN_RECORD_KEY.encode(): json.dumps(
            build_run_record(
                shard_dir, table, footers=footers, checksums=checksums, undigested=undigested,
            ),
            sort_keys=True,
        ).encode(),
    }
    merged: dict[bytes, bytes] = {}
    for part in (
        _footer_from_columns(table),
        stamp.footer_metadata() if stamp else None,
        record,
    ):
        if part:
            merged.update(part)
    return merged or None


# ---------------------------------------------------------------------------
# The run record
# ---------------------------------------------------------------------------


def _footers(shard_dir: Path) -> FooterList:
    """Every parquet in *shard_dir* as (footer metadata, row count).

    One footer read per file, which is what makes the record cheap: the row
    counts come out of the same metadata block as the stamps, so nothing reads
    a column to build the record.
    """
    out = []
    for path in sorted(shard_dir.glob("*.parquet")):
        try:
            meta = pq.read_metadata(str(path))
        except (OSError, pa.ArrowInvalid):  # unreadable, or not parquet
            logger.warning("run record: skipping unreadable %s", path.name)
            continue
        out.append((meta.metadata, meta.num_rows))
    return out


def _corpus_identity(table: pa.Table) -> dict[str, str | None]:
    """The one root and collection the rows agree on, or ``None`` where they do not.

    Same rule as the ingest-root footer block: a record naming one of several
    roots would be worse than one admitting it has more than one.
    """
    def one(column: str) -> str | None:
        values = {v for v in table.column(column).to_pylist() if v} if table.num_rows else set()
        return values.pop() if len(values) == 1 else None

    return {"ingest_root": one("ingest_root"), "collection_id": one("collection_id")}


def _observed_stages(footers: FooterList) -> list[dict]:
    """The stages the files say wrote them, in pipeline order, with their output.

    A stage appears because a file it wrote says so, never because a
    configuration asked for it. So a stage that ran and produced nothing is
    here with ``rows`` zero — the empty-but-schema-correct sidecar the writers
    already emit — and a stage that was configured and never ran is absent.
    That is the whole distinction, and it is only available from the artefacts.
    """
    files: Counter[str] = Counter()
    rows: Counter[str] = Counter()
    for meta, num_rows in footers:
        stage = read_footer_stamp(meta).get("stage")
        if not stage:
            continue
        files[stage] += 1
        rows[stage] += num_rows
    return [
        {"stage": stage, "files": files[stage], "rows": rows[stage]}
        for stage in sorted(files, key=stage_rank)
    ]


def _observed_models(footers: FooterList) -> list[dict]:
    """The local models the run's files recorded loading, with the stages that did.

    Keyed on name *and* digest, so a model swapped part-way through a run comes
    back as two entries rather than one of them silently winning. Each digest
    recomputes from the model files alone (`utils/models.py`), which is what
    makes the entry checkable by someone holding the model and not the run.
    """
    stages: dict[tuple[str, str], set[str]] = {}
    for meta, _rows in footers:
        stage = read_footer_stamp(meta).get("stage", "")
        for entry in read_footer_models(meta):
            key = (entry["name"], entry["digest"])
            stages.setdefault(key, set())
            if stage:
                stages[key].add(stage)
    return [
        {"name": name, "digest": digest, "stages": sorted(stages[(name, digest)])}
        for name, digest in sorted(stages)
    ]


def _observed_workflows(footers: FooterList) -> list[dict]:
    """The stored workflows whose digest the run's files carry, with the stages they wrote.

    Observed from the files like the stages are: a file written by `run-workflow` says so,
    a workflow that was only stored is absent. Two digests means two workflows (or one
    edited between runs) touched the directory, and both are listed.
    """
    stages: dict[str, set[str]] = {}
    for meta, _rows in footers:
        stamp = read_footer_stamp(meta)
        if digest := stamp.get("workflow_digest"):
            stages.setdefault(digest, set()).add(stamp.get("stage", ""))
    return [{"digest": d, "stages": sorted(s for s in stages[d] if s)} for d in sorted(stages)]


def _observed_slot_models(footers: FooterList) -> list[dict]:
    """The swappable-slot models the run's files recorded building, with the
    stages that built them.

    Keyed on everything a use reports (slot, name, distribution, version), so
    a model swapped for another under the same slot part-way through a run
    comes back as two entries rather than one silently winning — matching
    `_observed_model_checks`. Distinct from `_observed_models`: this is which
    *slot* built which *named* model, not which artefact file was loaded, so
    an API-backed engine with no local artefact still appears here.
    """
    stages: dict[str, set[str]] = {}
    entries: dict[str, dict] = {}
    for meta, _rows in footers:
        stage = read_footer_stamp(meta).get("stage", "")
        for use in read_footer_slot_models(meta):
            key = json.dumps(use, sort_keys=True)
            entries[key] = use
            stages.setdefault(key, set())
            if stage:
                stages[key].add(stage)
    return [
        {**entries[key], "stages": sorted(stages[key])}
        for key in sorted(entries, key=lambda k: (entries[k]["slot"], entries[k]["name"], k))
    ]


def _observed_model_checks(footers: FooterList) -> list[dict]:
    """The pre-run model checks the run's files recorded, with the stages that carried them.

    Keyed on everything a check reports, so one worker resolving PaddleOCR v5
    and another the wheel's v4 come back as two entries — mixed models are what
    the check exists to expose, and collapsing them would hide it.
    """
    stages: dict[str, set[str]] = {}
    entries: dict[str, dict] = {}
    for meta, _rows in footers:
        stage = read_footer_stamp(meta).get("stage", "")
        for check in (read_footer_model_check(meta) or {}).get("checks", []):
            key = json.dumps(check, sort_keys=True)
            entries[key] = check
            stages.setdefault(key, set())
            if stage:
                stages[key].add(stage)
    return [
        {**entries[key], "stages": sorted(stages[key])}
        for key in sorted(entries, key=lambda k: (entries[k]["slot"], entries[k]["name"], k))
    ]


def _called_models(shard_dir: Path) -> list[str]:
    """Isaacus models the run's embedding sidecars name as having been called."""
    seen: set[str] = set()
    for path in sorted(shard_dir.glob(f"*{_EMBEDDINGS_SUFFIX}")):
        try:
            table = pq.read_table(str(path), columns=["model"])
        except (OSError, KeyError, pa.ArrowInvalid):
            continue
        seen.update(m for m in table.column("model").to_pylist() if m)
    return sorted(seen)


#: An endpoint given as a full ARN carries the AWS account id in its fifth
#: field. The record must not, so the ARN is reduced to the endpoint name it
#: ends with and the region it names — better data as well as safer.
_ARN_PREFIX = "arn:aws:sagemaker:"


def _endpoint_identity(name: str, region: str | None) -> tuple[str, str | None]:
    """*name* and *region* with any account identifier removed.

    ``arn:aws:sagemaker:<region>:<account>:endpoint/<name>`` reduces to
    ``<name>`` plus ``<region>``. A string that is not an ARN passes through
    untouched; one that is, but in a shape this does not recognise, still does
    not — it yields ``unnamed-endpoint`` rather than risk emitting the account
    field as a name. The declared region wins over the ARN's, since a deployer
    who spelled one out meant it.
    """
    if not name.startswith(_ARN_PREFIX):
        return name, region
    # Parse strictly by ARN field index rather than by "the last segment":
    # field 4 is the account, and an ARN truncated at it would otherwise be
    # emitted as the endpoint name. Fields are region(3), account(4),
    # resource(5+) — so an ARN with no resource has no name to give.
    parts = name.split(":")
    resource = ":".join(parts[5:]) if len(parts) > 5 else ""
    resolved = resource.rsplit("/", 1)[-1].strip() or "unnamed-endpoint"
    arn_region = parts[3].strip() if len(parts) > 3 else ""
    return resolved, region or arn_region or None


def _services(shard_dir: Path) -> list[dict]:
    """The external services the run reached, by endpoint, region and models.

    The deployment is declared in the environment rather than recorded in the
    output, so this describes the environment the record is written in. That is
    the run's own environment at both finalisation moments; a regeneration
    elsewhere is noted in ``partial`` rather than left to look like a fact
    about the run.

    Credentials are structurally absent: an endpoint name, a region and a model
    id are the whole of what is read, and the API key is never consulted. An
    endpoint spelled as a full ARN is reduced to its name and region first, so
    the AWS account id in the ARN's fifth field does not reach the record.
    """
    called = _called_models(shard_dir)
    if sagemaker_configured():
        declared = []
        for e in endpoints_from_env():
            endpoint, region = _endpoint_identity(e.name, e.region)
            declared.append({
                "kind": "isaacus-sagemaker",
                "endpoint": endpoint,
                "region": region,
                "models": list(e.models) if e.models else called or None,
            })
        return sorted(declared, key=lambda s: s["endpoint"])
    if called:
        return [{
            "kind": "isaacus-hosted-api",
            # Not spelled out: the SDK resolves its own default endpoint, and
            # naming a URL this module does not read would be inventing one.
            "endpoint": "sdk-default",
            "region": None,
            "models": called,
        }]
    return []


def _manifests_only(shard_dir: Path) -> bool:
    """True when the directory holds manifest shards and no other parquet.

    That is never a real shard directory — extraction writes four siblings — so
    it is a staged subset, and everything a downstream stage would have said
    about itself is simply not present to be read.
    """
    manifests = set(shard_dir.glob("*._manifest.parquet"))
    return bool(manifests) and not (set(shard_dir.glob("*.parquet")) - manifests)


def _partial(
    record: dict,
    footers: FooterList,
    *,
    staged_subset: bool = False,
    whole_directory: bool = True,
) -> list[str]:
    """What the record could not establish, named rather than left to look absent."""
    gaps: list[str] = []
    if not footers:
        gaps.append("no readable parquet in the shard directory")
    if staged_subset:
        gaps.append(
            "built from manifest shards alone: any stage or model after "
            "extraction is not present to be observed, and its absence here "
            "does not mean it did not run",
        )
    if not whole_directory:
        # Stages and models come from footers a caller can supply from
        # elsewhere; served models come from a *column*, which it cannot. So
        # this gap outlives the one above rather than being covered by it.
        gaps.append(
            "served models are read from the embedding sidecars' rows, and "
            "this record was built over a directory that does not hold the "
            "whole run",
        )
    if record["run"] is None:
        gaps.append(
            "run identity: no file carries a run stamp, or the files name more "
            "than one run",
        )
    if not record["stages"]:
        gaps.append("stages: no file names the stage that wrote it")
    if not record["local_models"]:
        gaps.append(
            "local models: no file records one, so either none was loaded or "
            "the run predates the record",
        )
    if not record["slot_models"]:
        gaps.append(
            "slot models: no file records one, so no swappable-slot model "
            "was built, or the run predates the record",
        )
    if not record["model_check"]:
        off = any(
            (read_footer_model_check(meta) or {}).get("level") == "off" for meta, _ in footers
        )
        gaps.append(
            "model check: it was switched off, so the configured models were "
            "not verified before the run" if off else
            "model check: no file records one, so the run predates it or no "
            "process was asked to check",
        )
    gaps.append(
        "external services describe the environment this record was written "
        "in, not one recorded during the run",
    )
    image = record["image"]
    if image["digest"]:
        # Stated even when the digest is present, and especially then: the
        # record must not read as though it verified the image it names.
        gaps.append(
            "the image digest is supplied to the container by the deployment, "
            "not read from the running image — a digest is content-addressed "
            "after the push and cannot be baked into the image it names, so "
            "this is operator-honest rather than self-verifying",
        )
    else:
        gaps.append(f"image digest: {image['reason']}")
    if not any(m["slot"] == "ocr" for m in record["slot_models"]):
        # Established once any file's `slot_models` names an OCR entry
        # (O3) — no longer unconditional now that the OCR engine is
        # recorded there, by distribution and version, whenever extraction
        # actually ran one.
        gaps.append(
            "the OCR engine is not recorded on the element stream, so a VLM-OCR "
            "service cannot be established from the shard directory",
        )
    return gaps


def build_run_record(
    shard_dir: Path, table: pa.Table, *, footers: FooterList | None = None,
    checksums: dict[str, str] | None = None, undigested: list[str] | None = None,
) -> dict:
    """Assemble the run record for *shard_dir*, whose rows are *table*.

    *footers* lets a caller supply observations it has already made instead of
    having them read from *shard_dir*. That is what `womblex finalize` does: a
    distributed run's shards live in object storage and it stages in only the
    manifests, so a record built from the directory alone would show extraction
    and nothing else — silently, since the missing stages look identical to
    stages that never ran. The caller reads the footers where the files are and
    passes them here; no filesystem abstraction reaches this module.

    *checksums* maps each published file's run-relative key to the SHA-256
    taken when it was written. The record lists them under ``files`` so a whole
    run is auditable without opening every file; a record built without them
    has no ``files`` key and names the gap in ``partial``. *undigested* names the
    jobs that may have published files without a digest (failed, unfinished, or
    recorded before checksums); ``partial`` names them too.

    Deterministic apart from ``generated_at``: every collection is sorted, the
    stages by pipeline position and the rest by name, so re-running over an
    unchanged run reproduces the record.
    """
    supplied = footers
    footers = _footers(shard_dir) if supplied is None else list(supplied)
    stamp = stamp_from_footers(sorted(shard_dir.glob("*._manifest.parquet")), "manifest")
    methods = Counter(m for m in table.column("extraction_method").to_pylist() if m)
    statuses = Counter(s for s in table.column("status").to_pylist() if s)
    record: dict = {
        "record_version": RUN_RECORD_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": None if stamp is None else {
            "run_id": stamp.run_id,
            "version": stamp.version,
            "commit": stamp.commit,
            "config_digest": stamp.config_digest,
        },
        "inputs": {
            **_corpus_identity(table),
            "documents": table.num_rows,
            "extraction_methods": dict(sorted(methods.items())),
            "statuses": dict(sorted(statuses.items())),
        },
        "stages": _observed_stages(footers),
        "workflows": _observed_workflows(footers),
        "local_models": _observed_models(footers),
        "slot_models": _observed_slot_models(footers),
        "model_check": _observed_model_checks(footers),
        "services": _services(shard_dir),
        "image": image_info().as_record(),
    }
    # Two different limits. A caller that supplied footers read them where the
    # files live, so the stages and models are complete and the staged-subset
    # gap does not apply — but `_called_models` reads a column out of *this*
    # directory either way, so a partial directory still limits the services.
    if checksums is not None:
        record["files"] = [
            {"key": key, "sha256": digest} for key, digest in sorted(checksums.items())
        ]
    partial_dir = _manifests_only(shard_dir)
    record["partial"] = _partial(
        record, footers,
        staged_subset=supplied is None and partial_dir,
        whole_directory=not partial_dir,
    )
    if checksums is None:
        record["partial"].append(
            "file checksums: none were supplied, so the output files carry no "
            "digest taken at write time",
        )
    if undigested:
        record["partial"].append(
            f"file checksums: {len(undigested)} job(s) failed, had not finished or "
            f"predate checksums, so files they published may carry no digest: "
            f"{', '.join(sorted(undigested))}",
        )
    return record


def read_run_record(path: Path) -> dict | None:
    """The run record in the manifest at *path*, or ``None`` if it carries none."""
    try:
        meta = pq.read_metadata(str(path)).metadata
    except (OSError, pa.ArrowInvalid):
        return None
    raw = (meta or {}).get(RUN_RECORD_KEY.encode())
    if raw is None:
        return None
    try:
        record = json.loads(raw.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    # A record that is not an object is not one this reader wrote; treat it as
    # absent rather than handing a caller something it cannot index.
    return record if isinstance(record, dict) else None


def run_manifest_path_for(shard_dir: Path) -> Path:
    """Default consolidated-manifest path: ``<run_root>/manifest.parquet``."""
    return shard_dir.parent / RUN_MANIFEST_FILENAME


def write_run_manifest(
    shard_dir: Path,
    output_path: Path | None = None,
    *,
    footers: FooterList | None = None,
    checksums: dict[str, str] | None = None,
    undigested: list[str] | None = None,
) -> Path:
    """Consolidate all ``*._manifest.parquet`` in ``shard_dir`` into one parquet.

    Writes to ``output_path`` (default ``<run_root>/manifest.parquet``) and
    returns the path written. An empty shard directory still produces an
    empty-but-schema-correct file so downstream reads are safe.

    *footers*, *checksums* and *undigested* are passed through to :func:`build_run_record` for
    a caller whose shards are not all in *shard_dir* — see that function.
    """
    table = read_manifest(shard_dir)
    footer = _merged_footer(shard_dir, table, footers, checksums, undigested)
    target = output_path or run_manifest_path_for(shard_dir)
    table = table.replace_schema_metadata({**(footer or {}), **contract_footer(target, role="manifest")})
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(target), compression="zstd", compression_level=3)
    logger.info("Wrote run manifest %s: docs=%d", target, table.num_rows)
    return target


__all__ = [
    "RUN_MANIFEST_FILENAME",
    "RUN_RECORD_KEY",
    "RUN_RECORD_VERSION",
    "FooterList",
    "build_run_record",
    "read_run_record",
    "run_manifest_path_for",
    "write_run_manifest",
]
