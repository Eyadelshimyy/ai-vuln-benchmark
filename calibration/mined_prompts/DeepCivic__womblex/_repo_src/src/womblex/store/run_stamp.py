"""Which run produced this file: the run id, version, configuration and stage.

A run id exists today only as a directory name, so a shard copied out of its
run loses it and a file mixed in from another run is indistinguishable from a
native one. This stamps five facts into every pipeline Parquet's footer
key-value metadata at the moment it is written — run id, Womblex version,
source commit, configuration digest and the stage that wrote it — so
attribution survives the file being moved. Three further keys ride alongside
them, each written only when it has a value: the configuration's self-declared
name (``dataset.name``) as the run's *preset*, the local model artefacts the
writing process had loaded (``utils/models.py``, by name and content digest),
and the swappable-slot models it actually built (``utils/model_registry.py``,
by slot, name, and the distribution and version that supplied it — built-in or
plugin). Neither model key is a field on the stamp because a stamp is declared
before any model loads or builds, so each is read from its own module at
footer time (:meth:`RunStamp.footer_metadata`); the preset is a field, filled
from the config at :meth:`RunStamp.declare` and carried through inheritance.

The preset key is general provenance — which configuration produced this file —
not a ground-truth concern: no stage reads it, and it exists so the ground-truth
sidecar's ``preset`` field is inherited from the artefacts like its
``preset_digest`` sibling rather than asserted by a caller.

The keys share the ``womblex.*`` namespace ``store/source_provenance.py``
established (itself the convention ``store/register_manifest.py`` reads back
from the register ingests): one convention, not a second one. Footer metadata
is additive, so a reader that ignores it reads the file unchanged.

Two properties are deliberate:

**The digest is over the validated configuration, not the file bytes.** Pydantic
defaults and the AI-chunking validator both rewrite what the YAML supplied, so
the file is not what ran. Two runs whose YAML differs only in formatting, key
order or omitted defaults therefore produce one digest.

**``paths`` is excluded from the digest, and so is ``dataset.run_id``.** Paths
say where a deployment keeps its files, not what the pipeline does: a worker
extracts from a scratch directory it was handed, and digesting that would make
a distributed run's stamp differ from the local run of the same pipeline over
the same corpus. The run id is stamped in its own right, so folding it into the
digest would only make every run's digest unique and say nothing.

**The version and the commit describe the bytes, not the run.** A version
alone does not identify a build — the same version is cut from every commit
between two releases — so ``store/build_info.py`` resolves the commit beside
it, from the work tree or a build stamp, and reports ``unavailable`` with a
reason where neither answers. Both are the *running* values wherever a stamp is
made, which is why a stage run at a later build than extraction says so instead
of repeating what extraction claimed.

**A downstream stage inherits its run rather than declaring one.** Extraction's
caller knows the run; a stage is handed a shard directory, and its own
``dataset.run_id`` is whatever configuration it was launched with — a copied
config would have it stamp a run it did not produce. So a sidecar takes the run
id and digest from the extraction shard it sits beside (:func:`stamp_for_sidecar`)
and supplies its own version and stage. That also makes a worker's stamp match
the local run's by construction rather than by both being handed the same YAML.

Attribution, not reproduction: outputs are not bit-reproducible (OCR and model
variability mean two runs over one corpus differ), so the stamp says what
produced a file, never that the file can be recomputed from it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex import __version__
from womblex.store.build_info import resolve_commit
from womblex.store.source_provenance import NAMESPACE
from womblex.utils.model_check import footer_payload as model_check_payload
from womblex.utils.model_registry import distribution_version, used_entries
from womblex.utils.models import loaded_models

RUN_ID_KEY = f"{NAMESPACE}.run_id"
VERSION_KEY = f"{NAMESPACE}.version"
COMMIT_KEY = f"{NAMESPACE}.commit"
CONFIG_DIGEST_KEY = f"{NAMESPACE}.config_digest"
STAGE_KEY = f"{NAMESPACE}.stage"
MODELS_KEY = f"{NAMESPACE}.models"
MODEL_CHECK_KEY = f"{NAMESPACE}.model_check"
SLOT_MODELS_KEY = f"{NAMESPACE}.slot_models"
PRESET_KEY = f"{NAMESPACE}.preset"
WORKFLOW_DIGEST_KEY = f"{NAMESPACE}.workflow_digest"

_workflow_digest: ContextVar[str | None] = ContextVar("womblex_workflow_digest", default=None)


@contextmanager
def workflow_context(digest: str) -> Iterator[None]:
    """Mark every sidecar written inside the block as produced by the workflow *digest*.

    Beside `config_digest`, which says how the stages were configured: this says which
    stored workflow (preset reference, steps, config overlay) ran them. It is a contextvar
    so concurrent runs in one process cannot stamp each other's files.
    """
    token = _workflow_digest.set(digest)
    try:
        yield
    finally:
        _workflow_digest.reset(token)

# Excluded from the digest: see the module docstring. `paths` is deployment
# location and `dataset.run_id` is the run's own identity, already a key;
# `processing.models_check` is how much to verify, not what is produced.
_DIGEST_EXCLUDE: dict = {
    "paths": True, "dataset": {"run_id"}, "processing": {"models_check"},
}

# The siblings a downstream sidecar prefers to inherit its run from, in order.
# Spelled out rather than imported from `store/output.py`, which imports this
# module — two constants are cheaper than breaking the cycle.
_PREFERRED_SOURCES = (".elements.parquet", "._manifest.parquet")


def config_digest(config: object) -> str:
    """``sha256:…`` over the validated configuration, minus the excluded keys.

    Takes a ``WomblexConfig`` structurally rather than by import: ``config.py``
    already reaches into this package, and the same duck-typing keeps
    ``IngestProvenance.from_config`` import-cycle free.
    """
    payload = config.model_dump(mode="json", exclude=_DIGEST_EXCLUDE)  # type: ignore[attr-defined]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _preset_name(config: object) -> str:
    """The configuration's self-declared identity — ``dataset.name`` — for the
    ``preset`` footer key.

    Read structurally like :func:`config_digest`, and empty rather than raising
    for a config that has no dataset name, so an unnamed run writes no preset
    key on the same terms a modelless process writes no models key.
    """
    dataset = getattr(config, "dataset", None)
    return str(getattr(dataset, "name", "") or "")


@dataclass(frozen=True)
class RunStamp:
    """The facts a written file carries about the run that produced it.

    One stamp is declared per run and re-pointed at each stage that writes
    (:meth:`for_stage`), so run id, version, commit and digest cannot drift
    between the files of one run. ``preset`` is the config's ``dataset.name``,
    carried the same way and empty for a config that declares none.
    """

    run_id: str
    version: str
    commit: str
    config_digest: str
    stage: str
    preset: str = ""

    @classmethod
    def declare(cls, run_id: str, config: object, *, stage: str) -> RunStamp:
        """Stamp for *run_id* under *config*, written by *stage*.

        Both strings are required: an unstamped file is honest, a file stamped
        with an empty run id or an unnamed stage is not.
        """
        if not str(run_id).strip():
            raise ValueError("run_id is empty; a stamp names the run or is not written")
        if not str(stage).strip():
            raise ValueError("stage is empty; a stamp names the writer or is not written")
        return cls(
            str(run_id).strip(),
            __version__,
            resolve_commit(),
            config_digest(config),
            str(stage).strip(),
            preset=_preset_name(config),
        )

    @classmethod
    def inherit(cls, run_id: str, digest: str, *, stage: str, preset: str = "") -> RunStamp:
        """The stamp a downstream sidecar carries: the run it extends, by *stage*.

        A stage runs over a shard directory and has no independent knowledge of
        the run — its own ``dataset.run_id`` may be anything a copied config
        holds — so the run id and configuration digest are taken from the
        extraction shard the sidecar sits beside. Those two are facts about the
        *run*; the version and the commit are the Womblex that wrote these
        bytes, so they are the running ones rather than the inherited ones. A
        stage run at a later build than the extraction therefore says so.
        """
        if not str(run_id).strip():
            raise ValueError("run_id is empty; a stamp names the run or is not written")
        if not str(stage).strip():
            raise ValueError("stage is empty; a stamp names the writer or is not written")
        return cls(
            str(run_id).strip(),
            __version__,
            resolve_commit(),
            str(digest),
            str(stage).strip(),
            preset=str(preset),
        )

    def for_stage(self, stage: str) -> RunStamp:
        """The same run, stamped for another writer."""
        if not str(stage).strip():
            raise ValueError("stage is empty; a stamp names the writer or is not written")
        return replace(self, stage=str(stage).strip())

    def footer_metadata(self) -> dict[bytes, bytes]:
        """Namespaced footer metadata for a file this stage is writing.

        The local models are read from the resolver at call time rather than
        held on the stamp, because a stamp is declared once at the start of a
        run and models load lazily during it — a field filled in at
        :meth:`declare` would be empty on every file. So the key names what
        this process had loaded when it wrote *this* file, which is the honest
        file-grain claim; the union across a run's files is the run's set, and
        that is what ``store/run_manifest.py`` records. A process that loaded
        no local model writes no key, on the same terms an unstamped file is
        written for a run that cannot be named.
        """
        meta = {
            RUN_ID_KEY.encode(): self.run_id.encode(),
            VERSION_KEY.encode(): self.version.encode(),
            COMMIT_KEY.encode(): self.commit.encode(),
            CONFIG_DIGEST_KEY.encode(): self.config_digest.encode(),
            STAGE_KEY.encode(): self.stage.encode(),
        }
        if self.preset:
            meta[PRESET_KEY.encode()] = self.preset.encode()
        if models := loaded_models():
            payload = [{"name": m.name, "digest": m.digest} for m in models]
            meta[MODELS_KEY.encode()] = json.dumps(payload).encode()
        if slots := used_entries():
            slot_payload = []
            for entry in slots:
                dist, version = distribution_version(entry)
                slot_payload.append({
                    "slot": entry.slot, "name": entry.name,
                    "distribution": dist, "version": version,
                })
            meta[SLOT_MODELS_KEY.encode()] = json.dumps(slot_payload).encode()
        if (check := model_check_payload()) is not None:
            meta[MODEL_CHECK_KEY.encode()] = json.dumps(check, sort_keys=True).encode()
        return meta


def read_footer_stamp(metadata: Mapping[bytes, bytes] | None) -> dict[str, str]:
    """Decode the run-stamp keys out of a Parquet footer.

    ``{}`` when the file carries none — written before this landed, or by a
    caller that declared no run. Partial stamps come back partial rather than
    being completed with invented values.
    """
    if not metadata:
        return {}
    decoded = {k.decode(): v.decode() for k, v in metadata.items()}
    names = (
        (RUN_ID_KEY, "run_id"),
        (VERSION_KEY, "version"),
        (COMMIT_KEY, "commit"),
        (CONFIG_DIGEST_KEY, "config_digest"),
        (STAGE_KEY, "stage"),
        (PRESET_KEY, "preset"),
        (WORKFLOW_DIGEST_KEY, "workflow_digest"),
    )
    return {name: decoded[key] for key, name in names if key in decoded}


def read_footer_models(metadata: Mapping[bytes, bytes] | None) -> list[dict[str, str]]:
    """Decode the local-model key out of a Parquet footer.

    ``[]`` when the file carries none — written before this landed, or by a
    process that loaded no local model. A malformed value reads as ``[]``
    rather than raising: a footer is additive metadata, and a file whose
    contents are fine is not worth refusing over its record of itself.
    """
    if not metadata:
        return []
    raw = metadata.get(MODELS_KEY.encode())
    if raw is None:
        return []
    try:
        entries = json.loads(raw.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(entries, list):
        return []
    return [
        {"name": str(e["name"]), "digest": str(e["digest"])}
        for e in entries
        if isinstance(e, dict) and "name" in e and "digest" in e
    ]


_SLOT_MODEL_FIELDS = ("slot", "name", "distribution", "version")


def read_footer_slot_models(metadata: Mapping[bytes, bytes] | None) -> list[dict[str, str]]:
    """Decode the slot-models key out of a Parquet footer.

    ``[]`` when the file carries none — written before this landed, or by a
    process that built no swappable-slot model — for the same reason
    :func:`read_footer_models` reads one as empty.
    """
    if not metadata:
        return []
    raw = metadata.get(SLOT_MODELS_KEY.encode())
    if raw is None:
        return []
    try:
        entries = json.loads(raw.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(entries, list):
        return []
    return [
        {field: str(e[field]) for field in _SLOT_MODEL_FIELDS}
        for e in entries
        if isinstance(e, dict) and all(field in e for field in _SLOT_MODEL_FIELDS)
    ]


def read_footer_model_check(metadata: Mapping[bytes, bytes] | None) -> dict | None:
    """Decode the model-check key out of a Parquet footer.

    ``None`` when the file carries none (written before the check existed, or by
    a process that was never asked to check), and for a malformed value, for the
    reason :func:`read_footer_models` reads one as empty.
    """
    raw = (metadata or {}).get(MODEL_CHECK_KEY.encode())
    if raw is None:
        return None
    try:
        payload = json.loads(raw.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def stamp_from_footers(paths: Iterable[Path], stage: str) -> RunStamp | None:
    """The one run *paths* agree on, stamped for *stage*; ``None`` if there is none.

    ``None`` when no readable file carries a run, and also when the files name
    more than one run — a stamp naming one of several runs would be worse than
    no stamp, which is the rule the ingest-root footer already follows.
    """
    seen: set[tuple[str, str]] = set()
    preset = ""
    for path in paths:
        try:
            existing = read_footer_stamp(pq.read_schema(str(path)).metadata)
        except (OSError, pa.ArrowInvalid):  # absent, unreadable, or not parquet
            continue
        if run_id := existing.get("run_id", ""):
            seen.add((run_id, existing.get("config_digest", "")))
            preset = existing.get("preset", "") or preset
    if len(seen) != 1:
        return None
    run_id, digest = seen.pop()
    # `preset` is determined by the config digest (dataset.name is inside it),
    # so the one run the files agree on has one preset; carry it forward.
    return RunStamp.inherit(run_id, digest, stage=stage, preset=preset)


def stamp_for_sidecar(base_path: Path, stage: str) -> RunStamp | None:
    """The stamp a sidecar of *base_path* inherits, or ``None`` if there is none.

    *base_path* is the batch's base shard path (``batch-0001.parquet``). The
    run is read from a stamped sibling of that base: the extraction shard or
    its manifest by preference, and failing those any sibling of the batch,
    since every file of one run carries the same run id and digest.

    Falling back matters for the distributed path. A stage worker stages in
    only the inputs its contract declares, and five stages (``embed``,
    ``link``, ``pii``, ``quality``, ``graph-refresh``) do not declare the
    elements shard — so insisting on it would leave their sidecars unstamped
    on a worker and stamped locally, which is exactly the local/distributed
    divergence the stamp exists to rule out.

    ``None`` when no sibling carries a run, or when the siblings disagree — a
    sidecar of an unstamped shard is written unstamped, on the same terms
    extraction already writes one for a run it cannot name.
    """
    preferred = [base_path.parent / f"{base_path.stem}{s}" for s in _PREFERRED_SOURCES]
    for path in preferred:
        if (stamp := stamp_from_footers([path], stage)) is not None:
            return stamp
    others = sorted(set(base_path.parent.glob(f"{base_path.stem}.*.parquet")) - set(preferred))
    return stamp_from_footers(others, stage)


def sidecar_footer(base_path: Path, stage: str) -> dict[bytes, bytes] | None:
    """:func:`stamp_for_sidecar` as footer metadata, for a writer to pass on."""
    stamp = stamp_for_sidecar(base_path, stage)
    meta = stamp.footer_metadata() if stamp is not None else None
    if (digest := _workflow_digest.get()) is not None:
        # A workflow digest is a fact about this write even where the run cannot be named.
        meta = {**(meta or {}), WORKFLOW_DIGEST_KEY.encode(): digest.encode()}
    return meta


__all__ = [
    "COMMIT_KEY",
    "CONFIG_DIGEST_KEY",
    "MODELS_KEY",
    "MODEL_CHECK_KEY",
    "PRESET_KEY",
    "RUN_ID_KEY",
    "SLOT_MODELS_KEY",
    "STAGE_KEY",
    "VERSION_KEY",
    "WORKFLOW_DIGEST_KEY",
    "RunStamp",
    "config_digest",
    "read_footer_model_check",
    "read_footer_models",
    "read_footer_slot_models",
    "read_footer_stamp",
    "sidecar_footer",
    "stamp_for_sidecar",
    "stamp_from_footers",
    "workflow_context",
]
