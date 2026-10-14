"""Local model path resolution for offline / air-gapped deployment.

Roots are searched **per artefact**, in this order:

1. ``WOMBLEX_MODELS_DIR`` environment variable (explicit override).
2. ``_models/`` bundled inside the installed package — this is the path used
   after ``pip install womblex`` and is what makes air-gapped use viable.
3. ``models/`` sibling of ``src/`` — backward compatibility for editable
   installs and the historical repo layout.
4. Roots supplied by installed packages through the ``womblex.model_roots``
   entry-point group, searched last so a plugin cannot shadow a built-in.

Per artefact, not "first root wins", because the roots hold *different*
artefacts. A container image mounts the large ones (layout, embedding, OCR)
at ``WOMBLEX_MODELS_DIR=/app/models`` while the small ones (``en_AU``,
``kanon-2-tokenizer``) ship inside the wheel — so a single-root resolver makes
the override **shadow** the bundled artefacts rather than supplement them, and
``resolve_local_model_path("en_AU")`` starts returning the bare string.

Supports both HuggingFace hub-style cache layouts and flat model directories.

**Resolution is also the record of what a run loaded.** Which local models a
run used is not recoverable after the fact: nothing on disk says whether a
shard was OCR'd by one layout model or its replacement, and a model directory
is swapped in place far more often than it is renamed. So every resolution that
actually finds an artefact is recorded here, and the run stamp carries the
recorded set — name and content digest — into each Parquet footer as it is
written. `store/run_manifest.py` unions those per-file records into the run
record.

Recording at resolution rather than at load is deliberate: this is the one
choke point every model path in the library already goes through, so a new
model cannot be loaded without appearing in the record. A caller that is only
*probing* for a model it will not load passes ``record=False`` — the record
says what the run loaded, and a resolution that leads to no load would
overstate it.

Digests are over the artefact's bytes, so they recompute from the model files
alone, and they are taken lazily: a run that loads a model pays the hash once,
at the first file it writes afterwards, and a run that loads none pays nothing.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)

#: Bytes per read while digesting. Model artefacts run to hundreds of
#: megabytes; the whole file is never held in memory.
_DIGEST_CHUNK = 1 << 20


def _repo_models_dir() -> Path | None:
    """``models/`` beside ``src/`` — the editable-install / repo layout."""
    current = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = current / "models"
        if candidate.is_dir() and (current / "src").is_dir():
            return candidate
        current = current.parent
    return None


MODEL_ROOTS_GROUP = "womblex.model_roots"


@cache
def _plugin_roots() -> tuple[Path, ...]:
    """Roots installed packages declare under ``womblex.model_roots``.

    An entry point's value is a directory path (``str``/``Path``), or a
    callable returning one or an iterable of them. A broken entry point is
    logged and skipped.
    """
    from importlib.metadata import entry_points

    roots: list[Path] = []
    for ep in entry_points(group=MODEL_ROOTS_GROUP):
        try:
            value = ep.load()
            if callable(value):
                value = value()
            items = [value] if isinstance(value, (str, Path)) else list(value)
            roots.extend(Path(item) for item in items)
        except Exception:
            logger.exception("model root plugin %s failed to load", ep.name)
    return tuple(roots)


def model_roots() -> tuple[Path, ...]:
    """Every existing models root, in resolution order.

    Duplicates are dropped, so pointing ``WOMBLEX_MODELS_DIR`` at the bundled
    directory does not search it twice.
    """
    roots: list[Path] = []

    def add(path: Path | None) -> None:
        if path is not None and path.is_dir() and path not in roots:
            roots.append(path)

    env_override = os.environ.get("WOMBLEX_MODELS_DIR")
    add(Path(env_override) if env_override else None)
    add(Path(__file__).resolve().parent.parent / "_models")
    add(_repo_models_dir())
    for plugin_root in _plugin_roots():
        add(plugin_root)
    return tuple(roots)


def find_models_dir() -> Path | None:
    """The highest-priority existing models root, or None if there is none.

    Kept for callers that want *a* root rather than a resolved artefact. Do
    not use it to build an artefact path — the artefact may live under a
    lower-priority root; call :func:`resolve_local_model_path` instead.
    """
    roots = model_roots()
    return roots[0] if roots else None


def _resolve_under(root: Path, model_name: str) -> Path | None:
    """*model_name* under *root*, or None if this root does not hold it."""
    local = root / model_name

    if local.is_file():
        return local

    if local.is_dir():
        refs_main = local / "refs" / "main"
        if refs_main.is_file():
            snapshot_dir = local / "snapshots" / refs_main.read_text().strip()
            if snapshot_dir.is_dir():
                return snapshot_dir
        return local

    return None


def resolve_local_model_path(model_name: str, *, record: bool = True) -> str | Path:
    """Return a local path to *model_name* if pre-downloaded, else the name itself.

    Every root from :func:`model_roots` is searched, in order, and the first
    that actually holds *model_name* wins.

    Understands the HuggingFace hub cache layout::

        <root>/<model_name>/refs/main           → contains snapshot hash
        <root>/<model_name>/snapshots/<hash>/   → actual model files

    If a flat directory ``<root>/<model_name>/`` exists without the hub
    layout, that directory is returned directly.

    For non-directory artefacts (e.g. a single ``.onnx`` checkpoint), pass the
    filename as *model_name* and the full file path is returned if it exists.

    Args:
        model_name: HuggingFace model identifier or bare filename.
        record: Whether a successful resolution counts as this run having
            loaded the model. Left on for every real load; passed ``False`` by
            callers that only ask whether an artefact is present.

    Returns:
        Local ``Path`` if found, otherwise the original *model_name* string
        (so callers can pass the result directly to library constructors).
    """
    for root in model_roots():
        resolved = _resolve_under(root, model_name)
        if resolved is not None:
            if record:
                _record_resolved(model_name, resolved)
            return resolved
    return model_name


# ---------------------------------------------------------------------------
# What this process loaded
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedModel:
    """One local model artefact a run resolved, and the digest of its bytes.

    ``digest`` is ``sha256:…`` over the artefact's contents and nothing else,
    so a reader holding the same model files recomputes it without needing the
    run, the machine or the path it was resolved from. The path itself is
    deliberately not carried: it is deployment location, excluded for the same
    reason ``paths`` is excluded from the configuration digest.
    """

    name: str
    digest: str


#: Requested name -> resolved path, for every artefact this process actually
#: found. Insertion-ordered, but read back sorted so a run's record does not
#: depend on the order its stages happened to load models in.
_RESOLVED: dict[str, Path] = {}

#: True while :func:`suppress_recording` is active.
_suppress_record = False


def _record_resolved(model_name: str, path: Path) -> None:
    """Note that *model_name* resolved to *path*, for the run record.

    First resolution wins, matching the resolver itself: a name resolves to one
    artefact per process, so a later call cannot change what was loaded. A
    no-op while :func:`suppress_recording` is active.
    """
    if _suppress_record:
        return
    _RESOLVED.setdefault(model_name, path)


@contextmanager
def suppress_recording() -> Iterator[None]:
    """While active, a resolution is not recorded as loaded.

    The pre-run model check resolves every configured model's local artefact
    to confirm it is present, but that resolution is a probe, not the run
    loading it — ``utils/model_check.py`` wraps its whole pass in this (and
    the matching ``model_registry.suppress_use_recording``) so the check
    itself never appears in the ``womblex.models`` footer key. Unlike that
    sibling suppression, nothing here caches a resolution across calls, so a
    later, unsuppressed resolution of the same name always records.
    """
    global _suppress_record
    previous, _suppress_record = _suppress_record, True
    try:
        yield
    finally:
        _suppress_record = previous


def _digest_file(digest: hashlib._Hash, path: Path) -> None:
    with path.open("rb") as fh:
        while block := fh.read(_DIGEST_CHUNK):
            digest.update(block)


@cache
def digest_model_path(path: Path) -> str:
    """``sha256:…`` over the artefact at *path* — a file's bytes, or a tree's.

    A directory digests every file beneath it, each contributing its
    root-relative POSIX path and its bytes, walked in sorted order. Naming the
    relative path as well as the content is what makes a rename of two
    same-sized files inside the tree a different digest rather than the same
    one.

    Cached per path: a run writes many files and would otherwise re-hash
    hundreds of megabytes for each of them.
    """
    digest = hashlib.sha256()
    if path.is_file():
        _digest_file(digest, path)
        return "sha256:" + digest.hexdigest()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        _digest_file(digest, child)
    return "sha256:" + digest.hexdigest()


def loaded_models() -> tuple[LoadedModel, ...]:
    """Every local model this process resolved, name-sorted, each digested.

    An artefact that has become unreadable since it was resolved is dropped
    with a warning rather than failing the write it was asked for: the run's
    output is not worth losing to an incomplete record of it, and a missing
    entry is visible in the record as an absence.
    """
    out: list[LoadedModel] = []
    for name, path in sorted(_RESOLVED.items()):
        try:
            out.append(LoadedModel(name, digest_model_path(path)))
        except OSError as exc:
            logger.warning("model %s at %s could not be digested: %s", name, path, exc)
    return tuple(out)


def record_loaded_path(model_name: str, path: str | Path) -> None:
    """Record an artefact loaded from outside the models roots.

    :func:`resolve_local_model_path` covers every artefact Womblex resolves,
    but not one a *library* loads from inside its own wheel — RapidOCR falls
    back to its bundled PaddleOCR v4 models when the v5 directory is absent,
    and those never pass through the resolver. Without this the record would
    show no OCR model for a run that OCR'd, which is the silent kind of wrong
    the record exists to avoid. The caller names what it loaded and where.

    A path that does not exist is not recorded — there would be nothing to
    digest — but it is logged: a caller reaching for an artefact that has moved
    is exactly the case that would otherwise leave the record quietly short.
    """
    resolved = Path(path)
    if not resolved.exists():
        logger.warning("model %s not recorded: nothing at %s", model_name, resolved)
        return
    _record_resolved(model_name, resolved)


def reset_loaded_models() -> None:
    """Forget what this process resolved. For tests that swap models roots."""
    _RESOLVED.clear()
    digest_model_path.cache_clear()


__all__ = [
    "LoadedModel",
    "digest_model_path",
    "find_models_dir",
    "loaded_models",
    "model_roots",
    "record_loaded_path",
    "reset_loaded_models",
    "resolve_local_model_path",
    "suppress_recording",
]
