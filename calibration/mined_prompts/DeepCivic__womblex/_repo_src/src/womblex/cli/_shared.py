"""Shared CLI helpers and the Command record used by topic modules."""
from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import NamedTuple

logger = logging.getLogger("womblex")

# Standalone images open as one-page documents and take the scanned-page OCR
# route, so they are ingested like a PDF.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".jp2"}

SUPPORTED_EXTENSIONS = {
    ".pdf", ".csv", ".xlsx", ".xls", ".docx", ".md", ".markdown", *IMAGE_EXTENSIONS,
}


class Command(NamedTuple):
    """One CLI subcommand: name, help, parser registration, handler."""

    name: str
    help: str
    register: Callable[[argparse.ArgumentParser], None]
    handler: Callable[[argparse.Namespace], int]


LOG_FORMATS = ("text", "json")


def add_models_check_argument(p: argparse.ArgumentParser) -> None:
    """``--models-check``: override ``processing.models_check`` for this invocation."""
    p.add_argument(
        "--models-check", choices=("off", "load", "smoke"), default=None,
        help="How far to verify the configured models before work starts "
             "(overrides processing.models_check): off, load, or smoke "
             "(load plus one inference on a built-in input).",
    )


def apply_models_check(config: object, args: argparse.Namespace) -> None:
    """Apply a ``--models-check`` override to *config*, if one was given."""
    if getattr(args, "models_check", None):
        config.processing.models_check = args.models_check  # type: ignore[attr-defined]


def setup_logging(verbose: bool = False, log_format: str | None = None) -> None:
    """Configure logging; *log_format* falls back to ``WOMBLEX_LOG_FORMAT``, then text."""
    fmt = (log_format or os.environ.get("WOMBLEX_LOG_FORMAT") or "text").lower()
    if fmt not in LOG_FORMATS:
        raise ValueError(f"unknown log format {fmt!r}; expected one of {LOG_FORMATS}")
    level = logging.DEBUG if verbose else logging.INFO
    if fmt == "json":
        from womblex.utils.log_format import JsonFormatter

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.basicConfig(level=level, handlers=[handler], force=True)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class NestedCorpusError(ValueError):
    """Supported documents sit in subdirectories of the ingest location.

    A run ingests what is directly under the location it is given. Documents a
    level down would be ingested by some entry points and passed over by others,
    so every entry point refuses them instead — see ``select_supported``.

    Subclasses ``ValueError`` so the console's enqueue route reports it as bad
    input rather than a 500.
    """

    def __init__(self, location: str, nested: Mapping[str, int]) -> None:
        self.location = location
        self.nested = dict(nested)
        named = ", ".join(
            f"{d}/ ({n} document{'s' if n != 1 else ''})"
            for d, n in sorted(nested.items())[:3]
        )
        more = len(nested) - 3
        if more > 0:
            named += f", and {more} more subdirector{'ies' if more != 1 else 'y'}"
        super().__init__(
            f"{location} holds documents in subdirectories: {named}. "
            f"A run ingests only what is directly under the location it is given. "
            f"Point at the subdirectory that holds the documents, or flatten the corpus."
        )


def select_supported(relpaths: Iterable[str], *, location: str) -> list[str]:
    """The supported documents directly under *location*, from its full listing.

    *relpaths* is every file under the location, relative to it, from a
    recursive listing — the walk is what makes a nested corpus visible rather
    than silently skipped.

    Raises :class:`NestedCorpusError` if any supported document sits in a
    subdirectory, whether or not the top level holds documents too: a mixed
    layout ingests in part, which is the same defect as ingesting nothing.
    Subdirectories holding no supported document are ignored, so a corpus
    alongside ``.git`` or a notes directory still runs.

    The one enumeration rule, shared by the local CLI, the cloud enqueue and the
    console, so the three cannot drift apart again.
    """
    top: list[str] = []
    nested: dict[str, int] = {}
    for rel in relpaths:
        parts = PurePosixPath(rel).parts
        if not parts or PurePosixPath(rel).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if len(parts) == 1:
            top.append(rel)
        else:
            nested[parts[0]] = nested.get(parts[0], 0) + 1
    if nested:
        raise NestedCorpusError(location, nested)
    return sorted(top)


# An ingest-relative prefix becomes a path segment joined onto the ingest root,
# the way a run id is joined onto a run root. ``RemoteStore`` strips
# leading and trailing slashes but does not resolve ``..``, so a climbing prefix
# lists — and enqueues — outside the location the deployment configured.
# Refused rather than sanitised, on the same footing as ``is_safe_run_id``: a
# real prefix is a key scope inside the corpus, so there is nothing legitimate
# to rewrite.
_UNSAFE_IN_PREFIX = ("\\", "\x00", "://")


def normalise_prefix(input_prefix: str | None) -> str:
    """*input_prefix* as a store-key scope, or ``ValueError`` if it escapes.

    Returns the segments joined back with single slashes (``""`` for the whole
    ingest root), so every spelling of one location yields one scope and the
    entry points cannot disagree about what a prefix selects.

    Rebuilt from the path's parts rather than merely stripped, because the
    prefix is not only a listing scope: the keys under it are what an enqueue
    writes to the queue, what the worker downloads, and what is recorded as each
    document's ``source_relpath``. A ``.`` segment or doubled slash the operator
    typed would ride into all three — harmless on a local path, a different key
    on an object store, which has no path semantics to collapse it.
    """
    prefix = (input_prefix or "").strip().strip("/")
    parts = PurePosixPath(prefix).parts  # collapses `.` and `//`; keeps `..`
    if any(c in prefix for c in _UNSAFE_IN_PREFIX) or ".." in parts:
        raise ValueError(
            f"unsafe input_prefix: {input_prefix!r} — a prefix names a location "
            "inside the configured ingest root."
        )
    return "/".join(parts)


def discover_files(input_root: Path, limit: int | None = None, skip: int = 0) -> list[Path]:
    """Discover supported documents directly under *input_root*.

    Raises :class:`NestedCorpusError` for a corpus with documents a level down.
    """
    listing = (
        p.relative_to(input_root).as_posix()
        for p in input_root.rglob("*")
        if p.is_file()
    )
    files = [input_root / name for name in select_supported(listing, location=str(input_root))]
    if skip:
        files = files[skip:]
    if limit:
        files = files[:limit]
    return files


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


def check_stage_models(
    args: argparse.Namespace, scopes: tuple[str, ...], *, config: object = None, **sections: object,
) -> bool:
    """Check the models a standalone stage command is about to use; ``False`` stops it.

    *config* is the loaded ``--config`` (``None`` means stage defaults) and
    *sections* are the stage sections the command will actually run with, which
    may carry CLI overrides. Called before the command writes anything.
    """
    from womblex.config import WomblexConfig
    from womblex.utils.model_check import check_models

    base = config or WomblexConfig.model_validate({
        "dataset": {"name": "stage"},
        "paths": {"input_root": ".", "output_root": ".", "checkpoint_dir": "."},
    })
    cfg = base.model_copy(update=sections)  # type: ignore[attr-defined]
    apply_models_check(cfg, args)
    result = check_models(cfg, scopes=scopes)
    if result.failures:
        logger.error("Model check failed; nothing was processed: %s", result.message())
        return False
    return True
