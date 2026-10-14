"""Resolve a published row back to the source document it describes.

Every sidecar in a run joins on ``source_hash``, and the manifest is the only
mapping from that key back to a document — but a manifest row names a corpus
and a path, not a file that is known to still be there. This closes the loop:
given a run's manifest and access to the corpus, hand it a ``source_hash`` and
it returns the file, having verified the bytes, or says precisely why it cannot.
:meth:`SourceResolver.for_run` is the entry point from a run directory, and
``womblex resolve-source`` the command over it.

**Every call returns a** :class:`Resolution` — a status and a reason, never an
empty return and never an exception — so a consumer walking a run's rows gets
an account of the ones that failed rather than a truncated list.

**Two hash bases, and the resolver knows which it faces.** A file ingest hashes
the source bytes (``store/output.py``); the pre-extracted records path hashes
the record id plus its text (``ingest/records.py``), where there is no file to
hash. Records rows are therefore declined by name, naming
``*.provenance.parquet`` as the back-link that does answer for them — out of
scope for this resolver, which is not the same as missing.

**Root + relpath, so a corpus that moves re-resolves.** Resolution is by the
path under the ingest root that P1 records and then by content hash, never by
an absolute path, so a moved corpus re-resolves against its new root unchanged.

**The index is the run's own enumeration.** Both paths go through
``select_supported`` in ``cli/_shared.py``, the one rule every entry point
ingests by, so a document returned here is one a run could have processed and a
corpus refused here is one a run refuses too — agreement by construction rather
than by a second walk that happens to match.

Scoped to the NLP path. The standalone register ingests (G-NAF, ABN,
geospatial) carry no ``source_hash`` and are not addressed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

import pyarrow as pa

from womblex.cli._shared import NestedCorpusError, discover_files, select_supported
from womblex.store.output import _read_shard, _source_hash, read_manifest
from womblex.store.run_manifest import RUN_MANIFEST_FILENAME

logger = logging.getLogger(__name__)

#: What a ``source_hash`` was taken over. Reported on every resolution so a
#: consumer never has to infer it from the value's shape.
HASH_BASIS_FILE_BYTES = "file_bytes"
HASH_BASIS_RECORD_ID_TEXT = "record_id+text"

#: ``extraction_method`` values whose hash is not over file bytes.
_RECORD_METHODS = frozenset({"records"})

#: The sidecar that answers for a row this resolver declines.
_RECORDS_BACKLINK = "*.provenance.parquet"

ResolutionStatus = Literal["resolved", "hash_mismatch", "not_found", "unsupported_basis"]

RESOLVED: ResolutionStatus = "resolved"
HASH_MISMATCH: ResolutionStatus = "hash_mismatch"
NOT_FOUND: ResolutionStatus = "not_found"
UNSUPPORTED_BASIS: ResolutionStatus = "unsupported_basis"


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving one ``source_hash``.

    ``path`` is populated whenever a file was located, which includes
    ``hash_mismatch`` — a corpus re-saved with different bytes resolves by path
    and reports the mismatch, which is more use than reporting nothing found.
    """

    source_hash: str
    status: ResolutionStatus
    hash_basis: str
    detail: str
    path: Path | None = None
    doc_id: str = ""
    source_relpath: str = ""

    @property
    def ok(self) -> bool:
        return self.status == RESOLVED


def hash_basis_for(extraction_method: str) -> str:
    """The hash basis a manifest row's ``extraction_method`` implies."""
    return (
        HASH_BASIS_RECORD_ID_TEXT
        if extraction_method in _RECORD_METHODS
        else HASH_BASIS_FILE_BYTES
    )


def load_manifest(location: Path) -> pa.Table:
    """Read the document manifest for a run, from whichever shape is given.

    A run root (preferring the consolidated ``manifest.parquet`` P9 publishes,
    else its ``documents/`` shard directory), a shard directory, or a manifest
    Parquet directly. Raises ``FileNotFoundError`` when none of the three is
    there, rather than returning an empty table that would resolve nothing.
    """
    p = Path(location)
    if p.is_file():
        return _read_shard(p, "manifest")
    consolidated = p / RUN_MANIFEST_FILENAME
    if consolidated.is_file():
        return _read_shard(consolidated, "manifest")
    documents = p / "documents"
    shard_dir = documents if documents.is_dir() else p
    if not any(shard_dir.glob("*._manifest.parquet")):
        raise FileNotFoundError(
            f"no manifest under {p}: expected {RUN_MANIFEST_FILENAME}, "
            f"a documents/ shard directory, or *._manifest.parquet siblings"
        )
    return read_manifest(shard_dir)


def _ingestable(relpath: str, root: Path) -> bool:
    """Would a run ingest *relpath* from *root*? The shared enumeration rule."""
    if not relpath:
        return False
    try:
        return bool(select_supported([relpath], location=str(root)))
    except NestedCorpusError:
        return False


@dataclass(frozen=True)
class _Row:
    doc_id: str
    relpath: str
    filename: str
    extraction_method: str


class SourceResolver:
    """Resolve a run's ``source_hash`` values against a corpus on disk.

    Built over a manifest table and a local corpus root. The index — a full
    corpus scan, hashing every document — is built lazily and only when path
    resolution has not answered, so resolving a corpus that is still where the
    run found it costs one file read per lookup rather than a scan.
    """

    def __init__(self, manifest: pa.Table, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._rows = _rows_by_hash(manifest)
        self._index: dict[str, Path] | None = None
        self._index_error: str | None = None

    @classmethod
    def for_run(cls, location: Path, *, root: str | Path | None = None) -> SourceResolver:
        """Build from a run root / shard dir / manifest file.

        *root* overrides the corpus location — that is how a corpus that has
        moved is re-resolved. Without it the root is taken from the manifest's
        own ``ingest_root``, which must be a single ``file://`` root: an
        object-store root has no local corpus to index, and a mixed or absent
        root names nothing, so both raise here rather than resolving nothing
        row by row.
        """
        manifest = load_manifest(Path(location))
        return cls(manifest, root if root is not None else _declared_root(manifest, location))

    @property
    def index(self) -> dict[str, Path]:
        """``source_hash`` → path, over the documents a run would ingest.

        Empty when the corpus is one a run refuses (documents in
        subdirectories); the refusal is reported on each resolution rather than
        raised, so a caller walking every row still gets an account.
        """
        if self._index is None:
            try:
                files = discover_files(self.root)
            except NestedCorpusError as e:
                self._index, self._index_error = {}, str(e)
            else:
                self._index = {}
                for f in files:
                    digest = _source_hash(str(f))
                    if digest:
                        self._index.setdefault(digest, f)
                logger.info("source index: %d document(s) under %s", len(self._index), self.root)
        return self._index

    def resolve(self, source_hash: str) -> Resolution:
        """Resolve one ``source_hash``. Always returns a :class:`Resolution`."""
        row = self._rows.get(source_hash)
        if row is None:
            return Resolution(
                source_hash, NOT_FOUND, HASH_BASIS_FILE_BYTES,
                "no manifest row carries this source_hash",
            )
        basis = hash_basis_for(row.extraction_method)
        if basis != HASH_BASIS_FILE_BYTES:
            return Resolution(
                source_hash, UNSUPPORTED_BASIS, basis,
                f"hashed over {basis}, not file bytes — no file to resolve; "
                f"join {_RECORDS_BACKLINK} on source_hash for this row's origin",
                doc_id=row.doc_id, source_relpath=row.relpath,
            )
        return self._resolve_file(source_hash, row, basis)

    def resolve_all(self) -> list[Resolution]:
        """Resolve every distinct ``source_hash``, in manifest order.

        Rows, not documents, are what a manifest holds — but two rows can carry
        one hash (the same bytes ingested under two names), and they resolve to
        the same file, so the duplicate is collapsed rather than resolved twice.
        """
        return [self.resolve(h) for h in self._rows]

    def _resolve_file(self, source_hash: str, row: _Row, basis: str) -> Resolution:
        rel = row.relpath or row.filename

        def result(status: ResolutionStatus, detail: str, path: Path | None = None) -> Resolution:
            return Resolution(
                source_hash, status, basis, detail,
                path=path, doc_id=row.doc_id, source_relpath=rel,
            )

        # `_source_hash` returns "" for a file that is not there, so one read
        # answers both "is it where the manifest says" and "is it the same file".
        named = self.root / rel if _ingestable(rel, self.root) else None
        actual = _source_hash(str(named)) if named is not None else ""
        if actual and actual == source_hash:
            return result(RESOLVED, "hash verified against file bytes", named)

        # The path did not answer; only now does the full-corpus scan earn itself.
        found = self.index.get(source_hash)
        if found is not None:
            return result(
                RESOLVED,
                f"hash verified; not where the manifest names it ({rel or 'none'})",
                found,
            )
        if actual:
            return result(
                HASH_MISMATCH,
                f"resolved by path; contents differ "
                f"(manifest {source_hash[:12]}, file {actual[:12]})",
                named,
            )
        return result(
            NOT_FOUND,
            self._index_error
            or f"no document under {self.root} matches by path ({rel or 'none'}) or by hash",
        )


def _rows_by_hash(manifest: pa.Table) -> dict[str, _Row]:
    """Index the manifest by ``source_hash``, first row per hash winning.

    Two rows can share a hash — one document ingested under two names — and
    both name the same bytes, so the first is kept and the other's path is not
    reported. Columns are read defensively: a shard predating one of them
    back-fills to ``""`` here rather than raising, matching the reader
    convention.
    """
    names = ("source_hash", "doc_id", "source_relpath", "filename", "extraction_method")
    n = manifest.num_rows
    cols = {
        name: [str(v or "") for v in manifest.column(name).to_pylist()]
        if name in manifest.schema.names else [""] * n
        for name in names
    }
    rows: dict[str, _Row] = {}
    for i, key in enumerate(cols["source_hash"]):
        if key and key not in rows:
            rows[key] = _Row(
                cols["doc_id"][i], cols["source_relpath"][i],
                cols["filename"][i], cols["extraction_method"][i],
            )
    return rows


def _declared_root(manifest: pa.Table, location: Path) -> str:
    """The single ``file://`` ingest root the manifest declares."""
    if "ingest_root" not in manifest.schema.names:
        raise ValueError(f"{location} declares no ingest_root; pass root= explicitly")
    declared = {str(v or "") for v in manifest.column("ingest_root").to_pylist()} - {""}
    if len(declared) != 1:
        raise ValueError(
            f"{location} declares {len(declared)} ingest roots; pass root= explicitly"
        )
    root = declared.pop()
    if not root.startswith("file://"):
        raise ValueError(
            f"{location} was ingested from {root}, which has no local corpus to index; "
            "pass root= naming a local copy"
        )
    # `qualify_root` records the root via `Path.as_uri()`, which percent-encodes.
    return unquote(root.removeprefix("file://"))


__all__ = [
    "HASH_BASIS_FILE_BYTES",
    "HASH_BASIS_RECORD_ID_TEXT",
    "HASH_MISMATCH",
    "NOT_FOUND",
    "RESOLVED",
    "UNSUPPORTED_BASIS",
    "Resolution",
    "ResolutionStatus",
    "SourceResolver",
    "hash_basis_for",
    "load_manifest",
]
