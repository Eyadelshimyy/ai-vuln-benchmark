"""Where a source document came from: the ingest root and its root-relative path.

The manifest has always named a document by ``filename`` — a bare basename that
says nothing about the corpus it came from and joins back to nothing. This adds
the missing pair: a scheme-qualified ``ingest_root`` (``file:///srv/corpus``,
``s3://bucket/inbox``), declared by configuration or by the command that started
the run and never inferred from the working directory, and a ``source_relpath``
under it, so root + relpath names the document and a corpus that moves
re-resolves without a rewrite.

Both land twice: as manifest columns (document grain) and in every extraction
Parquet's footer key-value metadata (file grain), so a shard copied out of its
run still says where its documents came from. Footer keys are namespaced
``womblex.*`` — the convention ``store/register_manifest.py`` already reads back
from the register ingests — and are additive: a reader that ignores them reads
the file unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from womblex.store.remote import validate_location_uri

NAMESPACE = "womblex"
INGEST_ROOT_KEY = f"{NAMESPACE}.ingest_root"
SOURCE_RELPATH_KEY = f"{NAMESPACE}.source_relpath"
COLLECTION_ID_KEY = f"{NAMESPACE}.collection_id"


def qualify_root(root: str | Path) -> str:
    """Return *root* as a scheme-qualified, trailing-slash-free URI.

    A URI passes through validated; a plain path is made absolute and expressed
    as ``file://``. An empty root raises rather than defaulting to the working
    directory — an undeclared ingest root is the one thing never invented here.
    A declared *relative* path is resolved against the working directory once,
    and the resolved absolute root is what gets recorded.
    """
    raw = str(root).strip()
    if not raw:
        raise ValueError("ingest root is empty; declare it in config or on the command line")
    # Validates a URI and, for a plain path, catches the typo that would
    # otherwise become a directory literally named "s3:" (`s3:/bucket`).
    validate_location_uri(raw)
    if raw.startswith("file://"):
        return Path(raw[len("file://") :]).expanduser().as_uri().rstrip("/")
    if "://" in raw:
        return raw.rstrip("/")
    return Path(raw).expanduser().resolve().as_uri().rstrip("/")


def relpath_under(root_uri: str, source_path: str | Path) -> str:
    """The POSIX path of *source_path* relative to *root_uri*.

    Raises ``ValueError`` when the document does not live under the declared
    root, or when the root is an object store (no local path to subtract — the
    caller supplies the store key instead). Loud at the first batch beats a run
    of rows that resolve to nothing.
    """
    if "://" in root_uri and not root_uri.startswith("file://"):
        raise ValueError(
            f"cannot derive a relative path under object-store root {root_uri!r} "
            "from a local path; supply the store key explicitly"
        )
    base = Path(root_uri.removeprefix("file://")).expanduser().resolve()
    resolved = Path(source_path).expanduser().resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError as e:
        raise ValueError(f"{resolved} is not under the declared ingest root {root_uri}") from e


@dataclass(frozen=True)
class IngestProvenance:
    """The declared origin of a batch's documents.

    ``relpaths`` maps a *local* path to the root-relative path it came from. It
    exists for the distributed path, where documents are extracted from a
    scratch directory whose paths say nothing about the store keys they were
    downloaded from. A local run leaves it empty and the relative path is
    derived from the root.
    """

    ingest_root: str
    collection_id: str
    relpaths: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def declare(
        cls,
        root: str | Path,
        collection_id: str,
        *,
        relpaths: Mapping[str | Path, str] | None = None,
    ) -> IngestProvenance:
        if not collection_id:
            raise ValueError("collection_id is empty; every manifest row must name its corpus")
        keyed = {str(Path(k).expanduser().resolve()): v for k, v in (relpaths or {}).items()}
        return cls(qualify_root(root), collection_id, keyed)

    @classmethod
    def from_config(cls, config: object) -> IngestProvenance:
        """Derive from a ``WomblexConfig``: declared ``paths.ingest_root`` wins
        (the only way to name an object-store root for a run over staged
        copies), else the already-required ``paths.input_root``."""
        paths = config.paths  # type: ignore[attr-defined]
        return cls.declare(
            paths.ingest_root or paths.input_root,
            config.dataset.name,  # type: ignore[attr-defined]
        )

    def relpath_for(self, source_path: str | Path) -> str:
        """Root-relative path of *source_path*: explicit mapping, else derived."""
        mapped = self.relpaths.get(str(Path(source_path).expanduser().resolve()))
        if mapped is not None:
            return mapped.strip("/")
        return relpath_under(self.ingest_root, source_path)

    def footer_metadata(self, relpaths: Iterable[str]) -> dict[bytes, bytes]:
        """Namespaced footer metadata for a file covering *relpaths*.

        Root and collection are single values; the relative paths are a JSON
        array because a batch shard covers many documents. A one-document file
        is a one-element array — uniform rather than two shapes to read.
        """
        seen: list[str] = []
        for rel in relpaths:
            if rel not in seen:
                seen.append(rel)
        return {
            INGEST_ROOT_KEY.encode(): self.ingest_root.encode(),
            COLLECTION_ID_KEY.encode(): self.collection_id.encode(),
            SOURCE_RELPATH_KEY.encode(): json.dumps(seen).encode(),
        }


def read_footer_provenance(metadata: Mapping[bytes, bytes] | None) -> dict[str, object]:
    """Decode the ``womblex.*`` provenance keys out of a Parquet footer.

    ``{}`` when the file carries none (written before this landed, or another
    producer). ``source_relpath`` comes back as a list of strings.
    """
    if not metadata:
        return {}
    decoded = {k.decode(): v.decode() for k, v in metadata.items()}
    out: dict[str, object] = {}
    for key, name in ((INGEST_ROOT_KEY, "ingest_root"), (COLLECTION_ID_KEY, "collection_id")):
        if key in decoded:
            out[name] = decoded[key]
    if SOURCE_RELPATH_KEY in decoded:
        try:
            out["source_relpath"] = json.loads(decoded[SOURCE_RELPATH_KEY])
        except json.JSONDecodeError:
            out["source_relpath"] = [decoded[SOURCE_RELPATH_KEY]]
    return out


__all__ = [
    "COLLECTION_ID_KEY",
    "INGEST_ROOT_KEY",
    "NAMESPACE",
    "SOURCE_RELPATH_KEY",
    "IngestProvenance",
    "qualify_root",
    "read_footer_provenance",
    "relpath_under",
]
