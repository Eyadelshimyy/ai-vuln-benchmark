"""Object-storage staging for distributed runs (S3 / MinIO / GCS / local).

The cloud worker model is **stage-in / stage-out**: a worker pulls a job's
input documents from object storage to a local scratch dir, runs the ordinary
``Path``-based pipeline, then pushes the resulting ``batch-NNNN.*.parquet``
shards back. This keeps every extraction/stage module untouched — they still
see plain local paths — and confines all remote-storage knowledge to this one
adapter, exactly the way ``store/`` sidecar modules are self-contained.

fsspec gives one API across ``s3://`` (MinIO is S3-compatible), ``gs://``,
``az://`` and the local filesystem, so the air-gapped / CPU-first default is
free: a ``file://`` or bare local path uses the same code path with no S3
endpoint touched. fsspec + s3fs are core dependencies (see ``pyproject.toml``),
so ``s3://`` works on any install — they stay dormant until a remote URI names
them.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from womblex.utils.checksum import sha256_file

logger = logging.getLogger(__name__)

_REMOTE_PROTOCOLS = ("s3://", "gs://", "gcs://", "az://", "abfs://")


def is_remote_uri(uri: str) -> bool:
    """True when *uri* names an object store (not a local/``file://`` path)."""
    return uri.startswith(_REMOTE_PROTOCOLS)


def storage_options_from_env(uri: str, *, credentials: tuple[str, str] | None = None) -> dict:
    """Build fsspec storage options for *uri* from env vars.

    Only ``s3://`` gets explicit options: an S3 access key/secret plus an
    endpoint override (``WOMBLEX_S3_ENDPOINT`` or ``AWS_ENDPOINT_URL``) so
    MinIO and other S3-compatible stores work without code changes. Every
    other backend — ``gs://``, ``az://``, local — returns an empty dict and
    authenticates via its own native mechanism (gcsfs/adlfs ambient
    credentials); these kwargs are s3fs-shaped and would misconfigure them.

    *credentials*, when given as ``(key, secret)``, is an explicit override
    that wins over the environment — the path an operator-saved credential
    (Resources Console) takes so a rotated key is used *moving forward*
    without a container rebuild, even though the Dockerfile baked an older one
    into the env. When ``None``, the store's credentials are read from the
    **store-specific** ``WOMBLEX_S3_ACCESS_KEY_ID`` /
    ``WOMBLEX_S3_SECRET_ACCESS_KEY`` first, falling back to the AWS standard
    ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` for back-compatibility.
    The store-specific pair exists so a deployment can point s3fs at MinIO (a
    static key) **without** setting the ambient ``AWS_ACCESS_KEY_ID`` — which
    is process-global and, when set, disables boto3's instance-role resolution
    for *every* client in the process, including the ``isaacus-sagemaker``
    SigV4 signer (it would then sign MinIO's key against real SageMaker and
    403). Leaving ``AWS_ACCESS_KEY_ID`` unset lets that signer reach the EC2
    instance role, while s3fs still gets MinIO's key explicitly here.
    """
    if not uri.startswith("s3://"):
        return {}
    endpoint = os.environ.get("WOMBLEX_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
    opts: dict = {}
    key: str | None
    secret: str | None
    if credentials is not None:
        key, secret = credentials
    else:
        key = os.environ.get("WOMBLEX_S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
        secret = (
            os.environ.get("WOMBLEX_S3_SECRET_ACCESS_KEY")
            or os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
    if key and secret:
        opts["key"] = key
        opts["secret"] = secret
    if endpoint:
        opts["client_kwargs"] = {"endpoint_url": endpoint}
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        opts.setdefault("client_kwargs", {})["region_name"] = region
    return opts


#: Protocols :class:`RemoteStore` can open. A URI with any other scheme is
#: rejected before fsspec sees it — ``url_to_fs`` would otherwise try to
#: resolve the host of e.g. ``ftp://`` during what is meant to be validation.
SUPPORTED_PROTOCOLS = ("file", "s3", "s3a", "gs", "gcs", "az", "abfs")


def validate_location_uri(uri: str) -> None:
    """Raise ``ValueError`` unless *uri* names a location a store can open.

    Catches the typos a hand-typed location actually produces — ``s3:/bucket``
    (one slash), ``S3://bucket``, a bare ``s3://`` with no bucket, an
    unsupported scheme. Without this, ``s3:/bucket`` parses as a *relative
    local path* and documents land in a folder literally named ``s3:``.
    """
    if not uri or not uri.strip():
        raise ValueError("location is empty")
    scheme, sep, rest = uri.partition("://")
    if not sep:
        head = uri.split("/", 1)[0]
        if ":" in head:
            raise ValueError(
                f"{uri!r} is not a usable location — a URI needs '://' "
                f"(did you mean {head.rstrip(':')}://…?)"
            )
        return  # a plain local path
    if scheme != scheme.lower():
        raise ValueError(f"{uri!r} is not a usable location — the scheme must be lowercase")
    if scheme not in SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"{uri!r} is not a usable location — scheme {scheme!r} is not one of "
            f"{', '.join(SUPPORTED_PROTOCOLS)}"
        )
    if scheme != "file" and not rest.strip("/"):
        raise ValueError(f"{uri!r} names no bucket")


def store_root(uri: str) -> tuple[str, str]:
    """Normalise *uri* to ``(bucket_or_mount, prefix)`` for containment checks.

    Parsed with the same ``url_to_fs`` call and storage options
    :meth:`RemoteStore.from_uri` uses, so it agrees with what actually gets
    opened. Object-store URIs split the bucket out so two different buckets
    never compare as overlapping; local paths have no bucket, so the first
    element is ``""``.
    """
    validate_location_uri(uri)
    fsspec = _require_fsspec()
    fs, path = fsspec.core.url_to_fs(uri, **storage_options_from_env(uri))
    # Collapse empty segments so a doubled slash compares as the typo it is,
    # and so this agrees with `_path_contains`, which drops them too.
    path = "/".join(part for part in path.split("/") if part)
    protocol = fs.protocol[0] if isinstance(fs.protocol, (list, tuple)) else fs.protocol
    if protocol in ("s3", "s3a", "gs", "gcs", "az", "abfs"):
        bucket, _, prefix = path.partition("/")
        return bucket, prefix
    return "", path


def _path_contains(parent: str, child: str) -> bool:
    """True when path-segment sequence *parent* is *child* or an ancestor of it."""
    parent_parts = [p for p in parent.split("/") if p]
    child_parts = [p for p in child.split("/") if p]
    return child_parts[: len(parent_parts)] == parent_parts


def same_location(a: str, b: str) -> bool:
    """True when two URIs name the same bucket and prefix.

    Spelling differences that do not change what gets opened — a trailing
    slash, a redundant ``//`` — compare equal, which a raw string comparison
    of two independently-configured values would not.
    """
    return store_root(a) == store_root(b)


def assert_disjoint_locations(
    ingest_uri: str, store_uri: str, *, runs_prefix: str = "runs",
) -> None:
    """Raise ``ValueError`` unless *ingest_uri* and the store's effective output
    (``<store_uri>/<runs_prefix>``) live on disjoint paths.

    Same bucket, different folders is the normal case. Either location
    containing the other means raw documents and processed shards would
    accumulate in one folder. Comparison is by path *segment*, so
    ``s3://b/run`` and ``s3://b/runs`` are disjoint — matching how the
    delimiter-based listing in :meth:`RemoteStore.list_files` actually walks
    an object store, not how a raw ``ListObjectsV2`` prefix would.
    """
    output_uri = f"{store_uri.rstrip('/')}/{runs_prefix.strip('/')}"
    ingest_bucket, ingest_path = store_root(ingest_uri)
    output_bucket, output_path = store_root(output_uri)
    if ingest_bucket != output_bucket:
        return
    if _path_contains(ingest_path, output_path) or _path_contains(output_path, ingest_path):
        raise ValueError(
            f"Ingest location {ingest_uri!r} and output location {output_uri!r} "
            "are not disjoint (one contains the other) — documents and shards "
            "must live under separate prefixes."
        )


def _require_fsspec():  # type: ignore[no-untyped-def]
    # fsspec is a core dependency, so this is a plain import now. Kept as a
    # named helper (rather than inlined) so `from_uri` reads unchanged and any
    # future backend-availability check has one place to live.
    import fsspec

    return fsspec


@dataclass
class RemoteStore:
    """Thin fsspec wrapper rooted at a base URI.

    All paths passed to methods are *relative* to the store root; the store
    joins them. Build one with :meth:`from_uri`.
    """

    fs: object  # fsspec.AbstractFileSystem
    root: str

    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        storage_options: dict | None = None,
        credentials: tuple[str, str] | None = None,
    ) -> RemoteStore:
        """Open a store at *uri* (e.g. ``s3://bucket/runs`` or ``/data/runs``).

        *credentials* is an explicit ``(key, secret)`` S3 override that wins
        over the environment (see :func:`storage_options_from_env`) — the path
        an operator-saved credential takes. Ignored when *storage_options* is
        passed (the caller has already built the full options dict).
        """
        fsspec = _require_fsspec()
        opts = (
            storage_options
            if storage_options is not None
            else storage_options_from_env(uri, credentials=credentials)
        )
        fs, root = fsspec.core.url_to_fs(uri, **opts)
        return cls(fs=fs, root=root.rstrip("/"))

    def _full(self, rel: str) -> str:
        rel = rel.strip("/")
        return f"{self.root}/{rel}" if rel else self.root

    def exists(self, rel: str) -> bool:
        return bool(self.fs.exists(self._full(rel)))  # type: ignore[attr-defined]

    def read_parquet_footer(self, rel: str) -> tuple[dict[bytes, bytes] | None, int] | None:
        """A parquet's footer metadata and row count, read in place.

        Parquet keeps both in a footer, so this is a couple of ranged reads
        rather than a download — which is what makes it affordable to ask it of
        every file in a run at finalisation. ``None`` when the object cannot be
        read as parquet; the caller is building a record and a file it cannot
        read is one it reports as absent, not one it fails over.
        """
        import pyarrow.parquet as pq

        try:
            with self.fs.open(self._full(rel), "rb") as handle:  # type: ignore[attr-defined]
                meta = pq.read_metadata(handle)
        except Exception as exc:  # any backend or format error: report, never fail
            logger.warning("could not read parquet footer for %s: %s", rel, exc)
            return None
        return meta.metadata, meta.num_rows

    def read_text(self, rel: str, *, encoding: str = "utf-8") -> str:
        """Read a small text object in place (no staging) — e.g. a saved preset."""
        with self.fs.open(self._full(rel), "r", encoding=encoding) as handle:  # type: ignore[attr-defined]
            return cast(str, handle.read())

    def delete(self, rel: str) -> None:
        """Remove one object. Assumes it exists; callers check first if that matters."""
        self.fs.rm_file(self._full(rel))  # type: ignore[attr-defined]

    def list_files(self, rel: str, pattern: str = "*", *, recursive: bool = False) -> list[str]:
        """List paths under *rel* matching *pattern*, returned store-relative.

        ``recursive`` walks nested prefixes as well as the immediate level.
        Object stores have a flat keyspace — a document uploaded as
        ``inbox/2026-08/foo.pdf`` is one key, not a folder — so any caller
        enumerating *documents* (rather than a known sibling set, like a
        batch's parquet shards) needs it.
        """
        full = self._full(rel)
        glob = f"{full}/**/{pattern}" if recursive else f"{full}/{pattern}"
        matches: list[str] = self.fs.glob(glob)  # type: ignore[attr-defined]
        prefix = self.root + "/"
        return [m.removeprefix(prefix) for m in matches if m != self.root]

    def find(self, rel: str) -> list[str]:
        """Every object key under *rel*, recursively and store-relative — no pseudo-directories."""
        prefix = self.root + "/"
        found: list[str] = self.fs.find(self._full(rel))  # type: ignore[attr-defined]
        return [m.removeprefix(prefix) for m in found]

    def list_dirs(self, rel: str) -> list[str]:
        """List immediate child directory names under *rel* (name only, not full path).

        Object stores have no real directories, but fsspec surfaces the
        common-prefix convention (keys with more path segments beneath *rel*)
        as pseudo-directory entries. That is what lets a caller enumerate
        ``runs/<run_id>/`` without knowing the run ids ahead of time.
        """
        full = self._full(rel)
        if not self.fs.exists(full):  # type: ignore[attr-defined]
            return []
        entries: list[dict] = self.fs.ls(full, detail=True)  # type: ignore[attr-defined]
        return sorted(
            e["name"].rstrip("/").rsplit("/", 1)[-1]
            for e in entries
            if e.get("type") == "directory"
        )

    def download_file(self, rel: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.fs.get_file(self._full(rel), str(local_path))  # type: ignore[attr-defined]
        return local_path

    def upload_file(self, local_path: Path, rel: str) -> str:
        full = self._full(rel)
        parent = full.rsplit("/", 1)[0]
        self.fs.makedirs(parent, exist_ok=True)  # type: ignore[attr-defined]
        self.fs.put_file(str(local_path), full)  # type: ignore[attr-defined]
        return rel

    def write_stream(self, fileobj: BinaryIO, rel: str) -> str:
        """Copy *fileobj* to *rel* without staging it on local disk first."""
        full = self._full(rel)
        self.fs.makedirs(full.rsplit("/", 1)[0], exist_ok=True)  # type: ignore[attr-defined]
        with self.fs.open(full, "wb") as out:  # type: ignore[attr-defined]
            shutil.copyfileobj(fileobj, out)
        return rel

    def move(self, src_rel: str, dst_rel: str) -> str:
        """Move *src_rel* onto *dst_rel* within the store, replacing any existing object.

        Server-side where the backend allows it (S3 ``mv`` is copy-then-delete
        against the same bucket; the local filesystem is a rename). The single
        point of a temp-key-then-move commit: the destination key flips from its
        old bytes to the new ones in one backend operation, so a reader never
        sees a half-written object and a crash cannot leave the destination
        truncated. Overwrites the destination — this is the commit step of an
        in-place rewrite, not a create-if-absent.
        """
        src = self._full(src_rel)
        dst = self._full(dst_rel)
        parent = dst.rsplit("/", 1)[0]
        self.fs.makedirs(parent, exist_ok=True)  # type: ignore[attr-defined]
        self.fs.mv(src, dst)  # type: ignore[attr-defined]
        return dst_rel

    def download_to_dir(
        self, rels: list[str], local_dir: Path, *, nested: bool = False
    ) -> list[Path]:
        """Fetch each store-relative key in *rels* into *local_dir*.

        Flat by default: the key's basename is the local name. That is what a
        caller staging one prefix's siblings wants — a batch's shards, a
        checkpoint dir — because the local dir is then exactly the flat
        directory the ``*_shards()`` entry points and the manifest readers
        expect, and sibling keys cannot collide.

        ``nested=True`` mirrors each key's prefix under *local_dir* instead.
        Documents are the case that needs it: an ingest prefix is enumerated
        recursively, so ``2026-07/report.pdf`` and ``2026-08/report.pdf`` are
        two documents that flat staging would land on one local file — the
        second winning, one document extracted twice, one not at all, and the
        recorded source path the survivor's. A key is canonicalised and asserted
        to stay under *local_dir* before anything is written to it.
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        targets = [
            self._nested_target(local_dir, rel) if nested else local_dir / rel.rsplit("/", 1)[-1]
            for rel in rels
        ]
        shared = sorted({t.name for t in targets if targets.count(t) > 1})
        if shared:
            logger.warning(
                "Staging %d key(s) onto a local path another key already claims "
                "(%s); each such document is overwritten by its namesake",
                len(targets) - len(set(targets)), ", ".join(shared),
            )
        out: list[Path] = []
        for rel, target in zip(rels, targets, strict=True):
            out.append(self.download_file(rel, target))
        logger.info("Staged %d input file(s) -> %s", len(out), local_dir)
        return out

    @staticmethod
    def _nested_target(local_dir: Path, rel: str) -> Path:
        """Local path for *rel* under *local_dir*, with its prefix preserved.

        A store key is whatever the queue row or a listing named, so it is
        canonicalised and checked against the staging directory rather than
        trusted: ``../`` segments in a key would otherwise write outside the
        scratch dir the caller created.
        """
        base = local_dir.resolve()
        target = (base / rel.strip("/")).resolve()
        if target == base or not target.is_relative_to(base):
            raise ValueError(f"store key {rel!r} does not stage under {local_dir}")
        return target

    def upload_glob(self, local_dir: Path, glob: str, remote_rel_dir: str) -> dict[str, str]:
        """Push every file in *local_dir* matching *glob* under *remote_rel_dir*.

        Returns each published key with the SHA-256 of the bytes uploaded,
        hashed from the local file before it leaves.
        """
        uploaded: dict[str, str] = {}
        for p in sorted(local_dir.glob(glob)):
            if p.is_file():
                digest = sha256_file(p)
                uploaded[self.upload_file(p, f"{remote_rel_dir.strip('/')}/{p.name}")] = digest
        logger.info("Published %d shard file(s) -> %s/%s", len(uploaded), self.root, remote_rel_dir)
        return uploaded


__all__ = [
    "SUPPORTED_PROTOCOLS",
    "RemoteStore",
    "assert_disjoint_locations",
    "is_remote_uri",
    "same_location",
    "storage_options_from_env",
    "store_root",
    "validate_location_uri",
]
