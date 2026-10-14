"""Chunk-quality metrics + cross-batch duplicate clustering (pure core).

ML-readiness annotations over ``*.chunks.parquet``: per-chunk shape flags and
corpus-wide duplicate cluster ids. Annotation only — nothing here mutates chunk
text. The per-stage driver (:mod:`womblex.process.quality_stage`) applies these
across a shard directory and writes ``*.chunk_quality.parquet`` siblings.

Duplicate detection is deliberately self-contained (no datasketch dependency):
``exact_dup_id`` clusters chunks identical after whitespace/case/punctuation
normalisation; ``near_dup_id`` clusters near-duplicates via MinHash + LSH with
a fixed seed, so results are reproducible. Both are cross-batch, hence computed
in one global pass over all chunks rather than per batch.
"""

from __future__ import annotations

import collections
import hashlib
import re
import zlib

import numpy as np

_MH_PRIME = (1 << 61) - 1


def _stable_hash(s: str) -> int:
    """Deterministic 32-bit hash. Python's ``hash()`` is salted per process
    (PYTHONHASHSEED), which would make near-dup ids non-reproducible across
    runs — the opposite of what the stable-id contract promises."""
    return zlib.crc32(s.encode("utf-8"))


# ---- per-chunk shape flags --------------------------------------------------
def char_len(text: str) -> int:
    return len(text) if isinstance(text, str) else 0


def alpha_frac(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    return sum(c.isalpha() for c in text) / len(text)


def boilerplate_flag(text: str, patterns: list[re.Pattern]) -> bool:
    return isinstance(text, str) and any(p.search(text) for p in patterns)


def compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# ---- duplicate clustering (self-contained MinHash + LSH) --------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def _shingles(s: str, k: int) -> set[int]:
    toks = _norm(s).split()
    if len(toks) < k:
        return {_stable_hash(" ".join(toks))} if toks else set()
    return {_stable_hash(" ".join(toks[i:i + k])) for i in range(len(toks) - k + 1)}


def _signatures(texts: list[str], perms: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    a = rng.integers(1, _MH_PRIME, perms, dtype=np.uint64)
    b = rng.integers(0, _MH_PRIME, perms, dtype=np.uint64)
    sigs = np.full((len(texts), perms), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, s in enumerate(texts):
        sh = _shingles(s, k)
        if not sh:
            continue
        x = np.fromiter(sh, dtype=np.uint64)
        sigs[i] = ((a[:, None] * x[None, :] + b[:, None]) % _MH_PRIME).min(axis=1)
    return sigs


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def _cluster_ids(keys: list) -> list[int | None]:
    """Stable cluster ids (first-occurrence order); None for singletons."""
    groups: dict = collections.defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    ordered = sorted((m for m in groups.values() if len(m) > 1), key=lambda m: m[0])
    out: list[int | None] = [None] * len(keys)
    for cid, members in enumerate(ordered):
        for i in members:
            out[i] = cid
    return out


def exact_dup_ids(texts: list[str]) -> list[int | None]:
    # Content addressing for dup detection, not a security signature.
    # nosemgrep: python.lang.security.insecure-hash-algorithms-md5.insecure-hash-algorithm-md5
    return _cluster_ids(
        [hashlib.md5(_norm(t).encode(), usedforsecurity=False).hexdigest() for t in texts]
    )


def near_dup_ids(texts: list[str], perms: int, bands: int, k: int) -> list[int | None]:
    n = len(texts)
    if n == 0:
        return []
    sigs = _signatures(texts, perms, k)
    rows = perms // bands
    uf = _UnionFind(n)
    for band in range(bands):
        buckets: dict = collections.defaultdict(list)
        sub = sigs[:, band * rows:(band + 1) * rows]
        for i in range(n):
            buckets[sub[i].tobytes()].append(i)
        for members in buckets.values():
            for j in members[1:]:
                uf.union(members[0], j)
    return _cluster_ids([uf.find(i) for i in range(n)])


__all__ = [
    "alpha_frac", "boilerplate_flag", "char_len", "compile_patterns",
    "exact_dup_ids", "near_dup_ids",
]
