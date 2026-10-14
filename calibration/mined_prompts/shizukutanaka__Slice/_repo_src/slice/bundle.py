"""Dataset bundling — a KnowledgeStore as one portable archive.

`dataset` flattens docs into tables; `bundle` packages them for
transport: a single `.zip` containing every schema-valid document
plus a `manifest.json` recording ids, models and counts, so a
receiver knows what's inside without unzipping everything.

Only valid documents ship — the same `validate` gate the store
enforces on save is re-applied on export, and skipped files are
counted in the manifest rather than silently dropped.
"""

from __future__ import annotations

import json
import zipfile
import zlib
from typing import List

from .knowledge import validate

SCHEMA = "slice.bundle/v1"


def _body_model_name(d: dict):
    sk = d.get("skeleton")
    bm = sk.get("body_model") if isinstance(sk, dict) else None
    return bm.get("name") if isinstance(bm, dict) else None


def pack(store, path: str) -> dict:
    """Write every valid doc in `store` to a zip at `path`.

    Returns the manifest dict that was embedded.
    """
    docs, skipped = [], 0
    for entry in store.list():
        kid = entry.get("id")
        if not isinstance(kid, str):
            skipped += 1
            continue
        try:
            doc = store.get(kid)
        except (KeyError, OSError, json.JSONDecodeError):
            skipped += 1
            continue
        # validate raises on malformed internals — an unprocessable
        # doc counts as skipped, it must not kill the pack
        try:
            bad = not isinstance(doc, dict) or bool(validate(doc))
        except Exception:
            bad = True
        if bad:
            skipped += 1
            continue
        # validate does not type-check body_model — a malformed one
        # would ship and crash downstream consumers
        skel = doc.get("skeleton")
        bm = skel.get("body_model") if isinstance(skel, dict) else None
        if bm is not None and not isinstance(bm, dict):
            skipped += 1
            continue
        docs.append(doc)
    manifest = {
        "schema": SCHEMA,
        "count": len(docs),
        "skipped": skipped,
        "documents": [
            {"id": d["id"],
             "created_at": d.get("created_at"),
             "body_model": _body_model_name(d)}
            for d in docs
        ],
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",
                   json.dumps(manifest, ensure_ascii=False,
                              indent=2))
        for d in docs:
            z.writestr("docs/%s.json" % d["id"],
                       json.dumps(d, ensure_ascii=False))
    return manifest


def manifest(path: str) -> dict:
    """Read just the manifest from a bundle."""
    with zipfile.ZipFile(path) as z:
        m = json.loads(z.read("manifest.json"))
    if not isinstance(m, dict):
        raise ValueError("manifest.json is not an object")
    return m


def unpack(path: str) -> List[dict]:
    """Load all documents from a bundle back into memory."""
    docs = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.startswith("docs/"):
                continue
            try:
                doc = json.loads(z.read(name))
            except (json.JSONDecodeError, UnicodeDecodeError,
                    zipfile.BadZipFile, zlib.error):
                continue  # one corrupt member must not kill the archive
            # validate raises on malformed internals — an unprocessable
            # member is skipped, it must not kill the archive
            try:
                bad = not isinstance(doc, dict) or bool(validate(doc))
            except Exception:
                bad = True
            if bad:
                continue
            docs.append(doc)
    return docs
