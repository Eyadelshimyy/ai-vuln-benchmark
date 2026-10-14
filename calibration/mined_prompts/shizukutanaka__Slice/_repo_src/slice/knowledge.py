"""Knowledge Engine: the artifact Slice actually keeps.

Images are disposable; Knowledge JSON is the durable product.
Schema `slice.knowledge/v1` records the skeleton (with per-joint
confidence and observed/predicted state), body ratios, the selected
statistical body model, and an OpenPose-style x,y,c keypoint array for
interoperability.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Optional

from .landmarks import BONES, JOINTS
from .skeleton import OBSERVED, PREDICTED, Skeleton

SCHEMA = "slice.knowledge/v1"
SCHEMA_V11 = "slice.knowledge/v1.1"
SCHEMAS = (SCHEMA, SCHEMA_V11)


def build(skel: Skeleton, ratios: dict, pose: Optional[dict] = None,
          *, image_sha256: str = "", source_name: str = "",
          engine: dict, analysis: Optional[dict] = None) -> dict:
    joints = skel.joints
    flat = []
    for name in JOINTS:
        j = joints.get(name)
        flat += [j.x, j.y, j.confidence] if j else [0.0, 0.0, 0.0]
    doc = {
        "schema": SCHEMA_V11 if analysis is not None else SCHEMA,
        "id": "k_" + secrets.token_hex(6),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": engine,
        "source": {
            "name": source_name,
            "sha256": image_sha256,
            "image_retained": False,
        },
        "skeleton": skel.to_dict(),
        "pose": pose or {},
        "ratio": ratios,
        "prediction": {
            "observed": [n for n, j in joints.items() if j.state == OBSERVED],
            "predicted": [n for n, j in joints.items()
                          if j.state == PREDICTED],
        },
        # how much of the document rests on image evidence — the
        # honesty contract as numbers, not just colors
        "coverage": {
            "joints_total": len(JOINTS),
            "observed": sum(1 for j in joints.values()
                            if j.state == OBSERVED),
            "predicted": sum(1 for j in joints.values()
                             if j.state == PREDICTED),
            "unfilled": len(JOINTS) - len(joints),
            "observed_ratio": round(
                sum(1 for j in joints.values() if j.state == OBSERVED)
                / len(JOINTS), 3),
            "mean_observed_confidence": round(
                sum(j.confidence for j in joints.values()
                    if j.state == OBSERVED)
                / max(1, sum(1 for j in joints.values()
                             if j.state == OBSERVED)), 3),
        },
        "export": {
            "keypoints_2d": flat,
            "keypoint_order": JOINTS,
            # state per keypoint, same order — the flat array alone
            # can't tell prior fill from evidence
            "keypoints_state": [
                joints[n].state if n in joints else "absent"
                for n in JOINTS],
            "bones": [list(b) for b in BONES],
        },
    }
    if analysis is not None:
        doc["analysis"] = analysis
    return doc


def validate(doc: dict) -> list:
    """Return a list of schema problems; empty means valid."""
    errors = []
    if doc.get("schema") not in SCHEMAS:
        errors.append(f"schema must be one of {SCHEMAS}")
    for key in ("id", "created_at", "engine", "skeleton", "export"):
        if key not in doc:
            errors.append(f"missing {key}")
    # v1.1 extension slot: `analysis` is a dict of named layer outputs;
    # each layer is a free-form dict (its own state/basis vocabulary).
    # Tolerated on v1 too — docs written before the version bump stay
    # readable; new documents carrying analysis are built as v1.1.
    analysis = doc.get("analysis")
    if analysis is not None and not (
            isinstance(analysis, dict) and all(
                isinstance(v, dict) for v in analysis.values())):
        errors.append("analysis must be a dict of layer dicts")
    joints = (doc.get("skeleton") or {}).get("joints") or {}
    for name, j in joints.items():
        for f in ("x", "y", "confidence", "state"):
            if f not in j:
                errors.append(f"joint {name} missing {f}")
        if j.get("state") not in (OBSERVED, PREDICTED):
            errors.append(f"joint {name} bad state {j.get('state')}")
        c = j.get("confidence")
        if not isinstance(c, (int, float)) or not 0 <= c <= 1:
            errors.append(f"joint {name} bad confidence {c}")
    pred = doc.get("prediction")
    if isinstance(pred, dict):
        # the observed/predicted lists are the document's evidence
        # claim — they must not contradict the joints' own states,
        # or fill could be listed as sighted evidence
        for n in pred.get("observed") or []:
            j = joints.get(n)
            if not isinstance(j, dict):
                errors.append(
                    f"prediction.observed {n} not in skeleton.joints")
            elif j.get("state") != OBSERVED:
                errors.append(
                    f"prediction.observed {n} is {j.get('state')}"
                    " in skeleton.joints")
        for n in pred.get("predicted") or []:
            j = joints.get(n)
            if not isinstance(j, dict):
                errors.append(
                    f"prediction.predicted {n} not in skeleton.joints")
            elif j.get("state") != PREDICTED:
                errors.append(
                    f"prediction.predicted {n} is {j.get('state')}"
                    " in skeleton.joints")
    skel = doc.get("skeleton") or {}
    frame = skel.get("frame")
    if isinstance(frame, dict):
        for k in ("width", "height"):
            v = frame.get(k)
            if not isinstance(v, (int, float)) or v <= 0:
                errors.append(f"skeleton.frame.{k} must be > 0")
    norm = skel.get("normalized")
    if isinstance(norm, dict):
        # invariants of the normalized space: pelvis at the origin,
        # neck one torso-unit away — a stale/rewritten block breaks
        # these even when its numbers look plausible
        nj = norm.get("joints") or {}
        pv, nk = nj.get("pelvis"), nj.get("neck")
        if isinstance(pv, dict) and (
                abs(pv.get("x", 1)) > 0.01
                or abs(pv.get("y", 1)) > 0.01):
            errors.append("normalized pelvis not at origin")
        if isinstance(nk, dict):
            d = (nk.get("x", 0) ** 2 + nk.get("y", 0) ** 2) ** 0.5
            if abs(d - 1.0) > 0.01:
                errors.append("normalized neck not one unit away")
    export = doc.get("export") or {}
    order = export.get("keypoint_order")
    if order is not None:
        flat = export.get("keypoints_2d")
        if not isinstance(flat, list) \
                or len(flat) != 3 * len(order):
            errors.append(
                "export.keypoints_2d must be 3*len(keypoint_order)")
        states = export.get("keypoints_state")
        if states is not None:
            # a flat-export state array that contradicts skeleton.joints
            # would let fill masquerade as evidence downstream
            if len(states) != len(order):
                errors.append(
                    "export.keypoints_state must align with "
                    "keypoint_order")
            else:
                for name, st in zip(order, states):
                    j = joints.get(name)
                    expect = j.get("state", "absent") \
                        if isinstance(j, dict) else "absent"
                    if st != expect:
                        errors.append(
                            f"export state {name}: {st} != "
                            f"skeleton {expect}")
    return errors


INDEX_NAME = "_index.json"


def _list_entry(doc: dict) -> dict:
    skel = doc.get("skeleton")
    bm = skel.get("body_model") if isinstance(skel, dict) else None
    return {"id": doc.get("id"),
            "created_at": doc.get("created_at"),
            "body_model": bm.get("name") if isinstance(bm, dict) else None}


class KnowledgeStore:
    """File-backed store: one <id>.json per analysis, with a cached
    listing manifest so `list()` does not have to open every document."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def save(self, doc: dict) -> str:
        errors = validate(doc)
        if errors:
            raise ValueError("invalid knowledge: " + "; ".join(errors))
        path = os.path.join(self.root, doc["id"] + ".json")
        # atomic write: a crash mid-save must never leave a torn JSON
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        self._index_add(doc)
        return doc["id"]

    def get(self, kid: str) -> dict:
        if not re.fullmatch(r"k_[0-9a-f]{12}", kid):
            raise KeyError(kid)
        path = os.path.join(self.root, kid + ".json")
        if not os.path.isfile(path):
            raise KeyError(kid)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        # a file named <kid>.json must be the document <kid> — a
        # mismatched internal id means corruption, not the doc asked for
        if not isinstance(doc, dict) or doc.get("id") != kid:
            raise KeyError(kid)
        return doc

    def _index_path(self) -> str:
        return os.path.join(self.root, INDEX_NAME)

    def _index_load(self) -> Optional[dict]:
        """Load the cached index -> {kid: entry}, or None if unusable."""
        try:
            with open(self._index_path(), encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(d, dict) or not isinstance(d.get("entries"), dict):
            return None
        return d["entries"]

    def _index_add(self, doc: dict) -> None:
        """Best-effort index update; a broken index self-heals in list()."""
        entries = self._index_load()
        if entries is None:
            entries = {}
        entries[doc["id"]] = _list_entry(doc)
        try:
            with open(self._index_path(), "w", encoding="utf-8") as f:
                json.dump({"schema": "slice.index/v1", "entries": entries},
                          f, ensure_ascii=False)
        except OSError:
            pass

    def list(self) -> list:
        files = [fn for fn in sorted(os.listdir(self.root))
                 if fn.endswith(".json") and fn != INDEX_NAME]
        indexed = self._index_load()
        out, healed = [], indexed is None
        entries = dict(indexed) if indexed else {}
        index_mtime = 0.0
        if indexed is not None:
            try:
                index_mtime = os.path.getmtime(self._index_path())
            except OSError:
                index_mtime = 0.0
        for fn in files:
            kid = fn[:-5]
            ent = entries.pop(kid, None)
            if ent is not None and ent.get("id") != kid:
                # a cached entry whose id disagrees with the filename
                # names a document that can never be retrieved — drop
                # it and let the file itself be judged below
                healed = True
                ent = None
            if ent is not None:
                # a file rewritten after the index was built leaves a
                # stale entry — re-read it so list() never reports a
                # document that no longer exists on disk
                try:
                    if os.path.getmtime(
                            os.path.join(self.root, fn)) < index_mtime:
                        out.append(ent)
                        continue
                except OSError:
                    pass
                ent = None
            if ent is None:
                healed = True
                try:
                    with open(os.path.join(self.root, fn),
                              encoding="utf-8") as f:
                        d = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(d, dict) or d.get("id") != kid:
                    # filename is the document's identity — a file that
                    # claims another id is corruption, not a listable doc
                    continue
                ent = _list_entry(d)
            out.append(ent)
        if healed or entries:  # index was missing/stale -> rewrite it
            known = {e.get("id"): e for e in out if e.get("id")}
            try:
                with open(self._index_path(), "w", encoding="utf-8") as f:
                    json.dump({"schema": "slice.index/v1",
                               "entries": known}, f, ensure_ascii=False)
            except OSError:
                pass
        return out


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
