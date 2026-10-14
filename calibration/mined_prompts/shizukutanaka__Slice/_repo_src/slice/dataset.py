"""Dataset export — Knowledge Store docs as CSV / JSONL.

The store keeps one Knowledge JSON per analyzed image. `dataset`
flattens a set of docs into tabular exports for spreadsheets,
notebooks and training pipelines:

- `to_csv(docs)`            one row per document (summary stats)
- `to_csv(docs, joints=True)` one row per joint (long format)
- `to_jsonl(docs)`          the raw docs, one per line
- `from_store(store)`       load every valid doc in a KnowledgeStore

Honesty rules carry through: joint rows come from
`skeleton.joints` — joints absent from a document are absent from
the export (no 0,0,0 ghost rows from the flat keypoint array), and
each row keeps `state`/`basis` so downstream consumers can still
separate observed evidence from prediction.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Iterable, List

from .knowledge import validate

SUMMARY_FIELDS = [
    "id", "created_at", "body_model", "source_name", "source_sha256",
    "frame_w", "frame_h", "n_joints", "n_observed", "n_predicted",
    "observed_ratio", "mean_confidence", "mean_observed_confidence",
]

JOINT_FIELDS = [
    "id", "joint", "x", "y", "confidence", "state", "basis",
]


def from_store(store) -> List[dict]:
    """Load every valid document a KnowledgeStore can read."""
    docs = []
    for entry in store.list():
        kid = entry.get("id")
        if not isinstance(kid, str):
            continue  # hand-dropped dict without an id
        try:
            doc = store.get(kid)
        except (KeyError, OSError, json.JSONDecodeError):
            continue
        # A file edited after save, or dropped in by hand, must not
        # leak into exports: only schema-valid documents count.
        if not isinstance(doc, dict) or validate(doc):
            continue
        docs.append(doc)
    return docs


def summary_rows(docs: Iterable[dict]) -> List[dict]:
    """One row per document."""
    rows = []
    for d in docs:
        joints = ((d.get("skeleton") or {}).get("joints")) or {}
        n_obs = sum(1 for j in joints.values()
                    if j.get("state") == "observed")
        confs = [j.get("confidence", 0.0) for j in joints.values()]
        obs_confs = [j.get("confidence", 0.0) for j in joints.values()
                     if j.get("state") == "observed"]
        frame = (d.get("skeleton") or {}).get("frame") or {}
        rows.append({
            "id": d.get("id", ""),
            "created_at": d.get("created_at", ""),
            "body_model": ((d.get("skeleton") or {})
                           .get("body_model") or {}).get("name", ""),
            "source_name": (d.get("source") or {}).get("name", ""),
            "source_sha256": (d.get("source") or {}).get("sha256", ""),
            "frame_w": frame.get("width", ""),
            "frame_h": frame.get("height", ""),
            "n_joints": len(joints),
            "n_observed": n_obs,
            "n_predicted": len(joints) - n_obs,
            "observed_ratio": (round(n_obs / len(joints), 4)
                               if joints else ""),
            "mean_confidence": (round(sum(confs) / len(confs), 4)
                                if confs else ""),
            # observed joints only — confidence averaged over
            # evidence, not blended with prior-derived joints
            "mean_observed_confidence":
                (round(sum(obs_confs) / len(obs_confs), 4)
                 if obs_confs else ""),
        })
    return rows


def joint_rows(docs: Iterable[dict]) -> List[dict]:
    """One row per joint per document (long format).

    Only joints present in the document appear — absent joints are
    not emitted as fake zero rows.
    """
    rows = []
    for d in docs:
        joints = ((d.get("skeleton") or {}).get("joints")) or {}
        for name, j in joints.items():
            rows.append({
                "id": d.get("id", ""),
                "joint": name,
                "x": j.get("x", ""),
                "y": j.get("y", ""),
                "confidence": j.get("confidence", ""),
                "state": j.get("state", ""),
                "basis": j.get("basis", ""),
            })
    return rows


def to_csv(docs: Iterable[dict], *, joints: bool = False) -> str:
    """CSV text with a fixed header; rows in document order."""
    rows = joint_rows(docs) if joints else summary_rows(docs)
    fields = JOINT_FIELDS if joints else SUMMARY_FIELDS
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def to_jsonl(docs: Iterable[dict]) -> str:
    """Raw Knowledge docs, one JSON object per line."""
    return "".join(json.dumps(d, ensure_ascii=False) + "\n"
                   for d in docs)
