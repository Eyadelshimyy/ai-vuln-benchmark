"""KnowledgeStore audit — the doc-side of selfcheck.

`selfcheck` judges what comes out of one image; `audit_store`
judges what already lives in a store: per-document honesty lint
(`audit.audit`), near-duplicate detection (`dedup.dedup`), and an
observed-rate profile across the whole collection (which joints the
store's documents could actually see, and which they only ever
predicted).

Unreadable docs are reported, not swallowed: a store that fails to
load is part of the audit's findings, not a crash of the auditor.
"""

from __future__ import annotations

import os
from typing import Dict, List

from . import audit, dedup, knowledge
from .landmarks import JOINTS
from .skeleton import OBSERVED


def audit_store(root: str, eps: float = 0.15) -> Dict:
    """Audit every document in the store at `root`.

    Returns {n_docs, docs, duplicates, joint_observed, verdict,
    reasons} — `docs` maps id -> lint result, `joint_observed`
    maps joint name -> fraction of docs where it was observed.
    """
    store = knowledge.KnowledgeStore(root)
    ids = [i["id"] for i in store.list()]
    docs: List[dict] = []
    unreadable: List[dict] = []
    for kid in ids:
        try:
            docs.append(store.get(kid))
        except Exception as e:  # noqa: BLE001 - audit, not crash
            unreadable.append({"id": kid, "error": str(e)})
    # files the index never recorded are findings too, not invisible
    indexed = set(ids)
    for f in sorted(os.listdir(root)):
        if f.endswith(".json") and f != knowledge.INDEX_NAME \
                and f[:-5] not in indexed:
            unreadable.append({"id": f[:-5],
                               "error": "not in store index"})

    per: Dict[str, dict] = {}
    reasons: List[str] = ["unreadable:%s" % u["id"] for u in unreadable]
    for d in docs:
        r = audit.audit(d)
        per[d.get("id", "?")] = {
            "verdict": r["verdict"],
            "score": r["score"],
            "errors": r["errors"],
            "warnings": [w["code"] if isinstance(w, dict) else w
                         for w in r["warnings"]],
        }
        reasons += ["%s:error:%s" % (d.get("id"), e)
                    for e in r["errors"]]

    dup = dedup.dedup(docs, eps=eps)
    reasons += ["dedup:%s~%s" % (p["a"], p["b"]) for p in dup["pairs"]]

    # joint-level observed rate across the store
    joint_obs: Dict[str, float] = {}
    for name in JOINTS:
        obs = sum(
            1 for d in docs
            if (d.get("skeleton", {}).get("joints", {})
                .get(name, {}).get("state") == OBSERVED))
        if docs:
            joint_obs[name] = round(obs / len(docs), 3)
    blind = sorted(n for n, v in joint_obs.items() if v == 0)

    invalid = sum(1 for v in per.values() if v["verdict"] == "invalid")
    flagged = sum(1 for v in per.values() if v["verdict"] == "flagged")
    verdict = ("fail" if invalid or unreadable else
               "warn" if flagged or dup["pairs"] else
               "pass")

    return {
        "n_docs": len(docs),
        "docs": per,
        "unreadable": unreadable,
        "duplicates": dup,
        "joint_observed": joint_obs,
        "blind_joints": blind,
        "verdict": verdict,
        "reasons": reasons,
        "n_reasons": len(reasons),
        "state": "derived",
        "basis": "audit.audit per doc + dedup + joint observed-rate",
    }
