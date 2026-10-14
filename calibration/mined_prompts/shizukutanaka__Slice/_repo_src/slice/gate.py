"""Unified quality gate — one call, three layers, one verdict.

Quality checks are scattered: `consistency.audit` lints the skeleton's
geometry, `audit.audit` lints the knowledge document's honesty,
and an empty `Skeleton` itself is a detection outcome. A caller
wanting "is this analysis good enough to keep?" had to run all of
those and synthesize. `check` does that: it runs whichever layers
the caller supplies (skeleton / document / both) and folds the
results into one verdict.

Verdict ladder — `fail` (hard error: no detection, invalid document,
structural violation), `warn` (analysis produced output but flags
exist), `pass` (nothing to flag). The layers' own finding strings
are passed through verbatim in `reasons` — never reworded, so the
gate adds no new vocabulary.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import audit as doc_audit
from . import consistency
from .skeleton import OBSERVED, Skeleton


def check(skel: Optional[Skeleton] = None,
          doc: Optional[dict] = None,
          model: Optional[str] = None) -> Dict:
    """Run every quality layer the inputs support; return a verdict.

    `skel` — skeleton to lint with `consistency.audit`; also supplies
    the detection outcome (empty skeleton = no person found).
    `doc`  — knowledge document to lint with `audit.audit`.
    Returns {verdict, reasons, layers} where layers keeps each
    checker's raw output.
    """
    layers: Dict[str, dict] = {}
    reasons: List[str] = []

    if skel is not None:
        joints = list(skel.joints.values())
        detected = bool(joints)
        observed = sum(1 for j in joints if j.state == OBSERVED)
        issues = consistency.audit(skel, model) if detected else []
        layers["detection"] = {
            "found": detected,
            "joints": len(joints),
            "observed": observed,
        }
        layers["consistency"] = {"issues": issues}
        if not detected:
            reasons.append("detection:no_person")
        reasons += ["consistency:" + i for i in issues]

    if doc is not None:
        doc_res = doc_audit.audit(doc)
        layers["document"] = doc_res
        reasons += ["document:error:" + e for e in doc_res["errors"]]
        reasons += ["document:warn:" + w["code"]
                    if isinstance(w, dict) else "document:warn:" + w
                    for w in doc_res["warnings"]]

    hard = [r for r in reasons
            if r.startswith("detection:") or ":error:" in r
            or r.startswith("consistency:no_body")
            or r.startswith("consistency:inverted")]
    verdict = ("fail" if hard else
               "warn" if reasons else
               "pass")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "n_reasons": len(reasons),
        "layers": layers,
        "state": "derived",
        "basis": "consistency + document-audit + detection outcome",
    }


def keep(result: Dict) -> bool:
    """Is this analysis worth storing? pass or warn = yes."""
    return result["verdict"] != "fail"
