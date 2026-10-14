"""One-shot self-audit — every merged quality layer on one image.

The audit stack grew as separate library modules (imgqual, human,
evid, limbcov, fit, stability, contrad, gate). `run` drives them all
over a single decode and folds their per-layer verdicts into one
overall verdict with the reason codes preserved:

    input   imgqual   adequate | marginal | inadequate
    subject human     person_like (advisory, never a veto)
    skeleton evid     off_mask observed joints
            limbcov   covered | gaps | insufficient
            fit       good | poor | unmeasurable
            stability stable_fraction over threshold probes
    layers  contrad   consistent | contradicted
    doc     gate      pass | warn | fail

Layers that cannot measure (no skeleton, no foreground) report their
own `unmeasurable`/`insufficient` state and do not move the verdict —
absence of evidence is not downgraded as if it were bad evidence.
"""

from __future__ import annotations

from typing import Dict, Optional

from . import (anatomy, axis, balance, bitmap, classify, consistency,
               contrad, evid, fit, gate, ground, human, imgqual,
               knowledge, limbcov, mask as mask_mod, pipeline,
               stability)

_SEV = {"ok": 0, "advisory": 1, "problem": 2, "unmeasured": -1}

# verdict/state strings that legitimately mean "nothing to measure"
_ABSENT = ("unmeasurable", "insufficient")


def _unmapped(result: Dict) -> str:
    """Severity for a verdict outside the layer's known vocabulary.

    Absence of evidence ("unmeasurable"/"insufficient") stays
    unmeasured; *unrecognized* evidence is not absence — map it to
    advisory so a new verdict a layer starts emitting cannot be
    silently neutralized into "no evidence"."""
    v = result.get("verdict") or result.get("state")
    return "unmeasured" if v in _ABSENT else "advisory"


def _severity(layer: str, result: Dict) -> str:
    """Map one layer's own verdict vocabulary onto a common scale."""
    if layer == "imgqual":
        return {"adequate": "ok", "marginal": "advisory",
                "inadequate": "problem"}.get(result["verdict"],
                                             "unmeasured")
    if layer == "human":
        if result["state"] == "unmeasurable":
            return "unmeasured"
        return "ok" if result["person_like"] else "advisory"
    if layer == "evid":
        if result["state"] == "unmeasurable":
            return "unmeasured"
        return "advisory" if result["off_mask"] else "ok"
    if layer == "limbcov":
        return {"covered": "ok", "gaps": "advisory"}.get(
            result["verdict"], _unmapped(result))
    if layer == "fit":
        return {"good": "ok", "poor": "advisory"}.get(
            fit.verdict(result), _unmapped(result))
    if layer == "stability":
        if result["state"] == "unmeasurable":
            return "unmeasured"
        return "advisory" if stability.unstable(result) else "ok"
    if layer == "contrad":
        return {"consistent": "ok", "contradicted": "problem"}.get(
            result["verdict"], _unmapped(result))
    if layer == "consistency":
        # anatomical-prior violations are advisory, not fail: real
        # bodies legitimately exceed population bounds
        return {"consistent": "ok", "issues": "advisory"}.get(
            result["verdict"], _unmapped(result))
    if layer == "gate":
        return {"pass": "ok", "warn": "advisory",
                "fail": "problem"}.get(result["verdict"],
                                        _unmapped(result))
    return _unmapped(result)


def run(raw: bytes, *, model: Optional[str] = None,
        source_name: str = "", robust: bool = False) -> Dict:
    """Audit one image; return {verdict, reasons, layers, doc}.

    `doc` is the ordinary Knowledge document produced alongside the
    audit — the audit judges it, it is not the product itself.
    """
    bmp = bitmap.decode(raw)
    est = pipeline.ROBUST_ESTIMATOR if robust else pipeline.ESTIMATOR
    mdl = model or "adult"
    layers: Dict[str, dict] = {}
    reasons = []

    # an unknown model name silently estimates against the default
    # prior — the audit must say the requested model never ran
    if model and model not in anatomy.BODY_MODELS:
        reasons.append("model:unknown_body_model")

    layers["imgqual"] = imgqual.assess(bmp)
    if not imgqual.adequate(layers["imgqual"]):
        reasons.append("imgqual:" + layers["imgqual"]["verdict"])

    skel = est.estimate(bmp, mdl)
    # joint-based layers must compare at the estimator's own
    # resolution — skeleton coords live in downscaled space, so a
    # full-resolution mask would flag every joint as off-mask.
    small = bmp.downscale(est.max_dim)
    mask = mask_mod.foreground(small, est)

    if mask_mod.coverage(mask) > 0:
        labels, sizes = est._label_components(
            mask, small.width, small.height)
        best = max(sizes, key=sizes.get)
        comp = est._component_mask(
            labels, best, small.width, small.height)
        layers["human"] = human.assess(comp)
        if layers["human"]["state"] == "measured" \
                and not layers["human"]["person_like"]:
            reasons.append("human:not_person_like")

    if skel.joints:
        layers["evid"] = evid.locate(skel, mask)
        reasons += ["evid:off_mask:" + j
                    for j in evid.unsupported(layers["evid"])]
        layers["limbcov"] = limbcov.check(skel, mask)
        reasons += ["limbcov:" + b for b in limbcov.uncovered(
            layers["limbcov"])]
        layers["fit"] = fit.fit(skel, mask)
        if fit.verdict(layers["fit"]) == "poor":
            reasons.append("fit:poor")
        layers["stability"] = stability.probe(bmp, est, mdl)
        reasons += ["stability:unstable:" + j
                    for j in stability.unstable(layers["stability"])]
        layers["contrad"] = contrad.check({
            "classify": classify.analyze(skel),
            "axis": {"angle_deg": (
                (axis.principal(skel) or {}).get("angle_deg"))},
            "ground": ground.estimate(skel, small.height),
            "balance": balance.assess(skel),
        })
        reasons += ["contrad:" + c["id"]
                    for c in layers["contrad"]["contradictions"]]
        c_issues = consistency.audit(skel, mdl)
        layers["consistency"] = {
            "issues": c_issues,
            "verdict": "issues" if c_issues else "consistent"}
        reasons += ["consistency:" + i for i in c_issues]

    doc = pipeline._build_doc(
        skel, bmp, knowledge.sha256(raw), source_name, mdl,
        estimator=est, robust=robust)
    layers["gate"] = gate.check(skel, doc, mdl)
    reasons += layers["gate"]["reasons"]
    # the gate re-runs consistency.audit and re-reports the same
    # codes this function already appended — one finding, one reason
    reasons = list(dict.fromkeys(reasons))

    sev = {l: _severity(l, r) for l, r in layers.items()}
    worst = max((_SEV[s] for s in sev.values()), default=-1)
    verdict = {2: "fail", 1: "warn"}.get(worst, "pass")

    return {
        "verdict": verdict,
        "severity": sev,
        "reasons": reasons,
        "n_reasons": len(reasons),
        "layers": layers,
        "doc": pipeline.strip_runtime(doc),
        "state": "derived",
        "basis": "self-audit over imgqual/human/evid/limbcov/fit/"
                 "stability/contrad/consistency/gate",
    }
