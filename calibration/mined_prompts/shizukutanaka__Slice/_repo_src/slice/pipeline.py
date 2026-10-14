"""End-to-end pipeline: image bytes -> Knowledge dict.

Import -> Pose -> Skeleton -> Prediction -> Ratio -> Knowledge.
Shared by the CLI and the REST layer so both produce identical output.
"""

from __future__ import annotations

from typing import List, Optional

from . import (__version__, angles, balance, bitmap, classify,
               consistency, dynamics, framepos, gesture, knowledge,
               occlusion, pose, predict, ratio, spine, style, symmetry)
from .anatomy import BODY_MODELS
from .skeleton import OBSERVED

ESTIMATOR = pose.HeuristicPoseEstimator()
# opt-in profile for noisy/uneven real photos: adaptive threshold,
# cast-shadow rejection, morphological mask cleanup — all merged
# estimator features that the default profile leaves off.
ROBUST_ESTIMATOR = pose.HeuristicPoseEstimator(
    adaptive=True, reject_shadow=True, clean=True)


def _build_doc(skel, bmp, image_sha: str, source_name: str,
               model: Optional[str], estimator=None,
               robust: bool = False) -> dict:
    estimator = estimator or ESTIMATOR
    added = predict.complete(skel, model)
    ratios = ratio.analyze(skel, centroid=skel.centroid)
    cls = classify.analyze(skel)
    doc = knowledge.build(
        skel, ratios, cls,
        image_sha256=image_sha,
        source_name=source_name,
        engine={"name": estimator.name, "version": estimator.version,
                "slice": __version__,
                "profile": "robust" if robust else "default"},
    )
    # the mask decision that produced every joint's evidence —
    # Otsu value or the fixed fallback, whichever actually ran.
    # Carried on the skeleton (not read off the shared estimator) so
    # concurrent REST calls can't cross-attribute each other's choice
    thr = getattr(skel, "mask_threshold", None)
    if thr:
        doc["engine"]["mask_threshold"] = {
            "value": round(thr[0], 1), "method": thr[1]}
    doc["prediction"]["filled"] = [j.name for j in added]
    doc["style"] = style.analyze(bmp)
    analysis = {
        "angles": angles.analyze(skel),
        "symmetry": symmetry.score(skel),
        "balance": balance.assess(skel),
        "spine": spine.classify(skel),
        "gesture": gesture.summarize(skel),
        "dynamics": dynamics.score(skel),
        "occlusion": occlusion.audit(skel),
        "frame": framepos.analyze(skel, skel.image_width,
                                  skel.image_height),
        "consistency": {"issues": consistency.audit(skel, model)},
    }
    # every layer must be a dict (knowledge.validate); omit empty ones
    doc["analysis"] = {k: v for k, v in analysis.items() if v is not None}
    if (bmp.width, bmp.height) != (skel.image_width, skel.image_height):
        # joints live in the estimator's downscaled working space —
        # record the source resolution so coordinates can be mapped
        # back onto the image named by image_sha256
        doc["skeleton"]["frame"]["source"] = {
            "width": bmp.width, "height": bmp.height}
    doc["warnings"] = _warnings(skel)
    if model and model not in BODY_MODELS:
        # an unrecognized model name silently used the default prior in
        # both pose priors and predict.complete — the doc must say so
        doc["warnings"].append("unknown_body_model")
    doc["_bitmap"] = bmp      # runtime only: overlay rendering
    doc["_skeleton"] = skel   # runtime only
    return doc


def analyze(raw: bytes, *, model: Optional[str] = None,
            source_name: str = "", robust: bool = False) -> dict:
    bmp = bitmap.decode(raw)
    estimator = ROBUST_ESTIMATOR if robust else ESTIMATOR
    skel = estimator.estimate(bmp, model or "adult")
    return _build_doc(skel, bmp, knowledge.sha256(raw), source_name,
                      model, estimator=estimator, robust=robust)


def analyze_multi(raw: bytes, *, model: Optional[str] = None,
                  source_name: str = "", top_k: int = 4,
                  robust: bool = False) -> List[dict]:
    """One Knowledge document per detected person.

    Each document is a complete v1 analysis of its own skeleton and
    carries a `people` block recording which foreground component it
    came from — same-image siblings, not a merged figure. The list
    may be shorter than the visible crowd (touching silhouettes are
    one component) or empty (nothing detected): the pixels decide.
    """
    bmp = bitmap.decode(raw)
    estimator = ROBUST_ESTIMATOR if robust else ESTIMATOR
    skels = estimator.estimate_multi(bmp, model or "adult",
                                     top_k=top_k)
    sha = knowledge.sha256(raw)
    docs = [_build_doc(s, bmp, sha, source_name, model,
                       estimator=estimator, robust=robust)
            for s in skels]
    for i, d in enumerate(docs):
        d["people"] = {"index": i, "count": len(docs),
                       "state": "observed",
                       "basis": "foreground component"}
    return docs


def _warnings(skel) -> list:
    """Evidence-thinness flags: which parts of the document rest on
    prediction rather than on anything visible in the image."""
    obs = {n for n, j in skel.joints.items() if j.state == OBSERVED}
    w = []
    if len(obs) < 8:
        w.append("few_observed_joints")
    if not any(n in ("pelvis", "chest", "neck", "head") for n in obs):
        # the kinematic root itself is predicted — every coordinate in
        # the document hangs on an unmeasured anchor
        w.append("no_observed_torso")
    if not any(n.startswith("wrist") for n in obs):
        w.append("no_observed_wrists")
    if not any(n.startswith("ankle") or n.startswith("foot") for n in obs):
        w.append("no_observed_feet")
    if "head" not in obs:
        w.append("no_observed_head")
    return w


def strip_runtime(doc: dict) -> dict:
    """Knowledge JSON is the product; drop non-serializable internals."""
    return {k: v for k, v in doc.items() if not k.startswith("_")}


def observed_count(doc: dict) -> int:
    return len(doc["prediction"]["observed"])



def people_count(doc_or_docs) -> int:
    """How many person documents an analysis produced (1 for `analyze`)."""
    return len(doc_or_docs) if isinstance(doc_or_docs, list) else 1
