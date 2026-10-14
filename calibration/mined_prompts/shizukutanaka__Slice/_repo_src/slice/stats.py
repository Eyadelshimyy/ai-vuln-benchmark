"""Corpus statistics — meta-knowledge over many analyses.

Aggregate many estimated skeletons (or stored knowledge docs) to see
how the estimator performs across a dataset: per-joint observed rate,
mean confidence, which limbs go missing most. A wrist observed in 20%
of frames and a pelvis in 99% say different things about the input
domain and where the estimator needs help.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from .landmarks import JOINTS
from .skeleton import OBSERVED, Skeleton


def joint_stats(skels: Iterable[Skeleton]) -> Dict[str, dict]:
    """Per-joint observed rate / mean confidence / fill rate."""
    obs = {n: 0 for n in JOINTS}
    conf_sum = {n: 0.0 for n in JOINTS}
    obs_conf_sum = {n: 0.0 for n in JOINTS}
    present = {n: 0 for n in JOINTS}
    total = 0
    for skel in skels:
        total += 1
        for n in JOINTS:
            j = skel.joints.get(n)
            if j is None:
                continue
            present[n] += 1
            conf_sum[n] += j.confidence
            if j.state == OBSERVED:
                obs[n] += 1
                obs_conf_sum[n] += j.confidence
    out: Dict[str, dict] = {}
    for n in JOINTS:
        out[n] = {
            "observed_rate": round(obs[n] / total, 3) if total else 0.0,
            "fill_rate": round(present[n] / total, 3) if total else 0.0,
            # blends observed (evidence) and predicted (prior) joints —
            # kept for compatibility; prefer the observed-only mean
            "mean_confidence": (round(conf_sum[n] / present[n], 3)
                                if present[n] else 0.0),
            # confidence measured over observed joints only — the
            # number that actually describes evidence quality
            "mean_observed_confidence":
                (round(obs_conf_sum[n] / obs[n], 3)
                 if obs[n] else 0.0),
        }
    return out


def weakest_joints(stats: Dict[str, dict], n: int = 5) -> List[str]:
    """Joint names with the lowest observed rate (blind spots)."""
    return [name for name, _ in
            sorted(stats.items(),
                   key=lambda kv: kv[1]["observed_rate"])[:n]]


def summary(skels: Iterable[Skeleton]) -> dict:
    """Corpus-level digest."""
    skels = list(skels)
    js = joint_stats(skels)
    if not skels:
        return {"cases": 0}
    mean_obs = sum(s["observed_rate"] for s in js.values()) / len(js)
    return {
        "cases": len(skels),
        "mean_observed_rate": round(mean_obs, 3),
        "weakest_joints": weakest_joints(js),
        "per_joint": js,
    }
