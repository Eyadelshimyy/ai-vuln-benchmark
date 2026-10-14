"""Per-joint trust grade — synthesize the audit layers.

`calib` knows how accurate a confidence value tends to be,
`stability` knows how much the joint moves under parameter
perturbation, `evid` knows whether the joint sits on real
foreground. Each answers alone; a consumer that wants "should I
trust this wrist?" has to query all three and arbitrate.

`grade` is that arbitration — deliberately small: it takes the
*already-computed* results of those layers (any subset) and folds
them into one of three grades per joint:

- `low`    — observed joint off the mask, unstable under
             perturbation, or calibrated accuracy under 0.3
- `medium` — sensitive, single-run, calibrated accuracy < 0.5,
             or simply `predicted` (fill-in, not evidence)
- `high`   — nothing contradicted it

A predicted joint off the mask is *expected*, not a failure —
priors legitimately place joints outside the silhouette (same
rule evid.unsupported uses: off_mask only accuses an observed
claim), so it does not push a joint below `medium`.

Every joint's `factors` list names which inputs pushed it down, so
the grade stays an explanation, not a number. Missing inputs are
fine: grade what you have, say what you lacked.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .skeleton import OBSERVED, Skeleton

RANK = {"low": 0, "medium": 1, "high": 2}


def _cap(grade: str, ceiling: str) -> str:
    return grade if RANK[grade] <= RANK[ceiling] else ceiling


def grade(skel: Skeleton,
          calib_table: Optional[list] = None,
          stability: Optional[dict] = None,
          evidence: Optional[dict] = None) -> Dict:
    """Grade every joint; return {joints, distribution, inputs_used}.

    `calib_table` — output of `calib.reliability_table`
    `stability`   — output of `stability.probe`
    `evidence`    — output of `evid.locate`
    """
    used = []
    if calib_table is not None:
        used.append("calib")
    if stability is not None:
        used.append("stability")
    if evidence is not None:
        used.append("evid")

    joints: Dict[str, Dict] = {}
    dist = {"high": 0, "medium": 0, "low": 0}
    for name, j in sorted(skel.joints.items()):
        g = "high"
        factors: List[str] = []

        if j.state != OBSERVED:
            g = _cap(g, "medium")
            factors.append("predicted_fill")

        if evidence is not None:
            zone = (evidence.get("joints") or {}).get(name, {}) \
                .get("zone")
            if zone == "off_mask" and j.state == OBSERVED:
                g = "low"
                factors.append("off_mask")

        if stability is not None:
            v = (stability.get("joints") or {}).get(name, {}) \
                .get("verdict")
            if v == "unstable":
                g = "low"
                factors.append("unstable")
            elif v == "sensitive":
                g = _cap(g, "medium")
                factors.append("sensitive")
            elif v == "single_run":
                g = _cap(g, "medium")
                factors.append("single_run")

        if calib_table is not None and j.state == OBSERVED:
            i = min(len(calib_table) - 1,
                    int(j.confidence * len(calib_table)))
            acc = calib_table[i].get("accuracy")
            if acc is not None:
                if acc < 0.3:
                    g = "low"
                    factors.append("calibrated_accuracy %.2f" % acc)
                elif acc < 0.5:
                    g = _cap(g, "medium")
                    factors.append("calibrated_accuracy %.2f" % acc)

        joints[name] = {"grade": g, "factors": factors}
        dist[g] += 1

    return {
        "joints": joints,
        "distribution": dist,
        "inputs_used": used,
        "state": "graded" if used else "heuristic",
        "basis": "synthesis over " + "+".join(used)
        if used else "no audit inputs supplied — predicted/state only",
    }


def trusted(result: Dict, minimum: str = "medium") -> List[str]:
    """Joint names at or above `minimum` grade."""
    floor = RANK[minimum]
    return sorted(n for n, v in result["joints"].items()
                  if RANK[v["grade"]] >= floor)
