"""Self-evaluation harness: how accurate is the pose estimator?

`draw_case` renders a figure programmatically while recording the exact
joint coordinates used, giving ground truth for free. `evaluate` then
measures detection rate and positional error against an estimator —
the honest counterpart of claiming an accuracy number.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

from .bitmap import Bitmap
from .pose import HeuristicPoseEstimator
from .skeleton import OBSERVED

Truth = Dict[str, Tuple[float, float]]


def draw_case(width: int = 160, height: int = 300, *,
              pose: str = "stand",
              skin=(60, 60, 60, 255),
              bg=(235, 235, 235, 255)) -> Tuple[Bitmap, Truth]:
    """Standing front figure; returns (bitmap, ground-truth joints).

    `pose` selects the arm geometry: "stand" hangs the arms beside
    the torso; "t" spreads them horizontally so calibration/bias
    measurements do not rest on a single pose."""
    bmp = Bitmap.new(width, height, bg)
    cx = width // 2
    truth: Truth = {}

    def rect(x0, y0, x1, y1):
        for y in range(int(y0), int(y1)):
            for x in range(int(x0), int(x1)):
                bmp.set(x, y, skin)

    def disc(ccx, ccy, r):
        for y in range(int(ccy - r), int(ccy + r) + 1):
            for x in range(int(ccx - r), int(ccx + r) + 1):
                if (x - ccx) ** 2 + (y - ccy) ** 2 <= r * r:
                    bmp.set(x, y, skin)

    head_r = width * 0.11
    sh_y = height * 0.22
    hip_y = height * 0.55
    torso_w = width * 0.34
    gap = 8
    disc(cx, head_r + 8, head_r)
    truth["head"] = (cx, head_r + 8)
    rect(cx - 3, head_r + 6, cx + 3, sh_y + 6)
    truth["neck"] = (cx, sh_y + 6)
    if pose == "t":
        # horizontal arm strip at shoulder height; reach is the free
        # side space so the fixture fits narrow canvases too
        rect(cx - torso_w / 2, sh_y, cx + torso_w / 2, hip_y)
        reach = (cx - torso_w / 2) * 0.92
        ay0, ay1 = sh_y + 2, sh_y + 10
        rect(cx - torso_w / 2 - reach, ay0, cx - torso_w / 2, ay1)
        rect(cx + torso_w / 2, ay0, cx + torso_w / 2 + reach, ay1)
        amy = (ay0 + ay1) / 2
        truth["shoulder_l"] = (cx - torso_w / 2, amy)
        truth["shoulder_r"] = (cx + torso_w / 2 - 1, amy)
        truth["elbow_l"] = (cx - torso_w / 2 - reach / 2, amy)
        truth["elbow_r"] = (cx + torso_w / 2 + reach / 2, amy)
        truth["wrist_l"] = (cx - torso_w / 2 - reach, amy)
        truth["wrist_r"] = (cx + torso_w / 2 + reach - 1, amy)
    else:
        rect(cx - torso_w / 2 - 10, sh_y, cx + torso_w / 2 + 10, sh_y + 6)
        rect(cx - torso_w / 2, sh_y, cx + torso_w / 2, hip_y)
        truth["shoulder_l"] = (cx - torso_w / 2 - 10, sh_y + 3)
        truth["shoulder_r"] = (cx + torso_w / 2 + 9, sh_y + 3)
        rect(cx - torso_w / 2 - 10, sh_y + 6, cx - torso_w / 2 - 2,
             height * 0.75)
        rect(cx + torso_w / 2 + 2, sh_y + 6, cx + torso_w / 2 + 10,
             height * 0.75)
        truth["elbow_l"] = (cx - torso_w / 2 - 6, height * 0.5)
        truth["elbow_r"] = (cx + torso_w / 2 + 6, height * 0.5)
        truth["wrist_l"] = (cx - torso_w / 2 - 6, height * 0.75 - 1)
        truth["wrist_r"] = (cx + torso_w / 2 + 6, height * 0.75 - 1)
    truth["chest"] = (cx, (sh_y + hip_y) / 2)
    truth["pelvis"] = (cx, hip_y - 1)
    truth["hip_l"] = (cx - torso_w / 2, hip_y)
    truth["hip_r"] = (cx + torso_w / 2 - 1, hip_y)
    rect(cx - torso_w / 2, hip_y, cx - gap / 2, height - 10)
    rect(cx + gap / 2, hip_y, cx + torso_w / 2, height - 10)
    truth["knee_l"] = ((cx - torso_w / 2 + cx - gap / 2) / 2,
                       hip_y + (height - 10 - hip_y) * 0.55)
    truth["knee_r"] = ((cx + gap / 2 + cx + torso_w / 2) / 2,
                       hip_y + (height - 10 - hip_y) * 0.55)
    truth["ankle_l"] = (truth["knee_l"][0], height - 11)
    truth["ankle_r"] = (truth["knee_r"][0], height - 11)
    truth["foot_l"] = (truth["knee_l"][0], height - 11)
    truth["foot_r"] = (truth["knee_r"][0], height - 11)
    return bmp, truth


def evaluate(pairs: Iterable[Tuple[Bitmap, Truth]],
             estimator: Optional[HeuristicPoseEstimator] = None
             ) -> dict:
    """Detection rate and mean positional error over ground truth."""
    est = estimator or HeuristicPoseEstimator()
    cases = total = found = observed = 0
    err_sum = 0.0
    per_joint: Dict[str, list] = {}
    for bmp, truth in pairs:
        cases += 1
        skel = est.estimate(bmp)
        for name, (tx, ty) in truth.items():
            total += 1
            j = skel.get(name)
            if j is None:
                continue
            found += 1
            err = math.hypot(j.x - tx, j.y - ty)
            err_sum += err
            per_joint.setdefault(name, []).append(
                (err, j.state == OBSERVED))
            if j.state == OBSERVED:
                observed += 1
    stats = {
        n: {"cases": len(v),
            "mean_error": round(sum(e for e, _ in v) / len(v), 2),
            "observed": sum(1 for _, o in v if o)}
        for n, v in sorted(per_joint.items())
    }
    return {
        "cases": cases,
        "joints_expected": total,
        "joints_found": found,
        "detection_rate": round(found / max(1, total), 4),
        "observed_rate": round(observed / max(1, total), 4),
        "mean_error_px": round(err_sum / max(1, found), 2),
        "per_joint": stats,
    }
