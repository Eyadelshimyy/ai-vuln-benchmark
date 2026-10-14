"""End-to-end Phase 3 demo — 2D skeleton to 3D bridges.

Walks the full chain on a synthetic figure:

    draw_case -> estimate -> lift (pseudo-3D) -> rig (bones)
    -> retarget (different proportions) -> bvh + gltf (exports)

Every stage prints what it produced and what it *assumed* — the demo
exists to show the honesty contract holding across the 3D boundary:
depth cues say `basis`, retargeted joints say `predicted`, exports
carry the provenance through.

Run:
    python3 examples/demo_3d.py [out_dir]
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from slice import bvh, evaluate, gltf, lift, pose, retarget, rig
from slice.skeleton import Joint, Skeleton


def _scaled(skel: Skeleton, factor: float) -> Skeleton:
    """A differently-proportioned target body (all bones x `factor`)."""
    px, py = skel.point("pelvis")
    out = Skeleton(skel.image_width, skel.image_height)
    for name, j in skel.joints.items():
        out.set(Joint(name, px + (j.x - px) * factor,
                      py + (j.y - py) * factor, j.confidence,
                      state=j.state, basis="scaled demo target"))
    return out


def run(out_dir: str = "demo_3d_out") -> dict:
    os.makedirs(out_dir, exist_ok=True)

    bmp, truth = evaluate.draw_case()
    skel = pose.HeuristicPoseEstimator().estimate(bmp)
    obs = sum(1 for j in skel.joints.values() if j.state == "observed")
    print(f"[1] estimate: {len(skel.joints)} joints "
          f"({obs} observed, {len(skel.joints) - obs} predicted)")

    lifted = lift.lift(skel)
    print(f"[2] lift: {len(lifted)} joints lifted, "
          f"depth_spread={lift.depth_spread(lifted)} "
          f"(0 = flat; basis per joint is honest about cues)")

    bones = rig.build(skel)
    print(f"[3] rig: {len(bones)} bones, "
          f"total_length={rig.total_bone_length(bones)}px")

    target = _scaled(skel, 1.15)
    ret = retarget.retarget(skel, target)
    rpred = sum(1 for j in ret.joints.values() if j.state == "predicted")
    print(f"[4] retarget: pose copied onto 1.15x body, "
          f"{rpred}/{len(ret.joints)} joints predicted (new positions "
          f"are inferred, not measured)")

    bvh_path = os.path.join(out_dir, "figure.bvh")
    with open(bvh_path, "w") as f:
        f.write(bvh.export(skel))
    gltf_path = os.path.join(out_dir, "figure.gltf")
    with open(gltf_path, "w") as f:
        json.dump(gltf.to_gltf(skel), f, indent=2)
    print(f"[5] export: {bvh_path}, {gltf_path}")
    print("done — every stage carried its provenance forward")
    return {"bvh": bvh_path, "gltf": gltf_path,
            "joints": len(skel.joints), "bones": len(bones)}


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "demo_3d_out")
