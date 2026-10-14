"""Temporal ID — stable identity for skeletons across a frame sequence.

P3-19: the honest-contract tracker. Today each `analyze` produces an
independent skeleton; `track` threads them together so "the person in
frame 12" is the same `track_id` as in frame 3.

Scope is deliberately narrow — Slice estimates ONE person per image, so
this is a 1-track occupancy problem: does the next frame's skeleton
continue the current track, or was the person lost/re-acquired? Matching
is greedy nearest by pelvis distance (or centroid when pelvis is missing),
with a configurable max jump in torso-length units so a re-appearing
person elsewhere in frame starts a new track instead of teleporting.

Every assignment reports `state:"linked"` (continuity claim backed by a
distance measurement) — a track is a hypothesis, not a fact; the
`gap` field records frames skipped inside a track honestly.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .skeleton import Skeleton


def _anchor(skel: Skeleton) -> Optional[tuple]:
    """Tracking anchor: an OBSERVED pelvis if present, else the
    skeleton centroid. A predicted pelvis is prior fill — its position
    is a guess, so anchoring on it would compute the link distance on
    a guess."""
    p = skel.joints.get("pelvis")
    if p and p.state == "observed":
        return (p.x, p.y)
    return skel.centroid


def _torso(skel: Skeleton) -> float:
    """Normalising length (pelvis<->neck); falls back to a fraction of
    frame height when the torso is not observed — a predicted endpoint
    is a guess, not a measurement."""
    pj, nj = skel.joints.get("pelvis"), skel.joints.get("neck")
    p = (pj.x, pj.y) if pj and pj.state == "observed" else None
    n = (nj.x, nj.y) if nj and nj.state == "observed" else None
    if p and n:
        d = math.hypot(n[0] - p[0], n[1] - p[1])
        if d > 1e-6:
            return d
    return max(skel.image_height, 1) * 0.25


def track(frames: Sequence[Skeleton],
          max_jump_torso: float = 0.8) -> List[dict]:
    """Assign each frame to a track.

    Returns a list parallel to `frames`:
        {frame, track_id, state, gap, jump, anchor}
    `track_id` increments for each new track. `jump` is the pelvis
    displacement in torso units from the previous assigned frame
    (None on a track's first frame). `gap` counts frames where the
    track had no assignment between its previous hit and this one.
    Frames with no joints get `track_id: None, state: "empty"`.
    """
    out: List[dict] = []
    tracks: List[dict] = []  # {id, last_frame_idx, last_anchor, last_seen}
    next_id = 0
    for i, skel in enumerate(frames):
        anchor = _anchor(skel) if skel.joints else None
        if anchor is None:
            out.append({"frame": i, "track_id": None, "state": "empty",
                        "gap": None, "jump": None, "anchor": None})
            continue
        best = None
        for tr in tracks:
            jump = math.hypot(anchor[0] - tr["last_anchor"][0],
                              anchor[1] - tr["last_anchor"][1])
            jump_t = jump / _torso(skel)
            if best is None or jump_t < best[1]:
                best = (tr, jump_t)
        if best is not None and best[1] <= max_jump_torso:
            tr, jump_t = best
            gap = i - tr["last_seen"] - 1
            tr["last_anchor"], tr["last_seen"] = anchor, i
            out.append({"frame": i, "track_id": tr["id"],
                        "state": "linked" if gap == 0 else "reacquired",
                        "gap": gap, "jump": round(jump_t, 3),
                        "anchor": anchor})
        else:
            tracks.append({"id": next_id, "last_anchor": anchor,
                           "last_seen": i})
            out.append({"frame": i, "track_id": next_id,
                        "state": "new_track", "gap": None,
                        "jump": None, "anchor": anchor})
            next_id += 1
    return out


def summarize(assignments: Sequence[dict]) -> dict:
    """Roll-up of `track()` output: track count, occupancy, longest run."""
    ids = {}
    longest = 0
    cur, cur_id = 0, None
    for a in assignments:
        if a["track_id"] is not None:
            ids[a["track_id"]] = ids.get(a["track_id"], 0) + 1
        if a["track_id"] == cur_id and a["track_id"] is not None:
            cur += 1
        else:
            cur, cur_id = (1 if a["track_id"] is not None else 0,
                           a["track_id"])
        longest = max(longest, cur)
    return {
        "tracks": len(ids),
        "occupancy": {str(k): v for k, v in ids.items()},
        "empty_frames": sum(1 for a in assignments
                            if a["state"] == "empty"),
        "longest_run": longest,
        "assumption": ("single-person tracking: one track may hold the "
                       "person at a time; simultaneous people are not "
                       "distinguished by this tracker"),
    }
