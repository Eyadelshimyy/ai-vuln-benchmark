"""Knowledge document diff — what changed between two analyses.

Dataset QA needs to answer "is this second result actually
different?" without eyeballing JSON. `diff` compares two Knowledge
documents field by field: joints added / removed / moved, state
flips (observed -> predicted is a downgrade worth flagging),
confidence drift, body-model and pose-label changes.

A moved joint lists its displacement in pixels; a joint whose
`state` changed lists from -> to so regressions (observed becoming
predicted) are visible, not just motion.
"""

from __future__ import annotations


_MOVE_EPS = 0.5     # px below which a joint "didn't move"
_CONF_EPS = 0.01    # confidence delta worth reporting


def _joint_map(doc):
    """(comparable joints, malformed names) for one document.

    A hand-edited or foreign doc may hold non-dict joints or
    non-numeric coordinates; comparing those would crash or
    fabricate a move, so they are reported, not diffed.
    """
    sk = doc.get("skeleton") if isinstance(doc, dict) else None
    raw = sk.get("joints") if isinstance(sk, dict) else None
    joints = raw if isinstance(raw, dict) else {}
    good, bad = {}, []
    for name, j in joints.items():
        if (isinstance(j, dict)
                and isinstance(j.get("x"), (int, float))
                and isinstance(j.get("y"), (int, float))):
            good[name] = j
        else:
            bad.append(name)
    return good, sorted(bad)


def diff(a: dict, b: dict) -> dict:
    """Field-level diff of two Knowledge documents."""
    ja, bad_a = _joint_map(a)
    jb, bad_b = _joint_map(b)
    added = sorted(set(jb) - set(ja))
    removed = sorted(set(ja) - set(jb))
    moved, state_changed, conf_delta = [], [], {}
    for name in sorted(set(ja) & set(jb)):
        x0, y0 = ja[name].get("x", 0), ja[name].get("y", 0)
        x1, y1 = jb[name].get("x", 0), jb[name].get("y", 0)
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if dist > _MOVE_EPS:
            moved.append({
                "joint": name,
                "dx": round(x1 - x0, 2),
                "dy": round(y1 - y0, 2),
                "dist": round(dist, 2),
            })
        sa, sb = ja[name].get("state"), jb[name].get("state")
        if sa != sb:
            state_changed.append(
                {"joint": name, "from": sa, "to": sb})
        ca, cb = (ja[name].get("confidence"),
                  jb[name].get("confidence"))
        if (isinstance(ca, (int, float))
                and isinstance(cb, (int, float))):
            dc = cb - ca
            if abs(dc) > _CONF_EPS:
                conf_delta[name] = round(dc, 3)
    moved.sort(key=lambda m: -m["dist"])

    def _block(doc, *keys):
        cur = doc if isinstance(doc, dict) else {}
        for k in keys:
            cur = cur.get(k) if isinstance(cur, dict) else {}
        return cur if isinstance(cur, dict) else {}

    fa = _block(a, "skeleton", "frame")
    fb = _block(b, "skeleton", "frame")
    ma = _block(a, "skeleton", "body_model").get("name")
    mb = _block(b, "skeleton", "body_model").get("name")
    pa = _block(a, "pose").get("label")
    pb = _block(b, "pose").get("label")

    out = {
        "joints": {
            "added": added,
            "removed": removed,
            "moved": moved,
            "state_changed": state_changed,
            "confidence_delta": conf_delta,
            "malformed": {"a": bad_a, "b": bad_b},
        },
        "n_common": len(set(ja) & set(jb)),
        "frame_changed": fa != fb,
        "model_changed": (
            {"from": ma, "to": mb} if ma != mb else None),
        "pose_changed": (
            {"from": pa, "to": pb} if pa != pb else None),
        "identical": _identical(a, b),
    }
    out["summary"] = _summary(out)
    return out


def _identical(a: dict, b: dict) -> bool:
    import json
    aa = ({k: v for k, v in a.items() if k not in ("id", "created_at")}
          if isinstance(a, dict) else a)
    bb = ({k: v for k, v in b.items() if k not in ("id", "created_at")}
          if isinstance(b, dict) else b)
    try:
        return json.dumps(aa, sort_keys=True) == json.dumps(
            bb, sort_keys=True)
    except (TypeError, ValueError):
        return False


def _summary(d: dict) -> str:
    parts = []
    j = d["joints"]
    if j["moved"]:
        parts.append("%d joints moved (max %.1fpx)" % (
            len(j["moved"]), j["moved"][0]["dist"]))
    if j["added"]:
        parts.append("added: %s" % ",".join(j["added"]))
    if j["removed"]:
        parts.append("removed: %s" % ",".join(j["removed"]))
    if j["state_changed"]:
        down = [s for s in j["state_changed"]
                if s["from"] == "observed"]
        parts.append("%d state flips%s" % (
            len(j["state_changed"]),
            " (%d observed->predicted downgrades)" % len(down)
            if down else ""))
    if d["model_changed"]:
        m = d["model_changed"]
        parts.append("model %s->%s" % (m["from"], m["to"]))
    if d["pose_changed"]:
        p = d["pose_changed"]
        parts.append("pose %s->%s" % (p["from"], p["to"]))
    return "; ".join(parts) if parts else "no differences"
