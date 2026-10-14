"""Slice CLI.

    python -m slice analyze <image> [-o knowledge.json]
        [--model adult|child|deformed] [--robust] [--overlay out.png]
        [--store DIR]
    python -m slice batch <dir> --store DIR [--model M] [--robust] [-r]
    python -m slice audit <image> [--model M] [--robust] [-o audit.json]
        — run every quality layer over one image and print a verdict
    python -m slice audit --store DIR  — audit the stored documents
          (honesty lint, near-duplicates, joint observed-rate)
    python -m slice calib
        — confidence calibration vs ground-truth fixtures
    python -m slice bias
        — per-joint systematic vs random error on fixtures
    python -m slice batch <dir> --store DIR [--model M] [-r]
    python -m slice audit <image|dir> [--model M] [--robust] [-o audit.json]
        — run every quality layer over one image (or a directory)
          and print a verdict
    python -m slice serve [--port 8000] [--store DIR]
    python -m slice list [--store DIR]
    python -m slice priorchk
        — audit the BODY_MODELS prior tables
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os
import sys

from . import (__version__, bench, bitmap, calib, evaluate, knowledge,
               limbcov, pipeline, render, rest, selfcheck)
from . import (__version__, bitmap, calib, evaluate, knowledge, limbcov,
               pipeline, render, rest, selfcheck, storechk)
from . import (__version__, bitmap, calib, compare, diff, evaluate,
               knowledge, limbcov, pipeline, render, rest, selfcheck)
from . import (__version__, bitmap, calib, evaluate, knowledge, limbcov,
               pipeline, render, rest, selfcheck, storechk)
from . import (__version__, axis, bias, bitmap, calib, contact, dominance,
               evaluate, extjoints, framefit, ground, handpos, horizon,
               knowledge, limbcov, limbs, mass, mirror, pipeline, plumb, reach,
               render, rest, rom, selfcheck, storechk)
from . import (__version__, bitmap, knowledge, pipeline, priorchk,
               render, rest, selfcheck)
from . import (__version__, bitmap, calib, evaluate, knowledge, limbcov, 
               pipeline, priorchk, render, rest, selfcheck, storechk)
from . import (__version__, bitmap, knowledge, pipeline, render, rest,
               mutate, selfcheck)
from . import (__version__, bitmap, knowledge, limbcov, mutate, pipeline,
               render, rest, selfcheck)
from . import (__version__, bitmap, calib, evaluate, knowledge, limbcov, 
               mutate, pipeline, render, rest, selfcheck, storechk)
from . import (__version__, bitmap, calib, evaluate, knowledge, limbcov, 
               pipeline, render, rest, rig, selfcheck, storechk)
from .anatomy import BODY_MODELS
from . import (oks)
from . import (balance, classify, contrad)
from . import (reid)
from . import (describe)
from . import (autocrop, crop)
from . import (ascii, bvh, coco, gltf, heatmap, paf, svg)
from . import (evid)
from . import (imgqual)
from . import (human)


def _cmd_analyze(a) -> int:
    with open(a.image, "rb") as f:
        raw = f.read()
    if a.multi:
        return _cmd_analyze_multi(a, raw)
    try:
        doc = pipeline.analyze(raw, model=a.model, source_name=a.image,
                               robust=a.robust)
    except bitmap.UnsupportedFormat as e:
        print(f"unsupported image: {e}", file=sys.stderr)
        return 2
    out = pipeline.strip_runtime(doc)
    if a.store:
        kid = knowledge.KnowledgeStore(a.store).save(out)
        print(f"saved: {kid}", file=sys.stderr)
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {a.output}", file=sys.stderr)
    else:
        print(text)
    if a.overlay:
        png = render.overlay_png(doc["_bitmap"], doc["_skeleton"])
        with open(a.overlay, "wb") as f:
            f.write(png)
        print(f"wrote {a.overlay}", file=sys.stderr)
    j = doc["skeleton"]["joints"]
    obs = sum(1 for v in j.values() if v["state"] == "observed")
    pred = sum(1 for v in j.values() if v["state"] == "predicted")
    pose_l = (doc.get("pose") or {}).get("label", "?")
    style_l = (doc.get("style") or {}).get("label", "?")
    bm = doc["skeleton"].get("body_model") or {}
    print(f"joints: {obs} observed / {pred} predicted", file=sys.stderr)
    print(f"pose: {pose_l} | style: {style_l} | model: "
          f"{bm.get('label', '?')} ({bm.get('state', '?')})",
          file=sys.stderr)
    warns = out.get("warnings") or []
    if warns:
        print("warnings: " + ", ".join(warns), file=sys.stderr)
    return 0


def _cmd_batch(a) -> int:
    """Analyze every image under a directory into a KnowledgeStore."""
    import os
    store = knowledge.KnowledgeStore(a.store)
    exts = (".png", ".bmp", ".jpg", ".jpeg", ".webp")
    paths = []
    if a.recursive:
        for root, _dirs, files in os.walk(a.dir):
            paths += [os.path.join(root, f) for f in files]
    else:
        paths = [os.path.join(a.dir, f) for f in os.listdir(a.dir)]
    paths = sorted(p for p in paths
                   if p.lower().endswith(exts))
    saved = failed = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                raw = f.read()
            doc = pipeline.analyze(raw, model=a.model, source_name=p,
                                   robust=a.robust)
            kid = store.save(pipeline.strip_runtime(doc))
            obs = doc["coverage"]["observed"]
            print(f"{p}: {kid} ({obs} observed)")
            saved += 1
        except (bitmap.UnsupportedFormat, OSError, ValueError) as e:
            print(f"{p}: FAILED {e}", file=sys.stderr)
            failed += 1
    print(f"batch: {saved} saved, {failed} failed "
          f"({len(paths)} images)", file=sys.stderr)
    return 0 if saved else 1


def _cmd_analyze_multi(a, raw) -> int:
    try:
        docs = pipeline.analyze_multi(raw, model=a.model,
                                      source_name=a.image,
                                      robust=a.robust)
    except bitmap.UnsupportedFormat as e:
        print(f"unsupported image: {e}", file=sys.stderr)
        return 2
    outs = []
    for d in docs:
        out = pipeline.strip_runtime(d)
        if a.store:
            kid = knowledge.KnowledgeStore(a.store).save(out)
            print(f"saved: {kid}", file=sys.stderr)
        outs.append(out)
    text = json.dumps(outs, ensure_ascii=False, indent=2)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {a.output}", file=sys.stderr)
    else:
        print(text)
    if a.overlay:
        print("multi: --overlay skipped (per-person overlay not written)",
              file=sys.stderr)
    print(f"people: {len(outs)}", file=sys.stderr)
    return 0


def _cmd_audit(a) -> int:
    import os
    if a.image and os.path.isdir(a.image):
        return _cmd_audit_dir(a)
    if a.store:
        return _cmd_audit_store(a)
    if not a.image:
        print("audit: image path or --store DIR required",
              file=sys.stderr)
        return 2
    import os
    if os.path.isdir(a.image):
        return _cmd_audit_dir(a)
    with open(a.image, "rb") as f:
        raw = f.read()
    try:
        res = selfcheck.run(raw, model=a.model, source_name=a.image,
                            robust=a.robust)
    except bitmap.UnsupportedFormat as e:
        print(f"unsupported image: {e}", file=sys.stderr)
        return 2
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"wrote {a.output}", file=sys.stderr)
    else:
        print(json.dumps({k: res[k] for k in
                          ("verdict", "severity", "reasons")},
                         ensure_ascii=False, indent=2))
    print(f"verdict: {res['verdict']} "
          f"({res['n_reasons']} reasons)", file=sys.stderr)
    return 0 if res["verdict"] != "fail" else 1


def _cmd_bench(a) -> int:
    """Accuracy/latency gate over ground-truth fixtures."""
    rep = bench.run(repeats=a.repeats)
    ok, fails = bench.gate(rep)
    if a.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"estimate: {rep['ms_per_estimate']}ms/image "
              f"({rep['cases']} cases x{rep['repeats']})")
        print(f"detection={rep['detection_rate']} "
              f"observed={rep['observed_rate']} "
              f"err={rep['mean_error_px']}px "
              f"oks={rep['mean_oks']}")
    for f in fails:
        print(f"FAIL {f}", file=sys.stderr)
    return 0 if ok else 1


def _diff_load(arg, store_dir, model, robust):
    """arg = image path or k_<id> in the store."""
    import os
    if os.path.isfile(arg):
        with open(arg, "rb") as f:
            doc = pipeline.analyze(f.read(), model=model,
                                   source_name=arg, robust=robust)
        return pipeline.strip_runtime(doc)
    return knowledge.KnowledgeStore(store_dir).get(arg)


def _cmd_diff(a) -> int:
    try:
        da = _diff_load(a.a, a.store, a.model, a.robust)
        db = _diff_load(a.b, a.store, a.model, a.robust)
    except bitmap.UnsupportedFormat as e:
        print(f"unsupported image: {e}", file=sys.stderr)
        return 2
    except (KeyError, OSError) as e:
        print(f"cannot load: {e}", file=sys.stderr)
        return 2
    res = diff.diff(da, db)
    res["pose_distance"] = compare.pose_distance(da, db)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"wrote {a.output}", file=sys.stderr)
    else:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _probe_dispatch(layer, skel, mask):
    """layer name -> result dict, or None on unknown layer."""
    if layer == "axis":
        return {"principal": axis.principal(skel),
                "tilt": axis.tilt(skel)}
    if layer == "plumb":
        return {"line": plumb.line(skel),
                "forward_head": plumb.forward_head(skel),
                "assess": plumb.assess(skel)}
    if layer == "limbs":
        return limbs.profile(skel)
    if layer == "rom":
        return {"check": rom.check(skel),
                "violations": rom.violations(skel)}
    if layer == "contact":
        return contact.summary(skel)
    if layer == "dominance":
        return {"assess": dominance.assess(skel),
                "cues": dominance.cues(skel)}
    if layer == "handpos":
        return {"summary": handpos.summary(skel),
                "positions": handpos.positions(skel)}
    if layer == "framefit":
        return framefit.assess(skel)
    if layer == "ground":
        return {"estimate": ground.estimate(
                    skel, skel.image_height),
                "clearance": ground.clearance(skel)}
    if layer == "reach":
        return reach.workspace(skel)
    if layer == "horizon":
        return horizon.estimate(skel)
    if layer == "mass":
        area = sum(r.count(True) for r in mask)
        return {"estimate": mass.estimate(skel, area),
                "bmi": mass.bmi(skel, area),
                "area_px": area}
    if layer == "extjoints":
        derived = extjoints.derive(skel)
        return {"joints": {n: {"x": j.x, "y": j.y,
                               "confidence": j.confidence,
                               "state": j.state,
                               "basis": j.basis}
                           for n, j in derived.items()},
                "vocabulary": extjoints.vocabulary()}
    return None
_PROBE_LAYERS = ("axis", "plumb", "limbs", "rom", "contact",
                 "dominance", "handpos", "framefit", "ground",
                 "reach", "horizon", "mass", "extjoints")
def _cmd_probe(a) -> int:
    """Run one semantic layer directly on an image."""
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    try:
        with open(a.image, "rb") as f:
            bmp = bitmap.decode(f.read())
    except (bitmap.UnsupportedFormat, OSError) as e:
        print(f"cannot load {a.image}: {e}", file=sys.stderr)
        return 2
    skel = est.estimate(bmp, a.model or "adult")
    if not skel.joints:
        print("no person detected", file=sys.stderr)
        return 1
    small = bmp.downscale(est.max_dim)
    mask = est._mask(small)
    res = _probe_dispatch(a.layer, skel, mask)
    if res is None:
        print(f"unknown layer: {a.layer}", file=sys.stderr)
        return 2
    print(json.dumps({"layer": a.layer, "result": res},
                     ensure_ascii=False, indent=2))
    return 0
def _cmd_mirror(a) -> int:
    """Estimator left/right consistency: est(flip(img)) vs flip(est(img))."""
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    try:
        with open(a.image, "rb") as f:
            bmp = bitmap.decode(f.read())
    except (bitmap.UnsupportedFormat, OSError) as e:
        print(f"cannot load {a.image}: {e}", file=sys.stderr)
        return 2
    skel_a = est.estimate(bmp, a.model or "adult")
    if not skel_a.joints:
        print("no person detected", file=sys.stderr)
        return 1
    expected = mirror.flip_skeleton(skel_a)
    actual = est.estimate(mirror.flip_bitmap(bmp),
                          a.model or "adult")
    drift = {}
    for name, je in expected.joints.items():
        ja = actual.joints.get(name)
        if ja is None:
            drift[name] = None
            continue
        drift[name] = round(math.hypot(
            ja.x - je.x, ja.y - je.y), 2)
    vals = [v for v in drift.values() if v is not None]
    res = {
        "joints_compared": len(vals),
        "missing_in_flipped": [n for n, v in drift.items()
                               if v is None],
        "mean_drift_px": round(
            sum(vals) / len(vals), 2) if vals else 0.0,
        "max_drift": ({"joint": max(drift, key=lambda n:
                                   drift[n] or -1),
                       "px": max(vals)} if vals else None),
        "per_joint": drift,
        "verdict": ("symmetric"
                    if vals and max(vals) <= a.tolerance
                    else "drifted"),
        "tolerance_px": a.tolerance,
        "state": "derived",
        "basis": "est(flip(image)) vs flip(est(image)); "
                 "nonzero drift = estimator left/right bias",
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["verdict"] == "symmetric" else 1


def _norm_skel(skel):
    """Rebuild `skel` in normalized pose space (pelvis origin,
    torso unit) so cross-resolution comparisons are meaningful."""
    from .skeleton import Joint, Skeleton
    norm = skel.normalized()
    out = Skeleton(1, 1)
    if not norm:
        return out
    out.orientation = dict(skel.orientation)
    out.body_model = dict(skel.body_model)
    for name, pos in norm["joints"].items():
        j = skel.joints[name]
        out.set(Joint(name, pos["x"], pos["y"], j.confidence,
                      state=j.state, basis=j.basis))
    return out


def _cmd_oks(a) -> int:
    """OKS similarity of skeleton B against reference A."""
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    skels = []
    for p in (a.a, a.b):
        try:
            with open(p, "rb") as f:
                skels.append(_norm_skel(est.estimate(
                    bitmap.decode(f.read()), a.model or "adult")))
        except (bitmap.UnsupportedFormat, OSError) as e:
            print(f"cannot load {p}: {e}", file=sys.stderr)
            return 2
    if not all(sk.joints for sk in skels):
        print("no person in one or both images",
              file=sys.stderr)
        return 1
    score = oks.oks(skels[0], skels[1])
    res = {"oks": score,
           "per_joint": oks.per_joint(skels[0], skels[1]),
           "basis": "COCO OKS in normalized pose space "
                    "(pelvis origin, torso unit)"}
    if score is None:
        print("no comparable joints", file=sys.stderr)
        return 1
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _cmd_contrad(a) -> int:
    """Cross-layer contradiction audit (pose/axis/ground/balance)."""
    try:
        with open(a.image, "rb") as f:
            bmp = bitmap.decode(f.read())
    except (bitmap.UnsupportedFormat, OSError) as e:
        print(f"cannot load {a.image}: {e}", file=sys.stderr)
        return 2
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    skel = est.estimate(bmp, a.model or "adult")
    if not skel.joints:
        print("no person detected", file=sys.stderr)
        return 1
    layers = {
        "classify": classify.analyze(skel),
        "axis": axis.principal(skel),
        "ground": ground.estimate(skel, skel.image_height),
        "balance": balance.assess(skel),
    }
    layers = {k: v for k, v in layers.items() if v}
    res = contrad.check(layers)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["verdict"] == "consistent" else 1


def _cmd_reid(a) -> int:
    """Same-person check between two images (bone-ratio matching)."""
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    feats = []
    for p in (a.a, a.b):
        try:
            with open(p, "rb") as f:
                skel = est.estimate(
                    bitmap.decode(f.read()), a.model or "adult")
        except (bitmap.UnsupportedFormat, OSError) as e:
            print(f"cannot load {p}: {e}", file=sys.stderr)
            return 2
        if not skel.joints:
            print(f"no person in {p}", file=sys.stderr)
            return 1
        feats.append(reid.features(skel))
    res = reid.compare(feats[0], feats[1],
                       threshold=a.threshold)
    res["features_a"] = feats[0]["vector"]
    res["features_b"] = feats[1]["vector"]
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("same_person") else 1


def _cmd_describe(a) -> int:
    """Describe one image's figure in a sentence."""
    with open(a.image, "rb") as f:
        raw = f.read()
    try:
        bmp = bitmap.decode(raw)
    except bitmap.UnsupportedFormat as e:
        print(f"unsupported image: {e}", file=sys.stderr)
        return 2
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    skel = est.estimate(bmp, a.model or "adult")
    if not skel.joints:
        print("no figure data")
        return 1
    pose = (classify.analyze(skel) or {}).get("pose")
    print(describe.describe(skel, pose))
    return 0


def _compare_doc(path: str, est, model: str):
    """image or Knowledge JSON -> doc dict (skeleton only used)."""
    if path.endswith(".json") or path.startswith("k_"):
        if path.startswith("k_"):
            path = os.path.join(
                "knowledge_store", path + ".json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "rb") as f:
        bmp = bitmap.decode(f.read())
    skel = est.estimate(bmp, model)
    return {"skeleton": skel.to_dict()}


def _cmd_compare(a) -> int:
    """Normalized pose distance between two images/docs."""
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    try:
        da = _compare_doc(a.a, est, a.model or "adult")
        db = _compare_doc(a.b, est, a.model or "adult")
    except (bitmap.UnsupportedFormat, OSError,
            json.JSONDecodeError) as e:
        print(f"cannot load input: {e}", file=sys.stderr)
        return 2
    res = compare.pose_distance(da, db,
                                min_confidence=a.min_confidence)
    if res is None:
        print("cannot normalize one or both skeletons",
              file=sys.stderr)
        return 1
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _cmd_autocrop(a) -> int:
    """Suggest (and optionally apply) a person-bbox crop."""
    try:
        with open(a.image, "rb") as f:
            bmp = bitmap.decode(f.read())
    except (bitmap.UnsupportedFormat, OSError) as e:
        print(f"cannot load {a.image}: {e}", file=sys.stderr)
        return 2
    aspect = None
    if a.aspect:
        try:
            w, h = (float(v) for v in a.aspect.split(":"))
            aspect = w / h if h else None
        except ValueError:
            print("--aspect must be W:H (e.g. 3:4)",
                  file=sys.stderr)
            return 2
    res = autocrop.suggest(bmp, margin=a.margin,
                           aspect=aspect)
    if a.output and res.get("crop"):
        rect = tuple(res["crop"])
        out = crop.crop(bmp, rect)
        with open(a.output, "wb") as f:
            f.write(bitmap.encode_png(out))
        print(f"wrote {a.output} "
              f"({out.width}x{out.height})", file=sys.stderr)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("crop") else 1


def _cmd_export(a) -> int:
    """Render a skeleton into an external format."""
    with open(a.image, "rb") as f:
        raw = f.read()
    try:
        bmp = bitmap.decode(raw)
    except bitmap.UnsupportedFormat as e:
        print(f"unsupported image: {e}", file=sys.stderr)
        return 2
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    skel = est.estimate(bmp, a.model or "adult")
    if not skel.joints:
        print("no person detected", file=sys.stderr)
        return 1
    binary = None
    if a.format == "bvh":
        text = bvh.export(skel)
    elif a.format == "gltf":
        text = json.dumps(gltf.to_gltf(skel), ensure_ascii=False)
    elif a.format == "coco":
        text = json.dumps(coco.to_coco(skel), ensure_ascii=False)
    elif a.format == "svg":
        text = svg.render(skel)
    elif a.format == "ascii":
        text = ascii.render(skel)
    elif a.format == "paf":
        text = json.dumps(paf.field(skel), ensure_ascii=False)
    elif a.format == "heatmap":
        binary = bitmap.encode_png(heatmap.render(skel))
        text = None
    if a.output:
        if binary is not None:
            with open(a.output, "wb") as f:
                f.write(binary)
        else:
            with open(a.output, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        print(f"wrote {a.output}", file=sys.stderr)
    elif binary is not None:
        sys.stdout.buffer.write(binary)
    else:
        print(text)
    return 0


def _cmd_evid(a) -> int:
    """Per-joint evidence localization: interior/boundary/off_mask."""
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    try:
        with open(a.image, "rb") as f:
            bmp = bitmap.decode(f.read())
    except (bitmap.UnsupportedFormat, OSError) as e:
        print(f"cannot load {a.image}: {e}", file=sys.stderr)
        return 2
    skel = est.estimate(bmp, a.model or "adult")
    if not skel.joints:
        print("no person detected", file=sys.stderr)
        return 1
    small = bmp.downscale(est.max_dim)
    m = est._mask(small)
    res = evid.locate(skel, m)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if not evid.unsupported(res) else 1


def _cmd_imgqual(a) -> int:
    """Image evidence adequacy before estimation."""
    try:
        with open(a.image, "rb") as f:
            bmp = bitmap.decode(f.read())
    except (bitmap.UnsupportedFormat, OSError) as e:
        print(f"cannot load {a.image}: {e}", file=sys.stderr)
        return 2
    res = imgqual.assess(bmp)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["verdict"] != "inadequate" else 1


def _cmd_human(a) -> int:
    """Person-likeness score per foreground component."""
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    try:
        with open(a.image, "rb") as f:
            bmp = bitmap.decode(f.read())
    except (bitmap.UnsupportedFormat, OSError) as e:
        print(f"cannot load {a.image}: {e}", file=sys.stderr)
        return 2
    small = bmp.downscale(est.max_dim)
    w, h = small.width, small.height
    m = est._mask(small)
    labels, sizes = est._label_components(m, w, h)
    comps = []
    for lab in sorted(sizes, key=sizes.get, reverse=True):
        if sizes[lab] < a.min_px:
            continue
        comp = est._component_mask(labels, lab, w, h)
        res = human.assess(comp)
        res["px"] = sizes[lab]
        res["component"] = lab
        comps.append(res)
    out = {"components": comps, "n_components": len(comps)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not comps:
        return 1
    return 0 if any(c["person_like"] for c in comps) else 1


def _cmd_rig(a) -> int:
    """Animation rig: bone hierarchy, lengths, directions."""
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    try:
        with open(a.image, "rb") as f:
            bmp = bitmap.decode(f.read())
    except (bitmap.UnsupportedFormat, OSError) as e:
        print(f"cannot load {a.image}: {e}", file=sys.stderr)
        return 2
    skel = est.estimate(bmp, a.model or "adult")
    if not skel.joints:
        print("no person detected", file=sys.stderr)
        return 1
    bones = rig.build(skel)
    res = {"bones": bones,
           "hierarchy": rig.hierarchy(bones),
           "total_bone_length": rig.total_bone_length(bones),
           "frame": {"width": skel.image_width,
                     "height": skel.image_height},
           "basis": "bone endpoints observed/predicted as given; "
                    "no lengths inferred"}
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"wrote {a.output}", file=sys.stderr)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _cmd_audit_store(a) -> int:
    res = storechk.audit_store(a.store)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"wrote {a.output}", file=sys.stderr)
    else:
        out = {k: res[k] for k in ("verdict", "n_docs", "blind_joints",
                                   "reasons")}
        out["reasons"] = out["reasons"][:10]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"store: {res['n_docs']} docs, verdict={res['verdict']} "
          f"({res['n_reasons']} reasons)", file=sys.stderr)
    return 0 if res["verdict"] != "fail" else 1


def _cmd_limbcov(a) -> int:
    """Bone-level silhouette coverage: bones crossing background."""
    est = (pipeline.ROBUST_ESTIMATOR if a.robust
           else pipeline.ESTIMATOR)
    try:
        with open(a.image, "rb") as f:
            bmp = bitmap.decode(f.read())
    except (bitmap.UnsupportedFormat, OSError) as e:
        print(f"cannot load {a.image}: {e}", file=sys.stderr)
        return 2
    skel = est.estimate(bmp, a.model or "adult")
    if not skel.joints:
        print("no person detected", file=sys.stderr)
        return 1
    small = bmp.downscale(est.max_dim)
    m = est._mask(small)
    res = limbcov.check(skel, m)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if res["verdict"] == "insufficient":
        return 1
    return 0 if res["verdict"] == "covered" else 1


def _cmd_calib(a) -> int:
    """Confidence calibration: measured hit rate per reported bin."""
    pairs = [evaluate.draw_case(),
             evaluate.draw_case(width=240, height=320)]
    rep = calib.report(pairs)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    # overconfidence is the dangerous direction: reporting 0.9 when the
    # empirical hit rate is 0.5. Underconfidence is merely conservative.
    # An all-empty table measured nothing — fail rather than pass mute.
    measured = any(b["n"] for b in rep["bins"])
    return 1 if rep["overconfident_bins"] or not measured else 0


def _cmd_bias(a) -> int:
    """Per-joint systematic-error profile over ground-truth fixtures."""
    est = pipeline.ESTIMATOR
    pairs = []
    for bmp, truth in (evaluate.draw_case(),
                       evaluate.draw_case(width=240, height=320)):
        pairs.append((est.estimate(bmp).joints, truth))
    rep = bias.profile(pairs)
    # an entry in joints only means a joint was observed once —
    # "insufficient" is still unmeasured. measured = at least one
    # joint cleared the sample minimum, else the gate passes on
    # zero evidence precisely when the estimator is degraded.
    worst = rep.get("worst_joint")
    rep["state"] = "estimated" if worst is not None else "unmeasured"
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    # unmeasured must not pass; a worst joint beyond the bench gate is
    # a real estimator defect
    if worst is None or worst["mean_error_px"] > 10.0:
        return 1
    return 0


def _cmd_priorchk(a) -> int:
    """Structural audit of the BODY_MODELS prior tables."""
    res = priorchk.audit()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["verdict"] == "sane" else 1


def _parse_rect(s: str):
    try:
        x0, y0, x1, y1 = (int(v) for v in s.split(","))
    except ValueError:
        raise SystemExit("rect must be x0,y0,x1,y1")
    return x0, y0, x1, y1


def _cmd_mutate(a) -> int:
    """Deterministic robustness transforms: noise/occlude/crop."""
    try:
        with open(a.image, "rb") as f:
            bmp = bitmap.decode(f.read())
    except (bitmap.UnsupportedFormat, OSError) as e:
        print(f"cannot load {a.image}: {e}", file=sys.stderr)
        return 2
    if a.noise is not None:
        out = mutate.add_noise(bmp, a.noise, seed=a.seed)
        op = {"op": "noise", "amount": a.noise, "seed": a.seed}
    elif a.occlude:
        x0, y0, x1, y1 = _parse_rect(a.occlude)
        out = mutate.occlude(bmp, x0, y0, x1, y1)
        op = {"op": "occlude", "rect": [x0, y0, x1, y1]}
    elif a.crop:
        x0, y0, x1, y1 = _parse_rect(a.crop)
        out = mutate.crop(bmp, x0, y0, x1, y1)
        op = {"op": "crop", "rect": [x0, y0, x1, y1]}
    else:
        print("specify --noise N, --occlude or --crop",
              file=sys.stderr)
        return 2
    with open(a.output, "wb") as f:
        f.write(bitmap.encode_png(out))
    print(json.dumps({**op, "output": a.output,
                      "size": [out.width, out.height]},
                     ensure_ascii=False))
    return 0


def _cmd_audit_dir(a) -> int:
    """Audit every image under a directory; print a verdict per file."""
    import os
    exts = (".png", ".bmp", ".jpg", ".jpeg", ".webp")
    paths = sorted(os.path.join(a.image, f) for f in os.listdir(a.image)
                   if f.lower().endswith(exts)
                   and os.path.isfile(os.path.join(a.image, f)))
    if not paths:
        print(f"no images under {a.image}", file=sys.stderr)
        return 1
    tally = {"pass": 0, "warn": 0, "fail": 0}
    results = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                raw = f.read()
            res = selfcheck.run(raw, model=a.model, source_name=p,
                                robust=a.robust)
            tally[res["verdict"]] += 1
            results.append({"image": p, **res})
            print(f"{res['verdict']:>4}  {p}  "
                  f"{','.join(res['reasons'][:3])}")
        except (bitmap.UnsupportedFormat, OSError, ValueError) as e:
            tally["fail"] += 1
            results.append({"image": p, "verdict": "fail",
                            "reasons": [str(e)]})
            print(f"fail  {p}  {e}", file=sys.stderr)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump({"results": results, "tally": tally}, f,
                      ensure_ascii=False, indent=2)
        print(f"wrote {a.output}", file=sys.stderr)
    print(f"audit: {tally['pass']} pass / {tally['warn']} warn / "
          f"{tally['fail']} fail ({len(paths)} images)",
          file=sys.stderr)
    return 0 if not tally["fail"] else 1


def _cmd_serve(a) -> int:
    rest.serve(port=a.port, store_dir=a.store, token=a.token)
    return 0


def _cmd_list(a) -> int:
    for item in knowledge.KnowledgeStore(a.store).list():
        print(item["id"], item["created_at"], item.get("body_model"))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="slice", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze one image")
    a.add_argument("image")
    a.add_argument("-o", "--output")
    a.add_argument("--model", choices=sorted(BODY_MODELS), default=None,
                   help="body model to force (default: estimator's own pick)")
    a.add_argument("--robust", action="store_true",
                   help="robust profile: adaptive threshold, shadow "
                        "rejection, mask cleanup (for real photos)")
    a.add_argument("--overlay")
    a.add_argument("--store")
    a.add_argument("--multi", action="store_true",
                   help="detect every foreground person (JSON array out)")
    a.set_defaults(fn=_cmd_analyze)

    b = sub.add_parser("batch", help="analyze a directory of images")
    b.add_argument("dir")
    b.add_argument("--store", required=True,
                   help="KnowledgeStore directory to write into")
    b.add_argument("--model", choices=sorted(BODY_MODELS), default=None)
    b.add_argument("-r", "--recursive", action="store_true")
    b.add_argument("--robust", action="store_true",
                   help="robust estimation profile")
    b.set_defaults(fn=_cmd_batch)

    au = sub.add_parser("audit", help="run all quality layers on one image")
    au.add_argument("image", nargs="?",
                    help="image file or directory (a store dir needs --store)")
    au.add_argument("--store",
                    help="audit a KnowledgeStore directory instead "
                         "of an image")
    au.add_argument("--model", choices=sorted(BODY_MODELS), default=None)
    au.add_argument("--robust", action="store_true",
                    help="robust estimation profile")
    au.add_argument("-o", "--output",
                    help="write the full audit JSON (all layers + doc)")
    au.set_defaults(fn=_cmd_audit)

    be = sub.add_parser(
        "bench", help="accuracy/latency gate on fixtures")
    be.add_argument("--repeats", type=int, default=1,
                    help="latency repeats per case")
    be.add_argument("--json", action="store_true",
                    help="print the raw report JSON")
    be.set_defaults(fn=_cmd_bench)

    di = sub.add_parser("diff",
                        help="diff two images or stored documents")
    di.add_argument("a", help="image path or k_<id>")
    di.add_argument("b", help="image path or k_<id>")
    di.add_argument("--store", default="knowledge",
                    help="KnowledgeStore dir for id args")
    di.add_argument("--model", choices=sorted(BODY_MODELS),
                    default=None)
    di.add_argument("--robust", action="store_true",
                    help="robust estimation profile (image args only)")
    di.add_argument("-o", "--output",
                    help="write the diff JSON to a file")
    di.set_defaults(fn=_cmd_diff)

    pr = sub.add_parser(
        "probe", help="run one semantic layer on an image")
    pr.add_argument("layer", choices=sorted(_PROBE_LAYERS))
    pr.add_argument("image")
    pr.add_argument("--model", choices=sorted(BODY_MODELS),
                    default=None)
    pr.add_argument("--robust", action="store_true",
                    help="robust estimation profile")
    pr.set_defaults(fn=_cmd_probe)
    mi = sub.add_parser(
        "mirror",
        help="estimator left/right consistency audit")
    mi.add_argument("image")
    mi.add_argument("--model", choices=sorted(BODY_MODELS),
                    default=None)
    mi.add_argument("--robust", action="store_true",
                    help="robust estimation profile")
    mi.add_argument("--tolerance", type=float, default=4.0,
                    help="max per-joint drift px for 'symmetric'")
    mi.set_defaults(fn=_cmd_mirror)

    ok = sub.add_parser(
        "oks", help="COCO OKS similarity between two images")
    ok.add_argument("a", help="reference image")
    ok.add_argument("b", help="candidate image")
    ok.add_argument("--model", choices=sorted(BODY_MODELS),
                    default=None)
    ok.add_argument("--robust", action="store_true",
                    help="robust estimation profile")
    ok.set_defaults(fn=_cmd_oks)

    cd = sub.add_parser(
        "contrad", help="cross-layer contradiction audit")
    cd.add_argument("image")
    cd.add_argument("--model", choices=sorted(BODY_MODELS),
                    default=None)
    cd.add_argument("--robust", action="store_true",
                    help="robust estimation profile")
    cd.set_defaults(fn=_cmd_contrad)

    ri = sub.add_parser(
        "reid", help="same-person check between two images")
    ri.add_argument("a")
    ri.add_argument("b")
    ri.add_argument("--model", choices=sorted(BODY_MODELS),
                    default=None)
    ri.add_argument("--robust", action="store_true",
                    help="robust estimation profile")
    ri.add_argument("--threshold", type=float,
                    default=reid.DEFAULT_THRESHOLD,
                    help="mean |feature diff| below this = same person")
    ri.set_defaults(fn=_cmd_reid)

    d = sub.add_parser("describe",
                       help="describe one image's figure in a sentence")
    d.add_argument("image")
    d.add_argument("--model", choices=sorted(BODY_MODELS), default=None)
    d.add_argument("--robust", action="store_true",
                   help="robust estimation profile")
    d.set_defaults(fn=_cmd_describe)

    cp = sub.add_parser(
        "compare",
        help="pose distance between two images/docs")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--model", choices=sorted(BODY_MODELS),
                    default=None)
    cp.add_argument("--robust", action="store_true",
                    help="robust estimation profile")
    cp.add_argument("--min-confidence", type=float, default=0.0)
    cp.set_defaults(fn=_cmd_compare)

    ac = sub.add_parser(
        "autocrop", help="person-bbox crop suggestion")
    ac.add_argument("image")
    ac.add_argument("--margin", type=float, default=0.1,
                    help="pad fraction around the bbox")
    ac.add_argument("--aspect",
                    help="target aspect as W:H (e.g. 3:4)")
    ac.add_argument("-o", "--output",
                    help="write the cropped PNG here")
    ac.set_defaults(fn=_cmd_autocrop)

    e = sub.add_parser("export",
                       help="render a skeleton into an external format")
    e.add_argument("image")
    e.add_argument("--format", required=True,
                   choices=["bvh", "gltf", "coco", "svg", "ascii",
                            "paf", "heatmap"])
    e.add_argument("--model", choices=sorted(BODY_MODELS), default=None)
    e.add_argument("--robust", action="store_true",
                   help="robust estimation profile")
    e.add_argument("-o", "--output",
                   help="output file (default: stdout)")
    e.set_defaults(fn=_cmd_export)

    ev = sub.add_parser(
        "evid", help="per-joint evidence localization")
    ev.add_argument("image")
    ev.add_argument("--model", choices=sorted(BODY_MODELS),
                    default=None)
    ev.add_argument("--robust", action="store_true",
                    help="robust estimation profile")
    ev.set_defaults(fn=_cmd_evid)

    iq = sub.add_parser(
        "imgqual", help="image evidence adequacy")
    iq.add_argument("image")
    iq.set_defaults(fn=_cmd_imgqual)

    hu = sub.add_parser(
        "human", help="person-likeness per component")
    hu.add_argument("image")
    hu.add_argument("--min-px", type=int, default=100,
                    help="min component pixels")
    hu.add_argument("--robust", action="store_true",
                    help="robust estimation profile")
    hu.set_defaults(fn=_cmd_human)

    rg = sub.add_parser(
        "rig", help="animation rig export (bones + hierarchy)")
    rg.add_argument("image")
    rg.add_argument("--model", choices=sorted(BODY_MODELS),
                    default=None)
    rg.add_argument("--robust", action="store_true",
                    help="robust estimation profile")
    rg.add_argument("-o", "--output",
                    help="write the rig JSON here")
    rg.set_defaults(fn=_cmd_rig)

    lc = sub.add_parser(
        "limbcov", help="bone coverage vs silhouette")
    lc.add_argument("image")
    lc.add_argument("--model", choices=sorted(BODY_MODELS),
                    default=None)
    lc.add_argument("--robust", action="store_true",
                    help="robust estimation profile")
    lc.set_defaults(fn=_cmd_limbcov)

    cb = sub.add_parser(
        "calib", help="confidence calibration vs ground truth")
    cb.set_defaults(fn=_cmd_calib)

    bi = sub.add_parser(
        "bias", help="per-joint systematic vs random error profile")
    bi.set_defaults(fn=_cmd_bias)

    pc = sub.add_parser(
        "priorchk", help="BODY_MODELS structural audit")
    pc.set_defaults(fn=_cmd_priorchk)

    mu = sub.add_parser(
        "mutate", help="robustness transforms (noise/occlude/crop)")
    mu.add_argument("image")
    mu.add_argument("--noise", type=int, default=None,
                    help="uniform noise amplitude (seeded)")
    mu.add_argument("--occlude", metavar="X0,Y0,X1,Y1",
                    help="black-box occlusion rect")
    mu.add_argument("--crop", metavar="X0,Y0,X1,Y1",
                    help="crop rect")
    mu.add_argument("--seed", type=int, default=0,
                    help="noise seed (deterministic)")
    mu.add_argument("-o", "--output", required=True,
                    help="output PNG path")
    mu.set_defaults(fn=_cmd_mutate)

    s = sub.add_parser("serve", help="run the REST viewer server")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--store", default="knowledge")
    s.add_argument("--token", default=None,
                   help="require 'Authorization: Bearer TOKEN' on API "
                        "routes (default: SLICE_TOKEN env, else open)")
    s.set_defaults(fn=_cmd_serve)

    l = sub.add_parser("list", help="list stored knowledge")
    l.add_argument("--store", default="knowledge")
    l.set_defaults(fn=_cmd_list)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
