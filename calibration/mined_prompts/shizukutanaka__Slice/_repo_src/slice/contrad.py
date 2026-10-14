"""Cross-layer contradiction check — do the modules agree?

Each semantic layer reads the skeleton independently: `classify`
labels a pose, `axis` measures body orientation, `ground` judges
support, `balance` projects the centre of mass. Nobody checks the
layers against *each other* — a figure can be labelled "lying"
while its PCA axis is vertical, or "standing" while `ground`
reports the feet airborne. Those are real estimation failures:
the layers looked at the same evidence and disagreed.

`check(layers)` takes a dict of layer outputs
(e.g. {"classify": classify.analyze(sk), "axis": axis.principal(sk),
"ground": ground.estimate(sk), "balance": balance.assess(sk)}) and
runs a small rule set. Every fired rule reports both sides' raw
values — the caller sees the disagreement, not a re-scored
average. Absent layers are skipped, never assumed.
"""

from __future__ import annotations

from typing import Dict, List

VERTICAL_MIN, VERTICAL_MAX = 60.0, 120.0  # axis angle = vertical


def _v(layers: Dict, layer: str, key: str):
    v = layers.get(layer) or {}
    return v.get(key)


def check(layers: Dict) -> Dict:
    """Run the contradiction rules over a {layer_name: output} dict.

    Returns {contradictions: [{id, detail, evidence}], verdict,
    checked} — verdict is "consistent" | "contradicted" |
    "insufficient" (no comparable layer pairs present).
    """
    out: List[Dict] = []

    def rule(rid: str, detail: str, evidence: Dict) -> None:
        out.append({"id": rid, "detail": detail, "evidence": evidence})

    pose = _v(layers, "classify", "pose")
    axis_deg = _v(layers, "axis", "angle_deg")
    contact = _v(layers, "ground", "contact")
    projected = _v(layers, "balance", "projected")
    support = _v(layers, "ground", "support_joints")

    if pose in ("lie", "lying") and axis_deg is not None \
            and VERTICAL_MIN <= axis_deg <= VERTICAL_MAX:
        rule("lying_vertical_axis",
             "classified lying but body axis is vertical",
             {"classify.pose": pose, "axis.angle_deg": axis_deg})

    if pose in ("stand", "standing") and axis_deg is not None \
            and not (VERTICAL_MIN - 30 <= axis_deg <= VERTICAL_MAX + 30):
        rule("standing_horizontal_axis",
             "classified standing but body axis is near horizontal",
             {"classify.pose": pose, "axis.angle_deg": axis_deg})

    if pose in ("stand", "standing") and contact == "airborne":
        rule("standing_airborne",
             "classified standing but feet report airborne",
             {"classify.pose": pose, "ground.contact": contact})

    if projected == "inside" and contact == "airborne" \
            and not support:
        rule("stable_without_support",
             "balance says stable but no support joint is grounded",
             {"balance.projected": projected,
              "ground.contact": contact})

    if pose == "crouch" and axis_deg is not None \
            and not (VERTICAL_MIN - 45 <= axis_deg <= VERTICAL_MAX + 45):
        rule("crouch_horizontal",
             "classified crouching but axis is horizontal",
             {"classify.pose": pose, "axis.angle_deg": axis_deg})

    comparable = sum(1 for k in ("classify", "axis", "ground",
                                 "balance") if layers.get(k))
    return {
        "contradictions": out,
        "n_contradictions": len(out),
        "verdict": ("contradicted" if out else
                    "consistent" if comparable >= 2 else
                    "insufficient"),
        "layers_checked": comparable,
        "state": "measured",
        "basis": "cross-layer rule set over supplied layer outputs",
    }
