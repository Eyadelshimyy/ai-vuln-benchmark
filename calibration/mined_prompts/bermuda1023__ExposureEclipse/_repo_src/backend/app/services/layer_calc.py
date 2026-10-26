"""Reinsurance layer calculation engine.

Pure-math service that runs deterministic loss scenarios through one or more
excess-of-loss (XOL) layers. Used by the frontend (later) to answer questions
like "if the assumed damage ratio for our coastal exposure is 12%, what does
each layer pay out, and what's our net loss to the cedent?"

Concepts (industry-standard XOL math):
    ground_up_loss = TIV × damage_ratio       (or supplied directly)
    loss_to_layer  = max(0, min(gross − deductible, limit))
    ceded_loss     = loss_to_layer × share

Layers in a stack are evaluated INDEPENDENTLY against the same gross loss
(no cumulative carry-over). The reinsurer's total payout is the sum of
``ceded_loss`` across the stack. The cedent's NET loss is the gross loss
minus that sum.

Reinstatements / annual aggregates / event vs occurrence wording are out of
scope for v1 — single-event deterministic scenarios only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class LayerTerms:
    """One XOL layer's terms. ``share`` is the reinsurer's signed line on the
    layer (0..1). ``deductible`` and ``limit`` are in the same currency as the
    gross loss the layer is evaluated against."""

    deductible: float
    limit: float
    share: float = 1.0
    name: str | None = None

    def __post_init__(self) -> None:
        if self.deductible < 0:
            raise ValueError(f"deductible must be ≥ 0, got {self.deductible}")
        if self.limit <= 0:
            raise ValueError(f"limit must be > 0, got {self.limit}")
        if not 0.0 <= self.share <= 1.0:
            raise ValueError(f"share must be in [0, 1], got {self.share}")


@dataclass(slots=True, frozen=True)
class LayerOutcome:
    """Per-layer result for one scenario."""

    name: str | None
    deductible: float
    limit: float
    share: float
    loss_to_layer: float    # gross loss hitting the layer, before share
    ceded_loss: float       # what the reinsurer pays (loss_to_layer × share)
    exhausted: bool         # true when loss_to_layer == limit


@dataclass(slots=True, frozen=True)
class ScenarioResult:
    """One deterministic loss scenario evaluated through the layer stack."""

    label: str | None
    tiv: float | None
    damage_ratio: float | None     # 0..1; None when gross_loss was supplied directly
    ground_up_loss: float
    layers: list[LayerOutcome]
    total_ceded: float             # sum of ceded_loss across the stack
    cedent_net_loss: float         # ground_up_loss − total_ceded


# ─────────────────────────── primitives ───────────────────────────


def apply_layer(gross_loss: float, terms: LayerTerms) -> LayerOutcome:
    """Run one gross loss through one XOL layer."""
    if gross_loss < 0:
        gross_loss = 0.0
    in_layer = max(0.0, min(gross_loss - terms.deductible, terms.limit))
    ceded = in_layer * terms.share
    return LayerOutcome(
        name=terms.name,
        deductible=terms.deductible,
        limit=terms.limit,
        share=terms.share,
        loss_to_layer=in_layer,
        ceded_loss=ceded,
        exhausted=in_layer >= terms.limit,
    )


def apply_stack(gross_loss: float, layers: list[LayerTerms]) -> list[LayerOutcome]:
    """Run one gross loss through every layer in the stack."""
    return [apply_layer(gross_loss, layer) for layer in layers]


def run_scenario(
    layers: list[LayerTerms],
    *,
    gross_loss: float | None = None,
    tiv: float | None = None,
    damage_ratio: float | None = None,
    label: str | None = None,
) -> ScenarioResult:
    """Evaluate one scenario. Caller supplies EITHER ``gross_loss`` OR
    (``tiv`` and ``damage_ratio``); the other set of fields is derived
    so the result records both for downstream display."""
    if gross_loss is None:
        if tiv is None or damage_ratio is None:
            raise ValueError(
                "run_scenario needs either gross_loss or (tiv + damage_ratio)"
            )
        gross_loss = tiv * damage_ratio
    elif tiv is not None and damage_ratio is None and tiv > 0:
        damage_ratio = gross_loss / tiv

    outcomes = apply_stack(gross_loss, layers)
    total_ceded = sum(o.ceded_loss for o in outcomes)
    return ScenarioResult(
        label=label,
        tiv=tiv,
        damage_ratio=damage_ratio,
        ground_up_loss=gross_loss,
        layers=outcomes,
        total_ceded=total_ceded,
        cedent_net_loss=gross_loss - total_ceded,
    )


# Convenience for "TIV sweep" — give one TIV + a list of damage ratios, get
# back the loss curve through the layer stack. Useful for plotting "if
# the damage ratio is X%, our payout is Y" charts on the frontend.
DEFAULT_SWEEP_DAMAGE_RATIOS: tuple[float, ...] = (
    0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00,
)


def run_sweep(
    layers: list[LayerTerms],
    tiv: float,
    damage_ratios: tuple[float, ...] = DEFAULT_SWEEP_DAMAGE_RATIOS,
) -> list[ScenarioResult]:
    """Run a TIV against a series of damage ratios — yields a payout curve."""
    return [
        run_scenario(
            layers,
            tiv=tiv,
            damage_ratio=dr,
            label=f"{dr * 100:g}% damage ratio",
        )
        for dr in damage_ratios
    ]
