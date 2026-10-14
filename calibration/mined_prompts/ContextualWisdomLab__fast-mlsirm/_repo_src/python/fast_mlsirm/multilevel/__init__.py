"""Public contextual-membership contracts and crossed ``u_h`` estimation.

Contracts remain the sealed design layer. ``estimate_crossed_person_effects``
is the Rust-owned MAP estimator of multiple-membership / crossed person
effects (Fox & Glas, 2001; Browne, Goldstein, & Rasbash, 2001).
"""

from . import estimation as _estimation
from ._crossed_estimation_safety import install as _install_crossed_estimation_safety
from .contracts import (
    ContextMembership,
    ContextMembershipDesign,
    LongitudinalDesign,
    LongitudinalStateKind,
    LongitudinalStateSpec,
    MultilevelContractError,
    TemporalOccasion,
    build_context_membership,
    build_context_membership_design,
    build_longitudinal_design,
    build_longitudinal_state_spec,
    build_temporal_occasion,
)
from .estimation import CrossedPersonEffectResult

_install_crossed_estimation_safety(_estimation)
estimate_crossed_person_effects = _estimation.estimate_crossed_person_effects
weighted_contextual_effect = _estimation.weighted_contextual_effect
fit_hierarchical_longitudinal_irt = _estimation.fit_hierarchical_longitudinal_irt
fit_longitudinal_state = _estimation.fit_longitudinal_state
simulate_hierarchical_longitudinal_irt = _estimation.simulate_hierarchical_longitudinal_irt

del _install_crossed_estimation_safety, _estimation

__all__ = [
    "ContextMembership",
    "ContextMembershipDesign",
    "LongitudinalDesign",
    "LongitudinalStateKind",
    "LongitudinalStateSpec",
    "MultilevelContractError",
    "TemporalOccasion",
    "build_context_membership",
    "build_context_membership_design",
    "build_longitudinal_design",
    "build_longitudinal_state_spec",
    "build_temporal_occasion",
    "CrossedPersonEffectResult",
    "estimate_crossed_person_effects",
    "fit_hierarchical_longitudinal_irt",
    "fit_longitudinal_state",
    "simulate_hierarchical_longitudinal_irt",
    "weighted_contextual_effect",
]
