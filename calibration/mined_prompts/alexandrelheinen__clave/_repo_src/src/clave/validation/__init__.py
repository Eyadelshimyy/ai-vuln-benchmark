"""Validation metrics, the named confusions, and the gates that decide a pass.

Every metric here is a pure function of recorded outcomes. Nothing in this
package loads a model, steps a simulator, or reads a corpus, so a number can be
reproduced from a records file alone.

The record type is a placeholder. v0.6.0 owns rollout capture and supersedes it,
and the metric functions are written to survive that handover unchanged.
"""

from clave.validation.confusions import (
    NAMED_CONFUSIONS,
    ConfusionResult,
    NamedConfusion,
    confusion_rates,
)
from clave.validation.gates import (
    GateConfig,
    GateConfigError,
    GateOutcome,
    GateThresholds,
    evaluate,
)
from clave.validation.metrics import (
    ConfusionMatrix,
    MetricError,
    RoutingCounts,
    TimingSummary,
    ValidationSummary,
    default_channel_map,
    percentile,
)
from clave.validation.outcomes import (
    ObjectOutcome,
    OutcomeError,
    OutcomeSet,
    load_outcomes,
)
from clave.validation.report import ValidationReport

__all__ = [
    "NAMED_CONFUSIONS",
    "ConfusionMatrix",
    "ConfusionResult",
    "GateConfig",
    "GateConfigError",
    "GateOutcome",
    "GateThresholds",
    "MetricError",
    "NamedConfusion",
    "ObjectOutcome",
    "OutcomeError",
    "OutcomeSet",
    "RoutingCounts",
    "TimingSummary",
    "ValidationReport",
    "ValidationSummary",
    "confusion_rates",
    "default_channel_map",
    "evaluate",
    "load_outcomes",
    "percentile",
]
