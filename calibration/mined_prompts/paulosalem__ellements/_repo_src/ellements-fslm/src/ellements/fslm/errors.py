"""Exception hierarchy for `ellements.fslm`."""

from __future__ import annotations


class FSLMError(Exception):
    """Base error for FSLM runtime and specification failures."""


class SpecValidationError(FSLMError, ValueError):
    """Raised when a machine specification is invalid."""


class InvalidTransitionError(FSLMError):
    """Raised when a transition cannot be applied."""


class InvariantViolationError(FSLMError):
    """Raised when state invariants fail in strict contexts."""


class SafetyBlockedError(FSLMError):
    """Raised when an action is blocked by safety policy."""


class BudgetExhaustedError(FSLMError):
    """Raised when an FSLM run exceeds a declared budget."""


__all__ = [
    "BudgetExhaustedError",
    "FSLMError",
    "InvalidTransitionError",
    "InvariantViolationError",
    "SafetyBlockedError",
    "SpecValidationError",
]

