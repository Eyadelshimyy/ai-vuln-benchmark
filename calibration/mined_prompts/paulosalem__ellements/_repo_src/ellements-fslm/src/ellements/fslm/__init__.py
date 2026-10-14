"""Agentic finite-state linguistic machines for ellements."""

from __future__ import annotations

from . import det, nl
from .context import FSLMContext
from .definition import MachineDefinition, RuntimeBindings
from .dsl import MachineBuilder, StateBuilder, TransitionBuilder, machine
from .errors import (
    BudgetExhaustedError,
    FSLMError,
    InvalidTransitionError,
    InvariantViolationError,
    SafetyBlockedError,
    SpecValidationError,
)
from .evaluators import LLMDecisionEvaluator
from .kernel import ActionExecutor, FSLMKernel, GuardEvaluator, InvariantChecker
from .loading import load_machine_definition, load_machine_spec
from .models import (
    ActionResult,
    ActionSpec,
    AffordanceSpec,
    BudgetSpec,
    ConfidencePolicy,
    DecisionResult,
    EffectSpec,
    EventPatternSpec,
    ExecutionPolicy,
    FSLMEvent,
    GuardSpec,
    InvariantSpec,
    MachineSnapshot,
    MachineSpec,
    OutputRecord,
    OutputSpec,
    StateSpec,
    StepResult,
    ToolRef,
    TransitionSpec,
)
from .observers import (
    FSLMEventRecord,
    FSLMObserver,
    JsonlFSLMObserver,
    RecordingFSLMObserver,
    RichConsoleFSLMObserver,
)
from .persistence import InMemoryFSLMStore, LocalFSLMStore
from .rendering import make_console, render_snapshot, render_step, render_validation
from .visualization import to_mermaid

__all__ = [
    "ActionExecutor",
    "ActionResult",
    "ActionSpec",
    "AffordanceSpec",
    "BudgetExhaustedError",
    "BudgetSpec",
    "ConfidencePolicy",
    "DecisionResult",
    "EventPatternSpec",
    "ExecutionPolicy",
    "EffectSpec",
    "FSLMContext",
    "FSLMError",
    "FSLMEvent",
    "FSLMEventRecord",
    "FSLMKernel",
    "FSLMObserver",
    "GuardEvaluator",
    "GuardSpec",
    "InMemoryFSLMStore",
    "InvalidTransitionError",
    "InvariantChecker",
    "InvariantSpec",
    "InvariantViolationError",
    "JsonlFSLMObserver",
    "LocalFSLMStore",
    "LLMDecisionEvaluator",
    "MachineBuilder",
    "MachineDefinition",
    "MachineSnapshot",
    "MachineSpec",
    "OutputRecord",
    "OutputSpec",
    "RecordingFSLMObserver",
    "RichConsoleFSLMObserver",
    "RuntimeBindings",
    "SafetyBlockedError",
    "SpecValidationError",
    "StateBuilder",
    "StateSpec",
    "StepResult",
    "ToolRef",
    "TransitionBuilder",
    "TransitionSpec",
    "det",
    "load_machine_definition",
    "machine",
    "make_console",
    "load_machine_spec",
    "nl",
    "render_snapshot",
    "render_step",
    "render_validation",
    "to_mermaid",
]
