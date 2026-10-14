"""Explicit discovery-engine capabilities. Missing binaries stay unavailable."""

from __future__ import annotations

from enum import StrEnum


class EngineCapability(StrEnum):
    PLANNING_ONLY = "planning_only"
    STATIC_ANALYSIS = "static_analysis"
    FUZZING = "fuzzing"
    SYMBOLIC_EXECUTION = "symbolic_execution"
    PROPERTY_TESTING = "property_testing"
    INVARIANT_TESTING = "invariant_testing"
    COVERAGE_FEEDBACK = "coverage_feedback"
    SANITIZER_AWARE = "sanitizer_aware"
    RESULTS_INGESTION = "results_ingestion"
    BUILD = "build"
    TEST_EXECUTION = "test_execution"
    ECONOMIC_SIMULATION = "economic_simulation"
    CROSS_CONTRACT_ANALYSIS = "cross_contract_analysis"
    RUNTIME_VALIDATION = "runtime_validation"
    FORK_VALIDATION = "fork_validation"
    DIFFERENTIAL_VALIDATION = "differential_validation"
    BOUNTY_CONTEXT_ANALYSIS = "bounty_context_analysis"
    CALLER_CONTEXT_ANALYSIS = "caller_context_analysis"
    ORACLE_QUALITY_ANALYSIS = "oracle_quality_analysis"
    PROOF_BINDING_ANALYSIS = "proof_binding_analysis"
    ACCOUNT_ABSTRACTION_ANALYSIS = "account_abstraction_analysis"
    ACCOUNTING_ANALYSIS = "accounting_analysis"
    COMPILER_ADVISORY_ANALYSIS = "compiler_advisory_analysis"
    COMPILER_DIFFERENTIAL_VALIDATION = "compiler_differential_validation"
    VFCS_GENERATION = "vfcs_generation"


CAPABILITY_ORDER: tuple[EngineCapability, ...] = (
    EngineCapability.STATIC_ANALYSIS,
    EngineCapability.CROSS_CONTRACT_ANALYSIS,
    EngineCapability.ECONOMIC_SIMULATION,
    EngineCapability.RUNTIME_VALIDATION,
    EngineCapability.FORK_VALIDATION,
    EngineCapability.DIFFERENTIAL_VALIDATION,
    EngineCapability.BOUNTY_CONTEXT_ANALYSIS,
    EngineCapability.CALLER_CONTEXT_ANALYSIS,
    EngineCapability.ORACLE_QUALITY_ANALYSIS,
    EngineCapability.PROOF_BINDING_ANALYSIS,
    EngineCapability.ACCOUNT_ABSTRACTION_ANALYSIS,
    EngineCapability.ACCOUNTING_ANALYSIS,
    EngineCapability.COMPILER_ADVISORY_ANALYSIS,
    EngineCapability.COMPILER_DIFFERENTIAL_VALIDATION,
    EngineCapability.VFCS_GENERATION,
    EngineCapability.FUZZING,
    EngineCapability.SYMBOLIC_EXECUTION,
    EngineCapability.TEST_EXECUTION,
    EngineCapability.PROPERTY_TESTING,
    EngineCapability.INVARIANT_TESTING,
    EngineCapability.COVERAGE_FEEDBACK,
    EngineCapability.BUILD,
    EngineCapability.SANITIZER_AWARE,
    EngineCapability.RESULTS_INGESTION,
    EngineCapability.PLANNING_ONLY,
)
"""Fixed preference used wherever one capability must be named from a set."""


class EngineAvailability(StrEnum):
    UNAVAILABLE = "unavailable"
    AVAILABLE = "available"


class ResultStatus(StrEnum):
    """Execution truth. Planned is not executed, and unavailable is not a result."""

    PLANNED = "planned"
    UNAVAILABLE = "unavailable"
    NOT_IMPLEMENTED = "not_implemented"
    EXECUTED = "executed"
    INGESTED = "ingested"
    INTERESTING = "interesting"
    FAILED = "failed"
    TIMEOUT = "timeout"
    TOOL_FAILURE = "tool_failure"
    UNSUPPORTED = "unsupported"
