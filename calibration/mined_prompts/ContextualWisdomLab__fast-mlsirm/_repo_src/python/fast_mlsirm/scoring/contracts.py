"""Compatibility composition for public scoring contracts.

The implementation is split into bounded canonicalization, assessment-policy,
execution-contract, and engine-authorization modules. This file preserves one
stable internal import surface for the package namespace and future scoring
components.
"""

from . import execution as _execution
from ._execution_integer_safety import install as _install_execution_integer_safety

_install_execution_integer_safety(_execution)

from ._contract_safety import artifact_digest as artifact_digest
from ._contract_safety import canonical_json as canonical_json
from ._validation import ASSESSMENT_SCHEMA_VERSION as ASSESSMENT_SCHEMA_VERSION
from ._validation import (
    MAX_METADATA_COLLECTION_VALUES as MAX_METADATA_COLLECTION_VALUES,
)
from ._validation import MAX_METADATA_DEPTH as MAX_METADATA_DEPTH
from ._validation import MAX_METADATA_NODES as MAX_METADATA_NODES
from ._validation import AssessmentSpecError as AssessmentSpecError
from .assessment import AssessmentResponseType as AssessmentResponseType
from .assessment import AssessmentSpec as AssessmentSpec
from .assessment import ConstructSpec as ConstructSpec
from .assessment import build_assessment_spec as build_assessment_spec
from .authorization import StaticFixtureEngine as StaticFixtureEngine
from .authorization import build_scoring_request as build_scoring_request
from .authorization import build_scoring_result as build_scoring_result
from .execution import (
    LEGACY_SCORING_REQUEST_SCHEMA_VERSION as LEGACY_SCORING_REQUEST_SCHEMA_VERSION,
)
from .execution import (
    SCORING_REQUEST_SCHEMA_VERSION as SCORING_REQUEST_SCHEMA_VERSION,
)
from .execution import EngineDescriptor as EngineDescriptor
from .execution import EngineKind as EngineKind
from .execution import EvidenceReference as EvidenceReference
from .execution import EvidenceRole as EvidenceRole
from .execution import FixtureOutcome as FixtureOutcome
from .execution import ObservationGranularity as ObservationGranularity
from .execution import ObservationStatus as ObservationStatus
from .execution import ScoreObservation as ScoreObservation
from .execution import ScoringEngine as ScoringEngine
from .execution import ScoringRequest as ScoringRequest
from .execution import ScoringResult as ScoringResult
from .execution import build_engine_descriptor as build_engine_descriptor
from .execution import build_score_observation as build_score_observation
from .migrations import migrate_scoring_request_v1 as migrate_scoring_request_v1
from .policies import AdjudicationPolicy as AdjudicationPolicy
from .policies import CalibrationPolicy as CalibrationPolicy
from .policies import EnginePolicy as EnginePolicy
from .policies import MonitoringPolicy as MonitoringPolicy
from .policies import ReportingPolicy as ReportingPolicy
from .policies import ValidationPolicy as ValidationPolicy

__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "MAX_METADATA_COLLECTION_VALUES",
    "MAX_METADATA_DEPTH",
    "MAX_METADATA_NODES",
    "AdjudicationPolicy",
    "AssessmentResponseType",
    "AssessmentSpec",
    "AssessmentSpecError",
    "CalibrationPolicy",
    "ConstructSpec",
    "EngineDescriptor",
    "EngineKind",
    "EnginePolicy",
    "EvidenceReference",
    "EvidenceRole",
    "FixtureOutcome",
    "MonitoringPolicy",
    "ObservationGranularity",
    "ObservationStatus",
    "ReportingPolicy",
    "ScoreObservation",
    "ScoringEngine",
    "ScoringRequest",
    "ScoringResult",
    "StaticFixtureEngine",
    "ValidationPolicy",
    "artifact_digest",
    "build_assessment_spec",
    "build_engine_descriptor",
    "build_score_observation",
    "build_scoring_request",
    "build_scoring_result",
    "canonical_json",
]
