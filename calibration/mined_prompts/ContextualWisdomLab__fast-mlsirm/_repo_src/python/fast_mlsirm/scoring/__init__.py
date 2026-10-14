"""Provider-neutral automated-scoring contracts and policy specifications."""

from . import calibration as _calibration
from . import rag as _rag
from ._calibration_validation import install as _install_calibration_validation
from ._rag_metadata_validation import install as _install_rag_metadata_validation

_install_calibration_validation(_calibration)
_install_rag_metadata_validation(_rag)

from .calibration import MAX_SCORING_FACETS_CELLS as MAX_SCORING_FACETS_CELLS
from .calibration import MAX_SCORING_FACETS_RATINGS as MAX_SCORING_FACETS_RATINGS
from .calibration import (
    ScoringFacetsCalibrationBundle as ScoringFacetsCalibrationBundle,
)
from .calibration import ScoringFacetsDesign as ScoringFacetsDesign
from .calibration import ScoringFacetsRatingRecord as ScoringFacetsRatingRecord
from .calibration import (
    build_scoring_facets_calibration_bundle as build_scoring_facets_calibration_bundle,
)
from .calibration import (
    build_scoring_facets_rating_records as build_scoring_facets_rating_records,
)
from .calibration import fit_scoring_facets_bundle as fit_scoring_facets_bundle
from .calibration import fit_scoring_facets_design as fit_scoring_facets_design
from .contracts import ASSESSMENT_SCHEMA_VERSION as ASSESSMENT_SCHEMA_VERSION
from .contracts import (
    LEGACY_SCORING_REQUEST_SCHEMA_VERSION as LEGACY_SCORING_REQUEST_SCHEMA_VERSION,
)
from .contracts import (
    SCORING_REQUEST_SCHEMA_VERSION as SCORING_REQUEST_SCHEMA_VERSION,
)
from .contracts import (
    MAX_METADATA_COLLECTION_VALUES as MAX_METADATA_COLLECTION_VALUES,
)
from .contracts import MAX_METADATA_DEPTH as MAX_METADATA_DEPTH
from .contracts import MAX_METADATA_NODES as MAX_METADATA_NODES
from .contracts import AdjudicationPolicy as AdjudicationPolicy
from .contracts import AssessmentResponseType as AssessmentResponseType
from .contracts import AssessmentSpec as AssessmentSpec
from .contracts import AssessmentSpecError as AssessmentSpecError
from .contracts import CalibrationPolicy as CalibrationPolicy
from .contracts import ConstructSpec as ConstructSpec
from .contracts import EngineDescriptor as EngineDescriptor
from .contracts import EngineKind as EngineKind
from .contracts import EnginePolicy as EnginePolicy
from .contracts import EvidenceReference as EvidenceReference
from .contracts import EvidenceRole as EvidenceRole
from .contracts import FixtureOutcome as FixtureOutcome
from .contracts import MonitoringPolicy as MonitoringPolicy
from .contracts import ObservationGranularity as ObservationGranularity
from .contracts import ObservationStatus as ObservationStatus
from .contracts import ReportingPolicy as ReportingPolicy
from .contracts import ScoreObservation as ScoreObservation
from .contracts import ScoringEngine as ScoringEngine
from .contracts import ScoringRequest as ScoringRequest
from .contracts import ScoringResult as ScoringResult
from .contracts import StaticFixtureEngine as StaticFixtureEngine
from .contracts import ValidationPolicy as ValidationPolicy
from .contracts import artifact_digest as artifact_digest
from .contracts import build_assessment_spec as build_assessment_spec
from .contracts import build_engine_descriptor as build_engine_descriptor
from .contracts import build_score_observation as build_score_observation
from .contracts import build_scoring_request as build_scoring_request
from .contracts import build_scoring_result as build_scoring_result
from .contracts import canonical_json as canonical_json
from .contracts import migrate_scoring_request_v1 as migrate_scoring_request_v1
from .item_bank import ItemBankEntry as ItemBankEntry
from .item_bank import ItemBankRelease as ItemBankRelease
from .item_bank import ItemLifecycleState as ItemLifecycleState
from .item_bank import build_item_bank_entry as build_item_bank_entry
from .item_bank import build_item_bank_release as build_item_bank_release
from .rag_evidence import RAGEvidenceRegimeLimitations as RAGEvidenceRegimeLimitations
from .rag_evidence import (
    rag_evidence_regime_limitations as rag_evidence_regime_limitations,
)

# Preserve the pinned star-import contract. Execution, authorization,
# calibration, item-bank, and RAG adapter contracts remain explicit package
# attributes imported by their documented names, but are not added to
# ``__all__`` until the next public-surface version bump.
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
    "EnginePolicy",
    "MonitoringPolicy",
    "ReportingPolicy",
    "ValidationPolicy",
    "artifact_digest",
    "build_assessment_spec",
    "canonical_json",
]
