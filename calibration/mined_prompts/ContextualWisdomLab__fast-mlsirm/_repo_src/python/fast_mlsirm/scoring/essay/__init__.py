"""Essay-specific adapters for the shared provider-neutral scoring contracts."""

from . import contracts as _contracts
from ._integer_safety import install as _install_integer_safety

_install_integer_safety(_contracts)
del _contracts, _install_integer_safety

from .calibration_reporting import (
    MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS as MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS,
)
from .calibration_reporting import (
    EssayFacetsCalibrationReport as EssayFacetsCalibrationReport,
)
from .calibration_reporting import (
    build_essay_facets_calibration_report as build_essay_facets_calibration_report,
)
from .calibration_reporting import (
    fit_essay_facets_calibration_report as fit_essay_facets_calibration_report,
)
from .calibration_report_html import (
    render_essay_facets_calibration_report_html as render_essay_facets_calibration_report_html,
)
from .contracts import EssayEvidenceKind as EssayEvidenceKind
from .contracts import EssayPrompt as EssayPrompt
from .contracts import EssayResponseEvidence as EssayResponseEvidence
from .contracts import EssayReviewFlag as EssayReviewFlag
from .contracts import EssayScoringRequest as EssayScoringRequest
from .contracts import EssaySubmission as EssaySubmission
from .contracts import (
    MAX_ESSAY_EVIDENCE_REFERENCES as MAX_ESSAY_EVIDENCE_REFERENCES,
)
from .contracts import (
    MAX_ESSAY_RESPONSE_CHARACTERS as MAX_ESSAY_RESPONSE_CHARACTERS,
)
from .contracts import MAX_ESSAY_RESPONSE_UNITS as MAX_ESSAY_RESPONSE_UNITS
from .contracts import MAX_ESSAY_REVIEW_FLAGS as MAX_ESSAY_REVIEW_FLAGS
from .contracts import build_essay_prompt as build_essay_prompt
from .contracts import (
    build_essay_response_evidence as build_essay_response_evidence,
)
from .contracts import build_essay_scoring_request as build_essay_scoring_request
from .contracts import build_essay_submission as build_essay_submission
from .contracts import score_essay_request as score_essay_request
from .report_html import (
    render_essay_score_report_html as render_essay_score_report_html,
)
from .reporting import EssayScoreReport as EssayScoreReport
from .reporting import (
    MAX_ESSAY_REPORT_REVIEW_TRIGGERS as MAX_ESSAY_REPORT_REVIEW_TRIGGERS,
)
from .reporting import build_essay_score_report as build_essay_score_report
from .validation_report_html import (
    render_essay_validation_evidence_report_html as render_essay_validation_evidence_report_html,
)
from .validation_reporting import (
    MAX_ESSAY_VALIDATION_REVIEW_TRIGGERS as MAX_ESSAY_VALIDATION_REVIEW_TRIGGERS,
)
from .validation_reporting import (
    EssayValidationEvidenceReport as EssayValidationEvidenceReport,
)
from .validation_reporting import EssayValidationMetric as EssayValidationMetric
from .validation_reporting import (
    build_essay_validation_evidence_report as build_essay_validation_evidence_report,
)

__all__ = [
    "MAX_ESSAY_EVIDENCE_REFERENCES",
    "MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS",
    "MAX_ESSAY_REPORT_REVIEW_TRIGGERS",
    "MAX_ESSAY_RESPONSE_CHARACTERS",
    "MAX_ESSAY_RESPONSE_UNITS",
    "MAX_ESSAY_REVIEW_FLAGS",
    "MAX_ESSAY_VALIDATION_REVIEW_TRIGGERS",
    "EssayEvidenceKind",
    "EssayFacetsCalibrationReport",
    "EssayPrompt",
    "EssayResponseEvidence",
    "EssayReviewFlag",
    "EssayScoreReport",
    "EssayScoringRequest",
    "EssaySubmission",
    "EssayValidationEvidenceReport",
    "EssayValidationMetric",
    "build_essay_facets_calibration_report",
    "build_essay_prompt",
    "build_essay_response_evidence",
    "build_essay_score_report",
    "build_essay_scoring_request",
    "build_essay_submission",
    "build_essay_validation_evidence_report",
    "fit_essay_facets_calibration_report",
    "render_essay_facets_calibration_report_html",
    "render_essay_score_report_html",
    "render_essay_validation_evidence_report_html",
    "score_essay_request",
]