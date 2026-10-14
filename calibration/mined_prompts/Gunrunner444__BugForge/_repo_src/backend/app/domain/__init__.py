"""Vendor-neutral domain models for BugForge's extensible architecture.

This package must not import third-party tool SDKs (Burp, HackerOne, OpenAI, ZAP).
Adapters translate between those systems and these types.
"""

from app.domain.evidence import (
    VERIFICATION_PROVENANCE,
    Evidence,
    EvidenceBundle,
    EvidenceKind,
    EvidenceProvenance,
    EvidenceSource,
)
from app.domain.findings import (
    FindingStatus,
    HumanReviewState,
    SecurityFinding,
    SourceLocation,
)
from app.domain.http import HttpBodyMeta, HttpExchange, HttpHeader
from app.domain.language import LanguageCapability, LanguageStats
from app.domain.reports import ReportAsset, SecurityReport, VulnerabilityReportDraft
from app.domain.scope import ScopeConstraint
from app.domain.security import EvidenceTier, VulnerabilityClass
from app.domain.source import (
    LanguageParseResult,
    ParsedEntity,
    ParsedImport,
    ParsedParameter,
)

__all__ = [
    "VERIFICATION_PROVENANCE",
    "Evidence",
    "EvidenceBundle",
    "EvidenceKind",
    "EvidenceProvenance",
    "EvidenceSource",
    "FindingStatus",
    "HumanReviewState",
    "HttpBodyMeta",
    "HttpExchange",
    "HttpHeader",
    "LanguageCapability",
    "LanguageParseResult",
    "LanguageStats",
    "ParsedEntity",
    "ParsedImport",
    "ParsedParameter",
    "ReportAsset",
    "ScopeConstraint",
    "SecurityFinding",
    "SecurityReport",
    "SourceLocation",
    "EvidenceTier",
    "VulnerabilityClass",
    "VulnerabilityReportDraft",
]
