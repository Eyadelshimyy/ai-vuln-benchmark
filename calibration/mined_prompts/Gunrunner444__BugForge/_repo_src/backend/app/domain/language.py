"""Language capability flags used by language adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass
class LanguageStats:
    language: str
    file_count: int
    percentage: float


class ParserTier(StrEnum):
    """How a file was parsed. Never present profile fallback as full AST."""

    FULL_AST = "full_ast"
    PROFILE_FALLBACK = "profile_fallback"
    DETECTION_ONLY = "detection_only"
    SPECIALIZED = "specialized"


class LanguageCapability(StrEnum):
    """What a language adapter can actually do.

    Detection-only adapters are registered so repository language statistics
    stay accurate. They must not claim PARSE, STATIC_ANALYSIS, or
    SECURITY_ANALYSIS until a real implementation exists.
    """

    DETECTION = "detection"
    SOURCE = "source"
    PARSE = "parse"
    AST = "ast"
    ENTITY_EXTRACTION = "entity_extraction"
    IMPORT_EXTRACTION = "import_extraction"
    SCOPE_ANALYSIS = "scope_analysis"
    CALL_ANALYSIS = "call_analysis"
    DATA_FLOW = "data_flow"
    SECURITY_ANALYSIS = "security_analysis"
    CODE_QUALITY = "code_quality"
    # Legacy alias used by the quality-analysis engine. Adapters that implement
    # CODE_QUALITY also expose STATIC_ANALYSIS so existing callers keep working.
    STATIC_ANALYSIS = "static_analysis"


FULL_ANALYSIS_CAPS = frozenset(
    {
        LanguageCapability.DETECTION,
        LanguageCapability.SOURCE,
        LanguageCapability.PARSE,
        LanguageCapability.AST,
        LanguageCapability.ENTITY_EXTRACTION,
        LanguageCapability.IMPORT_EXTRACTION,
        LanguageCapability.SCOPE_ANALYSIS,
        LanguageCapability.CALL_ANALYSIS,
        LanguageCapability.DATA_FLOW,
        LanguageCapability.SECURITY_ANALYSIS,
    }
)
