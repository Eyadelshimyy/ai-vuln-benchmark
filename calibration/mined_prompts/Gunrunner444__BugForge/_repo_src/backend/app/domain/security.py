"""Vulnerability classes used by static security analysis and later reports."""

from __future__ import annotations

from enum import StrEnum


class VulnerabilityClass(StrEnum):
    INJECTION = "injection"
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    POTENTIAL_PATH_TRAVERSAL = "potential_path_traversal"
    USER_CONTROLLED_PATH = "user_controlled_path"
    SSRF = "ssrf"
    XSS = "xss"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    POTENTIAL_UNSAFE_DESERIALIZATION = "potential_unsafe_deserialization"
    AUTHENTICATION = "authentication_flaw"
    AUTHORIZATION = "authorization_flaw"
    IDOR = "insecure_direct_object_reference"
    HARDCODED_SECRET = "hardcoded_secret"
    INSECURE_FILE_HANDLING = "insecure_file_handling"
    DYNAMIC_EXECUTION = "dangerous_dynamic_execution"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    INSECURE_CONFIGURATION = "insecure_configuration"
    UNSAFE_REDIRECT = "unsafe_redirect"
    MISSING_SECURITY_CONTROL = "missing_security_control"
    BUSINESS_LOGIC = "business_logic_risk"
    REENTRANCY = "reentrancy"
    UNSAFE_EXTERNAL_CALL = "unsafe_external_call"
    UNSAFE_ARITHMETIC = "unsafe_arithmetic"
    SIGNATURE_FLAW = "signature_flaw"
    UNSAFE_PROXY = "unsafe_proxy"
    DENIAL_OF_SERVICE = "denial_of_service"


class EvidenceTier(StrEnum):
    """How strongly a finding is supported. Phase 2 never reaches verified."""

    STATIC_INDICATOR = "static_indicator"
    AI_HYPOTHESIS = "ai_hypothesis"
    CORROBORATED = "corroborated"
    REPRODUCED = "reproduced"
    VERIFIED = "verified"
