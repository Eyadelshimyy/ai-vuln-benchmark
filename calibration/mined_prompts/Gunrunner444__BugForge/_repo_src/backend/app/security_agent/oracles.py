"""Vulnerability-specific reproduction oracles. 'Looked interesting' is not an oracle."""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from typing import Any


class OracleType(StrEnum):
    EXACT_RESPONSE = "exact_response"
    RESPONSE_FIELD = "response_field"
    RESOURCE_IDENTITY = "resource_identity"
    AUTHORIZATION_DIFFERENCE = "authorization_difference"
    STATE_TRANSITION = "state_transition"
    INVARIANT_VIOLATION = "invariant_violation"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    SOURCE_RUNTIME_CONSISTENCY = "source_runtime_consistency"


class AuthorizationOracle(StrEnum):
    """Expected authorization behavior. A raw difference is never a vulnerability."""

    OWNER_ONLY = "owner_only"
    ROLE_REQUIRED = "role_required"
    TENANT_ISOLATION = "tenant_isolation"
    AUTHENTICATED_USER_ONLY = "authenticated_user_only"
    PUBLIC_RESOURCE = "public_resource"
    CUSTOM_EXPECTATION = "custom_expectation"


def artifact_hash(payload: str | bytes) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return sha256(data).hexdigest()


def evaluate_oracle(oracle: OracleType, *, expected: Any, actual: Any) -> bool:
    if oracle is OracleType.EXACT_RESPONSE:
        return bool(expected == actual)
    if oracle is OracleType.RESPONSE_FIELD:
        if isinstance(actual, dict) and isinstance(expected, dict):
            key = next(iter(expected))
            return bool(actual.get(key) == expected.get(key))
        return str(expected) in str(actual)
    if oracle is OracleType.RESOURCE_IDENTITY:
        return str(expected) == str(actual)
    if oracle is OracleType.AUTHORIZATION_DIFFERENCE:
        return bool(expected != actual)
    if oracle is OracleType.STATE_TRANSITION:
        return str(expected) == str(actual)
    if oracle is OracleType.INVARIANT_VIOLATION:
        return bool(actual) is False if expected is True else bool(actual != expected)
    if oracle is OracleType.SENSITIVE_DATA_EXPOSURE:
        needle = str(expected)
        return bool(needle) and needle in str(actual)
    if oracle is OracleType.SOURCE_RUNTIME_CONSISTENCY:
        return str(expected) in str(actual)
    return False
