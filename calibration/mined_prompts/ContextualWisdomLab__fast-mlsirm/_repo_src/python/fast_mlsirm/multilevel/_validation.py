"""Fail-closed validation helpers for contextual measurement contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

MULTILEVEL_SCHEMA_VERSION = "1.0"
MAX_IDENTIFIER_LENGTH = 128
MAX_SIGNED_INTEGER = (1 << 63) - 1
MIN_SIGNED_INTEGER = -(1 << 63)
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class MultilevelContractError(ValueError):
    """Stable non-reflective contextual-contract validation error."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Store machine-readable error metadata without rejected values."""
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


def contract_error(code: str, path: str, message: str) -> MultilevelContractError:
    """Return one structured contextual-contract error."""
    return MultilevelContractError(code, path, message)


def schema_version(value: Any) -> str:
    """Require the exact built-in current contextual-contract schema version."""
    if type(value) is not str or value != MULTILEVEL_SCHEMA_VERSION:
        raise contract_error(
            "invalid_schema_version",
            "$.schema_version",
            f"schema_version must be '{MULTILEVEL_SCHEMA_VERSION}'",
        )
    return MULTILEVEL_SCHEMA_VERSION


def descriptive_identifier(value: Any, name: str, path: str | None = None) -> str:
    """Return one exact built-in bounded lower-snake identifier."""
    resolved_path = path or f"$.{name}"
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > MAX_IDENTIFIER_LENGTH
        or _IDENTIFIER_PATTERN.fullmatch(value) is None
    ):
        raise contract_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must use two-or-more-token lower snake_case",
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise contract_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be valid UTF-8",
        ) from None
    return value


def fingerprint(value: Any, name: str, path: str | None = None) -> str:
    """Return one exact built-in complete lowercase SHA-256 fingerprint."""
    resolved_path = path or f"$.{name}"
    if type(value) is not str or _FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise contract_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be a 64-character lower hexadecimal digest",
        )
    return value


def strict_boolean(value: Any, name: str, path: str | None = None) -> bool:
    """Return a real Boolean without accepting integer coercion."""
    if not isinstance(value, bool):
        raise contract_error(
            f"invalid_{name}",
            path or f"$.{name}",
            f"{name} must be boolean",
        )
    return value


def exact_integer(
    value: Any,
    name: str,
    *,
    minimum: int = MIN_SIGNED_INTEGER,
    maximum: int = MAX_SIGNED_INTEGER,
    path: str | None = None,
) -> int:
    """Return one exact bounded integer while rejecting Boolean coercion."""
    resolved_path = path or f"$.{name}"
    if type(value) is not int or not minimum <= value <= maximum:
        raise contract_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be an integer in the supported range",
        )
    return value


def membership_weight(value: Any) -> float:
    """Return one finite membership weight in the interval ``(0, 1]``."""
    if type(value) not in (int, float):
        raise contract_error(
            "invalid_membership_weight",
            "$.membership_weight",
            "membership_weight must be a finite real number greater than zero and at most one",
        )
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 < normalized <= 1.0:
        raise contract_error(
            "invalid_membership_weight",
            "$.membership_weight",
            "membership_weight must be a finite real number greater than zero and at most one",
        )
    return normalized


def autoregressive_coefficient(value: Any) -> float:
    """Return one finite stationary AR(1) coefficient strictly inside unity."""
    if type(value) not in (int, float):
        raise contract_error(
            "invalid_autoregressive_coefficient",
            "$.autoregressive_coefficient",
            "autoregressive_coefficient must be finite and strictly between negative one and one",
        )
    normalized = float(value)
    if not math.isfinite(normalized) or not -1.0 < normalized < 1.0:
        raise contract_error(
            "invalid_autoregressive_coefficient",
            "$.autoregressive_coefficient",
            "autoregressive_coefficient must be finite and strictly between negative one and one",
        )
    return normalized


def bounded_values(
    values: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> tuple[Any, ...]:
    """Materialize one bounded iterable without accepting text as a collection."""
    path = f"$.{name}"
    if isinstance(values, (str, bytes, bytearray)):
        raise contract_error(
            f"invalid_{name}",
            path,
            f"{name} must be a bounded collection",
        )
    try:
        iterator = iter(values)
    except (TypeError, ValueError, OverflowError):
        raise contract_error(
            f"invalid_{name}",
            path,
            f"{name} must be a bounded collection",
        ) from None
    output: list[Any] = []
    try:
        for index, value in enumerate(iterator):
            if index >= maximum:
                raise contract_error(
                    f"invalid_{name}",
                    path,
                    f"{name} exceeds the configured collection bound",
                )
            output.append(value)
    except MultilevelContractError:
        raise
    except (TypeError, ValueError, OverflowError):
        raise contract_error(
            f"invalid_{name}",
            path,
            f"{name} could not be materialized safely",
        ) from None
    if len(output) < minimum:
        raise contract_error(
            f"invalid_{name}",
            path,
            f"{name} does not contain enough values",
        )
    return tuple(output)


def canonical_json(value: Any) -> str:
    """Return deterministic compact UTF-8 JSON for normalized contract content."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError):
        raise contract_error(
            "invalid_canonical_content",
            "$",
            "contract content could not be serialized canonically",
        ) from None


def artifact_digest(content: Any) -> str:
    """Return SHA-256 over deterministic normalized contract content."""
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()


__all__ = [
    "MAX_SIGNED_INTEGER",
    "MIN_SIGNED_INTEGER",
    "MULTILEVEL_SCHEMA_VERSION",
    "MultilevelContractError",
    "artifact_digest",
    "autoregressive_coefficient",
    "bounded_values",
    "canonical_json",
    "contract_error",
    "descriptive_identifier",
    "exact_integer",
    "fingerprint",
    "membership_weight",
    "schema_version",
    "strict_boolean",
]
