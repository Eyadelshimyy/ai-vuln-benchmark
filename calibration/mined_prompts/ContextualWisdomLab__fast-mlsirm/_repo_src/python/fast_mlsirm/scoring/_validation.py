"""Bounded fail-closed validation and canonicalization for scoring contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np

from fast_mlsirm.rubric.models import _identifier, _semantic_version, _text

ASSESSMENT_SCHEMA_VERSION = "1.0"
MAX_METADATA_COLLECTION_VALUES = 64
MAX_METADATA_DEPTH = 8
MAX_METADATA_NODES = 1_024
MAX_ASSESSMENT_CONSTRUCTS = 32
MAX_ASSESSMENT_RUBRICS = 64
MAX_POLICY_REFERENCES = 64
MAX_RATERS_PER_RESPONSE = 64
MAX_METADATA_KEY_LENGTH = 128
MAX_METADATA_TEXT_LENGTH = 8_192
MAX_CANONICAL_JSON_CHARACTERS = 262_144
MAX_ERROR_PATH_LENGTH = 256
MAX_ERROR_MESSAGE_LENGTH = 512
MIN_SIGNED_INTEGER = -(1 << 63)
MAX_SIGNED_INTEGER = (1 << 63) - 1
FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ERROR_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_NUMPY_INTEGER_SCALAR_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
)
_SENSITIVE_METADATA_FIELDS = frozenset(
    {
        "answer_text",
        "essay_text",
        "prompt_text",
        "raw_response",
        "response_content",
        "response_text",
        "source_content",
        "source_text",
    }
)

EnumValue = TypeVar("EnumValue", bound=Enum)


class AssessmentSpecError(ValueError):
    """Stable redacted assessment-contract validation error."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Store bounded machine metadata without caller-controlled values."""
        if type(code) is not str or _ERROR_CODE_PATTERN.fullmatch(code) is None:
            raise ValueError("code must use two-or-more-token lower snake_case")
        if type(path) is not str or not path.startswith("$"):
            raise ValueError("path must begin with '$'")
        if len(path) > MAX_ERROR_PATH_LENGTH:
            raise ValueError(
                f"path must contain at most {MAX_ERROR_PATH_LENGTH} characters"
            )
        if not path.isprintable():
            raise ValueError("path must not contain control characters")
        if type(message) is not str or not message.strip():
            raise ValueError("message must not be empty")
        if len(message) > MAX_ERROR_MESSAGE_LENGTH:
            raise ValueError(
                f"message must contain at most {MAX_ERROR_MESSAGE_LENGTH} characters"
            )
        if not message.isprintable():
            raise ValueError("message must not contain control characters")
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


def assessment_error(code: str, path: str, message: str) -> AssessmentSpecError:
    """Return one structured non-reflective scoring-contract error."""
    return AssessmentSpecError(code, path, message)


class CanonicalContract:
    """Marker base for package-owned contracts with authoritative content."""

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical contract content without derived identities."""
        raise NotImplementedError


def _require_utf8(value: str, path: str) -> str:
    """Reject lone surrogates before serialization or digest encoding."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise assessment_error(
            "invalid_utf8_text",
            path,
            "text must be valid UTF-8",
        ) from None
    return value


def descriptive_identifier(value: Any, name: str, path: str | None = None) -> str:
    """Return one descriptive lower-snake identifier or a domain error."""
    resolved_path = path or f"$.{name}"
    try:
        normalized = _identifier(value, name)
    except (TypeError, ValueError, OverflowError):
        raise assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must use two-or-more-token lower snake_case",
        ) from None
    return _require_utf8(normalized, resolved_path)


def bounded_text(
    value: Any,
    name: str,
    *,
    maximum: int = MAX_METADATA_TEXT_LENGTH,
    path: str | None = None,
) -> str:
    """Return bounded non-empty text or a stable domain error."""
    resolved_path = path or f"$.{name}"
    try:
        normalized = _text(value, name, maximum=maximum)
    except (TypeError, ValueError, OverflowError):
        raise assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be non-empty text containing at most {maximum} characters",
        ) from None
    return _require_utf8(normalized, resolved_path)


def semantic_version(value: Any, name: str, path: str | None = None) -> str:
    """Return a canonical semantic version or a stable domain error."""
    resolved_path = path or f"$.{name}"
    try:
        return _semantic_version(value, name)
    except (TypeError, ValueError, OverflowError):
        raise assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be a canonical semantic version",
        ) from None


def assessment_schema_version(value: Any) -> str:
    """Require the exact built-in assessment wire-schema version string."""
    if type(value) is not str or value != ASSESSMENT_SCHEMA_VERSION:
        raise assessment_error(
            "invalid_schema_version",
            "$.schema_version",
            f"schema_version must be '{ASSESSMENT_SCHEMA_VERSION}'",
        )
    return ASSESSMENT_SCHEMA_VERSION


def enum_value(
    value: Any,
    enum_type: type[EnumValue],
    name: str,
    path: str | None = None,
) -> EnumValue:
    """Return an exact enum member or admit only inert built-in wire text."""
    value_type = type(value)
    if value_type is enum_type:
        return value
    if value_type is not str:
        raise assessment_error(
            f"invalid_{name}",
            path or f"$.{name}",
            f"{name} must be one of the supported values",
        )
    try:
        return enum_type(value)
    except (TypeError, ValueError, OverflowError):
        raise assessment_error(
            f"invalid_{name}",
            path or f"$.{name}",
            f"{name} must be one of the supported values",
        ) from None


def strict_boolean(value: Any, name: str, path: str | None = None) -> bool:
    """Return a real Boolean without accepting integer coercion."""
    if not isinstance(value, bool):
        raise assessment_error(
            f"invalid_{name}",
            path or f"$.{name}",
            f"{name} must be boolean",
        )
    return value


def _has_exact_type(value: Any, trusted_types: tuple[type, ...]) -> bool:
    """Return whether a control has one exact package-trusted scalar type."""
    value_type = type(value)
    return any(value_type is trusted_type for trusted_type in trusted_types)


def bounded_positive_integer(
    value: Any,
    name: str,
    maximum: int,
    path: str | None = None,
) -> int:
    """Return a bounded positive integer without caller-controlled coercion."""
    resolved_path = path or f"$.{name}"
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif _has_exact_type(value, _NUMPY_INTEGER_SCALAR_TYPES):
        normalized = int(value)
    else:
        raise assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be an integer between 1 and {maximum}",
        )
    if not 1 <= normalized <= maximum:
        raise assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be between 1 and {maximum}",
        )
    return normalized


def bounded_values(
    values: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
    path: str | None = None,
) -> tuple[Any, ...]:
    """Materialize a bounded iterable without accepting text as a collection."""
    resolved_path = path or f"$.{name}"
    if isinstance(values, (str, bytes, bytearray)):
        raise assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be a collection",
        )
    try:
        iterator = iter(values)
    except (TypeError, ValueError, OverflowError):
        raise assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be a collection",
        ) from None
    output: list[Any] = []
    try:
        for index, value in enumerate(iterator):
            if index >= maximum:
                raise assessment_error(
                    f"invalid_{name}",
                    resolved_path,
                    f"{name} must contain at most {maximum} values",
                )
            output.append(value)
    except AssessmentSpecError:
        raise
    except (TypeError, ValueError, OverflowError):
        raise assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} could not be materialized safely",
        ) from None
    if len(output) < minimum:
        raise assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must contain at least {minimum} value",
        )
    return tuple(output)


def fingerprint(value: Any, name: str, path: str | None = None) -> str:
    """Return a validated exact built-in lowercase SHA-256 fingerprint."""
    resolved_path = path or f"$.{name}"
    if type(value) is not str or FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be a 64-character lower hexadecimal digest",
        )
    return value


def sorted_identifiers(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
    maximum: int = MAX_POLICY_REFERENCES,
) -> tuple[str, ...]:
    """Return a bounded sorted tuple of unique descriptive identifiers."""
    raw = bounded_values(
        values,
        name,
        minimum=minimum,
        maximum=maximum,
    )
    normalized = tuple(
        descriptive_identifier(
            value,
            name,
            f"$.{name}[{index}]",
        )
        for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise assessment_error(
            f"duplicate_{name}",
            f"$.{name}",
            f"{name} must not contain duplicates",
        )
    return tuple(sorted(normalized))


def sorted_fingerprints(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
    maximum: int = MAX_ASSESSMENT_RUBRICS,
) -> tuple[str, ...]:
    """Return a bounded sorted tuple of unique SHA-256 fingerprints."""
    raw = bounded_values(
        values,
        name,
        minimum=minimum,
        maximum=maximum,
    )
    normalized = tuple(
        fingerprint(value, name, f"$.{name}[{index}]")
        for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise assessment_error(
            f"duplicate_{name}",
            f"$.{name}",
            f"{name} must not contain duplicates",
        )
    return tuple(sorted(normalized))


def _metadata_key(value: Any, path: str) -> str:
    """Return one bounded safe metadata key without reflecting its value."""
    if type(value) is not str:
        raise assessment_error(
            "invalid_metadata_key",
            path,
            "metadata keys must be strings",
        )
    if not value or value != value.strip():
        raise assessment_error(
            "invalid_metadata_key",
            path,
            "metadata keys must be non-empty and trimmed",
        )
    if len(value) > MAX_METADATA_KEY_LENGTH:
        raise assessment_error(
            "invalid_metadata_key",
            path,
            (
                "metadata keys must contain at most "
                f"{MAX_METADATA_KEY_LENGTH} characters"
            ),
        )
    if not value.isprintable():
        raise assessment_error(
            "invalid_metadata_key",
            path,
            "metadata keys must not contain control characters",
        )
    _require_utf8(value, path)
    if value in _SENSITIVE_METADATA_FIELDS:
        raise assessment_error(
            "sensitive_metadata_field",
            path,
            "metadata cannot contain response or source content fields",
        )
    return value


def _mapping_entries(value: Mapping[Any, Any], path: str) -> tuple[tuple[str, Any], ...]:
    """Validate mapping keys in index paths before deterministic sorting."""
    entries: list[tuple[str, Any]] = []
    seen: set[str] = set()
    try:
        iterator = iter(value.items())
    except (TypeError, ValueError, OverflowError):
        raise assessment_error(
            "invalid_metadata_mapping",
            path,
            "metadata mapping entries could not be inspected safely",
        ) from None
    try:
        for index, entry in enumerate(iterator):
            if index >= MAX_METADATA_COLLECTION_VALUES:
                raise assessment_error(
                    "metadata_collection_too_large",
                    path,
                    (
                        "metadata mappings must contain at most "
                        f"{MAX_METADATA_COLLECTION_VALUES} values"
                    ),
                )
            try:
                raw_key, child = entry
            except (TypeError, ValueError, OverflowError):
                raise assessment_error(
                    "invalid_metadata_mapping",
                    f"{path}.entries[{index}]",
                    "metadata mapping entries must contain one key and value",
                ) from None
            key = _metadata_key(raw_key, f"{path}.keys[{index}]")
            if key in seen:
                raise assessment_error(
                    "duplicate_metadata_key",
                    f"{path}.keys[{index}]",
                    "metadata keys must be unique",
                )
            seen.add(key)
            entries.append((key, child))
    except AssessmentSpecError:
        raise
    except (TypeError, ValueError, OverflowError):
        raise assessment_error(
            "invalid_metadata_mapping",
            path,
            "metadata mapping entries could not be materialized safely",
        ) from None
    return tuple(entries)


def freeze_json_value(
    value: Any,
    path: str,
    *,
    depth: int,
    node_count: list[int],
) -> Any:
    """Validate and deeply freeze one bounded JSON-compatible value."""
    if depth > MAX_METADATA_DEPTH:
        raise assessment_error(
            "metadata_depth_exceeded",
            path,
            f"metadata exceeds the maximum depth of {MAX_METADATA_DEPTH}",
        )
    node_count[0] += 1
    if node_count[0] > MAX_METADATA_NODES:
        raise assessment_error(
            "metadata_node_budget_exceeded",
            path,
            f"metadata exceeds the maximum node count of {MAX_METADATA_NODES}",
        )
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if not MIN_SIGNED_INTEGER <= value <= MAX_SIGNED_INTEGER:
            raise assessment_error(
                "integer_out_of_range",
                path,
                "integer metadata must fit the signed 64-bit range",
            )
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise assessment_error(
                "non_finite_metadata_number",
                path,
                "numeric metadata must be finite",
            )
        return 0.0 if value == 0.0 else value
    if type(value) is str:
        if len(value) > MAX_METADATA_TEXT_LENGTH:
            raise assessment_error(
                "metadata_text_too_long",
                path,
                (
                    "string metadata must contain at most "
                    f"{MAX_METADATA_TEXT_LENGTH} characters"
                ),
            )
        return _require_utf8(value, path)
    if isinstance(value, Mapping):
        entries = _mapping_entries(value, path)
        frozen_entries = [
            (
                key,
                freeze_json_value(
                    child,
                    f"{path}.values[{index}]",
                    depth=depth + 1,
                    node_count=node_count,
                ),
            )
            for index, (key, child) in enumerate(entries)
        ]
        return MappingProxyType(
            {key: child for key, child in sorted(frozen_entries)}
        )
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_METADATA_COLLECTION_VALUES:
            raise assessment_error(
                "metadata_collection_too_large",
                path,
                (
                    "metadata collections must contain at most "
                    f"{MAX_METADATA_COLLECTION_VALUES} values"
                ),
            )
        return tuple(
            freeze_json_value(
                entry,
                f"{path}[{index}]",
                depth=depth + 1,
                node_count=node_count,
            )
            for index, entry in enumerate(value)
        )
    raise assessment_error(
        "unsupported_metadata_value",
        path,
        "metadata contains an unsupported JSON value",
    )


def freeze_metadata(value: Any) -> MappingProxyType:
    """Return one deeply immutable bounded metadata mapping."""
    if not isinstance(value, Mapping):
        raise assessment_error(
            "invalid_metadata",
            "$.metadata",
            "metadata must be a mapping",
        )
    return freeze_json_value(value, "$.metadata", depth=0, node_count=[0])


def thaw_json_value(value: Any) -> Any:
    """Return ordinary JSON-compatible containers from immutable domain values."""
    if isinstance(value, Mapping):
        return {key: thaw_json_value(entry) for key, entry in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(entry) for entry in value]
    return value


def _canonical_payload(value: Any) -> Any:
    """Return package-owned content without caller-defined converters."""
    if isinstance(value, CanonicalContract):
        return value._content_dict()
    if value is None or isinstance(
        value,
        (bool, int, float, str, Mapping, list, tuple),
    ):
        return value
    raise assessment_error(
        "unsupported_canonical_artifact",
        "$",
        "value contains an unsupported canonical artifact",
    )


def canonical_json(value: Any) -> str:
    """Serialize bounded JSON-compatible content deterministically as UTF-8 text."""
    frozen = freeze_json_value(
        _canonical_payload(value),
        "$",
        depth=0,
        node_count=[0],
    )
    encoded = json.dumps(
        thaw_json_value(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    encoded_bytes = encoded.encode("utf-8")
    if len(encoded_bytes) > MAX_CANONICAL_JSON_CHARACTERS:
        raise assessment_error(
            "canonical_json_too_large",
            "$",
            (
                "canonical JSON must contain at most "
                f"{MAX_CANONICAL_JSON_CHARACTERS} UTF-8 bytes"
            ),
        )
    return encoded


def artifact_digest(value: Any) -> str:
    """Return the SHA-256 identity of one canonical bounded artifact."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
