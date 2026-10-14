"""Contract-specific fail-closed wrappers around generic scoring validators."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import hashlib
import json
from typing import Any, TypeVar

import numpy as np

from fast_mlsirm.rubric.models import _identifier, _semantic_version, _text

from . import _validation as base

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
        "provider_output",
        "provider_response",
        "raw_response",
        "response_content",
        "response_text",
        "source_content",
        "source_text",
    }
)

EnumValue = TypeVar("EnumValue", bound=Enum)


def descriptive_identifier(
    value: Any,
    name: str,
    path: str | None = None,
) -> str:
    """Normalize one identifier while preserving package-owned domain errors."""
    resolved_path = path or f"$.{name}"
    try:
        normalized = _identifier(value, name)
    except base.AssessmentSpecError:
        raise
    except Exception:
        raise base.assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must use two-or-more-token lower snake_case",
        ) from None
    return base._require_utf8(normalized, resolved_path)


def bounded_text(
    value: Any,
    name: str,
    *,
    maximum: int = base.MAX_METADATA_TEXT_LENGTH,
    path: str | None = None,
) -> str:
    """Normalize bounded text while preserving package-owned domain errors."""
    resolved_path = path or f"$.{name}"
    try:
        normalized = _text(value, name, maximum=maximum)
    except base.AssessmentSpecError:
        raise
    except Exception:
        raise base.assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be non-empty text containing at most {maximum} characters",
        ) from None
    return base._require_utf8(normalized, resolved_path)


def semantic_version(
    value: Any,
    name: str,
    path: str | None = None,
) -> str:
    """Normalize a semantic version while preserving package-owned errors."""
    resolved_path = path or f"$.{name}"
    try:
        normalized = _semantic_version(value, name)
    except base.AssessmentSpecError:
        raise
    except Exception:
        raise base.assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be a canonical semantic version",
        ) from None
    return base._require_utf8(normalized, resolved_path)


def enum_value(
    value: Any,
    enum_type: type[EnumValue],
    name: str,
    path: str | None = None,
) -> EnumValue:
    """Accept enum members or exact built-in strings without alien equality."""
    resolved_path = path or f"$.{name}"
    if isinstance(value, enum_type):
        return value
    if type(value) is not str:
        raise base.assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be one of the supported values",
        )
    try:
        return enum_type(value)
    except base.AssessmentSpecError:
        raise
    except Exception:
        raise base.assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be one of the supported values",
        ) from None


def _has_exact_type(value: Any, trusted_types: tuple[type, ...]) -> bool:
    """Return whether a control has one exact package-trusted scalar type."""
    value_type = type(value)
    return any(value_type is trusted_type for trusted_type in trusted_types)


def _normalize_metadata_scalar(value: Any) -> tuple[bool, Any]:
    """Normalize JSON scalar subclasses through inert base-type descriptors."""
    value_type = type(value)
    if value is None or value_type in (bool, int, float, str):
        return True, value
    if isinstance(value, str):
        return True, str.__str__(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return True, int.__int__(value)
    if isinstance(value, float):
        return True, float.__float__(value)
    return False, value


def bounded_positive_integer(
    value: Any,
    name: str,
    maximum: int,
    path: str | None = None,
) -> int:
    """Normalize one positive integer without caller-controlled coercion."""
    resolved_path = path or f"$.{name}"
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif _has_exact_type(value, _NUMPY_INTEGER_SCALAR_TYPES):
        normalized = int(value)
    else:
        raise base.assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be an integer between 1 and {maximum}",
        )
    if not 1 <= normalized <= maximum:
        raise base.assessment_error(
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
    """Materialize a bounded iterable through the structured error boundary."""
    resolved_path = path or f"$.{name}"
    if isinstance(values, (str, bytes, bytearray)):
        raise base.assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be a collection",
        )
    try:
        iterator = iter(values)
    except base.AssessmentSpecError:
        raise
    except Exception:
        raise base.assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must be a collection",
        ) from None
    output: list[Any] = []
    try:
        for index, value in enumerate(iterator):
            if index >= maximum:
                raise base.assessment_error(
                    f"invalid_{name}",
                    resolved_path,
                    f"{name} must contain at most {maximum} values",
                )
            output.append(value)
    except base.AssessmentSpecError:
        raise
    except Exception:
        raise base.assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} could not be materialized safely",
        ) from None
    if len(output) < minimum:
        raise base.assessment_error(
            f"invalid_{name}",
            resolved_path,
            f"{name} must contain at least {minimum} value",
        )
    return tuple(output)


def sorted_identifiers(
    values: Any,
    name: str,
    *,
    minimum: int,
    maximum: int = base.MAX_POLICY_REFERENCES,
) -> tuple[str, ...]:
    """Return safe sorted identifiers using callback-hardened boundaries."""
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
        raise base.assessment_error(
            f"duplicate_{name}",
            f"$.{name}",
            f"{name} must not contain duplicates",
        )
    return tuple(sorted(normalized))


def sorted_fingerprints(
    values: Any,
    name: str,
    *,
    minimum: int,
    maximum: int = base.MAX_ASSESSMENT_RUBRICS,
) -> tuple[str, ...]:
    """Return safe sorted SHA-256 identities from a bounded collection."""
    raw = bounded_values(
        values,
        name,
        minimum=minimum,
        maximum=maximum,
    )
    normalized: list[str] = []
    for index, value in enumerate(raw):
        path = f"$.{name}[{index}]"
        try:
            normalized.append(base.fingerprint(value, name, path))
        except base.AssessmentSpecError:
            raise
        except Exception:
            raise base.assessment_error(
                f"invalid_{name}",
                path,
                f"{name} must contain 64-character lower hexadecimal digests",
            ) from None
    if len(set(normalized)) != len(normalized):
        raise base.assessment_error(
            f"duplicate_{name}",
            f"$.{name}",
            f"{name} must not contain duplicates",
        )
    return tuple(sorted(normalized))


def _preflight_metadata(
    value: Any,
    path: str,
    *,
    depth: int = 0,
    node_count: list[int] | None = None,
    active_containers: set[int] | None = None,
) -> Any:
    """Copy bounded acyclic metadata while redacting caller callback failures."""
    counts = [0] if node_count is None else node_count
    active = set() if active_containers is None else active_containers
    if depth > base.MAX_METADATA_DEPTH:
        raise base.assessment_error(
            "metadata_depth_exceeded",
            path,
            f"metadata exceeds the maximum depth of {base.MAX_METADATA_DEPTH}",
        )
    counts[0] += 1
    if counts[0] > base.MAX_METADATA_NODES:
        raise base.assessment_error(
            "metadata_node_budget_exceeded",
            path,
            f"metadata exceeds the maximum node count of {base.MAX_METADATA_NODES}",
        )

    is_scalar, normalized_scalar = _normalize_metadata_scalar(value)
    if is_scalar:
        return normalized_scalar

    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise base.assessment_error(
                "cyclic_metadata_reference",
                path,
                "metadata cannot contain cyclic container references",
            )
        active.add(marker)
        try:
            try:
                iterator = iter(value.items())
            except Exception:
                raise base.assessment_error(
                    "invalid_metadata_mapping",
                    path,
                    "metadata mapping entries could not be inspected safely",
                ) from None
            output: dict[str, Any] = {}
            try:
                for index, entry in enumerate(iterator):
                    if index >= base.MAX_METADATA_COLLECTION_VALUES:
                        raise base.assessment_error(
                            "metadata_collection_too_large",
                            path,
                            (
                                "metadata mappings must contain at most "
                                f"{base.MAX_METADATA_COLLECTION_VALUES} values"
                            ),
                        )
                    try:
                        raw_key, child = entry
                    except Exception:
                        raise base.assessment_error(
                            "invalid_metadata_mapping",
                            f"{path}.entries[{index}]",
                            "metadata mapping entries must contain one key and value",
                        ) from None
                    key_path = f"{path}.keys[{index}]"
                    key_is_scalar, key_input = _normalize_metadata_scalar(raw_key)
                    if not key_is_scalar or type(key_input) is not str:
                        key_input = raw_key
                    key = base._metadata_key(key_input, key_path)
                    if key.casefold() in _SENSITIVE_METADATA_FIELDS:
                        raise base.assessment_error(
                            "sensitive_metadata_field",
                            key_path,
                            "metadata cannot contain response or source content fields",
                        )
                    if key in output:
                        raise base.assessment_error(
                            "duplicate_metadata_key",
                            key_path,
                            "metadata keys must be unique",
                        )
                    output[key] = _preflight_metadata(
                        child,
                        f"{path}.values[{index}]",
                        depth=depth + 1,
                        node_count=counts,
                        active_containers=active,
                    )
            except base.AssessmentSpecError:
                raise
            except Exception:
                raise base.assessment_error(
                    "invalid_metadata_mapping",
                    path,
                    "metadata mapping entries could not be materialized safely",
                ) from None
            return output
        finally:
            active.remove(marker)

    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise base.assessment_error(
                "cyclic_metadata_reference",
                path,
                "metadata cannot contain cyclic container references",
            )
        active.add(marker)
        try:
            try:
                size = len(value)
            except Exception:
                raise base.assessment_error(
                    "invalid_metadata_collection",
                    path,
                    "metadata collection size could not be inspected safely",
                ) from None
            if size > base.MAX_METADATA_COLLECTION_VALUES:
                raise base.assessment_error(
                    "metadata_collection_too_large",
                    path,
                    (
                        "metadata collections must contain at most "
                        f"{base.MAX_METADATA_COLLECTION_VALUES} values"
                    ),
                )
            output: list[Any] = []
            try:
                iterator = iter(value)
                for index, child in enumerate(iterator):
                    if index >= base.MAX_METADATA_COLLECTION_VALUES:
                        raise base.assessment_error(
                            "metadata_collection_too_large",
                            path,
                            (
                                "metadata collections must contain at most "
                                f"{base.MAX_METADATA_COLLECTION_VALUES} values"
                            ),
                        )
                    output.append(
                        _preflight_metadata(
                            child,
                            f"{path}[{index}]",
                            depth=depth + 1,
                            node_count=counts,
                            active_containers=active,
                        )
                    )
            except base.AssessmentSpecError:
                raise
            except Exception:
                raise base.assessment_error(
                    "invalid_metadata_collection",
                    path,
                    "metadata collection could not be materialized safely",
                ) from None
            return tuple(output) if isinstance(value, tuple) else output
        finally:
            active.remove(marker)

    return value


def freeze_metadata(value: Any) -> Any:
    """Return immutable metadata after bounded callback-safe preflight."""
    return base.freeze_metadata(_preflight_metadata(value, "$.metadata"))


def canonical_json(value: Any) -> str:
    """Serialize complete contracts without reapplying the metadata-only node cap."""
    if not isinstance(value, base.CanonicalContract):
        return base.canonical_json(value)
    encoded = json.dumps(
        base.thaw_json_value(value._content_dict()),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    try:
        encoded_bytes = encoded.encode("utf-8")
    except UnicodeEncodeError:
        raise base.assessment_error(
            "invalid_utf8_text",
            "$",
            "text must be valid UTF-8",
        ) from None
    if len(encoded_bytes) > base.MAX_CANONICAL_JSON_CHARACTERS:
        raise base.assessment_error(
            "canonical_json_too_large",
            "$",
            (
                "canonical JSON must contain at most "
                f"{base.MAX_CANONICAL_JSON_CHARACTERS} UTF-8 bytes"
            ),
        )
    return encoded


def artifact_digest(value: Any) -> str:
    """Return SHA-256 over the callback-safe canonical contract representation."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
