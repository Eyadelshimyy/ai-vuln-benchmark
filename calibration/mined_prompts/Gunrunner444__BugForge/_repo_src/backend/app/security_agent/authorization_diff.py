"""Compare two identity-scoped HTTP observations. Never auto-declare a vulnerability.

A status/body difference is only a *security hypothesis* when an
:class:`AuthorizationOracle` (expected authorization rule) is known.
Dynamic fields (timestamps, request IDs, tracing, CSRF tokens, random
values) are normalized before comparison.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.security_agent.oracles import AuthorizationOracle

_DYNAMIC_KEY_FRAGMENTS = (
    "timestamp",
    "created_at",
    "updated_at",
    "date",
    "time",
    "request_id",
    "requestid",
    "trace_id",
    "traceid",
    "span_id",
    "correlation",
    "csrf",
    "nonce",
    "uuid",
    "rand",
    "random",
    "token",
)
_DYNAMIC_HEADER_FRAGMENTS = (
    "date",
    "request-id",
    "request_id",
    "trace",
    "span",
    "correlation",
    "csrf",
    "x-amzn",
    "x-amz",
    "cf-ray",
    "age",
    "x-runtime",
)
_ISO_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")
_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_SENSITIVE_FIELDS = (
    "password",
    "secret",
    "token",
    "ssn",
    "email",
    "credit_card",
    "authorization",
    "api_key",
    "session",
)


@dataclass
class AuthorizationHypothesis:
    title: str
    target: str
    difference: str
    status_a: int | None
    status_b: int | None
    evidence_ids: tuple[str, ...] = ()
    security_expectation: str = ""
    is_vulnerability: bool = False
    oracle: str = AuthorizationOracle.CUSTOM_EXPECTATION.value
    status_difference: str = ""
    response_structure_difference: str = ""
    resource_identity_difference: str = ""
    ownership_difference: str = ""
    sensitive_field_exposure: str = ""
    permission_difference: str = ""
    expected_authorization_rule: str = ""
    actual_authorization_result: str = ""
    meaningful_security_hypothesis: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "target": self.target,
            "difference": self.difference,
            "status_a": self.status_a,
            "status_b": self.status_b,
            "evidence_ids": list(self.evidence_ids),
            "security_expectation": self.security_expectation,
            "is_vulnerability": False,
            "oracle": self.oracle,
            "status_difference": self.status_difference,
            "response_structure_difference": self.response_structure_difference,
            "resource_identity_difference": self.resource_identity_difference,
            "ownership_difference": self.ownership_difference,
            "sensitive_field_exposure": self.sensitive_field_exposure,
            "permission_difference": self.permission_difference,
            "expected_authorization_rule": self.expected_authorization_rule,
            "actual_authorization_result": self.actual_authorization_result,
            "meaningful_security_hypothesis": self.meaningful_security_hypothesis,
        }


def compare_authorization(
    exchange_a: dict[str, Any],
    exchange_b: dict[str, Any],
    *,
    expectation: str,
    evidence_ids: tuple[str, ...] = (),
    oracle: AuthorizationOracle | str | None = None,
) -> AuthorizationHypothesis:
    parsed_oracle = _oracle(oracle, expectation)
    status_a = _status(exchange_a)
    status_b = _status(exchange_b)
    body_a = _normalized_body(exchange_a)
    body_b = _normalized_body(exchange_b)
    struct_a = _structure(body_a)
    struct_b = _structure(body_b)
    owner_a = str(exchange_a.get("resource_id") or _field(body_a, "id") or "")
    owner_b = str(exchange_b.get("resource_id") or _field(body_b, "id") or "")
    owner_field_a = str(exchange_a.get("owner") or _field(body_a, "owner") or "")
    owner_field_b = str(exchange_b.get("owner") or _field(body_b, "owner") or "")

    status_difference = ""
    if status_a != status_b:
        status_difference = f"status {status_a} vs {status_b}"
    structure_difference = ""
    if struct_a != struct_b:
        structure_difference = "response structure differs after normalizing dynamic fields"
    resource_difference = ""
    if owner_a and owner_b and owner_a != owner_b:
        resource_difference = f"resource identity {owner_a} vs {owner_b}"
    ownership_difference = ""
    if owner_field_a and owner_field_b and owner_field_a != owner_field_b:
        ownership_difference = f"owner {owner_field_a} vs {owner_field_b}"
    sensitive = _sensitive_exposure(body_a, body_b, parsed_oracle)
    permission = _permission_difference(status_a, status_b, parsed_oracle)

    differences = [
        item
        for item in (
            status_difference,
            structure_difference,
            resource_difference,
            ownership_difference,
            sensitive,
            permission,
        )
        if item
    ]
    expected_rule = expectation.strip() or _default_rule(parsed_oracle)
    actual = "; ".join(differences) or "no meaningful difference after normalization"
    has_expectation = bool(expected_rule)
    meaningful = has_expectation and bool(differences)
    return AuthorizationHypothesis(
        title="authorization differential",
        target=str(exchange_a.get("url") or exchange_b.get("url") or ""),
        difference=actual,
        status_a=status_a,
        status_b=status_b,
        evidence_ids=evidence_ids,
        security_expectation=expected_rule,
        is_vulnerability=False,
        oracle=parsed_oracle.value,
        status_difference=status_difference,
        response_structure_difference=structure_difference,
        resource_identity_difference=resource_difference,
        ownership_difference=ownership_difference,
        sensitive_field_exposure=sensitive,
        permission_difference=permission,
        expected_authorization_rule=expected_rule,
        actual_authorization_result=actual,
        meaningful_security_hypothesis=meaningful,
    )


def _oracle(oracle: AuthorizationOracle | str | None, expectation: str) -> AuthorizationOracle:
    if isinstance(oracle, AuthorizationOracle):
        return oracle
    if isinstance(oracle, str) and oracle.strip():
        try:
            return AuthorizationOracle(oracle.strip())
        except ValueError:
            return AuthorizationOracle.CUSTOM_EXPECTATION
    if expectation.strip():
        return AuthorizationOracle.CUSTOM_EXPECTATION
    return AuthorizationOracle.CUSTOM_EXPECTATION


def _default_rule(oracle: AuthorizationOracle) -> str:
    return {
        AuthorizationOracle.OWNER_ONLY: "Only the resource owner may read or mutate the object.",
        AuthorizationOracle.ROLE_REQUIRED: "The requested role is required for this operation.",
        AuthorizationOracle.TENANT_ISOLATION: "Tenant A data must be invisible to tenant B.",
        AuthorizationOracle.AUTHENTICATED_USER_ONLY: "Anonymous callers must not receive the resource.",
        AuthorizationOracle.PUBLIC_RESOURCE: "The resource is expected to be public; differences are not authorization bugs.",
        AuthorizationOracle.CUSTOM_EXPECTATION: "",
    }[oracle]


def _status(exchange: dict[str, Any]) -> int | None:
    raw = exchange.get("status") or (exchange.get("response") or {}).get("status")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _normalized_body(exchange: dict[str, Any]) -> Any:
    raw = (exchange.get("response") or {}).get("body")
    if raw is None:
        raw = exchange.get("body")
    if isinstance(raw, (dict, list)):
        return _strip_dynamic(raw)
    text = str(raw or "")
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return _UUID.sub("<id>", _ISO_TS.sub("<ts>", text))
    return _strip_dynamic(parsed)


def _strip_dynamic(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower().replace("-", "_")
            if any(fragment in lowered for fragment in _DYNAMIC_KEY_FRAGMENTS):
                continue
            cleaned[str(key)] = _strip_dynamic(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_dynamic(item) for item in value]
    if isinstance(value, str):
        return _UUID.sub("<id>", _ISO_TS.sub("<ts>", value))
    return value


def _structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _structure(item) for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, list):
        if not value:
            return []
        return [_structure(value[0])]
    return type(value).__name__


def _field(body: Any, name: str) -> Any:
    if isinstance(body, dict):
        return body.get(name)
    return None


def _sensitive_exposure(body_a: Any, body_b: Any, oracle: AuthorizationOracle) -> str:
    if oracle is AuthorizationOracle.PUBLIC_RESOURCE:
        return ""
    exposed_a = _sensitive_keys(body_a)
    exposed_b = _sensitive_keys(body_b)
    extra = sorted(set(exposed_b) - set(exposed_a))
    if extra:
        return "sensitive fields exposed to identity B: " + ", ".join(extra)
    extra_a = sorted(set(exposed_a) - set(exposed_b))
    if extra_a:
        return "sensitive fields exposed to identity A: " + ", ".join(extra_a)
    return ""


def _sensitive_keys(body: Any) -> list[str]:
    found: list[str] = []
    if isinstance(body, dict):
        for key, value in body.items():
            lowered = str(key).lower()
            if any(item in lowered for item in _SENSITIVE_FIELDS) and value not in (
                None,
                "",
                [],
                {},
            ):
                found.append(str(key))
            found.extend(_sensitive_keys(value))
    elif isinstance(body, list):
        for item in body:
            found.extend(_sensitive_keys(item))
    return found


def _permission_difference(
    status_a: int | None, status_b: int | None, oracle: AuthorizationOracle
) -> str:
    if status_a is None or status_b is None or status_a == status_b:
        return ""
    allowed = {200, 201, 204}
    denied = {401, 403, 404}
    if status_a in allowed and status_b in denied:
        if oracle is AuthorizationOracle.PUBLIC_RESOURCE:
            return "identity B was denied a resource marked public"
        return "identity A allowed, identity B denied"
    if status_a in denied and status_b in allowed:
        return "identity B allowed, identity A denied"
    return ""
