"""Controlled fuzzing. Never unrestricted. Always ScopeGuard + SafetyController."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.domain.evidence import Evidence, EvidenceKind
from app.domain.http import HttpExchange
from app.security_testing.approvals import ApprovalKind
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import ApprovalRequiredError, AuthorizationDeniedError
from app.security_testing.failures import ToolExecutionResult, ToolExecutionState
from app.security_testing.rate_limit import RateLimiter
from app.security_testing.safety import SafetyLimits


class MutationKind(StrEnum):
    QUERY = "query"
    PATH = "path"
    JSON = "json"
    MALFORMED_JSON = "malformed_json"
    FORM = "form"
    HEADER = "header"
    API = "api"


@dataclass(frozen=True)
class SeedRequest:
    method: str
    url: str
    headers: tuple[tuple[str, str], ...] = ()
    body: str | None = None


@dataclass(frozen=True)
class FuzzLimits:
    request_limit: int = 10
    requests_per_second: float = 1.0
    concurrency: int = 1
    timeout_seconds: float = 5.0
    payload_count: int = 8
    max_body_size: int = 2048

    def intersect(self, safety: SafetyLimits) -> FuzzLimits:
        """Enforce the stricter of fuzzer-specific and SafetyController limits."""
        return FuzzLimits(
            request_limit=min(self.request_limit, safety.max_requests),
            requests_per_second=min(self.requests_per_second, safety.requests_per_second),
            concurrency=min(self.concurrency, safety.max_concurrent),
            timeout_seconds=min(self.timeout_seconds, safety.timeout_seconds),
            payload_count=min(self.payload_count, safety.max_payload_count),
            max_body_size=min(self.max_body_size, safety.max_payload_bytes),
        )


_DEFAULT_PAYLOADS = (
    "'",
    '"',
    "<x>",
    "../",
    "%00",
    "1 OR 1=1",
    "{}",
    "A" * 64,
)


class ResponseDifferenceAnalyzer:
    def diff(self, baseline: HttpExchange, candidate: HttpExchange) -> str | None:
        if baseline.response_status != candidate.response_status:
            return f"status {baseline.response_status} -> {candidate.response_status}"
        base_len = len(baseline.response_body or "")
        cand_len = len(candidate.response_body or "")
        if abs(base_len - cand_len) > max(32, int(base_len * 0.3)):
            return f"body length {base_len} -> {cand_len}"
        return None


class FuzzingEngine:
    def __init__(self, engine: SecurityTestEngine, *, limits: FuzzLimits | None = None) -> None:
        self.engine = engine
        requested = limits or FuzzLimits()
        self.limits = requested.intersect(engine.safety.limits)
        self.analyzer = ResponseDifferenceAnalyzer()
        self._limiter = RateLimiter(
            SafetyLimits(
                requests_per_second=self.limits.requests_per_second,
                max_requests=self.limits.request_limit,
            )
        )

    async def fuzz(
        self,
        seed: SeedRequest,
        *,
        kinds: Sequence[MutationKind] = (MutationKind.QUERY,),
        operator: str | None = None,
    ) -> list[Evidence] | ToolExecutionResult:
        if not self.engine.session.fuzzing_enabled and not self.engine.approvals.is_granted(
            ApprovalKind.ENABLE_FUZZING
        ):
            if operator:
                self.engine.grant(ApprovalKind.ENABLE_FUZZING, operator=operator)
            else:
                raise ApprovalRequiredError(ApprovalKind.ENABLE_FUZZING.value)
        decision = self.engine.authorize(seed.url, method=seed.method, tool="fuzzer", active=True)
        if not decision.allowed:
            raise AuthorizationDeniedError(decision.reason, target=seed.url, tool="fuzzer")

        client = self.engine.http("fuzzer")
        baseline = await client.request(
            seed.method,
            seed.url,
            headers=dict(seed.headers),
            content=seed.body,
            timeout=self.limits.timeout_seconds,
        )
        if isinstance(baseline, ToolExecutionResult):
            return baseline
        evidence: list[Evidence] = [
            Evidence(
                kind=EvidenceKind.FUZZING,
                source="fuzzer",
                summary=f"Baseline {baseline.response_status} for {seed.method} {seed.url}",
            )
        ]
        sent = 0
        payloads = _DEFAULT_PAYLOADS[: self.limits.payload_count]
        for kind in kinds:
            for payload in payloads:
                if sent >= self.limits.request_limit:
                    evidence.append(
                        Evidence(
                            kind=EvidenceKind.FUZZING,
                            source="fuzzer",
                            summary=f"Stopped at request_limit={self.limits.request_limit}",
                        )
                    )
                    self.engine.add_evidence(evidence)
                    return evidence
                if self.engine.safety.remaining_requests() <= 0:
                    evidence.append(
                        Evidence(
                            kind=EvidenceKind.FUZZING,
                            source="fuzzer",
                            summary="Stopped at SafetyController max_requests",
                        )
                    )
                    self.engine.add_evidence(evidence)
                    return evidence
                if len(payload.encode()) > self.limits.max_body_size:
                    continue
                rate = self._limiter.acquire()
                if not rate.allowed:
                    evidence.append(
                        Evidence(
                            kind=EvidenceKind.FUZZING,
                            source="fuzzer",
                            summary=f"Stopped at fuzzer RPS limit {self.limits.requests_per_second}",
                        )
                    )
                    self.engine.add_evidence(evidence)
                    return evidence
                mutated = _mutate(seed, kind, payload, max_body=self.limits.max_body_size)
                if mutated.body and len(mutated.body.encode("utf-8")) > self.limits.max_body_size:
                    continue
                result = await client.request(
                    mutated.method,
                    mutated.url,
                    headers=dict(mutated.headers),
                    content=mutated.body,
                    timeout=self.limits.timeout_seconds,
                )
                sent += 1
                if isinstance(result, ToolExecutionResult):
                    if result.state is ToolExecutionState.DRY_RUN:
                        evidence.append(
                            Evidence(
                                kind=EvidenceKind.FUZZING,
                                source="fuzzer",
                                summary=result.detail or "dry-run",
                            )
                        )
                        return evidence
                    continue
                delta = self.analyzer.diff(baseline, result)
                if delta:
                    evidence.append(
                        Evidence(
                            kind=EvidenceKind.FUZZING,
                            source="fuzzer",
                            summary=f"{kind.value} payload {payload!r}: {delta}",
                            details=f"status={result.response_status}",
                        )
                    )
        self.engine.add_evidence(evidence)
        return evidence


def _mutate(seed: SeedRequest, kind: MutationKind, payload: str, *, max_body: int) -> SeedRequest:
    if kind is MutationKind.QUERY:
        parsed = urlparse(seed.url)
        query_pairs = list(parse_qsl(parsed.query, keep_blank_values=True))
        if not query_pairs:
            query_pairs = [("q", payload)]
        else:
            first_name = query_pairs[0][0]
            mutated = False
            rebuilt: list[tuple[str, str]] = []
            for name, value in query_pairs:
                if not mutated and name == first_name:
                    rebuilt.append((name, payload))
                    mutated = True
                else:
                    rebuilt.append((name, value))
            query_pairs = rebuilt
        url = urlunparse(parsed._replace(query=urlencode(query_pairs, doseq=True)))
        return replace(seed, url=url)
    if kind is MutationKind.PATH:
        parsed = urlparse(seed.url)
        path = parsed.path.rstrip("/") + "/" + payload
        return replace(seed, url=urlunparse(parsed._replace(path=path)))
    if kind is MutationKind.JSON:
        body = mutate_json_document(seed.body or "{}", payload)
        return replace(seed, body=body[:max_body])
    if kind is MutationKind.MALFORMED_JSON:
        return replace(seed, body=malformed_json_document(seed.body or "{}", payload)[:max_body])
    if kind is MutationKind.FORM:
        return replace(seed, body=f"fuzz={payload}"[:max_body])
    if kind is MutationKind.HEADER:
        return replace(seed, headers=seed.headers + (("X-BugForge-Fuzz", payload),))
    return replace(seed, body=(payload[:max_body] if payload else seed.body))


def mutate_json_document(body: str, payload: str) -> str:
    """JSON-aware mutation. Uses parsed JSON + json.dumps, never Python repr()."""
    try:
        data: Any = json.loads(body) if body.strip() else {}
    except json.JSONDecodeError:
        return json.dumps({"fuzz": payload}, ensure_ascii=False)
    mutated = _mutate_json_node(data, payload)
    return json.dumps(mutated, ensure_ascii=False)


def malformed_json_document(body: str, payload: str) -> str:
    """Explicit malformed-JSON mode. Output is intentionally not valid JSON."""
    return (body.rstrip() + payload + ",,}") if body.strip() else "{" + payload


def _mutate_json_node(value: Any, payload: str) -> Any:
    injected = _json_payload(payload)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        mutated_one = False
        for key, item in value.items():
            if not mutated_one:
                out[key] = _mutate_json_node(item, payload)
                mutated_one = True
            else:
                out[key] = item
        out["fuzz"] = injected
        return out
    if isinstance(value, list):
        items = [_mutate_json_node(item, payload) for item in value]
        items.append(injected)
        return items
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(injected, (int, float)) and not isinstance(injected, bool):
            return injected
        return 0
    if value is None:
        return injected
    if isinstance(value, str):
        return payload
    return injected


def _json_payload(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        lowered = payload.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        try:
            if "." in payload:
                return float(payload)
            return int(payload)
        except ValueError:
            return payload
