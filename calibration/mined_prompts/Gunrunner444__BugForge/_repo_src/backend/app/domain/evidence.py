"""Evidence-first domain types.

A model-generated hypothesis is never independent verification evidence.
Collectors convert tool output and repository artifacts into :class:`Evidence`
records. Verification requires observational or executable provenance.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


class EvidenceKind(StrEnum):
    AI_ANALYSIS = "ai_analysis"
    STATIC_ANALYSIS = "static_analysis"
    SOURCE_CODE = "source_code"
    GENERATED_TEST = "generated_test"
    BROWSER = "browser"
    PROXY = "proxy"
    SCANNER = "scanner"
    SCANNER_PLAN = "scanner_plan"
    TOOL_STATUS = "tool_status"
    FUZZING = "fuzzing"
    API_TEST = "api_test"
    REPRODUCTION = "reproduction"
    SCREENSHOT = "screenshot"
    LOG = "log"
    HTTP_REQUEST = "http_request"
    HTTP_RESPONSE = "http_response"
    TEST_FAILURE = "test_failure"
    REPLAY = "replay"
    ECONOMIC_OBSERVATION = "economic_observation"
    PROTOCOL_OBSERVATION = "protocol_observation"
    RUNTIME_OBSERVATION = "runtime_observation"


class EvidenceProvenance(StrEnum):
    """Where a piece of evidence came from.

    AI-generated reasoning is tracked explicitly so it can never be treated as
    independent verification.
    """

    AI_HYPOTHESIS = "ai_hypothesis"
    STATIC_ANALYSIS = "static_analysis"
    SOURCE_OBSERVATION = "source_observation"
    EXECUTION = "execution"
    REPRODUCTION = "reproduction"
    BROWSER_OBSERVATION = "browser_observation"
    HTTP_OBSERVATION = "http_observation"
    SCANNER_OBSERVATION = "scanner_observation"
    SCANNER_RESULT = "scanner_result"
    SCANNER_PLAN = "scanner_plan"
    TOOL_STATUS = "tool_status"
    API_TEST = "api_test"
    FUZZING_RESULT = "fuzzing_result"
    SCREENSHOT = "screenshot"
    LOG = "log"
    REPLAY = "replay"
    # Generated exploratory code inside Docker. Not live verification.
    SANDBOX_EXECUTION = "sandbox_execution"
    # In-process economic calculation. Not live verification.
    ECONOMIC_OBSERVATION = "economic_observation"
    # Protocol-graph candidate. Not live verification.
    PROTOCOL_OBSERVATION = "protocol_observation"
    # Sandboxed runtime or differential observation. Not live verification.
    RUNTIME_OBSERVATION = "runtime_observation"


# Provenance that can back a verified finding. AI text and sandbox runs are excluded.
VERIFICATION_PROVENANCE: frozenset[EvidenceProvenance] = frozenset(
    {
        EvidenceProvenance.EXECUTION,
        EvidenceProvenance.REPRODUCTION,
        EvidenceProvenance.BROWSER_OBSERVATION,
        EvidenceProvenance.HTTP_OBSERVATION,
        EvidenceProvenance.SCANNER_OBSERVATION,
        EvidenceProvenance.SCANNER_RESULT,
        EvidenceProvenance.API_TEST,
        EvidenceProvenance.FUZZING_RESULT,
        EvidenceProvenance.SCREENSHOT,
        EvidenceProvenance.LOG,
    }
)

_KIND_TO_PROVENANCE: dict[EvidenceKind, EvidenceProvenance] = {
    EvidenceKind.AI_ANALYSIS: EvidenceProvenance.AI_HYPOTHESIS,
    EvidenceKind.STATIC_ANALYSIS: EvidenceProvenance.STATIC_ANALYSIS,
    EvidenceKind.SOURCE_CODE: EvidenceProvenance.SOURCE_OBSERVATION,
    # Generated tests are artifacts, not execution results, until they are run.
    EvidenceKind.GENERATED_TEST: EvidenceProvenance.SOURCE_OBSERVATION,
    EvidenceKind.TEST_FAILURE: EvidenceProvenance.EXECUTION,
    EvidenceKind.REPRODUCTION: EvidenceProvenance.REPRODUCTION,
    EvidenceKind.BROWSER: EvidenceProvenance.BROWSER_OBSERVATION,
    EvidenceKind.PROXY: EvidenceProvenance.HTTP_OBSERVATION,
    EvidenceKind.HTTP_REQUEST: EvidenceProvenance.HTTP_OBSERVATION,
    EvidenceKind.HTTP_RESPONSE: EvidenceProvenance.HTTP_OBSERVATION,
    EvidenceKind.SCANNER: EvidenceProvenance.SCANNER_RESULT,
    EvidenceKind.SCANNER_PLAN: EvidenceProvenance.SCANNER_PLAN,
    EvidenceKind.TOOL_STATUS: EvidenceProvenance.TOOL_STATUS,
    EvidenceKind.FUZZING: EvidenceProvenance.FUZZING_RESULT,
    EvidenceKind.API_TEST: EvidenceProvenance.API_TEST,
    EvidenceKind.LOG: EvidenceProvenance.LOG,
    EvidenceKind.SCREENSHOT: EvidenceProvenance.SCREENSHOT,
    EvidenceKind.REPLAY: EvidenceProvenance.REPLAY,
    EvidenceKind.ECONOMIC_OBSERVATION: EvidenceProvenance.ECONOMIC_OBSERVATION,
    EvidenceKind.PROTOCOL_OBSERVATION: EvidenceProvenance.PROTOCOL_OBSERVATION,
    EvidenceKind.RUNTIME_OBSERVATION: EvidenceProvenance.RUNTIME_OBSERVATION,
}


@dataclass(frozen=True)
class Evidence:
    """One independently collected piece of evidence.

    Provenance is derived from ``kind`` so callers cannot relabel AI text as
    independent verification.
    """

    kind: EvidenceKind
    source: str
    summary: str
    details: str = ""
    artifact_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: EvidenceProvenance | None = None
    collected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("Evidence summary must be non-empty")
        if not self.source.strip():
            raise ValueError("Evidence source must be non-empty")
        derived = _KIND_TO_PROVENANCE.get(self.kind)
        if derived is None:
            raise ValueError(f"Unsupported evidence kind {self.kind!r}")
        if self.provenance is not None and self.provenance is not derived:
            raise ValueError(
                f"Provenance {self.provenance.value} is incompatible with kind {self.kind.value}"
            )
        object.__setattr__(self, "provenance", derived)

    @property
    def contributes_to_verification(self) -> bool:
        return self.provenance in VERIFICATION_PROVENANCE

    @classmethod
    def from_ai(cls, summary: str, *, details: str = "", source: str = "ai_provider") -> Evidence:
        """Wrap model-generated text. Never contributes to verification."""
        return cls(
            kind=EvidenceKind.AI_ANALYSIS,
            provenance=EvidenceProvenance.AI_HYPOTHESIS,
            source=source,
            summary=summary,
            details=details,
        )


@dataclass
class EvidenceSource:
    """Inputs available to evidence collectors.

    All fields are optional so collectors can ignore data they do not
    understand. This is intentionally not tied to Burp, browsers, or a
    particular AI vendor.
    """

    static_findings: Sequence[Any] = ()
    test_failures: Sequence[Any] = ()
    source_files: Sequence[Any] = ()
    generated_tests: Sequence[Any] = ()
    reproductions: Sequence[Any] = ()
    http_exchanges: Sequence[Any] = ()
    screenshots: Sequence[Any] = ()
    logs: Sequence[Any] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    """Combined evidence from one or more collectors."""

    items: tuple[Evidence, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def of_kind(self, kind: EvidenceKind) -> tuple[Evidence, ...]:
        return tuple(item for item in self.items if item.kind == kind)

    def verifying_items(self) -> tuple[Evidence, ...]:
        return tuple(item for item in self.items if item.contributes_to_verification)

    def extend(self, extra: Iterable[Evidence]) -> EvidenceBundle:
        return EvidenceBundle(items=self.items + tuple(extra))

    @classmethod
    def from_items(cls, items: Iterable[Evidence]) -> EvidenceBundle:
        return cls(items=tuple(items))


def observation_identity(item: Evidence) -> tuple[str, ...]:
    """Canonical observation identity. Object UUIDs are not part of it.

    Trust is a server signature, not this tuple. Deduplication uses the
    canonical event: kind, normalized summary, details digest, artifact,
    execution, outcome, finding binding, and evidence-type fields such as
    HTTP method, URL, and status. The summary is whitespace-normalized and
    is not an authorization signal. A client-supplied observation id is
    ignored, so it cannot mint a trusted identity or split an exact duplicate
    into a new observation. A new server observation id for the same canonical
    event still collapses.
    """
    digest = hashlib.sha256(item.details.encode("utf-8")).hexdigest()[:16]
    return (
        item.kind.value,
        item.source.strip(),
        " ".join(item.summary.split()),
        digest,
        (item.artifact_path or "").replace("\\", "/"),
        _meta_text(item, "line"),
        _meta_text(item, "contradicts") or _meta_text(item, "reached"),
        _meta_text(item, "execution_id"),
        _meta_text(item, "outcome"),
        _meta_text(item, "finding_id"),
        _meta_text(item, "finding_key"),
        _meta_text(item, "project_id"),
        *_type_identity(item),
    )


def _type_identity(item: Evidence) -> tuple[str, ...]:
    kind = item.kind
    if kind in {
        EvidenceKind.HTTP_REQUEST,
        EvidenceKind.HTTP_RESPONSE,
        EvidenceKind.PROXY,
        EvidenceKind.API_TEST,
    }:
        return (
            _norm_meta(item, "method"),
            _norm_meta(item, "url") or _norm_meta(item, "route"),
            _norm_meta(item, "status") or _norm_meta(item, "status_code"),
        )
    if kind is EvidenceKind.SCANNER:
        return (
            _norm_meta(item, "check_id") or _norm_meta(item, "rule_id"),
            _norm_meta(item, "target"),
            _norm_meta(item, "result_id"),
        )
    if kind is EvidenceKind.BROWSER:
        return (
            _norm_meta(item, "route") or _norm_meta(item, "url"),
            _norm_meta(item, "event"),
            _norm_meta(item, "session_id"),
        )
    if kind is EvidenceKind.FUZZING:
        return (_norm_meta(item, "target"), _norm_meta(item, "result_id"))
    if kind is EvidenceKind.REPRODUCTION:
        return (
            _norm_meta(item, "result_id"),
            _norm_meta(item, "reproduced"),
            _norm_meta(item, "observed_target"),
        )
    if kind in {EvidenceKind.STATIC_ANALYSIS, EvidenceKind.SOURCE_CODE}:
        return (
            _norm_meta(item, "rule") or _norm_meta(item, "sink"),
            _norm_meta(item, "field_path"),
            _norm_meta(item, "sink_occurrence"),
            _norm_meta(item, "scope_id"),
            _norm_meta(item, "argument_index"),
        )
    return ()


def _norm_meta(item: Evidence, key: str) -> str:
    return " ".join(_meta_text(item, key).split()).lower()


def evidence_contradicts(item: Evidence) -> bool:
    """True when the record says the suspected path was not reached."""
    flag = _meta_text(item, "contradicts").lower()
    reached = _meta_text(item, "reached").lower()
    return flag in {"1", "true", "yes"} or reached in {"0", "false", "no", "not_reached"}


def _meta_text(item: Evidence, key: str) -> str:
    value = item.metadata.get(key)
    return "" if value is None else str(value)
