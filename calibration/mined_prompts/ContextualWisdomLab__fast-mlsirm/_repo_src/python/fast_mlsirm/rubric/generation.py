"""Provider-neutral source packets, generation requests, and execution provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .contracts import build_generation_contract
from .models import (
    EvidenceMode,
    ItemBlueprint,
    RubricSpecification,
    SCHEMA_VERSION,
    _LOCALE_PATTERN,
    _bounded_values,
    _canonical_json,
    _identifier,
    _schema_version,
    _sha256_hex,
    _text,
    _unsigned_integer,
)

if TYPE_CHECKING:
    from .candidates import GeneratedItemCandidate

MAX_SOURCE_CHARACTERS = 262_144
MAX_SOURCES = 32
MAX_TOTAL_SOURCE_CHARACTERS = 1_048_576
MAX_RAW_RESPONSE_CHARACTERS = 262_144
_ALLOWED_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "text/csv",
        "text/markdown",
        "text/plain",
    }
)
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT_IDENTITY_FIELDS = frozenset(
    {"contract_id", "contract_handle", "contract_fingerprint"}
)


def _source_content(value: Any) -> str:
    """Validate bounded source text while preserving exact whitespace and offsets."""
    if type(value) is not str:
        raise ValueError("content must be a string")
    if not value.strip():
        raise ValueError("content must not be empty")
    if len(value) > MAX_SOURCE_CHARACTERS:
        raise ValueError(
            f"content must contain at most {MAX_SOURCE_CHARACTERS} characters"
        )
    return value


def _media_type(value: Any) -> str:
    """Normalize an allowlisted textual source media type."""
    normalized = _text(value, "media_type", maximum=64).lower()
    if normalized not in _ALLOWED_MEDIA_TYPES:
        raise ValueError(f"media_type must be one of {sorted(_ALLOWED_MEDIA_TYPES)}")
    return normalized


def _locale(value: Any) -> str:
    """Normalize a bounded BCP 47-style locale tag."""
    normalized = _text(value, "locale", maximum=64)
    if _LOCALE_PATTERN.fullmatch(normalized) is None:
        raise ValueError("locale must be a BCP 47-style tag")
    return normalized


def _digest(value: Any, name: str) -> str:
    """Validate a lower hexadecimal SHA-256 digest."""
    normalized = _text(value, name, maximum=64)
    if _DIGEST_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a 64-character lower hexadecimal digest")
    return normalized


def _validate_contract_depth(content: str) -> None:
    """Reject contract JSON strings whose nesting depth exceeds the maximum budget."""
    depth = 0
    in_string = False
    escaped = False
    for char in content:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > 128:  # MAX_JSON_NESTING_DEPTH
                raise ValueError(
                    "contract_json exceeds the maximum JSON nesting depth of 128"
                )
        elif char in "]}":
            depth -= 1


def _contract_object(contract_json: str) -> dict[str, Any]:
    """Parse canonical contract JSON and require a top-level object."""
    if type(contract_json) is not str or not contract_json:
        raise ValueError("contract_json must be non-empty JSON text")
    _validate_contract_depth(contract_json)

    def _reject_duplicates(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("contract_json contains duplicate keys")
            result[k] = v
        return result

    def _reject_nonfinite(literal):
        raise ValueError("contract_json contains non-finite numbers")

    def _reject_float_nonfinite(value):
        import math
        f_val = float(value)
        if not math.isfinite(f_val):
            raise ValueError("contract_json contains non-finite numbers")
        return f_val

    try:
        contract = json.loads(
            contract_json,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite,
            parse_float=_reject_float_nonfinite,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("contract_json must be valid JSON text") from exc
    if not isinstance(contract, dict):
        raise ValueError("contract_json must encode a JSON object")
    return contract


def _contract_string(contract: dict[str, Any], field: str) -> str:
    """Return one required bounded string from a generation contract."""
    value = contract.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"contract_json must contain a non-empty {field}")
    return value


def _contract_body(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical contract body whose digest defines its identity."""
    return {
        field_name: field_value
        for field_name, field_value in contract.items()
        if field_name not in _CONTRACT_IDENTITY_FIELDS
    }


def _validate_contract_identity(
    contract: dict[str, Any],
    expected_contract_id: str,
) -> None:
    """Recompute every contract identity field from the canonical body."""
    contract_id = _identifier(
        _contract_string(contract, "contract_id"),
        "contract_id",
    )
    contract_handle = _identifier(
        _contract_string(contract, "contract_handle"),
        "contract_handle",
    )
    contract_fingerprint = _digest(
        _contract_string(contract, "contract_fingerprint"),
        "contract_fingerprint",
    )
    expected_fingerprint = _sha256_hex(_contract_body(contract))
    if contract_fingerprint != expected_fingerprint:
        raise ValueError(
            "contract_fingerprint must match the canonical generation contract body"
        )
    derived_id = f"generation_contract_{expected_fingerprint[:16]}"
    if contract_id != expected_contract_id or contract_id != derived_id:
        raise ValueError("contract_id must match the generation contract fingerprint")
    derived_handle = f"generation_contract_{expected_fingerprint[:32]}"
    if contract_handle != derived_handle:
        raise ValueError(
            "contract_handle must match the generation contract fingerprint"
        )


def _validate_source_cardinality(
    evidence_mode: EvidenceMode,
    source_count: int,
) -> None:
    """Enforce the declared evidence mode before any provider invocation."""
    valid = False
    if evidence_mode is EvidenceMode.CLOSED_BOOK:
        valid = source_count == 0
    elif evidence_mode is EvidenceMode.SINGLE_SOURCE:
        valid = source_count == 1
    elif evidence_mode in {
        EvidenceMode.MULTI_SOURCE,
        EvidenceMode.ADVERSARIAL_CONTEXT,
    }:
        valid = source_count >= 2
    elif evidence_mode is EvidenceMode.UNANSWERABLE:
        valid = source_count >= 1
    if not valid:
        raise ValueError(
            f"source cardinality is invalid for evidence mode {evidence_mode.value}"
        )


@dataclass(frozen=True)
class SourceDocument:
    """One bounded untrusted source with content-addressed redacted metadata."""

    source_id: str
    content: str
    media_type: str = "text/plain"
    locale: str = "en"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize source identity and validate its exact preserved content."""
        object.__setattr__(
            self,
            "source_id",
            _identifier(self.source_id, "source_id"),
        )
        object.__setattr__(self, "content", _source_content(self.content))
        object.__setattr__(self, "media_type", _media_type(self.media_type))
        object.__setattr__(self, "locale", _locale(self.locale))
        object.__setattr__(
            self,
            "schema_version",
            _schema_version(self.schema_version),
        )

    @property
    def content_digest(self) -> str:
        """Return SHA-256 over the exact UTF-8 source content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return source provenance without disclosing source content."""
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "content_digest": self.content_digest,
            "character_count": len(self.content),
            "media_type": self.media_type,
            "locale": self.locale,
        }

    def to_provider_dict(self) -> dict[str, Any]:
        """Return the explicit untrusted-data payload supplied to a provider."""
        return {
            **self.to_metadata_dict(),
            "content": self.content,
            "trust_boundary": "untrusted_source_data",
        }


def _normalize_sources(values: Any) -> tuple[SourceDocument, ...]:
    """Return a bounded unique source packet and enforce its aggregate budget."""
    raw_sources = _bounded_values(
        values,
        "sources",
        minimum=0,
        maximum=MAX_SOURCES,
    )
    for index, source in enumerate(raw_sources):
        if not isinstance(source, SourceDocument):
            raise ValueError(f"sources[{index}] must be a SourceDocument")
    sources = tuple(raw_sources)
    source_ids = tuple(source.source_id for source in sources)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_id values must be unique")
    total_characters = sum(len(source.content) for source in sources)
    if total_characters > MAX_TOTAL_SOURCE_CHARACTERS:
        raise ValueError(
            f"aggregate source content exceeds {MAX_TOTAL_SOURCE_CHARACTERS} characters"
        )
    return sources


def _request_identity_payload(
    contract: dict[str, Any],
    blueprint: ItemBlueprint,
    sources: tuple[SourceDocument, ...],
    generation_seed: int,
    schema_version: str,
) -> dict[str, Any]:
    """Return complete immutable content used to address one generation request."""
    return {
        "schema_version": schema_version,
        "contract_id": contract["contract_id"],
        "contract_handle": contract["contract_handle"],
        "contract_fingerprint": contract["contract_fingerprint"],
        "blueprint_id": blueprint.blueprint_id,
        "blueprint_fingerprint": blueprint.blueprint_fingerprint,
        "generation_seed": generation_seed,
        "sources": [source.to_metadata_dict() for source in sources],
    }


@dataclass(frozen=True)
class GenerationRequest:
    """Content-addressed request binding one contract to an exact source packet."""

    request_id: str
    contract_id: str
    contract_json: str
    blueprint: ItemBlueprint
    sources: tuple[SourceDocument, ...]
    generation_seed: int
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Prevent direct construction from bypassing request provenance invariants."""
        object.__setattr__(
            self,
            "request_id",
            _identifier(self.request_id, "request_id"),
        )
        object.__setattr__(
            self,
            "contract_id",
            _identifier(self.contract_id, "contract_id"),
        )
        contract = _contract_object(self.contract_json)
        if contract.get("contract_id") != self.contract_id:
            raise ValueError("contract_json contract_id must match contract_id")
        if not isinstance(self.blueprint, ItemBlueprint):
            raise ValueError("blueprint must be an ItemBlueprint")
        blueprint_payload = contract.get("blueprint")
        if not isinstance(blueprint_payload, dict):
            raise ValueError("contract_json must contain a blueprint object")
        expected_blueprint = {
            "blueprint_id": self.blueprint.blueprint_id,
            "blueprint_fingerprint": self.blueprint.blueprint_fingerprint,
            "rubric_id": self.blueprint.rubric_id,
            "rubric_version": self.blueprint.rubric_version,
            "rubric_fingerprint": self.blueprint.rubric_fingerprint,
        }
        for field_name, expected_value in expected_blueprint.items():
            if blueprint_payload.get(field_name) != expected_value:
                raise ValueError(
                    f"contract_json blueprint {field_name} must match blueprint"
                )
        _validate_contract_identity(contract, self.contract_id)
        sources = _normalize_sources(self.sources)
        _validate_source_cardinality(self.blueprint.evidence_mode, len(sources))
        object.__setattr__(self, "sources", sources)
        generation_seed = _unsigned_integer(self.generation_seed, "generation_seed")
        if generation_seed != self.blueprint.generation_seed:
            raise ValueError("generation_seed must match blueprint generation_seed")
        object.__setattr__(self, "generation_seed", generation_seed)
        schema_version = _schema_version(self.schema_version)
        object.__setattr__(self, "schema_version", schema_version)
        identity = _request_identity_payload(
            contract,
            self.blueprint,
            sources,
            generation_seed,
            schema_version,
        )
        expected_request_id = f"generation_request_{_sha256_hex(identity)[:16]}"
        if self.request_id != expected_request_id:
            raise ValueError("request_id must match the generation request fingerprint")

    @property
    def contract(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible generation contract object."""
        return _contract_object(self.contract_json)

    @property
    def contract_handle(self) -> str:
        """Return the 128-bit public handle of the exact generation contract."""
        return _contract_string(self.contract, "contract_handle")

    @property
    def contract_fingerprint(self) -> str:
        """Return the full SHA-256 identity of the exact generation contract."""
        return _digest(
            _contract_string(self.contract, "contract_fingerprint"),
            "contract_fingerprint",
        )

    @property
    def request_fingerprint(self) -> str:
        """Return SHA-256 over contract, blueprint, seed, and redacted source metadata."""
        return _sha256_hex(
            _request_identity_payload(
                self.contract,
                self.blueprint,
                self.sources,
                self.generation_seed,
                self.schema_version,
            )
        )

    @property
    def request_handle(self) -> str:
        """Return the 128-bit public handle of this immutable request."""
        return f"generation_request_{self.request_fingerprint[:32]}"

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return auditable request metadata without source or provider text."""
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "request_handle": self.request_handle,
            "request_fingerprint": self.request_fingerprint,
            "contract_id": self.contract_id,
            "contract_handle": self.contract_handle,
            "contract_fingerprint": self.contract_fingerprint,
            "blueprint_id": self.blueprint.blueprint_id,
            "blueprint_fingerprint": self.blueprint.blueprint_fingerprint,
            "generation_seed": self.generation_seed,
            "sources": [source.to_metadata_dict() for source in self.sources],
        }

    def to_provider_dict(self) -> dict[str, Any]:
        """Return the complete explicit payload for an isolated provider adapter."""
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "request_handle": self.request_handle,
            "request_fingerprint": self.request_fingerprint,
            "contract_id": self.contract_id,
            "contract_handle": self.contract_handle,
            "contract_fingerprint": self.contract_fingerprint,
            "generation_seed": self.generation_seed,
            "generation_contract": self.contract,
            "sources": [source.to_provider_dict() for source in self.sources],
            "trust_boundary": "rubric_and_sources_are_untrusted_data",
        }


def build_generation_request(
    rubric: RubricSpecification,
    blueprint: ItemBlueprint,
    sources: Any = (),
) -> GenerationRequest:
    """Bind an exact rubric contract and bounded source packet into one request."""
    if not isinstance(rubric, RubricSpecification):
        raise TypeError("rubric must be a RubricSpecification")
    if not isinstance(blueprint, ItemBlueprint):
        raise TypeError("blueprint must be an ItemBlueprint")

    contract = build_generation_contract(rubric, blueprint)
    normalized_sources = _normalize_sources(sources)
    _validate_source_cardinality(blueprint.evidence_mode, len(normalized_sources))
    contract_json = _canonical_json(contract)
    identity = _request_identity_payload(
        contract,
        blueprint,
        normalized_sources,
        blueprint.generation_seed,
        SCHEMA_VERSION,
    )
    request_fingerprint = _sha256_hex(identity)
    return GenerationRequest(
        request_id=f"generation_request_{request_fingerprint[:16]}",
        contract_id=contract["contract_id"],
        contract_json=contract_json,
        blueprint=blueprint,
        sources=normalized_sources,
        generation_seed=blueprint.generation_seed,
        schema_version=SCHEMA_VERSION,
    )


@runtime_checkable
class ItemGenerationProvider(Protocol):
    """Minimal synchronous adapter contract for one provider generation call."""

    provider_id: str
    model_id: str

    def generate(self, request: GenerationRequest) -> str:
        """Return exactly one generated-item JSON text for ``request``."""
        ...


@dataclass
class StaticFixtureProvider:
    """Deterministic offline provider fixture; never evidence of model quality."""

    provider_id: str
    model_id: str
    response_text: str
    call_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate stable provider provenance and preserve explicit fixture text."""
        self.provider_id = _identifier(self.provider_id, "provider_id")
        self.model_id = _identifier(self.model_id, "model_id")
        if type(self.response_text) is not str:
            raise ValueError("response_text must be a string")

    def generate(self, request: GenerationRequest) -> str:
        """Return the injected response exactly once per explicit invocation."""
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")
        self.call_count += 1
        return self.response_text


class GenerationProviderError(RuntimeError):
    """Redacted provider-boundary failure that never includes provider text."""

    def __init__(self, code: str, message: str) -> None:
        """Store a stable machine code and generic public message."""
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class GenerationExecution:
    """Redacted deterministic provenance for one successful provider execution."""

    execution_id: str
    request_id: str
    contract_id: str
    provider_id: str
    model_id: str
    candidate: GeneratedItemCandidate
    raw_response_digest: str
    request_fingerprint: str = ""
    contract_fingerprint: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate exact request, contract, candidate, digest, and schema provenance."""
        from .candidates import GeneratedItemCandidate

        object.__setattr__(
            self,
            "execution_id",
            _identifier(self.execution_id, "execution_id"),
        )
        object.__setattr__(
            self,
            "request_id",
            _identifier(self.request_id, "request_id"),
        )
        object.__setattr__(
            self,
            "contract_id",
            _identifier(self.contract_id, "contract_id"),
        )
        object.__setattr__(
            self,
            "provider_id",
            _identifier(self.provider_id, "provider_id"),
        )
        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        if not isinstance(self.candidate, GeneratedItemCandidate):
            raise ValueError("candidate must be a GeneratedItemCandidate")
        self.candidate._verify_validation_proof()
        if self.candidate.request_id != self.request_id:
            raise ValueError("candidate request_id must match execution request_id")
        if self.candidate.contract_id != self.contract_id:
            raise ValueError("candidate contract_id must match execution contract_id")
        request_fingerprint = _digest(
            self.request_fingerprint,
            "request_fingerprint",
        )
        contract_fingerprint = _digest(
            self.contract_fingerprint,
            "contract_fingerprint",
        )
        if request_fingerprint != self.candidate.request_fingerprint:
            raise ValueError(
                "candidate request_fingerprint must match execution request_fingerprint"
            )
        if contract_fingerprint != self.candidate.contract_fingerprint:
            raise ValueError(
                "candidate contract_fingerprint must match execution contract_fingerprint"
            )
        object.__setattr__(self, "request_fingerprint", request_fingerprint)
        object.__setattr__(self, "contract_fingerprint", contract_fingerprint)
        object.__setattr__(
            self,
            "raw_response_digest",
            _digest(self.raw_response_digest, "raw_response_digest"),
        )
        object.__setattr__(
            self,
            "schema_version",
            _schema_version(self.schema_version),
        )
        expected_id = f"generation_execution_{self.execution_fingerprint[:16]}"
        if self.execution_id != expected_id:
            raise ValueError("execution_id must match the execution fingerprint")

    @property
    def execution_fingerprint(self) -> str:
        """Return SHA-256 over the complete redacted execution provenance."""
        return _sha256_hex(
            {
                "schema_version": self.schema_version,
                "request_id": self.request_id,
                "request_fingerprint": self.request_fingerprint,
                "contract_id": self.contract_id,
                "contract_fingerprint": self.contract_fingerprint,
                "provider_id": self.provider_id,
                "model_id": self.model_id,
                "candidate_fingerprint": self.candidate.candidate_fingerprint,
                "raw_response_digest": self.raw_response_digest,
            }
        )

    @property
    def execution_handle(self) -> str:
        """Return the 128-bit public handle of the execution record."""
        return f"generation_execution_{self.execution_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return execution provenance without raw source or provider response text."""
        self.candidate._verify_validation_proof()
        return {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "execution_handle": self.execution_handle,
            "execution_fingerprint": self.execution_fingerprint,
            "request_id": self.request_id,
            "request_fingerprint": self.request_fingerprint,
            "contract_id": self.contract_id,
            "contract_fingerprint": self.contract_fingerprint,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "candidate": self.candidate.to_dict(),
            "raw_response_digest": self.raw_response_digest,
        }


def execute_generation(
    provider: ItemGenerationProvider,
    request: GenerationRequest,
) -> GenerationExecution:
    """Invoke one provider exactly once and validate its untrusted JSON output."""
    if not isinstance(request, GenerationRequest):
        raise TypeError("request must be a GenerationRequest")
    if not isinstance(provider, ItemGenerationProvider):
        raise TypeError("provider must implement ItemGenerationProvider")
    provider_id = _identifier(provider.provider_id, "provider_id")
    model_id = _identifier(provider.model_id, "model_id")
    try:
        raw_response = provider.generate(request)
    except Exception:
        raise GenerationProviderError(
            "provider_failure",
            "provider generation failed without exposing provider diagnostics",
        ) from None
    if type(raw_response) is not str:
        raise GenerationProviderError(
            "invalid_provider_output",
            "provider must return JSON text",
        )

    from .candidates import parse_generated_item_candidate

    candidate = parse_generated_item_candidate(raw_response, request)
    raw_response_digest = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
    execution_identity = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request.request_id,
        "request_fingerprint": request.request_fingerprint,
        "contract_id": request.contract_id,
        "contract_fingerprint": request.contract_fingerprint,
        "provider_id": provider_id,
        "model_id": model_id,
        "candidate_fingerprint": candidate.candidate_fingerprint,
        "raw_response_digest": raw_response_digest,
    }
    execution_fingerprint = _sha256_hex(execution_identity)
    return GenerationExecution(
        execution_id=f"generation_execution_{execution_fingerprint[:16]}",
        request_id=request.request_id,
        contract_id=request.contract_id,
        provider_id=provider_id,
        model_id=model_id,
        candidate=candidate,
        raw_response_digest=raw_response_digest,
        request_fingerprint=request.request_fingerprint,
        contract_fingerprint=request.contract_fingerprint,
        schema_version=SCHEMA_VERSION,
    )
