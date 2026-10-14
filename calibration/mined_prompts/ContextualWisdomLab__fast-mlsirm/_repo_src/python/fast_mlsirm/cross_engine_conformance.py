"""Provider-neutral contracts for independent numerical conformance evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
import unicodedata

SCHEMA_VERSION = "1.0"
MAX_TEXT_LENGTH = 4_096
MAX_COLLECTION_VALUES = 128
MAX_MANIFEST_JSON_BYTES = 1_048_576
MAX_MANIFEST_NESTING = 128

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)


class ConformanceLayer(str, Enum):
    """Independent comparison layers required by the conformance protocol."""

    FIXED_PARAMETER_EQUATION = "fixed_parameter_equation"
    FITTED_RESULT = "fitted_result"
    NEUTRAL_TRUTH_ADVERSARIAL = "neutral_truth_adversarial"


class ConformanceCoverageStatus(str, Enum):
    """Coverage states for one advertised numerical capability."""

    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    NO_INDEPENDENT_ENGINE = "no_independent_engine"
    NOT_COMPARABLE = "not_comparable"
    PLANNED = "planned"


class ConformanceExecutionStatus(str, Enum):
    """Execution/verdict states that must remain semantically distinct."""

    PASSED = "passed"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"
    NOT_EXECUTED = "not_executed"
    NOT_APPLICABLE = "not_applicable"


class ConformanceEnvironmentKind(str, Enum):
    """Immutable environment artifact represented by the environment digest."""

    CONTAINER_IMAGE = "container_image"
    ENVIRONMENT_LOCK = "environment_lock"


class ConformanceRedistributionStatus(str, Enum):
    """Redistribution boundary for conformance evidence and derived artifacts."""

    REDISTRIBUTABLE = "redistributable"
    METADATA_ONLY = "metadata_only"
    RESTRICTED_NO_REDISTRIBUTION = "restricted_no_redistribution"


_EXECUTED_STATUSES = {
    ConformanceExecutionStatus.PASSED,
    ConformanceExecutionStatus.FAILED,
    ConformanceExecutionStatus.INDETERMINATE,
}


def _text(value: object, name: str, *, maximum: int = MAX_TEXT_LENGTH) -> str:
    """Normalize bounded exact built-in text without caller callback dispatch."""
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    return normalized


def _identifier(value: object, name: str) -> str:
    """Normalize a two-or-more-token lower-snake-case public identifier."""
    normalized = _text(value, name, maximum=128)
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must use two-or-more-token lower snake_case")
    return normalized


def _fingerprint(value: object, name: str) -> str:
    """Validate a lowercase SHA-256 hexadecimal identity."""
    normalized = _text(value, name, maximum=64)
    if _FINGERPRINT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return normalized


def _optional_fingerprint(value: object | None, name: str) -> str | None:
    """Validate an optional lowercase SHA-256 hexadecimal identity."""
    return None if value is None else _fingerprint(value, name)


def _git_sha(value: object, name: str) -> str:
    """Validate one immutable full Git SHA-1 or SHA-256 identity."""
    normalized = _text(value, name, maximum=64)
    if _GIT_SHA_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            f"{name} must be a full lowercase 40- or 64-character Git SHA"
        )
    return normalized


def _semantic_version(value: object, name: str) -> str:
    """Validate a canonical numeric semantic version."""
    normalized = _text(value, name, maximum=64)
    if _SEMANTIC_VERSION_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a canonical semantic version")
    return normalized


def _schema_version(value: object) -> str:
    """Accept only the conformance schema implemented by this package slice."""
    normalized = _text(value, "schema_version", maximum=16)
    if normalized != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be '{SCHEMA_VERSION}'")
    return normalized


def _enum_value(value: object, enum_type: type[Enum], name: str) -> Enum:
    """Admit an exact enum member or an exact built-in serialized value."""
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise ValueError(f"{name} must be a supported {enum_type.__name__} value")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = [member.value for member in enum_type]
        raise ValueError(f"{name} must be one of {choices}") from exc


def _canonical_json(payload: object) -> str:
    """Serialize source-free conformance content deterministically."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(payload: object) -> str:
    """Return SHA-256 over canonical UTF-8 JSON."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ConformanceRunProvenance:
    """Reproducibility metadata for one isolated conformance inventory run."""

    harness_commit: str
    environment_sha256: str
    environment_kind: ConformanceEnvironmentKind
    operating_system: str
    architecture: str
    rng_algorithm: str
    rng_seeds: tuple[int, ...]
    mapping_schema_version: str
    mapping_sha256: str
    model_configuration_sha256: str
    convergence_controls_sha256: str
    tolerance_sha256: str
    tolerance_rationale: str
    raw_output_sha256: str | None
    normalized_output_sha256: str | None
    license_classification: str
    redistribution_status: ConformanceRedistributionStatus

    def __post_init__(self) -> None:
        """Normalize exact reproducibility identities without raw result content."""
        if type(self) is not ConformanceRunProvenance:
            raise ValueError(
                "ConformanceRunProvenance must be an exact package record"
            )
        object.__setattr__(
            self,
            "harness_commit",
            _git_sha(self.harness_commit, "harness_commit"),
        )
        for field_name in (
            "environment_sha256",
            "mapping_sha256",
            "model_configuration_sha256",
            "convergence_controls_sha256",
            "tolerance_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _fingerprint(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "environment_kind",
            _enum_value(
                self.environment_kind,
                ConformanceEnvironmentKind,
                "environment_kind",
            ),
        )
        object.__setattr__(
            self,
            "operating_system",
            _text(self.operating_system, "operating_system", maximum=128),
        )
        object.__setattr__(
            self,
            "architecture",
            _text(self.architecture, "architecture", maximum=128),
        )
        object.__setattr__(
            self,
            "rng_algorithm",
            _text(self.rng_algorithm, "rng_algorithm"),
        )
        if type(self.rng_seeds) not in {tuple, list}:
            raise ValueError("rng_seeds must be a list or tuple")
        if len(self.rng_seeds) > MAX_COLLECTION_VALUES:
            raise ValueError(
                f"rng_seeds must contain at most {MAX_COLLECTION_VALUES} values"
            )
        seeds = tuple(self.rng_seeds)
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("rng_seeds must contain non-negative built-in integers")
        object.__setattr__(self, "rng_seeds", seeds)
        object.__setattr__(
            self,
            "mapping_schema_version",
            _semantic_version(self.mapping_schema_version, "mapping_schema_version"),
        )
        object.__setattr__(
            self,
            "tolerance_rationale",
            _text(self.tolerance_rationale, "tolerance_rationale"),
        )
        for field_name in ("raw_output_sha256", "normalized_output_sha256"):
            object.__setattr__(
                self,
                field_name,
                _optional_fingerprint(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "license_classification",
            _identifier(self.license_classification, "license_classification"),
        )
        object.__setattr__(
            self,
            "redistribution_status",
            _enum_value(
                self.redistribution_status,
                ConformanceRedistributionStatus,
                "redistribution_status",
            ),
        )

    def to_manifest(self) -> dict[str, object]:
        """Return revalidated source-free reproducibility metadata."""
        sealed = _run_provenance(self)
        return {
            "architecture": sealed.architecture,
            "convergence_controls_sha256": sealed.convergence_controls_sha256,
            "environment_kind": sealed.environment_kind.value,
            "environment_sha256": sealed.environment_sha256,
            "harness_commit": sealed.harness_commit,
            "license_classification": sealed.license_classification,
            "mapping_schema_version": sealed.mapping_schema_version,
            "mapping_sha256": sealed.mapping_sha256,
            "model_configuration_sha256": sealed.model_configuration_sha256,
            "normalized_output_sha256": sealed.normalized_output_sha256,
            "operating_system": sealed.operating_system,
            "raw_output_sha256": sealed.raw_output_sha256,
            "redistribution_status": sealed.redistribution_status.value,
            "rng_algorithm": sealed.rng_algorithm,
            "rng_seeds": list(sealed.rng_seeds),
            "tolerance_rationale": sealed.tolerance_rationale,
            "tolerance_sha256": sealed.tolerance_sha256,
        }


def _run_provenance(value: object) -> ConformanceRunProvenance:
    """Revalidate one nested run-provenance record before manifest hashing."""
    if type(value) is not ConformanceRunProvenance:
        raise ValueError("run_provenance must be a ConformanceRunProvenance")
    return ConformanceRunProvenance(
        harness_commit=value.harness_commit,
        environment_sha256=value.environment_sha256,
        environment_kind=value.environment_kind,
        operating_system=value.operating_system,
        architecture=value.architecture,
        rng_algorithm=value.rng_algorithm,
        rng_seeds=value.rng_seeds,
        mapping_schema_version=value.mapping_schema_version,
        mapping_sha256=value.mapping_sha256,
        model_configuration_sha256=value.model_configuration_sha256,
        convergence_controls_sha256=value.convergence_controls_sha256,
        tolerance_sha256=value.tolerance_sha256,
        tolerance_rationale=value.tolerance_rationale,
        raw_output_sha256=value.raw_output_sha256,
        normalized_output_sha256=value.normalized_output_sha256,
        license_classification=value.license_classification,
        redistribution_status=value.redistribution_status,
    )


@dataclass(frozen=True, slots=True)
class ComparisonEngine:
    """Versioned identity of one isolated independent comparison engine."""

    engine_id: str
    engine_version: str
    source_reference: str
    license_classification: str

    def __post_init__(self) -> None:
        """Normalize a sealed package-owned engine record."""
        if type(self) is not ComparisonEngine:
            raise ValueError("ComparisonEngine must be an exact package record")
        object.__setattr__(self, "engine_id", _identifier(self.engine_id, "engine_id"))
        object.__setattr__(
            self,
            "engine_version",
            _text(self.engine_version, "engine_version", maximum=128),
        )
        object.__setattr__(
            self,
            "source_reference",
            _text(self.source_reference, "source_reference", maximum=512),
        )
        object.__setattr__(
            self,
            "license_classification",
            _identifier(self.license_classification, "license_classification"),
        )

    def to_manifest(self) -> dict[str, str]:
        """Return a revalidated JSON-compatible external-engine identity."""
        sealed = _engine(self)
        return {
            "engine_id": sealed.engine_id,
            "engine_version": sealed.engine_version,
            "license_classification": sealed.license_classification,
            "source_reference": sealed.source_reference,
        }


def _engine(value: object, name: str = "engine") -> ComparisonEngine:
    """Revalidate one exact package-owned comparison-engine record."""
    if type(value) is not ComparisonEngine:
        raise ValueError(f"{name} must be a ComparisonEngine")
    return ComparisonEngine(
        engine_id=value.engine_id,
        engine_version=value.engine_version,
        source_reference=value.source_reference,
        license_classification=value.license_classification,
    )


@dataclass(frozen=True, slots=True)
class ConformanceEvidence:
    """Immutable provenance for one independent conformance execution or plan."""

    evidence_id: str
    engine: ComparisonEngine
    layer: ConformanceLayer
    execution_status: ConformanceExecutionStatus
    parameter_mapping_version: str
    parameter_mapping_sha256: str
    fixture_sha256: str
    environment_sha256: str
    artifact_sha256: str | None = None
    limitation: str | None = None

    def __post_init__(self) -> None:
        """Normalize a sealed package-owned evidence record."""
        if type(self) is not ConformanceEvidence:
            raise ValueError("ConformanceEvidence must be an exact package record")
        object.__setattr__(
            self,
            "evidence_id",
            _identifier(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(self, "engine", _engine(self.engine))
        object.__setattr__(
            self,
            "layer",
            _enum_value(self.layer, ConformanceLayer, "layer"),
        )
        object.__setattr__(
            self,
            "execution_status",
            _enum_value(
                self.execution_status,
                ConformanceExecutionStatus,
                "execution_status",
            ),
        )
        object.__setattr__(
            self,
            "parameter_mapping_version",
            _semantic_version(
                self.parameter_mapping_version,
                "parameter_mapping_version",
            ),
        )
        for field_name in (
            "parameter_mapping_sha256",
            "fixture_sha256",
            "environment_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _fingerprint(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "artifact_sha256",
            _optional_fingerprint(self.artifact_sha256, "artifact_sha256"),
        )
        if self.limitation is not None:
            object.__setattr__(
                self,
                "limitation",
                _text(self.limitation, "limitation"),
            )

        executed = self.execution_status in _EXECUTED_STATUSES
        if executed and self.artifact_sha256 is None:
            raise ValueError("artifact_sha256 is required for an executed status")
        if not executed and self.artifact_sha256 is not None:
            raise ValueError("artifact_sha256 must be omitted for a nonexecuted status")

    def to_manifest(self) -> dict[str, object]:
        """Return a revalidated JSON-compatible evidence projection."""
        sealed = _evidence(self)
        return {
            "artifact_sha256": sealed.artifact_sha256,
            "engine": sealed.engine.to_manifest(),
            "environment_sha256": sealed.environment_sha256,
            "evidence_id": sealed.evidence_id,
            "execution_status": sealed.execution_status.value,
            "fixture_sha256": sealed.fixture_sha256,
            "layer": sealed.layer.value,
            "limitation": sealed.limitation,
            "parameter_mapping_sha256": sealed.parameter_mapping_sha256,
            "parameter_mapping_version": sealed.parameter_mapping_version,
        }


def _evidence(value: object, name: str = "evidence") -> ConformanceEvidence:
    """Revalidate one exact package-owned conformance-evidence record."""
    if type(value) is not ConformanceEvidence:
        raise ValueError(f"{name} must be a ConformanceEvidence")
    return ConformanceEvidence(
        evidence_id=value.evidence_id,
        engine=value.engine,
        layer=value.layer,
        execution_status=value.execution_status,
        parameter_mapping_version=value.parameter_mapping_version,
        parameter_mapping_sha256=value.parameter_mapping_sha256,
        fixture_sha256=value.fixture_sha256,
        environment_sha256=value.environment_sha256,
        artifact_sha256=value.artifact_sha256,
        limitation=value.limitation,
    )


def _evidence_values(values: object) -> tuple[ConformanceEvidence, ...]:
    """Normalize bounded exact package-owned evidence records."""
    if type(values) not in {tuple, list}:
        raise ValueError("evidence must be a list or tuple")
    if len(values) > MAX_COLLECTION_VALUES:
        raise ValueError(
            f"evidence must contain at most {MAX_COLLECTION_VALUES} values"
        )
    normalized: list[ConformanceEvidence] = []
    for index, value in enumerate(values):
        normalized.append(_evidence(value, f"evidence[{index}]"))
    if len({row.evidence_id for row in normalized}) != len(normalized):
        raise ValueError("evidence_id values must be unique")
    return tuple(sorted(normalized, key=lambda row: row.evidence_id))


@dataclass(frozen=True, slots=True)
class ConformanceCapability:
    """One public numerical estimand and its independent comparison coverage."""

    capability_id: str
    public_entrypoint: str
    estimand: str
    likelihood_family: str
    parameterization: str
    identification: str
    comparison_scope: str
    coverage_status: ConformanceCoverageStatus
    evidence: tuple[ConformanceEvidence, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize a sealed package-owned capability record."""
        if type(self) is not ConformanceCapability:
            raise ValueError("ConformanceCapability must be an exact package record")
        object.__setattr__(
            self,
            "capability_id",
            _identifier(self.capability_id, "capability_id"),
        )
        for field_name in (
            "public_entrypoint",
            "estimand",
            "likelihood_family",
            "parameterization",
            "identification",
            "comparison_scope",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "coverage_status",
            _enum_value(
                self.coverage_status,
                ConformanceCoverageStatus,
                "coverage_status",
            ),
        )
        evidence = _evidence_values(self.evidence)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

        if self.coverage_status in {
            ConformanceCoverageStatus.COVERED,
            ConformanceCoverageStatus.PARTIALLY_COVERED,
        } and not evidence:
            raise ValueError("covered capability requires evidence")
        if self.coverage_status in {
            ConformanceCoverageStatus.COVERED,
            ConformanceCoverageStatus.PARTIALLY_COVERED,
        } and not any(row.execution_status in _EXECUTED_STATUSES for row in evidence):
            raise ValueError("covered capability requires executed evidence")
        if self.coverage_status in {
            ConformanceCoverageStatus.NO_INDEPENDENT_ENGINE,
            ConformanceCoverageStatus.NOT_COMPARABLE,
        } and evidence:
            raise ValueError(
                f"{self.coverage_status.value} must not contain comparison evidence"
            )
        if self.coverage_status is ConformanceCoverageStatus.PLANNED and any(
            row.execution_status
            not in {
                ConformanceExecutionStatus.NOT_EXECUTED,
                ConformanceExecutionStatus.NOT_APPLICABLE,
            }
            for row in evidence
        ):
            raise ValueError("planned coverage may contain only nonexecuted evidence")

    def to_manifest(self) -> dict[str, object]:
        """Return a revalidated JSON-compatible capability projection."""
        sealed = _capability(self)
        return {
            "capability_id": sealed.capability_id,
            "comparison_scope": sealed.comparison_scope,
            "coverage_status": sealed.coverage_status.value,
            "estimand": sealed.estimand,
            "evidence": [row.to_manifest() for row in sealed.evidence],
            "identification": sealed.identification,
            "likelihood_family": sealed.likelihood_family,
            "parameterization": sealed.parameterization,
            "public_entrypoint": sealed.public_entrypoint,
            "schema_version": sealed.schema_version,
        }


def _capability(value: object, name: str = "capability") -> ConformanceCapability:
    """Revalidate one exact package-owned conformance-capability record."""
    if type(value) is not ConformanceCapability:
        raise ValueError(f"{name} must be a ConformanceCapability")
    return ConformanceCapability(
        capability_id=value.capability_id,
        public_entrypoint=value.public_entrypoint,
        estimand=value.estimand,
        likelihood_family=value.likelihood_family,
        parameterization=value.parameterization,
        identification=value.identification,
        comparison_scope=value.comparison_scope,
        coverage_status=value.coverage_status,
        evidence=value.evidence,
        schema_version=value.schema_version,
    )


def _capability_values(values: object) -> tuple[ConformanceCapability, ...]:
    """Normalize a bounded non-empty set of exact package-owned capabilities."""
    if type(values) not in {tuple, list}:
        raise ValueError("capabilities must be a list or tuple")
    if not 1 <= len(values) <= MAX_COLLECTION_VALUES:
        raise ValueError(
            f"capabilities must contain between 1 and {MAX_COLLECTION_VALUES} values"
        )
    normalized: list[ConformanceCapability] = []
    for index, value in enumerate(values):
        normalized.append(_capability(value, f"capabilities[{index}]"))
    if len({row.capability_id for row in normalized}) != len(normalized):
        raise ValueError("capability_id values must be unique")
    return tuple(sorted(normalized, key=lambda row: row.capability_id))


def _has_executed_evidence(capabilities: tuple[ConformanceCapability, ...]) -> bool:
    """Return whether a sealed capability set contains any executed evidence."""
    return any(
        evidence.execution_status in _EXECUTED_STATUSES
        for capability in capabilities
        for evidence in capability.evidence
    )


def _manifest_mapping(
    value: object,
    name: str,
    expected_keys: frozenset[str],
) -> dict[str, object]:
    """Admit an exact persisted mapping with an exact built-in string key set."""
    if type(value) is not dict:
        raise ValueError(f"{name} must be a built-in dictionary")
    keys = tuple(dict.keys(value))
    if any(type(key) is not str for key in keys):
        raise ValueError(f"{name} keys must be built-in strings")
    if frozenset(keys) != expected_keys:
        raise ValueError(f"{name} keys must be exactly {sorted(expected_keys)}")
    return value


def _manifest_list(value: object, name: str) -> list[object]:
    """Admit an exact persisted JSON list before iteration."""
    if type(value) is not list:
        raise ValueError(f"{name} must be a built-in list")
    return value


def _manifest_string(mapping: dict[str, object], key: str, name: str) -> str:
    """Read one persisted exact built-in string without subclass callbacks."""
    value = dict.__getitem__(mapping, key)
    if type(value) is not str:
        raise ValueError(f"{name}.{key} must be a string")
    return value


def _manifest_optional_string(
    mapping: dict[str, object], key: str, name: str
) -> str | None:
    """Read one persisted optional exact built-in string."""
    value = dict.__getitem__(mapping, key)
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{name}.{key} must be a string or null")
    return value


_ENGINE_MANIFEST_KEYS = frozenset(
    {
        "engine_id",
        "engine_version",
        "license_classification",
        "source_reference",
    }
)
_EVIDENCE_MANIFEST_KEYS = frozenset(
    {
        "artifact_sha256",
        "engine",
        "environment_sha256",
        "evidence_id",
        "execution_status",
        "fixture_sha256",
        "layer",
        "limitation",
        "parameter_mapping_sha256",
        "parameter_mapping_version",
    }
)
_CAPABILITY_MANIFEST_KEYS = frozenset(
    {
        "capability_id",
        "comparison_scope",
        "coverage_status",
        "estimand",
        "evidence",
        "identification",
        "likelihood_family",
        "parameterization",
        "public_entrypoint",
        "schema_version",
    }
)
_RUN_PROVENANCE_MANIFEST_KEYS = frozenset(
    {
        "architecture",
        "convergence_controls_sha256",
        "environment_kind",
        "environment_sha256",
        "harness_commit",
        "license_classification",
        "mapping_schema_version",
        "mapping_sha256",
        "model_configuration_sha256",
        "normalized_output_sha256",
        "operating_system",
        "raw_output_sha256",
        "redistribution_status",
        "rng_algorithm",
        "rng_seeds",
        "tolerance_rationale",
        "tolerance_sha256",
    }
)
_INVENTORY_MANIFEST_KEYS = frozenset(
    {
        "capabilities",
        "inventory_fingerprint",
        "package_version",
        "run_provenance",
        "schema_version",
        "source_commit",
    }
)


def _engine_from_manifest(value: object) -> ComparisonEngine:
    """Rehydrate one persisted comparison-engine identity."""
    manifest = _manifest_mapping(value, "engine manifest", _ENGINE_MANIFEST_KEYS)
    return ComparisonEngine(
        engine_id=_manifest_string(manifest, "engine_id", "engine manifest"),
        engine_version=_manifest_string(manifest, "engine_version", "engine manifest"),
        source_reference=_manifest_string(
            manifest, "source_reference", "engine manifest"
        ),
        license_classification=_manifest_string(
            manifest, "license_classification", "engine manifest"
        ),
    )


def _evidence_from_manifest(value: object) -> ConformanceEvidence:
    """Rehydrate one persisted conformance-evidence record."""
    manifest = _manifest_mapping(value, "evidence manifest", _EVIDENCE_MANIFEST_KEYS)
    return ConformanceEvidence(
        evidence_id=_manifest_string(manifest, "evidence_id", "evidence manifest"),
        engine=_engine_from_manifest(dict.__getitem__(manifest, "engine")),
        layer=_manifest_string(manifest, "layer", "evidence manifest"),
        execution_status=_manifest_string(
            manifest, "execution_status", "evidence manifest"
        ),
        parameter_mapping_version=_manifest_string(
            manifest, "parameter_mapping_version", "evidence manifest"
        ),
        parameter_mapping_sha256=_manifest_string(
            manifest, "parameter_mapping_sha256", "evidence manifest"
        ),
        fixture_sha256=_manifest_string(
            manifest, "fixture_sha256", "evidence manifest"
        ),
        environment_sha256=_manifest_string(
            manifest, "environment_sha256", "evidence manifest"
        ),
        artifact_sha256=_manifest_optional_string(
            manifest, "artifact_sha256", "evidence manifest"
        ),
        limitation=_manifest_optional_string(manifest, "limitation", "evidence manifest"),
    )


def _capability_from_manifest(value: object) -> ConformanceCapability:
    """Rehydrate one persisted public-capability conformance record."""
    manifest = _manifest_mapping(
        value, "capability manifest", _CAPABILITY_MANIFEST_KEYS
    )
    evidence_values = _manifest_list(
        dict.__getitem__(manifest, "evidence"), "evidence"
    )
    return ConformanceCapability(
        capability_id=_manifest_string(
            manifest, "capability_id", "capability manifest"
        ),
        public_entrypoint=_manifest_string(
            manifest, "public_entrypoint", "capability manifest"
        ),
        estimand=_manifest_string(manifest, "estimand", "capability manifest"),
        likelihood_family=_manifest_string(
            manifest, "likelihood_family", "capability manifest"
        ),
        parameterization=_manifest_string(
            manifest, "parameterization", "capability manifest"
        ),
        identification=_manifest_string(
            manifest, "identification", "capability manifest"
        ),
        comparison_scope=_manifest_string(
            manifest, "comparison_scope", "capability manifest"
        ),
        coverage_status=_manifest_string(
            manifest, "coverage_status", "capability manifest"
        ),
        evidence=tuple(_evidence_from_manifest(row) for row in evidence_values),
        schema_version=_manifest_string(
            manifest, "schema_version", "capability manifest"
        ),
    )


def _run_provenance_from_manifest(value: object) -> ConformanceRunProvenance:
    """Rehydrate one persisted run-provenance record."""
    manifest = _manifest_mapping(
        value, "run_provenance manifest", _RUN_PROVENANCE_MANIFEST_KEYS
    )
    seeds = _manifest_list(dict.__getitem__(manifest, "rng_seeds"), "rng_seeds")
    return ConformanceRunProvenance(
        harness_commit=_manifest_string(
            manifest, "harness_commit", "run_provenance manifest"
        ),
        environment_sha256=_manifest_string(
            manifest, "environment_sha256", "run_provenance manifest"
        ),
        environment_kind=_manifest_string(
            manifest, "environment_kind", "run_provenance manifest"
        ),
        operating_system=_manifest_string(
            manifest, "operating_system", "run_provenance manifest"
        ),
        architecture=_manifest_string(
            manifest, "architecture", "run_provenance manifest"
        ),
        rng_algorithm=_manifest_string(
            manifest, "rng_algorithm", "run_provenance manifest"
        ),
        rng_seeds=seeds,
        mapping_schema_version=_manifest_string(
            manifest, "mapping_schema_version", "run_provenance manifest"
        ),
        mapping_sha256=_manifest_string(
            manifest, "mapping_sha256", "run_provenance manifest"
        ),
        model_configuration_sha256=_manifest_string(
            manifest, "model_configuration_sha256", "run_provenance manifest"
        ),
        convergence_controls_sha256=_manifest_string(
            manifest, "convergence_controls_sha256", "run_provenance manifest"
        ),
        tolerance_sha256=_manifest_string(
            manifest, "tolerance_sha256", "run_provenance manifest"
        ),
        tolerance_rationale=_manifest_string(
            manifest, "tolerance_rationale", "run_provenance manifest"
        ),
        raw_output_sha256=_manifest_optional_string(
            manifest, "raw_output_sha256", "run_provenance manifest"
        ),
        normalized_output_sha256=_manifest_optional_string(
            manifest, "normalized_output_sha256", "run_provenance manifest"
        ),
        license_classification=_manifest_string(
            manifest, "license_classification", "run_provenance manifest"
        ),
        redistribution_status=_manifest_string(
            manifest, "redistribution_status", "run_provenance manifest"
        ),
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while rejecting duplicate member names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    """Reject non-finite JSON extensions unsupported by the manifest contract."""
    raise ValueError(f"manifest JSON contains unsupported constant: {value}")


def _validate_raw_manifest_depth(content: str) -> None:
    """Reject JSON strings whose nesting depth exceeds the maximum budget."""
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
            if depth > MAX_MANIFEST_NESTING:
                raise ValueError("manifest JSON nesting is too deep")
        elif char in "]}":
            depth -= 1


def _validate_manifest_nesting(value: object) -> None:
    """Reject parsed JSON containers deeper than the replay contract allows."""
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if type(current) is dict:
            children = dict.values(current)
        elif type(current) is list:
            children = current
        else:
            continue
        if depth >= MAX_MANIFEST_NESTING:
            raise ValueError("manifest JSON nesting is too deep")
        stack.extend((child, depth + 1) for child in children)


@dataclass(frozen=True, slots=True)
class ConformanceInventory:
    """Content-addressed inventory of independent conformance coverage."""

    package_version: str
    source_commit: str
    capabilities: tuple[ConformanceCapability, ...]
    schema_version: str = SCHEMA_VERSION
    run_provenance: ConformanceRunProvenance | None = None

    def __post_init__(self) -> None:
        """Normalize a sealed package-owned inventory record."""
        if type(self) is not ConformanceInventory:
            raise ValueError("ConformanceInventory must be an exact package record")
        object.__setattr__(
            self,
            "package_version",
            _semantic_version(self.package_version, "package_version"),
        )
        object.__setattr__(
            self,
            "source_commit",
            _git_sha(self.source_commit, "source_commit"),
        )
        capabilities = _capability_values(self.capabilities)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

        provenance = None
        if self.run_provenance is not None:
            provenance = _run_provenance(self.run_provenance)
            object.__setattr__(self, "run_provenance", provenance)

        if _has_executed_evidence(capabilities):
            if provenance is None:
                raise ValueError("run_provenance is required for executed evidence")
            if provenance.raw_output_sha256 is None:
                raise ValueError("raw_output_sha256 is required for executed evidence")
            if provenance.normalized_output_sha256 is None:
                raise ValueError(
                    "normalized_output_sha256 is required for executed evidence"
                )

    def _manifest_without_fingerprint(self) -> dict[str, object]:
        """Return revalidated content used to derive immutable inventory identity."""
        sealed = _inventory(self)
        return {
            "capabilities": [row.to_manifest() for row in sealed.capabilities],
            "package_version": sealed.package_version,
            "run_provenance": (
                None
                if sealed.run_provenance is None
                else sealed.run_provenance.to_manifest()
            ),
            "schema_version": sealed.schema_version,
            "source_commit": sealed.source_commit,
        }

    @property
    def inventory_fingerprint(self) -> str:
        """Return SHA-256 identity of the normalized source-free inventory."""
        return _sha256(self._manifest_without_fingerprint())

    def to_manifest(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible source-free inventory."""
        payload = self._manifest_without_fingerprint()
        payload["inventory_fingerprint"] = _sha256(payload)
        return payload

    @classmethod
    def from_manifest(cls, value: object) -> ConformanceInventory:
        """Strictly rehydrate one canonical persisted inventory manifest."""
        if cls is not ConformanceInventory:
            raise ValueError("ConformanceInventory replay requires the exact package class")
        manifest = _manifest_mapping(value, "manifest", _INVENTORY_MANIFEST_KEYS)
        capabilities = _manifest_list(
            dict.__getitem__(manifest, "capabilities"), "capabilities"
        )
        run_value = dict.__getitem__(manifest, "run_provenance")
        run_provenance = (
            None if run_value is None else _run_provenance_from_manifest(run_value)
        )
        supplied_fingerprint = _fingerprint(
            _manifest_string(manifest, "inventory_fingerprint", "manifest"),
            "inventory_fingerprint",
        )
        replayed = cls(
            package_version=_manifest_string(manifest, "package_version", "manifest"),
            source_commit=_manifest_string(manifest, "source_commit", "manifest"),
            capabilities=tuple(
                _capability_from_manifest(row) for row in capabilities
            ),
            schema_version=_manifest_string(manifest, "schema_version", "manifest"),
            run_provenance=run_provenance,
        )
        if replayed.inventory_fingerprint != supplied_fingerprint:
            raise ValueError("inventory_fingerprint does not match canonical manifest")
        if replayed.to_manifest() != manifest:
            raise ValueError("manifest must already be canonical")
        return replayed

    @classmethod
    def from_json(cls, value: object) -> ConformanceInventory:
        """Strictly parse bounded JSON and replay its canonical inventory manifest."""
        if type(value) is not str:
            raise ValueError("manifest JSON must be a string")
        try:
            payload = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("manifest JSON must be UTF-8 encodable") from exc
        if len(payload) > MAX_MANIFEST_JSON_BYTES:
            raise ValueError(
                f"manifest JSON must contain at most {MAX_MANIFEST_JSON_BYTES} bytes"
            )
        _validate_raw_manifest_depth(value)
        try:
            parsed = json.loads(
                value,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise ValueError("manifest JSON must contain valid JSON") from exc
        except RecursionError as exc:
            raise ValueError("manifest JSON nesting is too deep") from exc
        _validate_manifest_nesting(parsed)
        return cls.from_manifest(parsed)


def _inventory(value: object) -> ConformanceInventory:
    """Revalidate one exact package-owned conformance-inventory record."""
    if type(value) is not ConformanceInventory:
        raise ValueError("inventory must be a ConformanceInventory")
    return ConformanceInventory(
        package_version=value.package_version,
        source_commit=value.source_commit,
        capabilities=value.capabilities,
        schema_version=value.schema_version,
        run_provenance=value.run_provenance,
    )


__all__ = [
    "ComparisonEngine",
    "ConformanceCapability",
    "ConformanceCoverageStatus",
    "ConformanceEnvironmentKind",
    "ConformanceEvidence",
    "ConformanceExecutionStatus",
    "ConformanceInventory",
    "ConformanceLayer",
    "ConformanceRedistributionStatus",
    "ConformanceRunProvenance",
]
