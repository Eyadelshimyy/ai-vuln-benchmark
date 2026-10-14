"""Persisted-manifest replay regressions for cross-engine conformance."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.cross_engine_conformance import (
    ComparisonEngine,
    ConformanceCapability,
    ConformanceCoverageStatus,
    ConformanceEvidence,
    ConformanceExecutionStatus,
    ConformanceInventory,
    ConformanceLayer,
    ConformanceRunProvenance,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SOURCE_COMMIT = "0" * 40


class _HostileDict(dict[str, object]):
    """Dictionary subtype that records unsafe replay callback dispatch."""

    callbacks = 0

    def __iter__(self):  # type: ignore[override]
        """Reject iteration before exact built-in mapping admission."""
        type(self).callbacks += 1
        raise AssertionError("hostile dictionary iteration executed")

    def __getitem__(self, key: str) -> object:
        """Reject lookup before exact built-in mapping admission."""
        type(self).callbacks += 1
        raise AssertionError("hostile dictionary lookup executed")


class _HostileList(list[object]):
    """List subtype that records unsafe replay callback dispatch."""

    callbacks = 0

    def __iter__(self):  # type: ignore[override]
        """Reject iteration before exact built-in list admission."""
        type(self).callbacks += 1
        raise AssertionError("hostile list iteration executed")


class _HostileText(str):
    """String subtype that records unsafe JSON-input callback dispatch."""

    callbacks = 0

    def __len__(self) -> int:
        """Reject length access before exact built-in text admission."""
        type(self).callbacks += 1
        raise AssertionError("hostile string length executed")


def _engine() -> ComparisonEngine:
    """Return one source-free external-engine identity."""
    return ComparisonEngine(
        engine_id="mirt_engine",
        engine_version="1.44.0",
        source_reference="doi:10.18637/jss.v048.i06",
        license_classification="gpl_3_reviewed",
    )


def _evidence() -> ConformanceEvidence:
    """Return one planned nonexecuted evidence record."""
    return ConformanceEvidence(
        evidence_id="rasch_probability_plan",
        engine=_engine(),
        layer=ConformanceLayer.FIXED_PARAMETER_EQUATION,
        execution_status=ConformanceExecutionStatus.NOT_EXECUTED,
        parameter_mapping_version="1.0.0",
        parameter_mapping_sha256=_SHA_A,
        fixture_sha256=_SHA_B,
        environment_sha256=_SHA_C,
        artifact_sha256=None,
        limitation="execution scheduled in isolated validation environment",
    )


def _capability(*, capability_id: str = "rasch_probability") -> ConformanceCapability:
    """Return one planned capability carrying nested engine/evidence metadata."""
    return ConformanceCapability(
        capability_id=capability_id,
        public_entrypoint="fast_mlsirm.rasch.probability",
        estimand="Dichotomous Rasch response probability",
        likelihood_family="bernoulli_logit",
        parameterization="difficulty with unit discrimination",
        identification="latent location fixed by the compared fixture",
        comparison_scope="fixed-parameter equation conformance",
        coverage_status=ConformanceCoverageStatus.PLANNED,
        evidence=(_evidence(),),
    )


def _provenance() -> ConformanceRunProvenance:
    """Return one canonical source-free run provenance record."""
    return ConformanceRunProvenance(
        harness_commit=_SOURCE_COMMIT,
        environment_sha256=_SHA_A,
        environment_kind="environment_lock",
        operating_system="linux",
        architecture="x86_64",
        rng_algorithm="pcg64_dxsm",
        rng_seeds=(17, 23),
        mapping_schema_version="1.0.0",
        mapping_sha256=_SHA_B,
        model_configuration_sha256=_SHA_C,
        convergence_controls_sha256=_SHA_D,
        tolerance_sha256=_SHA_A,
        tolerance_rationale="fixed-parameter double-precision comparison",
        raw_output_sha256=None,
        normalized_output_sha256=None,
        license_classification="synthetic_or_open",
        redistribution_status="metadata_only",
    )


def _inventory(*, two_capabilities: bool = False) -> ConformanceInventory:
    """Return one canonical nonexecuted inventory for replay tests."""
    capabilities = [_capability()]
    if two_capabilities:
        capabilities.append(_capability(capability_id="rasch_log_likelihood"))
    return ConformanceInventory(
        package_version="0.8.0",
        source_commit=_SOURCE_COMMIT,
        capabilities=capabilities,
        run_provenance=_provenance(),
    )


def _canonical_json(payload: object) -> str:
    """Return the same deterministic JSON shape emitted by conformance tooling."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _nested_evidence_manifest(manifest: dict[str, object]) -> dict[str, object]:
    """Return the first nested evidence mapping from a canonical manifest."""
    capabilities = manifest["capabilities"]
    assert type(capabilities) is list
    capability = capabilities[0]
    assert type(capability) is dict
    evidence = capability["evidence"]
    assert type(evidence) is list
    row = evidence[0]
    assert type(row) is dict
    return row


def test_manifest_round_trip_revalidates_exact_inventory_identity() -> None:
    """A canonical persisted manifest rehydrates through package-owned validation."""
    inventory = _inventory()
    manifest = inventory.to_manifest()

    replayed = ConformanceInventory.from_manifest(manifest)

    assert replayed == inventory
    assert replayed.inventory_fingerprint == inventory.inventory_fingerprint
    assert replayed.to_manifest() == manifest


def test_json_round_trip_revalidates_exact_inventory_identity() -> None:
    """Canonical JSON replays to the same inventory and manifest fingerprint."""
    inventory = _inventory()
    payload = _canonical_json(inventory.to_manifest())

    replayed = ConformanceInventory.from_json(payload)

    assert replayed == inventory
    assert replayed.to_manifest() == inventory.to_manifest()


def test_manifest_replay_rejects_tampered_fingerprint() -> None:
    """A stored fingerprint must match the canonical revalidated payload."""
    manifest = _inventory().to_manifest()
    manifest["inventory_fingerprint"] = "f" * 64

    with pytest.raises(ValueError, match="inventory_fingerprint does not match"):
        ConformanceInventory.from_manifest(manifest)


@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_manifest_replay_requires_exact_root_key_set(mutation: str) -> None:
    """Unknown and missing root fields fail closed instead of being ignored."""
    manifest = _inventory().to_manifest()
    if mutation == "unknown":
        manifest["unexpected"] = "value"
    else:
        del manifest["source_commit"]

    with pytest.raises(ValueError, match="manifest keys must be exactly"):
        ConformanceInventory.from_manifest(manifest)


def test_manifest_replay_requires_exact_capability_key_set() -> None:
    """Unknown nested capability fields cannot survive replay validation."""
    manifest = _inventory().to_manifest()
    capabilities = manifest["capabilities"]
    assert type(capabilities) is list
    capability = capabilities[0]
    assert type(capability) is dict
    capability["unexpected"] = "value"

    with pytest.raises(ValueError, match="capability manifest keys must be exactly"):
        ConformanceInventory.from_manifest(manifest)


def test_manifest_replay_requires_exact_evidence_key_set() -> None:
    """Missing evidence fields fail before package reconstruction."""
    manifest = _inventory().to_manifest()
    evidence = _nested_evidence_manifest(manifest)
    del evidence["fixture_sha256"]

    with pytest.raises(ValueError, match="evidence manifest keys must be exactly"):
        ConformanceInventory.from_manifest(manifest)


def test_manifest_replay_requires_exact_engine_key_set() -> None:
    """Unknown engine identity fields cannot be ignored during replay."""
    manifest = _inventory().to_manifest()
    evidence = _nested_evidence_manifest(manifest)
    engine = evidence["engine"]
    assert type(engine) is dict
    engine["unexpected"] = "value"

    with pytest.raises(ValueError, match="engine manifest keys must be exactly"):
        ConformanceInventory.from_manifest(manifest)


def test_manifest_replay_requires_exact_run_provenance_key_set() -> None:
    """Run provenance schema drift fails closed before reconstruction."""
    manifest = _inventory().to_manifest()
    provenance = manifest["run_provenance"]
    assert type(provenance) is dict
    del provenance["architecture"]

    with pytest.raises(
        ValueError, match="run_provenance manifest keys must be exactly"
    ):
        ConformanceInventory.from_manifest(manifest)


def test_manifest_replay_rejects_hostile_mapping_without_callbacks() -> None:
    """Caller-defined mapping subtypes are rejected before iteration or lookup."""
    _HostileDict.callbacks = 0
    hostile = _HostileDict(_inventory().to_manifest())

    with pytest.raises(ValueError, match="manifest must be a built-in dictionary"):
        ConformanceInventory.from_manifest(hostile)

    assert _HostileDict.callbacks == 0


def test_manifest_replay_rejects_hostile_nested_list_without_callbacks() -> None:
    """Caller-defined list subtypes are rejected before capability iteration."""
    _HostileList.callbacks = 0
    manifest = _inventory().to_manifest()
    manifest["capabilities"] = _HostileList(manifest["capabilities"])

    with pytest.raises(ValueError, match="capabilities must be a built-in list"):
        ConformanceInventory.from_manifest(manifest)

    assert _HostileList.callbacks == 0


def test_manifest_replay_rejects_hostile_nested_text_without_callbacks() -> None:
    """Persisted text subclasses fail before normalization callbacks can run."""
    _HostileText.callbacks = 0
    manifest = _inventory().to_manifest()
    manifest["package_version"] = _HostileText("0.8.0")

    with pytest.raises(ValueError, match="manifest.package_version must be a string"):
        ConformanceInventory.from_manifest(manifest)

    assert _HostileText.callbacks == 0


def test_json_replay_rejects_duplicate_object_keys() -> None:
    """Duplicate JSON keys cannot be collapsed before provenance validation."""
    payload = '{"schema_version":"1.0","schema_version":"1.0"}'

    with pytest.raises(ValueError, match="duplicate JSON object key: schema_version"):
        ConformanceInventory.from_json(payload)


def test_json_replay_rejects_oversized_payload_before_parsing() -> None:
    """Persisted manifest parsing has a fixed pre-parse resource bound."""
    payload = " " * 1_048_577

    with pytest.raises(ValueError, match="manifest JSON must contain at most 1048576"):
        ConformanceInventory.from_json(payload)


def test_json_replay_rejects_text_subclass_before_callbacks() -> None:
    """Caller-defined JSON text subtypes are rejected before length or parsing."""
    _HostileText.callbacks = 0

    with pytest.raises(ValueError, match="manifest JSON must be a string"):
        ConformanceInventory.from_json(_HostileText("{}"))

    assert _HostileText.callbacks == 0


def test_json_replay_rejects_malformed_json() -> None:
    """Malformed JSON is translated to the persisted-manifest validation error."""
    with pytest.raises(ValueError, match="manifest JSON must contain valid JSON"):
        ConformanceInventory.from_json("{")


def test_json_replay_rejects_nonfinite_constants() -> None:
    """Python JSON extensions such as NaN are outside the persisted contract."""
    with pytest.raises(ValueError, match="unsupported constant: NaN"):
        ConformanceInventory.from_json("NaN")


def test_manifest_replay_rejects_noncanonical_normalized_text() -> None:
    """Replay cannot silently normalize a persisted signed manifest in place."""
    manifest = _inventory().to_manifest()
    provenance = manifest["run_provenance"]
    assert type(provenance) is dict
    provenance["operating_system"] = " linux "

    with pytest.raises(ValueError, match="manifest must already be canonical"):
        ConformanceInventory.from_manifest(manifest)


def test_manifest_replay_rejects_noncanonical_capability_order() -> None:
    """Persisted capability order must already match canonical signed ordering."""
    manifest = _inventory(two_capabilities=True).to_manifest()
    capabilities = manifest["capabilities"]
    assert type(capabilities) is list
    capabilities.reverse()

    with pytest.raises(ValueError, match="manifest must already be canonical"):
        ConformanceInventory.from_manifest(manifest)
