"""Sandboxed runtime, fork, and differential validation.

A missing image is unavailable. Host execution is not a fallback. The network
stays disabled unless an operator has enabled a pinned fork. Results stay
candidates or unknown and cannot mark a finding verified.

Every execution is bound to an explicit request identity that travels to the
sandbox in a read-only manifest file. The runtime must echo that identity, and
an observation is attributed to the candidate only when the identity matches.
The process exit status is authoritative: a document left by a failed process
is never evidence.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.results import DynamicResult
from app.execution.base import ExecutionConfig, ExecutionResult
from app.parsing.solidity_runtime import (
    SCHEMA,
    ForkIdentity,
    ProcessOutcome,
    RuntimeObservation,
    RuntimeRequest,
    clamp_runtime_executions,
    compare_executions,
    fork_identity,
    network_allowed,
    parse_runtime_document,
    process_outcome,
    runtime_command,
    select_observation,
)

_VERSION = "phase48"
_INTERFACE = SCHEMA
_SECRET_KEYS = frozenset({"private_key", "mnemonic", "wallet", "secret"})
_PUBLIC = ("http://", "https://", "alchemy", "infura", "fork_url")
_MAX_PAIR = 2
_MODES = frozenset({"local", "fork", "differential", "replay"})
NETWORK_NONE = "none"
NETWORK_CONTROLLED_FORK = "controlled-fork"
_MANIFEST_NAME = "runtime-input.json"


class RuntimeEngine(DiscoveryEngine):
    @property
    def engine_id(self) -> str:
        return "bugforge-runtime"

    @property
    def display_name(self) -> str:
        return "BugForge runtime validation"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset(
            {
                EngineCapability.RUNTIME_VALIDATION,
                EngineCapability.FORK_VALIDATION,
                EngineCapability.DIFFERENTIAL_VALIDATION,
                EngineCapability.RESULTS_INGESTION,
            }
        )

    def availability(self) -> EngineAvailability:
        if _runtime_image() and _docker_present() and _local_image():
            return EngineAvailability.AVAILABLE
        return EngineAvailability.UNAVAILABLE

    def version(self) -> str:
        if not _runtime_image():
            return ""
        return _INTERFACE

    def selected_capability(self, request: AnalysisRequest) -> EngineCapability:
        return EngineCapability(_capability(runtime_mode(request)))

    def start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        mode = runtime_mode(request)
        if mode == "fork":
            blocked = _fork_gate(self, request)
            if blocked is not None:
                return blocked
        if self.availability() is not EngineAvailability.AVAILABLE:
            return _status(
                self,
                request,
                ResultStatus.UNAVAILABLE,
                "runtime validation requires Docker and a local "
                "security_agent_runtime_image; host execution is not started "
                "and no image is pulled",
                capability=_capability(mode),
                observation="unavailable",
            )
        return super().start_campaign(request)

    def analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        return self.start_campaign(request)

    def _campaign_capability(self, request: AnalysisRequest) -> EngineCapability:
        return self.selected_capability(request)

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        return _validate(self, request)

    def _analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        return _validate(self, request)


def runtime_mode(request: AnalysisRequest) -> str:
    """The requested mode. Unknown text stays unknown and is rejected later."""
    explicit = request.extra.get("mode", "")
    if explicit:
        return explicit
    source = request.extra.get("source", "")
    if request.extra.get("fork") == "true" or source == "fork":
        return "fork"
    if request.extra.get("differential") == "true" or source == "differential":
        return "differential"
    return "local"


def _validate(engine: RuntimeEngine, request: AnalysisRequest) -> DynamicResult:
    mode = runtime_mode(request)
    if mode not in _MODES:
        return _status(
            engine,
            request,
            ResultStatus.UNSUPPORTED,
            "the runtime mode is not supported",
            capability="runtime_validation",
            observation="unsupported",
        )
    if _secret_requested(request):
        return _status(
            engine,
            request,
            ResultStatus.FAILED,
            "private-key and wallet material are not accepted",
            capability=_capability(mode),
            observation="unknown",
        )
    if _public_network_requested(request):
        return _status(
            engine,
            request,
            ResultStatus.FAILED,
            "public RPC and fork URLs are not used",
            capability=_capability(mode),
            observation="unknown",
        )
    identity = _fork(request) if mode == "fork" else None
    if mode == "fork":
        blocked = _fork_gate(engine, request)
        if blocked is not None:
            return blocked
    if engine.availability() is not EngineAvailability.AVAILABLE:
        return _status(
            engine,
            request,
            ResultStatus.UNAVAILABLE,
            "runtime validation requires Docker and a local security_agent_runtime_image; "
            "host execution is not started and no image is pulled",
            capability=_capability(mode),
            observation="unavailable",
        )
    spec = runtime_request(request, mode, identity)
    absent = spec.missing()
    if absent:
        return _status(
            engine,
            request,
            ResultStatus.UNSUPPORTED,
            "the runtime request has no established " + " or ".join(absent),
            capability=_capability(mode),
            observation="incomplete",
            binding="deterministic_identity_missing:" + ",".join(absent),
            spec=spec,
        )
    runs = _run_count(request, mode)
    allow_network = network_allowed(
        mode="fork" if mode == "fork" else "local",
        fork_enabled=_fork_enabled(),
        deterministic=bool(identity and identity.deterministic),
        requested_network=request.extra.get("network", ""),
    )
    observations: list[RuntimeObservation] = []
    for run in range(runs):
        observed = _execute_once(engine, request, spec, mode, allow_network, run, identity)
        if isinstance(observed, DynamicResult):
            return observed
        observations.append(observed)
    report = None
    if mode in {"differential", "replay"}:
        if len(observations) < 2:
            return _status(
                engine,
                request,
                ResultStatus.UNSUPPORTED,
                "a comparison needs two executions inside the existing cap",
                capability=_capability(mode),
                observation="incomplete",
                spec=spec,
            )
        reset = request.extra.get("state_reset", "") == "true"
        report = compare_executions(observations[0], observations[1], reset=reset)
    return _ingested(engine, request, mode, spec, observations, report, identity, allow_network)


def runtime_request(
    request: AnalysisRequest, mode: str, identity: ForkIdentity | None
) -> RuntimeRequest:
    """Collect only identity the request actually established. Nothing is inferred."""
    extra = request.extra
    underlying = "fork" if mode == "fork" else "local"
    if identity is not None and mode == "fork":
        chain_id, block, snapshot = (
            identity.chain_id,
            identity.block_number,
            identity.state_snapshot,
        )
        reference = _fork_reference(identity)
    else:
        chain_id = extra.get("chain_id", "")
        block = extra.get("block_number", "")
        snapshot = extra.get("state_snapshot", "")
        reference = ""
    return RuntimeRequest(
        mode=underlying,
        capability=_capability(mode),
        project=extra.get("project_id", ""),
        source_snapshot=extra.get("source_snapshot", ""),
        compiler_configuration=extra.get("compiler_configuration", ""),
        runtime_configuration=extra.get("runtime_configuration", ""),
        target=request.target,
        contract=request.contract,
        function_identity=extra.get("function_identity", ""),
        deployment_address=extra.get("deployment_address", ""),
        sequence_id=extra.get("sequence_id", ""),
        transaction_index=extra.get("transaction_index", ""),
        actor=extra.get("actor", ""),
        chain_id=chain_id,
        block_number=block,
        state_snapshot=snapshot,
        fork_reference=reference,
    )


def _fork_reference(identity: ForkIdentity) -> str:
    blob = "|".join(
        (identity.fork_source, identity.chain_id, identity.block_number, identity.state_snapshot)
    )
    return "fork-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _execute_once(
    engine: RuntimeEngine,
    request: AnalysisRequest,
    spec: RuntimeRequest,
    mode: str,
    allow_network: bool,
    run: int,
    identity: ForkIdentity | None,
) -> RuntimeObservation | DynamicResult:
    capability = _capability(mode)
    network = NETWORK_CONTROLLED_FORK if allow_network else NETWORK_NONE
    source = identity.fork_source if identity is not None and mode == "fork" else ""
    with (
        tempfile.TemporaryDirectory(prefix="bugforge-runtime-in-") as input_dir,
        tempfile.TemporaryDirectory(prefix="bugforge-runtime-") as directory,
    ):
        manifest = Path(input_dir) / _MANIFEST_NAME
        manifest.write_text(
            json.dumps(spec.manifest(run, fork_source=source), sort_keys=True), encoding="utf-8"
        )
        config = ExecutionConfig(
            command=runtime_command(str(manifest)),
            working_directory=input_dir,
            timeout_seconds=60,
            memory_limit_mb=512,
            cpu_limit=1.0,
            environment={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
            allow_network=allow_network,
            output_dir=directory,
            read_only_volumes={str(request.repo_root): "/bugforge-src"},
        )
        if config.allow_network != allow_network:
            return _status(
                engine,
                request,
                ResultStatus.FAILED,
                "network disabled",
                capability=capability,
                observation="unknown",
                spec=spec,
            )
        result = _execute(config)
        document = result.artifact_contents.get("runtime.json", "")
        parsed = parse_runtime_document(document) if document else None
        started = not (result.error_message and result.exit_code == -1 and not result.timed_out)
        outcome = process_outcome(
            started=started,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            document_valid=parsed is not None,
        )
        if outcome != ProcessOutcome.ACCEPTED.value or parsed is None:
            return _process_failure(
                engine, request, spec, capability, outcome, result, bool(document), network
            )
        selection = select_observation(parsed, spec, run=run)
        if selection.observation is None or selection.status != "selected":
            observation = (
                "incomplete" if selection.status in {"not_found", "incomplete"} else "unknown"
            )
            return _status(
                engine,
                request,
                ResultStatus.UNSUPPORTED,
                "the runtime output does not answer the requested sequence and transaction",
                capability=capability,
                observation=observation,
                executed=True,
                network=network,
                binding=f"{selection.status}:{selection.reason}",
                exit_code=result.exit_code,
                spec=spec,
            )
        return selection.observation


def _process_failure(
    engine: RuntimeEngine,
    request: AnalysisRequest,
    spec: RuntimeRequest,
    capability: str,
    outcome: str,
    result: ExecutionResult,
    document_present: bool,
    network: str,
) -> DynamicResult:
    if outcome == ProcessOutcome.TIMEOUT.value:
        return _status(
            engine,
            request,
            ResultStatus.TIMEOUT,
            "the runtime timed out",
            capability=capability,
            observation="incomplete",
            executed=True,
            network=network,
            exit_code=result.exit_code,
            spec=spec,
        )
    if outcome == ProcessOutcome.NOT_STARTED.value:
        return _status(
            engine,
            request,
            ResultStatus.FAILED,
            "the runtime could not start",
            capability=capability,
            observation="unknown",
            exit_code=result.exit_code,
            spec=spec,
        )
    if outcome == ProcessOutcome.TOOL_FAILURE.value:
        failed = _status(
            engine,
            request,
            ResultStatus.TOOL_FAILURE,
            "the runtime process failed; any document it left is diagnostic only",
            capability=capability,
            observation="unknown",
            executed=True,
            network=network,
            exit_code=result.exit_code,
            spec=spec,
        )
        failed.metadata["document_present"] = str(document_present).lower()
        failed.metadata["document_attributed"] = "false"
        failed.stderr = result.stderr[:500]
        return failed
    return _status(
        engine,
        request,
        ResultStatus.UNSUPPORTED,
        "the runtime output did not match bugforge-runtime-v1",
        capability=capability,
        observation="incomplete",
        executed=True,
        network=network,
        exit_code=result.exit_code,
        spec=spec,
    )


def _ingested(
    engine: RuntimeEngine,
    request: AnalysisRequest,
    mode: str,
    spec: RuntimeRequest,
    observations: list[RuntimeObservation],
    report: object,
    identity: ForkIdentity | None,
    allow_network: bool,
) -> DynamicResult:
    first = observations[0]
    classification = getattr(report, "classification", "")
    report_status = getattr(report, "status", "")
    status = report_status or ("candidate" if first.status == "observed" else "unknown")
    explanation = "runtime evidence is not verification"
    if classification:
        explanation = f"{classification}; {explanation}"
    fork_pinned = bool(
        identity
        and identity.deterministic
        and first.block_number == identity.block_number
        and first.chain_id == identity.chain_id
        and first.state_snapshot == identity.state_snapshot
    )
    local_pinned = (
        mode != "fork"
        and first.chain_id == "31337"
        and first.block_number == "1"
        and bool(first.state_snapshot)
    )
    metadata = _meta(
        request,
        capability=_capability(mode),
        observation=status,
        deterministic="true" if fork_pinned or local_pinned else "false",
        observed=True,
        network=NETWORK_CONTROLLED_FORK if allow_network else NETWORK_NONE,
        spec=spec,
    )
    metadata["executions"] = str(len(observations))
    metadata["vulnerability"] = "unknown"
    metadata["tool"] = first.tool
    metadata["tool_version"] = first.tool_version
    metadata["sequence_id"] = first.sequence_id
    metadata["transaction_index"] = first.transaction_index
    metadata["state_snapshot"] = first.state_snapshot
    metadata["transaction_success"] = first.success
    metadata["transaction_outcome"] = {"true": "succeeded", "false": "reverted"}.get(
        first.success, "unknown"
    )
    metadata["execution_ids"] = ",".join(item.execution_id for item in observations)
    metadata["state_reset"] = "true" if request.extra.get("state_reset", "") == "true" else "false"
    if report is not None and classification:
        metadata["classification"] = str(classification)
        metadata["compared"] = (
            f"{getattr(report, 'left_execution', '')}|{getattr(report, 'right_execution', '')}"
        )
    return DynamicResult(
        engine=engine.engine_id,
        engine_version=engine.version() or _INTERFACE,
        language=request.language,
        target=request.target,
        campaign_id=request.campaign_id,
        source_file=request.source_file,
        contract=first.contract,
        function=first.function_identity,
        status=ResultStatus.INGESTED,
        executed=True,
        provenance="sandbox",
        oracle_explanation=explanation,
        oracle_kind="differential" if report is not None else "runtime",
        metadata=metadata,
        runtime_evidence=report or first,
    )


def _status(
    engine: RuntimeEngine,
    request: AnalysisRequest,
    status: ResultStatus,
    explanation: str,
    *,
    capability: str,
    observation: str,
    deterministic: str = "",
    executed: bool = False,
    network: str = NETWORK_NONE,
    binding: str = "",
    exit_code: int | None = None,
    spec: RuntimeRequest | None = None,
) -> DynamicResult:
    metadata = _meta(
        request,
        capability=capability,
        observation=observation,
        deterministic=deterministic,
        network=network,
        spec=spec,
    )
    if binding:
        metadata["binding"] = binding
    return DynamicResult(
        engine=engine.engine_id,
        engine_version=engine.version(),
        language=request.language,
        target=request.target,
        status=status,
        executed=executed,
        exit_code=exit_code,
        provenance="sandbox",
        oracle_explanation=explanation,
        oracle_kind="",
        metadata=metadata,
    )


def _meta(
    request: AnalysisRequest,
    *,
    capability: str,
    observation: str,
    deterministic: str,
    observed: bool = False,
    network: str = NETWORK_NONE,
    spec: RuntimeRequest | None = None,
) -> dict[str, str]:
    if not observed:
        evidence_class = "tool_status"
    elif capability == "differential_validation":
        evidence_class = "differential"
    else:
        evidence_class = "runtime"
    metadata = {
        "verified": "false",
        "vulnerability": "unknown",
        "capability": capability,
        "evidence_class": evidence_class,
        "observation_status": observation,
        "deterministic": deterministic,
        "network": network,
        "environment": "docker-sandbox",
        "project": request.extra.get("project_id", ""),
        "source_snapshot": request.extra.get("source_snapshot", ""),
        "compiler_configuration": request.extra.get("compiler_configuration", ""),
        "llm_invoked": "false",
    }
    if spec is not None:
        metadata["runtime_mode"] = spec.mode
        metadata["request_identity"] = spec.identity_hash()
        metadata["chain_id"] = spec.chain_id
        metadata["fork_block"] = spec.block_number if spec.mode == "fork" else ""
        metadata["fork_reference"] = spec.fork_reference
    return metadata


def _fork_gate(engine: RuntimeEngine, request: AnalysisRequest) -> DynamicResult | None:
    identity = _fork(request)
    if not identity.configured:
        return _status(
            engine,
            request,
            ResultStatus.UNAVAILABLE,
            "fork mode requires explicit operator configuration",
            capability="fork_validation",
            observation="unavailable",
        )
    if not identity.deterministic:
        return _status(
            engine,
            request,
            ResultStatus.UNSUPPORTED,
            "an unpinned fork is not deterministic validation",
            capability="fork_validation",
            observation="unknown",
            deterministic="false",
        )
    return None


def _capability(mode: str) -> str:
    if mode == "fork":
        return "fork_validation"
    if mode in {"differential", "replay"}:
        return "differential_validation"
    return "runtime_validation"


def _run_count(request: AnalysisRequest, mode: str) -> int:
    requested = request.extra.get("executions", "2" if mode in {"differential", "replay"} else "1")
    try:
        count = int(requested)
    except ValueError:
        count = 1
    capped = clamp_runtime_executions(count)
    if mode in {"differential", "replay"}:
        return min(capped, _MAX_PAIR)
    return capped


def _fork(request: AnalysisRequest) -> ForkIdentity:
    enabled = _fork_enabled() and bool(_fork_source())
    return fork_identity(
        configured=enabled,
        fork_source=_fork_source() if enabled else "",
        chain_id=request.extra.get("chain_id", ""),
        block_number=request.extra.get("fork_block", ""),
        state_snapshot=request.extra.get("state_snapshot", ""),
        compiler_configuration=request.extra.get("compiler_configuration", ""),
        project=request.extra.get("project_id", ""),
        target=request.target,
        deployments=tuple(item for item in request.extra.get("deployments", "").split(",") if item),
    )


def _secret_requested(request: AnalysisRequest) -> bool:
    return any(key.lower() in _SECRET_KEYS for key in request.extra)


def _public_network_requested(request: AnalysisRequest) -> bool:
    blob = " ".join(f"{key}={value}" for key, value in request.extra.items()).lower()
    return any(mark in blob for mark in _PUBLIC)


def _execute(config: ExecutionConfig) -> ExecutionResult:
    """Run inside Docker. This function is the only execution path."""
    import asyncio

    from app.execution.docker_executor import DockerTestExecutor

    if config.allow_network and not _fork_enabled():
        return ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr="network disabled",
            duration_seconds=0.0,
            error_message="network disabled",
        )
    image = _runtime_image()
    return asyncio.run(DockerTestExecutor(image).execute(config))


def _runtime_image() -> str:
    from app.core.config import get_settings

    return get_settings().security_agent_runtime_image.strip()


def _fork_enabled() -> bool:
    from app.core.config import get_settings

    return bool(get_settings().security_agent_fork_enabled)


def _fork_source() -> str:
    from app.core.config import get_settings

    return get_settings().security_agent_fork_source.strip()


def _docker_present() -> bool:
    import shutil

    return shutil.which("docker") is not None


def _local_image() -> bool:
    from app.execution.docker_executor import local_image_present

    return local_image_present(_runtime_image())
