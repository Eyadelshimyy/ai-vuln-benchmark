"""Optional ItyFuzz exploration inside the controlled Docker sandbox.

Interface targeted: fuzzland/ityfuzz off-chain stdout, inspected 2026-09-28
from the upstream quickstart, integration test, and published campaign logs.
The command is ``ityfuzz evm -t <directory-glob>``. The directory contains
``[name].abi`` and ``[name].bin``. ItyFuzz interprets the glob itself.

Established stdout forms:

- ``Found vulnerabilities`` with optional ``Description`` and ``Trace`` sections
- non-finding logs: ``EVM Fuzzer Start``, ``[Stats #N]``, ``New Corpus Item``,
  ``Coverage Summary``, ``Instruction Covered``, ``Branch Covered``

There is no verified ``{"vulnerabilities": ..., "transactions": ...}`` schema.
JSON and any other unrecognized text produce no finding. The label recorded
as the engine version is ``ityfuzz-stdout-v1``. That label is the interface
name, not a claim that a particular semver was executed.

The executable is not vendored and is not an authority. A missing local image
is ``UNAVAILABLE``. Host ``ityfuzz`` is not a fallback, and no image is pulled.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.adapters.discovery.external import ExternalDiscoveryEngine
from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest
from app.discovery.process import ProcessResult
from app.discovery.results import DynamicFinding, DynamicResult

_LICENSE = "fuzzland/ityfuzz; source is not vendored and is not relicensed by BugForge"
_INTERFACE = "ityfuzz-stdout-v1"
_NETWORK = (
    "fork",
    "rpc",
    "onchain",
    "etherscan",
    "http://",
    "https://",
    "eth_rpc",
    "alchemy",
    "infura",
)
_MAX_TIMEOUT = 60.0
_BANNER = re.compile(r"Found vulnerabilities", re.IGNORECASE)
_DESCRIPTION = re.compile(
    r"=+\s*Description\s*=+\s*(?P<body>.*?)(?:=+\s*Trace\s*=+|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_TRACE = re.compile(r"=+\s*Trace\s*=+\s*(?P<body>.*)\Z", re.IGNORECASE | re.DOTALL)
_KNOWN_LOG = re.compile(
    r"EVM Fuzzer Start|\[Stats\s*#|New Corpus Item|Coverage Summary|Instruction Covered|Branch Covered",
    re.IGNORECASE,
)
_INSTRUCTION = re.compile(
    r"Instruction Covered:\s*(?P<value>[0-9]+(?:\.[0-9]+)?)%",
    re.IGNORECASE,
)
_BRANCH = re.compile(r"Branch Covered:\s*(?P<value>[0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)
_SENDER = re.compile(r"\[Sender\]|[└├]─")


@dataclass(frozen=True)
class ItyFuzzParse:
    status: ResultStatus
    findings: tuple[DynamicFinding, ...]
    minimized: str
    limitation: str
    coverage: dict[str, str]
    diagnostic: str = ""


class ItyFuzzEngine(ExternalDiscoveryEngine):
    binary = "ityfuzz"
    license_note = _LICENSE

    @property
    def engine_id(self) -> str:
        return "ityfuzz"

    @property
    def display_name(self) -> str:
        return "ItyFuzz"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset(
            {
                EngineCapability.FUZZING,
                EngineCapability.COVERAGE_FEEDBACK,
                EngineCapability.RESULTS_INGESTION,
            }
        )

    def availability(self) -> EngineAvailability:
        if _sandbox_image() and _docker_present() and _local_image():
            return EngineAvailability.AVAILABLE
        return EngineAvailability.UNAVAILABLE

    def version(self) -> str:
        """Interface label only. This does not execute a host or container binary."""
        if not _sandbox_image():
            return ""
        return _INTERFACE

    def _unavailable(self, request: AnalysisRequest) -> DynamicResult:
        return DynamicResult(
            engine=self.engine_id,
            language=request.language,
            target=request.target,
            status=ResultStatus.UNAVAILABLE,
            executed=False,
            provenance="sandbox",
            oracle_explanation=(
                "ItyFuzz requires Docker and a local security_agent_ityfuzz_image; "
                "host ityfuzz is not started and no image is pulled"
            ),
            metadata={
                "verified": "false",
                "environment": "docker-sandbox",
                "license": self.license_note,
                "interface": _INTERFACE,
            },
        )

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        if _rejects_network(request):
            return DynamicResult(
                engine=self.engine_id,
                language=request.language,
                target=request.target,
                status=ResultStatus.FAILED,
                executed=False,
                provenance="scope",
                oracle_explanation="public RPC and fork campaigns are rejected in the default sandbox",
                metadata={
                    "verified": "false",
                    "environment": "docker-sandbox",
                    "license": self.license_note,
                },
            )
        if self.availability() is not EngineAvailability.AVAILABLE:
            return self._unavailable(request)
        artifacts = _artifact_dir(request)
        if artifacts is None:
            return DynamicResult(
                engine=self.engine_id,
                language=request.language,
                target=request.target,
                status=ResultStatus.UNSUPPORTED,
                executed=False,
                provenance="capability",
                oracle_explanation=(
                    "a local ItyFuzz campaign needs compiled .abi and .bin files inside "
                    "the repository; BugForge does not fetch them"
                ),
                metadata={
                    "verified": "false",
                    "environment": "docker-sandbox",
                    "license": self.license_note,
                },
            )
        proc = _execute_campaign(artifacts, _timeout(request))
        return _from_process(self, request, proc)

    def collect_results(self, request: AnalysisRequest) -> DynamicResult:
        """Ingest captured stdout without requiring a sandbox image."""
        if request.extra.get("ityfuzz_output", ""):
            return self._collect_results(request)
        return super().collect_results(request)

    def _collect_results(self, request: AnalysisRequest) -> DynamicResult:
        raw = request.extra.get("ityfuzz_output", "")
        if not raw:
            return self._start_campaign(request)
        parsed = parse_ityfuzz_output(raw)
        return _ingested(self, request, parsed, raw)


def ityfuzz_command() -> list[str]:
    """Argv for the container. The glob is one argument; ItyFuzz interprets it."""
    return ["ityfuzz", "evm", "-t", "/bugforge-output/artifacts/*"]


def parse_ityfuzz_streams(stdout: str, stderr: str) -> ItyFuzzParse:
    """Keep a finding that appears in either stream. A trace is not an executable seed."""
    parts: list[ItyFuzzParse] = []
    if stdout.strip():
        parts.append(parse_ityfuzz_output(stdout))
    if stderr.strip():
        parts.append(parse_ityfuzz_output(stderr))
    if not parts:
        return parse_ityfuzz_output("")
    findings: list[DynamicFinding] = []
    seen: set[str] = set()
    coverage: dict[str, str] = {}
    diagnostics: list[str] = []
    chosen = parts[0]
    for item in parts:
        coverage.update(item.coverage)
        if item.diagnostic:
            diagnostics.append(item.diagnostic)
        if item.findings and not chosen.findings:
            chosen = item
        for finding in item.findings:
            if finding.title in seen:
                continue
            seen.add(finding.title)
            findings.append(finding)
    status = ResultStatus.INTERESTING if findings else chosen.status
    limitation = "" if findings else chosen.limitation
    return ItyFuzzParse(status, tuple(findings), "", limitation, coverage, "\n".join(diagnostics))


def parse_ityfuzz_output(text: str) -> ItyFuzzParse:
    """Parse established stdout only. Other text never becomes a finding."""
    stripped = text.strip()
    coverage = _coverage(stripped)
    if not stripped:
        return ItyFuzzParse(
            ResultStatus.EXECUTED,
            (),
            "",
            "missing output is not a vulnerability",
            {},
        )
    if _BANNER.search(stripped):
        return _from_banner(stripped, coverage)
    if _json_document(stripped):
        return ItyFuzzParse(
            ResultStatus.UNSUPPORTED,
            (),
            "",
            "ItyFuzz stdout has no vulnerabilities JSON schema; no finding was fabricated",
            {},
        )
    if _KNOWN_LOG.search(stripped):
        return ItyFuzzParse(
            ResultStatus.EXECUTED,
            (),
            "",
            "coverage and corpus logs are not vulnerabilities",
            coverage,
        )
    return ItyFuzzParse(
        ResultStatus.UNSUPPORTED,
        (),
        "",
        "unrecognized ItyFuzz output; no finding was fabricated",
        {},
    )


def exploration_authority(status: ResultStatus) -> str:
    """ItyFuzz never promotes a result to reproduced or verified."""
    if status is ResultStatus.INTERESTING:
        return "interesting"
    if status is ResultStatus.EXECUTED:
        return "executed"
    if status is ResultStatus.UNAVAILABLE:
        return "unavailable"
    if status is ResultStatus.UNSUPPORTED:
        return "unsupported"
    if status is ResultStatus.TIMEOUT:
        return "timeout"
    if status is ResultStatus.FAILED:
        return "failed"
    return "unknown"


def _from_banner(text: str, coverage: dict[str, str]) -> ItyFuzzParse:
    description = ""
    described = _DESCRIPTION.search(text)
    if described:
        description = " ".join(described.group("body").split())
    title = description.split(". ", 1)[0][:180] if description else "Found vulnerabilities"
    trace = ""
    traced = _TRACE.search(text)
    if traced and _SENDER.search(traced.group("body")):
        trace = traced.group("body").strip()[:2000]
    finding = DynamicFinding(
        detector_id="ityfuzz",
        title=title[:180],
        description=description[:500],
        status="potential",
    )
    if trace:
        description = f"{description} trace: {trace[:240]}".strip()
        finding = DynamicFinding(
            detector_id="ityfuzz",
            title=title[:180],
            description=description[:500],
            status="potential",
        )
    return ItyFuzzParse(ResultStatus.INTERESTING, (finding,), "", "", coverage, trace)


def _json_document(text: str) -> bool:
    if not text.startswith("{") and not text.startswith("["):
        return False
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _coverage(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    instruction = _INSTRUCTION.search(text)
    branch = _BRANCH.search(text)
    if instruction:
        found["percent"] = instruction.group("value")
        found["coverage_available"] = "true"
        found["instruction_percent"] = instruction.group("value")
    if branch:
        found["branch_percent"] = branch.group("value")
    return found


def _ingested(
    engine: ItyFuzzEngine, request: AnalysisRequest, parsed: ItyFuzzParse, raw: str
) -> DynamicResult:
    return DynamicResult(
        engine=engine.engine_id,
        engine_version=engine.version(),
        language=request.language,
        target=request.target,
        source_file=request.source_file,
        contract=request.contract,
        function=request.function,
        campaign_id=request.campaign_id,
        status=parsed.status,
        executed=False,
        minimized_input="",
        findings=parsed.findings,
        coverage=dict(parsed.coverage),
        stdout=raw[:4000],
        provenance="ityfuzz_output",
        oracle_explanation=parsed.limitation
        or "ingested caller-supplied ItyFuzz output; the tool was not executed",
        oracle_kind="fuzzing" if parsed.findings else "",
        metadata=_metadata(engine, request, parsed),
    )


def _from_process(
    engine: ItyFuzzEngine, request: AnalysisRequest, proc: ProcessResult
) -> DynamicResult:
    if proc.timed_out:
        return DynamicResult(
            engine=engine.engine_id,
            engine_version=engine.version(),
            language=request.language,
            target=request.target,
            campaign_id=request.campaign_id,
            status=ResultStatus.TIMEOUT,
            executed=False,
            stderr=proc.stderr[:2000],
            stdout=proc.stdout[:4000],
            provenance="ityfuzz",
            metadata=_metadata(engine, request, None),
        )
    if not proc.started:
        return DynamicResult(
            engine=engine.engine_id,
            engine_version=engine.version(),
            language=request.language,
            target=request.target,
            status=ResultStatus.FAILED,
            executed=False,
            stderr=proc.stderr[:2000],
            provenance="ityfuzz",
            metadata=_metadata(engine, request, None),
        )
    parsed = parse_ityfuzz_streams(proc.stdout or "", proc.stderr or "")
    return DynamicResult(
        engine=engine.engine_id,
        engine_version=engine.version(),
        language=request.language,
        target=request.target,
        source_file=request.source_file,
        contract=request.contract,
        function=request.function,
        campaign_id=request.campaign_id,
        status=parsed.status,
        executed=True,
        exit_code=proc.return_code,
        stdout=(proc.stdout or "")[:4000],
        stderr=(proc.stderr or "")[:2000],
        minimized_input="",
        findings=parsed.findings,
        coverage=dict(parsed.coverage),
        provenance="ityfuzz",
        oracle_explanation=parsed.limitation,
        oracle_kind="fuzzing" if parsed.findings else "",
        metadata=_metadata(engine, request, parsed),
    )


def _metadata(
    engine: ItyFuzzEngine, request: AnalysisRequest, parsed: ItyFuzzParse | None
) -> dict[str, str]:
    status = parsed.status if parsed else ResultStatus.FAILED
    return {
        "verified": "false",
        "authority": exploration_authority(status),
        "environment": "docker-sandbox",
        "evidence_class": "fuzzing",
        "interface": _INTERFACE,
        "license": engine.license_note,
        "tool_version": engine.version(),
        "campaign": request.campaign_id,
        "source_snapshot": request.extra.get("source_snapshot", ""),
        "compiler_configuration": request.extra.get("compiler_configuration", ""),
        "oracle_type": "found-vulnerabilities" if parsed and parsed.findings else "",
        "diagnostic_trace": (parsed.diagnostic[:500] if parsed else ""),
        "executable_input": "false",
        "structured": "true" if parsed and not parsed.limitation else "false",
    }


def _execute_campaign(artifacts: Path, timeout: float) -> ProcessResult:
    """Copy local artifacts into scratch and run ityfuzz with the network off."""
    import asyncio

    from app.execution.base import ExecutionConfig
    from app.execution.docker_executor import DockerTestExecutor

    image = _sandbox_image()
    command = ityfuzz_command()
    with tempfile.TemporaryDirectory(prefix="bugforge-ityfuzz-") as directory:
        scratch = Path(directory)
        destination = scratch / "artifacts"
        destination.mkdir()
        for item in artifacts.iterdir():
            if item.is_symlink() or not item.is_file():
                continue
            if item.suffix not in {".abi", ".bin"}:
                continue
            shutil.copyfile(item, destination / item.name)
        config = ExecutionConfig(
            command=command,
            working_directory=str(scratch),
            timeout_seconds=max(1, int(timeout)),
            memory_limit_mb=512,
            cpu_limit=1.0,
            environment={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
            allow_network=False,
            output_dir=str(scratch),
        )
        if config.allow_network or config.command != command:
            return ProcessResult(False, False, None, "", "network disabled", False)
        result = asyncio.run(DockerTestExecutor(image).execute(config))
        if result.timed_out:
            return ProcessResult(False, True, result.exit_code, result.stdout, result.stderr, True)
        if result.error_message and result.exit_code == -1:
            return ProcessResult(False, False, result.exit_code, result.stdout, result.stderr, True)
        return ProcessResult(True, False, result.exit_code, result.stdout, result.stderr, True)


def _sandbox_image() -> str:
    from app.core.config import get_settings

    return get_settings().security_agent_ityfuzz_image.strip()


def _docker_present() -> bool:
    return shutil.which("docker") is not None


def _local_image() -> bool:
    from app.execution.docker_executor import local_image_present

    return local_image_present(_sandbox_image())


def _artifact_dir(request: AnalysisRequest) -> Path | None:
    raw = request.extra.get("artifact_dir", "")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = request.repo_root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(request.repo_root.resolve())
    except ValueError:
        return None
    if not resolved.is_dir():
        return None
    names = {item.name for item in resolved.iterdir() if item.is_file() and not item.is_symlink()}
    if not any(name.endswith(".abi") for name in names) or not any(
        name.endswith(".bin") for name in names
    ):
        return None
    return resolved


def _rejects_network(request: AnalysisRequest) -> bool:
    blob = " ".join(request.extra.values()).lower()
    return any(marker in blob for marker in _NETWORK)


def _timeout(request: AnalysisRequest) -> float:
    raw = request.extra.get("timeout", "")
    try:
        requested = float(raw) if raw else 30.0
    except ValueError:
        requested = 30.0
    return max(1.0, min(requested, _MAX_TIMEOUT))
