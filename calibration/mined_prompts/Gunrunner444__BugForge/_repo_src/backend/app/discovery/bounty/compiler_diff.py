"""Bounded compiler differential validation.

When a compiler is installed and host compilation is explicitly enabled, the same
source is compiled through a few pipeline configurations and the metadata-stripped
runtime bytecode is compared. Without a compiler the result is UNAVAILABLE; a
compiler is never downloaded or installed. Different bytecode across pipelines is
normal, so a difference is only a candidate. It becomes interesting only together
with a matching advisory, a source trigger, and an observed behavioral difference,
and even then it is not verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.discovery.bounty.advisories import AdvisoryReport, parse_version
from app.discovery.bounty.campaign import UNKNOWN, CompilerConfiguration

MAX_PIPELINES = 4
MAX_COMPARISONS = 6
MAX_SOURCE_FILES = 64
MAX_CONTRACTS = 64

UNAVAILABLE = "unavailable"
COMPLETED = "completed"
INCOMPLETE = "incomplete"
IDENTITY_MISMATCH = "identity_mismatch"


@dataclass(frozen=True)
class Pipeline:
    via_ir: bool
    optimizer: bool
    runs: int = 200
    evm_version: str = ""

    @property
    def label(self) -> str:
        return (
            f"viaIR={str(self.via_ir).lower()},optimizer={str(self.optimizer).lower()}"
            f",runs={self.runs},evm={self.evm_version or 'default'}"
        )


@dataclass(frozen=True)
class CompileOutcome:
    ok: bool
    runtime_digests: tuple[tuple[str, str], ...] = ()
    reason: str = ""


class CompilerBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def version(self) -> str: ...

    def compile(self, sources: Mapping[str, str], pipeline: Pipeline) -> CompileOutcome: ...


class NoCompilerBackend:
    """Used when no compiler exists. It never runs anything."""

    name = "none"

    def available(self) -> bool:
        return False

    def version(self) -> str:
        return ""

    def compile(self, sources: Mapping[str, str], pipeline: Pipeline) -> CompileOutcome:
        return CompileOutcome(False, reason="no compiler is installed")


class HostSolcBackend:
    """The installed ``solc``. Available only when host compilation is enabled and a binary exists."""

    name = "host-solc"

    def __init__(self, binary: str = "solc") -> None:
        self.binary = binary
        self._version = ""

    def available(self) -> bool:
        from app.core.config import get_settings

        return bool(get_settings().solidity_host_compiler) and shutil.which(self.binary) is not None

    def version(self) -> str:
        if self._version or not self.available():
            return self._version
        try:
            completed = subprocess.run(
                [self.binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        for line in completed.stdout.splitlines():
            if line.startswith("Version:"):
                parsed = parse_version(line.split(":", 1)[1].strip())
                self._version = ".".join(map(str, parsed)) if parsed else ""
        return self._version

    def compile(self, sources: Mapping[str, str], pipeline: Pipeline) -> CompileOutcome:
        from app.parsing.solidity_compiler import run_solc_standard_json

        settings: dict[str, object] = {
            "optimizer": {"enabled": pipeline.optimizer, "runs": pipeline.runs},
            "viaIR": pipeline.via_ir,
            "outputSelection": {"*": {"*": ["evm.deployedBytecode.object"]}},
        }
        if pipeline.evm_version:
            settings["evmVersion"] = pipeline.evm_version
        payload = json.dumps(
            {
                "language": "Solidity",
                "sources": {path: {"content": text} for path, text in sorted(sources.items())},
                "settings": settings,
            }
        )
        try:
            raw = run_solc_standard_json(payload, binary=self.binary)
            data = json.loads(raw)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return CompileOutcome(False, reason=str(exc)[:200])
        errors = [e for e in data.get("errors", []) if e.get("severity") == "error"]
        if errors:
            return CompileOutcome(
                False, reason=str(errors[0].get("message", "compile error"))[:200]
            )
        digests: list[tuple[str, str]] = []
        for path, contracts in sorted(data.get("contracts", {}).items()):
            for name, artifact in sorted(contracts.items()):
                code = artifact.get("evm", {}).get("deployedBytecode", {}).get("object", "")
                if code:
                    digests.append((f"{path}:{name}", strip_metadata_digest(code)))
        return CompileOutcome(True, tuple(digests[:MAX_CONTRACTS]))


def strip_metadata_digest(hex_code: str) -> str:
    """Digest of runtime bytecode without the trailing CBOR metadata blob."""
    text = hex_code[2:] if hex_code.startswith("0x") else hex_code
    try:
        raw = bytes.fromhex(text)
    except ValueError:
        return "invalid"
    if len(raw) >= 2:
        length = int.from_bytes(raw[-2:], "big")
        if 0 < length + 2 <= len(raw):
            raw = raw[: -(length + 2)]
    return hashlib.sha256(raw).hexdigest()[:24]


@dataclass(frozen=True)
class BytecodeDifference:
    contract: str
    pipeline_a: str
    pipeline_b: str
    digest_a: str
    digest_b: str


@dataclass(frozen=True)
class DifferentialResult:
    status: str
    reason: str
    backend: str
    compiler_version: str
    pipelines: tuple[str, ...]
    differences: tuple[BytecodeDifference, ...]
    comparisons: int
    truncated: bool
    executed: bool
    verified: bool = False


def default_pipelines(config: CompilerConfiguration | None = None) -> tuple[Pipeline, ...]:
    """A small fixed set: legacy and IR, each with and without the optimizer."""
    evm = ""
    if config is not None and config.evm_version not in {UNKNOWN, ""}:
        evm = config.evm_version
    return tuple(
        Pipeline(via_ir=ir, optimizer=opt, evm_version=evm)
        for ir in (False, True)
        for opt in (False, True)
    )[:MAX_PIPELINES]


def run_differential(
    sources: Mapping[str, str],
    *,
    backend: CompilerBackend | None = None,
    pipelines: Sequence[Pipeline] | None = None,
    expected: CompilerConfiguration | None = None,
) -> DifferentialResult:
    backend = backend or HostSolcBackend()
    chosen = tuple(pipelines or default_pipelines(expected))[:MAX_PIPELINES]
    labels = tuple(p.label for p in chosen)
    if not backend.available():
        return DifferentialResult(
            UNAVAILABLE,
            "no usable compiler; none is downloaded",
            backend.name,
            "",
            labels,
            (),
            0,
            False,
            False,
        )
    version = backend.version()
    if expected is not None and expected.version != UNKNOWN and version:
        if parse_version(expected.version) != parse_version(version):
            return DifferentialResult(
                IDENTITY_MISMATCH,
                f"installed compiler {version} is not the campaign compiler {expected.version}",
                backend.name,
                version,
                labels,
                (),
                0,
                False,
                False,
            )
    if len(sources) > MAX_SOURCE_FILES:
        return DifferentialResult(
            INCOMPLETE,
            "too many source files for a bounded comparison",
            backend.name,
            version,
            labels,
            (),
            0,
            False,
            False,
        )
    outcomes: dict[str, dict[str, str]] = {}
    for pipeline in chosen:
        outcome = backend.compile(sources, pipeline)
        if not outcome.ok:
            return DifferentialResult(
                INCOMPLETE,
                f"{pipeline.label}: {outcome.reason or 'compilation failed'}",
                backend.name,
                version,
                labels,
                (),
                0,
                True,
                False,
            )
        outcomes[pipeline.label] = dict(outcome.runtime_digests)
    differences: list[BytecodeDifference] = []
    comparisons = 0
    truncated = False
    base_label = labels[0] if labels else ""
    for other in labels[1:]:
        if comparisons >= MAX_COMPARISONS:
            truncated = True
            break
        comparisons += 1
        for contract in sorted(set(outcomes[base_label]) | set(outcomes[other])):
            a, b = outcomes[base_label].get(contract, ""), outcomes[other].get(contract, "")
            if a != b:
                differences.append(BytecodeDifference(contract, base_label, other, a, b))
    return DifferentialResult(
        COMPLETED,
        "bytecode compared across pipelines",
        backend.name,
        version,
        labels,
        tuple(differences),
        comparisons,
        truncated,
        True,
    )


@dataclass(frozen=True)
class Promotion:
    """Whether a bytecode difference deserves research attention. Never verification."""

    status: str  # candidate | not_promoted
    missing: tuple[str, ...]
    advisories: tuple[str, ...]
    verified: bool = False


def promote(
    difference: BytecodeDifference,
    advisories: AdvisoryReport,
    *,
    behavioral_evidence_ids: Sequence[str] = (),
    identity_bound: bool,
) -> Promotion:
    contract = difference.contract.split(":")[-1]
    applicable = tuple(
        m.uid
        for m in advisories.matches
        if m.status == "applicable_candidate"
        and any(contract in loc for loc in m.trigger_locations)
    )
    missing: list[str] = []
    if not applicable:
        missing.append("a matching advisory with a source trigger in this contract")
    if not behavioral_evidence_ids:
        missing.append(
            "an observed behavioral difference; bytecode differs across pipelines by design"
        )
    if not identity_bound:
        missing.append("compiler and source identity bound to the campaign")
    return Promotion("candidate" if not missing else "not_promoted", tuple(missing), applicable)
