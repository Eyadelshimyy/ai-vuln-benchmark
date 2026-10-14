"""Second-engine property adapters and the research engine registry (Phase 52, Slice B).

The Foundry harness reports structured observations and the property is judged in
Python. To re-judge the *same* property along a materially independent path, this
module hosts the identical harness plan inside Echidna (hevm) or Medusa (geth EVM)
with an **independently implemented in-EVM judge** generated from the oracle
specification. Three flag properties make the engine's pass/fail output structured:

* ``echidna_bf_flag_ran``       fails once the planned sequence ran to completion;
* ``echidna_bf_flag_unjudged``  fails when it ran but the property could not be judged;
* ``echidna_bf_flag_violated``  fails when the property is violated.

A flag that never appears in the engine's report is a parse failure (INCONCLUSIVE),
never a pass. Every run is local, offline, bounded, with a server-owned config and a
minimal environment (no RPC URL, no fork, no corpus or keys from the server). An
engine binary being on PATH is ``installed``; it is ``usable`` only after a smoke
check judged a known-violated and a known-held property correctly. Nothing here
verifies a finding: agreement is a corroborated candidate at most.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.discovery.bounty.properties import (
    OracleKind,
    OracleSpec,
    PropertyVerdict,
    precondition_indices,
    probe_index,
)
from app.discovery.bounty.stateful import (
    _FIXTURE_TOKEN,
    K_AFTER,
    K_BEFORE,
    K_CALL,
    K_CREDITED,
    K_PRIMITIVE,
    K_READ_FAILED,
    K_RECEIVED,
    HarnessArtifact,
    Outcome,
    _safe_env,
    _tail,
    execution_enabled,
    sha256_text,
    tool_status,
    untrusted_cheatcode_use,
)
from app.discovery.bounty.vfcs import Vfcs
from app.discovery.process import tool_path

ENGINE_HARNESS_SCHEMA = "bugforge.engine_harness/1"
ENGINE_CONTRACT = "BugForgeEngineHarness"
ENGINE_FILE = "BugForgeEngineHarness.sol"
FLAG_RAN = "echidna_bf_flag_ran"
FLAG_UNJUDGED = "echidna_bf_flag_unjudged"
FLAG_VIOLATED = "echidna_bf_flag_violated"
FLAGS = (FLAG_RAN, FLAG_UNJUDGED, FLAG_VIOLATED)
ENGINE_TIMEOUT = 120
ENGINE_TEST_LIMIT = 40  # the plan is one deterministic transaction; a few suffice


class EngineStatus(StrEnum):
    """Registry status. A binary alone is ``installed``; ``usable`` needs a passing smoke."""

    USABLE = "usable"
    INSTALLED = "installed"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    BLOCKED_BY_POLICY = "blocked_by_policy"


# ---- in-EVM judge ---------------------------------------------------------------------------


def _judge(sequence: Vfcs, oracle: OracleSpec) -> tuple[str, str] | None:
    """Solidity bodies for (_bfJudged, _bfViolated), or None when not expressible."""
    if not oracle.executable:
        return None
    probe = probe_index(sequence, oracle)
    if probe < 0:
        return None
    required = precondition_indices(sequence, oracle, probe)
    present = {sequence.calls[i].role for i in required}
    if any(role not in present for role in oracle.precondition_roles):
        return None
    pre = "".join(f"        if (!_bfOk({i})) return false;\n" for i in required)
    reached = f"        if (!_bfReached({probe})) return false;\n"
    kind = oracle.kind
    if kind in {OracleKind.CALL_MUST_FAIL, OracleKind.SECOND_CALL_MUST_FAIL}:
        judged = "        return true;\n"
        violated = f"        return _bfOk({probe});\n"
    elif kind is OracleKind.STATE_UNCHANGED:
        judged = (
            f"        return !bfReadFailed && bfSeen[{K_BEFORE}][{probe}] "
            f"&& bfSeen[{K_AFTER}][{probe}];\n"
        )
        violated = f"        return bfVal[{K_BEFORE}][{probe}] != bfVal[{K_AFTER}][{probe}];\n"
    elif kind in {OracleKind.CREDIT_LE_RECEIVED, OracleKind.BALANCE_NOT_INCREASED}:
        judged = (
            f"        return !bfReadFailed && bfSeen[{K_RECEIVED}][0] && bfSeen[{K_RECEIVED}][1]"
            f" && bfSeen[{K_CREDITED}][0] && bfSeen[{K_CREDITED}][1] && _bfOk({probe});\n"
        )
        violated = (
            f"        return _bfGreaterDelta(bfVal[{K_CREDITED}][0], bfVal[{K_CREDITED}][1], "
            f"bfVal[{K_RECEIVED}][0], bfVal[{K_RECEIVED}][1]);\n"
        )
    elif kind is OracleKind.CREDIT_LE_HOLDINGS:
        judged = (
            f"        return !bfReadFailed && bfSeen[{K_RECEIVED}][1] && bfSeen[{K_CREDITED}][1]"
            f" && _bfOk({probe});\n"
        )
        violated = f"        return bfVal[{K_CREDITED}][1] > bfVal[{K_RECEIVED}][1];\n"
    else:
        return None
    return pre + reached + judged, violated


_ENGINE_TEMPLATE = """// SPDX-License-Identifier: UNLICENSED
// BUGFORGE ENGINE HARNESS ({schema}). Local, offline, bounded. Not a proof.
// The run body is the Foundry harness plan verbatim; the judge below is an
// independent in-EVM implementation of oracle {oracle}.
pragma solidity >=0.7.0;

{imports}{fixture}
interface BugForgeVm {{
    function prank(address) external;
    function warp(uint256) external;
    function roll(uint256) external;
    function deal(address, uint256) external;
}}

contract {contract} {{
    BugForgeVm internal constant vm =
        BugForgeVm(address(uint160(uint256(keccak256("hevm cheat code")))));
    bool internal bfRan;
    bool internal bfReadFailed;
    mapping(uint256 => mapping(uint256 => uint256)) internal bfVal;
    mapping(uint256 => mapping(uint256 => bool)) internal bfSeen;

    function _bfObs(uint256 kind, uint256 index, uint256 value) internal {{
        if (kind == {read_failed}) bfReadFailed = true;
        bfVal[kind][index] = value;
        bfSeen[kind][index] = true;
    }}

    function _bfRead(address target, bytes memory data) internal view returns (bool, uint256) {{
        (bool ok, bytes memory out) = target.staticcall(data);
        if (!ok || out.length < 32) return (false, 0);
        return (true, abi.decode(out, (uint256)));
    }}

    function _bfOk(uint256 i) internal view returns (bool) {{
        return (bfSeen[{call}][i] && bfVal[{call}][i] == 1)
            || (bfSeen[{prim}][i] && bfVal[{prim}][i] == 1);
    }}

    function _bfReached(uint256 i) internal view returns (bool) {{
        return bfSeen[{call}][i] || bfSeen[{prim}][i];
    }}

    // signed (c1 - c0) > (r1 - r0) without overflow
    function _bfGreaterDelta(uint256 c0, uint256 c1, uint256 r0, uint256 r1)
        internal pure returns (bool)
    {{
        bool cNeg = c1 < c0;
        bool rNeg = r1 < r0;
        uint256 cAbs = cNeg ? c0 - c1 : c1 - c0;
        uint256 rAbs = rNeg ? r0 - r1 : r1 - r0;
        if (!cNeg && rNeg) return cAbs > 0 || rAbs > 0;
        if (cNeg && !rNeg) return false;
        if (!cNeg) return cAbs > rAbs;
        return cAbs < rAbs;
    }}

    function _bfJudged() internal view returns (bool) {{
{judged}    }}

    function _bfViolated() internal view returns (bool) {{
        if (!bfRan || !_bfJudged()) return false;
{violated}    }}

    function bf_run() public {{
        if (bfRan) return;
{body}
        bfRan = true;
    }}

    function {flag_ran}() public view returns (bool) {{
        return !bfRan;
    }}

    function {flag_unjudged}() public view returns (bool) {{
        return !(bfRan && !_bfJudged());
    }}

    function {flag_violated}() public view returns (bool) {{
        return !_bfViolated();
    }}
}}
"""


def engine_harness_source(harness: HarnessArtifact, sequence: Vfcs) -> str:
    """The engine-hosted harness, or "" when the plan or oracle is not expressible."""
    if not harness.buildable or not harness.body:
        return ""
    judge = _judge(sequence, harness.oracle)
    if judge is None:
        return ""
    body = harness.body.replace("emit BugForgeObservation(", "_bfObs(")
    if "BugForgeObservation" in body:
        return ""
    return _ENGINE_TEMPLATE.format(
        schema=ENGINE_HARNESS_SCHEMA,
        oracle=harness.oracle.kind.value,
        imports="".join(f'import "./src/{path}";\n' for path in harness.imports),
        fixture=_FIXTURE_TOKEN if harness.fixture else "",
        contract=ENGINE_CONTRACT,
        read_failed=K_READ_FAILED,
        call=K_CALL,
        prim=K_PRIMITIVE,
        judged=judge[0],
        violated=judge[1],
        body=body,
        flag_ran=FLAG_RAN,
        flag_unjudged=FLAG_UNJUDGED,
        flag_violated=FLAG_VIOLATED,
    )


# ---- engine runs ------------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineRun:
    """One second-engine judgment of one property. Structured; never verified."""

    engine: str
    process_status: str  # not_run | completed | timeout | spawn_failed
    outcome: Outcome
    verdict: str
    reason: str
    flags: tuple[tuple[str, str], ...] = ()  # (flag, failed|passed)
    version: str = ""
    harness_hash: str = ""
    config_hash: str = ""
    command: tuple[str, ...] = ()
    return_code: int | None = None
    duration_ms: int = 0
    output_tail: str = ""
    verified: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "process_status": self.process_status,
            "outcome": self.outcome.value,
            "verdict": self.verdict,
            "reason": self.reason[:300],
            "flags": dict(self.flags),
            "version": self.version,
            "harness_hash": self.harness_hash,
            "config_hash": self.config_hash,
            "command": list(self.command),
            "return_code": self.return_code,
            "duration_ms": self.duration_ms,
            "output_tail": self.output_tail,
            "verified": False,
        }


def verdict_from_flags(flags: Mapping[str, str]) -> tuple[Outcome, str, str]:
    """(outcome, verdict, reason) from the three flag results; missing flags never pass."""
    missing = [flag for flag in FLAGS if flags.get(flag) not in {"failed", "passed"}]
    if missing:
        return (
            Outcome.INCONCLUSIVE,
            PropertyVerdict.NOT_EVALUATED.value,
            f"the engine report did not include {', '.join(missing)}",
        )
    if flags[FLAG_RAN] != "failed":
        return (
            Outcome.INCONCLUSIVE,
            PropertyVerdict.NOT_EVALUATED.value,
            "the engine never completed the planned sequence",
        )
    if flags[FLAG_UNJUDGED] == "failed":
        return (
            Outcome.INCONCLUSIVE,
            PropertyVerdict.NOT_EVALUATED.value,
            "the sequence ran but the in-EVM judge could not evaluate the property",
        )
    if flags[FLAG_VIOLATED] == "failed":
        return (
            Outcome.PROPERTY_VIOLATED,
            PropertyVerdict.PROPERTY_VIOLATED.value,
            "the independent in-EVM judge found the property violated",
        )
    return (
        Outcome.PROPERTY_HELD,
        PropertyVerdict.PROPERTY_HELD.value,
        "the independent in-EVM judge found the property held",
    )


_ECHIDNA_LINE = re.compile(r"^\s*(echidna_bf_flag_\w+)\s*:\s*(failed|passing|passed)", re.M)
_MEDUSA_LINE = re.compile(
    r"\[(FAILED|PASSED)\]\s+Property Test:\s+\w+\.(echidna_bf_flag_\w+)\(\)", re.M
)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_echidna(output: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for name, state in _ECHIDNA_LINE.findall(_ANSI.sub("", output)):
        if name in FLAGS:
            found[name] = "failed" if state == "failed" else "passed"
    return found


def parse_medusa(output: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for state, name in _MEDUSA_LINE.findall(_ANSI.sub("", output)):
        if name in FLAGS:
            found[name] = "failed" if state == "FAILED" else "passed"
    return found


def echidna_config(seed: int = 1) -> str:
    # Server-owned. No rpcUrl, no corpus, no coverage files, one worker, fixed seed.
    return (
        "testMode: property\n"
        f"testLimit: {ENGINE_TEST_LIMIT}\n"
        "seqLen: 1\n"
        "shrinkLimit: 0\n"
        "workers: 1\n"
        f"seed: {seed}\n"
        "coverage: false\n"
        "format: text\n"
        "allowFFI: false\n"
    )


def medusa_config() -> str:
    # Server-owned. Fork mode off, empty rpcUrl, coverage off, property tests only.
    config = {
        "fuzzing": {
            "workers": 1,
            "workerResetLimit": 50,
            "timeout": ENGINE_TIMEOUT,
            "testLimit": ENGINE_TEST_LIMIT,
            "shrinkLimit": 0,
            "callSequenceLength": 1,
            "corpusDirectory": "",
            # medusa 1.5 dereferences a nil tracer when coverage is off; reports stay
            # inside the temporary run directory and are discarded with it.
            "coverageEnabled": True,
            "coverageFormats": [],
            "targetContracts": [ENGINE_CONTRACT],
            "predeployedContracts": {},
            "targetContractsBalances": [],
            "constructorArgs": {},
            "deployerAddress": "0x30000",
            "senderAddresses": ["0x10000"],
            "blockNumberDelayMax": 1,
            "blockTimestampDelayMax": 1,
            "transactionGasLimit": 12500000,
            "testing": {
                "stopOnFailedTest": False,
                "stopOnFailedContractMatching": False,
                "stopOnNoTests": True,
                "testAllContracts": False,
                "testViewMethods": False,
                "assertionTesting": {"enabled": False},
                "propertyTesting": {"enabled": True, "testPrefixes": ["echidna_bf_flag_"]},
                "optimizationTesting": {"enabled": False},
            },
            "chainConfig": {
                "codeSizeCheckDisabled": True,
                "cheatCodes": {"cheatCodesEnabled": True, "enableFFI": False},
                "forkConfig": {"forkModeEnabled": False, "rpcUrl": "", "rpcBlock": 1},
            },
        },
        "compilation": {
            "platform": "crytic-compile",
            "platformConfig": {
                "target": ENGINE_FILE,
                "solcVersion": "",
                "exportDirectory": "",
                "args": ["--compile-force-framework", "solc"],
            },
        },
        "logging": {"level": "info", "logDirectory": "", "noColor": True},
    }
    return json.dumps(config, indent=1, sort_keys=True)


@dataclass(frozen=True)
class EngineSpec:
    name: str
    binary: str
    config_name: str
    config: str
    argv: tuple[str, ...]
    parse: Any  # Callable[[str], dict[str, str]]


def _specs() -> dict[str, EngineSpec]:
    return {
        "echidna": EngineSpec(
            "echidna",
            "echidna",
            "echidna.yaml",
            echidna_config(),
            (
                ENGINE_FILE,
                "--contract",
                ENGINE_CONTRACT,
                "--config",
                "echidna.yaml",
                "--format",
                "text",
                "--disable-slither",
                "--disable-onchain-sources",
                "--crytic-args",
                "--compile-force-framework solc",
            ),
            parse_echidna,
        ),
        "medusa": EngineSpec(
            "medusa",
            "medusa",
            "medusa.json",
            medusa_config(),
            ("fuzz", "--config", "medusa.json", "--no-color"),
            parse_medusa,
        ),
    }


PROPERTY_ENGINES = ("echidna", "medusa")


@lru_cache(maxsize=8)
def _engine_version(binary: str) -> str:
    path = tool_path(binary)
    if not path:
        return ""
    try:
        proc = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    match = re.search(r"(\d+\.\d+\.\d+[\w.+-]*)", proc.stdout + proc.stderr)
    return match.group(1) if match else ""


def _layout(root: Path, sources: Mapping[str, str], harness: str, spec: EngineSpec) -> None:
    base = (root / "src").resolve()
    base.mkdir(parents=True, exist_ok=True)
    for rel, text in sorted(sources.items()):
        target = (base / rel).resolve()
        if base not in target.parents:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (root / ENGINE_FILE).write_text(harness, encoding="utf-8")
    (root / spec.config_name).write_text(spec.config, encoding="utf-8")


class PropertyEngine:
    """Runs one engine-hosted harness, offline and bounded."""

    def __init__(self, name: str, *, timeout: int = ENGINE_TIMEOUT) -> None:
        specs = _specs()
        if name not in specs:
            raise ValueError(f"no property adapter for {name}")
        self.name = name
        self.spec = specs[name]
        self.timeout = timeout

    def installed(self) -> bool:
        return bool(tool_path(self.spec.binary)) and bool(tool_path("solc"))

    def run(
        self, harness: HarnessArtifact, sequence: Vfcs, sources: Mapping[str, str]
    ) -> EngineRun:
        source = engine_harness_source(harness, sequence)
        common: dict[str, Any] = {
            "version": _engine_version(self.spec.binary),
            "config_hash": sha256_text(self.spec.config),
            "command": (self.spec.binary, *self.spec.argv),
        }
        if not source:
            return EngineRun(
                self.name,
                "not_run",
                Outcome.INCONCLUSIVE,
                PropertyVerdict.NOT_EVALUATED.value,
                "the harness plan or oracle is not expressible for an engine judge",
                **common,
            )
        common["harness_hash"] = sha256_text(source)
        cheats = untrusted_cheatcode_use(sources)
        if cheats:
            return EngineRun(
                self.name,
                "not_run",
                Outcome.INCONCLUSIVE,
                PropertyVerdict.NOT_EVALUATED.value,
                f"blocked_by_policy: cheatcode references in sources ({'; '.join(cheats)})",
                **common,
            )
        binary = tool_path(self.spec.binary)
        if not binary or not tool_path("solc") or not tool_path("crytic-compile"):
            return EngineRun(
                self.name,
                "not_run",
                Outcome.UNAVAILABLE,
                PropertyVerdict.NOT_EVALUATED.value,
                f"{self.spec.binary}, solc or crytic-compile is not installed",
                **common,
            )
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"bugforge-{self.name}-") as tmp:
            root = Path(tmp)
            _layout(root, sources, source, self.spec)
            try:
                proc = subprocess.run(
                    [binary, *self.spec.argv],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                    env=_safe_env(tmp),
                )
            except subprocess.TimeoutExpired:
                return EngineRun(
                    self.name,
                    "timeout",
                    Outcome.TIMEOUT,
                    PropertyVerdict.NOT_EVALUATED.value,
                    f"{self.name} exceeded {self.timeout}s",
                    **common,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return EngineRun(
                    self.name,
                    "spawn_failed",
                    Outcome.EXECUTION_FAILED,
                    PropertyVerdict.NOT_EVALUATED.value,
                    f"{self.name} did not start: {str(exc)[:160]}",
                    **common,
                )
        output = proc.stdout + "\n" + proc.stderr
        flags = self.spec.parse(output)
        outcome, verdict, reason = verdict_from_flags(flags)
        if not flags:
            # No structured flag results: the engine failed (compile, crash, config).
            # Its output is kept as a tail; it is never guessed into a verdict.
            outcome = Outcome.EXECUTION_FAILED
            reason = f"{self.name} exited {proc.returncode} without flag results"
        return EngineRun(
            self.name,
            "completed",
            outcome,
            verdict,
            reason,
            flags=tuple(sorted(flags.items())),
            return_code=proc.returncode,
            duration_ms=int((time.monotonic() - started) * 1000),
            output_tail=_tail(_ANSI.sub("", output)),
            **common,
        )


# ---- smoke check & registry -----------------------------------------------------------------

_SMOKE_VULNERABLE = """// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract BugForgeSmoke {
    address public owner;
    function initialize(address o) external { owner = o; }
}
"""
_SMOKE_SAFE = """// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract BugForgeSmoke {
    address public owner;
    bool public done;
    function initialize(address o) external { require(!done, "done"); done = true; owner = o; }
}
"""


def _smoke_inputs(text: str) -> tuple[HarnessArtifact, Vfcs, dict[str, str]]:
    from app.discovery.bounty.properties import build_property
    from app.discovery.bounty.stateful import build_harness
    from app.discovery.bounty.vfcs import SequenceIdentity, VfcsCall
    from app.parsing.solidity_research import build_research_model

    sources = {"BugForgeSmoke.sol": text}
    sig = "initialize(address)"
    sequence = Vfcs(
        sequence_id="vf_engine_smoke",
        template="initialize→reinitialize",
        origin="template:initialize→reinitialize",
        calls=(
            VfcsCall("BugForgeSmoke", sig, "initialize", "attacker", (("o", "attacker"),)),
            VfcsCall("BugForgeSmoke", sig, "reinitialize", "victim", (("o", "second_owner"),)),
        ),
        property_under_test="an account initializes exactly once",
        derived_from=f"aa.unprotected_account_initializer@BugForgeSmoke.{sig}",
        identity=SequenceIdentity(),
    )
    model = build_research_model(sources)
    spec = build_property(sequence, model)
    return build_harness(sequence, model, oracle=spec.oracle, sources=sources), sequence, sources


@dataclass(frozen=True)
class SmokeResult:
    engine: str
    usable: bool
    reason: str
    observed: tuple[tuple[str, str], ...] = ()


@lru_cache(maxsize=4)
def smoke_check(name: str) -> SmokeResult:
    """Binary alone is not usable: the engine must judge a known violation and a known
    held property correctly through the same adapter path real checks use."""
    engine = PropertyEngine(name, timeout=60)
    if not engine.installed():
        return SmokeResult(name, False, f"{name} is not installed")
    observed: list[tuple[str, str]] = []
    for label, text, want in (
        ("vulnerable", _SMOKE_VULNERABLE, Outcome.PROPERTY_VIOLATED),
        ("safe", _SMOKE_SAFE, Outcome.PROPERTY_HELD),
    ):
        harness, sequence, sources = _smoke_inputs(text)
        run = engine.run(harness, sequence, sources)
        observed.append((label, run.outcome.value))
        if run.outcome is not want:
            return SmokeResult(
                name,
                False,
                f"smoke {label}: expected {want.value}, got {run.outcome.value} ({run.reason})",
                tuple(observed),
            )
    return SmokeResult(
        name, True, "judged a known violation and a known held property", tuple(observed)
    )


@dataclass(frozen=True)
class EngineEntry:
    name: str
    role: str
    status: EngineStatus
    reason: str
    version: str = ""
    property_adapter: bool = False
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status.value,
            "reason": self.reason,
            "version": self.version,
            "property_adapter": self.property_adapter,
            "evidence": list(self.evidence),
        }


# Tools BugForge knows about but has no campaign adapter for: never counted as usable.
_NO_ADAPTER = {
    "ityfuzz": "fuzzer",
    "halmos": "symbolic",
    "slither": "static",
}


def engine_registry(*, smoke: bool = True) -> tuple[EngineEntry, ...]:
    """Honest per-engine status for this box. ``smoke=False`` reports at most installed."""
    entries: list[EngineEntry] = [
        EngineEntry(
            "bugforge-static",
            "static_semantic",
            EngineStatus.USABLE,
            "in-process semantic detectors over the exact source set",
        )
    ]
    tools = tool_status()
    enabled = execution_enabled()
    if not enabled:
        entries.append(
            EngineEntry(
                "foundry",
                "stateful_execution",
                EngineStatus.BLOCKED_BY_POLICY,
                "local stateful execution is disabled by server settings",
                tools.forge_version,
                True,
            )
        )
    elif not tools.available:
        entries.append(
            EngineEntry(
                "foundry",
                "stateful_execution",
                EngineStatus.UNAVAILABLE,
                "forge or solc is not installed",
                tools.forge_version,
                True,
            )
        )
    else:
        entries.append(
            EngineEntry(
                "foundry",
                "stateful_execution",
                EngineStatus.USABLE if smoke else EngineStatus.INSTALLED,
                "forge and solc are installed; harness execution is exercised per run",
                tools.forge_version,
                True,
                (f"solc {tools.solc_version}",),
            )
        )
    for name in PROPERTY_ENGINES:
        engine = PropertyEngine(name)
        version = _engine_version(name) if tool_path(name) else ""
        if not enabled:
            status, reason = EngineStatus.BLOCKED_BY_POLICY, "local execution is disabled"
            evidence: tuple[str, ...] = ()
        elif not engine.installed() or not tool_path("crytic-compile"):
            status, reason, evidence = (
                EngineStatus.UNAVAILABLE,
                f"{name}, solc or crytic-compile is not installed",
                (),
            )
        elif not smoke:
            status, reason, evidence = (
                EngineStatus.INSTALLED,
                "binary present; not smoke-checked",
                (),
            )
        else:
            result = smoke_check(name)
            status = EngineStatus.USABLE if result.usable else EngineStatus.INSTALLED
            reason = result.reason
            evidence = tuple(f"{label}:{outcome}" for label, outcome in result.observed)
        entries.append(
            EngineEntry(name, "property_engine", status, reason, version, True, evidence)
        )
    for name, role in sorted(_NO_ADAPTER.items()):
        if tool_path(name):
            entries.append(
                EngineEntry(
                    name,
                    role,
                    EngineStatus.UNSUPPORTED,
                    "installed, but BugForge has no campaign adapter for it",
                )
            )
        else:
            entries.append(EngineEntry(name, role, EngineStatus.UNAVAILABLE, "not installed"))
    entries.append(
        EngineEntry(
            "fork_replay",
            "pinned_fork_replay",
            EngineStatus.BLOCKED_BY_POLICY,
            "no RPC is used by research tooling; fork replay needs an approved, pinned "
            "fork outside this path",
        )
    )
    return tuple(entries)


def usable_property_engines(registry: Sequence[EngineEntry] | None = None) -> tuple[str, ...]:
    entries = registry if registry is not None else engine_registry()
    return tuple(
        entry.name
        for entry in entries
        if entry.role == "property_engine" and entry.status is EngineStatus.USABLE
    )


def engine_status_map(registry: Sequence[EngineEntry]) -> dict[str, str]:
    return {entry.name: entry.status.value for entry in registry}
