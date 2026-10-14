"""Optional external discovery tools. Missing executables do not invent results."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.process import ProcessResult, run_command, tool_path, tool_version
from app.discovery.results import DynamicFinding, DynamicResult, unavailable_result


class ExternalDiscoveryEngine(DiscoveryEngine):
    binary: str = ""
    license_note: str = ""

    def availability(self) -> EngineAvailability:
        if tool_path(self.binary):
            return EngineAvailability.AVAILABLE
        return EngineAvailability.UNAVAILABLE

    def version(self) -> str:
        return tool_version(self.binary)

    def _unavailable(self, request: AnalysisRequest) -> DynamicResult:
        return unavailable_result(
            self.engine_id,
            request.language,
            request.target,
            reason=f"{self.display_name} executable {self.binary!r} is not installed",
        )


class SlitherEngine(ExternalDiscoveryEngine):
    binary = "slither"
    license_note = "AGPLv3 external executable; source is not vendored"

    @property
    def engine_id(self) -> str:
        return "slither"

    @property
    def display_name(self) -> str:
        return "Slither"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity", "vyper"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset({EngineCapability.STATIC_ANALYSIS, EngineCapability.RESULTS_INGESTION})

    def _analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        if self.availability() is not EngineAvailability.AVAILABLE:
            return self._unavailable(request)
        proc = run_command(
            ["slither", ".", "--json", "-"],
            cwd=request.repo_root,
            timeout=60,
        )
        findings = normalize_slither(proc.stdout)
        return DynamicResult(
            engine=self.engine_id,
            engine_version=self.version(),
            language=request.language,
            target=request.target or str(request.repo_root),
            status=_process_status(proc, ingested=bool(findings)),
            executed=_executed(proc),
            exit_code=proc.return_code,
            stdout=proc.stdout[:4000],
            stderr=proc.stderr[:2000],
            findings=tuple(findings),
            provenance="slither",
            metadata={"license": self.license_note, "verified": "false", **_proc_meta(proc)},
        )

    def _collect_results(self, request: AnalysisRequest) -> DynamicResult:
        raw = request.extra.get("slither_json", "")
        if not raw:
            return self._analyze_target(request)
        findings = normalize_slither(raw)
        return DynamicResult(
            engine=self.engine_id,
            language=request.language,
            target=request.target,
            status=ResultStatus.INGESTED,
            executed=False,
            findings=tuple(findings),
            provenance="slither_json",
            oracle_explanation="Ingested caller-supplied Slither JSON. The tool was not executed.",
        )


class FoundryEngine(ExternalDiscoveryEngine):
    binary = "forge"
    license_note = "MIT/Apache external Foundry executable"

    @property
    def engine_id(self) -> str:
        return "foundry"

    @property
    def display_name(self) -> str:
        return "Foundry"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset(
            {
                EngineCapability.BUILD,
                EngineCapability.TEST_EXECUTION,
                EngineCapability.FUZZING,
                EngineCapability.INVARIANT_TESTING,
                EngineCapability.COVERAGE_FEEDBACK,
                EngineCapability.RESULTS_INGESTION,
            }
        )

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        if self.availability() is not EngineAvailability.AVAILABLE:
            return self._unavailable(request)
        if _rejects_fork(request):
            return DynamicResult(
                engine=self.engine_id,
                language=request.language,
                target=request.target,
                status=ResultStatus.FAILED,
                executed=False,
                provenance="scope",
                oracle_explanation="Live or forked network execution is not started from this adapter.",
            )
        mode = request.extra.get("mode", "test")
        config = read_foundry_config(request.repo_root)
        tests = discover_forge_tests(request.repo_root)
        argv, campaign = foundry_invocation(mode, request, tests=tests)
        proc = run_command(argv, cwd=request.repo_root, timeout=120)
        parsed = parse_foundry_output(
            proc.stdout,
            baseline=request.extra.get("coverage_baseline", ""),
            mode=campaign,
        )
        coverage = _as_map(parsed["coverage"])
        assertion = _as_str(parsed["assertion"])
        status = _process_status(proc, ingested=bool(assertion))
        if assertion and status in {ResultStatus.EXECUTED, ResultStatus.INGESTED}:
            status = ResultStatus.INTERESTING
        return DynamicResult(
            engine=self.engine_id,
            engine_version=self.version(),
            language=request.language,
            target=request.target,
            function=request.function,
            contract=request.contract,
            source_file=request.source_file,
            status=status,
            executed=_executed(proc),
            exit_code=proc.return_code,
            stdout=proc.stdout[:4000],
            stderr=proc.stderr[:2000],
            assertion=assertion,
            coverage=coverage,
            reproduction_command=" ".join(argv),
            provenance="foundry",
            oracle_kind="assertion" if assertion else "",
            oracle_explanation=assertion,
            metadata={
                "fork": "false",
                "verified": "false",
                "mode": mode,
                "campaign": campaign,
                "license": self.license_note,
                **_proc_meta(proc),
                **{f"foundry_{key}": value for key, value in config.items()},
            },
        )


class EchidnaEngine(ExternalDiscoveryEngine):
    binary = "echidna"
    license_note = "AGPL external executable; source is not vendored"

    @property
    def engine_id(self) -> str:
        return "echidna"

    @property
    def display_name(self) -> str:
        return "Echidna"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset(
            {
                EngineCapability.PROPERTY_TESTING,
                EngineCapability.FUZZING,
                EngineCapability.RESULTS_INGESTION,
            }
        )

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        if self.availability() is not EngineAvailability.AVAILABLE:
            return self._unavailable(request)
        target = request.source_file or request.target
        if not target:
            return DynamicResult(
                engine=self.engine_id,
                language=request.language,
                target="",
                status=ResultStatus.NOT_IMPLEMENTED,
                executed=False,
                oracle_explanation="Echidna needs a contract file.",
            )
        argv = ["echidna", target, "--format", "json"]
        if request.contract:
            argv[2:2] = ["--contract", request.contract]
        for name in ("echidna.yaml", "echidna.config.yaml"):
            if (request.repo_root / name).is_file():
                argv.extend(["--config", name])
                break
        proc = run_command(argv, cwd=request.repo_root, timeout=90)
        parsed = parse_echidna_output(proc.stdout + "\n" + proc.stderr)
        assertion = _as_str(parsed["assertion"])
        sequence = _as_str(parsed["sequence"])
        coverage = _as_map(parsed["coverage"])
        status = _process_status(proc, ingested=bool(assertion))
        if assertion and status in {ResultStatus.EXECUTED, ResultStatus.INGESTED}:
            status = ResultStatus.INTERESTING
        campaign_ok = _as_str(parsed["campaign_success"])
        return DynamicResult(
            engine=self.engine_id,
            engine_version=self.version(),
            language=request.language,
            target=target,
            contract=_as_str(parsed["contract"]) or request.contract,
            function=_as_str(parsed["property"]),
            status=status,
            executed=_executed(proc),
            exit_code=proc.return_code,
            stdout=proc.stdout[:4000],
            stderr=proc.stderr[:2000],
            assertion=assertion,
            minimized_input=sequence,
            coverage=coverage,
            reproduction_command=" ".join(argv),
            provenance="echidna",
            oracle_kind="property" if assertion else "",
            oracle_explanation=(
                assertion or "No counterexample was found in this campaign. That is not a proof."
            ),
            metadata={
                "verified": "false",
                "license": self.license_note,
                "campaign_success": campaign_ok,
                "proof": "false",
                **_proc_meta(proc),
            },
        )


class MedusaEngine(ExternalDiscoveryEngine):
    binary = "medusa"
    license_note = "AGPL external executable; source is not vendored"

    @property
    def engine_id(self) -> str:
        return "medusa"

    @property
    def display_name(self) -> str:
        return "Medusa"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset(
            {
                EngineCapability.FUZZING,
                EngineCapability.PROPERTY_TESTING,
                EngineCapability.COVERAGE_FEEDBACK,
                EngineCapability.RESULTS_INGESTION,
            }
        )

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        if self.availability() is not EngineAvailability.AVAILABLE:
            return self._unavailable(request)
        argv = ["medusa", "fuzz"]
        if (request.repo_root / "medusa.json").is_file():
            argv.extend(["--config", "medusa.json"])
        proc = run_command(argv, cwd=request.repo_root, timeout=90)
        parsed = parse_medusa_output(
            proc.stdout + "\n" + proc.stderr,
            baseline=request.extra.get("coverage_baseline", ""),
        )
        assertion = _as_str(parsed["assertion"])
        coverage = _as_map(parsed["coverage"])
        status = _process_status(proc, ingested=bool(assertion))
        if assertion and status in {ResultStatus.EXECUTED, ResultStatus.INGESTED}:
            status = ResultStatus.INTERESTING
        return DynamicResult(
            engine=self.engine_id,
            engine_version=self.version(),
            language=request.language,
            target=request.target,
            contract=request.contract,
            function=_as_str(parsed["property"]) or request.function,
            status=status,
            executed=_executed(proc),
            exit_code=proc.return_code,
            stdout=proc.stdout[:4000],
            stderr=proc.stderr[:2000],
            assertion=assertion,
            coverage=coverage,
            minimized_input=_as_str(parsed["sequence"]),
            reproduction_command=" ".join(argv),
            provenance="medusa",
            oracle_kind="property" if assertion else "",
            oracle_explanation=assertion,
            metadata={"verified": "false", "license": self.license_note, **_proc_meta(proc)},
        )


class HalmosEngine(ExternalDiscoveryEngine):
    binary = "halmos"
    license_note = "AGPL external symbolic-testing executable; source is not vendored"

    @property
    def engine_id(self) -> str:
        return "halmos"

    @property
    def display_name(self) -> str:
        return "Halmos"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset({EngineCapability.SYMBOLIC_EXECUTION, EngineCapability.RESULTS_INGESTION})

    def start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        if not request.difficult:
            return DynamicResult(
                engine=self.engine_id,
                language=request.language,
                target=request.target,
                status=ResultStatus.PLANNED,
                executed=False,
                oracle_explanation="Halmos runs only for a target marked difficult to reach.",
            )
        if not halmos_target_supported(request):
            return DynamicResult(
                engine=self.engine_id,
                language=request.language,
                target=request.target,
                status=ResultStatus.NOT_IMPLEMENTED,
                executed=False,
                oracle_explanation="Halmos runs a Foundry symbolic test. This target is not one.",
            )
        return super().start_campaign(request)

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        if not request.difficult:
            return DynamicResult(
                engine=self.engine_id,
                language=request.language,
                target=request.target,
                status=ResultStatus.PLANNED,
                executed=False,
                oracle_explanation="Halmos runs only for a target marked difficult to reach.",
            )
        if not halmos_target_supported(request):
            return DynamicResult(
                engine=self.engine_id,
                language=request.language,
                target=request.target,
                status=ResultStatus.NOT_IMPLEMENTED,
                executed=False,
                oracle_explanation=("Halmos runs a Foundry symbolic test. This target is not one."),
            )
        if self.availability() is not EngineAvailability.AVAILABLE:
            return self._unavailable(request)
        test_name = request.match_test or request.function
        argv = ["halmos", "--match-test", test_name]
        proc = run_command(argv, cwd=request.repo_root, timeout=120)
        parsed = parse_halmos_output(proc.stdout)
        status = _process_status(proc, ingested=bool(parsed["counterexample"]))
        if parsed["counterexample"] and status in {ResultStatus.EXECUTED, ResultStatus.INGESTED}:
            status = ResultStatus.INTERESTING
        return DynamicResult(
            engine=self.engine_id,
            engine_version=self.version(),
            language=request.language,
            target=request.target,
            function=parsed["test"] or request.function,
            contract=request.contract,
            source_file=request.source_file,
            status=status,
            executed=_executed(proc),
            exit_code=proc.return_code,
            stdout=proc.stdout[:4000],
            stderr=proc.stderr[:2000],
            minimized_input=parsed["counterexample"],
            assertion=parsed["counterexample"],
            reproduction_command=" ".join(argv),
            provenance="halmos",
            oracle_kind="symbolic" if parsed["counterexample"] else "",
            oracle_explanation=parsed["counterexample"],
            metadata={
                "seed_source": "symbolic" if parsed["counterexample"] else "",
                "verified": "false",
                "license": self.license_note,
                **_proc_meta(proc),
            },
        )


class WakeEngine(ExternalDiscoveryEngine):
    binary = "wake"
    license_note = "ISC external executable; BugForge does not depend on Wake"

    @property
    def engine_id(self) -> str:
        return "wake"

    @property
    def display_name(self) -> str:
        return "Wake"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset(
            {
                EngineCapability.STATIC_ANALYSIS,
                EngineCapability.FUZZING,
                EngineCapability.RESULTS_INGESTION,
            }
        )

    def _analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        return self._run(request, ["wake", "detect"], fuzz=False)

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        return self._run(request, ["wake", "fuzz"], fuzz=True)

    def _run(self, request: AnalysisRequest, argv: list[str], *, fuzz: bool) -> DynamicResult:
        if self.availability() is not EngineAvailability.AVAILABLE:
            return self._unavailable(request)
        proc = run_command(argv, cwd=request.repo_root, timeout=90)
        findings = () if fuzz else tuple(parse_wake_detect(proc.stdout))
        status = _process_status(proc, ingested=bool(findings))
        return DynamicResult(
            engine=self.engine_id,
            engine_version=self.version(),
            language=request.language,
            target=request.target,
            status=status,
            executed=_executed(proc),
            exit_code=proc.return_code,
            stdout=proc.stdout[:4000],
            stderr=proc.stderr[:2000],
            findings=findings,
            provenance="wake",
            metadata={
                "license": self.license_note,
                "verified": "false",
                "mode": argv[-1],
                **_proc_meta(proc),
            },
        )


def normalize_slither(payload: str) -> list[DynamicFinding]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    detectors = data.get("results", {}).get("detectors", []) if isinstance(data, dict) else []
    findings: list[DynamicFinding] = []
    if not isinstance(detectors, list):
        return []
    for item in detectors:
        if not isinstance(item, dict):
            continue
        raw_elements = item.get("elements")
        elements = raw_elements if isinstance(raw_elements, list) else []
        located = [element for element in elements if isinstance(element, dict)]
        if not located:
            located = [{}]
        for element in located:
            raw_mapping = element.get("source_mapping")
            mapping = raw_mapping if isinstance(raw_mapping, dict) else {}
            raw_lines = mapping.get("lines")
            lines = raw_lines if isinstance(raw_lines, list) else []
            if not mapping and len(located) > 1:
                continue
            findings.append(
                DynamicFinding(
                    detector_id=str(item.get("check") or "slither"),
                    title=str(item.get("check") or "slither"),
                    severity=str(item.get("impact") or ""),
                    confidence=str(item.get("confidence") or ""),
                    function=str(element.get("name") or ""),
                    contract=_slither_contract(element),
                    file_path=str(
                        mapping.get("filename_relative") or mapping.get("filename") or ""
                    ),
                    line=int(lines[0]) if lines and isinstance(lines[0], int) else 0,
                    description=str(item.get("description") or "")[:500],
                    status="potential",
                )
            )
    return findings


def _slither_contract(element: dict[str, object]) -> str:
    specific = element.get("type_specific_fields")
    if not isinstance(specific, dict):
        return ""
    parent = specific.get("parent")
    if not isinstance(parent, dict):
        return ""
    return str(parent.get("name") or "")


def project_uses_foundry(root: Path) -> bool:
    return (root / "foundry.toml").is_file()


def read_foundry_config(root: Path) -> dict[str, str]:
    path = root / "foundry.toml"
    found: dict[str, str] = {}
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="replace")
        for key in ("src", "test", "script", "libs", "solc_version", "solc"):
            match = re.search(rf"^{key}\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
            if match:
                found[key] = match.group(1)
    for dirname in ("src", "test", "script", "lib"):
        if (root / dirname).is_dir():
            found.setdefault(dirname, dirname)
    return found


def parse_foundry_output(stdout: str, *, baseline: str = "", mode: str = "") -> dict[str, object]:
    fails = re.findall(r"\[FAIL[^\]]*\]\s*([^\n]+)", stdout)
    assertion = fails[0].strip() if fails else ""
    coverage: dict[str, str] = {"coverage_available": "false", "coverage_source": "foundry"}
    if mode == "coverage" or re.search(r"\bcoverage\b", stdout, re.IGNORECASE):
        match = re.search(r"(\d+(?:\.\d+)?)%", stdout)
        if match:
            _record_percent(coverage, match.group(1), baseline, source="foundry")
    return {"assertion": assertion, "coverage": coverage}


def parse_echidna_output(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        parsed = _parse_echidna_json(stripped)
        if parsed is not None:
            return parsed
    assertion = ""
    property_name = ""
    for line in text.splitlines():
        if "falsified" in line.lower() or re.search(r"\bfailed!?\b", line.lower()):
            assertion = line.strip()[:240]
            property_name = line.split(":", 1)[0].strip()
            break
    sequence = ""
    if "call sequence" in text.lower():
        chunk = re.split(r"call sequence", text, flags=re.IGNORECASE, maxsplit=1)[-1]
        sequence = "\n".join(line.strip() for line in chunk.splitlines()[1:8] if line.strip())
    coverage: dict[str, str] = {"coverage_available": "false", "coverage_source": "echidna"}
    corpus = re.search(r"corpus[^\d]*(\d+)", text, re.IGNORECASE)
    if corpus:
        coverage["corpus"] = corpus.group(1)
        coverage["coverage_available"] = "true"
    passed = bool(re.search(r"\bpassing\b|\bpassed\b", text, re.IGNORECASE)) and not assertion
    return {
        "assertion": assertion,
        "property": property_name,
        "sequence": sequence[:500],
        "coverage": coverage,
        "contract": "",
        "campaign_success": "true" if passed else "false" if assertion else "",
    }


def parse_medusa_output(text: str, *, baseline: str = "") -> dict[str, object]:
    assertion = ""
    property_name = ""
    for line in text.splitlines():
        if re.search(r"assertion failed|property failed|\[FAIL", line, re.IGNORECASE):
            assertion = line.strip()[:240]
            property_name = line.strip()[:80]
            break
    coverage: dict[str, str] = {"coverage_available": "false", "coverage_source": "medusa"}
    percent = re.search(r"coverage[^\d]*(\d+(?:\.\d+)?)%", text, re.IGNORECASE)
    if percent:
        _record_percent(coverage, percent.group(1), baseline, source="medusa")
    corpus = re.search(r"corpus[^\d]*(\d+)", text, re.IGNORECASE)
    if corpus:
        coverage["corpus"] = corpus.group(1)
        coverage["coverage_available"] = "true"
    sequence = ""
    if "call sequence" in text.lower():
        sequence = text[
            text.lower().find("call sequence") : text.lower().find("call sequence") + 300
        ]
    return {
        "assertion": assertion,
        "property": property_name,
        "coverage": coverage,
        "sequence": sequence[:500],
    }


def parse_halmos_output(stdout: str) -> dict[str, str]:
    counterexample = ""
    if "Counterexample" in stdout:
        start = stdout.find("Counterexample")
        counterexample = stdout[start : start + 500].strip()
    test = ""
    match = re.search(r"\[(?:FAIL|PASS)[^\]]*\]\s*([^\n(]+)", stdout)
    if match:
        test = match.group(1).strip()
    return {"counterexample": counterexample, "test": test}


def parse_wake_detect(stdout: str) -> list[DynamicFinding]:
    findings: list[DynamicFinding] = []
    for line in stdout.splitlines():
        match = re.search(r"([A-Za-z0-9_.-]+):(\d+):\s*(.+)", line)
        if match is None:
            continue
        findings.append(
            DynamicFinding(
                detector_id="wake",
                title=match.group(3)[:120],
                file_path=match.group(1),
                line=int(match.group(2)),
                description=line.strip()[:500],
                status="potential",
            )
        )
    return findings


def discover_forge_tests(root: Path) -> tuple[str, ...]:
    names: list[str] = []
    test_dir = root / "test"
    if not test_dir.is_dir():
        return ()
    for path in sorted(test_dir.rglob("*.sol")):
        text = path.read_text(encoding="utf-8", errors="replace")
        names.extend(re.findall(r"function\s+((?:test|invariant)\w*)\s*\(", text))
    return tuple(dict.fromkeys(names))


def foundry_invocation(
    mode: str, request: AnalysisRequest, *, tests: tuple[str, ...] = ()
) -> tuple[list[str], str]:
    """Build a Foundry command and an honest campaign label.

    A generic ``forge test`` is a test run. It becomes a fuzz or invariant
    campaign only when a real matching test is known.
    """
    if mode == "build":
        return ["forge", "build"], "build"
    if mode == "coverage":
        return ["forge", "coverage", "--report", "summary"], "coverage"
    argv = ["forge", "test"]
    label = "test"
    match = ""
    if mode == "fuzz":
        match = _known_match(request, tests, _looks_fuzz)
        label = "fuzz" if match else "test"
    elif mode == "invariant":
        match = _known_match(request, tests, _looks_invariant)
        label = "invariant" if match else "test"
    elif request.match_test:
        match = request.match_test
    if match:
        argv.extend(["--match-test", match, "-vv"])
    if label == "fuzz":
        runs = request.extra.get("fuzz_runs", "64")
        if runs.isdigit():
            argv.extend(["--fuzz-runs", runs])
    return argv, label


def halmos_target_supported(request: AnalysisRequest) -> bool:
    name = request.match_test or request.function
    if not name:
        return False
    if request.extra.get("symbolic_test") == "true":
        return True
    lowered = name.lower()
    return (
        lowered.startswith("test")
        or lowered.startswith("check_")
        or lowered.startswith("invariant")
    )


def _known_match(
    request: AnalysisRequest,
    tests: tuple[str, ...],
    predicate: Callable[[str], bool],
) -> str:
    requested = request.match_test or request.function
    if requested and requested in tests and predicate(requested):
        return requested
    if requested and not tests and predicate(requested):
        return requested
    matches = [name for name in tests if predicate(name)]
    if len(matches) == 1:
        return matches[0]
    return ""


def _looks_fuzz(name: str) -> bool:
    text = name.lower()
    return text.startswith("testfuzz") or "fuzz" in text


def _looks_invariant(name: str) -> bool:
    return name.lower().startswith("invariant")


def _process_status(proc: ProcessResult, *, ingested: bool) -> ResultStatus:
    if not proc.available:
        return ResultStatus.UNAVAILABLE
    if proc.timed_out:
        return ResultStatus.TIMEOUT
    if not proc.started:
        return ResultStatus.TOOL_FAILURE
    tool_error = (
        "compiler run failed" in proc.stderr.lower()
        or "traceback (most recent call last)" in (proc.stderr + proc.stdout).lower()
    )
    if tool_error and not ingested:
        return ResultStatus.TOOL_FAILURE
    if ingested:
        return ResultStatus.INGESTED
    return ResultStatus.EXECUTED


def _executed(proc: ProcessResult) -> bool:
    return proc.started and not proc.timed_out


def _proc_meta(proc: ProcessResult) -> dict[str, str]:
    code = "" if proc.return_code is None else str(proc.return_code)
    return {
        "started": str(proc.started).lower(),
        "timed_out": str(proc.timed_out).lower(),
        "available": str(proc.available).lower(),
        "return_code": code,
    }


def _record_percent(coverage: dict[str, str], percent: str, baseline: str, *, source: str) -> None:
    coverage["percent"] = percent
    coverage["coverage_percent"] = percent
    coverage["coverage_available"] = "true"
    coverage["coverage_source"] = source
    if not baseline:
        return
    try:
        delta = float(percent) - float(baseline)
    except ValueError:
        return
    coverage["coverage_delta"] = str(delta)
    increased = delta > 0
    coverage["new_coverage"] = "true" if increased else "false"


def _parse_echidna_json(text: str) -> dict[str, object] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        tests = data.get("tests", [])
    elif isinstance(data, list):
        tests = data
    else:
        return None
    if not isinstance(tests, list):
        return None
    assertion = ""
    property_name = ""
    sequence = ""
    contract = ""
    failed = False
    for item in tests:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").lower()
        name = str(item.get("name") or item.get("property") or "")
        if status in {"falsified", "failed", "solved"} or item.get("error"):
            failed = True
            assertion = str(item.get("error") or status or "falsified")[:240]
            property_name = name
            transactions = item.get("transactions") or item.get("call_sequence") or []
            if isinstance(transactions, list):
                sequence = "\n".join(str(step) for step in transactions[:8])
            contract = str(item.get("contract") or "")
            break
        if status in {"passing", "passed"}:
            property_name = property_name or name
    coverage: dict[str, str] = {"coverage_available": "false", "coverage_source": "echidna"}
    corpus = data.get("corpus") if isinstance(data, dict) else None
    if corpus:
        coverage["corpus"] = str(corpus)
        coverage["coverage_available"] = "true"
    campaign = "false" if failed else "true" if tests else ""
    if isinstance(data, dict) and isinstance(data.get("success"), bool) and not tests:
        campaign = "true" if data["success"] else "false"
        if not data["success"]:
            assertion = str(data.get("error") or "campaign failed")
    return {
        "assertion": assertion,
        "property": property_name,
        "sequence": sequence[:500],
        "coverage": coverage,
        "contract": contract,
        "campaign_success": campaign,
    }


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _rejects_fork(request: AnalysisRequest) -> bool:
    blob = " ".join(request.extra.values()).lower()
    return "fork-url" in blob or "fork_url" in blob or request.extra.get("fork") == "true"
