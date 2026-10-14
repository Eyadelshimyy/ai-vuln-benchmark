"""Runtime bridges for languages other than Solidity.

These adapters select a real local tool and record execution truth. They do
not invent a fuzz campaign when the repository has no harness, and a missing
executable stays UNAVAILABLE.
"""

from __future__ import annotations

from app.adapters.discovery.external import (
    ExternalDiscoveryEngine,
    _executed,
    _proc_meta,
    _process_status,
)
from app.discovery.capabilities import EngineAvailability, EngineCapability
from app.discovery.engine import AnalysisRequest
from app.discovery.process import run_command, tool_path
from app.discovery.results import DynamicResult, not_implemented_result


class _RuntimeEngine(ExternalDiscoveryEngine):
    campaign_label = "test"

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        built = self.build_command(request)
        if built is None:
            return not_implemented_result(
                self.engine_id, request.language, request.target, "campaign"
            )
        argv, label = built
        if self.availability() is not EngineAvailability.AVAILABLE:
            return self._unavailable(request)
        proc = run_command(argv, cwd=request.repo_root, timeout=60)
        status = _process_status(proc, ingested=False)
        return DynamicResult(
            engine=self.engine_id,
            engine_version=self.version(),
            language=request.language,
            target=request.target,
            function=request.function,
            status=status,
            executed=_executed(proc),
            exit_code=proc.return_code,
            stdout=proc.stdout[:4000],
            stderr=proc.stderr[:2000],
            reproduction_command=" ".join(argv),
            provenance=self.engine_id,
            oracle_explanation=f"{label} campaign recorded. Exit status alone is not a vulnerability.",
            metadata={"verified": "false", "campaign": label, **_proc_meta(proc)},
        )

    def build_command(self, request: AnalysisRequest) -> tuple[list[str], str] | None:
        return None


class GoTestEngine(_RuntimeEngine):
    binary = "go"
    license_note = "Go toolchain, used only when already installed"

    @property
    def engine_id(self) -> str:
        return "go-test"

    @property
    def display_name(self) -> str:
        return "Go test"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"go"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset(
            {
                EngineCapability.TEST_EXECUTION,
                EngineCapability.FUZZING,
                EngineCapability.COVERAGE_FEEDBACK,
            }
        )

    def build_command(self, request: AnalysisRequest) -> tuple[list[str], str] | None:
        return go_test_command(request)


class CargoTestEngine(_RuntimeEngine):
    binary = "cargo"
    license_note = "Rust toolchain, used only when already installed"

    @property
    def engine_id(self) -> str:
        return "cargo-test"

    @property
    def display_name(self) -> str:
        return "Cargo test"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"rust"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset(
            {
                EngineCapability.TEST_EXECUTION,
                EngineCapability.FUZZING,
                EngineCapability.SANITIZER_AWARE,
            }
        )

    def build_command(self, request: AnalysisRequest) -> tuple[list[str], str] | None:
        return cargo_test_command(request)


class NativeFuzzEngine(_RuntimeEngine):
    binary = "clang"
    license_note = "System compiler or AFL++, used only when already installed"

    @property
    def engine_id(self) -> str:
        return "native-fuzz"

    @property
    def display_name(self) -> str:
        return "Native coverage fuzzing"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"c", "cpp"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset({EngineCapability.FUZZING, EngineCapability.COVERAGE_FEEDBACK})

    def availability(self) -> EngineAvailability:
        if tool_path("afl-fuzz") or tool_path("clang"):
            return EngineAvailability.AVAILABLE
        return EngineAvailability.UNAVAILABLE

    def build_command(self, request: AnalysisRequest) -> tuple[list[str], str] | None:
        harness = request.extra.get("fuzz_harness", "")
        if not harness:
            return None
        if tool_path("afl-fuzz"):
            return ["afl-fuzz", "-i", "in", "-o", "out", "--", harness], "coverage-fuzz"
        return None


class SanitizerEngine(_RuntimeEngine):
    binary = "clang"
    license_note = "Sanitizer-enabled compilation requires a local clang"

    @property
    def engine_id(self) -> str:
        return "sanitizer"

    @property
    def display_name(self) -> str:
        return "Sanitizer build"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"c", "cpp", "rust"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset({EngineCapability.SANITIZER_AWARE, EngineCapability.TEST_EXECUTION})

    def build_command(self, request: AnalysisRequest) -> tuple[list[str], str] | None:
        source = request.source_file
        if request.language == "rust":
            if request.extra.get("sanitizer") != "true":
                return None
            return ["cargo", "test"], "sanitizer"
        if not source or not source.endswith((".c", ".cc", ".cpp", ".cxx")):
            return None
        return [
            "clang",
            "-fsanitize=address,undefined",
            "-g",
            source,
            "-o",
            "/tmp/bugforge-sanitizer-probe",
        ], "sanitizer"


def go_test_command(request: AnalysisRequest) -> tuple[list[str], str] | None:
    name = request.function or request.match_test
    if request.extra.get("mode") == "fuzz" or (name and name.startswith("Fuzz")):
        if not name or not name.startswith("Fuzz"):
            return None
        return ["go", "test", "-fuzz", f"^{name}$", "-fuzztime", "1s"], "fuzz"
    return ["go", "test", "./..."], "test"


def cargo_test_command(request: AnalysisRequest) -> tuple[list[str], str] | None:
    name = request.function or request.match_test
    if request.extra.get("mode") == "fuzz":
        if tool_path("cargo-fuzz") is None:
            return None
        target = name or request.target
        if not target:
            return None
        return ["cargo", "fuzz", "run", target], "fuzz"
    argv = ["cargo", "test"]
    if name:
        argv.append(name)
    return argv, "test"
