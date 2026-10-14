"""Controlled exploratory tests.

The external model may propose a hypothesis and a test. BugForge validates
the proposal, chooses the command, and executes it only in Docker with the
network disabled. Generated code is untrusted. A failing test is not a
verified bug, and a passing test is not a vulnerability.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.execution.base import OUTPUT_CONTAINER_PATH, ExecutionConfig, ExecutionResult, TestExecutor
from app.execution.local_executor import LocalTestExecutor
from app.security_agent.evidence_graph import EvidenceGraph
from app.security_testing.sanitization import wrap_untrusted
from app.security_testing.secrets import redact_text
from app.testing.test_validator import validate_test_code

_REPO_MOUNT = "/bugforge-repo"


class ExploratoryClassification(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"
    ERROR = "ERROR"
    DUPLICATE = "DUPLICATE"
    UNSUPPORTED = "UNSUPPORTED"
    COMPILATION_FAILED = "COMPILATION_FAILED"


class RepeatBehavior(StrEnum):
    DETERMINISTIC_FAILURE = "deterministic failure"
    DETERMINISTIC_SUCCESS = "deterministic success"
    FLAKY = "flaky"
    ENVIRONMENT_ERROR = "environment error"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ExploratoryOracle:
    kind: str
    expected_behavior: str


@dataclass(frozen=True)
class ExecutionProfile:
    profile_id: str
    language: str
    framework: str
    command: tuple[str, ...]
    network: str = "none"
    timeout_seconds: int = 30
    memory_mb: int = 256
    cpu: float = 0.5
    test_name: str = "test_exploratory.py"

    def container_command(self) -> list[str]:
        test_path = f"{OUTPUT_CONTAINER_PATH}/{self.test_name}"
        return [part.replace("{test}", test_path) for part in self.command]


@dataclass(frozen=True)
class ExploratoryTestPolicy:
    enabled: bool = False
    lab_only: bool = True
    require_docker: bool = True
    network: str = "none"
    max_tests: int = 4
    max_iterations: int = 3
    max_seconds: float = 60.0
    max_test_bytes: int = 16_384
    max_artifact_bytes: int = 65_536
    timeout_seconds: int = 30
    memory_mb: int = 256
    cpu: float = 0.5

    @classmethod
    def from_settings(cls) -> ExploratoryTestPolicy:
        from app.core.config import get_settings

        settings = get_settings()
        return cls(
            enabled=bool(settings.security_agent_exploratory_enabled),
            require_docker=bool(settings.security_agent_exploratory_require_docker),
            max_tests=int(settings.security_agent_max_exploratory_tests),
            max_iterations=int(settings.security_agent_max_exploratory_iterations),
            max_seconds=float(settings.security_agent_max_exploratory_seconds),
            max_test_bytes=int(settings.security_agent_max_generated_test_bytes),
            max_artifact_bytes=int(settings.security_agent_max_exploratory_artifact_bytes),
        )


@dataclass
class ExploratoryTestCandidate:
    target: str
    language: str
    framework: str
    target_symbol: str
    hypothesis_id: str
    test_code: str
    rationale: str
    expected_behavior: str
    oracle: str
    confidence: str = "low"
    parent_attempt_id: str = ""
    target_file: str = ""
    project_id: str = ""
    session_id: str = ""
    candidate_hash: str = ""
    follow_up: str = ""


@dataclass
class ExploratoryTestAttempt:
    attempt_id: str
    candidate_id: str
    repository_snapshot: str
    repository_commit: str
    test_hash: str
    execution_profile: str
    command_identity: tuple[str, ...]
    classification: str
    stdout: str = ""
    stderr: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    duration: float = 0.0
    timeout: bool = False
    evidence_ids: list[str] = field(default_factory=list)
    parent_attempt: str = ""
    hypothesis_id: str = ""
    session_id: str = ""
    project_id: str = ""
    detail: str = ""
    executed: bool = False
    meaningful: bool = False
    follow_up_recommended: bool = False
    repeat: str = RepeatBehavior.INCONCLUSIVE.value
    verified: bool = False


@dataclass
class ExploratoryContext:
    session_id: str
    project_id: str
    mode: str
    hypotheses: dict[str, str]
    repo_path: str
    commit_sha: str
    snapshot_hash: str
    graph: EvidenceGraph | None = None


_FOLLOW_UPS = frozenset(
    {"confirmation", "falsification", "boundary", "negative_control", "alternative"}
)
_MAX_FOLLOW_UPS = 4
_SKIP_SNAPSHOT = {".git", "__pycache__", "bugforge-explore", "bugforge-output"}


@dataclass
class ExploratoryRecord:
    """Persisted exploratory attempt. Secrets and process environment are omitted."""

    attempt_id: str
    session_id: str
    project_id: str
    hypothesis_id: str
    parent_attempt_id: str
    candidate_id: str
    test_hash: str
    target_file: str
    target_symbol: str
    language: str
    framework: str
    test_code: str
    rationale: str
    expected_behavior: str
    oracle: str
    confidence: str
    repository_snapshot: str
    repository_commit: str
    execution_profile: str
    command_identity: list[str]
    classification: str
    stdout_summary: str
    stderr_summary: str
    artifacts: dict[str, str]
    duration: float
    timeout: bool
    executed: bool
    meaningful: bool
    follow_up_recommended: bool
    repeat_classification: str
    verified: bool = False
    follow_up: str = ""
    created_at: str = ""
    replayable: bool = True


@dataclass
class _StoredAttempt:
    attempt: ExploratoryTestAttempt
    test_code: str
    target_symbol: str
    target_file: str
    language: str
    framework: str
    oracle: str
    expected_behavior: str
    confidence: str
    rationale: str
    follow_up: str
    replayable: bool = True


class ExploratoryEngine:
    """Validate and run one exploratory candidate. Does not verify findings."""

    def __init__(
        self,
        *,
        policy: ExploratoryTestPolicy | None = None,
        executor: TestExecutor | None = None,
        docker_available: bool | None = None,
        forge_available: bool | None = None,
        pytest_ready: bool | None = None,
        records: list[ExploratoryRecord] | None = None,
    ) -> None:
        self.policy = policy or ExploratoryTestPolicy.from_settings()
        self.executor = executor
        self.docker_available = docker_available
        self.forge_available = forge_available
        self.pytest_ready = pytest_ready
        self.records = records if records is not None else []
        self._attempts: dict[str, _StoredAttempt] = {}
        self._by_hash: dict[str, str] = {}
        self.tests_run = 0
        self.seconds_used = 0.0
        self.iterations = 0
        self._reindex()

    async def run(
        self, candidate: ExploratoryTestCandidate, context: ExploratoryContext
    ) -> ExploratoryTestAttempt:
        profile = profile_for(
            candidate.language,
            candidate.framework,
            forge_available=self._forge_available(),
        )
        identity = self._identity(candidate, context, profile.profile_id if profile else "none")
        stored = self._by_hash.get(identity)
        if stored and not candidate.rationale.startswith("replay:"):
            prior = self._attempts[stored].attempt
            return self._duplicate(prior)
        rejection = self._reject(candidate, context, profile)
        if rejection is not None:
            return rejection
        assert profile is not None
        if (
            candidate.language == "python"
            and candidate.framework == "pytest"
            and not self._pytest_ok()
        ):
            return self._result(
                candidate,
                context,
                identity,
                profile,
                ExploratoryClassification.UNAVAILABLE,
                "UNAVAILABLE: the exploratory Python image does not provide pytest. "
                "BugForge will not enable the network or pip install dependencies.",
            )
        if self.tests_run >= self.policy.max_tests or self.iterations >= self.policy.max_iterations:
            return self._result(
                candidate,
                context,
                identity,
                profile,
                ExploratoryClassification.BLOCKED,
                "exploratory budget exhausted",
            )
        if self.seconds_used >= self.policy.max_seconds:
            return self._result(
                candidate,
                context,
                identity,
                profile,
                ExploratoryClassification.BLOCKED,
                "exploratory time budget exhausted",
            )
        if not self._docker_ok():
            return self._result(
                candidate,
                context,
                identity,
                profile,
                ExploratoryClassification.UNAVAILABLE,
                "Docker is required for exploratory code and is not available",
            )
        if isinstance(self.executor, LocalTestExecutor):
            return self._result(
                candidate,
                context,
                identity,
                profile,
                ExploratoryClassification.UNAVAILABLE,
                "local execution is not allowed for untrusted exploratory code",
            )
        return await self._execute(candidate, context, profile, identity)

    def replay(
        self, attempt_id: str, context: ExploratoryContext
    ) -> ExploratoryTestCandidate | None:
        stored = self._attempts.get(attempt_id)
        if stored is None:
            return None
        attempt = stored.attempt
        if (
            attempt.repository_snapshot != context.snapshot_hash
            or attempt.repository_commit != context.commit_sha
        ):
            return None
        if not stored.replayable:
            return None
        if hashlib.sha256(stored.test_code.encode("utf-8")).hexdigest() != attempt.test_hash:
            return None
        return ExploratoryTestCandidate(
            target=stored.target_file or stored.target_symbol,
            language=stored.language,
            framework=stored.framework,
            target_symbol=stored.target_symbol,
            hypothesis_id=attempt.hypothesis_id,
            test_code=stored.test_code,
            rationale="replay:" + attempt_id,
            expected_behavior=stored.expected_behavior,
            oracle=stored.oracle,
            confidence=stored.confidence,
            parent_attempt_id=attempt_id,
            target_file=stored.target_file,
            project_id=attempt.project_id,
            session_id=attempt.session_id,
            follow_up=stored.follow_up,
        )

    def repeat_behavior(self, candidate_hash: str) -> str:
        group = [
            item.attempt
            for item in self._attempts.values()
            if item.attempt.candidate_id == candidate_hash and item.attempt.executed
        ]
        if not group:
            return RepeatBehavior.INCONCLUSIVE.value
        kinds = {item.classification for item in group}
        if kinds <= {
            ExploratoryClassification.ERROR.value,
            ExploratoryClassification.TIMEOUT.value,
        }:
            return RepeatBehavior.ENVIRONMENT_ERROR.value
        if (
            ExploratoryClassification.PASSED.value in kinds
            and ExploratoryClassification.FAILED.value in kinds
        ):
            return RepeatBehavior.FLAKY.value
        if kinds == {ExploratoryClassification.FAILED.value}:
            return RepeatBehavior.DETERMINISTIC_FAILURE.value
        if kinds == {ExploratoryClassification.PASSED.value}:
            return RepeatBehavior.DETERMINISTIC_SUCCESS.value
        return RepeatBehavior.INCONCLUSIVE.value

    def _reject(
        self,
        candidate: ExploratoryTestCandidate,
        context: ExploratoryContext,
        profile: ExecutionProfile | None,
    ) -> ExploratoryTestAttempt | None:
        if not self.policy.enabled:
            return self._result(
                candidate,
                context,
                "",
                profile,
                ExploratoryClassification.BLOCKED,
                "exploratory testing is disabled",
            )
        if self.policy.lab_only and context.mode != "lab":
            return self._result(
                candidate,
                context,
                "",
                profile,
                ExploratoryClassification.BLOCKED,
                "exploratory generated code is lab-only",
            )
        if candidate.session_id and candidate.session_id != context.session_id:
            return self._result(
                candidate, context, "", profile, ExploratoryClassification.INVALID, "wrong session"
            )
        if candidate.project_id and candidate.project_id != context.project_id:
            return self._result(
                candidate, context, "", profile, ExploratoryClassification.INVALID, "wrong project"
            )
        if candidate.hypothesis_id not in context.hypotheses:
            return self._result(
                candidate,
                context,
                "",
                profile,
                ExploratoryClassification.INVALID,
                "wrong hypothesis",
            )
        if candidate.parent_attempt_id and candidate.parent_attempt_id not in self._attempts:
            return self._result(
                candidate,
                context,
                "",
                profile,
                ExploratoryClassification.INVALID,
                "unknown parent attempt",
            )
        if candidate.parent_attempt_id:
            parent = self._attempts[candidate.parent_attempt_id].attempt
            if (
                parent.session_id != context.session_id
                or parent.project_id != context.project_id
                or parent.hypothesis_id != candidate.hypothesis_id
                or parent.repository_snapshot != context.snapshot_hash
            ):
                return self._result(
                    candidate,
                    context,
                    "",
                    profile,
                    ExploratoryClassification.INVALID,
                    "parent attempt identity does not match",
                )
            if self._chain_depth(candidate.parent_attempt_id) >= _MAX_FOLLOW_UPS:
                return self._result(
                    candidate,
                    context,
                    "",
                    profile,
                    ExploratoryClassification.BLOCKED,
                    "exploratory follow-up chain limit",
                )
        if candidate.follow_up and candidate.follow_up not in _FOLLOW_UPS:
            return self._result(
                candidate,
                context,
                "",
                profile,
                ExploratoryClassification.INVALID,
                "unknown follow-up kind",
            )
        if len(candidate.test_code.encode("utf-8")) > self.policy.max_test_bytes:
            return self._result(
                candidate,
                context,
                "",
                profile,
                ExploratoryClassification.BLOCKED,
                "generated test exceeds the byte budget",
            )
        if profile is None:
            state = profile_availability(
                candidate.language,
                candidate.framework,
                pytest_ready=self._pytest_ok(),
                forge_available=self._forge_available(),
            )
            if state == "UNSUPPORTED":
                return self._result(
                    candidate,
                    context,
                    "",
                    None,
                    ExploratoryClassification.UNSUPPORTED,
                    "UNSUPPORTED",
                )
            detail = (
                "UNAVAILABLE: forge is not installed. No Foundry test was executed."
                if candidate.language == "solidity"
                else "UNAVAILABLE: this exploratory profile cannot run in the current environment."
            )
            return self._result(
                candidate,
                context,
                "",
                None,
                ExploratoryClassification.UNAVAILABLE,
                detail,
            )
        problem = _validate_code(candidate.language, candidate.test_code)
        if problem:
            if problem.startswith("BLOCKED"):
                classification = ExploratoryClassification.BLOCKED
            elif problem.startswith("UNSUPPORTED"):
                classification = ExploratoryClassification.UNSUPPORTED
            else:
                classification = ExploratoryClassification.INVALID
            return self._result(candidate, context, "", profile, classification, problem)
        return None

    async def _execute(
        self,
        candidate: ExploratoryTestCandidate,
        context: ExploratoryContext,
        profile: ExecutionProfile,
        identity: str,
    ) -> ExploratoryTestAttempt:
        if self.executor is None:
            return self._result(
                candidate,
                context,
                identity,
                profile,
                ExploratoryClassification.UNAVAILABLE,
                "no Docker executor is bound",
            )
        scratch = Path(tempfile.mkdtemp(prefix="bugforge-explore-"))
        output = scratch / "output"
        output.mkdir()
        if profile.framework == "foundry":
            prepare_foundry_workspace(Path(context.repo_path), output, candidate.test_code)
        else:
            (output / profile.test_name).write_text(candidate.test_code, encoding="utf-8")
        repo = str(Path(context.repo_path).resolve())
        config = ExecutionConfig(
            command=profile.container_command(),
            working_directory=str(scratch),
            timeout_seconds=min(profile.timeout_seconds, self.policy.timeout_seconds),
            memory_limit_mb=min(profile.memory_mb, self.policy.memory_mb),
            cpu_limit=min(profile.cpu, self.policy.cpu),
            environment=_minimal_environment(profile.language),
            allow_network=False,
            output_dir=str(output),
            read_only_volumes={repo: _REPO_MOUNT},
        )
        self.iterations += 1
        try:
            result = await self.executor.execute(config)
        except Exception as exc:
            attempt = self._result(
                candidate,
                context,
                identity,
                profile,
                ExploratoryClassification.ERROR,
                redact_text(str(exc))[:300],
            )
            self._remember(attempt, candidate, identity)
            return attempt
        self.tests_run += 1
        self.seconds_used += result.duration_seconds
        classification = _classify_execution(result)
        stdout = _sanitize_output(result.stdout)
        stderr = _sanitize_output(result.stderr)
        attempt = self._result(
            candidate,
            context,
            identity,
            profile,
            classification,
            candidate.expected_behavior,
        )
        attempt.stdout = stdout
        attempt.stderr = stderr
        attempt.artifacts = _bounded_artifacts(
            result.artifact_contents, self.policy.max_artifact_bytes
        )
        attempt.duration = result.duration_seconds
        attempt.timeout = result.timed_out
        attempt.executed = True
        attempt.meaningful = _meaningful(attempt, candidate)
        attempt.follow_up_recommended = (
            classification
            in {
                ExploratoryClassification.FAILED,
                ExploratoryClassification.TIMEOUT,
            }
            and self.iterations < self.policy.max_iterations
        )
        attempt.verified = False
        self._remember(attempt, candidate, identity)
        attempt.repeat = self.repeat_behavior(identity)
        self._evidence(attempt, candidate, context, profile)
        return attempt

    def _remember(
        self, attempt: ExploratoryTestAttempt, candidate: ExploratoryTestCandidate, identity: str
    ) -> None:
        attempt.candidate_id = identity
        self._attempts[attempt.attempt_id] = _StoredAttempt(
            attempt,
            candidate.test_code,
            candidate.target_symbol,
            candidate.target_file,
            candidate.language,
            candidate.framework,
            candidate.oracle,
            candidate.expected_behavior,
            candidate.confidence,
            candidate.rationale,
            candidate.follow_up,
            redact_text(candidate.test_code) == candidate.test_code,
        )
        self._by_hash.setdefault(identity, attempt.attempt_id)
        self.records.append(_record_from(attempt, candidate))

    def _evidence(
        self,
        attempt: ExploratoryTestAttempt,
        candidate: ExploratoryTestCandidate,
        context: ExploratoryContext,
        profile: ExecutionProfile,
    ) -> None:
        graph = context.graph
        if graph is None:
            return
        hyp_id = candidate.hypothesis_id
        if hyp_id not in graph.nodes:
            graph.add(
                kind="hypothesis",
                provenance="ai_hypothesis",
                summary=hyp_id,
                source="exploratory_test",
                node_id=hyp_id,
                extra={"hypothesis_id": hyp_id, "verified": False},
            )
        source = graph.add(
            kind="exploratory_test",
            provenance="generated_test",
            summary=wrap_untrusted("generated-test", candidate.rationale)[:500],
            source="exploratory_test",
            extra={
                "hypothesis_id": hyp_id,
                "test_hash": attempt.test_hash,
                "profile": profile.profile_id,
                "verified": False,
                "sandbox": True,
            },
        )
        graph.link(hyp_id, source.id, "motivates")
        execution = graph.add(
            kind="exploratory_attempt",
            provenance="sandbox_execution",
            summary=redact_text(attempt.stdout or attempt.classification)[:500],
            source="exploratory_test",
            parent_id=source.id,
            relation="executes",
            extra={
                "hypothesis_id": hyp_id,
                "candidate_hash": attempt.candidate_id,
                "classification": attempt.classification,
                "command": list(attempt.command_identity),
                "commit": attempt.repository_commit,
                "snapshot": attempt.repository_snapshot,
                "verified": False,
                "meaningful": attempt.meaningful,
                "sandbox": True,
                "repeat": attempt.repeat,
                "network": profile.network,
                "timeout_seconds": profile.timeout_seconds,
                "memory_mb": profile.memory_mb,
                "cpu": profile.cpu,
                "image": _executor_image(self.executor),
                "foundry_version": "",
                "compiler_version": "",
                "toolchain_failure": attempt.classification
                == ExploratoryClassification.COMPILATION_FAILED.value,
            },
        )
        if attempt.meaningful:
            graph.link(execution.id, hyp_id, "supports")
        elif attempt.classification == ExploratoryClassification.PASSED.value:
            graph.link(execution.id, hyp_id, "contradicted_by")
        attempt.evidence_ids = [source.id, execution.id]

    def _duplicate(self, prior: ExploratoryTestAttempt) -> ExploratoryTestAttempt:
        return ExploratoryTestAttempt(
            attempt_id=prior.attempt_id,
            candidate_id=prior.candidate_id,
            repository_snapshot=prior.repository_snapshot,
            repository_commit=prior.repository_commit,
            test_hash=prior.test_hash,
            execution_profile=prior.execution_profile,
            command_identity=prior.command_identity,
            classification=ExploratoryClassification.DUPLICATE.value,
            hypothesis_id=prior.hypothesis_id,
            session_id=prior.session_id,
            project_id=prior.project_id,
            detail="duplicate exploratory candidate was not executed again",
            parent_attempt=prior.parent_attempt,
            verified=False,
        )

    def _result(
        self,
        candidate: ExploratoryTestCandidate,
        context: ExploratoryContext,
        identity: str,
        profile: ExecutionProfile | None,
        classification: ExploratoryClassification,
        detail: str,
    ) -> ExploratoryTestAttempt:
        return ExploratoryTestAttempt(
            attempt_id=uuid4().hex,
            candidate_id=identity,
            repository_snapshot=context.snapshot_hash,
            repository_commit=context.commit_sha,
            test_hash=hashlib.sha256(candidate.test_code.encode("utf-8")).hexdigest(),
            execution_profile=profile.profile_id if profile else "",
            command_identity=tuple(profile.container_command()) if profile else (),
            classification=classification.value,
            hypothesis_id=candidate.hypothesis_id,
            session_id=context.session_id,
            project_id=context.project_id,
            parent_attempt=candidate.parent_attempt_id,
            detail=detail,
            verified=False,
        )

    def _identity(
        self, candidate: ExploratoryTestCandidate, context: ExploratoryContext, profile_id: str
    ) -> str:
        payload = {
            "commit": context.commit_sha,
            "expected_behavior": candidate.expected_behavior,
            "framework": candidate.framework,
            "hypothesis": candidate.hypothesis_id,
            "language": candidate.language,
            "oracle": candidate.oracle,
            "profile": profile_id,
            "project": context.project_id,
            "session": context.session_id,
            "snapshot": context.snapshot_hash,
            "target_file": candidate.target_file,
            "compiler_version": "",
            "foundry_version": "",
            "image": _executor_image(self.executor),
            "target_symbol": candidate.target_symbol,
            "test_code": candidate.test_code,
            "test_hash": hashlib.sha256(candidate.test_code.encode("utf-8")).hexdigest(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        candidate.candidate_hash = digest
        return digest

    def _chain_depth(self, attempt_id: str) -> int:
        depth = 0
        seen: set[str] = set()
        current = attempt_id
        while current and current not in seen and depth < 8:
            seen.add(current)
            stored = self._attempts.get(current)
            if stored is None:
                break
            current = stored.attempt.parent_attempt
            depth += 1
        return depth

    def _reindex(self) -> None:
        self._attempts.clear()
        self._by_hash.clear()
        self.tests_run = 0
        self.iterations = 0
        self.seconds_used = 0.0
        for record in self.records:
            stored = _stored_from_record(record)
            self._attempts[record.attempt_id] = stored
            if (
                record.classification != ExploratoryClassification.DUPLICATE.value
                and record.candidate_id
            ):
                self._by_hash.setdefault(record.candidate_id, record.attempt_id)
            if record.executed:
                self.tests_run += 1
                self.iterations += 1
                self.seconds_used += record.duration

    def _pytest_ok(self) -> bool:
        if self.pytest_ready is not None:
            return self.pytest_ready
        from app.core.config import get_settings

        return bool(get_settings().security_agent_exploratory_pytest_ready)

    def _docker_ok(self) -> bool:
        if not self.policy.require_docker:
            return False
        if self.docker_available is not None:
            return self.docker_available
        return shutil.which("docker") is not None

    def _forge_available(self) -> bool:
        if self.forge_available is not None:
            return self.forge_available
        return shutil.which("forge") is not None


def _executor_image(executor: TestExecutor | None) -> str:
    """Configured sandbox image name. Unknown stays empty. Secrets are not read."""
    image = getattr(executor, "_image", "")
    if not isinstance(image, str):
        return ""
    cleaned = image.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,200}", cleaned):
        return ""
    return cleaned


def docker_executor_for_exploratory(available: bool, image: str = "") -> TestExecutor | None:
    """Bind Docker when it is available. Never bind the host executor."""
    if not available:
        return None
    from app.execution.docker_executor import DockerTestExecutor

    cleaned = image.strip()
    if cleaned and not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,200}", cleaned):
        return None
    return DockerTestExecutor(image=cleaned) if cleaned else DockerTestExecutor()


def prepare_foundry_workspace(repo: Path, output: Path, test_code: str) -> Path:
    """Copy a bounded source snapshot into a disposable Foundry root and write the test there.

    The source repository is not modified. The generated test is not committed.
    """
    project = output / "project"
    project.mkdir(parents=True, exist_ok=True)
    root = repo.resolve() if repo.exists() else repo
    copied = 0
    total = 0
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if copied >= 40 or total >= 200_000:
                break
            if any(
                part in {".git", "out", "cache", "broadcast", "node_modules", "bugforge-output"}
                for part in path.parts
            ):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix != ".sol" and path.name not in {"foundry.toml", "remappings.txt"}:
                continue
            try:
                relative = path.resolve().relative_to(root)
            except ValueError:
                continue
            data = path.read_bytes()
            if total + len(data) > 200_000:
                break
            destination = project / relative
            if not destination.resolve().is_relative_to(project.resolve()):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            copied += 1
            total += len(data)
    test_path = project / "test" / "Exploratory.t.sol"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(test_code, encoding="utf-8")
    return project


def profile_for(
    language: str, framework: str, *, forge_available: bool = False
) -> ExecutionProfile | None:
    key = (language.strip().lower(), framework.strip().lower())
    if key == ("python", "pytest"):
        return ExecutionProfile(
            "python-pytest-v1",
            "python",
            "pytest",
            ("python", "-m", "pytest", "-p", "no:cacheprovider", "{test}"),
            test_name="test_exploratory.py",
        )
    if key == ("solidity", "foundry") and forge_available:
        return ExecutionProfile(
            "solidity-foundry-v1",
            "solidity",
            "foundry",
            (
                "forge",
                "test",
                "--root",
                "/bugforge-output/project",
                "--match-path",
                "test/Exploratory.t.sol",
                "--offline",
                "--out",
                "/bugforge-output/out",
                "--cache-path",
                "/bugforge-output/cache",
            ),
            test_name="project/test/Exploratory.t.sol",
        )
    return None


def profile_availability(
    language: str, framework: str, *, pytest_ready: bool, forge_available: bool
) -> str:
    """AVAILABLE, UNAVAILABLE, or UNSUPPORTED. Missing tools are not test failures."""
    key = (language.strip().lower(), framework.strip().lower())
    if key == ("python", "pytest"):
        return "AVAILABLE" if pytest_ready else "UNAVAILABLE"
    if key == ("solidity", "foundry"):
        return "AVAILABLE" if forge_available else "UNAVAILABLE"
    return "UNSUPPORTED"


_BLOCKED_CHEATS = (
    (r"\b(?:tryFfi|ffi)\s*\(", "BLOCKED ffi"),
    (r"\bvm\s*\.\s*(?:ffi|tryFfi)\b", "BLOCKED ffi"),
    (r"--fork-url|--fork-block-number|--fork-chain-id", "BLOCKED network fork"),
    (
        r"\bvm\s*\.\s*(?:createSelectFork|createFork|selectFork|activeFork|rollFork|transact)\b",
        "BLOCKED network fork",
    ),
    (r"\bvm\s*\.\s*(?:rpc|rpcUrl|rpcUrls)\b", "BLOCKED arbitrary RPC"),
    (
        r"\bvm\s*\.\s*(?:getEnv|setEnv|envAddress|envBool|envBytes32|envBytes|envInt|envOr|"
        r"envString|envUint|envExists|envAddressOr|envBoolOr|envBytes32Or|envBytesOr|"
        r"envIntOr|envStringOr|envUintOr)\b",
        "BLOCKED environment-secret access",
    ),
    (
        r"\bvm\s*\.\s*(?:readFile|readFileBinary|readLine|readLink|readDir|writeFile|"
        r"writeFileBinary|writeLine|copyFile|removeFile|removeDir|createDir|fsMetadata|"
        r"projectRoot|getCode|getDeployedCode|exists|isFile|isDir|closeFile|writeJson|"
        r"parseJson|keyExistsJson|serializeJson)\b",
        "BLOCKED filesystem",
    ),
    (r"\b(?:vm\s*\.\s*)?process\s*\(", "BLOCKED process"),
    (r"\bvm\s*\.\s*prompt\s*\(", "BLOCKED process"),
    (
        r"(?m)^\s*//\s*forge-config:.*\b(?:fork|ffi|fs_permissions|rpc_endpoints)\b",
        "BLOCKED forge-config",
    ),
)
_ALLOWED_CHEATS = frozenset(
    {
        "prank",
        "startPrank",
        "stopPrank",
        "hoax",
        "deal",
        "warp",
        "roll",
        "store",
        "load",
        "etch",
        "label",
        "assume",
        "expectRevert",
        "expectEmit",
        "expectCall",
        "mockCall",
        "clearMockedCalls",
        "pauseGasMetering",
        "resumeGasMetering",
        "addr",
        "sign",
        "assertEq",
        "assertTrue",
        "assertFalse",
        "assertGt",
        "assertGe",
        "assertLt",
        "assertLe",
        "assertApproxEqAbs",
        "assertApproxEqRel",
    }
)


def validate_solidity_test(code: str) -> str | None:
    """Return a BLOCKED, UNSUPPORTED, or invalid reason, or None when acceptable.

    An unrecognized cheatcode is unsupported. That is not a claim that the test is safe.
    """
    if "function test" not in code:
        return "Solidity exploratory test needs a function whose name starts with test"
    for pattern, reason in _BLOCKED_CHEATS:
        if re.search(pattern, code):
            return reason
    for match in re.finditer(r"\bvm\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)", code):
        name = match.group(1)
        if name not in _ALLOWED_CHEATS:
            return f"UNSUPPORTED unreviewed cheatcode vm.{name}"
    if re.search(r"(?m)^\s*pragma\s+solidity", code) is None:
        return "Solidity exploratory test must declare a pragma"
    return None


def _validate_code(language: str, code: str) -> str | None:
    if language == "python":
        result = validate_test_code(code)
        if not result.valid:
            reason = result.error or "invalid python"
            if "unsafe" in reason or "dangerous" in reason:
                return f"BLOCKED {reason}"
            return reason
        return None
    if language == "solidity":
        return validate_solidity_test(code)
    return "UNSUPPORTED"


def _minimal_environment(language: str) -> dict[str, str]:
    if language == "python":
        return {"PYTHONPATH": _REPO_MOUNT, "PYTHONDONTWRITEBYTECODE": "1"}
    if language == "solidity":
        return {"FOUNDRY_OFFLINE": "1"}
    return {}


_COMPILE_MARKERS = (
    "Compiler run failed",
    "SolcError",
    "ParserError",
    "DeclarationError",
    "TypeError:",
    "Error (",
)


def _classify_execution(result: ExecutionResult) -> ExploratoryClassification:
    if result.timed_out:
        return ExploratoryClassification.TIMEOUT
    if result.error_message:
        return ExploratoryClassification.ERROR
    if result.exit_code == 0:
        return ExploratoryClassification.PASSED
    blob = f"{result.stdout}\n{result.stderr}"
    if any(marker in blob for marker in _COMPILE_MARKERS):
        return ExploratoryClassification.COMPILATION_FAILED
    return ExploratoryClassification.FAILED


def _meaningful(attempt: ExploratoryTestAttempt, candidate: ExploratoryTestCandidate) -> bool:
    return bool(
        attempt.classification == ExploratoryClassification.FAILED.value
        and candidate.expected_behavior.strip()
        and candidate.oracle.strip()
        and candidate.hypothesis_id
        and attempt.executed
    )


def _bounded_artifacts(contents: dict[str, str], limit: int) -> dict[str, str]:
    """Keep artifact hashes, not bodies, and stop once the byte cap is passed."""
    kept: dict[str, str] = {}
    used = 0
    for name in sorted(contents):
        body = contents[name]
        encoded = body.encode("utf-8", errors="replace")
        digest = hashlib.sha256(encoded).hexdigest()
        if used + len(encoded) > limit:
            kept[name] = f"omitted:{digest}"
            continue
        used += len(encoded)
        kept[name] = digest
    return kept


def _sanitize_output(value: str) -> str:
    redacted = redact_text(value or "")
    redacted = re.sub(r"/bugforge-[a-z]+", "[container]", redacted)
    if len(redacted.encode("utf-8")) > 4000:
        redacted = redacted[:4000]
    return wrap_untrusted("exploratory-output", redacted)


def snapshot_hash(path: str) -> str:
    """Hash sorted relative paths and file contents. Timestamps are ignored."""
    root = Path(path)
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    files: list[Path] = []
    for item in root.rglob("*"):
        if item.is_symlink() or not item.is_file():
            continue
        if any(
            part in _SKIP_SNAPSHOT or part.startswith("bugforge-explore") for part in item.parts
        ):
            continue
        files.append(item)
    for item in sorted(files, key=lambda entry: entry.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def environment_fingerprint(environment: dict[str, str]) -> str:
    safe = {
        key: value
        for key, value in environment.items()
        if key in {"PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "FOUNDRY_OFFLINE"}
    }
    payload = "|".join(f"{key}={safe[key]}" for key in sorted(safe))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Host environment is intentionally not read here.


def _record_from(
    attempt: ExploratoryTestAttempt, candidate: ExploratoryTestCandidate
) -> ExploratoryRecord:
    return ExploratoryRecord(
        attempt_id=attempt.attempt_id,
        session_id=attempt.session_id,
        project_id=attempt.project_id,
        hypothesis_id=attempt.hypothesis_id,
        parent_attempt_id=attempt.parent_attempt,
        candidate_id=attempt.candidate_id,
        test_hash=attempt.test_hash,
        target_file=candidate.target_file,
        target_symbol=candidate.target_symbol,
        language=candidate.language,
        framework=candidate.framework,
        test_code=candidate.test_code
        if redact_text(candidate.test_code) == candidate.test_code
        else "",
        replayable=redact_text(candidate.test_code) == candidate.test_code,
        rationale=redact_text(candidate.rationale)[:2000],
        expected_behavior=redact_text(candidate.expected_behavior)[:2000],
        oracle=candidate.oracle,
        confidence=candidate.confidence,
        repository_snapshot=attempt.repository_snapshot,
        repository_commit=attempt.repository_commit,
        execution_profile=attempt.execution_profile,
        command_identity=list(attempt.command_identity),
        classification=attempt.classification,
        stdout_summary=attempt.stdout[:2000],
        stderr_summary=attempt.stderr[:2000],
        artifacts=dict(attempt.artifacts),
        duration=attempt.duration,
        timeout=attempt.timeout,
        executed=attempt.executed,
        meaningful=attempt.meaningful,
        follow_up_recommended=attempt.follow_up_recommended,
        repeat_classification=attempt.repeat,
        verified=False,
        follow_up=candidate.follow_up,
    )


def _stored_from_record(record: ExploratoryRecord) -> _StoredAttempt:
    attempt = ExploratoryTestAttempt(
        attempt_id=record.attempt_id,
        candidate_id=record.candidate_id,
        repository_snapshot=record.repository_snapshot,
        repository_commit=record.repository_commit,
        test_hash=record.test_hash,
        execution_profile=record.execution_profile,
        command_identity=tuple(record.command_identity),
        classification=record.classification,
        stdout=record.stdout_summary,
        stderr=record.stderr_summary,
        artifacts=dict(record.artifacts),
        duration=record.duration,
        timeout=record.timeout,
        parent_attempt=record.parent_attempt_id,
        hypothesis_id=record.hypothesis_id,
        session_id=record.session_id,
        project_id=record.project_id,
        detail=record.expected_behavior,
        executed=record.executed,
        meaningful=record.meaningful,
        follow_up_recommended=record.follow_up_recommended,
        repeat=record.repeat_classification,
        verified=False,
    )
    return _StoredAttempt(
        attempt,
        record.test_code,
        record.target_symbol,
        record.target_file,
        record.language,
        record.framework,
        record.oracle,
        record.expected_behavior,
        record.confidence,
        record.rationale,
        record.follow_up,
        record.replayable,
    )


_CACHE: dict[str, ExploratoryEngine] = {}
_CACHE_LIMIT = 32


def release_exploratory_engine(session_id: str) -> None:
    """Drop the runtime engine. Persisted attempts stay in the repository."""
    _CACHE.pop(session_id, None)


def cached_exploratory_engine(
    session_id: str,
    records: list[ExploratoryRecord],
    **kwargs: Any,
) -> ExploratoryEngine:
    current = _CACHE.get(session_id)
    if current is not None and current.records is records:
        return current
    engine = ExploratoryEngine(records=records, **kwargs)
    _CACHE.pop(session_id, None)
    while len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[session_id] = engine
    return engine


def public_attempt(attempt: ExploratoryTestAttempt) -> dict[str, Any]:
    return {
        "attempt_id": attempt.attempt_id,
        "candidate_id": attempt.candidate_id,
        "classification": attempt.classification,
        "hypothesis_id": attempt.hypothesis_id,
        "executed": attempt.executed,
        "meaningful": attempt.meaningful,
        "follow_up_recommended": attempt.follow_up_recommended,
        "verified": False,
        "repeat": attempt.repeat,
        "stdout": attempt.stdout,
        "stderr": attempt.stderr,
        "detail": redact_text(attempt.detail)[:500],
        "evidence_ids": list(attempt.evidence_ids),
        "profile": attempt.execution_profile,
        "commit": attempt.repository_commit,
        "snapshot": attempt.repository_snapshot,
        "parent_attempt": attempt.parent_attempt,
        "command_supplied_by_model": False,
    }
