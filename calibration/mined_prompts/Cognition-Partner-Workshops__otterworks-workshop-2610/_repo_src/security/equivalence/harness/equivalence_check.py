"""Functional-equivalence gate for source-level security refactors.

The harness records how a vulnerable class behaves *before* it is refactored and
then grades the refactor against that recording:

* ``contract`` cases must be byte-identical afterwards - that is the
  functional-equivalence claim, and it is what makes "we only changed how it is
  implemented" checkable instead of reviewable.
* ``attack`` cases must stop firing - that is the vulnerability-closed claim.
* the public interface of the refactored members must be unchanged.
* the module's own suite must still pass every test that passed at record time.

Everything the recording depends on is fingerprinted (cases, seed, emitter,
subject sources), so evidence that no longer describes the code under test is
reported as ``stale`` and refuses to grade rather than passing quietly. Exit
codes: 0 pass, 1 a real divergence, 2 no verdict (missing/stale/unmeasured
evidence), 3 no verdict for a probe-only run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from defusedxml import ElementTree
from tabulate import tabulate

HARNESS_VERSION = 2

ROOT = Path(__file__).resolve().parents[3]
EQUIVALENCE_DIR = ROOT / "security" / "equivalence"
CASES_DIR = EQUIVALENCE_DIR / "cases"
EXPECTED_DIR = EQUIVALENCE_DIR / "expected"
REPORTS_DIR = EQUIVALENCE_DIR / "reports"

OK = "ok"
FAIL = "fail"
MISSING = "missing"
STALE = "stale"
UNMEASURED = "unmeasured"
UNRECORDED = "unrecorded"
NO_VERDICT = "no-verdict"

# Anything in this set means the harness could not reach a verdict. It is never a
# pass: grading a refactor against evidence that does not describe the code under
# test would invent confidence rather than measure it.
INCONCLUSIVE = {MISSING, STALE, UNMEASURED, UNRECORDED}

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_INCONCLUSIVE = 2
EXIT_NO_VERDICT = 3

POLICIES = {"contract", "attack"}
STAGES = {"baseline", "remediated"}
# "auto" resolves per finding from the fingerprints: a subject that still matches
# the recording is graded as the before-state, a subject that changed is graded as
# a refactor. A run therefore cannot pick the easier contract for itself.
STAGE_ARGUMENTS = STAGES | {"auto"}


class HarnessError(RuntimeError):
    """A defect in the fixture or in how the harness was invoked."""


@dataclass
class Module:
    name: str
    path: str
    seed: str
    emitter: str
    emit_command: str
    test_command: str


@dataclass
class Finding:
    id: str
    title: str
    cwe: str
    module: Module
    subject: list[str]
    secure_pattern: str
    observes: list[str] = field(default_factory=list)
    klass: str = ""
    methods: list[str] = field(default_factory=list)
    dast_finding: str = ""
    dast_route: str = ""

    @property
    def cases_path(self) -> Path:
        return CASES_DIR / f"{self.id}.json"

    @property
    def expected_path(self) -> Path:
        return EXPECTED_DIR / f"{self.id}.json"

    @property
    def module_dir(self) -> Path:
        return ROOT / self.module.path


def load_registry() -> tuple[dict[str, Module], list[Finding]]:
    document = yaml.safe_load((EQUIVALENCE_DIR / "findings.yaml").read_text())
    if document.get("harness_version") != HARNESS_VERSION:
        raise HarnessError(
            f"findings.yaml targets harness_version {document.get('harness_version')}, "
            f"this harness is {HARNESS_VERSION}"
        )
    modules = {
        name: Module(name=name, **spec) for name, spec in document["modules"].items()
    }
    findings = []
    for spec in document["findings"]:
        module = modules[spec["module"]]
        findings.append(
            Finding(
                id=spec["id"],
                title=spec["title"],
                cwe=spec["cwe"],
                module=module,
                subject=spec["subject"],
                secure_pattern=" ".join(spec["secure_pattern"].split()),
                observes=spec.get("observes", []),
                klass=spec.get("class", ""),
                methods=spec.get("methods", []),
                dast_finding=spec.get("dast_finding", ""),
                dast_route=spec.get("dast_route", ""),
            )
        )
    return modules, findings


def select(findings: list[Finding], only: str | None) -> list[Finding]:
    if not only:
        return findings
    chosen = [f for f in findings if f.id == only]
    if not chosen:
        raise HarnessError(
            f"unknown finding {only}; registered: {', '.join(f.id for f in findings)}"
        )
    return chosen


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(finding: Finding) -> dict[str, Any]:
    """Hash every input that can change what the recorded cases observe."""
    return {
        "cases_sha256": sha256(finding.cases_path),
        "seed_sha256": sha256(ROOT / finding.module.seed),
        "emitter_sha256": sha256(ROOT / finding.module.emitter),
        # Only the finding's own class decides whether it has been refactored.
        "subject_sha256": {path: sha256(ROOT / path) for path in finding.subject},
        # Provenance for the files the cases travel through but that other
        # findings also own; never used to pick a grading stage.
        "observed_sha256": {path: sha256(ROOT / path) for path in finding.observes},
    }


def load_cases(finding: Finding) -> dict[str, Any]:
    spec = json.loads(finding.cases_path.read_text())
    seen: set[str] = set()
    for case in spec["cases"]:
        if case["policy"] not in POLICIES:
            raise HarnessError(f"{finding.id}/{case['id']}: unknown policy {case['policy']}")
        if case["id"] in seen:
            raise HarnessError(f"{finding.id}: duplicate case id {case['id']}")
        seen.add(case["id"])
        if case["policy"] == "attack":
            detector = case.get("detect", {}).get("kind")
            if detector not in DETECTORS:
                raise HarnessError(
                    f"{finding.id}/{case['id']}: unknown detector {detector!r}; "
                    f"available: {', '.join(sorted(DETECTORS))}"
                )
    return spec


# --- detectors -------------------------------------------------------------
# A closed set, evaluated by the harness (never by the emitter) so "exploited"
# is decided by the same code before and after the refactor.


def observation_text(observation: dict[str, Any]) -> str:
    return json.dumps(observation, sort_keys=True)


def detect_text_matches(observation: dict[str, Any], spec: dict[str, Any]) -> bool:
    return re.search(spec["pattern"], observation_text(observation)) is not None


def detect_rows_contain_owner(observation: dict[str, Any], spec: dict[str, Any]) -> bool:
    rows = observation.get("value")
    if not isinstance(rows, list):
        return False
    return any(
        isinstance(row, dict) and row.get("owner_id") == spec["owner_id"] for row in rows
    )


def detect_json_field_true(observation: dict[str, Any], spec: dict[str, Any]) -> bool:
    value = observation.get("value")
    return isinstance(value, dict) and value.get(spec["field"]) is True


DETECTORS = {
    "text_matches": detect_text_matches,
    "rows_contain_owner": detect_rows_contain_owner,
    "json_field_true": detect_json_field_true,
}


def exploited(case: dict[str, Any], observation: dict[str, Any]) -> bool:
    spec = case["detect"]
    return DETECTORS[spec["kind"]](observation, spec)


def after_violations(case: dict[str, Any], observation: dict[str, Any]) -> list[str]:
    """Check the optional shape constraints on a neutralised attack case."""
    problems = []
    for key, wanted in case.get("after", {}).items():
        if key == "outcome":
            if observation.get("outcome") != wanted:
                problems.append(f"expected outcome {wanted}, got {observation.get('outcome')}")
        elif key == "status":
            actual = (observation.get("value") or {}).get("status")
            if actual != wanted:
                problems.append(f"expected HTTP {wanted}, got {actual}")
        elif key == "error_type":
            if observation.get("error_type") != wanted:
                problems.append(
                    f"expected error type {wanted}, got {observation.get('error_type')}"
                )
        else:
            raise HarnessError(f"{case['id']}: unknown 'after' constraint {key!r}")
    return problems


# --- running the module ----------------------------------------------------


def render(command: str, **paths: Path) -> str:
    return command.format(**{key: str(value) for key, value in paths.items()})


def run_module(finding: Finding, command: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("JWT_SECRET", None)  # the fixture authenticates via the forwarded header
    # The harness itself runs under `uv run`, whose ephemeral environment would
    # otherwise capture the module's own toolchain (poetry honours VIRTUAL_ENV).
    for variable in ("VIRTUAL_ENV", "POETRY_ACTIVE", "PYTHONPATH", "PYTHONHOME"):
        env.pop(variable, None)
    return subprocess.run(
        shlex.split(command),
        cwd=finding.module_dir,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def emit(finding: Finding) -> dict[str, Any]:
    """Observe the finding's cases against the current working tree."""
    with tempfile.TemporaryDirectory(prefix="ow-equivalence-") as tmp:
        out = Path(tmp) / "observed.json"
        command = render(
            finding.module.emit_command,
            emitter=ROOT / finding.module.emitter,
            cases=finding.cases_path,
            seed=ROOT / finding.module.seed,
            out=out,
        )
        result = run_module(finding, command)
        if result.returncode != 0 or not out.exists():
            raise HarnessError(
                f"{finding.id}: the emitter failed (exit {result.returncode})\n"
                f"$ {command}\n{result.stdout[-4000:]}{result.stderr[-4000:]}"
            )
        return json.loads(out.read_text())


#: One suite run per module per invocation. Several findings share a module, and
#: the suite result is a property of the module, not of the finding being graded.
_SUITE_RUNS: dict[str, dict[str, Any]] = {}


def run_suite(finding: Finding) -> dict[str, Any]:
    """Run the module's own suite and return its per-test outcomes."""
    cached = _SUITE_RUNS.get(finding.module.name)
    if cached is not None:
        return cached
    suite = _execute_suite(finding)
    _SUITE_RUNS[finding.module.name] = suite
    return suite


def _execute_suite(finding: Finding) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ow-equivalence-junit-") as tmp:
        junit = Path(tmp) / "junit.xml"
        command = render(finding.module.test_command, junit=junit)
        result = run_module(finding, command)
        if not junit.exists():
            raise HarnessError(
                f"{finding.module.name}: the suite produced no JUnit report "
                f"(exit {result.returncode})\n$ {command}\n"
                f"{result.stdout[-4000:]}{result.stderr[-4000:]}"
            )
        return summarise_junit(junit, result.returncode, command)


def summarise_junit(junit: Path, returncode: int, command: str) -> dict[str, Any]:
    tree = ElementTree.parse(junit)
    passed: list[str] = []
    failed: list[str] = []
    for case in tree.iter("testcase"):
        node = f"{case.get('classname', '')}::{case.get('name', '')}"
        broke = any(child.tag in {"failure", "error"} for child in case)
        skipped = any(child.tag == "skipped" for child in case)
        if broke:
            failed.append(node)
        elif not skipped:
            passed.append(node)
    return {
        "command": command,
        "exit_code": returncode,
        "passed": sorted(passed),
        "failed": sorted(failed),
    }


# --- reports ---------------------------------------------------------------


def write_report(name: str, payload: dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"report: {path.relative_to(ROOT)}")
    return path


def worst(statuses: list[str]) -> str:
    if any(status in INCONCLUSIVE for status in statuses):
        return next(status for status in statuses if status in INCONCLUSIVE)
    if FAIL in statuses:
        return FAIL
    # A finding nothing could be judged for is not closed: an exploit run over a
    # finding with no attack cases must exit 3, never 0.
    return NO_VERDICT if NO_VERDICT in statuses else OK


def exit_code_for(status: str) -> int:
    if status == OK:
        return EXIT_OK
    if status == FAIL:
        return EXIT_FAIL
    if status == NO_VERDICT:
        return EXIT_NO_VERDICT
    return EXIT_INCONCLUSIVE


# --- recording ------------------------------------------------------------


def command_record(findings: list[Finding], reason: str, allow_rerecord: bool) -> int:
    payloads = []
    for finding in findings:
        spec = load_cases(finding)
        if finding.expected_path.exists() and not allow_rerecord:
            print(
                f"{finding.id}: {finding.expected_path.relative_to(ROOT)} already exists. "
                "Re-recording replaces the reference the refactor is graded against; "
                "pass ALLOW_RERECORD=1 with a REASON that says why.",
                file=sys.stderr,
            )
            return EXIT_INCONCLUSIVE

        observed = emit(finding)
        observations = {case["id"]: case for case in observed["cases"]}
        unreproduced = [
            case["id"]
            for case in spec["cases"]
            if case["policy"] == "attack"
            and not exploited(case, observations[case["id"]])
        ]
        if unreproduced:
            # A recording is only a valid before-state if the vulnerability is
            # actually present in it. This is also what makes the recording
            # impossible to refresh after a fix in order to force a pass.
            print(
                f"{finding.id}: refusing to record - the attack cases "
                f"{', '.join(unreproduced)} do not reproduce against this working tree. "
                "Record the before-state, not the fix.",
                file=sys.stderr,
            )
            return EXIT_INCONCLUSIVE

        suite = run_suite(finding)
        previous = None
        if finding.expected_path.exists():
            old = json.loads(finding.expected_path.read_text())
            previous = {
                "recorded_at": old.get("recorded_at"),
                "reason": old.get("reason"),
                "fingerprint": old.get("fingerprint"),
            }
        payload = {
            "finding": finding.id,
            "module": finding.module.name,
            "harness_version": HARNESS_VERSION,
            "recorded_at": datetime.now(UTC).isoformat(),
            "reason": reason,
            "rerecorded_from": previous,
            "fingerprint": fingerprint(finding),
            "interface": observed["interface"],
            "cases": observed["cases"],
            "suite": suite,
        }
        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        finding.expected_path.write_text(
            json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8"
        )
        print(
            f"{finding.id}: recorded {len(observed['cases'])} cases and "
            f"{len(suite['passed'])} passing tests -> "
            f"{finding.expected_path.relative_to(ROOT)}"
        )
        payloads.append(payload)
    write_report("record", {"reason": reason, "findings": [p["finding"] for p in payloads]})
    return EXIT_OK


# --- grading --------------------------------------------------------------


def load_recording(finding: Finding) -> tuple[dict[str, Any] | None, str]:
    if not finding.expected_path.exists():
        return None, UNRECORDED
    recording = json.loads(finding.expected_path.read_text())
    if recording.get("harness_version") != HARNESS_VERSION:
        return recording, STALE
    current = fingerprint(finding)
    for key in ("cases_sha256", "seed_sha256", "emitter_sha256"):
        if recording["fingerprint"].get(key) != current[key]:
            return recording, STALE
    return recording, OK


def subject_changed(finding: Finding, recording: dict[str, Any]) -> list[str]:
    """Return the finding's own subject files that differ from the recording.

    Deliberately blind to ``observes`` files: the route module is shared by every
    finding in the module, so a refactor of one finding must not make its
    neighbours look refactored and be graded against the post-fix contract.
    """
    current = fingerprint(finding)["subject_sha256"]
    recorded = recording["fingerprint"]["subject_sha256"]
    if set(current) != set(recorded):
        raise HarnessError(
            f"{finding.id}: the subject file list changed since the recording "
            "(findings.yaml no longer matches the evidence); re-record with a reason"
        )
    return [path for path, digest in current.items() if recorded[path] != digest]


def grade_finding(finding: Finding, stage: str) -> dict[str, Any]:
    spec = load_cases(finding)
    recording, state = load_recording(finding)
    if state != OK:
        detail = {
            UNRECORDED: f"no recording at {finding.expected_path.relative_to(ROOT)}",
            STALE: "the recording does not describe these cases/seed/emitter any more",
        }[state]
        return {"finding": finding.id, "stage": stage, "status": state, "detail": detail}

    changed = subject_changed(finding, recording)
    if stage == "baseline" and changed:
        return {
            "finding": finding.id,
            "stage": stage,
            "status": STALE,
            "detail": f"the subject changed since the recording ({', '.join(changed)}); "
            "grade --stage remediated instead",
        }
    if stage == "remediated" and not changed:
        return {
            "finding": finding.id,
            "stage": stage,
            "status": UNMEASURED,
            "detail": "no subject file changed, so there is nothing to grade as a "
            "refactor; the before-state is still in place",
        }

    observed = emit(finding)
    observations = {case["id"]: case for case in observed["cases"]}
    recorded = {case["id"]: case for case in recording["cases"]}
    results = []
    for case in spec["cases"]:
        results.append(grade_case(case, recorded.get(case["id"]), observations[case["id"]], stage))
    # Grading only the cases still registered would let a refactor delete the
    # coverage that fails it - drop a finding's attack cases and every remaining
    # case passes. A recorded case that is no longer registered is missing
    # coverage, which is inconclusive, never a pass.
    for case_id in recorded:
        if case_id not in {case["id"] for case in spec["cases"]}:
            results.append(
                {
                    "id": case_id,
                    "policy": "recorded",
                    "status": MISSING,
                    "detail": "recorded in the evidence but no longer registered in "
                    f"{finding.cases_path.relative_to(ROOT)}; restore the case or "
                    "re-record with a reason",
                }
            )

    interface = grade_interface(recording, observed)
    statuses = [row["status"] for row in results] + [interface["status"]]
    return {
        "finding": finding.id,
        "stage": stage,
        "status": worst(statuses),
        "subject_changed": changed,
        "interface": interface,
        "cases": results,
    }


def grade_case(
    case: dict[str, Any],
    recorded: dict[str, Any] | None,
    observed: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    row = {"id": case["id"], "policy": case["policy"]}
    if recorded is None:
        return row | {"status": UNRECORDED, "detail": "not present in the recording"}

    if case["policy"] == "contract":
        if recorded == observed:
            return row | {"status": OK, "detail": summarise(observed)}
        return row | {
            "status": FAIL,
            "detail": "behaviour changed",
            "recorded": recorded,
            "observed": observed,
        }

    was = exploited(case, recorded)
    now = exploited(case, observed)
    if not was:
        return row | {
            "status": STALE,
            "detail": "the recording does not reproduce this exploit, so there is no "
            "before-state to compare against",
        }
    if stage == "baseline":
        if now:
            return row | {"status": OK, "detail": "still exploitable (expected before-state)"}
        return row | {
            "status": FAIL,
            "detail": "the recorded exploit no longer reproduces, but the subject is "
            "unchanged - the fixture no longer measures what it claims to",
            "observed": observed,
        }
    if now:
        return row | {
            "status": FAIL,
            "detail": "still exploitable after the refactor",
            "observed": observed,
        }
    problems = after_violations(case, observed)
    if problems:
        return row | {
            "status": FAIL,
            "detail": "no longer exploitable, but the shape of the rejection is wrong: "
            + "; ".join(problems),
            "observed": observed,
        }
    return row | {"status": OK, "detail": "neutralised"}


def grade_interface(recording: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    recorded = recording.get("interface", {})
    current = observed.get("interface", {})
    if recorded == current:
        return {"status": OK, "members": len(recorded), "detail": "unchanged"}
    drift = {
        member: {"recorded": recorded.get(member), "observed": current.get(member)}
        for member in set(recorded) | set(current)
        if recorded.get(member) != current.get(member)
    }
    return {"status": FAIL, "detail": "the public interface changed", "drift": drift}


def summarise(observation: dict[str, Any]) -> str:
    if observation.get("outcome") == "error":
        return f"{observation.get('error_type')}: {observation.get('error_message', '')[:80]}"
    text = json.dumps(observation.get("value"), sort_keys=True)
    return text if len(text) <= 90 else f"{text[:87]}..."


def resolve_stage(finding: Finding, stage: str) -> str:
    if stage != "auto":
        return stage
    recording, state = load_recording(finding)
    if recording is None or state != OK:
        return "baseline"
    return "remediated" if subject_changed(finding, recording) else "baseline"


def command_grade(findings: list[Finding], stage: str) -> int:
    if stage not in STAGE_ARGUMENTS:
        raise HarnessError(f"unknown stage {stage!r}")
    results = [grade_finding(finding, resolve_stage(finding, stage)) for finding in findings]
    rows = []
    for result in results:
        for case in result.get("cases", []):
            rows.append([result["finding"], result["stage"], case["id"], case["policy"],
                         case["status"], case.get("detail", "")[:60]])
        if not result.get("cases"):
            rows.append([result["finding"], result["stage"], "-", "-", result["status"],
                         result.get("detail", "")])
    print(tabulate(rows, headers=["finding", "stage", "case", "policy", "status", "detail"]))
    for result in results:
        for case in result.get("cases", []):
            if case["status"] == FAIL and "recorded" in case:
                print(f"\n{result['finding']}/{case['id']}: behaviour changed")
                print(f"  recorded: {summarise(case['recorded'])}")
                print(f"  observed: {summarise(case['observed'])}")
        if result.get("interface", {}).get("status") == FAIL:
            print(f"\n{result['finding']}: {result['interface']['detail']}")
            for member, drift in result["interface"]["drift"].items():
                print(f"  {member}")
                print(f"    recorded: {drift['recorded']}")
                print(f"    observed: {drift['observed']}")
    status = worst([result["status"] for result in results])
    write_report(f"grade-{stage}", {"stage": stage, "status": status, "findings": results})
    if stage == "auto":
        for result in results:
            print(f"{result['finding']}: graded as {result['stage']} -> {result['status']}")
    print(f"\n{stage}: {status}")
    return exit_code_for(status)


# --- exploit-only run -----------------------------------------------------


def command_exploit(findings: list[Finding], refactored_only: bool = False) -> int:
    rows = []
    statuses = []
    payload = []
    for finding in findings:
        if refactored_only:
            # CI runs this on every branch, including the before-state, where the
            # attacks are meant to fire. Only a finding whose own subject moved is
            # claiming to be fixed, so only that one owes a closed verdict - and a
            # recording that cannot say whether the subject moved is inconclusive.
            recording, state = load_recording(finding)
            if state != OK:
                rows.append([finding.id, "-", state, "cannot tell the state from the recording"])
                statuses.append(state)
                continue
            if not subject_changed(finding, recording):
                rows.append([finding.id, "-", "skipped", "subject unchanged (before-state)"])
                continue
        spec = load_cases(finding)
        attacks = [case for case in spec["cases"] if case["policy"] == "attack"]
        if not attacks:
            rows.append([finding.id, "-", NO_VERDICT, "no attack cases registered"])
            statuses.append(NO_VERDICT)
            continue
        observed = emit(finding)
        observations = {case["id"]: case for case in observed["cases"]}
        for case in attacks:
            observation = observations[case["id"]]
            fires = exploited(case, observation)
            status = FAIL if fires else OK
            statuses.append(status)
            rows.append([finding.id, case["id"], status, summarise(observation)[:70]])
            payload.append({"finding": finding.id, "case": case["id"], "exploited": fires,
                            "observation": observation})
    print(tabulate(rows, headers=["finding", "case", "status", "observation"]))
    if statuses:
        status = worst(statuses)
    elif refactored_only:
        # Nothing claimed to be fixed, so there is nothing to hold to a closed
        # verdict; the before-state's own exploitability is eq-baseline's job.
        status = OK
    else:
        status = NO_VERDICT
    write_report("exploit", {"status": status, "cases": payload})
    if not statuses and refactored_only:
        print(f"\nexploit: {status} (no subject changed, nothing claims to be fixed)")
    elif status == FAIL:
        print(f"\nexploit: {status} (exploitable)")
    elif status == OK:
        print(f"\nexploit: {status} (closed)")
    else:
        print(f"\nexploit: {status} (no verdict, which is never a pass)")
    return exit_code_for(status)


# --- suite regression gate ------------------------------------------------


def command_tests(findings: list[Finding]) -> int:
    statuses = []
    payload = []
    for finding in findings:
        recording, state = load_recording(finding)
        if state != OK:
            # Stale evidence is not a weaker pass than missing evidence: a pass
            # list recorded against a different fixture says nothing about this
            # working tree.
            print(f"{finding.id}: {state} - record the before-state first", file=sys.stderr)
            statuses.append(state)
            continue
        suite = run_suite(finding)
        recorded_pass = set(recording["suite"]["passed"])
        now_pass = set(suite["passed"])
        now_fail = set(suite["failed"])
        recorded_fail = set(recording["suite"]["failed"])
        regressed = sorted(recorded_pass & now_fail)
        vanished = sorted(recorded_pass - now_pass - now_fail)
        # A test that was already red at record time is a documented pre-existing
        # failure, not a regression; anything else that fails is new.
        new_failures = sorted(now_fail - recorded_pass - recorded_fail)
        fixed = sorted(recorded_fail & now_pass)
        if regressed or new_failures:
            status = FAIL
        elif vanished:
            # A test that passed at record time and no longer exists cannot be
            # used as evidence, and deleting it must not be a way to go green.
            status = UNMEASURED
        else:
            status = OK
        statuses.append(status)
        payload.append({
            "finding": finding.id,
            "module": finding.module.name,
            "status": status,
            "regressed": regressed,
            "new_failures": new_failures,
            "vanished": vanished,
            "newly_passing": fixed,
            "counts": {"passed": len(now_pass), "failed": len(now_fail),
                       "recorded_passed": len(recorded_pass)},
        })
        print(
            f"{finding.id} [{finding.module.name}]: {status} - "
            f"{len(now_pass)} passing, {len(now_fail)} failing "
            f"(recorded {len(recorded_pass)} passing)"
        )
        for label, nodes in (("regressed", regressed), ("new failure", new_failures),
                             ("no longer present", vanished), ("newly passing", fixed)):
            for node in nodes:
                print(f"  {label}: {node}")
    status = worst(statuses) if statuses else UNMEASURED
    write_report("tests", {"status": status, "modules": payload})
    print(f"\ntests: {status}")
    return exit_code_for(status)


# --- inventory ------------------------------------------------------------


def command_list(findings: list[Finding]) -> int:
    rows = []
    for finding in findings:
        spec = load_cases(finding)
        contract = sum(1 for case in spec["cases"] if case["policy"] == "contract")
        attack = sum(1 for case in spec["cases"] if case["policy"] == "attack")
        recording, state = load_recording(finding)
        if recording is None or state != OK:
            evidence = state  # missing, stale or unreadable evidence is never a pass
        else:
            changed = subject_changed(finding, recording)
            evidence = "changed subject" if changed else "matches before-state"
        rows.append([
            finding.id,
            finding.cwe,
            finding.module.name,
            f"{finding.klass}.{'/'.join(finding.methods)}",
            f"{contract}/{attack}",
            evidence,
        ])
    print(tabulate(
        rows,
        headers=["finding", "cwe", "module", "subject", "contract/attack", "evidence"],
    ))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "record", "grade", "exploit", "tests"])
    parser.add_argument("--finding")
    parser.add_argument("--stage", default="remediated")
    parser.add_argument("--reason", default="")
    parser.add_argument("--allow-rerecord", action="store_true")
    parser.add_argument("--refactored-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        _, findings = load_registry()
        chosen = select(findings, args.finding)
        if args.command == "list":
            return command_list(chosen)
        if args.command == "record":
            if not args.reason.strip():
                raise HarnessError("record requires a reason")
            return command_record(chosen, args.reason.strip(), args.allow_rerecord)
        if args.command == "grade":
            return command_grade(chosen, args.stage)
        if args.command == "exploit":
            return command_exploit(chosen, refactored_only=args.refactored_only)
        return command_tests(chosen)
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INCONCLUSIVE


if __name__ == "__main__":
    sys.exit(main())
