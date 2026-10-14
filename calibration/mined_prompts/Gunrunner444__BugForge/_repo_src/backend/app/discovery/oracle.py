"""Execution oracles. A nonzero exit or HTTP 500 is not, by itself, a bug."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OracleKind(StrEnum):
    CRASH = "crash"
    EXIT_STATUS = "exit_status"
    ASSERTION = "assertion"
    PROPERTY = "property"
    INVARIANT = "invariant"
    SANITIZER = "sanitizer"
    EXPECTED_OUTPUT = "expected_output"
    STATE_DELTA = "state_delta"
    DIFFERENTIAL = "differential"
    METAMORPHIC = "metamorphic"
    EXPECTED_REVERT = "expected_revert"
    UNEXPECTED_REVERT = "unexpected_revert"
    PANIC = "panic"
    OUT_OF_GAS = "out_of_gas"
    TOOL_FAILURE = "tool_failure"
    FUZZER_CRASH = "fuzzer_crash"


@dataclass(frozen=True)
class OracleVerdict:
    kind: OracleKind
    meaningful: bool
    explanation: str


def evaluate_oracle(
    kind: OracleKind,
    *,
    expected: str = "",
    actual: str = "",
    exit_code: int | None = None,
    sanitizer: str = "",
    crashed: bool = False,
) -> OracleVerdict:
    """Decide whether an execution produced evidence worth keeping.

    The explanation is the reason. Callers must not treat a meaningful oracle
    as a verified vulnerability.
    """
    if kind is OracleKind.CRASH:
        if crashed or (exit_code not in {None, 0} and "sanitizer" in actual.lower()):
            return OracleVerdict(kind, True, "Process crashed or a sanitizer aborted execution.")
        return OracleVerdict(
            kind, False, "No crash was observed. A nonzero exit alone is not a crash."
        )
    if kind is OracleKind.EXIT_STATUS:
        if exit_code is None:
            return OracleVerdict(kind, False, "No exit status was captured.")
        if exit_code != 0 and expected and expected in actual:
            return OracleVerdict(
                kind,
                True,
                f"Exit {exit_code} matched the expected failure marker {expected!r}.",
            )
        return OracleVerdict(
            kind,
            False,
            f"Exit {exit_code} is not meaningful without an expected failure marker.",
        )
    if kind is OracleKind.SANITIZER:
        if sanitizer.strip():
            return OracleVerdict(kind, True, f"Sanitizer reported {sanitizer.strip()}.")
        return OracleVerdict(kind, False, "No sanitizer diagnostic was present.")
    if kind in {OracleKind.ASSERTION, OracleKind.PROPERTY, OracleKind.INVARIANT}:
        if actual.strip() and (not expected or expected in actual):
            return OracleVerdict(
                kind,
                True,
                f"{kind.value} failed: {actual.strip()[:240]}",
            )
        return OracleVerdict(kind, False, f"No {kind.value} failure was identified.")
    if kind is OracleKind.EXPECTED_OUTPUT:
        if expected and expected in actual:
            return OracleVerdict(kind, True, "Actual output contained the expected marker.")
        return OracleVerdict(kind, False, "Expected output marker was absent.")
    if kind is OracleKind.STATE_DELTA:
        if expected and actual and expected != actual:
            return OracleVerdict(kind, True, "Observed state differed from the expected state.")
        return OracleVerdict(kind, False, "State matched the expectation or was not comparable.")
    if kind in {OracleKind.DIFFERENTIAL, OracleKind.METAMORPHIC}:
        if expected and actual and expected != actual:
            return OracleVerdict(
                kind,
                True,
                "Two executions that should agree produced different outputs.",
            )
        return OracleVerdict(kind, False, "No differential or metamorphic mismatch was shown.")
    if kind is OracleKind.EXPECTED_REVERT:
        return OracleVerdict(
            kind,
            False,
            "An expected revert is correct behavior, not a vulnerability.",
        )
    if kind is OracleKind.UNEXPECTED_REVERT:
        if actual.strip():
            return OracleVerdict(
                kind, True, "Execution reverted although the call was expected to succeed."
            )
        return OracleVerdict(kind, False, "No unexpected revert was observed.")
    if kind is OracleKind.PANIC:
        if actual.strip():
            return OracleVerdict(kind, True, f"Solidity panic: {actual.strip()[:180]}")
        return OracleVerdict(kind, False, "No panic code was observed.")
    if kind is OracleKind.OUT_OF_GAS:
        if actual.strip():
            return OracleVerdict(
                kind,
                True,
                "Out of gas was observed. That is evidence of a gas boundary, not a confirmed bug.",
            )
        return OracleVerdict(kind, False, "Execution did not run out of gas.")
    if kind is OracleKind.TOOL_FAILURE:
        return OracleVerdict(
            kind,
            False,
            "The tool or compiler failed. That is not a contract vulnerability.",
        )
    if kind is OracleKind.FUZZER_CRASH:
        if crashed or actual.strip():
            return OracleVerdict(kind, True, "The fuzzer reported a crashing input.")
        return OracleVerdict(kind, False, "The fuzzer did not report a crash.")
    return OracleVerdict(kind, False, "Oracle kind was not recognized.")
