"""Session-level fail-closed enforcement for hidden pytest outcomes.

Any collected test that ends as a skip (including ``pytest.skip``,
``pytest.importorskip``, ``skipif``, and collection-time module skips),
``xfail``, or ``xpass`` escalates an otherwise-successful session to
``TESTS_FAILED``. Sessions whose accounting cannot be observed also fail
closed instead of reporting GREEN.

Implementation basis: fail-closed outcome policy of Issue #1732
(ContextualWisdomLab, 2026).

References:
    ContextualWisdomLab. (2026). Test governance: fail closed on skipped
    and expected-failure outcomes (Issue #1732).
    https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

_ALLOWLIST_NAME = "allowed_non_execution.txt"
_CONFIG_MARKER = "_fail_closed_accounting_active"

# kind, nodeid pairs observed by the reporters below.
_ACCOUNTING: dict[str, object] = {"initialized": False, "events": []}


def _events() -> list[tuple[str, str]]:
    """Return the mutable outcome-accounting event log for this session.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    return _ACCOUNTING["events"]  # type: ignore[return-value]


def _allowlist_path() -> Path:
    """Return the reviewed non-execution allowlist co-located with this plugin.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    return Path(__file__).with_name(_ALLOWLIST_NAME)


def _load_allowlist_patterns() -> list[str]:
    """Load allowlist patterns, ignoring blanks, comments, and rationale text.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    path = _allowlist_path()
    if not path.exists():
        return []
    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            patterns.append(entry)
    return patterns


def _is_allowlisted(nodeid: str, patterns: list[str]) -> bool:
    """Check a node ID against exact entries or trailing-glob family patterns.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    for pattern in patterns:
        if any(char in pattern for char in ("*", "?", "[")):
            if fnmatch.fnmatchcase(nodeid, pattern):
                return True
        elif nodeid == pattern:
            return True
    return False


def pytest_configure(config: pytest.Config) -> None:
    """Mark outcome accounting active so the finish verdict can trust it.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    setattr(config, _CONFIG_MARKER, True)
    _ACCOUNTING["initialized"] = True


def pytest_collectreport(report: pytest.CollectReport) -> None:
    """Record collection-time skips (module skip, importorskip, collection skip).

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    if report.skipped:
        _events().append(("collection-skip", report.nodeid))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record setup/call/teardown skips and xfail/xpass outcomes.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    wasxfail = getattr(report, "wasxfail", None)
    if wasxfail is not None:
        kind = "xpass" if report.passed else "xfail"
        _events().append((kind, report.nodeid))
    elif report.skipped:
        _events().append(("skip", report.nodeid))


def _unexpected_events() -> list[tuple[str, str]]:
    """Return observed non-execution events absent from the reviewed allowlist.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    patterns = _load_allowlist_patterns()
    return [
        (kind, nodeid)
        for kind, nodeid in _events()
        if not _is_allowlisted(nodeid, patterns)
    ]


def _report_escalation(session: pytest.Session, lines: list[str]) -> None:
    """Emit fail-closed escalation evidence through the terminal reporter.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:  # pragma: no cover - reporter always present in practice
        return
    reporter.write_sep("=", "FAIL-CLOSED: non-executed or expected-failure outcomes")
    for line in lines:
        reporter.write_line(line)


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_sessionfinish(
    session: pytest.Session, exitstatus: int | pytest.ExitCode
):
    """Escalate unexpected non-execution to TESTS_FAILED without downgrading.

    Runs inner hooks first via the wrapper yield, then escalates an
    otherwise-OK session when unexpected skip/xfail/xpass events were
    observed, or when accounting was never initialized (fail closed). A
    stronger non-success status is never downgraded.

    Implementation basis: fail-closed outcome policy of Issue #1732
    (ContextualWisdomLab, 2026).

    References:
        ContextualWisdomLab. (2026). Test governance: fail closed on skipped
        and expected-failure outcomes (Issue #1732).
        https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1732
    """
    yield
    if exitstatus != pytest.ExitCode.OK:
        return exitstatus
    if not getattr(session.config, _CONFIG_MARKER, False) or not _ACCOUNTING["initialized"]:
        _report_escalation(
            session,
            ["outcome accounting was never initialized; failing closed"],
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        return session.exitstatus
    unexpected = _unexpected_events()
    if unexpected:
        _report_escalation(
            session,
            [f"{kind}: {nodeid}" for kind, nodeid in unexpected],
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        return session.exitstatus
    return exitstatus
