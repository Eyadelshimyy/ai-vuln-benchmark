"""Contract tests for the hourly PR-queue governance workflow."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


_WORKFLOW = (
    Path(__file__).parents[1]
    / ".github"
    / "workflows"
    / "hourly-pr-governance.yml"
)


def _workflow_text() -> str:
    """Return the scheduled workflow as UTF-8 text."""
    return _WORKFLOW.read_text(encoding="utf-8")


def _build_step_script() -> str:
    """Return the live governance shell block with YAML indentation removed."""
    text = _workflow_text()
    step = text.index("      - name: Build live PR queue governance evidence")
    run = text.index("        run: |\n", step) + len("        run: |\n")
    end = text.index("      - name:", run)
    return textwrap.dedent(text[run:end])


def _retry_classifier_script() -> str:
    """Extract the Python classifier used by the retry shell function."""
    script = _build_step_script()
    function = script.index("is_retryable_github_failure() {")
    start = script.index("python - <<'PY'\n", function) + len("python - <<'PY'\n")
    end = script.index("\nPY\n", start)
    return script[start:end]


def _classify_retryability(manifest: dict[str, Any]) -> str:
    """Run the embedded classifier against one temporary governance manifest."""
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        output = root / "hourly-pr-queue-governance"
        output.mkdir()
        (output / "pr_queue_governance_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-c", _retry_classifier_script()],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _manifest(
    *failed_check_names: str,
    stderr: str = "HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
) -> dict[str, Any]:
    """Return a minimal failed manifest with one caller-selected GitHub error."""
    return {
        "status": "failed",
        "failed_checks": [
            {"name": name, "ok": False} for name in failed_check_names
        ],
        "github": {
            "errors": [
                {
                    "command": ["pr", "list"],
                    "stderr": stderr,
                    "returncode": 1,
                }
            ]
        },
    }


def test_hourly_governance_workflow_exists_and_runs_every_hour():
    """The repository records queue-governance evidence on an hourly cadence."""
    text = _workflow_text()
    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert 'cron: "0 * * * *"' in text


def test_hourly_governance_workflow_is_read_only_and_bounded():
    """The scheduled loop cannot mutate pull requests or repository contents."""
    text = _workflow_text()
    assert "contents: read" in text
    assert "pull-requests: read" in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "cancel-in-progress: true" in text
    assert "timeout-minutes: 10" in text


def test_hourly_governance_workflow_builds_and_publishes_evidence():
    """Each run invokes the governed builder and uploads both audit artifacts."""
    text = _workflow_text()
    assert "python scripts/build_pr_queue_governance.py" in text
    assert "hourly-pr-queue-governance" in text
    assert "pr_queue_governance_manifest.json" in text
    assert "pr_queue_governance_report.html" in text
    assert "actions/upload-artifact" in text
    assert "if: always()" in text


def test_hourly_governance_workflow_tracks_implementation_and_contract_tests():
    """Changes to duplicate-claim governance trigger the evidence workflow."""
    text = _workflow_text()
    assert "scripts/build_pr_queue_governance.py" in text
    assert "tests/test_pr_queue_governance.py" in text
    assert "tests/test_hourly_pr_governance_workflow.py" in text


def test_governance_workflow_fails_closed_when_no_tests_are_discovered():
    """A renamed or removed test suite cannot make self-verification silently pass."""
    text = _workflow_text()
    assert "test_names = [" in text
    assert "if not test_names:" in text
    guard = 'raise RuntimeError("no hourly governance contract tests were discovered")'
    assert guard in text
    assert text.index("if not test_names:") < text.index(guard)
    assert text.index(guard) < text.index("for name in test_names:")


def test_hourly_governance_workflow_retries_only_safe_server_statuses():
    """Only exact gateway and service-unavailable HTTP statuses receive retries."""
    classifier = _retry_classifier_script()
    text = _workflow_text()
    assert "max_attempts=3" in text
    assert "import re" in classifier
    assert r"\bHTTP (?:502|503|504)\b" in classifier
    assert '"HTTP 429"' not in classifier
    assert '"HTTP 500"' not in classifier
    assert '"timeout"' not in classifier
    assert '"connection reset"' not in classifier
    assert 'delay=$((10 * (2 ** (attempt - 1))))' in text
    assert 'attempt=$((attempt + 1))' in text


def test_hourly_governance_workflow_does_not_retry_governance_failures():
    """Only approved GitHub statuses retry; real governance failures stay failed."""
    text = _workflow_text()
    assert 'manifest.get("github", {})' in text
    assert 'manifest.get("status") == "failed"' in text
    assert 'manifest.get("failed_checks")' in text
    assert 'check.get("ok") is False' in text
    assert 'type(error.get("returncode")) is int' in text
    assert 'error.get("returncode") != 0' in text
    assert '"github:snapshot"' in text
    assert '"github:base_sha"' in text
    assert "issubset(retryable_failed_check_names)" in text
    guard = 'if [[ "$retryable" != "true" || "$attempt" -ge "$max_attempts" ]]; then'
    assert guard in text
    assert text.index(guard) < text.index("exit 1")
    assert "continue-on-error: true" not in text


def test_hourly_governance_workflow_classifies_safe_server_statuses_as_retryable():
    """Each explicitly approved GitHub server status is eligible for retry."""
    messages = (
        "HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
        "HTTP 503: Service Unavailable (https://api.github.com/graphql)",
        "HTTP 504: Gateway Timeout (https://api.github.com/graphql)",
        "prefix http 502 suffix",
    )
    for stderr in messages:
        manifest = _manifest("github:snapshot", stderr=stderr)
        assert _classify_retryability(manifest) == "true"


def test_hourly_governance_workflow_classifies_repo_502_as_retryable():
    """A repository 502 may fail both snapshot and base-SHA evidence."""
    manifest = _manifest("github:snapshot", "github:base_sha")
    assert _classify_retryability(manifest) == "true"


def test_hourly_governance_workflow_rejects_rate_limit_without_headers():
    """A 429 cannot use a short retry without Retry-After or reset evidence."""
    manifest = _manifest(
        "github:snapshot",
        stderr="HTTP 429: API rate limit exceeded",
    )
    assert _classify_retryability(manifest) == "false"


def test_hourly_governance_workflow_rejects_status_prefix_collisions():
    """Longer numeric tokens cannot masquerade as approved HTTP statuses."""
    for status in (4290, 5020, 5030, 5040):
        manifest = _manifest(
            "github:snapshot",
            stderr=f"HTTP {status}: unexpected upstream error",
        )
        assert _classify_retryability(manifest) == "false"


def test_hourly_governance_workflow_rejects_unapproved_transient_markers():
    """HTTP 500 and generic network messages do not authorize retries."""
    messages = (
        "HTTP 500: Internal Server Error",
        "request timed out",
        "connection reset by peer",
        "temporary failure resolving api.github.com",
        "temporarily unavailable",
    )
    for stderr in messages:
        manifest = _manifest("github:snapshot", stderr=stderr)
        assert _classify_retryability(manifest) == "false"


def test_hourly_governance_workflow_rejects_mixed_governance_and_502_failures():
    """A genuine queue failure is never retried alongside an API outage."""
    manifest = _manifest("github:snapshot", "queue:duplicate_issue_claims")
    assert _classify_retryability(manifest) == "false"


def test_hourly_governance_workflow_rejects_malformed_failed_check_evidence():
    """Malformed failed-check evidence remains fail-closed."""
    manifest = _manifest("github:snapshot")
    manifest["failed_checks"] = "github:snapshot"
    assert _classify_retryability(manifest) == "false"


def test_hourly_governance_workflow_rejects_contradictory_passing_check_evidence():
    """A failed-check record marked successful cannot authorize a retry."""
    manifest = _manifest("github:snapshot")
    manifest["failed_checks"][0]["ok"] = True
    assert _classify_retryability(manifest) == "false"


def test_hourly_governance_workflow_rejects_contradictory_success_status():
    """A manifest marked successful cannot authorize retry after command failure."""
    manifest = _manifest("github:snapshot")
    manifest["status"] = "ok"
    assert _classify_retryability(manifest) == "false"


def test_hourly_governance_workflow_rejects_successful_error_command_evidence():
    """An error record with a zero return code is contradictory and non-retryable."""
    manifest = _manifest("github:snapshot")
    manifest["github"]["errors"][0]["returncode"] = 0
    assert _classify_retryability(manifest) == "false"


def test_hourly_governance_workflow_rejects_boolean_error_return_code():
    """A Boolean is not a valid process exit status despite subclassing int."""
    manifest = _manifest("github:snapshot")
    manifest["github"]["errors"][0]["returncode"] = True
    assert _classify_retryability(manifest) == "false"


def test_hourly_governance_workflow_retry_shell_is_syntactically_valid():
    """The bounded retry block remains valid Bash after YAML de-indentation."""
    completed = subprocess.run(
        ["bash", "-n"],
        input=_build_step_script(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def test_hourly_governance_workflow_does_not_persist_checkout_credentials():
    """Read-only governance steps cannot inherit a persisted checkout token."""
    text = _workflow_text()
    checkout = text.index("      - name: Check out reviewed source")
    setup = text.index("      - name: Set up Python", checkout)
    checkout_step = text[checkout:setup]
    assert "persist-credentials: false" in checkout_step
