import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


def _load_governance():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_pr_queue_governance.py"
    spec = importlib.util.spec_from_file_location("build_pr_queue_governance", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Build-level fixtures patch ``_source_commit`` with this canonical identity so
# queue-classification tests do not depend on a real Git checkout; boundary
# behavior of the real helper is covered by test_pr_queue_governance_git_timeout.py.
_CANONICAL_TEST_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _base_pr(
    number: int,
    *,
    body: str = "",
    head: str | None = None,
    state: str = "OPEN",
    files: list[str] | None = None,
) -> dict:
    return {
        "number": number,
        "title": f"PR {number}",
        "body": body,
        "headRefName": head or f"feat/pr-{number}",
        "headRefOid": f"{number:040x}",
        "baseRefName": "main",
        "isDraft": False,
        "mergeStateStatus": "UNKNOWN",
        "reviewDecision": "",
        "state": state,
        "updatedAt": "2026-07-02T00:00:00Z",
        "closedAt": "2026-07-02T03:00:00Z" if state != "OPEN" else None,
        "mergedAt": "2026-07-02T02:00:00Z" if state == "MERGED" else None,
        "url": f"https://github.com/ContextualWisdomLab/fast-mlsirm/pull/{number}",
        "labels": [],
        "files": [{"path": path} for path in (files or [])],
    }


def _write_snapshot(path: Path) -> Path:
    snapshot = {
        "mode": "offline",
        "repo": "ContextualWisdomLab/fast-mlsirm",
        "default_branch": "main",
        "base_sha": "a" * 40,
        "open_prs": [
            {
                **_base_pr(60),
                "title": "Palette: CLI simulation command error handling",
                "headRefName": "feat/cli-dx-improvement",
                "reviewDecision": "CHANGES_REQUESTED",
                "updatedAt": "2026-06-20T00:00:00Z",
            },
            {
                **_base_pr(59),
                "title": "Sentinel: add CSP to HTML reports",
                "headRefName": "sentinel-add-csp-to-html-reports",
                "reviewDecision": "CHANGES_REQUESTED",
                "updatedAt": "2026-06-19T00:00:00Z",
            },
            {
                **_base_pr(51),
                "title": "Add GPU support for estimation models",
                "headRefName": "copilot/add-gpu-support-for-estimation-models",
                "mergeStateStatus": "BLOCKED",
            },
            {
                **_base_pr(73),
                "title": "Add procurement due diligence evidence gate",
                "headRefName": "codex/20b-procurement-due-diligence",
                "mergeStateStatus": "QUEUED",
                "reviewDecision": "REVIEW_REQUIRED",
                "updatedAt": "2026-07-02T22:09:54Z",
            },
        ],
    }
    snapshot["pr_history"] = list(snapshot["open_prs"])
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path


def _args(root: Path, snapshot: Path | None, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(root),
        out=str(out),
        repo="ContextualWisdomLab/fast-mlsirm",
        contract_value_krw=2_000_000_000,
        offline_snapshot=str(snapshot) if snapshot else None,
        offline_github=snapshot is None,
        max_stale_days=7,
        generated_at="2026-07-03T00:00:00+00:00",
    )


def _write_custom_snapshot(
    path: Path,
    *,
    open_prs: list[dict],
    history: list[dict] | None = None,
) -> Path:
    snapshot = {
        "mode": "offline",
        "repo": "ContextualWisdomLab/fast-mlsirm",
        "default_branch": "main",
        "base_sha": "b" * 40,
        "open_prs": open_prs,
        "pr_history": history if history is not None else open_prs,
        "errors": [],
    }
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path


def test_pr_queue_governance_creates_manifest_and_report(tmp_path, monkeypatch):
    module = _load_governance()
    monkeypatch.setattr(module, "_source_commit", lambda root: _CANONICAL_TEST_COMMIT)
    snapshot = _write_snapshot(tmp_path / "snapshot.json")

    manifest = module.build_pr_queue_governance(
        _args(tmp_path, snapshot, tmp_path / "pr-queue-governance")
    )

    assert manifest["status"] == "ok"
    assert manifest["open_pr_count"] == 4
    assert manifest["risk_counts"]["changes_requested"] == 2
    assert manifest["risk_counts"]["stale"] == 2
    assert manifest["risk_counts"]["release_scope_conflict"] == 1
    assert manifest["risk_counts"]["review_or_check_delay"] == 1
    assert manifest["failed_checks"] == []
    assert manifest["checks"][0]["category"] == "github"
    out = tmp_path / "pr-queue-governance"
    manifest_path = out / "pr_queue_governance_manifest.json"
    html_path = out / "pr_queue_governance_report.html"
    assert manifest_path.exists()
    assert html_path.exists()
    assert manifest["html_report_sha256"] == _sha256(html_path)
    html = html_path.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in html
    assert "PR queue governance table" in html
    assert "Duplicate issue claim table" in html
    assert "Issue claim audit history table" in html
    assert "a" * 40 in html


def test_classify_pr_marks_duplicate_and_release_scope_risks():
    module = _load_governance()

    classified = module.classify_pr(
        {
            **_base_pr(51),
            "title": "Add GPU support for estimation models",
            "headRefName": "copilot/add-gpu-support-for-estimation-models",
            "mergeStateStatus": "BLOCKED",
            "updatedAt": "not-a-date",
            "body": "Fixes #394\nCanonical-For: #394",
            "labels": [{"name": "research"}, "security"],
            "files": [{"path": "src/a.py"}, "src/b.py", None],
        },
        now=module._parse_datetime("2026-07-03T00:00:00+00:00"),
        max_stale_days=7,
    )

    assert classified["release_scope_conflict"] is True
    assert "model_or_backend_scope" in classified["risk_reasons"]
    assert classified["age_days"] is None
    assert classified["closing_issue_references"] == [394]
    assert classified["canonical_issue_references"] == [394]
    assert classified["label_names"] == ["research", "security"]
    assert classified["changed_files"] == ["src/a.py", "src/b.py"]


def test_pr_queue_governance_fails_without_snapshot_when_offline(tmp_path, monkeypatch):
    module = _load_governance()
    monkeypatch.setattr(module, "_source_commit", lambda root: _CANONICAL_TEST_COMMIT)

    manifest = module.build_pr_queue_governance(
        _args(tmp_path, None, tmp_path / "pr-queue-governance")
    )

    assert manifest["status"] == "failed"
    failed = {check["name"] for check in manifest["failed_checks"]}
    assert "github:snapshot" in failed


def test_duplicate_issue_claim_without_canonical_designation_fails_closed(
    tmp_path, monkeypatch
):
    module = _load_governance()
    monkeypatch.setattr(module, "_source_commit", lambda root: _CANONICAL_TEST_COMMIT)
    prs = [
        _base_pr(396, body="Closes #394", files=["src/a.py", "tests/a.py"]),
        _base_pr(399, body="Fixes: #394", files=["src/b.py", "tests/b.py"]),
    ]
    snapshot = _write_custom_snapshot(tmp_path / "snapshot.json", open_prs=prs)

    manifest = module.build_pr_queue_governance(
        _args(tmp_path, snapshot, tmp_path / "out")
    )

    assert manifest["status"] == "failed"
    claim = manifest["duplicate_issue_claims"]["394"]
    assert claim["status"] == "conflict"
    assert claim["canonical_pr"] is None
    assert [record["pr_number"] for record in claim["claimant_prs"]] == [396, 399]
    assert manifest["unresolved_duplicate_issue_claim_count"] == 1
    failed = {check["name"] for check in manifest["failed_checks"]}
    assert "queue:duplicate_issue_claims" in failed


def test_exactly_one_canonical_designation_resolves_duplicate_claim(
    tmp_path, monkeypatch
):
    module = _load_governance()
    monkeypatch.setattr(module, "_source_commit", lambda root: _CANONICAL_TEST_COMMIT)
    prs = [
        _base_pr(
            396,
            body="Closes #394\n\nCanonical-For: #394",
            files=["src/a.py", "tests/a.py"],
        ),
        _base_pr(399, body="Resolves #394", files=["src/b.py", "tests/b.py"]),
    ]
    snapshot = _write_custom_snapshot(tmp_path / "snapshot.json", open_prs=prs)

    manifest = module.build_pr_queue_governance(
        _args(tmp_path, snapshot, tmp_path / "out")
    )

    claim = manifest["duplicate_issue_claims"]["394"]
    assert manifest["status"] == "ok"
    assert claim["status"] == "resolved"
    assert claim["canonical_pr"] == 396


def test_multiple_canonical_designations_remain_conflicted(tmp_path, monkeypatch):
    module = _load_governance()
    monkeypatch.setattr(module, "_source_commit", lambda root: _CANONICAL_TEST_COMMIT)
    prs = [
        _base_pr(396, body="Closes #394\nCanonical-For: #394"),
        _base_pr(399, body="Closes #394\nCanonical-For: #394"),
    ]
    snapshot = _write_custom_snapshot(tmp_path / "snapshot.json", open_prs=prs)

    manifest = module.build_pr_queue_governance(
        _args(tmp_path, snapshot, tmp_path / "out")
    )

    assert manifest["status"] == "failed"
    assert manifest["duplicate_issue_claims"]["394"]["canonical_pr"] is None


def test_closed_claimant_leaves_active_queue_and_remains_in_history(
    tmp_path, monkeypatch
):
    module = _load_governance()
    monkeypatch.setattr(module, "_source_commit", lambda root: _CANONICAL_TEST_COMMIT)
    active = _base_pr(396, body="Closes #394")
    closed = _base_pr(399, body="Closes #394", state="CLOSED")
    snapshot = _write_custom_snapshot(
        tmp_path / "snapshot.json",
        open_prs=[active],
        history=[active, closed],
    )

    manifest = module.build_pr_queue_governance(
        _args(tmp_path, snapshot, tmp_path / "out")
    )

    assert manifest["status"] == "ok"
    assert manifest["duplicate_issue_claims"] == {}
    assert [record["pr_number"] for record in manifest["issue_claim_history"]] == [396, 399]
    assert manifest["issue_claim_history"][1]["state"] == "CLOSED"


def test_identical_changed_file_sets_emit_nonblocking_warning(tmp_path, monkeypatch):
    module = _load_governance()
    monkeypatch.setattr(module, "_source_commit", lambda root: _CANONICAL_TEST_COMMIT)
    files = ["src/shared.py", "tests/test_shared.py"]
    prs = [
        _base_pr(1, files=files),
        _base_pr(2, files=list(reversed(files))),
    ]
    snapshot = _write_custom_snapshot(tmp_path / "snapshot.json", open_prs=prs)

    manifest = module.build_pr_queue_governance(
        _args(tmp_path, snapshot, tmp_path / "out")
    )

    assert manifest["status"] == "ok"
    assert manifest["changed_file_overlap_warnings"] == [
        {
            "pr_numbers": [1, 2],
            "identical": True,
            "jaccard": 1.0,
            "intersection": sorted(files),
            "union_size": 2,
        }
    ]


def test_high_overlap_sets_warn_but_one_shared_central_file_does_not(tmp_path):
    module = _load_governance()
    prs = [
        _base_pr(1, files=["src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/e.py"]),
        _base_pr(2, files=["src/a.py", "src/b.py", "src/c.py", "src/d.py"]),
        _base_pr(3, files=["src/a.py", "src/unique.py"]),
    ]
    classified = [
        module.classify_pr(
            pr,
            now=module._parse_datetime("2026-07-03T00:00:00+00:00"),
            max_stale_days=7,
        )
        for pr in prs
    ]

    warnings = module._changed_file_overlap_warnings(classified)

    assert len(warnings) == 1
    assert warnings[0]["pr_numbers"] == [1, 2]
    assert warnings[0]["jaccard"] == 0.8


def test_duplicate_head_branch_is_a_secondary_warning():
    module = _load_governance()
    prs = [
        module.classify_pr(
            _base_pr(1, head="feat/shared"),
            now=module._parse_datetime("2026-07-03T00:00:00+00:00"),
            max_stale_days=7,
        ),
        module.classify_pr(
            _base_pr(2, head="feat/shared"),
            now=module._parse_datetime("2026-07-03T00:00:00+00:00"),
            max_stale_days=7,
        ),
        module.classify_pr(
            _base_pr(3, head=""),
            now=module._parse_datetime("2026-07-03T00:00:00+00:00"),
            max_stale_days=7,
        ),
    ]

    assert module._duplicate_head_warnings(prs) == [
        {"head_ref": "feat/shared", "pr_numbers": [1, 2]}
    ]


def test_issue_reference_parsing_is_case_insensitive_deduplicated_and_bounded():
    module = _load_governance()

    assert module._issue_references(
        "closes #4\nFixes: #4\nRESOLVED #9\nmentions #10 only"
    ) == [4, 9]
    assert module._issue_references(None) == []
    assert module._canonical_issue_references(
        "Canonical-For: #9\ncanonical-for: #4\nCanonical-For: #9"
    ) == [4, 9]
    assert module._canonical_issue_references(42) == []


def test_extract_prs_supports_wrapped_snapshots_and_history_fallback():
    module = _load_governance()
    pr = _base_pr(1)

    assert module._extract_prs({"open_prs": {"data": [pr, "bad"]}}, "open_prs") == [pr]
    assert module._extract_prs({"open_prs": "bad"}, "open_prs") == []
    assert module._extract_pr_history({"open_prs": [pr]}) == [pr]


def test_run_gh_snapshot_records_base_sha_history_and_errors(monkeypatch):
    module = _load_governance()
    responses = iter(
        [
            subprocess.CompletedProcess(
                [],
                0,
                json.dumps({"defaultBranchRef": {"name": "main"}}),
                "",
            ),
            subprocess.CompletedProcess([], 0, json.dumps([_base_pr(1)]), ""),
            subprocess.CompletedProcess([], 0, json.dumps([_base_pr(1)]), ""),
            subprocess.CompletedProcess([], 0, json.dumps({"sha": "c" * 40}), ""),
        ]
    )
    monkeypatch.setattr(module, "run_bounded_capture", lambda *args, **kwargs: next(responses))

    snapshot = module._run_gh_snapshot("ContextualWisdomLab/fast-mlsirm")

    assert snapshot["default_branch"] == "main"
    assert snapshot["base_sha"] == "c" * 40
    assert len(snapshot["open_prs"]) == 1
    assert len(snapshot["pr_history"]) == 1
    assert snapshot["errors"] == []


def test_run_gh_snapshot_fails_closed_on_command_errors(monkeypatch):
    module = _load_governance()
    responses = iter(
        [
            subprocess.CompletedProcess([], 1, "", "repo failed"),
            subprocess.CompletedProcess([], 1, "", "open failed"),
            subprocess.CompletedProcess([], 1, "", "history failed"),
        ]
    )
    monkeypatch.setattr(module, "run_bounded_capture", lambda *args, **kwargs: next(responses))

    snapshot = module._run_gh_snapshot("ContextualWisdomLab/fast-mlsirm")

    assert snapshot["base_sha"] == ""
    assert snapshot["open_prs"] == []
    assert snapshot["pr_history"] == []
    assert len(snapshot["errors"]) == 3


def test_history_pr_list_omits_heavy_nested_fields_that_502():
    """Audit history must not request files/labels (run 31374029017 failure mode)."""
    module = _load_governance()

    assert "files" not in module._HISTORY_PR_JSON_FIELDS.split(",")
    assert "labels" not in module._HISTORY_PR_JSON_FIELDS.split(",")
    assert "mergeStateStatus" not in module._HISTORY_PR_JSON_FIELDS.split(",")
    # History still needs body for closing-keyword audit.
    assert "body" in module._HISTORY_PR_JSON_FIELDS.split(",")
    assert "number" in module._HISTORY_PR_JSON_FIELDS.split(",")
    # Open queue still needs full classification fields.
    open_fields = module._OPEN_PR_JSON_FIELDS.split(",")
    assert "files" in open_fields
    assert "labels" in open_fields
    assert "mergeStateStatus" in open_fields


def test_pr_list_command_binds_state_limit_and_fields():
    module = _load_governance()
    command = module._pr_list_command(
        "ContextualWisdomLab/fast-mlsirm",
        state="all",
        limit=100,
        fields=module._HISTORY_PR_JSON_FIELDS,
    )

    assert command[:3] == ["gh", "pr", "list"]
    assert "--state" in command and command[command.index("--state") + 1] == "all"
    assert "--limit" in command and command[command.index("--limit") + 1] == "100"
    assert command[command.index("--json") + 1] == module._HISTORY_PR_JSON_FIELDS
    assert "files" not in command[command.index("--json") + 1]


def test_run_gh_json_retries_only_transient_gateway_statuses(monkeypatch):
    """502/503/504 retry; non-transient failures fail closed without sleep loops."""
    module = _load_governance()
    sleeps: list[float] = []
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))

    transient = iter(
        [
            subprocess.CompletedProcess(
                [],
                1,
                "",
                "HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
            ),
            subprocess.CompletedProcess([], 0, json.dumps([{"number": 1}]), ""),
        ]
    )
    monkeypatch.setattr(
        module,
        "run_bounded_capture",
        lambda *args, **kwargs: next(transient),
    )
    payload, error = module._run_gh_json(
        ["gh", "pr", "list"],
        max_attempts=3,
        retry_sleep_seconds=0.01,
    )
    assert error is None
    assert payload == [{"number": 1}]
    assert sleeps == [0.01]

    permanent = iter(
        [
            subprocess.CompletedProcess([], 1, "", "HTTP 401: Bad credentials"),
            subprocess.CompletedProcess([], 0, json.dumps([{"number": 9}]), ""),
        ]
    )
    monkeypatch.setattr(
        module,
        "run_bounded_capture",
        lambda *args, **kwargs: next(permanent),
    )
    payload, error = module._run_gh_json(
        ["gh", "pr", "list"],
        max_attempts=3,
        retry_sleep_seconds=0.01,
    )
    assert payload is None
    assert error is not None
    assert error["returncode"] == 1
    assert "401" in error["stderr"]
    assert sleeps == [0.01]  # no additional sleep for non-transient


def test_run_gh_snapshot_uses_light_history_fields_and_recovers_from_502(
    monkeypatch,
):
    """Open list keeps full fields; history uses light fields and recovers after 502."""
    module = _load_governance()
    recorded: list[list[str]] = []
    history_attempts = {"n": 0}

    def fake_run(command, **kwargs):
        recorded.append(list(command))
        if command[1:3] == ["repo", "view"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"defaultBranchRef": {"name": "main"}}),
                "",
            )
        if command[1:3] == ["pr", "list"]:
            state = command[command.index("--state") + 1]
            fields = command[command.index("--json") + 1]
            if state == "open":
                assert "files" in fields.split(",")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps([_base_pr(10, files=["a.py", "b.py"])]),
                    "",
                )
            if state == "all":
                assert "files" not in fields.split(",")
                assert "labels" not in fields.split(",")
                history_attempts["n"] += 1
                if history_attempts["n"] == 1:
                    return subprocess.CompletedProcess(
                        command,
                        1,
                        "",
                        "HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
                    )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps(
                        [
                            _base_pr(10, files=["a.py"]),
                            _base_pr(9, state="MERGED", body="Fixes #1"),
                        ]
                    ),
                    "",
                )
        if command[1] == "api":
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"sha": "d" * 40}),
                "",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(module, "run_bounded_capture", fake_run)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    snapshot = module._run_gh_snapshot("ContextualWisdomLab/fast-mlsirm")

    assert snapshot["errors"] == []
    assert snapshot["base_sha"] == "d" * 40
    assert len(snapshot["open_prs"]) == 1
    assert len(snapshot["pr_history"]) == 2
    assert history_attempts["n"] == 2
    history_commands = [
        cmd for cmd in recorded if cmd[1:3] == ["pr", "list"] and "--state" in cmd
    ]
    all_cmds = [cmd for cmd in history_commands if cmd[cmd.index("--state") + 1] == "all"]
    assert all_cmds
    assert all(
        "files" not in cmd[cmd.index("--json") + 1].split(",") for cmd in all_cmds
    )


def test_run_gh_snapshot_records_exhausted_history_502_without_dropping_open_prs(
    monkeypatch,
):
    """When history stays 502 after retries, open PRs remain and the error is kept."""
    module = _load_governance()

    def fake_run(command, **kwargs):
        if command[1:3] == ["repo", "view"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"defaultBranchRef": {"name": "main"}}),
                "",
            )
        if command[1:3] == ["pr", "list"]:
            state = command[command.index("--state") + 1]
            if state == "open":
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps([_base_pr(25)]),
                    "",
                )
            return subprocess.CompletedProcess(
                command,
                1,
                "",
                "HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
            )
        if command[1] == "api":
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"sha": "e" * 40}),
                "",
            )
        raise AssertionError(command)

    monkeypatch.setattr(module, "run_bounded_capture", fake_run)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    snapshot = module._run_gh_snapshot("ContextualWisdomLab/fast-mlsirm")

    assert len(snapshot["open_prs"]) == 1
    assert snapshot["pr_history"] == []
    assert snapshot["base_sha"] == "e" * 40
    assert len(snapshot["errors"]) == 1
    assert snapshot["errors"][0]["command"] == ["pr", "list"]
    assert "HTTP 502" in snapshot["errors"][0]["stderr"]


def test_build_fails_when_history_snapshot_still_errors_after_retries(
    tmp_path, monkeypatch
):
    """Real non-recovered transport failure still hard-fails github:snapshot."""
    module = _load_governance()
    monkeypatch.setattr(module, "_source_commit", lambda root: _CANONICAL_TEST_COMMIT)

    def broken_snapshot(repo: str) -> dict:
        return {
            "mode": "live",
            "repo": repo,
            "default_branch": "main",
            "base_sha": "f" * 40,
            "open_prs": [_base_pr(1)],
            "pr_history": [],
            "errors": [
                {
                    "command": ["pr", "list"],
                    "returncode": 1,
                    "stderr": "HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
                }
            ],
        }

    monkeypatch.setattr(module, "_run_gh_snapshot", broken_snapshot)
    args = argparse.Namespace(
        repo_root=str(tmp_path),
        out=str(tmp_path / "out"),
        repo="ContextualWisdomLab/fast-mlsirm",
        contract_value_krw=2_000_000_000,
        offline_snapshot=None,
        offline_github=False,
        max_stale_days=7,
        generated_at="2026-07-03T00:00:00+00:00",
    )
    manifest = module.build_pr_queue_governance(args)

    assert manifest["status"] == "failed"
    assert any(
        check["name"] == "github:snapshot" and check["ok"] is False
        for check in manifest["failed_checks"]
    )
    assert manifest["open_pr_count"] == 1


def test_source_commit_and_json_helpers_cover_success_and_failure(tmp_path, monkeypatch):
    module = _load_governance()
    canonical_sha1 = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, f"{canonical_sha1}\n", ""),
    )
    assert module._source_commit(tmp_path) == canonical_sha1

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    try:
        module._source_commit(tmp_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-timeout git failure must not degrade to 'unknown'")

    object_path = tmp_path / "object.json"
    object_path.write_text('{"a": 1}', encoding="utf-8")
    assert module._read_json(object_path) == {"a": 1}

    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    try:
        module._read_json(array_path)
    except RuntimeError as exc:
        assert "must be an object" in str(exc)
    else:
        raise AssertionError("array JSON must fail closed")


def test_safe_url_allows_known_safe_schemes_and_blocks_unsafe_schemes():
    module = _load_governance()

    assert module._safe_url("https://github.com/org/repo/pull/1") == "https://github.com/org/repo/pull/1"
    assert module._safe_url("http://example.test/report") == "http://example.test/report"
    assert module._safe_url("mailto:security@example.test") == "mailto:security@example.test"
    assert module._safe_url("/relative/report.html") == "/relative/report.html"
    assert module._safe_url("javascript:alert(1)") == "#"
    assert module._safe_url("data:text/html,<script>alert(1)</script>") == "#"
    assert module._safe_url("vbscript:msgbox(1)") == "#"
    assert module._safe_url("ftp://example.test/report") == "#"
    assert module._safe_url("") == "#"
    assert module._safe_url(None) == "#"


def test_datetime_path_and_empty_normalizers_cover_edge_branches(tmp_path):
    module = _load_governance()

    naive = module._parse_datetime("2026-07-03T00:00:00")
    assert naive.tzinfo == module.UTC
    assert module._resolve_path("child/file.json", base=tmp_path) == tmp_path / "child/file.json"
    absolute = tmp_path / "absolute.json"
    assert module._resolve_path(absolute, base=Path("/ignored")) == absolute
    assert module._normalize_pr_list({"not": "a list"}) == []
    assert module._label_names({"labels": "bad"}) == []
    assert module._label_names({"labels": [None, {}, {"name": " "}, ""]}) == []
    assert module._changed_files({"files": "bad"}) == []
    assert module._changed_files({"files": [None, {}, {"path": " "}, ""]}) == []


def test_snapshot_from_args_uses_live_capture_when_not_offline(tmp_path, monkeypatch):
    module = _load_governance()
    expected = {"mode": "live", "open_prs": [], "pr_history": [], "errors": []}
    monkeypatch.setattr(module, "_run_gh_snapshot", lambda repo: {**expected, "repo": repo})
    args = argparse.Namespace(
        repo_root=str(tmp_path),
        out=str(tmp_path / "out"),
        repo="ContextualWisdomLab/fast-mlsirm",
        contract_value_krw=2_000_000_000,
        offline_snapshot=None,
        offline_github=False,
        max_stale_days=7,
        generated_at="2026-07-03T00:00:00+00:00",
    )

    snapshot = module._snapshot_from_args(args, tmp_path)

    assert snapshot["mode"] == "live"
    assert snapshot["repo"] == args.repo


def test_run_gh_snapshot_handles_missing_branch_and_nonobject_base(monkeypatch):
    module = _load_governance()
    no_branch_responses = iter(
        [
            subprocess.CompletedProcess([], 0, json.dumps({"defaultBranchRef": "main"}), ""),
            subprocess.CompletedProcess([], 0, "[]", ""),
            subprocess.CompletedProcess([], 0, "[]", ""),
        ]
    )
    monkeypatch.setattr(
        module,
        "run_bounded_capture",
        lambda *args, **kwargs: next(no_branch_responses),
    )
    snapshot = module._run_gh_snapshot("ContextualWisdomLab/fast-mlsirm")
    assert snapshot["default_branch"] == ""
    assert snapshot["base_sha"] == ""

    base_responses = iter(
        [
            subprocess.CompletedProcess(
                [],
                0,
                json.dumps({"defaultBranchRef": {"name": "main"}}),
                "",
            ),
            subprocess.CompletedProcess([], 0, "[]", ""),
            subprocess.CompletedProcess([], 0, "[]", ""),
            subprocess.CompletedProcess([], 0, json.dumps(["not", "an", "object"]), ""),
        ]
    )
    monkeypatch.setattr(
        module,
        "run_bounded_capture",
        lambda *args, **kwargs: next(base_responses),
    )
    snapshot = module._run_gh_snapshot("ContextualWisdomLab/fast-mlsirm")
    assert snapshot["default_branch"] == "main"
    assert snapshot["base_sha"] == ""


def test_classify_pr_without_timestamp_or_risks_is_clean():
    module = _load_governance()
    pr = _base_pr(5)
    pr.update(
        {
            "title": "Documentation cleanup",
            "headRefName": "docs/cleanup",
            "updatedAt": "",
            "reviewDecision": "",
            "mergeStateStatus": "",
        }
    )

    classified = module.classify_pr(
        pr,
        now=module._parse_datetime("2026-07-03T00:00:00+00:00"),
        max_stale_days=7,
    )

    assert classified["age_days"] is None
    assert classified["risk_reasons"] == []
    assert "updated_at_utc" not in classified


def test_no_duplicate_heads_or_overlap_for_empty_and_small_sets():
    module = _load_governance()
    prs = [
        module.classify_pr(
            {**_base_pr(1), "headRefName": "", "files": []},
            now=module._parse_datetime("2026-07-03T00:00:00+00:00"),
            max_stale_days=7,
        ),
        module.classify_pr(
            {**_base_pr(2), "files": [{"path": "central.py"}]},
            now=module._parse_datetime("2026-07-03T00:00:00+00:00"),
            max_stale_days=7,
        ),
    ]

    assert module._duplicate_head_warnings(prs) == []
    assert module._changed_file_overlap_warnings(prs) == []


def test_issue_claim_history_skips_unreferenced_prs():
    module = _load_governance()

    assert module._issue_claim_history([_base_pr(1), "bad"]) == []


def test_render_report_ignores_malformed_optional_records():
    module = _load_governance()
    html = module._render_report(
        {
            "status": "ok",
            "repo": "ContextualWisdomLab/fast-mlsirm",
            "base_sha": "d" * 40,
            "generated_at": "2026-07-03T00:00:00+00:00",
            "open_pr_count": 0,
            "risk_counts": {},
            "pull_requests": ["bad"],
            "duplicate_issue_claims": {"1": "bad"},
            "changed_file_overlap_warnings": ["bad"],
            "issue_claim_history": ["bad"],
            "checks": ["bad"],
            "duplicate_head_warnings": [],
        }
    )

    assert "PR Queue Governance" in html
    assert "d" * 40 in html


def test_main_returns_success_failure_and_exception(tmp_path, monkeypatch, capsys):
    module = _load_governance()
    monkeypatch.setattr(module, "_source_commit", lambda root: _CANONICAL_TEST_COMMIT)
    ok_snapshot = _write_custom_snapshot(tmp_path / "ok.json", open_prs=[])
    ok_code = module.main(
        [
            "--repo-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "ok-out"),
            "--offline-snapshot",
            str(ok_snapshot),
        ]
    )
    assert ok_code == 0
    assert '"status": "ok"' in capsys.readouterr().out

    conflict_snapshot = _write_custom_snapshot(
        tmp_path / "conflict.json",
        open_prs=[
            _base_pr(1, body="Closes #8"),
            _base_pr(2, body="Fixes #8"),
        ],
    )
    failed_code = module.main(
        [
            "--repo-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "failed-out"),
            "--offline-snapshot",
            str(conflict_snapshot),
        ]
    )
    assert failed_code == 1
    assert '"status": "failed"' in capsys.readouterr().out

    monkeypatch.setattr(
        module,
        "build_pr_queue_governance",
        lambda args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    exception_code = module.main(["--repo-root", str(tmp_path)])
    assert exception_code == 1
    assert '"error": "boom"' in capsys.readouterr().out


def test_extract_prs_rejects_wrapped_nonlist_data():
    module = _load_governance()

    assert module._extract_prs({"open_prs": {"data": "bad"}}, "open_prs") == []


def test_render_report_handles_nonmapping_duplicate_claims():
    module = _load_governance()
    html = module._render_report(
        {
            "status": "ok",
            "repo": "repo/name",
            "base_sha": "",
            "generated_at": "",
            "open_pr_count": 0,
            "risk_counts": {},
            "pull_requests": [],
            "duplicate_issue_claims": [],
            "changed_file_overlap_warnings": [],
            "issue_claim_history": [],
            "checks": [],
            "duplicate_head_warnings": [],
        }
    )

    assert "Duplicate Issue Claims" in html
