"""Smoke: Windows/POSIX MCP server launch — regression guard for #61."""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MCP_JSON = ROOT / ".mcp.json"
RUN_SERVER = ROOT / "bin" / "run-server"
RUN_SERVER_CMD = ROOT / "bin" / "run-server.cmd"
CONFIG_EXAMPLE = ROOT / "config" / "config.example.yaml"
REQUIREMENTS = ROOT / "requirements.txt"
REQUIREMENTS_DEV = ROOT / "requirements-dev.txt"

SKILLS_REQUIRING_OS_BRANCH = [
    ROOT / "skills" / "setup" / "SKILL.md",
    ROOT / "skills" / "session-start" / "SKILL.md",
    ROOT / "skills" / "configure" / "SKILL.md",
]

SKILLS_REQUIRING_PY_LAUNCHER_FALLBACK = [
    ROOT / "skills" / "setup" / "SKILL.md",
    ROOT / "skills" / "session-start" / "SKILL.md",
]

SKILLS_WITH_MULTILINE_PYTHON = [
    ROOT / "skills" / "setup" / "SKILL.md",
    ROOT / "skills" / "session-start" / "SKILL.md",
    ROOT / "skills" / "configure" / "SKILL.md",
]

DEV_ONLY_PACKAGES = ["pytest", "ruff", "mypy", "types-PyYAML"]


def test_mcp_json_is_valid_json():
    json.loads(MCP_JSON.read_text(encoding="utf-8"))


def test_mcp_json_command_has_no_hardcoded_venv_subpath():
    """command must go through the OS-agnostic bin/run-server wrapper, not hardcode
    venv/bin (POSIX) or venv\\Scripts (Windows) directly — that's exactly bug #61."""
    config = json.loads(MCP_JSON.read_text(encoding="utf-8"))
    command = config["mcpServers"]["project-hub-mcp"]["command"]
    assert "venv/bin" not in command
    assert "venv\\Scripts" not in command and "venv/Scripts" not in command
    assert command.endswith("bin/run-server")


def test_mcp_json_schema():
    """A dropped/typo'd field here silently breaks the MCP server for every user."""
    config = json.loads(MCP_JSON.read_text(encoding="utf-8"))
    server = config["mcpServers"]["project-hub-mcp"]
    assert server["type"] == "stdio"
    assert isinstance(server["args"], list) and len(server["args"]) == 1
    assert "CLAUDE_PLUGIN_ROOT" in server["env"]


def test_run_server_wrapper_exists_and_is_executable():
    assert RUN_SERVER.exists(), "bin/run-server not found"
    assert os.access(RUN_SERVER, os.X_OK), "bin/run-server must have the executable bit set"
    first_line = RUN_SERVER.read_text(encoding="utf-8").splitlines()[0]
    assert first_line in ("#!/bin/sh", "#!/bin/bash"), f"unexpected shebang: {first_line}"


def test_run_server_cmd_wrapper_targets_windows_venv():
    assert RUN_SERVER_CMD.exists(), "bin/run-server.cmd not found"
    content = RUN_SERVER_CMD.read_text(encoding="utf-8")
    assert "%USERPROFILE%" in content
    assert "Scripts\\python.exe" in content


def test_setup_session_configure_skills_document_both_platforms():
    """Regression guard: a file that documents only the POSIX venv path (venv/bin/python3)
    without a Windows equivalent (venv\\Scripts\\python.exe) is exactly the #61 bug —
    check both markers are present together, not that the POSIX one is absent (it must
    legitimately remain, for the POSIX branch)."""
    for skill_md in SKILLS_REQUIRING_OS_BRANCH:
        body = skill_md.read_text(encoding="utf-8")
        assert "venv/bin/python3" in body or "venv/bin/pip" in body, (
            f"{skill_md}: missing POSIX venv interpreter path"
        )
        assert "Scripts\\python.exe" in body or "Scripts\\pip.exe" in body, (
            f"{skill_md}: missing Windows venv interpreter path — likely not yet "
            f"OS-branched (regression for #61)"
        )


def test_setup_session_configure_skills_have_matching_path_counts():
    """Catches a copy-paste swap (e.g. a Windows bullet showing the POSIX path) that
    the plain presence check above wouldn't — POSIX and Windows mentions should pair up."""
    for skill_md in SKILLS_REQUIRING_OS_BRANCH:
        body = skill_md.read_text(encoding="utf-8")
        posix = len(re.findall(r"venv/bin/(?:python3|pip)", body))
        windows = len(re.findall(r"Scripts\\(?:python\.exe|pip\.exe)", body))
        assert posix == windows, (
            f"{skill_md}: POSIX/Windows path mention count mismatch ({posix} vs {windows})"
        )


def test_setup_and_session_start_document_py_launcher_fallback():
    """Regression guard for #69: on managed Windows devices, bare `python`/`python3`
    can resolve to the Microsoft Store app-execution-alias stub (exit code 49) even
    when Python is genuinely installed. `py -3` (the Python Launcher, always on PATH)
    must be documented as a fallback, or setup/session-start silently tell a managed
    device user to install Python they already have."""
    for skill_md in SKILLS_REQUIRING_PY_LAUNCHER_FALLBACK:
        body = skill_md.read_text(encoding="utf-8")
        assert "py -3" in body, (
            f"{skill_md}: missing `py -3` fallback for Windows Store-alias detection failures"
        )


def test_setup_step3_venv_creation_uses_resolved_interpreter_not_hardcoded():
    """Regression guard for #69: venv creation must reuse whichever interpreter Step 0
    resolved (<PY>), not hardcode `python`/`python3` — hardcoding breaks again exactly
    where Step 0 just worked around a Store-alias failure."""
    body = (ROOT / "skills" / "setup" / "SKILL.md").read_text(encoding="utf-8")
    step3 = body.split("### Step 3:", 1)[1].split("### Step 4:", 1)[0]
    assert "<PY> -m venv" in step3, "Step 3 must invoke <PY>, not a hardcoded python/python3"
    assert not re.search(r"^- (?:POSIX|Windows): `python3? -m venv", step3, re.MULTILINE), (
        "Step 3 still hardcodes python/python3 for venv creation"
    )


def test_setup_steps_1_2_5_use_resolved_interpreter_not_hardcoded():
    """Regression guard for #69: Steps 1, 2, and 5 (system-level Python invocations
    before the venv exists) must all reuse <PY> from Step 0, not hardcode python3 —
    the same bug Step 3 already guards against, just for the other three call sites.
    Step 1 uses the write-then-run pattern (#71: <PY> against a file path) rather
    than <PY> -c; Step 2 and Step 5's config-copy remain single-line -c invocations."""
    body = (ROOT / "skills" / "setup" / "SKILL.md").read_text(encoding="utf-8")
    for start, end, label in [
        ("### Step 2:", "### Step 3:", "Step 2"),
        ("### Step 5:", "### Step 5b:", "Step 5"),
    ]:
        section = body.split(start, 1)[1].split(end, 1)[0]
        assert "<PY> -c" in section, f"{label} must invoke <PY>, not a hardcoded python3"
        assert not re.search(r"^python3? -c", section, re.MULTILINE), (
            f"{label} still hardcodes python3/python"
        )

    step1 = body.split("### Step 1:", 1)[1].split("### Step 2:", 1)[0]
    assert "<PY>" in step1, "Step 1 must invoke <PY>, not a hardcoded python3"
    assert not re.search(r"^python3? ", step1, re.MULTILINE), "Step 1 still hardcodes python3/python"


def test_setup_documents_windows_quoting_note_for_py_fallback():
    """Regression guard: PowerShell call-operator quoting note for space-containing
    full paths from the known-install fallback must stay present — without it, an
    Intune-managed device whose Python only resolves via C:\\Program Files\\... fails
    on the exact scenario #69 was filed for."""
    body = (ROOT / "skills" / "setup" / "SKILL.md").read_text(encoding="utf-8")
    assert "Windows quoting note" in body
    assert '& "<PY>"' in body


def test_setup_known_install_fallback_uses_powershell_env_syntax():
    """Regression guard: known-install-path hints must use PowerShell $env: syntax,
    not cmd %VAR% syntax — they sit inside PowerShell call-operator commands, %VAR%
    would silently fail to expand there."""
    body = (ROOT / "skills" / "setup" / "SKILL.md").read_text(encoding="utf-8")
    assert "$env:ProgramFiles" in body
    assert "%ProgramFiles%" not in body
    assert "%LocalAppData%" not in body


def test_setup_step5b_uses_claude_plugin_root_not_candidate_guessing():
    """Regression guard for #70: Step 5b must use the interpolated ${CLAUDE_PLUGIN_ROOT}
    (already proven to work in Steps 4/5/6) instead of guessing the plugin root via a
    hardcoded candidate-path search — the candidate list didn't include the actual cache
    path and could silently install knowledge templates from the wrong version, or skip
    them entirely, once cache and checkout diverge."""
    body = (ROOT / "skills" / "setup" / "SKILL.md").read_text(encoding="utf-8")
    step5b = body.split("### Step 5b:", 1)[1].split("### Step 6:", 1)[0]
    assert "${CLAUDE_PLUGIN_ROOT}/knowledge" in step5b, (
        "Step 5b must interpolate ${CLAUDE_PLUGIN_ROOT} like Steps 4/5/6 do"
    )
    assert "candidates" not in step5b, (
        "Step 5b still guesses the plugin root via a candidate-path search (regression for #70)"
    )
    assert "PLUGIN_ROOT_NOT_FOUND" not in step5b, (
        "Step 5b still has the silent-skip fallback from the candidate-guessing heuristic"
    )


def _extract_python_block(section_text):
    """Pull the content of the first ```python fenced block out of a SKILL.md section."""
    match = re.search(r"```python\n(.*?)```", section_text, re.DOTALL)
    assert match, "no ```python fenced block found in section"
    return match.group(1)


def test_setup_step5b_script_copies_new_but_never_overwrites_existing():
    """Real execution of Step 5b's embedded script, not just a text check. This is the
    exact mechanism the #70 breaking-change analysis relies on to argue re-running setup
    is safe: templates are copied once but never overwritten on a later run, so a user's
    own edits to a placeholder file survive. That guarantee had zero executable coverage —
    close it by running the actual script against a real directory tree, twice."""
    body = (ROOT / "skills" / "setup" / "SKILL.md").read_text(encoding="utf-8")
    step5b = body.split("### Step 5b:", 1)[1].split("### Step 6:", 1)[0]
    script = _extract_python_block(step5b)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        plugin_root = tmp_path / "plugin_root"
        (plugin_root / "knowledge" / "generic").mkdir(parents=True)
        (plugin_root / "knowledge" / "generic" / "charter.md").write_text("ORIGINAL TEMPLATE")

        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        env = {**os.environ, "HOME": str(fake_home), "USERPROFILE": str(fake_home)}

        resolved_script = script.replace("${CLAUDE_PLUGIN_ROOT}", str(plugin_root))
        dest_file = fake_home / ".project-hub" / "knowledge" / "generic" / "charter.md"

        # First run: nothing installed yet -> must copy.
        result = subprocess.run(
            [sys.executable, "-c", resolved_script], env=env, capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0, f"first run failed: {result.stderr}"
        assert dest_file.exists(), "first run must install the template"
        assert dest_file.read_text() == "ORIGINAL TEMPLATE"

        # Simulate the user editing their placeholder.
        dest_file.write_text("USER EDITED CONTENT")

        # Second run: file already exists -> must NOT overwrite the user's edit.
        result = subprocess.run(
            [sys.executable, "-c", resolved_script], env=env, capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0, f"second run failed: {result.stderr}"
        assert dest_file.read_text() == "USER EDITED CONTENT", (
            "Step 5b overwrote an already-installed template — this would silently destroy "
            "user-authored content on every /project-hub:setup re-run"
        )


def test_claude_plugin_root_interpolation_survives_windows_backslash_paths():
    """Regression guard: ${CLAUDE_PLUGIN_ROOT} can be interpolated as a Windows path
    containing backslashes (e.g. C:\\Users\\...\\project-hub). Embedded in a plain
    (non-raw) Python string literal, a stray \\U/\\u/\\N sequence becomes an invalid
    Unicode escape and the script fails with a SyntaxError before it even runs — this
    broke test_setup_step5b_script_copies_new_but_never_overwrites_existing for real on
    Windows CI. Verify Steps 5, 5b, and 6 all use a raw string, so this can't recur
    regardless of which OS actually generates the substituted path (checked here via
    compile(), not subprocess, so it runs identically on every CI platform)."""
    body = (ROOT / "skills" / "setup" / "SKILL.md").read_text(encoding="utf-8")
    fake_windows_root = r"C:\Users\RUNNER~1\AppData\Local\Temp\plugin_root"
    for start, end, label in [
        ("### Step 5:", "### Step 5b:", "Step 5"),
        ("### Step 5b:", "### Step 6:", "Step 5b"),
        ("### Step 6:", "### Step 7:", "Step 6"),
    ]:
        section = body.split(start, 1)[1].split(end, 1)[0]
        script = _extract_python_block(section)
        resolved = script.replace("${CLAUDE_PLUGIN_ROOT}", fake_windows_root)
        try:
            compile(resolved, f"<{label}>", "exec")
        except SyntaxError as e:
            raise AssertionError(
                f"{label}: script is not valid Python once ${{CLAUDE_PLUGIN_ROOT}} is "
                f"substituted with a Windows backslash path — missing raw string (r'...')"
                f" around the placeholder: {e}"
            ) from e


def test_db_path_examples_do_not_recommend_cloud_sync():
    """Regression guard for #71: config.example.yaml must not list a cloud-sync
    folder (Dropbox et al.) as an equivalent db_path option to a real network share
    — cloud-sync clients copy files instead of respecting locks, and under WAL mode
    that causes silent data corruption, not a recoverable SQLITE_BUSY error."""
    content = CONFIG_EXAMPLE.read_text(encoding="utf-8")
    assert "db_path: ~/Dropbox" not in content, (
        "config.example.yaml still recommends a Dropbox path as a db_path example"
    )
    assert "DO NOT use a cloud-sync folder" in content or "Cloud-Sync" in content, (
        "config.example.yaml is missing an explicit warning against cloud-sync db_path"
    )


def test_setup_and_configure_distinguish_cloud_sync_from_network_share():
    """Regression guard for #71: the runtime db_path check in setup and configure must
    treat cloud-sync paths (Dropbox/OneDrive/Google Drive/iCloud) as a distinct,
    stronger-worded risk from a genuine network share (NFS/Samba) — lumping them into
    one generic 'team use' warning hides the silent-corruption risk of the former."""
    setup_body = (ROOT / "skills" / "setup" / "SKILL.md").read_text(encoding="utf-8")
    assert "cloud_sync_hints" in setup_body, "setup must detect cloud-sync paths separately"
    assert "CLOUD_SYNC:" in setup_body and "NETWORK_SHARE:" in setup_body, (
        "setup must distinguish CLOUD_SYNC from NETWORK_SHARE outcomes"
    )
    # macOS iCloud Drive's real path is ~/Library/Mobile Documents/... (no literal
    # "iCloud"); modern Google Drive mounts as ".../My Drive/..." (no literal "Google
    # Drive") — both must be covered, not just the four obvious brand names.
    assert "Mobile Documents" in setup_body and "My Drive" in setup_body, (
        "setup's cloud_sync_hints must cover iCloud Drive's and Google Drive's real "
        "on-disk path fragments, not just the brand name substrings"
    )
    configure_body = (ROOT / "skills" / "configure" / "SKILL.md").read_text(encoding="utf-8")
    assert "Cloud-Sync-Pfad erkannt" in configure_body, (
        "configure must also warn distinctly about cloud-sync db_path changes"
    )


def test_docs_root_default_matches_config_example():
    """Regression guard for #71: skills/setup/SKILL.md's stated docs_root default must
    match the actual default shipped in config.example.yaml — they drifted apart once
    already (SKILL.md said ~/.project-hub/projects, the real default was
    ~/Documents/project-hub) and nothing would catch a repeat without this test."""
    config_content = CONFIG_EXAMPLE.read_text(encoding="utf-8")
    match = re.search(r"^docs_root:\s*(\S+)", config_content, re.MULTILINE)
    assert match, "config.example.yaml must set a default docs_root"
    actual_default = match.group(1)

    setup_body = (ROOT / "skills" / "setup" / "SKILL.md").read_text(encoding="utf-8")
    assert f"`{actual_default}`" in setup_body, (
        f"setup/SKILL.md's documented docs_root default doesn't match config.example.yaml's "
        f"actual default ({actual_default})"
    )


def test_skills_use_write_then_run_for_multiline_python_not_inline_c():
    """Regression guard for #71: a `-c "..."` argument that spans multiple lines parses
    differently across bash/PowerShell/cmd and reliably breaks under PowerShell — the
    exact failure mode the Windows field test hit. No fenced code block in these skills
    should be paired with a multi-line -c invocation; multi-line scripts must be saved
    to a file and run as a plain path argument instead."""
    multiline_c_pattern = re.compile(r'-c\s+"\s*\n')
    for skill_md in SKILLS_WITH_MULTILINE_PYTHON:
        body = skill_md.read_text(encoding="utf-8")
        assert not multiline_c_pattern.search(body), (
            f"{skill_md}: found a multi-line `-c \"...` invocation — breaks under "
            f"PowerShell, use the write-then-run pattern instead"
        )


def test_requirements_split_runtime_from_dev_tooling():
    """Regression guard for #71: dev/test tooling (pytest, ruff, mypy, types-PyYAML)
    must not ship in the runtime requirements.txt that /project-hub:setup installs on
    end users' machines — it costs install time/disk space for tooling they never run.
    requirements-dev.txt must pull in requirements.txt and add the dev tooling."""
    assert REQUIREMENTS.exists() and REQUIREMENTS_DEV.exists()
    runtime_content = REQUIREMENTS.read_text(encoding="utf-8")
    dev_content = REQUIREMENTS_DEV.read_text(encoding="utf-8")
    for pkg in DEV_ONLY_PACKAGES:
        assert pkg not in runtime_content, f"{pkg} must not be in runtime requirements.txt"
        assert pkg in dev_content, f"{pkg} must be in requirements-dev.txt"
    assert "requirements.txt" in dev_content, (
        "requirements-dev.txt must reference requirements.txt (e.g. -r requirements.txt) "
        "so a dev install still gets the runtime deps"
    )


def test_setup_installs_runtime_requirements_not_dev():
    """Regression guard for #71: /project-hub:setup (Step 4) must install
    requirements.txt for end users, never requirements-dev.txt — swapping this
    would silently reintroduce pytest/ruff/mypy on every user's machine even
    though test_requirements_split_runtime_from_dev_tooling stays green (it only
    checks file *content*, not which file Step 4 actually wires up to pip)."""
    body = (ROOT / "skills" / "setup" / "SKILL.md").read_text(encoding="utf-8")
    step4 = body.split("### Step 4:", 1)[1].split("### Step 5:", 1)[0]
    assert "requirements.txt" in step4
    assert "requirements-dev.txt" not in step4


def test_contributing_and_ci_reference_same_dev_requirements_file():
    """Regression guard for #71: CONTRIBUTING.md's install commands and CI's install
    step must reference the identical dev-requirements filename — a typo'd/renamed
    divergence here isn't caught by CI (CI never reads CONTRIBUTING.md) and would only
    surface as a contributor's pip install failing on a file that doesn't exist."""
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    contributing_refs = set(re.findall(r"requirements[-\w]*\.txt", contributing))
    ci_install_refs = set(re.findall(r"pip install -r (requirements[-\w]*\.txt)", ci))

    assert REQUIREMENTS_DEV.name in contributing_refs, (
        "CONTRIBUTING.md must reference requirements-dev.txt for local dev setup"
    )
    assert ci_install_refs == {REQUIREMENTS_DEV.name}, (
        f"CI's install step references {ci_install_refs}, expected only "
        f"{{'{REQUIREMENTS_DEV.name}'}} — must match CONTRIBUTING.md"
    )
    for fname in contributing_refs | ci_install_refs:
        assert (ROOT / fname).exists(), f"{fname} referenced but not found in repo root"


def test_run_server_wrapper_actually_launches_python():
    """Real subprocess spawn through the OS-appropriate wrapper — proves shebang
    execution / %USERPROFILE%-%*-quoting actually work, not just that the files exist."""
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        if sys.platform == "win32":
            venv_scripts = home / ".project-hub" / "venv" / "Scripts"
            venv_scripts.mkdir(parents=True)
            (venv_scripts / "python.exe").write_bytes(Path(sys.executable).read_bytes())
            env = {**os.environ, "USERPROFILE": str(home)}
            cmd = [str(RUN_SERVER_CMD), "-c", "print('OK')"]
        else:
            venv_bin = home / ".project-hub" / "venv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python3").symlink_to(sys.executable)
            env = {**os.environ, "HOME": str(home)}
            cmd = [str(RUN_SERVER), "-c", "print('OK')"]

        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, f"wrapper failed: {result.stderr}"
        assert "OK" in result.stdout
