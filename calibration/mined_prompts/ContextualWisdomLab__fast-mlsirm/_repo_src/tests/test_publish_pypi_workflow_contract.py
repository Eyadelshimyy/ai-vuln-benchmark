from __future__ import annotations

from pathlib import Path
import gzip
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import textwrap
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml"
RELEASE_TAG_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-tag.yml"
PYPI_PUBLISH_SHA = "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _release_tag_workflow_text() -> str:
    return RELEASE_TAG_WORKFLOW.read_text(encoding="utf-8")


def _job_block(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        text,
    )
    assert match is not None, f"missing {name!r} job"
    return match.group(0)


def test_release_builds_are_bound_to_the_reviewed_source_commit() -> None:
    text = _workflow_text()
    verify = _job_block(text, "verify-release")
    sdist = _job_block(text, "sdist")
    wheels = _job_block(text, "wheels")

    assert "      release_commit:\n" in text
    assert re.search(
        r"(?m)^      release_commit:\n(?:        .*\n)*?        required: true$",
        text,
    )
    assert "release_commit: ${{ steps.release-source.outputs.value }}" in verify
    assert "ref: ${{ github.sha }}" in verify
    assert "ref: ${{ inputs.release_commit }}" not in text
    assert "ref: ${{ inputs.control_plane_commit }}" not in text
    for job in (sdist, wheels):
        assert "ref: ${{ needs.verify-release.outputs.release_commit }}" not in job
        assert "persist-credentials: false" in job
        assert "ref: ${{ inputs.release_tag }}" not in job

    assert "fetch-depth: 0" in verify
    assert 'RELEASE_TAG: ${{ inputs.release_tag }}' in verify
    assert 'RELEASE_COMMIT: ${{ inputs.release_commit }}' in verify
    assert 'RELEASE_COMMIT: ${{ steps.release-source.outputs.value }}' in verify
    assert "release_commit must be a canonical 40-character lowercase SHA-1" in verify
    assert 'git rev-parse --verify "$RELEASE_COMMIT^{commit}"' in verify
    assert "release commit must be an ancestor of the publication control plane" in verify
    assert 'git checkout --detach "$canonical_release_commit"' in verify
    assert 'git rev-parse -q --verify "$RELEASE_TAG^{commit}"' in verify
    assert "checked-out release source does not match release_commit" in verify
    assert "release tag does not target release_commit" in verify
    assert 'tomllib.load' in verify or 'tomllib.loads' in verify
    assert 'f"v{project[\'version\']}"' in verify
    assert text.count("maturin-version: v1.15.0") == 5


def test_release_checkout_rejects_unvalidated_dispatch_sha_authority() -> None:
    text = _workflow_text()
    verify = _job_block(text, "verify-release")

    assert "ref: ${{ inputs.release_commit }}" not in text
    assert "ref: ${{ inputs.control_plane_commit }}" not in text
    assert "release_commit: ${{ steps.release-source.outputs.value }}" in verify
    assert "id: release-source" in verify
    assert "ref: ${{ github.sha }}" in verify
    assert verify.index("ref: ${{ github.sha }}") < verify.index("id: release-source")
    assert "git merge-base --is-ancestor" in verify
    assert 'echo "value=$canonical_release_commit" >> "$GITHUB_OUTPUT"' in verify

    for job_name in (
        "sdist",
        "wheels",
        "reproducibility-record",
        "release-admission",
        "create-tag-and-release",
    ):
        job = _job_block(text, job_name)
        assert "ref: ${{ needs.verify-release.outputs.release_commit }}" not in job


def test_wheels_cover_supported_cpython_versions_on_every_platform() -> None:
    wheels = _job_block(_workflow_text(), "wheels")

    # The extension is not built with PyO3 abi3, so a CPython 3.12 wheel cannot
    # satisfy 3.13/3.14 callers. Each release platform must build all currently
    # evidenced supported CPython versions instead of forcing newer callers to
    # compile the sdist with a local Rust toolchain.
    for version in ("3.12", "3.13", "3.14"):
        assert wheels.count(f'python-version: "{version}"') == 4
        assert wheels.count(f"interpreter: python{version}") == 2

    assert wheels.count("interpreter: python\n") == 6
    assert "python-version: ${{ matrix.python-version }}" in wheels
    assert "args: --release --out dist -i ${{ matrix.interpreter }}" in wheels
    assert "name: dist-wheel-${{ matrix.target }}-py${{ matrix.python-version }}" in wheels


@pytest.mark.parametrize("job_name", ["verify-release", "sdist", "wheels", "macos-x86-runtime", "reproducibility-record", "release-admission", "create-tag-and-release"])
def test_direct_publish_ancestry_guard_against_real_git_history(tmp_path: Path, job_name: str) -> None:
    verify = _job_block(_workflow_text(), job_name)
    name = "Validate and select release source ancestry" if job_name == "verify-release" else "Validate release ancestry in this job"
    step = verify.split(f"- name: {name}\n", 1)[1].split("\n      - ", 1)[0]
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    assert verify.index("ref: ${{ github.sha }}") < verify.index(name)
    if job_name == "verify-release":
        assert verify.index(name) < verify.index("Require release tag and source commit")
        assert 'CONTROL_PLANE_COMMIT: ${{ inputs.control_plane_commit }}' in step
    else:
        assert 'CONTROL_PLANE_COMMIT: ${{ github.sha }}' in step
        assert 'RELEASE_COMMIT: ${{ needs.verify-release.outputs.release_commit }}' in step
        assert "ref: ${{ needs.verify-release.outputs.release_commit }}" not in verify
        assert "fetch-depth: 0" in verify
    assert 'shell: bash --noprofile --norc -e -o pipefail {0}' in step
    assert "fetch-tags: true" in verify

    repo = tmp_path / "git-history"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
           "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, env=env, text=True).strip()

    git("init", "-q")
    git("commit", "--allow-empty", "-qm", "root")
    root = git("rev-parse", "HEAD")
    git("commit", "--allow-empty", "-qm", "control")
    control = git("rev-parse", "HEAD")
    git("checkout", "--detach", "-q", root)
    git("commit", "--allow-empty", "-qm", "sibling")
    sibling = git("rev-parse", "HEAD")
    absent = "0" * 40
    cases = [
        (control, control, True),
        (root, control, True),
        (sibling, control, False),
        (absent, control, False),
        (root, root, False),
        (root.upper(), control, False),
    ]
    for index, (release, caller, allowed) in enumerate(cases):
        git("checkout", "--detach", "-q", control)
        marker = tmp_path / f"downstream-{index}"
        output = tmp_path / f"github-output-{index}"
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c",
             script + '\nprintf reached > "$DOWNSTREAM_MARKER"\n'],
            cwd=repo, env={**env, "RELEASE_COMMIT": release, "CONTROL_PLANE_COMMIT": caller,
                           "DOWNSTREAM_MARKER": str(marker), "GITHUB_OUTPUT": str(output)},
            capture_output=True, text=True,
        )
        assert (result.returncode == 0) is allowed, result.stderr
        assert marker.exists() is allowed
        if allowed:
            assert git("rev-parse", "HEAD") == release
            if job_name == "verify-release":
                assert output.read_text() == f"value={release}\n"
        assert git("tag", "--list") == ""


def test_release_tag_selects_only_verified_ancestors(tmp_path: Path) -> None:
    job = _job_block(_release_tag_workflow_text(), "publish-release-tag")
    name = "Verify the release source is on the current default-branch lineage"
    step = job.split(f"- name: {name}\n", 1)[1].split("\n      - ", 1)[0]
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    assert "ref: ${{ inputs.release_commit }}" not in job
    assert job.index("ref: ${{ github.sha }}") < job.index(name)

    env = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
           "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=tmp_path, env=env, text=True).strip()

    git("init", "-q")
    git("commit", "--allow-empty", "-qm", "root")
    root = git("rev-parse", "HEAD")
    git("commit", "--allow-empty", "-qm", "control")
    control = git("rev-parse", "HEAD")
    git("checkout", "--detach", "-q", root)
    git("commit", "--allow-empty", "-qm", "sibling")
    sibling = git("rev-parse", "HEAD")
    git("tag", "-a", "-m", "tag object", "annotated", root)
    tag_object = git("rev-parse", "annotated")
    for release, allowed in ((root, True), (control, True), (sibling, False),
                             ("0" * 40, False), (tag_object, False)):
        git("checkout", "--detach", "-q", control)
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
            cwd=tmp_path, env={**env, "DISPATCH_SHA": control, "RELEASE_COMMIT": release},
            capture_output=True, text=True,
        )
        assert (result.returncode == 0) is allowed, result.stderr
        assert git("rev-parse", "HEAD") == (release if allowed else control)


def test_release_tag_workflow_explicitly_dispatches_package_publish() -> None:
    publish_text = _workflow_text()
    release_text = _release_tag_workflow_text()
    release_job = _job_block(release_text, "publish-release-tag")
    verify = _job_block(publish_text, "verify-release")

    # GITHUB_TOKEN-created release events do not recursively start ordinary
    # event-triggered workflows. Package publication therefore uses the one
    # supported recursive trigger. The control-plane workflow comes from the
    # protected default branch while artifact source identity is an immutable
    # explicit commit, so an older release tag cannot select an outdated
    # publication workflow definition. The dispatch also carries the exact
    # release-tag run commit so a moving default branch cannot silently select a
    # different publication control plane between verification and dispatch.
    assert "  workflow_dispatch:\n" in publish_text
    assert "      release_tag:\n" in publish_text
    assert "      release_commit:\n" in publish_text
    assert "      control_plane_commit:\n" in publish_text
    assert publish_text.count("        required: true\n") >= 3
    assert "types: [published]" not in publish_text

    assert 'DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}' in verify
    assert 'PUBLISH_REF: ${{ github.ref }}' in verify
    assert 'CONTROL_PLANE_COMMIT: ${{ inputs.control_plane_commit }}' in verify
    assert 'CONTROL_PLANE_SHA: ${{ github.sha }}' in verify
    assert 'expected_ref="refs/heads/$DEFAULT_BRANCH"' in verify
    assert 'if [ "$PUBLISH_REF" != "$expected_ref" ]' in verify
    assert 'if [ "$CONTROL_PLANE_SHA" != "$CONTROL_PLANE_COMMIT" ]' in verify
    assert "publication control plane moved after release verification" in verify

    assert "permissions:\n      contents: read\n      actions: write" in release_job
    assert "gh workflow run publish-pypi.yml" in release_job
    assert 'DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}' in release_job
    assert 'git rev-parse "origin/$DEFAULT_BRANCH"' in release_job
    assert "release commit must be an ancestor of the current default branch" in release_job
    assert '--ref "$DEFAULT_BRANCH"' in release_job
    assert '--ref "v$RELEASE_VERSION"' not in release_job
    assert '-f release_tag="v$RELEASE_VERSION"' in release_job
    assert '-f release_commit="$RELEASE_COMMIT"' in release_job
    assert '-f control_plane_commit="$CONTROL_PLANE_COMMIT"' in release_job
    assert 'RELEASE_COMMIT: ${{ inputs.release_commit }}' in release_job
    # R5a: release-tag.yml verifies and dispatches; it never creates the tag or release.
    assert "gh release create" not in release_job
    assert "/git/refs" not in release_job


def test_release_asset_write_is_isolated_from_pypi_credentials() -> None:
    text = _workflow_text()
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")

    assert "permissions:\n      contents: write" in assets
    assert "GH_TOKEN: ${{ github.token }}" in assets
    assert "RELEASE_TAG: ${{ inputs.release_tag }}" in assets
    assert 'gh release upload "$RELEASE_TAG"' in assets
    assert "--clobber" not in assets
    assert "secrets.PIPY_TOKEN" not in assets
    assert "release $RELEASE_TAG is immutable; skipping GitHub asset upload" in assets
    assert ".immutable // false" in assets

    assert "environment: pypi" in publish
    assert "permissions:\n      contents: read" in publish
    assert "gh release upload" not in publish
    assert "contents: write" not in publish


def test_pypi_publish_uses_a_pinned_package_owned_uploader() -> None:
    text = _workflow_text()
    publish = _job_block(text, "publish-pypi")
    sdist = _job_block(text, "sdist")

    assert "python -m pip install" not in text
    assert "python -m twine upload" not in text
    assert "TWINE_USERNAME" not in text
    assert f"uses: pypa/gh-action-pypi-publish@{PYPI_PUBLISH_SHA}" in publish
    assert "password: ${{ secrets.PIPY_TOKEN }}" in publish
    assert "attestations: false" in publish
    assert "skip-existing: true" in publish
    assert "Ensure LICENSE is present in the sdist" in sdist
    assert 'license_member = f"{root}/LICENSE"' in sdist


def test_pypi_publish_reads_back_the_admitted_public_matrix() -> None:
    publish = _job_block(_workflow_text(), "publish-pypi")
    assert "ref: ${{ github.sha }}" in publish
    assert "persist-credentials: false" in publish
    assert publish.index("skip-existing: true") < publish.index(
        "Verify published PyPI matrix against admission"
    )
    assert "python3 scripts/ci/verify_published_pypi.py admission/admitted-manifest.tsv '${{ inputs.release_tag }}'" in publish


def test_pypi_publish_can_recover_independently_of_immutable_asset_upload() -> None:
    text = _workflow_text()
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")

    # Release assets and PyPI are two independent publication sinks fed by the
    # same verified build artifacts. Immutable releases skip asset upload
    # without failing the job so a previously failed PyPI publication can be
    # retried to a green overall run.
    assert "needs: [sdist, wheels, reproducibility-record, release-admission, create-tag-and-release]" in assets
    assert "needs: [sdist, wheels, reproducibility-record, release-admission, create-tag-and-release]" in publish
    assert "release-assets" not in publish.split("needs:", 1)[1].split("\n", 1)[0]
    assert "skipping GitHub asset upload" in assets
    assert "skip-existing: true" in publish


def _step_python(job: str, step_name: str) -> str:
    step = job.split(f"- name: {step_name}\n", 1)[1]
    body = step.split(" - <<'PY'\n", 1)[1].split("\n          PY\n", 1)[0]
    return textwrap.dedent(body)


def test_every_release_build_is_reproducible_from_the_release_commit_clock() -> None:
    text = _workflow_text()
    verify = _job_block(text, "verify-release")
    sdist = _job_block(text, "sdist")
    wheels = _job_block(text, "wheels")
    record = _job_block(text, "reproducibility-record")
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")

    # SOURCE_DATE_EPOCH is the committer time of the exact release commit,
    # derived once and fanned out, never the wall clock of the build runner.
    assert 'epoch="$(git log -1 --format=%ct "$RELEASE_COMMIT")"' in verify
    assert "source_date_epoch: ${{ steps.source-date-epoch.outputs.value }}" in verify
    for job in (verify, sdist, wheels, record):
        assert "date +%s" not in job
    epoch_env = "SOURCE_DATE_EPOCH: ${{ needs.verify-release.outputs.source_date_epoch }}"
    assert f"    env:\n      {epoch_env}\n" in sdist
    assert f"    env:\n      {epoch_env}\n" in wheels

    # maturin-action does not forward SOURCE_DATE_EPOCH into the manylinux
    # container on its own, so every wheel build step passes it explicitly.
    builds = re.findall(r"(?ms)^      - name: (?:Build|Rebuild) wheel.*?(?=^      - )", wheels)
    assert len(builds) == 2
    for build in builds:
        assert "docker-options: -e SOURCE_DATE_EPOCH\n" in build
        assert "before-script-linux: $CARGO_BUILD_PYTHON trusted-control/scripts/ci/capture_release_build_scope.py" in build
        assert "if:" not in build
    assert wheels.count("python trusted-control/scripts/ci/capture_release_build_scope.py") == 2

    # Every publishable artifact (12 wheels + sdist) is rebuilt from a clean
    # target and compared; no leg may opt out of the double build.
    assert "verify-reproducible" not in wheels
    assert "--out dist-rebuild --target-dir target-rebuild" in builds[1]
    assert "- name: Rebuild sdist for byte-reproducibility check" in sdist
    assert "- name: Capture first sdist build tools" in sdist
    assert "- name: Capture second sdist build tools" in sdist
    assert "needs: [verify-release, sdist]" in wheels
    assert "- name: Verify and unpack the same-run sdist" in wheels
    assert "- name: Build this wheel target from the verified sdist" in wheels
    assert "working-directory: sdist-consumer/source" in wheels
    assert "- name: Capture target sdist consumer wheel" in wheels
    assert "- name: Install target sdist consumer wheel" in wheels
    assert "args: --out dist-rebuild" in sdist
    compare_wheel = _step_python(wheels, "Compare double-build wheel digests and record them")
    compare_sdist = _step_python(sdist, "Compare double-build sdist digests and record them")
    assert compare_wheel == compare_sdist
    assert 'second = digest("dist-rebuild")' in compare_wheel
    assert "if first != second:" in compare_wheel
    assert "is not byte-reproducible" in compare_wheel
    assert "clean-target-repeat-same-env" in compare_wheel
    assert '"docker", "images"' not in compare_wheel  # never scan unrelated cached images
    assert "container: ${{ matrix.container }}" in builds[0]
    assert "container: ${{ matrix.container }}" in builds[1]
    containers = re.findall(r"target: (\S+)\n(?:            .*\n)*?            container: \"([^\"]*)\"", wheels)
    assert len(containers) == 12
    for target, image in containers:
        if "linux" in target:
            assert re.fullmatch(r"quay\.io/pypa/manylinux2014_(?:x86_64|aarch64)@sha256:[0-9a-f]{64}", image), image
        else:
            assert image == "", (target, image)
    assert "name: repro-rebuild-${{ matrix.target }}-py${{ matrix.python-version }}" in wheels
    assert "name: repro-rebuild-sdist" in sdist

    # The record fails closed: every downloaded publishable artifact needs a
    # byte-verified row whose digests match the published bytes, and both
    # publication sinks depend on it.
    gate = _step_python(record, "Require a byte-verified row for every publishable artifact")
    assert "pattern: dist-*" in record
    assert "pattern: repro-digest-*" in record
    assert "pattern: repro-digest-*\n          path: repro-digest\n          merge-multiple: false" in record
    assert 'Path("repro-digest").glob("repro-digest-*/*.tsv")' in gate
    assert 'failures.append(f"{name}: no byte-verification row")' in gate
    assert 'row["sha256"] != sha or row["rebuild_sha256"] != sha' in gate
    assert "if failures:" in gate
    assert "len(published) != 13" in gate
    assert "NOT independent-environment or" in gate
    assert "name: reproducibility-record" in record
    for sink in (assets, publish):
        assert "needs: [sdist, wheels, reproducibility-record, release-admission, create-tag-and-release]" in sink
        assert "always()" not in sink
        assert "|| true" not in sink
    assert "|| true" not in record

    # SOURCE_DATE_EPOCH alone is not enough: with the default 16 codegen units
    # fresh builds of the binding crate differ in `.llvm.<hash>` symbol
    # suffixes, so the shipped release profile pins a single codegen unit.
    binding = (REPO_ROOT / "crates" / "fast-mlsirm-py" / "Cargo.toml").read_text(encoding="utf-8")
    assert re.search(r"(?m)^\[profile\.release\]\n(?:(?!\[).*\n)*?codegen-units = 1$", binding)
    assert "cargo_manifest_path: crates/fast-mlsirm-py/Cargo.toml" in text
    assert "cargo_dev_manifest_path: crates/mlsirm-core/Cargo.toml" in text


def _sdist_without_license(path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as writer:
        for name, payload in (("fast_mlsirm-0.0.0/PKG-INFO", b"Metadata-Version: 2.4\n"),
                              ("fast_mlsirm-0.0.0/pyproject.toml", b"[project]\n")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 1_700_000_000
            writer.addfile(info, io.BytesIO(payload))
    path.write_bytes(gzip.compress(buffer.getvalue(), mtime=1_700_000_000))


def test_historical_sdist_license_injection_is_byte_reproducible(tmp_path: Path) -> None:
    script = _step_python(_job_block(_workflow_text(), "sdist"), "Ensure LICENSE is present in the sdist")
    env = {**os.environ, "SOURCE_DATE_EPOCH": "1790068752"}
    outputs = []
    for attempt in range(2):
        work = tmp_path / f"run{attempt}"
        for dist in ("dist", "dist-rebuild"):
            (work / dist).mkdir(parents=True)
            _sdist_without_license(work / dist / "fast_mlsirm-0.0.0.tar.gz")
        (work / "LICENSE").write_bytes((REPO_ROOT / "LICENSE").read_bytes())
        if attempt:
            time.sleep(1.1)  # a wall-clock gzip mtime would now differ
        subprocess.run([sys.executable, "-c", script], cwd=work, env=env, check=True)
        for dist in ("dist", "dist-rebuild"):
            archive = work / dist / "fast_mlsirm-0.0.0.tar.gz"
            with tarfile.open(archive, "r:gz") as reader:
                assert reader.extractfile("fast_mlsirm-0.0.0/LICENSE").read() == (REPO_ROOT / "LICENSE").read_bytes()
            outputs.append(hashlib.sha256(archive.read_bytes()).hexdigest())
    assert len(set(outputs)) == 1, outputs


_PINNED = "quay.io/pypa/manylinux2014_x86_64@sha256:" + "a" * 64
_UNRELATED = "quay.io/pypa/manylinux2014_x86_64@sha256:" + "b" * 64


def _run_compare(tmp_path: Path, name: str, *, leg: str, runner_os: str, container: str,
                 docker_digests: list[str] | None, runner_identity: bool = True) -> subprocess.CompletedProcess:
    script = _step_python(_job_block(_workflow_text(), "wheels"), "Compare double-build wheel digests and record them")
    work = tmp_path / name
    for dist in ("dist", "dist-rebuild"):
        (work / dist).mkdir(parents=True)
        (work / dist / "pkg-0-py3-none-any.whl").write_bytes(b"same bytes")
    stub = work / "bin"
    stub.mkdir()
    docker = stub / "docker"
    # Stub runner docker: `image inspect <ref>` succeeds only for images in docker_digests.
    docker.write_text(
        "#!/bin/sh\n"
        f"KNOWN='{' '.join(docker_digests or [])}'\n"
        'for ref; do :; done\n'  # the image reference is the last argument
        'for d in $KNOWN; do [ "$d" = "$ref" ] && { printf \'["%s"]\\n\' "$ref"; exit 0; }; done\n'
        'echo "Error: No such image: $ref" >&2; exit 1\n'
    )
    docker.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in {"ImageOS", "ImageVersion", "RUNNER_ARCH"}}
    env.update({"PATH": f"{stub}{os.pathsep}{env['PATH']}", "SOURCE_DATE_EPOCH": "1790068752", "LEG": leg,
                "ARTIFACT_GLOB": "*.whl", "RUNNER_OS": runner_os, "CONTAINER_IMAGE": container})
    if runner_identity:
        env.update({"ImageOS": "ubuntu24", "ImageVersion": "20260920.1", "RUNNER_ARCH": "X64"})
    return subprocess.run([sys.executable, "-c", script], cwd=work, env=env, capture_output=True, text=True)


def test_build_env_provenance_is_bound_to_the_invocation_and_fails_closed(tmp_path: Path) -> None:
    linux = "x86_64-unknown-linux-gnu-py3.12"
    ok = _run_compare(tmp_path, "linux-ok", leg=linux, runner_os="Linux", container=_PINNED,
                      docker_digests=[_PINNED, _UNRELATED])
    assert ok.returncode == 0, ok.stderr
    row = (tmp_path / "linux-ok" / "repro-digest" / f"{linux}.tsv").read_text().rstrip("\n").split("\t")
    assert row[-1] == f"container:{_PINNED}"

    # Negative shape 1: a Linux leg with only an unrelated cached image, or no bound digest.
    cached_only = _run_compare(tmp_path, "linux-cached-only", leg=linux, runner_os="Linux", container=_PINNED,
                               docker_digests=[_UNRELATED])
    assert cached_only.returncode != 0 and "is not present on this runner" in cached_only.stderr
    unbound = _run_compare(tmp_path, "linux-unbound", leg=linux, runner_os="Linux", container="",
                           docker_digests=[_UNRELATED])
    assert unbound.returncode != 0 and "no digest-pinned build container" in unbound.stderr
    tag_only = _run_compare(tmp_path, "linux-tag", leg=linux, runner_os="Linux",
                            container="quay.io/pypa/manylinux2014_x86_64:latest", docker_digests=[_UNRELATED])
    assert tag_only.returncode != 0 and "no digest-pinned build container" in tag_only.stderr

    # Negative shape 2: a container-less row (macOS/Windows/sdist) with a container digest or no runner identity.
    for leg, runner_os in (("universal2-apple-darwin-py3.12", "macOS"), ("x86_64-pc-windows-msvc-py3.12", "Windows"),
                           ("sdist", "Linux")):
        carries = _run_compare(tmp_path, f"{runner_os}-{leg}-carries", leg=leg, runner_os=runner_os,
                               container=_PINNED, docker_digests=[_PINNED])
        assert carries.returncode != 0 and "carries container provenance" in carries.stderr, (leg, carries.stderr)
        anonymous = _run_compare(tmp_path, f"{runner_os}-{leg}-anon", leg=leg, runner_os=runner_os, container="",
                                 docker_digests=[_PINNED], runner_identity=False)
        assert anonymous.returncode != 0 and "missing runner identity" in anonymous.stderr, (leg, anonymous.stderr)
        good = _run_compare(tmp_path, f"{runner_os}-{leg}-ok", leg=leg, runner_os=runner_os, container="",
                            docker_digests=[_PINNED])
        assert good.returncode == 0, (leg, good.stderr)
        row = (tmp_path / f"{runner_os}-{leg}-ok" / "repro-digest" / f"{leg}.tsv").read_text().rstrip("\n").split("\t")
        assert row[-1] == f"runner:ubuntu24/20260920.1/{runner_os}/X64"


_RELEASE_COMMIT = "c" * 40
_RELEASE_TAG = "v1.2.3"
_RUN_ID = 424242


def _needs(job: str) -> set[str]:
    match = re.search(r"(?m)^    needs: (?:\[(?P<many>[^\]]*)\]|(?P<one>\S+))$", job)
    if match is None:
        return set()
    if match.group("one"):
        return {match.group("one")}
    return {name.strip() for name in match.group("many").split(",")}


def _requires(text: str, job: str, dependency: str) -> bool:
    pending, seen = [job], set()
    while pending:
        current = pending.pop()
        for parent in _needs(_job_block(text, current)):
            if parent == dependency:
                return True
            if parent not in seen:
                seen.add(parent)
                pending.append(parent)
    return False


def _expected_legs() -> list[str]:
    admission = _job_block(_workflow_text(), "release-admission")
    match = re.search(r'EXPECTED_WHEEL_LEGS: "([^"]+)"', admission)
    assert match is not None
    return match.group(1).split()


def test_tag_and_release_are_created_only_after_release_admission() -> None:
    text = _workflow_text()
    release_job = _job_block(_release_tag_workflow_text(), "publish-release-tag")

    # R5a: only the post-admission job creates the immutable tag and release.
    creator = _job_block(text, "create-tag-and-release")
    assert '"$GITHUB_API_URL/repos/$GITHUB_REPOSITORY/git/refs"' in creator
    assert 'gh release create "v$RELEASE_VERSION"' in creator
    for name in ("verify-release", "sdist", "wheels", "reproducibility-record", "release-admission",
                 "release-assets", "publish-pypi"):
        job = _job_block(text, name)
        assert "gh release create" not in job and "/git/refs\"" not in job, name
    assert "gh release create" not in release_job and "/git/refs" not in release_job
    assert "contents: write" not in release_job

    assert _requires(text, "create-tag-and-release", "release-admission")
    assert _requires(text, "release-admission", "reproducibility-record")
    assert _requires(text, "release-admission", "verify-release")
    assert _requires(text, "release-admission", "dependency-gate")
    for sink in ("release-assets", "publish-pypi"):
        assert _requires(text, sink, "create-tag-and-release"), sink
        assert _requires(text, sink, "release-admission"), sink
    for name in ("release-admission", "create-tag-and-release", "release-assets", "publish-pypi"):
        job = _job_block(text, name)
        assert "always()" not in job and "continue-on-error" not in job and "|| true" not in job, name
        assert not re.search(r"(?m)^    if:", job), name

    # Condition 4: the admitted wheel legs are exactly the build matrix.
    wheels = _job_block(text, "wheels")
    matrix = re.findall(r"target: (\S+)\n(?:            .*\n)*?            python-version: \"([^\"]+)\"", wheels)
    assert sorted(f"{target}-py{version}" for target, version in matrix) == sorted(_expected_legs())


def test_central_full_set_gate_is_required_before_admission() -> None:
    workflow = _workflow_text()
    record = _job_block(workflow, "reproducibility-record")
    central = _job_block(workflow, "dependency-gate")
    admission = _job_block(workflow, "release-admission")
    assert "selected_wheel_filename: ${{ steps.bind-distributions.outputs.selected_wheel_filename }}" in record
    assert "selected_sdist_filename: ${{ steps.bind-distributions.outputs.selected_sdist_filename }}" in record
    assert "release-dependency-license-strix-gate.yml@b6cebb36dc11afe409a7fee8a3262827255c029c" in central
    assert "needs: [verify-release, reproducibility-record]" in central
    assert "secrets: inherit" in central
    assert "needs: [verify-release, reproducibility-record, dependency-gate]" in admission
    assert "full_set_verdict_artifact_id" in admission
    assert "full_set_verdict_artifact_digest" in admission
    assert "verify_release_full_set_verdict.py" in admission
    writer = 'with Path("admitted-manifest.tsv").open("x", encoding="utf-8")'
    for check in ("verify_full_set_verdict", "verify_runtime_dependency_coverage",
                  "verify_native_link_inventory", "verify_scope_declarations",
                  "verify_distribution_requirements", "verify_bundled_license_notices",
                  "verify_license_selection_report", "verify_cargo_graph_closure",
                  "verify_cargo_dependency_coverage"):
        assert admission.index(check) < admission.index(writer)


def test_admission_reexports_source_requirements_before_verifying_receipts(tmp_path: Path) -> None:
    admission = _job_block(_workflow_text(), "release-admission")
    step_name = "Recompute the complete source dependency export"
    assert admission.index("Download selected immutable IDs") < admission.index(step_name)
    assert admission.index(step_name) < admission.index("Admit exactly the verified bytes")
    assert 'version: "0.12.5"' in admission
    step = admission.split(f"- name: {step_name}\n", 1)[1].split("\n      - ", 1)[0]
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    assert "--project release-source --locked --offline --no-dev --no-emit-project --all-extras" in script
    assert "--no-header" in script

    binary = tmp_path / "uv"
    binary.write_text("#!/usr/bin/env python3\n"
                      "import pathlib, sys\n"
                      "pathlib.Path(sys.argv[-1]).write_bytes(b'numpy==2.5.1\\n')\n")
    binary.chmod(0o755)
    legs = _expected_legs()
    for leg in legs:
        folder = tmp_path / "downloaded" / f"repro-digest-{leg}"
        folder.mkdir(parents=True)
        (folder / f"{leg}.runtime-requirements.txt").write_bytes(
            b"# This file was autogenerated by uv via the following command:\n"
            b"#    uv export --locked --all-extras --output-file captured.txt\n"
            b"numpy==2.5.1\n"
        )
        if leg.startswith("universal2-"):
            intel = tmp_path / "downloaded" / f"repro-macos-x86-{leg}"
            intel.mkdir(parents=True)
            (intel / f"{leg}.runtime-requirements.txt").write_bytes(
                (folder / f"{leg}.runtime-requirements.txt").read_bytes())
    env = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}",
                        "RUNNER_TEMP": str(tmp_path), "EXPECTED_WHEEL_LEGS": " ".join(legs)}
    def run() -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env,
                              text=True, capture_output=True, check=False)

    assert run().returncode == 0
    (tmp_path / "downloaded" / f"repro-digest-{legs[0]}" /
     f"{legs[0]}.runtime-requirements.txt").write_bytes(
         b"# This file was autogenerated by uv via the following command:\n"
         b"#    uv export --locked --all-extras --output-file captured.txt\n"
         b""
     )
    failed = run()
    assert failed.returncode != 0
    assert "runtime requirements differ from release source export" in failed.stderr


def test_universal2_x86_runtime_capture_precedes_release_record() -> None:
    workflow = _workflow_text()
    job = _job_block(workflow, "macos-x86-runtime")
    record = _job_block(workflow, "reproducibility-record")
    assert "needs: [verify-release, wheels]" in job
    assert "runs-on: macos-15-intel" in job
    assert 'python-version: ["3.12", "3.13", "3.14"]' in job
    assert "architecture: x64" in job
    assert "ref: ${{ needs.verify-release.outputs.release_commit }}" not in job
    assert "ref: ${{ github.sha }}\n          path: trusted-control" in job
    assert "name: dist-wheel-${{ env.LEG }}" in job
    assert "name: repro-digest-${{ env.LEG }}" in job
    assert "python trusted-control/scripts/ci/capture_release_runtime.py \"runtime-proof/$LEG.tsv\" dist" in job
    assert "python scripts/ci/capture_release_runtime.py" not in job
    assert "name: repro-macos-x86-${{ env.LEG }}" in job
    assert "needs: [verify-release, sdist, wheels, macos-x86-runtime]" in record
    assert "runtime_variants.append(runtime)" in _job_block(workflow, "release-admission")


def _admission_fixture(root: Path) -> dict:
    import zipfile
    import json
    import runpy
    import zipfile
    legs = _expected_legs()
    platforms = {"x86_64-unknown-linux-gnu": "manylinux2014_x86_64",
                 "aarch64-unknown-linux-gnu": "manylinux2014_aarch64",
                 "universal2-apple-darwin": "macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2",
                 "x86_64-pc-windows-msvc": "win_amd64"}
    files = {}
    for leg in legs:
        target, version = leg.rsplit("-py", 1)
        cp = "cp" + version.replace(".", "")
        files[leg] = f"pkg-1.2.3-{cp}-{cp}-{platforms[target]}.whl"
    files["sdist"] = "pkg-1.2.3.tar.gz"
    payload = {leg: f"bytes of {name}".encode() for leg, name in files.items()}
    extension_member = "fast_mlsirm/_core.fixture.so"
    extension_bytes = b"synthetic extension member"
    dependency_archive_name = "numpy-2.5.1-py3-none-any.whl"
    dependency_buffer = io.BytesIO()
    with zipfile.ZipFile(dependency_buffer, "w") as archive:
        archive.writestr("numpy-2.5.1.dist-info/METADATA", "Name: numpy\nVersion: 2.5.1\n")
    dependency_archive_bytes = dependency_buffer.getvalue()
    for leg in legs:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            _, abi, platform = files[leg][:-4].rsplit("-", 3)[1:]
            tags = "".join(f"Tag: {abi}-{abi}-{part}\n" for part in platform.split("."))
            archive.writestr("pkg-1.2.3.dist-info/WHEEL", f"Wheel-Version: 1.0\n{tags}")
            archive.writestr("pkg-1.2.3.dist-info/METADATA", "Name: pkg\nVersion: 1.2.3\n")
            archive.writestr(extension_member, extension_bytes)
        payload[leg] = buffer.getvalue()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = b"Name: pkg\nVersion: 1.2.3\n"
        member = tarfile.TarInfo("pkg-1.2.3/PKG-INFO")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    payload["sdist"] = buffer.getvalue()
    sha = {leg: hashlib.sha256(data).hexdigest() for leg, data in payload.items()}
    def build_env(leg: str) -> str:
        target = leg.rsplit("-py", 1)[0] if leg != "sdist" else "sdist"
        if target == "sdist":
            return "runner:ubuntu/test/Linux/X64"
        if target.endswith("linux-gnu"):
            return "container:quay.io/pypa/fixture@sha256:" + "a" * 64
        if target == "x86_64-pc-windows-msvc":
            return "runner:windows/test/Windows/X64"
        return "runner:macos/test/macOS/ARM64"
    (root / "dist").mkdir(parents=True)
    for leg, name in files.items():
        (root / "dist" / name).write_bytes(payload[leg])
    (root / "record").mkdir()
    rows = "".join(
        f"{leg}\ttrue\tclean-target-repeat-same-env\t{sha[leg]}\t{sha[leg]}\t{files[leg]}\t{build_env(leg)}\n"
        for leg in sorted(files)
    )
    (root / "record" / "reproducibility-record.tsv").write_text(
        f"# release {_RELEASE_TAG} @ {_RELEASE_COMMIT}, SOURCE_DATE_EPOCH=1\n"
        "target\tbyte_verified\tverification\tsha256\trebuild_sha256\tfile\tbuild_env\n" + rows
    )
    source = root / "release-source"
    source.mkdir()
    (source / "uv.lock").write_text("fixture lock\n")
    (source / "pyproject.toml").write_text('[project]\nname = "fast-mlsirm"\nversion = "1.2.3"\n')
    crate = source / "crates/fast-mlsirm-py"
    crate.mkdir(parents=True)
    (crate / "Cargo.toml").write_text('[package]\nname = "fast-mlsirm-py"\nversion = "0.11.4"\n')
    (crate / "Cargo.lock").write_text('version = 4\n[[package]]\nname = "fast-mlsirm-py"\nversion = "0.11.4"\n[[package]]\nname = "mlsirm-core"\nversion = "0.11.4"\n')
    transport = runpy.run_path(str(REPO_ROOT / "scripts/ci/release_artifact_transport.py"))
    bundle_inventory = transport["bundle_inventory"]
    expected_maturin = transport["expected_maturin_binary_sha256"]
    for leg in files:
        folder = root / "scope-evidence" / f"repro-digest-{leg}"
        folder.mkdir(parents=True)
        row = next(line for line in rows.splitlines() if line.startswith(f"{leg}\t"))
        (folder / f"{leg}.tsv").write_text(row + "\n")
        inventory = bundle_inventory(root / "dist" / files[leg], leg, _RELEASE_COMMIT, build_env(leg))
        (folder / f"{leg}.bundle.json").write_text(json.dumps(inventory, sort_keys=True) + "\n")
        if leg != "sdist":
            requirements = folder / f"{leg}.runtime-requirements.txt"
            requirements.write_text("numpy==2.5.1 --hash=sha256:" + hashlib.sha256(dependency_archive_bytes).hexdigest() + "\n")
            target, version = leg.rsplit("-py", 1)
            system, machine = {
                "x86_64-unknown-linux-gnu": ("linux", "x86_64"),
                "aarch64-unknown-linux-gnu": ("linux", "aarch64"),
                "universal2-apple-darwin": ("darwin", "arm64"),
                "x86_64-pc-windows-msvc": ("win32", "AMD64"),
            }[target]
            before = [{"name": "numpy", "version": "2.5.1"}]
            (folder / dependency_archive_name).write_bytes(dependency_archive_bytes)
            runtime = {
                "schema_version": 1, "source_sha": _RELEASE_COMMIT, "leg": leg,
                "file": files[leg], "sha256": sha[leg], "build_env": build_env(leg),
                "uv_version": "uv 0.12.5", "python_version": version,
                "python_full_version": version + ".0",
                "implementation": "cpython", "sys_platform": system, "machine": machine,
                "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
                "uv_lock_sha256": hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest(),
                "locked_dependencies": before,
                "installed": [{"name": "fast-mlsirm", "version": "1.2.3"}, *before],
                "imported_extension": {"member": extension_member,
                                       "sha256": hashlib.sha256(extension_bytes).hexdigest()},
                "archives": [{"file": dependency_archive_name,
                              "size": len(dependency_archive_bytes),
                              "sha256": hashlib.sha256(dependency_archive_bytes).hexdigest(),
                              "name": "numpy", "version": "2.5.1"}],
            }
            (folder / f"{leg}.runtime.json").write_text(json.dumps(runtime, sort_keys=True) + "\n")
            (folder / f"{leg}.consumer.whl").write_bytes(payload[leg])
            metadata = {item["path"]: item["sha256"] for item in inventory["members"]
                        if item["path"].endswith((".dist-info/METADATA", ".dist-info/WHEEL"))}
            receipt = {"schema_version": 1, "source_sha": _RELEASE_COMMIT,
                       "leg": leg, "build_env": build_env(leg),
                       "sdist_file": files["sdist"], "sdist_sha256": sha["sdist"],
                       "file": files[leg], "published_sha256": sha[leg],
                       "consumer_sha256": sha[leg], "metadata_members": metadata,
                       "native_extension": {"member": extension_member,
                                            "sha256": hashlib.sha256(extension_bytes).hexdigest()}}
            receipt["installation"] = {key: runtime[key] for key in (
                "uv_version", "python_version", "implementation", "sys_platform", "machine",
                "requirements_sha256", "uv_lock_sha256", "locked_dependencies", "installed")}
            receipt["installation"]["imported_extension"] = receipt["native_extension"]
            (folder / f"{leg}.consumer.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
            if target == "universal2-apple-darwin":
                intel = root / "scope-evidence" / f"repro-macos-x86-{leg}"
                intel.mkdir()
                for filename in (f"{leg}.tsv", f"{leg}.runtime-requirements.txt",
                                 dependency_archive_name):
                    (intel / filename).write_bytes((folder / filename).read_bytes())
                runtime["machine"] = "x86_64"
                (intel / f"{leg}.runtime.json").write_text(json.dumps(runtime, sort_keys=True) + "\n")
        target, version = ("sdist", "3.12") if leg == "sdist" else leg.rsplit("-py", 1)
        targets = ([] if target == "sdist" else ["aarch64-apple-darwin", "x86_64-apple-darwin"]
                   if target == "universal2-apple-darwin" else [target])
        graph = [{"name": name, "version": "0.11.4", "source": None,
                  "checksum": None, "features": []}
                 for name in ("fast-mlsirm-py", "mlsirm-core")]
        snapshot = folder / f"{leg}.build-python.zip"
        with zipfile.ZipFile(snapshot, "w") as archive:
            archive.writestr("pip/pip/__init__.py", b"x")
        snapshot_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        for build_pass in ("first", "second"):
            build = {"schema_version": 1, "source_sha": _RELEASE_COMMIT, "leg": leg,
                     "pass": build_pass, "build_env": build_env(leg),
                     "cargo_lock_sha256": hashlib.sha256((crate / "Cargo.lock").read_bytes()).hexdigest(),
                     "pyproject_sha256": hashlib.sha256((source / "pyproject.toml").read_bytes()).hexdigest(),
                     "cargo_version": "cargo 1.90.0", "rustc_version": "rustc 1.90.0",
                     "maturin_version": "maturin 1.15.0",
                     "maturin_binary_sha256": expected_maturin(leg, build_env(leg)),
                     "python_version": f"Python {version}.0",
                     "python_packages": [{"name": "pip", "version": "25.2", "files": [
                         {"path": "pip/__init__.py", "size": 1,
                          "sha256": hashlib.sha256(b"x").hexdigest()}]}],
                     "python_snapshot_sha256": snapshot_sha,
                     "cargo_features": [] if target == "sdist" else ["pyo3/extension-module"],
                     "cargo_targets": {triple: graph for triple in targets}}
            (folder / f"{leg}.build-{build_pass}.json").write_text(json.dumps(build, sort_keys=True) + "\n")
    artifacts = [f"dist-wheel-{leg}" for leg in legs] + [
        "dist-sdist", "reproducibility-record", "release-dependency-sealed-evidence",
        "release-dependency-sealed-evidence--full-set-verdict",
    ] + [f"repro-digest-{leg}" for leg in files] + [
        f"repro-macos-x86-universal2-apple-darwin-py{version}"
        for version in ("3.12", "3.13", "3.14")]
    for name in ("release-dependency-sealed-evidence", "release-dependency-sealed-evidence--full-set-verdict"):
        bundle = root / "evidence" / name
        bundle.mkdir(parents=True)
        (bundle / "placeholder.txt").write_text("inert test artifact\n")
    listing = [{"id": index, "name": name, "workflow_run": {"id": _RUN_ID}, "expired": False, "digest": "sha256:" + "d" * 64}
               for index, name in enumerate(artifacts, 1)]
    return {"legs": legs, "files": files, "listing": listing,
            "build_env": {leg: build_env(leg) for leg in files}}


def test_build_scope_receipts_bind_wheel_lock_and_repeat(tmp_path: Path) -> None:
    import json
    import runpy
    import pytest

    fixture = _admission_fixture(tmp_path)
    leg = fixture["legs"][0]
    folder = tmp_path / "scope-evidence" / f"repro-digest-{leg}"
    first = json.loads((folder / f"{leg}.build-first.json").read_text())
    second = json.loads((folder / f"{leg}.build-second.json").read_text())
    verify = runpy.run_path(str(REPO_ROOT / "scripts/ci/release_artifact_transport.py"))["verify_build_scope"]
    row = {"target": leg, "build_env": fixture["build_env"][leg]}
    snapshot = folder / f"{leg}.build-python.zip"
    verify(first, second, row, tmp_path / "release-source", _RELEASE_COMMIT, snapshot)
    second["cargo_targets"][leg.rsplit("-py", 1)[0]][0]["version"] = "forged"
    with pytest.raises(ValueError, match="graph differs from selected lock"):
        verify(first, second, row, tmp_path / "release-source", _RELEASE_COMMIT, snapshot)
    second = json.loads((folder / f"{leg}.build-second.json").read_text())
    second["maturin_version"] = "maturin 1.14.1"
    with pytest.raises(ValueError, match="toolchain or leg"):
        verify(first, second, row, tmp_path / "release-source", _RELEASE_COMMIT, snapshot)
    second = json.loads((folder / f"{leg}.build-second.json").read_text())
    second["maturin_binary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="toolchain or leg"):
        verify(first, second, row, tmp_path / "release-source", _RELEASE_COMMIT, snapshot)
    second = json.loads((folder / f"{leg}.build-second.json").read_text())
    second["python_packages"][0]["files"][0]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="snapshot member hash differs"):
        verify(first, second, row, tmp_path / "release-source", _RELEASE_COMMIT, snapshot)
    second["python_packages"][0]["files"][0]["path"] = "../../unexpected\\file"
    with pytest.raises(ValueError, match="distribution file is malformed"):
        verify(first, second, row, tmp_path / "release-source", _RELEASE_COMMIT, snapshot)


def test_admission_recomputes_complete_cargo_graph(tmp_path: Path, monkeypatch) -> None:
    import json
    import runpy

    import pytest

    source = tmp_path / "release-source"
    crate = source / "crates/fast-mlsirm-py"
    crate.mkdir(parents=True)
    (crate / "Cargo.lock").write_text(
        'version = 4\n[[package]]\nname = "fast-mlsirm-py"\nversion = "0.11.4"\n'
        '[[package]]\nname = "mlsirm-core"\nversion = "0.11.4"\n')
    graph = [{"name": name, "version": "0.11.4", "source": None,
              "checksum": None, "features": []}
             for name in ("fast-mlsirm-py", "mlsirm-core")]
    metadata = {"packages": [
        {"id": name, "name": name, "version": "0.11.4", "source": None}
        for name in ("fast-mlsirm-py", "mlsirm-core")],
        "resolve": {"root": "fast-mlsirm-py", "nodes": [
            {"id": name, "features": []}
            for name in ("fast-mlsirm-py", "mlsirm-core")]}}
    declarations = {"crates/fast-mlsirm-py/Cargo.lock": (crate / "Cargo.lock").read_bytes(),
                    "crates/fast-mlsirm-py/Cargo.toml": b'[package]\nname = "fast-mlsirm-py"\n',
                    "crates/mlsirm-core/Cargo.toml": b'[package]\nname = "mlsirm-core"\n'}
    for relative, content in declarations.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    commands = []
    def output(command, *, text=False, cwd=None):
        commands.append(command)
        if command[0] != "git":
            return json.dumps(metadata)
        if "rev-parse" in command:
            return _RELEASE_COMMIT + "\n"
        if "ls-tree" in command:
            return "\n".join(declarations) + "\n"
        assert "show" in command
        return declarations[command[-1].split(":", 1)[1]]
    transport = runpy.run_path(str(REPO_ROOT / "scripts/ci/release_artifact_transport.py"))
    monkeypatch.setattr(transport["subprocess"], "check_output", output)
    receipts = []
    for leg in ["sdist", *_expected_legs()]:
        target = leg.rsplit("-py", 1)[0]
        triples = ([] if leg == "sdist" else
                   ["aarch64-apple-darwin", "x86_64-apple-darwin"]
                   if target == "universal2-apple-darwin" else [target])
        receipts.append({"leg": leg, "cargo_targets": {triple: graph for triple in triples}})
    verify = transport["verify_cargo_graph_closure"]
    verify(receipts, source, _RELEASE_COMMIT)
    assert len([command for command in commands if command[0] == "cargo"]) == 5
    receipts[1]["cargo_targets"][receipts[1]["leg"].rsplit("-py", 1)[0]] = graph[:1]
    with pytest.raises(ValueError, match="differs from selected source closure"):
        verify(receipts, source, _RELEASE_COMMIT)
    admission = _job_block(_workflow_text(), "release-admission")
    assert "transport.verify_cargo_graph_closure(build_graph_receipts, Path(\"release-source\"), commit)" in admission


def test_runtime_rejects_unaccounted_native_member(tmp_path: Path) -> None:
    import json
    import runpy
    import zipfile
    import pytest

    fixture = _admission_fixture(tmp_path)
    leg = fixture["legs"][0]
    folder = tmp_path / "scope-evidence" / f"repro-digest-{leg}"
    distribution = tmp_path / "dist" / fixture["files"][leg]
    with zipfile.ZipFile(distribution, "a") as archive:
        archive.writestr("fast_mlsirm/hidden.dat", b"MZhidden native code")
    transport = runpy.run_path(str(REPO_ROOT / "scripts/ci/release_artifact_transport.py"))
    runtime = json.loads((folder / f"{leg}.runtime.json").read_text())
    bundle = transport["bundle_inventory"](
        distribution, leg, _RELEASE_COMMIT, fixture["build_env"][leg],
    )
    row = {"target": leg, "file": fixture["files"][leg],
           "sha256": runtime["sha256"], "build_env": fixture["build_env"][leg]}
    with pytest.raises(ValueError, match="unaccounted bundled native binary"):
        transport["verify_runtime_inventory"](
            runtime, folder / f"{leg}.runtime-requirements.txt", row,
            tmp_path / "release-source", _RELEASE_COMMIT, bundle, distribution,
            {path.name: path for path in folder.iterdir()},
        )


def test_sdist_build_receipts_report_no_compiled_cargo_graph(tmp_path: Path) -> None:
    import json
    import runpy
    import pytest

    fixture = _admission_fixture(tmp_path)
    folder = tmp_path / "scope-evidence/repro-digest-sdist"
    first = json.loads((folder / "sdist.build-first.json").read_text())
    second = json.loads((folder / "sdist.build-second.json").read_text())
    verify = runpy.run_path(str(REPO_ROOT / "scripts/ci/release_artifact_transport.py"))["verify_build_scope"]
    row = {"target": "sdist", "build_env": fixture["build_env"]["sdist"]}
    snapshot = folder / "sdist.build-python.zip"
    verify(first, second, row, tmp_path / "release-source", _RELEASE_COMMIT, snapshot)
    second["cargo_targets"] = {"x86_64-unknown-linux-gnu": []}
    with pytest.raises(ValueError, match="toolchain or leg"):
        verify(first, second, row, tmp_path / "release-source", _RELEASE_COMMIT, snapshot)


def test_sdist_capture_records_runner_tools_without_cargo_metadata(tmp_path: Path, monkeypatch) -> None:
    import runpy

    _admission_fixture(tmp_path)
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts/ci"))
    capture = runpy.run_path(str(REPO_ROOT / "scripts/ci/capture_release_build_scope.py"))["capture"]

    def run(*args: str) -> str:
        if args[:3] == ("git", "-C", str(tmp_path / "release-source")):
            return _RELEASE_COMMIT
        if args == ("python", "--version"):
            return "Python 3.12.0"
        if args[:2] == ("python", "-c"):
            return sys.executable
        if args == ("cargo", "--version"):
            return "cargo 1.90.0"
        if args == ("rustc", "--version"):
            return "rustc 1.90.0"
        if args == ("maturin", "--version"):
            return "maturin 1.15.0"
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setitem(capture.__globals__, "_run", run)
    packages = [{"name": "pip", "version": "25.2", "files": [
        {"path": "pip/__init__.py", "size": 1, "sha256": "a" * 64}]}]
    monkeypatch.setitem(capture.__globals__, "_python_packages_with_files", lambda: packages)
    monkeypatch.setitem(capture.__globals__, "expected_maturin_binary_sha256", lambda *_: "asset-hash")
    monkeypatch.setitem(capture.__globals__, "hash_file", lambda _: "asset-hash")
    monkeypatch.setitem(capture.__globals__, "write_python_snapshot", lambda *_: "asset-hash")
    monkeypatch.setattr(capture.__globals__["shutil"], "which", lambda _: "/tmp/maturin")
    from types import SimpleNamespace

    monkeypatch.setitem(capture.__globals__, "sys", SimpleNamespace(
        executable=sys.executable, prefix=sys.prefix, platform="linux",
        implementation=SimpleNamespace(name="cpython")))
    monkeypatch.setitem(capture.__globals__, "platform", SimpleNamespace(machine=lambda: "x86_64"))
    receipt = capture(tmp_path / "release-source", {
        "CARGO_RELEASE_LEG": "sdist", "CARGO_RELEASE_SHA": _RELEASE_COMMIT,
        "CARGO_BUILD_PASS": "first", "CARGO_BUILD_PYTHON": "python",
        "ImageOS": "ubuntu", "ImageVersion": "test",
        "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64",
    })
    assert receipt["build_env"] == "runner:ubuntu/test/Linux/X64"
    assert receipt["cargo_targets"] == {}
    assert receipt["cargo_features"] == []
    assert receipt["python_packages"] == packages
    assert receipt["python_snapshot_sha256"] == "asset-hash"


def test_build_package_inventory_hashes_installed_files(tmp_path: Path, monkeypatch) -> None:
    import runpy
    from pathlib import PurePosixPath
    import pytest

    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts/ci"))
    inventory = runpy.run_path(str(REPO_ROOT / "scripts/ci/capture_release_build_scope.py"))["_python_packages_with_files"]
    prefix = tmp_path / "python"
    target = prefix / "site-packages/pkg.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"installed bytes")

    class Distribution:
        metadata = {"Name": "My_Pkg"}
        version = "1.0"
        files = [PurePosixPath("site-packages/pkg.py")]

        def locate_file(self, member):
            return prefix / member

    monkeypatch.setattr(inventory.__globals__["sys"], "prefix", str(prefix))
    monkeypatch.setitem(inventory.__globals__, "distributions", lambda: [Distribution()])
    monkeypatch.setattr(inventory.__globals__["sysconfig"], "get_paths",
                        lambda: {"purelib": str(target.parent), "platlib": str(target.parent)})
    assert inventory() == [{"name": "my-pkg", "version": "1.0", "files": [
        {"path": "site-packages/pkg.py", "size": len(b"installed bytes"),
         "sha256": hashlib.sha256(b"installed bytes").hexdigest()}]}]
    orphan = target.with_name("unlisted.py")
    orphan.write_bytes(b"unreviewed bytes")
    with pytest.raises(ValueError, match="omitted from RECORD"):
        inventory()
    orphan.unlink()
    orphan.symlink_to(target)
    with pytest.raises(ValueError, match="contains a symlink"):
        inventory()
    orphan.unlink()
    Distribution.files = [PurePosixPath("../outside.py")]
    (tmp_path / "outside.py").write_bytes(b"foreign bytes")
    with pytest.raises(ValueError, match="unsafe"):
        inventory()


def _run_admission(root: Path, step: str, listing: list[dict] | None = None) -> subprocess.CompletedProcess:
    import json
    import runpy
    import zipfile

    selected = []

    if listing is not None:
        (root / "run-artifacts.jsonl").write_text("".join(json.dumps(item) + "\n" for item in listing))
    if step == _BYTES_STEP and not (root / "downloaded").exists():
        # Exercise the actual transport with synthetic ZIPs, not a fake receipt.
        artifacts = [("reproducibility-record", list((root / "record").iterdir()))]
        record = (root / "record/reproducibility-record.tsv").read_text().splitlines()
        for line in record[2:]:
            fields = line.split("\t")
            name = "dist-sdist" if fields[0] == "sdist" else "dist-wheel-" + fields[0]
            path = root / "dist" / fields[5]
            # Missing files still reach the admission check as an empty mismatch.
            artifacts.append((name, [path] if path.exists() else []))
        artifacts += [(p.name, list(p.iterdir())) for p in (root / "evidence").iterdir()]
        artifacts += [(p.name, list(p.iterdir())) for p in (root / "scope-evidence").iterdir()]
        recorded_names = {line.split("\t")[5] for line in record[2:]}
        extras = [p for p in (root / "dist").iterdir() if p.name not in recorded_names]
        for name, paths in artifacts:
            if name == "dist-sdist":
                paths.extend(extras)
        selected, archives = [], {}
        # A duplicated target is diagnosed by record admission, not this test's
        # archive constructor; no metadata or record content is repaired.
        seen = set()
        for name, paths in artifacts:
            if name in seen:
                continue
            seen.add(name)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as z:
                for path in paths:
                    if path.is_file():
                        z.writestr(path.name, path.read_bytes())
                if not paths:
                    z.writestr("missing-member", b"missing")
            index = len(selected) + 1
            archives[index] = buffer.getvalue()
            selected.append({"id": index, "name": name, "digest": "sha256:" + hashlib.sha256(buffer.getvalue()).hexdigest()})
        by_name = {item["name"]: item for item in selected}
        distributions = []
        for line in record[2:]:
            leg, _, _, sha, _, filename, _ = line.split("\t")
            name = "dist-sdist" if leg == "sdist" else f"dist-wheel-{leg}"
            artifact = by_name[name]
            distributions.append({"leg": leg, "file": filename, "sha256": sha,
                                  "artifact_id": artifact["id"], "artifact_name": name,
                                  "artifact_digest": artifact["digest"]})
        identity = {"source_repository": "owner/repo", "source_sha": _RELEASE_COMMIT,
                    "control_sha": "d" * 40, "run_id": _RUN_ID, "run_attempt": 2}
        manifest = {"schema_version": 1, **identity, "distributions": distributions}
        scope_set = {"schema_version": 1, **identity, "evidence": [
            {"leg": leg, "artifact_id": by_name[f"repro-digest-{leg}"]["id"],
             "artifact_name": f"repro-digest-{leg}",
             "artifact_digest": by_name[f"repro-digest-{leg}"]["digest"]}
            for leg in sorted([*_expected_legs(), "sdist"])
        ]}
        runtime_variants = [{"leg": leg, "arch": "x86_64",
                             "artifact_id": by_name[f"repro-macos-x86-{leg}"]["id"],
                             "artifact_name": f"repro-macos-x86-{leg}",
                             "artifact_digest": by_name[f"repro-macos-x86-{leg}"]["digest"]}
                            for leg in sorted(_expected_legs()) if leg.startswith("universal2-")]
        record_artifact = by_name["reproducibility-record"]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for path in (root / "record").iterdir():
                archive.writestr(path.name, path.read_bytes())
            archive.writestr("release-gate-distribution-set.json", json.dumps(manifest))
            archive.writestr("release-scope-evidence-set.json", json.dumps(scope_set))
            if not (root / "record/release-scope-identities.json").exists():
                archive.writestr("release-scope-identities.json", "[]")
        archives[record_artifact["id"]] = buffer.getvalue()
        record_artifact["digest"] = "sha256:" + hashlib.sha256(buffer.getvalue()).hexdigest()
        binding = {"key": "pypi/numpy@2.5.1", "name": "release-strix-binding-a2-"
                   + hashlib.sha256(b"pypi/numpy@2.5.1").hexdigest(),
                   "id": 1000, "digest": "sha256:" + "b" * 64}
        dependency_sha = hashlib.sha256((root / "scope-evidence" /
            f"repro-digest-{_expected_legs()[0]}" / "numpy-2.5.1-py3-none-any.whl").read_bytes()).hexdigest()
        archive_key = f"{binding['key']}/sha256/{dependency_sha}"
        archive_binding = {"key": archive_key, "name": "release-strix-binding-a2-"
                           + hashlib.sha256(archive_key.encode()).hexdigest(),
                           "id": 1001, "digest": "sha256:" + "c" * 64}
        build_receipt = json.loads((root / "scope-evidence" / "repro-digest-sdist" /
                                    "sdist.build-first.json").read_text())
        build_package = build_receipt["python_packages"][0]
        build_sha = hashlib.sha256(json.dumps(build_package, sort_keys=True,
                                              separators=(",", ":")).encode()).hexdigest()
        build_key = f"pypi/pip@25.2/sha256/{build_sha}"
        build_binding = {"key": build_key, "name": "release-strix-binding-a2-"
                         + hashlib.sha256(build_key.encode()).hexdigest(),
                         "id": 1002, "digest": "sha256:" + "e" * 64}
        build_fixture = {"id": build_key,
                         "dependency": {"ecosystem": "pypi", "name": "pip", "version": "25.2",
                                        "source_sha256": build_sha}}
        build_fixture_sha = hashlib.sha256(json.dumps(build_fixture, sort_keys=True,
                                                      separators=(",", ":")).encode()).hexdigest()
        build_legs = sorted([*_expected_legs(), "sdist"])
        build_snapshots = {leg: json.loads((root / "scope-evidence" / f"repro-digest-{leg}" /
                                           f"{leg}.build-first.json").read_text())["python_snapshot_sha256"]
                           for leg in build_legs}
        fixture = {"id": archive_key,
                   "dependency": {"ecosystem": "pypi", "name": "numpy", "version": "2.5.1",
                                  "source_sha256": dependency_sha}}
        fixture_sha = hashlib.sha256(json.dumps(fixture, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
        tool_rows = {}
        for leg in build_legs:
            tool_receipt = json.loads((root / "scope-evidence" / f"repro-digest-{leg}" /
                                       f"{leg}.build-first.json").read_text())
            tool_sha = tool_receipt["maturin_binary_sha256"]
            tool_key = f"github-release/maturin@1.15.0/sha256/{tool_sha}"
            tool = tool_rows.setdefault(tool_key, {"key": tool_key,
                "package_key": "github-release/maturin@1.15.0", "name": "maturin",
                "version": "1.15.0", "source_sha256": tool_sha, "license": "Apache-2.0",
                "legs": [], "build_envs": {}, "fixture": {"id": tool_key,
                    "dependency": {"ecosystem": "github-release", "name": "maturin",
                                   "version": "1.15.0", "source_sha256": tool_sha}}})
            tool["legs"].append(leg)
            tool["build_envs"][leg] = tool_receipt["build_env"]
        for tool in tool_rows.values():
            tool["fixture_sha256"] = hashlib.sha256(json.dumps(tool["fixture"], sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
        tool_bindings = [{"key": key, "name": "release-strix-binding-a2-"
                          + hashlib.sha256(key.encode()).hexdigest(), "id": 1100 + index,
                          "digest": "sha256:" + "f" * 64}
                         for index, key in enumerate(sorted(tool_rows))]
        archive_report = {"schema": "cwl.release-runtime-archive-licenses/3", "archives": [
            {"key": archive_key, "package_key": binding["key"], "name": "numpy",
             "version": "2.5.1", "source_sha256": dependency_sha,
             "license": "BSD-3-Clause", "fixture": fixture,
             "fixture_sha256": fixture_sha, "legs": _expected_legs()}],
            "build_packages": [{"key": build_key, "package_key": "pypi/pip@25.2",
                                "name": "pip", "version": "25.2", "source_sha256": build_sha,
                                "license": "MIT", "fixture": build_fixture,
                                "fixture_sha256": build_fixture_sha, "legs": build_legs,
                                "snapshots": build_snapshots}],
            "build_tools": list(tool_rows.values())}
        archive_report_bytes = (json.dumps(archive_report, sort_keys=True) + "\n").encode()
        verdict_artifact = by_name["release-dependency-sealed-evidence--full-set-verdict"]
        report = {"schema": "cwl.release-dependency-gate/1", "result": "PASS",
                  "stage": "full", "source_repository": "owner/repo", "source_sha": _RELEASE_COMMIT,
                  "failures": [], "dependency_count": 1, "license_selections_sha256": "",
                  "dependencies": [{"key": binding["key"], "ecosystem": "pypi", "name": "numpy",
                                    "version": "2.5.1", "license": "BSD-3-Clause",
                                    "source_sha256": dependency_sha,
                                    "fixture_sha256": "c" * 64}],
                  "runtime_archive_reviews": [{"key": archive_key, "package_key": binding["key"],
                                               "source_sha256": dependency_sha,
                                               "license": "BSD-3-Clause", "fixture_sha256": fixture_sha,
                                               "legs": _expected_legs()}],
                  "build_package_reviews": [{"key": build_key, "package_key": "pypi/pip@25.2",
                                             "source_sha256": build_sha, "license": "MIT",
                                             "fixture_sha256": build_fixture_sha, "legs": build_legs}],
                  "build_tool_reviews": [{field: tool[field] for field in
                      ("key", "package_key", "source_sha256", "license", "fixture_sha256", "legs")}
                      for tool in tool_rows.values()]}
        report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        analyzer = {"path": "/usr/lib/llvm-18/bin/llvm-readobj", "version": "18.1.3",
                    "sha256": "d" * 64}
        native = {"schema": "cwl.release-native-links/2", "source_sha": _RELEASE_COMMIT,
                  "analyzer": analyzer,
                  "wheels": []}
        for distribution in distributions:
            leg = distribution["leg"]
            if leg == "sdist":
                continue
            target = leg.rsplit("-py", 1)[0]
            arches = ["aarch64", "x86_64"] if target == "universal2-apple-darwin" else [
                "aarch64" if target == "aarch64-unknown-linux-gnu" else "x86_64"]
            native["wheels"].append({"leg": leg, "file": distribution["file"],
                                     "sha256": distribution["sha256"],
                                     "member": "fast_mlsirm/_core.fixture.so",
                                     "member_sha256": hashlib.sha256(b"synthetic extension member").hexdigest(),
                                     "links": [{"arch": arch, "format": "fixture", "needed": [],
                                                "reviews": []}
                                               for arch in arches]})
        native["wheels"].sort(key=lambda row: row["leg"])
        native_bytes = (json.dumps(native, sort_keys=True) + "\n").encode()
        verdict = {"schema": "cwl.release-full-set-verdict/1", "result": "PASS", **identity,
                   "record_artifact_id": record_artifact["id"],
                   "record_artifact_digest": record_artifact["digest"],
                   "distributions": distributions, "binding_artifacts": [binding],
                   "scope_evidence": scope_set["evidence"],
                   "runtime_variants": runtime_variants,
                   "runtime_archive_binding_artifacts": [archive_binding],
                   "build_package_binding_artifacts": [build_binding],
                   "build_tool_binding_artifacts": tool_bindings,
                   "runtime_archive_license_sha256": hashlib.sha256(archive_report_bytes).hexdigest(),
                   "native_links_sha256": hashlib.sha256(native_bytes).hexdigest(),
                   "native_link_analyzer": analyzer,
                   "gate_report_sha256": hashlib.sha256(report_bytes).hexdigest()}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("full-set-verdict.json", json.dumps(verdict))
            archive.writestr("gate-report.json", report_bytes)
            archive.writestr("runtime-archive-license-report.json", archive_report_bytes)
            archive.writestr("release-native-links.json", native_bytes)
        archives[verdict_artifact["id"]] = buffer.getvalue()
        verdict_artifact["digest"] = "sha256:" + hashlib.sha256(buffer.getvalue()).hexdigest()
        module = runpy.run_path(str(REPO_ROOT / "scripts/ci/release_artifact_transport.py"))
        receipt = module["materialize"](selected, "owner/repo", root / "downloaded", lambda repo, index, output: output.write(archives[index]))
        (root / "selected-artifacts.json").write_text(json.dumps(selected))
        (root / "transport-receipt.json").write_text(json.dumps(receipt))
        (root / "run-artifacts.jsonl").write_text("".join(json.dumps({
            **item, "workflow_run": {"id": _RUN_ID, "head_sha": "d" * 40},
            "created_at": "2026-09-26T12:01:00Z", "expired": False,
        }) + "\n" for item in selected + [binding, archive_binding, build_binding, *tool_bindings]))
        (root / "run-attempt.json").write_text(json.dumps({
            "id": _RUN_ID, "run_attempt": 2, "head_sha": "d" * 40,
            "run_started_at": "2026-09-26T12:00:00Z",
        }))
        (root / "trusted-control").symlink_to(REPO_ROOT, target_is_directory=True)
        if os.environ.get("CWL_GATE_FIXTURE_ROOT"):
            (root / "trusted-gate").symlink_to(os.environ["CWL_GATE_FIXTURE_ROOT"], target_is_directory=True)
    if step == _BYTES_STEP and not selected:
        selected = json.loads((root / "selected-artifacts.json").read_text())
    script = _step_python(_job_block(_workflow_text(), "release-admission"), step)
    env = {**os.environ, "EXPECTED_WHEEL_LEGS": " ".join(_expected_legs()), "RUN_ID": str(_RUN_ID),
           "RELEASE_COMMIT": _RELEASE_COMMIT, "RELEASE_TAG": _RELEASE_TAG, "REPOSITORY": "owner/repo",
           "VERDICT_NAME": "release-dependency-sealed-evidence--full-set-verdict",
           "SEALED_NAME": "release-dependency-sealed-evidence",
           "RECORD_ID": "14", "VERDICT_ID": "16", "SEALED_ID": "15",
           "RECORD_DIGEST": "sha256:" + "d" * 64,
           "VERDICT_DIGEST": "sha256:" + "d" * 64,
           "SEALED_DIGEST": "sha256:" + "d" * 64, "GITHUB_SHA": "d" * 40,
           "GITHUB_RUN_ATTEMPT": "2"}
    if step == _BYTES_STEP:
        by_name = {item["name"]: item for item in selected}
        env.update(RECORD_ID=str(by_name["reproducibility-record"]["id"]),
                   RECORD_DIGEST=by_name["reproducibility-record"]["digest"],
                   VERDICT_ID=str(by_name[env["VERDICT_NAME"]]["id"]),
                   VERDICT_DIGEST=by_name[env["VERDICT_NAME"]]["digest"],
                   SEALED_ID=str(by_name[env["SEALED_NAME"]]["id"]),
                   SEALED_DIGEST=by_name[env["SEALED_NAME"]]["digest"])
    return subprocess.run([sys.executable, "-c", script], cwd=root, env=env, capture_output=True, text=True)


_SET_STEP = "Require the exact same-run artifact set"
_BYTES_STEP = "Admit exactly the verified bytes of release_commit"


def test_distribution_set_manifest_binds_exact_same_run_bytes(tmp_path: Path) -> None:
    import json

    job = _job_block(_workflow_text(), "reproducibility-record")
    assert "actions: read" in job
    assert "distribution_set_artifact_id: ${{ steps.record-upload.outputs.artifact-id }}" in job
    assert "release-gate-distribution-set.json" in job
    script = _step_python(job, "Bind all verified distribution bytes to immutable artifact IDs")
    base_env = {**os.environ, "EXPECTED_WHEEL_LEGS": " ".join(_expected_legs()),
                "RELEASE_COMMIT": _RELEASE_COMMIT, "RELEASE_TAG": _RELEASE_TAG,
                "GITHUB_REPOSITORY": "owner/repo", "GITHUB_RUN_ID": str(_RUN_ID),
                "GITHUB_RUN_ATTEMPT": "2", "GITHUB_SHA": "d" * 40,
                "GITHUB_OUTPUT": str(tmp_path / "producer-output.txt")}

    def run(name: str, mutate=None) -> subprocess.CompletedProcess[str]:
        root = tmp_path / name
        fixture = _admission_fixture(root)
        (root / "reproducibility-record.tsv").write_bytes(
            (root / "record/reproducibility-record.tsv").read_bytes()
        )
        listing = [dict(item, workflow_run={"id": _RUN_ID, "head_sha": "d" * 40},
                        created_at="2026-09-26T12:01:00Z") for item in fixture["listing"]]
        if mutate is not None:
            listing = mutate(root, listing)
        (root / "run-attempt.json").write_text(json.dumps({
            "id": _RUN_ID, "run_attempt": 2, "head_sha": "d" * 40,
            "run_started_at": "2026-09-26T12:00:00Z",
        }), encoding="utf-8")
        (root / "run-artifacts.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in listing), encoding="utf-8"
        )
        return subprocess.run([sys.executable, "-c", script], cwd=root,
                              env=base_env, capture_output=True, text=True)

    ok = run("ok")
    assert ok.returncode == 0, ok.stderr
    manifest = json.loads((tmp_path / "ok/release-gate-distribution-set.json").read_text())
    assert manifest["source_sha"] == _RELEASE_COMMIT
    assert manifest["control_sha"] == "d" * 40
    assert (manifest["run_id"], manifest["run_attempt"]) == (_RUN_ID, 2)
    assert len(manifest["distributions"]) == 13
    assert len({row["artifact_id"] for row in manifest["distributions"]}) == 13
    scope_set = json.loads((tmp_path / "ok/release-scope-evidence-set.json").read_text())
    assert len(scope_set["evidence"]) == 13
    assert not {row["artifact_id"] for row in scope_set["evidence"]} & {row["artifact_id"] for row in manifest["distributions"]}

    def tamper_sdist(root: Path, items: list[dict]) -> list[dict]:
        (root / "dist/pkg-1.2.3.tar.gz").write_bytes(b"altered")
        return items

    cases = [
        ("missing", lambda root, items: [a for a in items if a["name"] != "dist-sdist"]),
        ("other-run", lambda root, items: [dict(a, workflow_run={"id": 1}) if a["name"] == "dist-sdist" else a for a in items]),
        ("earlier-attempt", lambda root, items: [dict(a, created_at="2026-09-26T11:59:59Z") if a["name"] == "dist-sdist" else a for a in items]),
        ("other-control-head", lambda root, items: [dict(a, workflow_run={"id": _RUN_ID, "head_sha": "e" * 40}) if a["name"] == "dist-sdist" else a for a in items]),
        ("duplicate-id", lambda root, items: [dict(a, id=1) if a["name"] == "dist-sdist" else a for a in items]),
        ("extra", lambda root, items: items + [dict(items[0], name="dist-wheel-extra")]),
        ("tampered", tamper_sdist),
        ("missing-scope", lambda root, items: [a for a in items if a["name"] != "repro-digest-sdist"]),
        ("stale-scope", lambda root, items: [dict(a, created_at="2026-09-26T11:59:59Z") if a["name"] == "repro-digest-sdist" else a for a in items]),
    ]
    for name, mutate in cases:
        result = run(name, mutate)
        assert result.returncode != 0, (name, result.stderr)
        assert not (tmp_path / name / "release-gate-distribution-set.json").exists()


def test_release_admission_admits_only_verified_same_run_bytes(tmp_path: Path) -> None:
    import json
    import zipfile

    fixture = _admission_fixture(tmp_path / "ok")
    ok_set = _run_admission(tmp_path / "ok", _SET_STEP, fixture["listing"])
    assert ok_set.returncode == 0, ok_set.stderr
    ok_bytes = _run_admission(tmp_path / "ok", _BYTES_STEP)
    # The synthetic verdict passes; the deliberately empty scope inventory still refuses.
    assert ok_bytes.returncode != 0 and "scope identity set missing" in ok_bytes.stderr

    assert not (tmp_path / "ok" / "admitted-manifest.tsv").exists()

    def refuse_set(name: str, mutate, expected: str) -> None:
        root = tmp_path / name
        listing = _admission_fixture(root)["listing"]
        result = _run_admission(root, _SET_STEP, mutate(listing))
        assert result.returncode != 0 and expected in result.stderr, (name, result.stderr)

    refuse_set("no-evidence", lambda l: [a for a in l if a["name"] not in (
        "release-dependency-sealed-evidence", "release-dependency-sealed-evidence--full-set-verdict")],
        "central release evidence missing")
    refuse_set("missing-verdict", lambda l: [a for a in l if not a["name"].endswith("--full-set-verdict")],
        "central release evidence missing")
    refuse_set("other-run", lambda l: [dict(a, workflow_run={"id": 1}) if a["name"] == "dist-sdist" else a for a in l],
               "dist-sdist: not produced by run")
    refuse_set("expired", lambda l: [dict(a, expired=True) if a["name"] == "reproducibility-record" else a for a in l],
               "reproducibility-record: expired")
    refuse_set("no-digest", lambda l: [dict(a, digest=None) if a["name"].startswith("dist-wheel-") else a for a in l],
               "no sha256 artifact digest")
    refuse_set("duplicate", lambda l: l + [l[0]], "duplicate artifact name")
    refuse_set("extra-dist", lambda l: l + [dict(l[0], name="dist-wheel-extra")], "unexpected publishable")
    refuse_set("missing-dist", lambda l: [a for a in l if a["name"] != "dist-sdist"], "dist-sdist: not uploaded")
    refuse_set("missing-scope", lambda l: [a for a in l if a["name"] != "repro-digest-sdist"], "repro-digest-sdist: not uploaded")
    intel = "universal2-apple-darwin-py3.12"
    intel_name = f"repro-macos-x86-{intel}"
    refuse_set("missing-intel", lambda l: [a for a in l if a["name"] != intel_name],
               f"{intel_name}: not uploaded")
    refuse_set("foreign-intel", lambda l: [dict(a, workflow_run={"id": 1})
                  if a["name"] == intel_name else a for a in l],
               f"{intel_name}: not produced by run")
    refuse_set("missing-ids", lambda l: [{k: v for k, v in a.items() if k != "id"} for a in l], "immutable artifact ID")
    refuse_set("duplicate-ids", lambda l: [dict(a, id=1) for a in l], "immutable artifact ID")

    def refuse_bytes(name: str, mutate, expected: str) -> None:
        root = tmp_path / name
        fixture = _admission_fixture(root)
        mutate(root, fixture)
        result = _run_admission(root, _BYTES_STEP)
        assert result.returncode != 0 and expected in result.stderr, (name, result.stderr)

    first = _expected_legs()[0]

    refuse_bytes("extra-dist-file", lambda r, f: (r / "dist" / "extra.whl").write_bytes(b"x"),
                 "distribution files differ from the record")
    refuse_bytes("changed-dist-bytes", lambda r, f: (r / "dist" / f["files"][first]).write_bytes(b"other"),
                 "distribution files differ from the record")
    refuse_bytes("changed-intel-archive", lambda r, f: (r / "scope-evidence" / intel_name /
                 "numpy-2.5.1-py3-none-any.whl").write_bytes(b"forged"),
                 "runtime archive bytes differ from receipt")
    refuse_bytes("missing-intel-runtime", lambda r, f: (r / "scope-evidence" / intel_name /
                 f"{intel}.runtime.json").unlink(),
                 "Intel runtime evidence is incomplete")

    def record_mutation(root: Path, old: str, new: str) -> None:
        path = root / "record" / "reproducibility-record.tsv"
        path.write_text(path.read_text().replace(old, new, 1))

    refuse_bytes("record-commit", lambda r, f: record_mutation(r, _RELEASE_COMMIT, "f" * 40),
                 "reproducibility record is not bound")
    refuse_bytes("record-unverified", lambda r, f: record_mutation(r, "\ttrue\t", "\tfalse\t"),
                 "record row is not byte-verified")
    refuse_bytes(
        "changed-scope-row",
        lambda r, f: (r / "scope-evidence" / f"repro-digest-{first}" / f"{first}.tsv").write_text("forged\n"),
        "scope evidence row differs from reproducibility record",
    )
    refuse_bytes(
        "extra-scope-member",
        lambda r, f: (r / "scope-evidence" / f"repro-digest-{first}" / "extra.json").write_text("{}"),
        "scope evidence artifact members differ from build output",
    )
    def snapshot(root: Path) -> Path:
        return root / "scope-evidence" / f"repro-digest-{first}" / f"{first}.build-python.zip"
    refuse_bytes("missing-build-snapshot", lambda r, f: snapshot(r).unlink(),
                 "scope evidence artifact members differ from build output")
    def change_build_snapshot(root: Path, fixture: dict) -> None:
        with zipfile.ZipFile(snapshot(root), "a") as archive:
            archive.writestr("extra.txt", b"changed")
    refuse_bytes("changed-build-snapshot", change_build_snapshot,
                 "build snapshot members differ from installed files")
    refuse_bytes(
        "missing-consumer",
        lambda r, f: (r / "scope-evidence" / f"repro-digest-{first}" / f"{first}.consumer.json").unlink(),
        "scope evidence artifact members differ from build output",
    )
    def change_consumer(root: Path, fixture: dict) -> None:
        path = root / "scope-evidence" / f"repro-digest-{first}" / f"{first}.consumer.whl"
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("extra.txt", b"changed")
    refuse_bytes("changed-consumer", change_consumer,
                 "sdist consumer receipt differs from selected artifacts")
    def forge_bundle(root: Path, fixture: dict) -> None:
        path = root / "scope-evidence" / f"repro-digest-{first}" / f"{first}.bundle.json"
        payload = json.loads(path.read_text())
        payload["members"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(payload))

    refuse_bytes("forged-bundle", forge_bundle,
                 "build-leg bundle inventory differs from distribution bytes")
    def forge_runtime(root: Path, fixture: dict) -> None:
        path = root / "scope-evidence" / f"repro-digest-{first}" / f"{first}.runtime.json"
        payload = json.loads(path.read_text())
        payload["source_sha"] = "f" * 40
        path.write_text(json.dumps(payload))

    refuse_bytes("forged-runtime", forge_runtime,
                 "runtime inventory differs from selected source or wheel")
    def forge_runtime_target(root: Path, fixture: dict) -> None:
        path = root / "scope-evidence" / f"repro-digest-{first}" / f"{first}.runtime.json"
        payload = json.loads(path.read_text())
        payload["sys_platform"] = "win32" if payload["sys_platform"] != "win32" else "linux"
        path.write_text(json.dumps(payload))

    refuse_bytes("wrong-runtime-target", forge_runtime_target,
                 "runtime interpreter differs from wheel target")
    def forge_extension_hash(root: Path, fixture: dict) -> None:
        path = root / "scope-evidence" / f"repro-digest-{first}" / f"{first}.runtime.json"
        payload = json.loads(path.read_text())
        payload["imported_extension"]["sha256"] = "0" * 64
        path.write_text(json.dumps(payload))

    refuse_bytes("forged-imported-extension", forge_extension_hash,
                 "imported extension differs from selected wheel member")
    refuse_bytes(
        "changed-runtime-archive",
        lambda r, f: (r / "scope-evidence" / f"repro-digest-{first}" / "numpy-2.5.1-py3-none-any.whl").write_bytes(b"forged"),
        "runtime archive bytes differ from receipt",
    )
    refuse_bytes(
        "missing-runtime",
        lambda r, f: (r / "scope-evidence" / f"repro-digest-{first}" / f"{first}.runtime.json").unlink(),
        "scope evidence artifact members differ from build output",
    )
    refuse_bytes(
        "changed-runtime-requirements",
        lambda r, f: (r / "scope-evidence" / f"repro-digest-{first}" / f"{first}.runtime-requirements.txt").write_text("forged\n"),
        "runtime inventory differs from selected source or wheel",
    )


def test_build_package_licence_and_strix_must_cover_installed_files(tmp_path: Path) -> None:
    import json
    import runpy
    import pytest

    _admission_fixture(tmp_path)
    result = _run_admission(tmp_path, _BYTES_STEP)
    assert "scope identity set missing" in result.stderr
    folder = tmp_path / "downloaded/release-dependency-sealed-evidence--full-set-verdict"
    verdict = json.loads((folder / "full-set-verdict.json").read_text())
    report_bytes = (folder / "gate-report.json").read_bytes()
    report = json.loads(report_bytes)
    archive_bytes = (folder / "runtime-archive-license-report.json").read_bytes()
    archive_report = json.loads(archive_bytes)
    legs = [*_expected_legs(), "sdist"]
    scopes = [tmp_path / "scope-evidence" / f"repro-digest-{leg}" for leg in legs]
    builds = [json.loads((path / f"{leg}.build-first.json").read_text()) for leg, path in zip(legs, scopes)]
    runtimes = [json.loads((path / f"{leg}.runtime.json").read_text())
                for leg, path in zip(legs, scopes) if leg != "sdist"]
    variants = [json.loads((tmp_path / "scope-evidence" / f"repro-macos-x86-{leg}" /
                            f"{leg}.runtime.json").read_text())
                for leg in legs if leg.startswith("universal2-")]
    verify = runpy.run_path(str(REPO_ROOT / "scripts/ci/verify_release_full_set_verdict.py"))[
        "verify_runtime_dependency_coverage"]
    verify(verdict, report, report_bytes, archive_report, archive_bytes, runtimes, variants, builds,
           repository="owner/repo", source_sha=_RELEASE_COMMIT)
    bad = dict(archive_report, build_packages=[])
    bad_bytes = (json.dumps(bad, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="complete dependency set"):
        verify({**verdict, "runtime_archive_license_sha256": hashlib.sha256(bad_bytes).hexdigest()},
               report, report_bytes, bad, bad_bytes, runtimes, variants, builds,
               repository="owner/repo", source_sha=_RELEASE_COMMIT)
    with pytest.raises(ValueError, match="complete dependency set"):
        verify({**verdict, "build_package_binding_artifacts": []}, report, report_bytes,
               archive_report, archive_bytes, runtimes, variants, builds,
               repository="owner/repo", source_sha=_RELEASE_COMMIT)
    forged = json.loads(archive_bytes)
    forged["build_packages"][0]["snapshots"]["sdist"] = "0" * 64
    forged_bytes = (json.dumps(forged, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="build package licence differs"):
        verify({**verdict, "runtime_archive_license_sha256": hashlib.sha256(forged_bytes).hexdigest()},
               report, report_bytes, forged, forged_bytes, runtimes, variants, builds,
               repository="owner/repo", source_sha=_RELEASE_COMMIT)
    with pytest.raises(ValueError, match="complete dependency set"):
        verify({**verdict, "build_tool_binding_artifacts": []}, report, report_bytes,
               archive_report, archive_bytes, runtimes, variants, builds,
               repository="owner/repo", source_sha=_RELEASE_COMMIT)
    forged = json.loads(archive_bytes)
    forged["build_tools"][0]["build_envs"]["sdist"] = "runner:forged"
    forged_bytes = (json.dumps(forged, sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="build tool licence differs"):
        verify({**verdict, "runtime_archive_license_sha256": hashlib.sha256(forged_bytes).hexdigest()},
               report, report_bytes, forged, forged_bytes, runtimes, variants, builds,
               repository="owner/repo", source_sha=_RELEASE_COMMIT)
    forged_builds = [dict(build) for build in builds]
    forged_builds[0]["maturin_binary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="build tool licence and Strix sets differ"):
        verify(verdict, report, report_bytes, archive_report, archive_bytes,
               runtimes, variants, forged_builds, repository="owner/repo", source_sha=_RELEASE_COMMIT)


def test_build_leg_captures_finished_distribution_bytes(tmp_path: Path) -> None:
    import json

    fixture = _admission_fixture(tmp_path)
    leg = fixture["legs"][0]
    row = tmp_path / "scope-evidence" / f"repro-digest-{leg}" / f"{leg}.tsv"
    inventory = row.with_suffix(".bundle.json")
    expected = json.loads(inventory.read_text())
    inventory.unlink()
    command = [sys.executable, str(REPO_ROOT / "scripts/ci/capture_release_bundle.py"),
               str(row), str(tmp_path / "dist")]
    result = subprocess.run(command, env={**os.environ, "RELEASE_COMMIT": _RELEASE_COMMIT},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(inventory.read_text()) == expected
    row.write_text(row.read_text().replace("\ttrue\t", "\tfalse\t"))
    result = subprocess.run(command, env={**os.environ, "RELEASE_COMMIT": _RELEASE_COMMIT},
                            capture_output=True, text=True)
    assert result.returncode != 0 and "not byte-verified" in result.stderr



def test_publication_sinks_consume_only_the_admitted_bytes(tmp_path: Path) -> None:
    text = _workflow_text()
    for sink in ("release-assets", "publish-pypi"):
        job = _job_block(text, sink)
        step = job.split("- name: Require the admitted bytes\n", 1)[1].split("run: |\n", 1)[1]
        script = textwrap.dedent(step.split("\n      - ", 1)[0])
        for name, tamper in (("ok", None), ("changed", b"tampered"), ("extra", "extra")):
            root = tmp_path / f"{sink}-{name}"
            (root / "dist").mkdir(parents=True)
            (root / "admission").mkdir()
            (root / "dist" / "a.whl").write_bytes(b"a")
            (root / "admission" / "admitted-manifest.tsv").write_text(
                f"{hashlib.sha256(b'a').hexdigest()}  a.whl\n")
            if tamper == b"tampered":
                (root / "dist" / "a.whl").write_bytes(tamper)
            elif tamper == "extra":
                (root / "dist" / "b.whl").write_bytes(b"b")
            result = subprocess.run(["bash", "-c", script], cwd=root, capture_output=True, text=True)
            if tamper is None:
                assert result.returncode == 0, (sink, result.stderr)
            else:
                assert result.returncode != 0 and "not the admitted bytes" in result.stderr, (sink, name)


def test_admission_rejects_record_collapse_before_dict_coalescing(tmp_path: Path) -> None:
    for case in ("target", "filename", "abi", "platform"):
        root = tmp_path / case
        fixture = _admission_fixture(root)
        path = root / "record/reproducibility-record.tsv"
        lines = path.read_text().splitlines()
        wheel_rows = [i for i, line in enumerate(lines[2:], 2) if "\t" in line and not line.startswith("sdist\t")]
        first, second = wheel_rows[:2]
        row = lines[second].split("\t")
        if case == "target":
            row[0] = lines[first].split("\t")[0]
        elif case == "filename":
            row[5] = lines[first].split("\t")[5]
        elif case == "abi":
            fields = row[5].split("-")
            fields[-2] = "abi3"
            row[5] = "-".join(fields)
        else:
            row[5] = row[5].rsplit("-", 1)[0] + "-win_arm64.whl"
        lines[second] = "\t".join(row)
        path.write_text("\n".join(lines) + "\n")
        result = _run_admission(root, _BYTES_STEP)
        assert result.returncode != 0, case
        assert ("duplicate record" if case in ("target", "filename") else "platform/ABI mismatch") in result.stderr


def test_release_admission_ignores_legitimate_non_distribution_artifacts(tmp_path: Path) -> None:
    # Non-distribution artifacts a full run also uploads must not be confused
    # with the distribution set: central gate diagnostic reports (names per
    # ContextualWisdomLab/.github 00c6551183cca101cfc97c43656a17cc2491c1b4, gate
    # L567/L575), the per-leg licence pair inputs, and this workflow's own
    # reproducibility digest/rebuild artifacts.
    legs = _expected_legs()
    diagnostics = [
        f"release-dependency-{kind}-report--license-evidence-{leg}" for kind in ("license", "gate") for leg in legs
    ] + [f"license-pair-{leg}" for leg in legs] + [f"repro-rebuild-{leg}" for leg in legs] + ["repro-rebuild-sdist"]
    assert len(diagnostics) == 24 + 12 + 13

    def listing_with(root: Path, extra: list[str]) -> list[dict]:
        listing = _admission_fixture(root)["listing"]
        template = dict(listing[0])
        return listing + [dict(template, name=name) for name in extra]

    ok = _run_admission(tmp_path / "ok", _SET_STEP, listing_with(tmp_path / "ok", diagnostics))
    assert ok.returncode == 0, ok.stderr

    # Rejections that must survive the relaxation-free check above.
    for name, extra, expected in (
        ("unknown-evidence", diagnostics + ["license-evidence-unknown-leg"], "unexpected publishable or evidence"),
        ("unknown-dist", diagnostics + ["dist-wheel-unknown-leg"], "unexpected publishable or evidence"),
        ("duplicate-leg", diagnostics + [f"dist-wheel-{legs[0]}"], "duplicate artifact name"),
        ("duplicate-evidence", diagnostics + ["release-dependency-sealed-evidence"], "duplicate artifact name"),
    ):
        result = _run_admission(tmp_path / name, _SET_STEP, listing_with(tmp_path / name, extra))
        assert result.returncode != 0 and expected in result.stderr, (name, result.stderr)


def test_primary_macos_runtime_cannot_duplicate_intel_coverage(tmp_path: Path) -> None:
    import json
    import runpy
    import pytest

    fixture = _admission_fixture(tmp_path)
    leg = "universal2-apple-darwin-py3.12"
    folder = tmp_path / "scope-evidence" / f"repro-digest-{leg}"
    distribution = tmp_path / "dist" / fixture["files"][leg]
    transport = runpy.run_path(str(REPO_ROOT / "scripts/ci/release_artifact_transport.py"))
    runtime = json.loads((folder / f"{leg}.runtime.json").read_text())
    bundle = transport["bundle_inventory"](distribution, leg, _RELEASE_COMMIT, fixture["build_env"][leg])
    row = {"target": leg, "file": fixture["files"][leg],
           "sha256": transport["hash_file"](distribution), "build_env": fixture["build_env"][leg]}
    members = {path.name: path for path in folder.iterdir()}
    verify = transport["verify_runtime_inventory"]
    verify(runtime, folder / f"{leg}.runtime-requirements.txt", row,
           tmp_path / "release-source", _RELEASE_COMMIT, bundle, distribution, members)
    runtime["machine"] = "x86_64"
    with pytest.raises(ValueError, match="primary macOS runtime interpreter is not arm64"):
        verify(runtime, folder / f"{leg}.runtime-requirements.txt", row,
               tmp_path / "release-source", _RELEASE_COMMIT, bundle, distribution, members)


@pytest.mark.parametrize("target,runner_arch,os_name,machine", [
    ("sdist", "X64", "linux", "x86_64"),
    ("x86_64-unknown-linux-gnu-py3.12", "X64", "linux", "x86_64"),
    ("aarch64-unknown-linux-gnu-py3.12", "ARM64", "linux", "aarch64"),
    ("universal2-apple-darwin-py3.12", "ARM64", "darwin", "arm64"),
    ("universal2-apple-darwin-py3.12", "X64", "darwin", "x86_64"),
    ("x86_64-pc-windows-msvc-py3.12", "X64", "win32", "AMD64"),
])
def test_build_capture_observes_actual_interpreter_before_snapshot(
        tmp_path, monkeypatch, target, runner_arch, os_name, machine):
    import runpy
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts/ci"))
    capture = runpy.run_path(str(REPO_ROOT / "scripts/ci/capture_release_build_scope.py"))["capture"]
    actual = SimpleNamespace(executable=sys.executable, platform=os_name,
                             implementation=SimpleNamespace(name="cpython"))
    native = SimpleNamespace(machine=lambda: machine)
    monkeypatch.setitem(capture.__globals__, "sys", actual)
    monkeypatch.setitem(capture.__globals__, "platform", native)
    def run(*args):
        if args[0] == "git":
            return _RELEASE_COMMIT
        return "Python 3.12.0" if args[-1] == "--version" else sys.executable
    monkeypatch.setitem(capture.__globals__, "_run", run)
    def inventory():
        raise RuntimeError("validated interpreter reached snapshot inventory")
    monkeypatch.setitem(capture.__globals__, "_python_packages_with_files", inventory)
    environment = {"CARGO_RELEASE_LEG": target, "CARGO_RELEASE_SHA": _RELEASE_COMMIT,
                   "CARGO_BUILD_PASS": "first", "CARGO_BUILD_PYTHON": "python",
                   "RUNNER_ARCH": runner_arch}
    if "linux-gnu" in target:
        environment["CARGO_BUILD_IMAGE"] = "quay.io/test@sha256:" + "a" * 64
    with pytest.raises(RuntimeError, match="validated interpreter"):
        capture(tmp_path, environment)
    for field, invalid in (("machine", "foreign-cpu"), ("platform", "foreign-os"),
                           ("implementation", "pypy")):
        native.machine = lambda: machine
        actual.platform = os_name
        actual.implementation.name = "cpython"
        if field == "machine":
            native.machine = lambda: invalid
        elif field == "platform":
            actual.platform = invalid
        else:
            actual.implementation.name = invalid
        with pytest.raises(ValueError, match="actual build interpreter"):
            capture(tmp_path, environment)
    if os_name == "darwin":
        actual.implementation.name = "cpython"
        native.machine = lambda: "x86_64" if machine == "arm64" else "arm64"
        with pytest.raises(ValueError, match="actual build interpreter"):
            capture(tmp_path, environment)


def _complete_admission_fixture(root: Path, monkeypatch) -> dict:
    """Bind inert distributions to real Git blobs and locally resolved Cargo graphs."""
    import json
    import runpy
    import tomllib
    import zipfile

    fixture = _admission_fixture(root)
    source = root / "release-source"
    project = ('[project]\nname="fast-mlsirm"\nversion="1.2.3"\n'
               'requires-python=">=3.12"\ndependencies=["numpy==2.5.1"]\n').encode()
    (source / "pyproject.toml").write_bytes(project)
    (source / "Cargo.toml").write_text(
        '[workspace]\nmembers=["crates/mlsirm-core", "crates/pyo3"]\n'
        'exclude=["crates/fast-mlsirm-py"]\nresolver="2"\n')
    for name in ("fast-mlsirm-py", "mlsirm-core", "pyo3"):
        crate = source / "crates" / name
        (crate / "src").mkdir(parents=True, exist_ok=True)
        (crate / "src/lib.rs").write_text("// Inert Cargo metadata fixture; never compiled.\n")
        manifest = f'[package]\nname="{name}"\nversion="0.11.4"\nedition="2021"\n'
        if name == "fast-mlsirm-py":
            manifest += ('[workspace]\n[dependencies]\n'
                         'mlsirm-core={path="../mlsirm-core"}\npyo3={path="../pyo3"}\n')
        elif name == "pyo3":
            manifest += '[features]\nextension-module=[]\n'
        (crate / "Cargo.toml").write_text(manifest)
    wheel_manifest = source / "crates/fast-mlsirm-py/Cargo.toml"
    for manifest in (source / "Cargo.toml", wheel_manifest):
        subprocess.run(["cargo", "generate-lockfile", "--offline", "--manifest-path", str(manifest)],
                       cwd=source, check=True, capture_output=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_COMMITTER_NAME": "fixture",
           "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
    for args in (("init", "-q"), ("add", "."), ("commit", "-qm", "complete inert release source")):
        subprocess.run(["git", "-C", str(source), *args], env=env, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    previous_commit = _RELEASE_COMMIT
    monkeypatch.setitem(globals(), "_RELEASE_COMMIT", commit)
    transport = runpy.run_path(str(REPO_ROOT / "scripts/ci/release_artifact_transport.py"))
    metadata = (b"Name: fast-mlsirm\nVersion: 1.2.3\nRequires-Python: >=3.12\n"
                b"Requires-Dist: numpy==2.5.1\n")
    for leg, filename in list(fixture["files"].items()):
        old = root / "dist" / filename
        name = filename.replace("pkg-1.2.3", "fast_mlsirm-1.2.3")
        path = old.with_name(name)
        if leg == "sdist":
            with tarfile.open(path, "w:gz") as archive:
                for member_name, body in (("PKG-INFO", metadata), ("pyproject.toml", project)):
                    member = tarfile.TarInfo("fast_mlsirm-1.2.3/" + member_name)
                    member.size = len(body)
                    archive.addfile(member, io.BytesIO(body))
        else:
            with zipfile.ZipFile(old) as archive:
                members = {item.filename: archive.read(item) for item in archive.infolist()}
            with zipfile.ZipFile(path, "w") as archive:
                for member_name, body in members.items():
                    archive.writestr(member_name.replace("pkg-1.2.3", "fast_mlsirm-1.2.3"),
                                     metadata if member_name.endswith("/METADATA") else body)
        old.unlink()
        fixture["files"][leg] = name
    record_path = root / "record/reproducibility-record.tsv"
    record = record_path.read_text().replace(previous_commit, commit)
    lines = record.splitlines()
    rows = {}
    for line in lines[2:]:
        values = line.split("\t")
        leg = values[0]
        values[3] = values[4] = transport["hash_file"](root / "dist" / fixture["files"][leg])
        values[5] = fixture["files"][leg]
        rows[leg] = dict(zip(lines[1].split("\t"), values))
    record_path.write_text("\n".join(lines[:2]) + "\n" + "".join(
        "\t".join(row.values()) + "\n" for row in rows.values()))
    lock_path = source / "crates/fast-mlsirm-py/Cargo.lock"
    locked = {(p["name"], p["version"], p.get("source")): p.get("checksum")
              for p in tomllib.loads(lock_path.read_text())["package"]}
    graphs = {}
    scopes = []
    for leg, row in rows.items():
        path = root / "dist" / row["file"]
        inventory = transport["bundle_inventory"](path, leg, commit, row["build_env"])
        scopes.append(transport["scope_identity"](path, leg, source, commit, row["build_env"]))
        folders = [root / "scope-evidence" / f"repro-digest-{leg}"]
        if leg.startswith("universal2-"):
            folders.append(root / "scope-evidence" / f"repro-macos-x86-{leg}")
        for folder in folders:
            (folder / f"{leg}.tsv").write_text("\t".join(row.values()) + "\n")
            for receipt_path in folder.glob("*.json"):
                receipt = json.loads(receipt_path.read_text().replace(previous_commit, commit))
                if receipt_path.name.endswith(".bundle.json"):
                    receipt = inventory
                elif ".build-" in receipt_path.name:
                    receipt["pyproject_sha256"] = transport["hash_file"](source / "pyproject.toml")
                    receipt["cargo_lock_sha256"] = transport["hash_file"](lock_path)
                    for triple in receipt["cargo_targets"]:
                        if triple not in graphs:
                            cargo = json.loads(subprocess.check_output([
                                "cargo", "metadata", "--locked", "--offline", "--format-version", "1",
                                "--filter-platform", triple, "--manifest-path", str(wheel_manifest),
                                "--features", "pyo3/extension-module"], text=True, cwd=source))
                            packages = {p["id"]: p for p in cargo["packages"]}
                            graphs[triple] = sorted(({
                                "name": packages[n["id"]]["name"], "version": packages[n["id"]]["version"],
                                "source": packages[n["id"]]["source"],
                                "checksum": locked[(packages[n["id"]]["name"], packages[n["id"]]["version"],
                                                    packages[n["id"]]["source"])],
                                "features": sorted(n["features"]),
                            } for n in cargo["resolve"]["nodes"]),
                                key=lambda p: (p["name"], p["version"], p["source"] or ""))
                        receipt["cargo_targets"][triple] = graphs[triple]
                elif receipt_path.name.endswith(".runtime.json"):
                    receipt.update(file=row["file"], sha256=row["sha256"])
                elif receipt_path.name.endswith(".consumer.json"):
                    receipt.update(file=row["file"], published_sha256=row["sha256"],
                                   consumer_sha256=row["sha256"], sdist_file=rows["sdist"]["file"],
                                   sdist_sha256=rows["sdist"]["sha256"])
                    receipt["metadata_members"] = {m["path"]: m["sha256"] for m in inventory["members"]
                                                  if m["path"].endswith((".dist-info/METADATA", ".dist-info/WHEEL"))}
                    (folder / f"{leg}.consumer.whl").write_bytes(path.read_bytes())
                receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    (root / "record/release-scope-identities.json").write_text(json.dumps(scopes) + "\n")
    return fixture


def test_complete_admission_writes_only_verified_manifest_with_real_source_and_cargo(
        tmp_path, monkeypatch):
    fixture = _complete_admission_fixture(tmp_path, monkeypatch)
    result = _run_admission(tmp_path, _BYTES_STEP)
    assert result.returncode == 0, result.stderr
    expected = "".join(f"{hashlib.sha256((tmp_path / 'dist' / name).read_bytes()).hexdigest()}  {name}\n"
                       for name in sorted(fixture["files"].values()))
    manifest = (tmp_path / "admitted-manifest.tsv").read_text()
    assert manifest == expected and len(manifest.splitlines()) == 13
    # The shipped release sinks compare this exact sha256sum-compatible form.
    for sink in ("release-assets", "publish-pypi"):
        body = _job_block(_workflow_text(), sink)
        assert "diff -u admitted.tsv downloaded.tsv" in body


@pytest.mark.parametrize("corruption,expected", [
    ("source-lock", "source declaration differs from selected commit"),
    ("omitted-cargo-node", "Cargo graph differs from selected source closure"),
    ("asserted-complete-scope", "scope identity or unresolved evidence changed"),
    ("existing-manifest", "FileExistsError"),
])
def test_complete_admission_refuses_self_consistent_transport_with_unverified_scope(
        tmp_path, monkeypatch, corruption, expected):
    import json

    fixture = _complete_admission_fixture(tmp_path, monkeypatch)
    if corruption == "source-lock":
        lock = tmp_path / "release-source/Cargo.lock"
        lock.write_text(lock.read_text() + "\n# Changed after the source commit.\n")
    elif corruption == "omitted-cargo-node":
        leg = sorted(fixture["legs"])[0]
        folder = tmp_path / "scope-evidence" / f"repro-digest-{leg}"
        for build_pass in ("first", "second"):
            path = folder / f"{leg}.build-{build_pass}.json"
            record = json.loads(path.read_text())
            record["cargo_targets"] = {triple: [node for node in graph if node["name"] != "pyo3"]
                                       for triple, graph in record["cargo_targets"].items()}
            path.write_text(json.dumps(record) + "\n")
    elif corruption == "existing-manifest":
        (tmp_path / "admitted-manifest.tsv").write_text("untrusted pre-existing bytes\n")
    else:
        path = tmp_path / "record/release-scope-identities.json"
        records = json.loads(path.read_text())
        for record in records:
            record["scopes"] = {scope: {"status": "PASS", "evidence": []} for scope in record["scopes"]}
        path.write_text(json.dumps(records) + "\n")
    # Rebuild the ZIPs and their authenticated digests; reject the source/closure
    # mismatch rather than relying on an earlier archive transport failure.
    result = _run_admission(tmp_path, _BYTES_STEP)
    assert result.returncode != 0 and expected in result.stderr, result.stderr
    if corruption == "existing-manifest":
        assert (tmp_path / "admitted-manifest.tsv").read_text() == "untrusted pre-existing bytes\n"
    else:
        assert not (tmp_path / "admitted-manifest.tsv").exists()


@pytest.mark.parametrize("corruption,expected", [
    ("foreign-attempt", "workflow attempt differs from the release"),
    ("tampered-distribution", "downloaded artifact members changed after digest verification"),
])
def test_full_admission_writes_no_manifest_for_foreign_run_or_changed_transport(
        tmp_path, monkeypatch, corruption, expected):
    import json

    fixture = _complete_admission_fixture(tmp_path, monkeypatch)
    manifest = tmp_path / "admitted-manifest.tsv"
    manifest.write_text("private test sentinel\n")
    first = _run_admission(tmp_path, _BYTES_STEP)
    assert "FileExistsError" in first.stderr, first.stderr
    manifest.unlink()
    if corruption == "foreign-attempt":
        path = tmp_path / "run-attempt.json"
        attempt = json.loads(path.read_text())
        attempt["run_attempt"] += 1
        path.write_text(json.dumps(attempt) + "\n")
    else:
        leg = fixture["legs"][0]
        path = tmp_path / "downloaded" / f"dist-wheel-{leg}" / fixture["files"][leg]
        path.write_bytes(path.read_bytes() + b"changed after transport")
    result = _run_admission(tmp_path, _BYTES_STEP)
    assert result.returncode != 0 and expected in result.stderr, result.stderr
    assert not manifest.exists()
