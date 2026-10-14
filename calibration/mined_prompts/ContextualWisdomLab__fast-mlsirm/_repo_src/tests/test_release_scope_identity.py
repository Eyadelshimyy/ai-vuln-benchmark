"""Inert archive/Git fixtures through the workflow's actual identity consumers."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import tarfile
import textwrap
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / "scripts/ci/release_artifact_transport.py"))
WORKFLOW = (ROOT / ".github/workflows/publish-pypi.yml").read_text()


@pytest.fixture
def scope_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "release-source"
    source.mkdir()
    project = b'[build-system]\nrequires=["maturin>=1"]\n[project]\nname="fixture"\nversion="1"\nrequires-python=">=3.12"\ndependencies=["numpy>=1"]\n[project.optional-dependencies]\ndev=["pytest"]\nfuzz=["atheris"]\n'
    (source / "pyproject.toml").write_bytes(project)
    (source / "Cargo.lock").write_text("# synthetic lock\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_COMMITTER_NAME": "fixture",
           "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
    for args in [("init", "-q"), ("add", "."), ("commit", "-qm", "fixture")]:
        subprocess.run(["git", "-C", str(source), *args], env=env, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    legs = re.search(r'EXPECTED_WHEEL_LEGS: "([^"]+)"', WORKFLOW).group(1).split()
    platforms = {"x86_64-unknown-linux-gnu": "manylinux2014_x86_64", "aarch64-unknown-linux-gnu": "manylinux2014_aarch64",
                 "universal2-apple-darwin": "macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2", "x86_64-pc-windows-msvc": "win_amd64"}
    dist = tmp_path / "dist"
    dist.mkdir()
    rows = {}
    for leg in legs:
        target, version = leg.rsplit("-py", 1)
        cp = "cp" + version.replace(".", "")
        platform = platforms[target]
        path = dist / f"fixture-1-{cp}-{cp}-{platform}.whl"
        with zipfile.ZipFile(path, "w") as z:
            tags = "".join(f"Tag: {cp}-{cp}-{part}\n" for part in platform.split("."))
            z.writestr("fixture-1.dist-info/WHEEL", f"Wheel-Version: 1.0\n{tags}")
            z.writestr("fixture-1.dist-info/METADATA", "Name: fixture\nVersion: 1\nRequires-Dist: numpy\n")
        rows[leg] = {"target": leg, "file": path.name, "sha256": M["hash_file"](path), "build_env": "fixture:" + target}
    path = dist / "fixture-1.tar.gz"
    with tarfile.open(path, "w:gz") as t:
        for name, data in [("pyproject.toml", project), ("PKG-INFO", b"Name: fixture\nVersion: 1\nRequires-Python: >=3.12\n")]:
            member = tarfile.TarInfo("fixture-1/" + name)
            member.size = len(data)
            t.addfile(member, io.BytesIO(data))
    rows["sdist"] = {"target": "sdist", "file": path.name, "sha256": M["hash_file"](path), "build_env": "fixture:source"}
    (tmp_path / "trusted-control").symlink_to(ROOT, target_is_directory=True)
    monkeypatch.setenv("RELEASE_COMMIT", sha)
    # Run the precise producer block in the real workflow, not a copied implementation.
    producer = WORKFLOW.split('          record = Path("reproducibility-record.tsv")\n', 1)[1].split('          with record.open', 1)[0]
    exec(textwrap.dedent(producer), {"Path": Path, "os": os, "rows": rows})
    records = json.loads(Path("release-scope-identities.json").read_text())
    return source, sha, rows, records


def test_workflow_producer_transport_and_actual_declaration_consumer(scope_fixture):
    source, sha, rows, records = scope_fixture
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as z:
        z.write("release-scope-identities.json")
        z.writestr("reproducibility-record.tsv", "fixture record")
    data = payload.getvalue()
    selected = [{"id": 71, "name": "reproducibility-record", "digest": "sha256:" + hashlib.sha256(data).hexdigest()}]
    receipt = M["materialize"](selected, "owner/repo", Path("downloaded"), lambda r, i, out: out.write(data))
    M["verify_materialized"](selected, receipt, Path("downloaded"))
    for leg, row in rows.items():
        folder = Path("downloaded") / ("dist-sdist" if leg == "sdist" else f"dist-wheel-{leg}")
        folder.mkdir()
        (folder / row["file"]).write_bytes((Path("dist") / row["file"]).read_bytes())
    assert len(records) == 13
    assert sum(r["kind"] == "sdist" for r in records) == 1
    assert all(v == {"status": "UNKNOWN", "evidence": None} for r in records for v in r["scopes"].values())
    start = WORKFLOW.index('          scope_records = json.loads(Path("downloaded/reproducibility-record/release-scope-identities.json")')
    end = WORKFLOW.index('          # Source declarations cannot approve a closure.', start)
    import types
    calls = []
    transport = types.SimpleNamespace(**M)
    transport.verify_distribution_requirements = lambda *args: calls.append("metadata")
    transport.verify_bundled_license_notices = lambda *args: None
    transport.verify_license_selection_report = lambda *args: None
    transport.verify_cargo_graph_closure = lambda *args: calls.append("cargo")
    # This focused check covers transported declarations only. Complete admission
    # is exercised separately with real Git/Cargo and all actual verifiers.
    exec(textwrap.dedent(WORKFLOW[start:end]), {
        "json": json, "Path": Path, "rows": rows, "commit": sha,
        "transport": transport, "build_graph_receipts": [], "report": {},
        "verifier": types.SimpleNamespace(verify_cargo_dependency_coverage=lambda *args, **kwargs: calls.append("review"))})
    assert calls == [*(["metadata"] * 13), "cargo", "review"]
    assert 'Path("admitted-manifest.tsv").open("x"' in WORKFLOW[end:]
    assert 'name: reproducibility-record\n          path: |\n            reproducibility-record.tsv\n            release-scope-identities.json' in WORKFLOW
    assert not Path("admitted-manifest.tsv").exists()


def test_distribution_requirements_bind_source_extras_across_targets(tmp_path):
    source = tmp_path / "release-source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="fast-mlsirm"\nversion="0.11.4"\nrequires-python=">=3.12"\n'
        'dependencies=["numpy>=1.24"]\n[project.optional-dependencies]\n'
        'dev=["pytest>=8"]\n'
        'fuzz=["atheris>=2.3; implementation_name == \'cpython\' and sys_platform == \'linux\' '
        'and platform_machine == \'x86_64\' and python_version < \'3.15\'"]\n')
    env = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_COMMITTER_NAME": "fixture",
           "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
    for args in (("init", "-q"), ("add", "."), ("commit", "-qm", "fixture")):
        subprocess.run(["git", "-C", str(source), *args], env=env, check=True, capture_output=True)
    source_sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    metadata = ("Name: fast-mlsirm\nVersion: 0.11.4\nRequires-Python: >=3.12\n"
                "Requires-Dist: numpy>=1.24\n"
                "Requires-Dist: pytest>=8 ; extra == 'dev'\n"
                "Requires-Dist: atheris>=2.3 ; python_full_version < '3.15' and "
                "implementation_name == 'cpython' and platform_machine == 'x86_64' "
                "and sys_platform == 'linux' and extra == 'fuzz'\n"
                "Provides-Extra: dev\nProvides-Extra: fuzz\n")
    wheel = tmp_path / "fast_mlsirm-0.11.4-cp314-cp314-manylinux2014_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fast_mlsirm-0.11.4.dist-info/METADATA", metadata)
    sdist = tmp_path / "fast_mlsirm-0.11.4.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        data = metadata.encode()
        member = tarfile.TarInfo("fast_mlsirm-0.11.4/PKG-INFO")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    verify = M["verify_distribution_requirements"]
    verify(wheel, "x86_64-unknown-linux-gnu-py3.14", source, source_sha)
    verify(sdist, "sdist", source, source_sha)
    (source / "pyproject.toml").write_text("[project]\nname='tampered'\n")
    verify(wheel, "x86_64-unknown-linux-gnu-py3.14", source, source_sha)
    for changed in (metadata.replace("Requires-Dist: atheris>=2.3 ;", "Requires-Dist: atheris>=3 ;"),
                    metadata.replace("Requires-Dist: atheris>=2.3 ;", "Requires-Dist: missing>=2.3 ;"),
                    metadata.replace("sys_platform == 'linux'", "sys_platform == 'win32'")):
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("fast_mlsirm-0.11.4.dist-info/METADATA", changed)
        with pytest.raises(ValueError, match="requirements differ from source"):
            verify(wheel, "x86_64-unknown-linux-gnu-py3.14", source, source_sha)
    parser_step = WORKFLOW.split("- name: Provision hash-locked metadata parser", 1)[1].split("- name:", 1)[0]
    assert "--require-hashes -r trusted-control/requirements/package.txt" in parser_step
    assert "release-source/requirements" not in parser_step
    assert "transport.verify_distribution_requirements(distribution_paths[row[\"file\"]], leg," in WORKFLOW


@pytest.mark.parametrize("case", ["missing", "duplicate", "other-platform", "abi", "metadata", "lock", "source", "complete", "empty-scope", "wheel-as-sdist", "schema-bool"])
def test_scope_record_mutations_refuse_before_generic_hold(scope_fixture, case):
    source, sha, rows, records = scope_fixture
    records = copy.deepcopy(records)
    if case == "missing":
        records.pop()
    elif case == "duplicate":
        records[-1] = records[0]
    elif case == "other-platform":
        records[0]["target"] = "x86_64-pc-windows-msvc"
    elif case == "abi":
        records[0]["abi"] = "cp313"
    elif case == "metadata":
        records[0]["metadata_members"] = {}
    elif case == "lock":
        records[0]["source_declarations"]["Cargo.lock"] = "0" * 64
    elif case == "source":
        records[0]["source_sha"] = "0" * 40
    elif case == "complete":
        records[0]["scopes"]["runtime"] = {"status": "complete", "evidence": []}
    elif case == "empty-scope":
        records[0]["scopes"] = {}
    elif case == "wheel-as-sdist":
        records[-1]["kind"] = "wheel"
    elif case == "schema-bool":
        records[0]["schema_version"] = True
    with pytest.raises(ValueError) as error:
        M["verify_scope_identities"](records, rows, {r["file"]: Path("dist") / r["file"] for r in rows.values()}, source, sha)
    assert "evidence UNKNOWN" not in str(error.value)


def test_actual_artifact_replacement_and_source_head_refuse(scope_fixture):
    source, sha, rows, records = scope_fixture
    row = next(iter(rows.values()))
    artifact = Path("dist") / row["file"]
    with zipfile.ZipFile(artifact, "a") as z:
        z.writestr("additional-condition.txt", "synthetic")
    with pytest.raises(ValueError, match="identity"):
        M["verify_scope_identities"](records, rows, {r["file"]: Path("dist") / r["file"] for r in rows.values()}, source, sha)
    with pytest.raises(ValueError, match="checkout mismatch"):
        M["scope_identity"](artifact, row["target"], source, "0" * 40, row["build_env"])


def test_universal2_leg_requires_universal2_platform_tag(scope_fixture):
    source, sha, rows, _ = scope_fixture
    leg = "universal2-apple-darwin-py3.14"
    row = rows[leg]
    artifact = Path("dist") / row["file"]
    without_universal = artifact.with_name(artifact.name.replace(".macosx_10_12_universal2", ""))
    without_universal.write_bytes(artifact.read_bytes())
    with pytest.raises(ValueError, match="no universal2 platform tag"):
        M["scope_identity"](without_universal, leg, source, sha, row["build_env"])


@pytest.mark.parametrize("case", ["missing-metadata", "duplicate-metadata", "wrong-tag", "wrong-platform", "missing-sdist-declaration", "changed-sdist-pyproject"])
def test_actual_archive_declaration_failures(scope_fixture, case):
    source, sha, rows, records = scope_fixture
    leg = "sdist" if "sdist" in case else next(iter(rows))
    row = rows[leg]
    artifact = Path("dist") / row["file"]
    if leg == "sdist":
        with tarfile.open(artifact, "w:gz") as archive:
            data = b"different source bytes"
            member = tarfile.TarInfo("fixture-1/PKG-INFO")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
            if case == "changed-sdist-pyproject":
                member = tarfile.TarInfo("fixture-1/pyproject.toml")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    elif case == "wrong-platform":
        leg = "x86_64-pc-windows-msvc-py3.12"
    else:
        with zipfile.ZipFile(artifact, "w") as archive:
            tag = "cp313-cp313-win_amd64" if case == "wrong-tag" else records[0]["wheel_tags"][0]
            archive.writestr("fixture-1.dist-info/WHEEL", f"Tag: {tag}\n")
            if case != "missing-metadata":
                archive.writestr("fixture-1.dist-info/METADATA", "Name: fixture\n")
            if case == "duplicate-metadata":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("fixture-1.dist-info/METADATA", "Name: changed\n")
    with pytest.raises(ValueError):
        M["scope_identity"](artifact, leg, source, sha, row["build_env"])


@pytest.mark.parametrize("field,value", [
    ("Name", "impostor"),
    ("Version", "2"),
    ("Requires-Python", ">=3.14"),
])
def test_sdist_package_info_must_match_exact_source(scope_fixture, field, value):
    source, sha, rows, _ = scope_fixture
    artifact = Path("dist/fixture-1.tar.gz")
    project = (source / "pyproject.toml").read_bytes()
    metadata = {"Name": "fixture", "Version": "1", "Requires-Python": ">=3.12"}
    metadata[field] = value
    with tarfile.open(artifact, "w:gz") as archive:
        for name, data in (("pyproject.toml", project),
                           ("PKG-INFO", "".join(f"{key}: {item}\n" for key, item in metadata.items()).encode())):
            member = tarfile.TarInfo("fixture-1/" + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="PKG-INFO differs"):
        M["scope_identity"](artifact, "sdist", source, sha, rows["sdist"]["build_env"])


@pytest.mark.parametrize("path,data,error", [
    ("untracked.py", b"print('unexpected')\n", "absent from release commit"),
    ("Cargo.lock", b"changed lock\n", "differs from release commit"),
])
def test_sdist_source_members_must_match_exact_commit(scope_fixture, path, data, error):
    source, sha, rows, _ = scope_fixture
    artifact = Path("dist/fixture-1.tar.gz")
    entries = {
        "pyproject.toml": (source / "pyproject.toml").read_bytes(),
        "PKG-INFO": b"Name: fixture\nVersion: 1\nRequires-Python: >=3.12\n",
        path: data,
    }
    with tarfile.open(artifact, "w:gz") as archive:
        for name, payload in entries.items():
            member = tarfile.TarInfo("fixture-1/" + name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match=error):
        M["scope_identity"](artifact, "sdist", source, sha, rows["sdist"]["build_env"])


def test_sdist_consumer_receipt_binds_both_finished_distributions(scope_fixture, monkeypatch):
    import shutil

    source, sha, rows, _ = scope_fixture
    monkeypatch.syspath_prepend(str(ROOT / "scripts/ci"))
    consumer = runpy.run_path(str(ROOT / "scripts/ci/release_sdist_consumer.py"))
    leg = next(iter(rows))
    direct = Path("dist") / rows[leg]["file"]
    with zipfile.ZipFile(direct, "a") as archive:
        archive.writestr("fast_mlsirm/_core.fixture.so", b"native extension")
        archive.writestr("fast_mlsirm/_licenses/example-MIT.txt", b"retained licence notice")
    rows[leg]["sha256"] = M["hash_file"](direct)
    sdist = Path("dist") / rows["sdist"]["file"]
    sdist_input = Path("sdist-input")
    sdist_input.mkdir()
    shutil.copyfile(sdist, sdist_input / sdist.name)
    sdist_row = Path("sdist.tsv")
    wheel_row = Path(f"{leg}.tsv")
    for path, row in ((sdist_row, rows["sdist"]), (wheel_row, rows[leg])):
        path.write_text("\t".join([row["target"], "true", "clean-target-repeat-same-env",
                                    row["sha256"], row["sha256"], row["file"], row["build_env"]]) + "\n")
    prepared = Path("sdist-consumer")
    consumer["prepare"](source, sha, sdist_row, sdist_input, prepared)
    built = prepared / "source/dist-consumer"
    built.mkdir()
    shutil.copyfile(direct, built / direct.name)
    output = Path("repro-digest")
    output.mkdir()
    receipt = consumer["capture"](source, sha, wheel_row, Path("dist"), prepared, output)
    (source / "uv.lock").write_text("synthetic lock\n")
    requirements = output / f"{leg}.runtime-requirements.txt"
    requirements.write_text("synthetic requirements\n")
    installed = [{"name": "fast-mlsirm", "version": "1"}]
    runtime = {key: value for key, value in (
        ("source_sha", sha), ("leg", leg), ("file", rows[leg]["file"]),
        ("sha256", rows[leg]["sha256"]), ("uv_version", "uv 0.12.5"),
        ("python_version", f"{consumer['sys'].version_info.major}.{consumer['sys'].version_info.minor}"),
        ("implementation", consumer["sys"].implementation.name),
        ("sys_platform", consumer["sys"].platform), ("machine", consumer["platform"].machine()),
        ("requirements_sha256", M["hash_file"](requirements)),
        ("uv_lock_sha256", M["hash_file"](source / "uv.lock")),
        ("locked_dependencies", []), ("installed", installed))}
    (output / f"{leg}.runtime.json").write_text(json.dumps(runtime))
    lists = iter(([], installed))
    calls = []

    def fake_run(*args, cwd):
        calls.append(args)
        if args[:2] == ("uv", "--version"):
            return "uv 0.12.5\n"
        if args[:3] == ("uv", "pip", "list"):
            return json.dumps(next(lists))
        return ""

    monkeypatch.setitem(consumer["install"].__globals__, "_run", fake_run)
    monkeypatch.setitem(consumer["install"].__globals__, "installed_extension",
                        lambda interpreter, venv: receipt["native_extension"])
    receipt = consumer["install"](source, sha, wheel_row, output, Path.cwd())
    assert any(args[:3] == ("uv", "pip", "sync") and "--require-hashes" in args
               and "--no-index" in args for args in calls)
    assert any(args[:3] == ("uv", "pip", "install") and "--no-deps" in args
               and "--no-index" in args and "--no-cache" in args for args in calls)
    M["verify_sdist_consumer"](receipt, output / f"{leg}.consumer.whl", direct,
                               rows[leg], rows["sdist"], sha, runtime)
    consumer_wheel = output / f"{leg}.consumer.whl"
    original = consumer_wheel.read_bytes()
    with zipfile.ZipFile(consumer_wheel) as archive:
        members = {entry.filename: archive.read(entry) for entry in archive.infolist()}
    for replacement in (None, b"changed licence notice"):
        with zipfile.ZipFile(consumer_wheel, "w") as archive:
            for name, body in members.items():
                if name.startswith("fast_mlsirm/_licenses/"):
                    if replacement is None:
                        continue
                    body = replacement
                archive.writestr(name, body)
        forged_notice = copy.deepcopy(receipt)
        forged_notice["consumer_sha256"] = M["hash_file"](consumer_wheel)
        with pytest.raises(ValueError, match="omits or changes licence notices"):
            M["verify_sdist_consumer"](forged_notice, consumer_wheel, direct,
                                       rows[leg], rows["sdist"], sha, runtime)
    consumer_wheel.write_bytes(original)
    forged_install = copy.deepcopy(receipt)
    forged_install["installation"]["installed"] = []
    with pytest.raises(ValueError, match="consumer receipt differs"):
        M["verify_sdist_consumer"](forged_install, output / f"{leg}.consumer.whl", direct,
                                   rows[leg], rows["sdist"], sha, runtime)
    receipt["sdist_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="consumer receipt differs"):
        M["verify_sdist_consumer"](receipt, output / f"{leg}.consumer.whl", direct,
                                   rows[leg], rows["sdist"], sha, runtime)
    consumer_wheel = output / f"{leg}.consumer.whl"
    with zipfile.ZipFile(consumer_wheel, "a") as archive:
        archive.writestr("fast_mlsirm/hidden-data.bin", b"\x7fELFhidden native code")
    forged = json.loads((output / f"{leg}.consumer.json").read_text())
    forged["consumer_sha256"] = M["hash_file"](consumer_wheel)
    with pytest.raises(ValueError, match="unaccounted consumer bundled native binary"):
        M["verify_sdist_consumer"](forged, consumer_wheel, direct,
                                   rows[leg], rows["sdist"], sha, runtime)


def test_selected_licence_notice_must_survive_distribution(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = "python/fast_mlsirm/_licenses/example-MIT.txt"
    body = b"Permission notice retained verbatim."
    (source / path).parent.mkdir(parents=True)
    (source / path).write_bytes(body)
    (source / "docs").mkdir()
    digest = hashlib.sha256(body).hexdigest()
    (source / "docs/release-license-selections.json").write_text(json.dumps([
        {"bundled_notice": {"path": path, "sha256": digest}}]))
    env = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_COMMITTER_NAME": "fixture",
           "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
    for args in (("init", "-q"), ("add", "."), ("commit", "-qm", "notice")):
        subprocess.run(["git", "-C", str(source), *args], env=env, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    verify = M["verify_bundled_license_notices"]
    (source / path).write_bytes(b"mutable replacement")
    for kind, member in (("wheel", path.removeprefix("python/")), ("sdist", "root/" + path)):
        inventory = {"leg": "sdist" if kind == "sdist" else "target-py3.14", "members": [{"path": member, "sha256": digest}]}
        verify(inventory, source, sha)
        for members in ([], [{"path": member, "sha256": "a" * 64}]):
            with pytest.raises(ValueError, match="omits or changes"):
                verify({"leg": "sdist" if kind == "sdist" else "target-py3.14", "members": members}, source, sha)


def test_selection_report_binds_immutable_choices_and_archive(tmp_path):
    source = tmp_path / "source"
    (source / "docs").mkdir(parents=True)
    choice = {"ecosystem": "cargo", "name": "example", "version": "1", "chosen": "MIT",
              "rationale": "Reviewed exact archive text", "archive_sha256": "a" * 64}
    blob = json.dumps([choice]).encode()
    declaration = source / "docs/release-license-selections.json"
    declaration.write_bytes(blob)
    env = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_COMMITTER_NAME": "fixture",
           "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
    for args in (("init", "-q"), ("add", "."), ("commit", "-qm", "choices")):
        subprocess.run(["git", "-C", str(source), *args], env=env, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    report = {"source_sha": sha, "license_selections_sha256": hashlib.sha256(blob).hexdigest(),
              "dependencies": [{"key": "cargo/example@1", "license": choice["chosen"],
                                "license_selection_rationale": choice["rationale"],
                                "source_sha256": choice["archive_sha256"]}]}
    verify = M["verify_license_selection_report"]
    declaration.write_text("mutable replacement")
    verify(report, source, sha)
    added_single_licence = copy.deepcopy(report)
    added_single_licence["dependencies"].append({
        "key": "cargo/unselected@1", "license": "ISC", "source_sha256": "c" * 64})
    with pytest.raises(ValueError, match="lack source-bound bundled notice"):
        verify(added_single_licence, source, sha)
    for field, value in (("license", "Apache-2.0"), ("license_selection_rationale", "unreviewed"),
                         ("source_sha256", "b" * 64)):
        changed = copy.deepcopy(report)
        changed["dependencies"][0][field] = value
        with pytest.raises(ValueError, match="choice differs"):
            verify(changed, source, sha)
    for field, value in (("license_selections_sha256", "b" * 64), ("source_sha", "b" * 40),
                         ("dependencies", [])):
        changed = copy.deepcopy(report)
        changed[field] = value
        with pytest.raises(ValueError):
            verify(changed, source, sha)


@pytest.mark.parametrize("changed", ["Cargo.toml", "Cargo.lock", "crates/mlsirm-core/Cargo.toml",
                                   "crates/fast-mlsirm-py/Cargo.toml",
                                   "crates/fast-mlsirm-py/Cargo.lock", ".cargo/config.toml"])
def test_cargo_closure_rejects_changed_source_before_cargo(tmp_path, monkeypatch, changed):
    source = tmp_path / "source"
    source.mkdir()
    for relative in ("Cargo.toml", "Cargo.lock", "crates/mlsirm-core/Cargo.toml",
                     "crates/fast-mlsirm-py/Cargo.toml", "crates/fast-mlsirm-py/Cargo.lock"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# committed declaration\n")
    for args in [("init", "-q"), ("add", "."),
                 ("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                  "commit", "-qm", "source")]:
        subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    path = source / changed
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# changed after source selection\n")
    legs = re.search(r'EXPECTED_WHEEL_LEGS: "([^"]+)"', WORKFLOW).group(1).split() + ["sdist"]
    receipts = [{"leg": leg, "cargo_targets": {}} for leg in legs]
    original = subprocess.check_output

    def no_cargo(command, **kwargs):
        assert command[0] != "cargo"
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", no_cargo)
    with pytest.raises(ValueError, match="source declaration differs|untracked configuration"):
        M["verify_cargo_graph_closure"](receipts, source, sha)


def test_cargo_closure_requires_every_target_graph(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for relative in ("crates/mlsirm-core/Cargo.toml", "crates/fast-mlsirm-py/Cargo.toml",
                     "crates/fast-mlsirm-py/Cargo.lock"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("package = []\n" if relative.endswith(".lock") else "# fixture\n")
    for args in [("init", "-q"), ("add", "."),
                 ("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                  "commit", "-qm", "source")]:
        subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    legs = re.search(r'EXPECTED_WHEEL_LEGS: "([^"]+)"', WORKFLOW).group(1).split() + ["sdist"]
    receipts = [{"leg": leg, "cargo_targets": {}} for leg in legs]
    with pytest.raises(ValueError, match="target graph set is incomplete"):
        M["verify_cargo_graph_closure"](receipts, source, sha)


@pytest.mark.parametrize("mutation", [None, "missing-proof", "foreign-source", "changed-path", "changed-digest",
                                      "changed-archive", "foreign-upstream", "unselected-proof", "missing-notice",
                                      "changed-notice", "symlink-notice", "foreign-secondary", "missing-secondary",
                                      "mutable-secondary"])
@pytest.mark.parametrize("independent_input", [False, True])
def test_supplemental_license_notice_matches_real_source_git(tmp_path, mutation, independent_input):
    source = tmp_path / "source"
    source.mkdir()
    path = "python/fast_mlsirm/_licenses/example.txt"
    notice = source / path
    notice.parent.mkdir(parents=True)
    notice.write_bytes(b"inert notice fixture")
    digest = hashlib.sha256(notice.read_bytes()).hexdigest()
    upstream = [{"url": "https://raw.githubusercontent.com/example/library/" + "1" * 40 + "/LICENSE",
                 "sha256": "2" * 64}]
    if independent_input:
        upstream.append({"url": "https://raw.githubusercontent.com/example/generator-input/" + "5" * 40 + "/LICENSE",
                         "sha256": "6" * 64})
    if mutation == "mutable-secondary":
        upstream[-1]["url"] = "https://raw.githubusercontent.com/example/library/main/LICENSE"
    choice = {"ecosystem": "cargo", "name": "example", "version": "1", "chosen": "MIT",
              "rationale": "Exact source notice", "archive_sha256": "3" * 64,
              "bundled_notice": {"path": path, "sha256": digest}, "upstream_licenses": upstream}
    if mutation == "missing-notice":
        notice.unlink()
    elif mutation == "changed-notice":
        notice.write_bytes(b"changed bytes")
    elif mutation == "symlink-notice":
        notice.unlink()
        notice.symlink_to("/missing")
    elif mutation == "unselected-proof":
        del choice["upstream_licenses"]
    blob = json.dumps([choice]).encode()
    declaration = source / "docs/release-license-selections.json"
    declaration.parent.mkdir()
    declaration.write_bytes(blob)
    for args in [("init", "-q"), ("add", "."), ("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                                               "-c", "commit.gpgsign=false", "commit", "-qm", "source")]:
        subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    proof = {"source_sha": sha, "path": path, "sha256": digest, "archive_sha256": choice["archive_sha256"],
             "upstream_commit": "1" * 40, "upstream_licenses": upstream}
    report = {"source_sha": sha, "license_selections_sha256": hashlib.sha256(blob).hexdigest(),
              "dependencies": [{"key": "cargo/example@1", "license": "MIT", "source_sha256": "3" * 64,
                                "license_selection_rationale": choice["rationale"], "source_license_notice": proof}]}
    if mutation == "missing-proof":
        del report["dependencies"][0]["source_license_notice"]
    elif mutation in {"foreign-source", "changed-path", "changed-digest", "changed-archive", "foreign-upstream"}:
        field, value = {"foreign-source": ("source_sha", "4" * 40), "changed-path": ("path", path + ".other"),
                        "changed-digest": ("sha256", "4" * 64), "changed-archive": ("archive_sha256", "4" * 64),
                        "foreign-upstream": ("upstream_commit", "4" * 40)}[mutation]
        proof[field] = value
    elif mutation == "foreign-secondary":
        proof["upstream_licenses"][-1]["url"] = "https://raw.githubusercontent.com/example/library/" + "7" * 40 + "/LICENSE"
    elif mutation == "missing-secondary":
        proof["upstream_licenses"].pop()
    # Dirty working-tree bytes cannot substitute for the selected source commit.
    if notice.is_file():
        notice.write_bytes(b"dirty replacement")
    verify = M["verify_license_selection_report"]
    if mutation is None:
        verify(report, source, sha)
    else:
        with pytest.raises(ValueError):
            verify(report, source, sha)
