"""The complete central verdict must name this run's exact thirteen files."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.ci.verify_release_full_set_verdict import (
    verify_cargo_dependency_coverage,
    verify_full_set_verdict, verify_native_link_inventory,
)


SOURCE = "a" * 40
CONTROL = "b" * 40
DIGEST = "sha256:" + "c" * 64


def test_cargo_build_crates_require_exact_reviewed_archive_hash(tmp_path) -> None:
    source, sha = _cargo_source(tmp_path)
    package = {"name": "serde", "version": "1.0", "source": "registry+index",
               "checksum": "d" * 64}
    records = [{"leg": "wheel", "cargo_targets": {"target": [package]}}]
    report = {"source_sha": sha, "dependencies": [{"key": "cargo/serde@1.0", "source_sha256": "d" * 64}]}
    verify_cargo_dependency_coverage(report, records, source=source, source_sha=sha)
    for broken in ({"dependencies": []}, {"dependencies": [
            {"key": "cargo/serde@1.0", "source_sha256": "e" * 64}]}):
        with pytest.raises(ValueError, match="matching licence/Strix review"):
            broken["source_sha"] = sha
            verify_cargo_dependency_coverage(broken, records, source=source, source_sha=sha)


def test_native_links_require_exact_wheels_extensions_and_sealed_bytes() -> None:
    targets = {"x86_64-unknown-linux-gnu": ["x86_64"],
               "aarch64-unknown-linux-gnu": ["aarch64"],
               "universal2-apple-darwin": ["aarch64", "x86_64"],
               "x86_64-pc-windows-msvc": ["x86_64"]}
    distributions = []
    runtimes = []
    wheels = []
    for target, arches in targets.items():
        for version in ("3.12", "3.13", "3.14"):
            leg = f"{target}-py{version}"
            filename = f"{leg}.whl"
            member = "fast_mlsirm/_core.fixture." + ("pyd" if "windows" in target else "so")
            distributions.append({"leg": leg, "file": filename, "sha256": "a" * 64})
            runtimes.append({"leg": leg, "imported_extension": {"member": member, "sha256": "b" * 64}})
            wheels.append({"leg": leg, "file": filename, "sha256": "a" * 64,
                           "member": member, "member_sha256": "b" * 64,
                           "links": [{"arch": arch, "format": "native", "needed": ["system"],
                                      "reviews": [{"name": "system", "kind": "system-runtime",
                                                   "basis": "test platform runtime"}]}
                                     for arch in arches]})
    distributions.append({"leg": "sdist", "file": "source.tar.gz", "sha256": "c" * 64})
    analyzer = {"path": "/usr/lib/llvm-18/bin/llvm-readobj", "version": "18.1.3",
                "sha256": "d" * 64}
    native = {"schema": "cwl.release-native-links/2", "source_sha": SOURCE,
              "analyzer": analyzer, "wheels": wheels}
    raw = json.dumps(native).encode()
    verdict = {"distributions": distributions,
               "native_links_sha256": hashlib.sha256(raw).hexdigest(),
               "native_link_analyzer": analyzer}
    verify_native_link_inventory(verdict, native, raw, runtimes, SOURCE)

    with pytest.raises(ValueError, match="sealed verdict"):
        verify_native_link_inventory(verdict, native, raw + b" ", runtimes, SOURCE)
    with pytest.raises(ValueError, match="analyzer"):
        verify_native_link_inventory({**verdict, "native_link_analyzer": None}, native, raw, runtimes, SOURCE)
    missing = copy.deepcopy(native)
    missing["wheels"].pop()
    with pytest.raises(ValueError, match="twelve wheels"):
        verify_native_link_inventory(verdict, missing, raw, runtimes, SOURCE)
    foreign = copy.deepcopy(native)
    foreign["wheels"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="selected distribution"):
        verify_native_link_inventory(verdict, foreign, raw, runtimes, SOURCE)
    partial = copy.deepcopy(native)
    mac = next(row for row in partial["wheels"] if "darwin" in row["leg"])
    mac["links"].pop()
    with pytest.raises(ValueError, match="architectures"):
        verify_native_link_inventory(verdict, partial, raw, runtimes, SOURCE)
    unreviewed = copy.deepcopy(native)
    unreviewed["wheels"][0]["links"][0]["reviews"] = []
    with pytest.raises(ValueError, match="dependencies are incomplete"):
        verify_native_link_inventory(verdict, unreviewed, raw, runtimes, SOURCE)


def _case() -> dict:
    identity = {"source_repository": "ContextualWisdomLab/fast-mlsirm",
                "source_sha": SOURCE, "control_sha": CONTROL,
                "run_id": 42, "run_attempt": 2}
    rows = [{"leg": f"wheel-{number}", "file": f"wheel-{number}.whl",
             "sha256": f"{number:064x}", "artifact_id": number + 10,
             "artifact_name": f"dist-wheel-wheel-{number}", "artifact_digest": DIGEST}
            for number in range(12)]
    rows.append({"leg": "sdist", "file": "source.tar.gz", "sha256": "d" * 64,
                 "artifact_id": 99, "artifact_name": "dist-sdist", "artifact_digest": DIGEST})
    manifest = {"schema_version": 1, **identity, "distributions": rows}
    scope_rows = [{"leg": row["leg"], "artifact_id": index + 200,
                   "artifact_name": f"repro-digest-{row['leg']}", "artifact_digest": DIGEST}
                  for index, row in enumerate(rows)]
    scope_set = {"schema_version": 1, **identity, "evidence": scope_rows}
    variants = [{"leg": f"universal2-apple-darwin-py{version}", "arch": "x86_64",
                 "artifact_id": 300 + index,
                 "artifact_name": f"repro-macos-x86-universal2-apple-darwin-py{version}",
                 "artifact_digest": DIGEST}
                for index, version in enumerate(("3.12", "3.13", "3.14"))]
    name = "release-dependency-sealed-evidence--full-set-verdict"
    binding = {"key": "pypi/example@1", "name": "release-strix-binding-a2-"
               + hashlib.sha256(b"pypi/example@1").hexdigest(),
               "id": 102, "digest": DIGEST}
    archive_key = "pypi/example@1/sha256/" + "f" * 64
    archive_binding = {"key": archive_key, "name": "release-strix-binding-a2-"
                       + hashlib.sha256(archive_key.encode()).hexdigest(),
                       "id": 103, "digest": DIGEST}
    build_key = "pypi/pip@25.2/sha256/" + "e" * 64
    build_binding = {"key": build_key, "name": "release-strix-binding-a2-"
                     + hashlib.sha256(build_key.encode()).hexdigest(),
                     "id": 104, "digest": DIGEST}
    tool_key = "github-release/maturin@1.15.0/sha256/" + "f" * 64
    tool_binding = {"key": tool_key, "name": "release-strix-binding-a2-"
                    + hashlib.sha256(tool_key.encode()).hexdigest(),
                    "id": 105, "digest": DIGEST}
    verdict = {"schema": "cwl.release-full-set-verdict/1", "result": "PASS", **identity,
               "record_artifact_id": 100, "record_artifact_digest": DIGEST,
               "distributions": copy.deepcopy(rows), "binding_artifacts": [binding],
               "runtime_archive_binding_artifacts": [archive_binding],
               "build_package_binding_artifacts": [build_binding],
               "build_tool_binding_artifacts": [tool_binding],
               "runtime_archive_license_sha256": "f" * 64,
               "scope_evidence": sorted(copy.deepcopy(scope_rows), key=lambda row: row["leg"]),
               "runtime_variants": variants}

    def artifact(name: str, artifact_id: int) -> dict:
        return {"name": name, "id": artifact_id, "digest": DIGEST,
                "created_at": "2026-09-26T12:01:00Z", "expired": False,
                "workflow_run": {"id": 42, "head_sha": CONTROL}}

    return {"manifest": manifest, "scope_set": scope_set, "verdict": verdict,
            "artifacts": [artifact("reproducibility-record", 100), artifact(name, 101),
                          artifact(binding["name"], 102), artifact(archive_binding["name"], 103)]
                         + [artifact(build_binding["name"], 104), artifact(tool_binding["name"], 105)]
                         + [artifact(row["artifact_name"], row["artifact_id"]) for row in rows]
                         + [artifact(row["artifact_name"], row["artifact_id"]) for row in scope_rows]
                         + [artifact(row["artifact_name"], row["artifact_id"]) for row in variants],
            "attempt": {"id": 42, "run_attempt": 2, "head_sha": CONTROL,
                        "run_started_at": "2026-09-26T12:00:00Z"},
            "name": name}


def _verify(case: dict) -> None:
    verify_full_set_verdict(
        case["manifest"], case["scope_set"], case["verdict"], case["artifacts"], case["attempt"],
        repository="ContextualWisdomLab/fast-mlsirm", source_sha=SOURCE,
        control_sha=CONTROL, run_id=42, run_attempt=2, record_id=100,
        record_digest=DIGEST, verdict_id=101, verdict_digest=DIGEST,
        verdict_name=case["name"],
    )


def test_accepts_exact_current_run_full_set() -> None:
    _verify(_case())


def test_rejects_forged_missing_stale_or_changed_verdict() -> None:
    def other_run(case):
        case["artifacts"][1]["workflow_run"]["id"] = 41

    def stale(case):
        case["artifacts"][2]["created_at"] = "2026-09-26T11:59:00Z"

    def wrong_source(case):
        case["verdict"]["source_sha"] = "f" * 40

    def changed_file(case):
        case["verdict"]["distributions"][0]["sha256"] = "f" * 64

    def missing_file(case):
        case["verdict"]["distributions"].pop()

    def missing_binding(case):
        case["artifacts"].pop(2)

    def extra_binding(case):
        case["artifacts"].append({**case["artifacts"][2], "id": 104,
                                  "name": "release-strix-binding-a2-" + "f" * 64})

    def wrong_attempt(case):
        case["attempt"]["run_attempt"] = 1

    def changed_verdict_digest(case):
        case["artifacts"][1]["digest"] = "sha256:" + "f" * 64

    def changed_binding_key(case):
        case["verdict"]["binding_artifacts"][0]["key"] = "pypi/other@1"

    def changed_scope_digest(case):
        case["verdict"]["scope_evidence"][0]["artifact_digest"] = "sha256:" + "f" * 64

    def foreign_scope_artifact(case):
        next(item for item in case["artifacts"] if item["name"].startswith("repro-digest-"))[
            "workflow_run"]["id"] = 1

    def missing_scope_artifact(case):
        case["artifacts"] = [item for item in case["artifacts"]
                             if item["name"] != case["scope_set"]["evidence"][0]["artifact_name"]]

    def missing_intel_artifact(case):
        case["artifacts"] = [item for item in case["artifacts"]
                             if item["name"] != case["verdict"]["runtime_variants"][0]["artifact_name"]]

    def foreign_intel_artifact(case):
        next(item for item in case["artifacts"] if item["name"].startswith("repro-macos-x86-"))[
            "workflow_run"]["id"] = 1

    def forged_intel_digest(case):
        case["verdict"]["runtime_variants"][0]["artifact_digest"] = "sha256:" + "f" * 64

    for mutate in (other_run, stale, wrong_source, changed_file, missing_file,
                   missing_binding, extra_binding, wrong_attempt, changed_verdict_digest,
                   changed_binding_key, changed_scope_digest, foreign_scope_artifact,
                   missing_scope_artifact, missing_intel_artifact,
                   foreign_intel_artifact, forged_intel_digest):
        case = _case()
        mutate(case)
        with pytest.raises(ValueError):
            _verify(case)


def _cargo_source(tmp_path, *, development=False, conflict=False):
    import subprocess
    source = tmp_path / "source"
    wheel = source / "crates/fast-mlsirm-py"
    wheel.mkdir(parents=True)
    lock = 'version=4\n[[package]]\nname="serde"\nversion="1.0"\nsource="registry+index"\nchecksum="' + "d" * 64 + '"\n'
    (wheel / "Cargo.lock").write_text(lock)
    if development:
        lock += '[[package]]\nname="proptest"\nversion="1.0"\nsource="registry+index"\nchecksum="' + "e" * 64 + '"\n'
    if conflict:
        lock = lock.replace("d" * 64, "f" * 64)
    (source / "Cargo.lock").write_text(lock)
    for command in (("init", "-q"), ("add", "."),
                    ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "locks")):
        subprocess.run(["git", "-C", str(source), *command], check=True)
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    return source, sha


@pytest.mark.parametrize("mutation", [None, "omitted_dev", "wrong_dev_hash", "conflict", "foreign_source"])
def test_reviewed_cargo_set_covers_source_root_development_lock(tmp_path, mutation):
    source, sha = _cargo_source(tmp_path, development=True, conflict=mutation == "conflict")
    report = {"source_sha": sha, "dependencies": [
        {"key": "cargo/serde@1.0", "source_sha256": "d" * 64},
        {"key": "cargo/proptest@1.0", "source_sha256": "e" * 64}]}
    if mutation == "omitted_dev":
        report["dependencies"].pop()
    elif mutation == "wrong_dev_hash":
        report["dependencies"][-1]["source_sha256"] = "f" * 64
    elif mutation == "foreign_source":
        report["source_sha"] = "b" * 40
    if mutation is None:
        verify_cargo_dependency_coverage(report, [], source=source, source_sha=sha)
    else:
        with pytest.raises(ValueError):
            verify_cargo_dependency_coverage(report, [], source=source, source_sha=sha)
