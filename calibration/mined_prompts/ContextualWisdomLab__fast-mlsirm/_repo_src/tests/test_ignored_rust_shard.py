"""Tests for deterministic target-qualified ignored Rust test sharding."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_ignored_rust_shard.py"
_SPEC = importlib.util.spec_from_file_location("run_ignored_rust_shard", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
sharder = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sharder
_SPEC.loader.exec_module(sharder)


def _metadata() -> dict[str, object]:
    """Return a workspace plus one package explicitly outside that workspace."""
    return {
        "workspace_members": ["core-id"],
        "packages": [
            {
                "id": "core-id",
                "name": "mlsirm-core",
                "targets": [
                    {
                        "name": "mlsirm_core",
                        "kind": ["lib"],
                        "crate_types": ["lib"],
                        "test": True,
                        "doctest": True,
                    },
                    {
                        "name": "literature_true_parameter_recovery",
                        "kind": ["test"],
                        "crate_types": ["bin"],
                        "test": True,
                        "doctest": False,
                    },
                    {
                        "name": "helper",
                        "kind": ["bin"],
                        "crate_types": ["bin"],
                        "test": False,
                        "doctest": False,
                    },
                    {
                        "name": "criterion_bench",
                        "kind": ["bench"],
                        "crate_types": ["bin"],
                        "test": True,
                        "doctest": False,
                    },
                ],
            },
            {
                "id": "pyo3-id",
                "name": "fast-mlsirm-py",
                "targets": [
                    {
                        "name": "fast_mlsirm_core",
                        "kind": ["cdylib"],
                        "crate_types": ["cdylib"],
                        "test": True,
                        "doctest": False,
                    }
                ],
            },
        ],
    }


def test_sharding_runner_contains_no_source_mutation_contract():
    """The ignored-test runner cannot rewrite statistical source before testing."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "write_text(" not in source
    assert "apply_statistical_contract_overlay" not in source
    assert "_CDM_TEST_PATH" not in source


def test_parse_test_names_filters_noise_and_deduplicates_within_one_target():
    """Only one target's lines ending in ``: test`` become test names."""
    output = """
warning: ignored
alpha::one: test
beta: benchmark
alpha::one: test
zeta::two: test
2 tests, 0 benchmarks
"""
    assert sharder.parse_test_names(output) == ["alpha::one", "zeta::two"]


def test_workspace_targets_are_explicit_and_nonmembers_stay_out_of_scope():
    """Cargo metadata becomes exact selectors for workspace members only."""
    targets = sharder.parse_workspace_targets(json.dumps(_metadata()))
    assert targets == [
        sharder.CargoTarget("mlsirm-core", "doc", "mlsirm_core"),
        sharder.CargoTarget("mlsirm-core", "lib", "mlsirm_core"),
        sharder.CargoTarget(
            "mlsirm-core",
            "test",
            "literature_true_parameter_recovery",
        ),
    ]
    assert all(target.package != "fast-mlsirm-py" for target in targets)


def test_custom_library_crate_types_use_the_lib_selector():
    """A workspace cdylib target remains testable through Cargo's lib selector."""
    metadata = _metadata()
    metadata["workspace_members"] = ["core-id", "pyo3-id"]
    targets = sharder.parse_workspace_targets(json.dumps(metadata))
    assert sharder.CargoTarget(
        "fast-mlsirm-py",
        "lib",
        "fast_mlsirm_core",
    ) in targets


def test_explicit_target_category_wins_over_library_crate_type():
    """A library-producing example is still selected with ``--example``."""
    metadata = _metadata()
    packages = metadata["packages"]
    assert isinstance(packages, list)
    core = packages[0]
    assert isinstance(core, dict)
    targets = core["targets"]
    assert isinstance(targets, list)
    targets.append(
        {
            "name": "library_example",
            "kind": ["example"],
            "crate_types": ["rlib"],
            "test": True,
            "doctest": False,
        }
    )
    parsed = sharder.parse_workspace_targets(json.dumps(metadata))
    assert sharder.CargoTarget(
        "mlsirm-core",
        "example",
        "library_example",
    ) in parsed


def test_workspace_metadata_rejects_duplicate_member_package_names():
    """Two workspace members cannot collapse onto one package audit identity."""
    metadata = _metadata()
    metadata["workspace_members"] = ["core-id", "duplicate-id"]
    packages = metadata["packages"]
    assert isinstance(packages, list)
    packages.append(
        {
            "id": "duplicate-id",
            "name": "mlsirm-core",
            "targets": [],
        }
    )
    with pytest.raises(ValueError, match="duplicate Cargo workspace package name"):
        sharder.parse_workspace_targets(json.dumps(metadata))


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ([], "root must be an object"),
        ({"packages": [], "workspace_members": []}, "workspace package inventory"),
        (
            {"packages": ["bad"], "workspace_members": ["core-id"]},
            "package entries must be objects",
        ),
        (
            {
                "packages": [{"id": "core-id", "name": "", "targets": []}],
                "workspace_members": ["core-id"],
            },
            "no valid name",
        ),
    ],
)
def test_workspace_metadata_rejects_malformed_inventories(metadata, message):
    """Malformed Cargo metadata fails before any target command is constructed."""
    with pytest.raises(ValueError, match=message):
        sharder.parse_workspace_targets(json.dumps(metadata))


def test_workspace_metadata_rejects_unknown_default_test_target_kinds():
    """A new test-bearing target kind cannot disappear from evidence silently."""
    metadata = _metadata()
    packages = metadata["packages"]
    assert isinstance(packages, list)
    core = packages[0]
    assert isinstance(core, dict)
    targets = core["targets"]
    assert isinstance(targets, list)
    targets.append(
        {
            "name": "future_target",
            "kind": ["future-kind"],
            "crate_types": ["future-crate-type"],
            "test": True,
            "doctest": False,
        }
    )
    with pytest.raises(ValueError, match="unsupported default-tested Cargo target"):
        sharder.parse_workspace_targets(json.dumps(metadata))


def test_workspace_metadata_rejects_malformed_crate_types():
    """Target crate types must remain a bounded metadata string list."""
    metadata = _metadata()
    packages = metadata["packages"]
    assert isinstance(packages, list)
    core = packages[0]
    assert isinstance(core, dict)
    targets = core["targets"]
    assert isinstance(targets, list)
    target = targets[0]
    assert isinstance(target, dict)
    target["crate_types"] = "lib"
    with pytest.raises(ValueError, match="invalid crate_types"):
        sharder.parse_workspace_targets(json.dumps(metadata))


def test_target_qualified_identifiers_preserve_same_named_tests():
    """Same-named tests in different targets remain distinct evidence entries."""
    lib_target = sharder.CargoTarget("mlsirm-core", "lib", "mlsirm_core")
    integration_target = sharder.CargoTarget(
        "mlsirm-core",
        "test",
        "literature_true_parameter_recovery",
    )
    left = f"{lib_target.identity}::shared_name"
    right = f"{integration_target.identity}::shared_name"
    assert left != right
    shards = sharder.partition_inventory([left, right], 1)
    assert set(shards[0]) == {left, right}


def test_round_robin_shards_are_exhaustive_disjoint_and_nonempty():
    """Every retained target-qualified test belongs to one stable shard."""
    names = [f"core/test/target_{index:02d}::test" for index in range(17)]
    shards = [set(values) for values in sharder.partition_inventory(names, 4)]
    union = set().union(*shards)
    assert union == set(names)
    assert all(shards)
    for left in range(len(shards)):
        for right in range(left + 1, len(shards)):
            assert shards[left].isdisjoint(shards[right])


def test_skip_requires_an_exact_target_qualified_inventory_identifier():
    """A function-only name cannot exclude same-named tests across targets."""
    names = [
        "core/lib/core::dedicated",
        "core/test/recovery::dedicated",
        "core/test/recovery::keep",
    ]
    with pytest.raises(ValueError, match="target-qualified identifier"):
        sharder.select_shard(names, 0, 1, ["dedicated"])
    selected = sharder.select_shard(
        names,
        0,
        1,
        ["core/lib/core::dedicated"],
    )
    assert selected == [
        "core/test/recovery::dedicated",
        "core/test/recovery::keep",
    ]


def test_skip_names_are_unique_and_each_matches_one_inventory_entry():
    """Duplicate or stale dedicated-test declarations fail before execution."""
    names = ["core/lib/core::one", "core/lib/core::two"]
    with pytest.raises(ValueError, match="must be unique"):
        sharder.validated_skip_set(
            names,
            ["core/lib/core::one", "core/lib/core::one"],
        )
    with pytest.raises(ValueError, match="core/lib/core::missing"):
        sharder.validated_skip_set(names, ["core/lib/core::missing"])
    assert sharder.validated_skip_set(
        names,
        ["core/lib/core::one"],
    ) == {"core/lib/core::one"}


def test_partition_rejects_duplicate_identifiers_and_silently_empty_shards():
    """Collapsed identities and matrix drift fail before any test execution."""
    with pytest.raises(ValueError, match="must be unique"):
        sharder.partition_inventory(["core/lib/core::one"] * 2, 1)
    with pytest.raises(ValueError, match="empty shards: 2, 3"):
        sharder.partition_inventory(["a", "b"], 4)


@pytest.mark.parametrize(
    ("shard", "count", "message"),
    [
        (0, 0, "at least 1"),
        (-1, 2, "0 <= shard"),
        (2, 2, "0 <= shard"),
    ],
)
def test_invalid_partition_parameters_fail(shard, count, message):
    """Invalid shard coordinates are rejected before any process execution."""
    with pytest.raises(ValueError, match=message):
        sharder.select_shard(["test"], shard, count)


def test_cargo_commands_are_target_exact_and_injection_safe():
    """Commands select one Cargo target and pass names as argument-vector fields."""
    target = sharder.CargoTarget(
        "mlsirm-core",
        "test",
        "literature_true_parameter_recovery",
    )
    assert sharder.cargo_metadata_command() == [
        "cargo",
        "metadata",
        "--no-deps",
        "--format-version",
        "1",
    ]
    assert sharder.cargo_list_command(target) == [
        "cargo",
        "test",
        "--release",
        "-p",
        "mlsirm-core",
        "--test",
        "literature_true_parameter_recovery",
        "--",
        "--ignored",
        "--list",
    ]
    name = "module::test name; echo unsafe"
    command = sharder.cargo_test_command(target, name)
    assert command[7] == name
    assert command[-4:] == [
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]


def test_library_and_doctest_commands_use_unambiguous_single_flag_selectors():
    """Library and doctest inventories never rely on flattened workspace output."""
    lib_target = sharder.CargoTarget("mlsirm-core", "lib", "mlsirm_core")
    doc_target = sharder.CargoTarget("mlsirm-core", "doc", "mlsirm_core")
    assert sharder.cargo_list_command(lib_target)[5:7] == ["--lib", "--"]
    assert sharder.cargo_list_command(doc_target)[5:7] == ["--doc", "--"]
