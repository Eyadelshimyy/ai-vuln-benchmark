#!/usr/bin/env python3
"""List and execute one deterministic shard of ignored Rust tests.

Cargo's test harness lists test names only within one compiled target. Flattening
``cargo test --workspace -- --list`` output loses that target identity, so two
same-named tests in different libraries or integration binaries can be collapsed
or invoked ambiguously. This helper inventories each default-tested workspace
target independently, qualifies every test as
``package/selector/target::test_name``, validates exact dedicated-test
exclusions, and invokes the matching Cargo target with ``--ignored --exact``.

The runner is source-read-only. Its package scope comes exclusively from
Cargo's ``workspace_members`` metadata, matching the former ``--workspace``
command and leaving explicitly excluded packages to their dedicated workflows.
Every child process is additionally bounded by an operation-specific deadline;
GitHub Actions job timeouts remain an independent outer ceiling.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts._bounded_json import parse_json_bounded
except ModuleNotFoundError:
    from _bounded_json import parse_json_bounded

# The helper must remain importable both when this file is executed directly and
# when repository tests load it through ``spec_from_file_location``.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _subprocess_deadlines import (  # noqa: E402
    BoundedSubprocessTimeout,
    SubprocessOperation,
    run_bounded,
)

_TEST_LINE = re.compile(r"^(?P<name>.+): test$")
_SUPPORTED_SELECTORS = frozenset({"lib", "bin", "test", "example", "doc"})
_LIBRARY_TARGET_KINDS = frozenset(
    {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"}
)
_NON_DEFAULT_TARGET_KINDS = frozenset({"bench", "custom-build"})


@dataclass(frozen=True, order=True)
class CargoTarget:
    """One Cargo test-bearing target with a stable audit identity."""

    package: str
    selector: str
    target_name: str

    def __post_init__(self) -> None:
        """Reject malformed target metadata before constructing commands."""
        if not self.package or not self.target_name:
            raise ValueError("Cargo target package and target_name must be non-empty")
        if self.selector not in _SUPPORTED_SELECTORS:
            raise ValueError(f"unsupported Cargo selector: {self.selector}")

    @property
    def identity(self) -> str:
        """Return the stable package/target prefix used by test identifiers."""
        return f"{self.package}/{self.selector}/{self.target_name}"


@dataclass(frozen=True, order=True)
class IgnoredTest:
    """One ignored Rust test tied to exactly one Cargo target."""

    identifier: str
    target: CargoTarget
    test_name: str


def parse_test_names(output: str) -> list[str]:
    """Return sorted unique Rust test names from one target's list output."""
    names = {
        match.group("name")
        for line in output.splitlines()
        if (match := _TEST_LINE.match(line.strip())) is not None
    }
    return sorted(names)


def cargo_metadata_command() -> list[str]:
    """Return the read-only command used to enumerate workspace targets."""
    return ["cargo", "metadata", "--no-deps", "--format-version", "1"]


def _selector_for_target(kinds: set[str], crate_types: set[str]) -> str | None:
    """Map Cargo metadata target categories to one explicit test selector."""
    for selector in ("test", "example", "bin"):
        if selector in kinds:
            return selector
    if kinds & _LIBRARY_TARGET_KINDS or crate_types & _LIBRARY_TARGET_KINDS:
        return "lib"
    return None


def _metadata_string_set(
    target: dict[str, object],
    field: str,
    package_name: str,
    target_name: str,
) -> set[str]:
    """Return one validated list-of-strings metadata field as a set."""
    raw_values = target.get(field, ())
    if not isinstance(raw_values, list) or not all(
        isinstance(value, str) for value in raw_values
    ):
        raise ValueError(
            f"Cargo target {package_name}/{target_name} has invalid {field}"
        )
    return set(raw_values)


def parse_workspace_targets(metadata_output: str) -> list[CargoTarget]:
    """Return exact default-tested targets for Cargo workspace members."""
    try:
        metadata = parse_json_bounded(metadata_output)
    except ValueError as exc:
        raise ValueError("Cargo metadata output is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Cargo metadata root must be an object")
    packages = metadata.get("packages")
    workspace_members = set(metadata.get("workspace_members", ()))
    if not isinstance(packages, list) or not workspace_members:
        raise ValueError("Cargo metadata did not contain a workspace package inventory")

    workspace_packages: dict[str, dict[str, object]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("Cargo metadata package entries must be objects")
        if package.get("id") not in workspace_members:
            continue
        package_name = package.get("name")
        if not isinstance(package_name, str) or not package_name:
            raise ValueError("Cargo workspace package has no valid name")
        if package_name in workspace_packages:
            raise ValueError(f"duplicate Cargo workspace package name: {package_name}")
        workspace_packages[package_name] = package
    if not workspace_packages:
        raise ValueError("Cargo metadata resolved no workspace member packages")

    targets: set[CargoTarget] = set()
    for package_name, package in workspace_packages.items():
        raw_targets = package.get("targets", ())
        if not isinstance(raw_targets, list):
            raise ValueError(f"Cargo package {package_name} has no valid target list")
        for target in raw_targets:
            if not isinstance(target, dict):
                raise ValueError(f"Cargo target in {package_name} must be an object")
            if not target.get("test", False):
                continue
            target_name = target.get("name")
            if not isinstance(target_name, str) or not target_name:
                raise ValueError(f"Cargo target in {package_name} has no valid name")
            kinds = _metadata_string_set(target, "kind", package_name, target_name)
            crate_types = _metadata_string_set(
                target,
                "crate_types",
                package_name,
                target_name,
            )
            selector = _selector_for_target(kinds, crate_types)
            if selector is None:
                if kinds and kinds <= _NON_DEFAULT_TARGET_KINDS:
                    continue
                rendered_kinds = ", ".join(sorted(kinds)) or "<empty>"
                rendered_crate_types = ", ".join(sorted(crate_types)) or "<empty>"
                raise ValueError(
                    f"unsupported default-tested Cargo target {package_name}/{target_name}: "
                    f"kind={rendered_kinds}; crate_types={rendered_crate_types}"
                )
            targets.add(CargoTarget(package_name, selector, target_name))
            if selector == "lib" and target.get("doctest", False):
                targets.add(CargoTarget(package_name, "doc", target_name))
    if not targets:
        raise ValueError("Cargo metadata did not expose any default-tested targets")
    return sorted(targets)


def _target_selector_arguments(target: CargoTarget) -> list[str]:
    """Return the exact Cargo selector arguments for one target."""
    if target.selector in {"lib", "doc"}:
        return [f"--{target.selector}"]
    return [f"--{target.selector}", target.target_name]


def cargo_list_command(target: CargoTarget) -> list[str]:
    """Return the command that lists ignored tests for exactly one target."""
    return [
        "cargo",
        "test",
        "--release",
        "-p",
        target.package,
        *_target_selector_arguments(target),
        "--",
        "--ignored",
        "--list",
    ]


def cargo_test_command(target: CargoTarget, test_name: str) -> list[str]:
    """Return the exact command for one ignored test in one Cargo target."""
    return [
        "cargo",
        "test",
        "--release",
        "-p",
        target.package,
        *_target_selector_arguments(target),
        test_name,
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]


def inventory_ignored_tests(targets: Sequence[CargoTarget]) -> list[IgnoredTest]:
    """List ignored tests target-by-target without collapsing duplicate names."""
    inventory: list[IgnoredTest] = []
    seen: set[str] = set()
    for target in targets:
        completed = run_bounded(
            cargo_list_command(target),
            operation=SubprocessOperation.CARGO_TEST_LIST,
            check=True,
            capture_output=True,
            text=True,
        )
        for test_name in parse_test_names(completed.stdout):
            identifier = f"{target.identity}::{test_name}"
            if identifier in seen:
                raise ValueError(f"duplicate target-qualified test: {identifier}")
            seen.add(identifier)
            inventory.append(IgnoredTest(identifier, target, test_name))
    return sorted(inventory)


def validated_skip_set(
    identifiers: Sequence[str],
    skipped: Iterable[str],
) -> frozenset[str]:
    """Validate unique exact dedicated-test identifiers against the inventory."""
    inventory = set(identifiers)
    requested = tuple(skipped)
    if len(set(requested)) != len(requested):
        raise ValueError("skipped test identifiers must be unique")
    missing = sorted(set(requested) - inventory)
    if missing:
        raise ValueError(
            "skipped tests were not found by exact target-qualified identifier: "
            + ", ".join(missing)
        )
    return frozenset(requested)


def partition_inventory(
    identifiers: Sequence[str],
    shard_count: int,
    skipped: Iterable[str] = (),
) -> tuple[tuple[str, ...], ...]:
    """Return an exhaustive disjoint partition with no silently empty shard."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    inventory = sorted(set(identifiers))
    if len(inventory) != len(identifiers):
        raise ValueError("target-qualified ignored-test identifiers must be unique")
    skip_set = validated_skip_set(inventory, skipped)
    retained = [identifier for identifier in inventory if identifier not in skip_set]
    shards = tuple(
        tuple(
            identifier
            for index, identifier in enumerate(retained)
            if index % shard_count == shard
        )
        for shard in range(shard_count)
    )
    empty = [str(index) for index, selected in enumerate(shards) if not selected]
    if empty:
        raise ValueError(
            "ignored Rust partition contains empty shards: " + ", ".join(empty)
        )
    return shards


def select_shard(
    identifiers: Sequence[str],
    shard: int,
    shard_count: int,
    skipped: Iterable[str] = (),
) -> list[str]:
    """Select one stable shard after exact dedicated-test exclusions."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard < shard_count:
        raise ValueError("shard must satisfy 0 <= shard < shard_count")
    return list(partition_inventory(identifiers, shard_count, skipped)[shard])


def run_shard(shard: int, shard_count: int, skipped: Sequence[str]) -> int:
    """Run one target-qualified ignored-test shard and return its process code."""
    metadata = run_bounded(
        cargo_metadata_command(),
        operation=SubprocessOperation.CARGO_METADATA,
        check=True,
        capture_output=True,
        text=True,
    )
    targets = parse_workspace_targets(metadata.stdout)
    inventory = inventory_ignored_tests(targets)
    if not inventory:
        raise RuntimeError("Cargo did not report any ignored workspace tests")

    by_identifier = {test.identifier: test for test in inventory}
    selected = select_shard(
        tuple(by_identifier),
        shard,
        shard_count,
        skipped,
    )
    print(
        f"ignored Rust shard {shard + 1}/{shard_count}: "
        f"{len(selected)} of {len(inventory)} target-qualified tests",
        flush=True,
    )
    for identifier in selected:
        test = by_identifier[identifier]
        command = cargo_test_command(test.target, test.test_name)
        print("+ " + " ".join(command), flush=True)
        completed = run_bounded(
            command,
            operation=SubprocessOperation.STATISTICAL_TEST,
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the sharding runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        type=int,
        required=True,
        help="zero-based shard index",
    )
    parser.add_argument(
        "--shards",
        type=int,
        required=True,
        help="total shard count",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help=(
            "exact package/selector/target::test_name identifier executed by "
            "a dedicated job"
        ),
    )
    args = parser.parse_args(argv)
    if args.shards < 1:
        parser.error("--shards must be at least 1")
    if not 0 <= args.shard < args.shards:
        parser.error("--shard must satisfy 0 <= shard < shards")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the requested ignored-test shard with redacted timeout evidence."""
    args = parse_args(argv)
    try:
        return run_shard(args.shard, args.shards, args.skip)
    except BoundedSubprocessTimeout as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 124


if __name__ == "__main__":
    sys.exit(main())
