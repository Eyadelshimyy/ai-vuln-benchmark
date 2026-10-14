from __future__ import annotations

from pathlib import Path

from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).parent.resolve()
ROOT_PACKAGE_DIR = "ellements"
ELLEMENTS_SOURCE_ROOTS = [
    "ellements-agents/src",
    "ellements-benchmarking/src",
    "ellements-cli/src",
    "ellements-core/src",
    "ellements-domain-specific/src",
    "ellements-execution/src",
    "ellements-fslm/src",
    "ellements-reporting/src",
    "ellements-standard-tools/src",
]


def _discover_packages() -> tuple[list[str], dict[str, str]]:
    packages: set[str] = {"ellements"}
    package_dir: dict[str, str] = {"ellements": ROOT_PACKAGE_DIR}

    for relative_root in ELLEMENTS_SOURCE_ROOTS:
        root = ROOT / relative_root
        for package in find_namespace_packages(
            where=str(root),
            include=["ellements", "ellements.*"],
        ):
            if package in packages:
                continue
            package_path = root / Path(package.replace(".", "/"))
            package_dir[package] = str(package_path.relative_to(ROOT))
            packages.add(package)

    return sorted(packages), package_dir


DISCOVERED_PACKAGES, DISCOVERED_PACKAGE_DIR = _discover_packages()

setup(
    packages=DISCOVERED_PACKAGES,
    package_dir=DISCOVERED_PACKAGE_DIR,
)
