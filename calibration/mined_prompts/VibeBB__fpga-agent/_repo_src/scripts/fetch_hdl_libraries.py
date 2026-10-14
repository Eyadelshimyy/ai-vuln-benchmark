"""Fetch pinned external HDL libraries into ``third_party/``.

The registry in ``hdl_libraries.json`` maps a library name to its pinned
upstream URL and revision. Checkouts are gitignored and referenced
unmodified from contract ``libraries[]`` entries. Run on a networked
host before starting the network-isolated tools container.

Usage:
    fetch_hdl_libraries.py [--dest DIR] [name ...]   fetch registry entries
    fetch_hdl_libraries.py --contract CONTRACT        fetch what a contract declares

``--dest`` defaults to the repo's ``third_party/``. With ``--contract``
each declared library is cloned at ``source_url``/``revision`` into the
contract-relative ``root`` path, so agents working in a product workspace
fetch into that workspace instead of this repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

REGISTRY = Path(__file__).resolve().parent / "hdl_libraries.json"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "third_party"


def _git(target: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(target), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def fetch(name: str, url: str, revision: str, target: Path) -> str:
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", url, str(target)], check=True)
    if _git(target, "rev-parse", "HEAD") != revision:
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--quiet", "origin", revision], check=True
        )
        _git(target, "checkout", "--quiet", "--detach", revision)
    head = _git(target, "rev-parse", "HEAD")
    if head != revision:
        raise RuntimeError(f"{name}: checkout is at {head}, expected {revision}")
    if (target / ".gitmodules").exists():
        subprocess.run(
            [
                "git",
                "-C",
                str(target),
                "submodule",
                "update",
                "--init",
                "--recursive",
                "--quiet",
            ],
            check=True,
        )
    return f"{name} {head} at {target}"


def _load_registry() -> dict[str, dict[str, str]]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{REGISTRY} is not a JSON object")
    return cast(dict[str, dict[str, str]], data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="registry names to fetch (default: all)")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="checkout parent dir")
    parser.add_argument("--contract", type=Path, help="fetch the libraries a contract declares")
    args = parser.parse_args(argv)

    if args.contract:
        contract_path = args.contract.resolve()
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        for lib in contract.get("libraries", []):
            target = (contract_path.parent / lib["root"]).resolve()
            print(fetch(lib["name"], lib["source_url"], lib["revision"], target))
        return 0

    registry = _load_registry()
    names = args.names or list(registry)
    unknown = [n for n in names if n not in registry]
    if unknown:
        known = ", ".join(sorted(registry))
        parser.error(f"unknown libraries {unknown}; registry knows: {known}")
    for name in names:
        entry = registry[name]
        print(fetch(name, entry["url"], entry["revision"], args.dest / name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
