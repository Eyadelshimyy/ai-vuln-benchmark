"""Smoke-test the real install path for the FPGA plugin."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_plugin_load", REPO_ROOT / "scripts" / "check_plugin_load.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--ref", required=True, help="branch, tag, or commit")
    parser.add_argument("--repo-path", required=True, help="plugin subdirectory")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="fpga-plugin-install-") as tmp:
        tmpdir = Path(tmp)
        os.environ["OH_PERSISTENCE_DIR"] = str(tmpdir / "persistence")
        from openhands.sdk.plugin import (  # pyright: ignore[reportMissingImports,reportMissingModuleSource]
            install_plugin,
        )

        checker = _load_checker()
        want_sha = bool(SHA_RE.fullmatch(args.ref))
        # The SDK supports every ref form: 40-hex SHAs get a full clone +
        # checkout, branch/tag names a shallow --branch clone. Passing the
        # sha through keeps resolved_ref pinned to the release commit —
        # installing default-branch HEAD races with concurrent merges.
        fetch_ref = args.ref
        try:
            info = install_plugin(
                f"github:{args.repo}",
                ref=fetch_ref,
                repo_path=args.repo_path,
                installed_dir=tmpdir / "installed",
            )
        except Exception as exc:
            print(f"install_plugin failed: {exc}")
            return 1
        if want_sha and info.resolved_ref != args.ref:
            print(f"resolved ref {info.resolved_ref!r} != requested {args.ref}")
            return 1
        reasons = checker.check_plugin(info.install_path)
    if reasons:
        for reason in reasons:
            print(reason)
        return 1
    print(
        f"install-smoke OK: github:{args.repo}@{args.ref} "
        f"repo-path={args.repo_path} resolved={info.resolved_ref}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
