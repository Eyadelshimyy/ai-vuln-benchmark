"""Write the finished distribution's member inventory beside its build row."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from release_artifact_transport import bundle_inventory


def main(row_path: Path, dist: Path, source_sha: str) -> None:
    lines = row_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError("expected one build evidence row")
    fields = lines[0].split("\t")
    if len(fields) != 7 or fields[1:3] != ["true", "clean-target-repeat-same-env"]:
        raise ValueError("build evidence row is not byte-verified")
    leg, _, _, digest, rebuilt, filename, build_env = fields
    if row_path.name != f"{leg}.tsv" or digest != rebuilt:
        raise ValueError("build evidence row differs from its leg or rebuild")
    inventory = bundle_inventory(dist / filename, leg, source_sha, build_env)
    if inventory["sha256"] != digest:
        raise ValueError("finished distribution differs from build evidence")
    row_path.with_suffix(".bundle.json").write_text(json.dumps(inventory, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: capture_release_bundle.py ROW DIST_DIR")
    main(Path(sys.argv[1]), Path(sys.argv[2]), os.environ["RELEASE_COMMIT"])
