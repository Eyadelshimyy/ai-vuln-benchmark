#!/usr/bin/env python3
"""Print a digest-pinned image reference from the repository lock."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def locked_image(path: Path, entry: str) -> str:
    if not path.is_file():
        raise ValueError(f"image digest lock is missing: {path}")
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid image digest lock: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("image digest lock must be a JSON object")
    raw_item = cast(dict[str, Any], value).get(entry)
    if not isinstance(raw_item, dict):
        raise ValueError(f"image digest lock entry is missing: {entry}")
    item = cast(dict[str, Any], raw_item)
    image = item.get("image")
    digest = item.get("digest")
    if not isinstance(image, str) or not isinstance(digest, str):
        raise ValueError(f"image digest lock entry is malformed: {entry}")
    if not image or _DIGEST.fullmatch(digest) is None or digest == f"sha256:{'0' * 64}":
        raise ValueError(f"image digest lock entry is not digest-pinned: {entry}")
    return f"{image}@{digest}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("docker/image-digests.json"))
    parser.add_argument("--entry", required=True)
    args = parser.parse_args(argv)
    try:
        print(locked_image(args.lock, args.entry))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
