#!/usr/bin/env python3
"""Pull a digest-pinned image from the repository lock."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

try:
    from .print_locked_image import locked_image
except ImportError:
    from print_locked_image import locked_image

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("docker/image-digests.json"))
    parser.add_argument("--entry", required=True)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    try:
        image_ref = locked_image(args.lock, args.entry)
        digest = image_ref.rsplit("@", 1)[1]
        if _DIGEST.fullmatch(digest) is None:
            raise ValueError("locked image is not digest-pinned")
        result = subprocess.run(
            ["docker", "pull", image_ref],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "docker pull failed")
        record = {"entry": args.entry, "image": image_ref, "status": "pulled"}
        if args.record is not None:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(record, ensure_ascii=False))
    except (OSError, UnicodeDecodeError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
