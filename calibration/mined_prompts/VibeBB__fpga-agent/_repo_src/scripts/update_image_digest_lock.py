#!/usr/bin/env python3
"""Create or update one published image entry in the digest lock."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ENTRIES = {"fpga_tools"}
_IMAGES = {"fpga_tools": "ghcr.io/vibebb/fpga-tools"}


def _is_https_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and hostname is not None
        and not any(char.isspace() for char in parsed.netloc)
        and "\r" not in value
        and "\n" not in value
    )


def _digest(value: str) -> str:
    if _DIGEST.fullmatch(value) is None or value == f"sha256:{'0' * 64}":
        raise ValueError("digest must be a non-placeholder sha256 digest")
    return value


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid image lock: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("image lock must be a JSON object")
    payload = cast(dict[str, Any], value)
    unknown = set(payload) - _ENTRIES
    if unknown:
        raise ValueError(f"unknown image lock entries: {sorted(unknown)}")
    for name, item in payload.items():
        if not isinstance(item, dict):
            raise ValueError(f"image lock entry is malformed: {name}")
        fields = cast(dict[str, Any], item)
        if not isinstance(fields.get("image"), str) or not fields["image"]:
            raise ValueError(f"image lock entry lacks image: {name}")
        if fields.get("digest") is None and fields.get("tag") is None:
            if set(fields) == {"image", "digest", "tag"}:
                continue
            raise ValueError(f"uninitialized image lock entry is malformed: {name}")
        for field in ("tag", "digest"):
            if not isinstance(fields.get(field), str) or not fields[field]:
                raise ValueError(f"image lock entry lacks {field}: {name}")
        _digest(cast(str, fields["digest"]))
    return payload


def update_lock(
    path: Path,
    *,
    entry: str,
    image: str,
    tag: str,
    digest: str,
    published_at: str,
    workflow_run: str,
    dockerfile: str,
    tools: dict[str, str],
    attestation: str | None = None,
    sbom_attestation: str | None = None,
) -> bool:
    if entry not in _ENTRIES:
        raise ValueError(f"unknown image lock entry: {entry}")
    if image != _IMAGES[entry]:
        raise ValueError(f"unexpected image for {entry}: {image}")
    if not image or "@" in image or ":" in image.rsplit("/", 1)[-1]:
        raise ValueError("image must be an untagged repository name")
    if not tag:
        raise ValueError("tag must not be empty")
    _digest(digest)
    try:
        datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("published_at must be ISO-8601") from exc
    if not workflow_run or not dockerfile:
        raise ValueError("workflow_run and dockerfile must not be empty")
    if attestation is not None and not _is_https_url(attestation):
        raise ValueError("attestation must be an HTTPS URL")
    if sbom_attestation is not None and not _is_https_url(sbom_attestation):
        raise ValueError("sbom_attestation must be an HTTPS URL")
    if not tools or any(not key or not value for key, value in tools.items()):
        raise ValueError("tools must contain non-empty string values")
    payload = _load(path)
    before = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    image_entry: dict[str, Any] = {
        "digest": digest,
        "dockerfile": dockerfile,
        "image": image,
        "published_at": published_at,
        "tag": tag,
        "tools": dict(sorted(tools.items())),
        "workflow_run": workflow_run,
    }
    if attestation is not None:
        image_entry["attestation"] = attestation
    if sbom_attestation is not None:
        image_entry["sbom_attestation"] = sbom_attestation
    payload[entry] = image_entry
    after = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if before == after:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(after, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--published-at", required=True)
    parser.add_argument("--workflow-run", required=True)
    parser.add_argument("--dockerfile", required=True)
    parser.add_argument("--attestation")
    parser.add_argument("--sbom-attestation")
    parser.add_argument("--tools-json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value: Any = json.loads(args.tools_json.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in cast(dict[Any, Any], value).items()
        ):
            raise ValueError("tools JSON must be an object of string values")
        changed = update_lock(
            args.lock,
            entry=args.entry,
            image=args.image,
            tag=args.tag,
            digest=args.digest,
            published_at=args.published_at,
            workflow_run=args.workflow_run,
            dockerfile=args.dockerfile,
            tools=cast(dict[str, str], value),
            attestation=args.attestation,
            sbom_attestation=args.sbom_attestation,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("UPDATED" if changed else "UNCHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
