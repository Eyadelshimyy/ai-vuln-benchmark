from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

_COMMENT = (
    "File-level entries were removed for the 16 MiB attestation limit. "
    "The full Syft SBOM is attached to the workflow run as an artifact "
    "retained for 90 days."
)


def transform(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("SPDX document must be an object")
    document = cast(dict[str, Any], document)
    if document.get("spdxVersion") != "SPDX-2.3":
        raise ValueError("SPDX document must use SPDX-2.3")
    document_id = document.get("SPDXID")
    if not isinstance(document_id, str):
        raise ValueError("SPDX document is missing its SPDXID")

    package_entries = document.get("packages")
    if not isinstance(package_entries, list) or not package_entries:
        raise ValueError("SPDX document must contain at least one package")
    packages = cast(list[Any], package_entries)
    package_ids: set[str] = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("every package must have an SPDXID")
        package_entry = cast(dict[str, Any], package)
        package_id = package_entry.get("SPDXID")
        if not isinstance(package_id, str):
            raise ValueError("every package must have an SPDXID")
        package_ids.add(package_id)

    file_entries = document.get("files", [])
    if not isinstance(file_entries, list):
        raise ValueError("SPDX files must be an array")
    files = cast(list[Any], file_entries)
    file_ids: set[str] = set()
    for file_entry in files:
        if not isinstance(file_entry, dict):
            raise ValueError("every file entry must have an SPDXID")
        file_record = cast(dict[str, Any], file_entry)
        file_id = file_record.get("SPDXID")
        if not isinstance(file_id, str):
            raise ValueError("every file entry must have an SPDXID")
        file_ids.add(file_id)
    document.pop("files", None)

    relationship_entries = document.get("relationships", [])
    if not isinstance(relationship_entries, list):
        raise ValueError("SPDX relationships must be an array")
    relationships = cast(list[Any], relationship_entries)
    kept_relationships: list[dict[str, Any]] = []
    for relationship in relationships:
        if not isinstance(relationship, dict):
            raise ValueError("every relationship must be an object")
        relationship_record = cast(dict[str, Any], relationship)
        source = relationship_record.get("spdxElementId")
        target = relationship_record.get("relatedSpdxElement")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("every relationship must reference SPDXIDs")
        if source not in file_ids and target not in file_ids:
            kept_relationships.append(relationship_record)
    if "relationships" in document:
        document["relationships"] = kept_relationships

    allowed_ids = package_ids | {document_id, "NOASSERTION", "NONE"}
    for relationship in kept_relationships:
        for identifier in (
            relationship["spdxElementId"],
            relationship["relatedSpdxElement"],
        ):
            if identifier not in allowed_ids:
                raise ValueError(f"unknown SPDXID {identifier!r}")

    existing_comment = document.get("comment", "")
    if not isinstance(existing_comment, str):
        raise ValueError("SPDX document comment must be a string")
    document["comment"] = "\n\n".join(
        comment for comment in (existing_comment, _COMMENT) if comment
    )
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        result = transform(document)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"package_level_sbom.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
