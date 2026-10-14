from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_level_sbom.py"
COMMENT = (
    "File-level entries were removed for the 16 MiB attestation limit. "
    "The full Syft SBOM is attached to the workflow run as an artifact "
    "retained for 90 days."
)


def _document() -> dict[str, Any]:
    return {
        "spdxVersion": "SPDX-2.3",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "Café packages",
        "creationInfo": {"creators": ["Tool: Syft"], "created": "2026-10-02"},
        "packages": [
            {"name": "package-a", "SPDXID": "SPDXRef-Package-a"},
            {"name": "package-b", "SPDXID": "SPDXRef-Package-b"},
        ],
        "files": [{"fileName": "src/a.py", "SPDXID": "SPDXRef-File-a"}],
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Package-a",
            },
            {
                "spdxElementId": "SPDXRef-Package-a",
                "relationshipType": "DEPENDENCY_OF",
                "relatedSpdxElement": "SPDXRef-Package-b",
            },
            {
                "spdxElementId": "SPDXRef-Package-a",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": "SPDXRef-File-a",
            },
            {
                "spdxElementId": "SPDXRef-File-a",
                "relationshipType": "GENERATED_FROM",
                "relatedSpdxElement": "SPDXRef-Package-a",
            },
        ],
        "hasExtractedLicensingInfos": [{"licenseId": "LicenseRef-test"}],
        "comment": "Existing document comment",
    }


_RunResult = tuple[subprocess.CompletedProcess[str], Path]


def _run(tmp_path: Path, document: dict[str, Any]) -> _RunResult:
    input_path = tmp_path / "input.spdx.json"
    output_path = tmp_path / "output.spdx.json"
    input_path.write_text(json.dumps(document), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(input_path), str(output_path)],
        capture_output=True,
        check=False,
        text=True,
    )
    return result, output_path


def test_removes_file_entries_and_keeps_package_relationships(
    tmp_path: Path,
) -> None:
    original = _document()
    result, output_path = _run(tmp_path, original)

    assert result.returncode == 0, result.stderr
    serialized = output_path.read_text(encoding="utf-8")
    transformed = json.loads(serialized)
    expected = dict(original)
    expected.pop("files")
    expected["relationships"] = original["relationships"][:2]
    expected["comment"] = f"Existing document comment\n\n{COMMENT}"
    assert transformed == expected
    assert "SPDXRef-File-a" not in serialized
    assert "Café" in serialized
    assert r"\u00e9" not in serialized
    compact = json.dumps(transformed, ensure_ascii=False, separators=(",", ":"))
    assert serialized == compact


def test_rejects_dangling_relationship_reference(tmp_path: Path) -> None:
    document = _document()
    document["relationships"].append(
        {
            "spdxElementId": "SPDXRef-Package-a",
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": "SPDXRef-Missing",
        }
    )
    result, output_path = _run(tmp_path, document)

    assert result.returncode != 0
    assert "unknown SPDXID" in result.stderr
    assert not output_path.exists()


def test_rejects_empty_packages(tmp_path: Path) -> None:
    document = _document()
    document["packages"] = []
    document["relationships"] = []
    result, output_path = _run(tmp_path, document)

    assert result.returncode != 0
    assert "at least one package" in result.stderr
    assert not output_path.exists()


def test_rejects_wrong_spdx_version(tmp_path: Path) -> None:
    document = _document()
    document["spdxVersion"] = "SPDX-2.2"
    result, output_path = _run(tmp_path, document)

    assert result.returncode != 0
    assert "SPDX-2.3" in result.stderr
    assert not output_path.exists()
