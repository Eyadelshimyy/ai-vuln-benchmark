"""Drive scripts/ghcr_orphan_tag_prune.sh through a gh stub.

Only publish-time <40-hex>-tools tags older than the age floor, unpinned
by docker/image-digests.json and carrying no attestation may be deleted;
these tests pin each keep condition down before any real deletion runs.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPOSITORY = "VibeBB/fpga-agent"
SCRIPT = Path(__file__).parents[1] / "scripts/ghcr_orphan_tag_prune.sh"
LOCK_DIGEST = "sha256:" + "a" * 64
ORPHAN_DIGEST = "sha256:" + "b" * 64
ATTESTED_DIGEST = "sha256:" + "c" * 64
YOUNG_DIGEST = "sha256:" + "d" * 64


def _tsv_row(version_id: int, digest: str, days_old: float, tags: str) -> str:
    created = (datetime.now(UTC) - timedelta(days=days_old)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{version_id}\t{digest}\t{created}\t{tags}"


def _run(
    tmp_path: Path,
    *,
    rows: list[str],
    dry_run: str = "0",
    attest: str = "0",
    lock_digest: str = LOCK_DIGEST,
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$GH_STUB_CALLS"
case "$*" in
  *"packages/container"*)
    if [[ "$*" == *"-X DELETE"* ]]; then
      exit 0
    fi
    cat "$GH_STUB_VERSIONS"
    ;;
  *attestations*) printf '%s\\n' "$GH_STUB_ATTEST" ;;
  *) echo "unexpected gh call: $*" >&2; exit 1 ;;
esac
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    calls = tmp_path / "calls.log"
    versions = tmp_path / "versions.tsv"
    versions.write_text("\n".join(rows) + "\n", encoding="utf-8")
    lock = tmp_path / "image-digests.json"
    lock.write_text(
        json.dumps(
            {
                "fpga_tools": {
                    "image": "ghcr.io/vibebb/fpga-tools",
                    "digest": lock_digest,
                    "tag": "f" * 40 + "-tools",
                }
            }
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "GH_STUB_CALLS": str(calls),
            "GH_STUB_VERSIONS": str(versions),
            "GH_STUB_ATTEST": attest,
            "GITHUB_REPOSITORY_OWNER": "VibeBB",
            "GITHUB_REPOSITORY": REPOSITORY,
            "LOCK_FILE": str(lock),
            "GHCR_PRUNE_DRY_RUN": dry_run,
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    return result, calls.read_text(encoding="utf-8") if calls.exists() else ""


def test_old_unpinned_unattested_tag_is_deleted(tmp_path: Path) -> None:
    sha = "b" * 40
    result, log = _run(tmp_path, rows=[_tsv_row(101, ORPHAN_DIGEST, 30, f"{sha}-tools")])
    assert result.returncode == 0, result.stderr
    assert "packages/container/fpga-tools/versions/101" in log
    assert "-X DELETE" in log


def test_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    sha = "b" * 40
    result, log = _run(
        tmp_path,
        rows=[_tsv_row(101, ORPHAN_DIGEST, 30, f"{sha}-tools")],
        dry_run="1",
    )
    assert result.returncode == 0, result.stderr
    assert "-X DELETE" not in log
    assert "would delete" in result.stdout


def test_pinned_digest_is_kept(tmp_path: Path) -> None:
    sha = "a" * 40
    result, log = _run(tmp_path, rows=[_tsv_row(101, LOCK_DIGEST, 60, f"{sha}-tools")])
    assert result.returncode == 0, result.stderr
    assert "-X DELETE" not in log
    assert "pinned" in result.stdout


def test_young_tag_is_kept(tmp_path: Path) -> None:
    sha = "d" * 40
    result, log = _run(tmp_path, rows=[_tsv_row(101, YOUNG_DIGEST, 2, f"{sha}-tools")])
    assert result.returncode == 0, result.stderr
    assert "-X DELETE" not in log
    assert "younger" in result.stdout


def test_attested_tag_is_kept(tmp_path: Path) -> None:
    sha = "c" * 40
    result, log = _run(
        tmp_path,
        rows=[_tsv_row(101, ATTESTED_DIGEST, 30, f"{sha}-tools")],
        attest="1",
    )
    assert result.returncode == 0, result.stderr
    assert "-X DELETE" not in log
    assert "attestation" in result.stdout


def test_non_sha_tags_are_never_candidates(tmp_path: Path) -> None:
    result, log = _run(
        tmp_path,
        rows=[
            _tsv_row(101, ORPHAN_DIGEST, 90, "latest"),
            _tsv_row(102, "sha256:" + "e" * 64, 90, "sha256-attestation.sig"),
        ],
        attest="0",
    )
    assert result.returncode == 0, result.stderr
    assert "-X DELETE" not in log
    assert "attestations/" not in log
