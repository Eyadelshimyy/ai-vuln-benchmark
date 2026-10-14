import subprocess
from pathlib import Path

from rox_dox.features import FeatureFile, FeatureMap


def test_R9_feature_evidence_is_pinned_and_files_have_feature_peers(
    synthetic_feature_map: tuple[Path, str, FeatureMap],
) -> None:
    repo, commit, feature_map = synthetic_feature_map

    for feature in feature_map.features:
        for file in feature.files:
            for evidence in file.evidence:
                _assert_line_exists(repo, commit, evidence.path, evidence.line)
        for link in feature.table_links:
            _assert_line_exists(repo, commit, link.path, link.line)

        for file in feature.files:
            peers = [peer for peer in feature.files if peer.path != file.path]
            assert any(_shares_table_or_call(file, peer) for peer in peers), (
                f"{file.path} has no table or call relationship in feature {feature.id}"
            )


def _assert_line_exists(repo: Path, commit: str, path: str, line: int) -> None:
    source = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert 1 <= line <= len(source)


def _shares_table_or_call(file: FeatureFile, peer: FeatureFile) -> bool:
    if any(
        evidence.kind == "call" and evidence.to == peer.path
        for evidence in file.evidence
    ):
        return True
    if any(
        evidence.kind == "call" and evidence.to == file.path
        for evidence in peer.evidence
    ):
        return True
    return any(
        evidence.kind == "table"
        and any(
            peer_evidence.kind == "table" and peer_evidence.table == evidence.table
            for peer_evidence in peer.evidence
        )
        for evidence in file.evidence
    )
