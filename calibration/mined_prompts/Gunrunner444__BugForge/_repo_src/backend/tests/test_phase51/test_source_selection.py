"""Phase 51 truncation-aware source selection (owner Phase 4)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from app.discovery.bounty.campaign import BountyManifest
from app.discovery.bounty.source_selection import Priority, select_sources
from tests.test_phase50.phase50_support import MANIFEST_DATA


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "SelfTrustMulticall.sol").write_text(
        "contract SelfTrustMulticall { function f() external {} }"
    )
    (tmp_path / "Proxy.sol").write_text(
        "contract Proxy is UUPSUpgradeable { function _authorizeUpgrade(address) internal {} }"
    )
    (tmp_path / "Sensitive.sol").write_text(
        "contract S { function x() external { selfdestruct(payable(msg.sender)); } }"
    )
    (tmp_path / "Boring.sol").write_text("contract B { uint x; }")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "Dep.sol").write_text("contract Dep {}")
    return tmp_path


def _manifest(**overrides) -> BountyManifest:
    data = copy.deepcopy(MANIFEST_DATA)
    data.update(overrides)
    return BountyManifest.from_dict(data)


def test_scope_proxy_and_sensitive_rank_above_boring(repo: Path) -> None:
    sel = select_sources(repo, manifest=_manifest(), focus_contract="SelfTrustMulticall")
    by_path = {item.path: item.priority for item in sel.ranked}
    assert by_path["SelfTrustMulticall.sol"] is Priority.SCOPE
    assert by_path["Proxy.sol"] is Priority.PROXY
    assert by_path["Sensitive.sol"] is Priority.SECURITY_SENSITIVE
    assert by_path["Boring.sol"] is Priority.OTHER


def test_skip_dirs_are_excluded(repo: Path) -> None:
    sel = select_sources(repo, manifest=_manifest())
    assert not any("node_modules" in p for p in sel.selected)


def test_truncation_is_explicit_and_keeps_the_important_files(repo: Path) -> None:
    sel = select_sources(repo, manifest=_manifest(), focus_contract="SelfTrustMulticall", limit=2)
    assert sel.truncated is True
    assert sel.considered == 4  # node_modules excluded
    assert sel.selected == ("SelfTrustMulticall.sol", "Proxy.sol")
    # the dropped files are reported, and dropping does not imply they are safe
    assert "Boring.sol" in sel.dropped
    data = sel.as_dict()
    assert data["truncated"] is True
    assert "not implied to be clean" in data["note"]


def test_no_truncation_when_everything_fits(repo: Path) -> None:
    sel = select_sources(repo, manifest=_manifest(), limit=50)
    assert sel.truncated is False
    assert sel.dropped == ()


def test_missing_repo_root_is_empty_selection(tmp_path: Path) -> None:
    sel = select_sources(tmp_path / "nope", manifest=_manifest())
    assert sel.selected == () and sel.truncated is False
