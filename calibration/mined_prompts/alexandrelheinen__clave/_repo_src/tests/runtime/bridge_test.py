"""Tests for the process boundary.

These start the real runtime, so they build it when it is not built and skip
when cargo is absent.
"""

from pathlib import Path
from typing import Any

import pytest

from clave.runtime.bridge import Bridge, BridgeError, BridgePaths, locate_binary

ROOT = Path(__file__).resolve().parents[2]


def test_the_bridge_reports_a_runtime_that_was_never_built(tmp_path: Path) -> None:
    """The error says how to build it rather than what file is missing."""
    with pytest.raises(BridgeError, match="cargo build"):
        locate_binary(tmp_path)


def test_every_verdict_crosses_the_boundary_and_is_counted(
    tmp_path: Path,
    runtime_binary: Path,
    runtime_config: dict[str, Any],
    make_proposal: Any,
) -> None:
    """One round trip per proposal, and the runtime's own counts at the end."""
    paths = BridgePaths.under(tmp_path)
    with Bridge(runtime_binary, paths, runtime_config) as bridge:
        sorted_outcome = bridge.submit(make_proposal(0.0, 0.0, 0.95, 0.9))
        assert sorted_outcome.accepted
        assert sorted_outcome.channel == 1

        overridden = bridge.submit(make_proposal(0.0, 2.20, 0.95, 0.9))
        assert overridden.overridden
        assert overridden.check == "reach"

        rejected = bridge.submit(make_proposal(0.0, 0.0, 0.95, 0.1))
        assert rejected.accepted
        assert rejected.channel == 0
        assert rejected.reject_reason == "below_threshold"

        assert bridge.decisions_received == 2
        counters = bridge.counters()

    assert counters["proposals"] == 3
    assert counters["accepted"] == 1
    assert counters["overridden_reach"] == 1
    assert counters["rejected_low_confidence"] == 1
    assert counters["published"] == 2


def test_a_proposal_submitted_before_the_runtime_starts_is_refused(
    tmp_path: Path,
    runtime_config: dict[str, Any],
    make_proposal: Any,
) -> None:
    """A bridge that is not running says so rather than dropping the message."""
    bridge = Bridge(Path("unused"), BridgePaths.under(tmp_path), runtime_config)
    with pytest.raises(BridgeError, match="not running"):
        bridge.submit(make_proposal(0.0, 0.0, 0.95, 0.9))
