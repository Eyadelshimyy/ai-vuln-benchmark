"""Docker integration for the runtime engine.

Runs only when Docker exists and `security_agent_runtime_image` names an image
that is already present locally. Otherwise the engine reports UNAVAILABLE and
the live assertions are skipped. Nothing here pulls an image, uses the network,
or needs host Foundry. The image must implement bugforge-runtime-v1 and honor
`bugforge-runtime --input <manifest> --output <file>`. A repository-owned
contract fixture is in tests/fixtures/runtime_image.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.discovery.runtime import (
    RuntimeEngine,
    _docker_present,
    _local_image,
    _runtime_image,
)
from app.discovery.capabilities import EngineAvailability, ResultStatus
from tests.test_phase48.phase48_support import request


def _live() -> bool:
    return bool(_runtime_image()) and _docker_present() and _local_image()


def test_unavailable_runtime_is_reported_without_fabricating_success(tmp_path: Path) -> None:
    engine = RuntimeEngine()
    if _live():
        assert engine.availability() is EngineAvailability.AVAILABLE
        return
    assert engine.availability() is EngineAvailability.UNAVAILABLE
    result = engine.start_campaign(request(tmp_path))
    assert result.status is ResultStatus.UNAVAILABLE
    assert result.executed is False
    assert result.runtime_evidence is None
    assert result.metadata["observation_status"] == "unavailable"


@pytest.mark.skipif(not _live(), reason="UNAVAILABLE: no local Docker runtime image configured")
def test_real_runtime_image_binds_the_requested_transaction(tmp_path: Path) -> None:
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.executed is True
    assert result.status is ResultStatus.INGESTED, result.oracle_explanation
    assert result.metadata["transaction_index"] == "1"
    assert result.metadata["network"] == "none"
    assert result.metadata["verified"] == "false"
    assert result.runtime_evidence.request_identity == result.metadata["request_identity"]
    assert result.to_evidence().contributes_to_verification is False
