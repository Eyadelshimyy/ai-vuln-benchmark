"""Per-test isolation for Phase 51/52 campaigns.

Each test gets its own shared-in-memory SQLite database and a fresh process-wide
service, so persisted campaigns, control state, and background jobs from one test
never leak into the next. The database is real: the Phase 49 SqlStore and the
campaign record store run end to end against it.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

import app.discovery.bounty.service as service_module
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def isolated_campaign_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    name = f"phase51_{uuid.uuid4().hex}"
    url = f"sqlite:///file:{name}?mode=memory&cache=shared&uri=true"
    monkeypatch.setattr(get_settings(), "bounty_campaign_database_url", url, raising=False)
    monkeypatch.setattr(service_module, "_SERVICE", None, raising=False)
    yield
    monkeypatch.setattr(service_module, "_SERVICE", None, raising=False)
