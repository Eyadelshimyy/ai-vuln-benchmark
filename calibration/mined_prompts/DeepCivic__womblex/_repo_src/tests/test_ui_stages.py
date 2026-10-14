"""Tests for the console's downstream-stage dispatch (issue 5, part 3).

Kept out of ``test_ui.py`` (already past the 750-line file cap) for the same
reason ``test_ui_resources.py`` is: every case here builds its own
``TestClient`` and needs none of that file's shared fixtures.

The behaviour under test is that the console is a *skin over the CLI*: the
press writes exactly the rows ``womblex enqueue-stages`` writes, with the stage
list derived from the posted config by the one shared gate — never re-decided
here, and never widened past ``DOWNSTREAM_STAGES``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Self

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from womblex.pipeline_order import DOWNSTREAM_STAGES
from womblex.ui.app import create_app


class _FakeQueue:
    """A `RunBoard` stand-in: the dispatch path must not need Postgres."""

    last: _FakeQueue | None = None
    schema_ensured = False

    def __init__(self, dsn: str, **_kw: object) -> None:
        self.dsn = dsn
        self.staged: tuple[str, list[str], str, int] | None = None
        _FakeQueue.last = self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def enqueue_stages(
        self, run_id: str, stages: list[str], shard_prefix: str, *, max_attempts: int = 3,
        owner: str | None = None,
    ) -> int:
        self.staged = (run_id, stages, shard_prefix, max_attempts)
        return len(stages)


@pytest.fixture(autouse=True)
def _fake_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch provisions the DBOS schema through a module function; record, don't run it."""
    _FakeQueue.schema_ensured = False
    monkeypatch.setattr(
        "womblex.cloud.dispatch.ensure_schema",
        lambda _dsn: setattr(_FakeQueue, "schema_ensured", True),
    )


def _client(tmp_path: Path, **kw: object) -> TestClient:
    """A console wired to dispatch: store + queue. No ingest — stages read shards."""
    settings: dict[str, object] = {
        "store_uri": str(tmp_path / "store"), "db_dsn": "postgresql://x/y",
    }
    settings.update(kw)
    return TestClient(create_app(**settings))  # type: ignore[arg-type]


#: Three stages spread across the pipeline, so the assertion is about *order*
#: and not just membership. `money` is switched off explicitly: it defaults on,
#: and leaving it would make these cases read as if the gate were ignoring the
#: config rather than honouring its defaults.
_CONFIG = {
    "chunking": {"enabled": True},
    "enrichment": {"enabled": True},
    "embedding": {"enabled": True},
    "money": {"enabled": False},
}


class TestStageDispatchGuards:
    """The same guard `POST /enqueue` answers to — minus ingest."""

    def test_conflict_without_a_store(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path, db_dsn="postgresql://x/y"))
        resp = client.post("/api/execute/stages", json={"run_id": "run-1", "config": _CONFIG})
        assert resp.status_code == 409

    def test_conflict_without_a_queue(self, tmp_path: Path) -> None:
        client = TestClient(create_app(store_uri=str(tmp_path / "store")))
        resp = client.post("/api/execute/stages", json={"run_id": "run-1", "config": _CONFIG})
        assert resp.status_code == 409

    def test_no_ingest_location_is_not_a_blocker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stages read shards the store already holds; they never look at ingest.

        A run whose ingest location has since been unset (or was never set on
        this console) must still be finishable — hence `_guard(needs_ingest=False)`.
        """
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        resp = _client(tmp_path).post(
            "/api/execute/stages", json={"run_id": "run-1", "config": _CONFIG},
        )
        assert resp.status_code == 200

    def test_unsafe_run_id_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`is_safe_run_id` keeps the `runs/<id>/documents` join contained."""
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        _FakeQueue.last = None
        resp = _client(tmp_path).post(
            "/api/execute/stages", json={"run_id": "../../etc", "config": _CONFIG},
        )
        assert resp.status_code == 400
        assert _FakeQueue.last is None  # refused before the queue was opened

    def test_a_missing_run_id_is_refused_at_the_boundary(self, tmp_path: Path) -> None:
        """Never minted here: stages run over shards that already exist."""
        assert _client(tmp_path).post(
            "/api/execute/stages", json={"config": _CONFIG},
        ).status_code == 422


class TestStageDispatchPlan:
    """One press, the config-gated list, in pipeline order."""

    def test_dispatches_the_enabled_stages_in_pipeline_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        resp = _client(tmp_path).post(
            "/api/execute/stages",
            json={"run_id": "run-exec", "config": _CONFIG, "max_attempts": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        # enrich before chunk (AI-chunking reuse), embed after both.
        assert body["stages"] == ["enrich", "chunk", "embed"]
        assert body["run_id"] == "run-exec"
        assert body["newly_enqueued"] == 3
        assert body["shard_prefix"] == "runs/run-exec/documents"

        queue = _FakeQueue.last
        assert queue is not None and queue.schema_ensured
        assert queue.staged == ("run-exec", ["enrich", "chunk", "embed"],
                                "runs/run-exec/documents", 5)

    def test_a_disabled_stage_is_not_dispatched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        body = _client(tmp_path).post(
            "/api/execute/stages",
            json={
                "run_id": "run-1",
                "config": {"chunking": {"enabled": True}, "money": {"enabled": False}},
            },
        ).json()
        assert body["stages"] == ["chunk"]

    def test_pii_and_quality_are_never_dispatched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Masking is irreversible: it must not run off a flag left on in a
        copied config. The bound is `DOWNSTREAM_STAGES`, so no request widens it."""
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        body = _client(tmp_path).post(
            "/api/execute/stages",
            json={
                "run_id": "run-1",
                "config": {
                    "chunking": {"enabled": True},
                    "money": {"enabled": False},
                    "pii": {"enabled": True},
                    "quality": {"enabled": True},
                },
            },
        ).json()
        assert "pii" not in body["stages"]
        assert "quality" not in body["stages"]
        assert set(body["stages"]) <= set(DOWNSTREAM_STAGES)

    def test_a_config_enabling_nothing_is_a_400_not_an_empty_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A press that silently dispatched nothing reads as a broken button."""
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        resp = _client(tmp_path).post(
            "/api/execute/stages",
            json={
                "run_id": "run-1",
                "config": {"chunking": {"enabled": False}, "money": {"enabled": False}},
            },
        )
        assert resp.status_code == 400
        assert "no downstream stages" in resp.json()["detail"]

    def test_an_invalid_config_is_a_400_carrying_pydantics_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        resp = _client(tmp_path).post(
            "/api/execute/stages",
            json={"run_id": "run-1", "config": {"chunking": {"chunk_size": "not-a-number"}}},
        )
        assert resp.status_code == 400
        assert isinstance(resp.json()["detail"], list)


def test_dispatch_bounds_the_queue_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead DSN must fail the press, not hang the request thread."""
    seen: dict[str, object] = {}

    class _Capturing(_FakeQueue):
        def __init__(self, dsn: str, **kw: object) -> None:
            seen.update(kw)
            super().__init__(dsn, **kw)

    monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _Capturing)
    resp = _client(tmp_path).post(
        "/api/execute/stages", json={"run_id": "run-1", "config": _CONFIG},
    )
    assert resp.status_code == 200
    assert seen == {"connect_timeout": 5.0}


def test_a_layer_no_stage_produces_is_refused_before_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
    _FakeQueue.last = None
    config = {**_CONFIG, "processing": {"text_source": "spellfix"}, "spellfix": {"enabled": False}}
    resp = _client(tmp_path).post("/api/execute/stages", json={"run_id": "run-1", "config": config})
    assert resp.status_code == 400
    assert "[input_missing] chunk: .spellfix_text.parquet" in resp.json()["detail"]
    assert _FakeQueue.last is None and not _FakeQueue.schema_ensured
