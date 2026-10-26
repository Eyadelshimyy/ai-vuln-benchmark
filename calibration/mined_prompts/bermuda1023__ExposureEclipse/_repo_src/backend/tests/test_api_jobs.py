"""Tests for `/api/ert-jobs/run` + `/status/{id}` + `/{id}/cancel`."""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.enums import ErrorCode, JobStatus
from app.services import jobs as jobs_service


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    """Each test starts with a clean job registry."""
    jobs_service.reset_registry_for_tests()


async def _poll_until_terminal(
    client: AsyncClient,
    job_id: str,
    *,
    timeout_s: float = 5.0,
    interval_s: float = 0.05,
) -> dict:
    """Poll status until the job leaves queued/running, or timeout."""
    elapsed = 0.0
    while elapsed < timeout_s:
        resp = await client.get(f"/api/ert-jobs/status/{job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            return body
        await asyncio.sleep(interval_s)
        elapsed += interval_s
    raise AssertionError(f"Job {job_id} did not reach terminal state in {timeout_s}s")


@pytest.mark.asyncio
async def test_run_returns_202_with_job_id() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/ert-jobs/run",
            json={
                "serverName": "BERMUDA-SQL01",
                "edmDatabaseName": "Re_BER_27_Farmers_WS_EDM_01",
                "treatyYear": 2027,
                "currency": "USD",
                "peril": "WS",
                "aggregationLevels": ["COUNTRY", "STATE"],
                "rerun": False,
            },
        )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["jobId"].startswith("job-")
    assert body["status"] in (JobStatus.QUEUED.value, JobStatus.RUNNING.value)


@pytest.mark.asyncio
async def test_successful_job_eventually_completes() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        run_resp = await client.post(
            "/api/ert-jobs/run",
            json={
                "serverName": "BERMUDA-SQL01",
                "edmDatabaseName": "Re_BER_27_Farmers_WS_EDM_01",
                "treatyYear": 2027,
                "currency": "USD",
                "peril": "WS",
                "aggregationLevels": ["COUNTRY", "STATE"],
                "rerun": False,
            },
        )
        job_id = run_resp.json()["jobId"]
        final = await _poll_until_terminal(client, job_id)

    assert final["status"] == JobStatus.COMPLETED.value
    assert final["error"] is None
    assert len(final["outputTablesGenerated"]) > 0


@pytest.mark.asyncio
async def test_always_fails_job_reports_failure_with_technical_block() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        run_resp = await client.post(
            "/api/ert-jobs/run",
            json={
                "serverName": "BERMUDA-SQL01",
                "edmDatabaseName": "Re_BER_27_AlwaysFails_WS_EDM_01",
                "treatyYear": 2027,
                "currency": "USD",
                "peril": "WS",
                "aggregationLevels": ["COUNTRY"],
                "rerun": False,
            },
        )
        job_id = run_resp.json()["jobId"]
        final = await _poll_until_terminal(client, job_id)

    assert final["status"] == JobStatus.FAILED.value
    err = final["error"]
    assert err is not None
    assert err["emailSent"] is True
    tech = err["technical"]
    assert tech["databaseName"] == "Re_BER_27_AlwaysFails_WS_EDM_01"
    assert tech["procedureName"]
    assert tech["inputParameters"]


@pytest.mark.asyncio
async def test_unknown_job_returns_404_envelope() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/ert-jobs/status/job-does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == ErrorCode.JOB_NOT_FOUND.value


@pytest.mark.asyncio
async def test_duplicate_concurrent_submit_returns_existing_job_id() -> None:
    transport = ASGITransport(app=app)
    payload = {
        "serverName": "BERMUDA-SQL01",
        "edmDatabaseName": "Re_BER_27_Farmers_WS_EDM_01",
        "treatyYear": 2027,
        "currency": "USD",
        "peril": "WS",
        "aggregationLevels": ["COUNTRY"],
        "rerun": False,
    }
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/api/ert-jobs/run", json=payload)
        second = await client.post("/api/ert-jobs/run", json=payload)
        # Both should resolve to the same job (queued/running) before completion.
        assert first.status_code == 202
        assert second.status_code == 202
        if first.json()["status"] in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            assert first.json()["jobId"] == second.json()["jobId"]
