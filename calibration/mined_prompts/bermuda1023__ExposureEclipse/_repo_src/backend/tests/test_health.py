"""Phase 0 smoke test — /api/health responds and enums import cleanly."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.models import enums


def test_health_returns_ok() -> None:
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "peril-vista-backend"
    assert "dataProvider" in body


def test_canonical_enums_present() -> None:
    """Sanity-check: every enum group from CONTRACTS.md is importable from one module."""
    for name in (
        "MetricKey",
        "Measure",
        "AggregationLevel",
        "ErtStatus",
        "CombinationMethod",
        "Peril",
        "OccupancySegment",
        "JobStatus",
        "PortfolioScope",
        "YoyStatus",
        "WarningCode",
        "WarningSeverity",
        "ErrorCode",
        "GeocodingQuality",
        "DistanceToCoastBand",
        "YearBuiltBand",
    ):
        assert hasattr(enums, name), f"Missing enum: {name}"

    assert enums.CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN.value == (
        "MAX_ACROSS_PERILS_AT_VIEW_GRAIN"
    )
    assert enums.JobStatus.RUNNING.value == "running"
