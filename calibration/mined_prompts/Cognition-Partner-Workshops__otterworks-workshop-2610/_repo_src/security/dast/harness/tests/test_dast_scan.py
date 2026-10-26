"""What the scan records about itself, which is what the coverage gate grades."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dast_scan import request_recorder  # noqa: E402


def send(record, method: str, url: str, status: int = 200, **kwargs) -> None:
    """Deliver a response, since a request that was never answered is not coverage."""
    record(httpx.Response(status, request=httpx.Request(method, url, **kwargs)))


def test_requests_are_recorded_with_the_identity_that_made_them() -> None:
    exercised, record = request_recorder("http://localhost:8080")
    send(record, "GET", "http://localhost:8080/api/v1/documents")
    send(
        record,
        "GET",
        "http://localhost:8080/api/v1/documents",
        headers={"Authorization": "Bearer x"},
    )
    send(record, "GET", "http://localhost:8080/api/v1/documents")  # duplicate
    assert exercised == [
        {
            "method": "GET",
            "path": "/api/v1/documents",
            "authenticated": False,
            "probe": "",
            "status": 200,
        },
        {
            "method": "GET",
            "path": "/api/v1/documents",
            "authenticated": True,
            "probe": "",
            "status": 200,
        },
    ]


def test_a_request_to_another_origin_is_not_edge_coverage() -> None:
    """The direct-backend probe leaves the gateway on purpose; that is not coverage."""
    exercised, record = request_recorder("http://localhost:8080")
    send(record, "GET", "http://localhost:8082/api/v1/files")
    assert exercised == []


def test_a_path_routed_target_records_the_route_the_service_declares() -> None:
    """Without a tenant DNS zone the target is `<host>/<tenant>`; the route is not."""
    exercised, record = request_recorder("https://nlb.example.test/t-abc")
    send(record, "GET", "https://nlb.example.test/t-abc/api/v1/documents")
    send(record, "GET", "https://nlb.example.test/t-other/api/v1/documents")
    assert exercised == [
        {
            "method": "GET",
            "path": "/api/v1/documents",
            "authenticated": False,
            "probe": "",
            "status": 200,
        }
    ]


def test_the_probe_that_made_a_request_is_recorded_with_it() -> None:
    """Which probe reached a route is the difference between enumerated and attacked."""
    running = ["DAST-ANONYMOUS-ROUTE-SWEEP"]
    exercised, record = request_recorder("http://localhost:8080", lambda: running[0])
    send(record, "GET", "http://localhost:8080/api/v1/documents")
    running[0] = "DAST-BOLA-DOCUMENTS"
    send(record, "GET", "http://localhost:8080/api/v1/documents")
    assert [entry["probe"] for entry in exercised] == [
        "DAST-ANONYMOUS-ROUTE-SWEEP",
        "DAST-BOLA-DOCUMENTS",
    ]


def test_a_target_written_with_its_default_port_still_matches_its_own_traffic() -> None:
    """httpx normalizes host and port on the way out; the origin has to be normalized too."""
    exercised, record = request_recorder("https://API.example.test:443")
    send(record, "GET", "https://api.example.test/api/v1/files")
    assert [entry["path"] for entry in exercised] == ["/api/v1/files"]


def test_the_status_the_target_answered_with_is_recorded() -> None:
    """Hooked on the response: a route that only ever 502'd was reached by nothing."""
    exercised, record = request_recorder("http://localhost:8080")
    send(record, "DELETE", "http://localhost:8080/api/v1/documents/x", status=502)
    assert [entry["status"] for entry in exercised] == [502]
