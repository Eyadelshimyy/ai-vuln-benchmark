from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from gvas.interfaces.http.app import create_app


def test_health_endpoint() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    api_paths = [route.path for route in create_app().routes if isinstance(route, APIRoute)]
    assert api_paths == ["/healthz"]
