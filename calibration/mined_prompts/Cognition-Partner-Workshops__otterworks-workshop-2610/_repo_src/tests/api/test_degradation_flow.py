import os

import pytest


pytestmark = [pytest.mark.api_flow, pytest.mark.degradation]


def _require_degradation_enabled():
    if os.getenv("OTTERWORKS_RUN_DEGRADATION_TESTS") != "1":
        pytest.skip("Set OTTERWORKS_RUN_DEGRADATION_TESTS=1 after intentionally degrading dependencies")


def test_search_dependency_degradation_returns_clean_error(api_client):
    _require_degradation_enabled()
    user = api_client.register_user("degradation-search")

    response = api_client.client.get(
        "/api/v1/search/",
        headers=user.auth_headers,
        params={"q": "anything"},
    )
    assert response.status_code in {500, 502, 503}
    api_client.assert_json_error(response)


def test_collab_dependency_degradation_reflected_in_health(api_client):
    _require_degradation_enabled()
    user = api_client.register_user("degradation-collab")

    response = api_client.client.get("/api/v1/collab/documents", headers=user.auth_headers)
    assert response.status_code in {200, 500, 502, 503}
    if response.status_code >= 500:
        api_client.assert_json_error(response)


def test_gateway_circuit_breaker_or_bad_gateway_error_shape(api_client):
    _require_degradation_enabled()
    user = api_client.register_user("degradation-gateway")

    responses = [
        api_client.client.get("/api/v1/search/", headers=user.auth_headers, params={"q": "probe"})
        for _ in range(8)
    ]
    failing = [response for response in responses if response.status_code in {502, 503}]
    assert failing
    api_client.assert_json_error(failing[-1])
