import pytest


pytestmark = pytest.mark.api_flow


def test_notification_listing_read_preferences_and_route_gaps(api_client):
    user = api_client.register_user("notification-flow")

    missing_user_response = api_client.client.get("/api/v1/notifications", headers=user.auth_headers)
    assert missing_user_response.status_code == 400

    list_response = api_client.client.get(
        "/api/v1/notifications",
        headers=user.auth_headers,
        params={"user_id": user.id, "page": 1, "page_size": 20},
    )
    assert list_response.status_code == 200, list_response.text
    data = list_response.json()
    assert data["page"] == 1
    assert "data" in data

    unread_response = api_client.client.get(
        "/api/v1/notifications/unread-count",
        headers=user.auth_headers,
        params={"user_id": user.id},
    )
    assert unread_response.status_code == 200, unread_response.text
    assert unread_response.json()["userId"] == user.id

    mark_all_response = api_client.client.put(
        "/api/v1/notifications/read-all",
        headers=user.auth_headers,
        params={"user_id": user.id},
    )
    assert mark_all_response.status_code == 200, mark_all_response.text
    assert "markedCount" in mark_all_response.json()

    missing_notification = api_client.client.get(
        "/api/v1/notifications/00000000-0000-0000-0000-000000000000",
        headers=user.auth_headers,
    )
    assert missing_notification.status_code == 404

    preferences_get = api_client.client.get(
        "/api/v1/preferences",
        headers=user.auth_headers,
        params={"user_id": user.id},
    )
    api_client.assert_gateway_route_available(preferences_get, "/api/v1/preferences")
    assert preferences_get.status_code == 200, preferences_get.text

    preferences_put = api_client.client.put(
        "/api/v1/preferences",
        headers=user.auth_headers,
        json={"userId": user.id, "eventType": "document_edited", "channels": ["IN_APP"]},
    )
    assert preferences_put.status_code == 204, preferences_put.text


def test_admin_health_users_features_and_permissions(api_client):
    user = api_client.register_user("admin-negative")

    health_response = api_client.client.get("/api/v1/admin/health/services", headers=user.auth_headers)
    api_client.assert_gateway_route_available(health_response, "/api/v1/admin")
    assert health_response.status_code in {200, 401, 403, 503}, health_response.text

    users_response = api_client.client.get("/api/v1/admin/users", headers=user.auth_headers)
    assert users_response.status_code in {401, 403}

    metrics_response = api_client.client.get("/api/v1/admin/metrics/summary", headers=user.auth_headers)
    assert metrics_response.status_code in {200, 401, 403}

    feature_create_response = api_client.client.post(
        "/api/v1/admin/features",
        headers=user.auth_headers,
        json={
            "key": f"api_flow_feature_{api_client.run_id}",
            "name": "API Flow Feature",
            "description": "Feature flag created by API flow test",
            "enabled": False,
        },
    )
    assert feature_create_response.status_code in {201, 401, 403, 422}, feature_create_response.text

    announcement_response = api_client.client.get(
        "/api/v1/admin/announcements",
        headers=user.auth_headers,
        params={"page": 1, "per_page": 10},
    )
    assert announcement_response.status_code in {200, 401, 403}

    audit_logs_response = api_client.client.get(
        "/api/v1/admin/audit-logs",
        headers=user.auth_headers,
        params={"page": 1, "per_page": 10},
    )
    assert audit_logs_response.status_code in {200, 401, 403}


def test_gateway_health_cors_request_id_and_unavailable_route_behavior(api_client):
    health_response = api_client.client.get("/health")
    assert health_response.status_code in {200, 503}

    request_id = f"api-flow-{api_client.run_id}"
    request_id_response = api_client.client.get("/health", headers={"X-Request-ID": request_id})
    assert request_id_response.status_code in {200, 503}

    cors_response = api_client.client.options(
        "/api/v1/auth/login",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, Authorization",
        },
    )
    assert cors_response.status_code in {200, 204}
    assert cors_response.headers.get("access-control-allow-origin") in {
        "http://localhost:3000",
        "*",
    }

    unknown_route_response = api_client.client.get("/api/v1/does-not-exist")
    assert unknown_route_response.status_code == 404
