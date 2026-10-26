import pytest


pytestmark = [pytest.mark.api_flow, pytest.mark.side_effect]


def test_document_create_update_delete_eventually_reflects_in_search(api_client):
    user = api_client.register_user("side-effect-search")
    unique_term = f"side-effect-{api_client.run_id}"
    document = api_client.create_document(
        user,
        title=f"Search Fanout {api_client.run_id}",
        content=f"Initial searchable content {unique_term}",
    )
    document_id = document["id"]

    def search_for_initial_term():
        response = api_client.client.get(
            "/api/v1/search/",
            headers=user.auth_headers,
            params={"q": unique_term, "owner_id": user.id, "page": 1, "size": 10},
        )
        if response.status_code != 200:
            return []
        return response.json().get("items") or response.json().get("results") or []

    initial_results = api_client.poll_until(
        search_for_initial_term,
        lambda results: any(str(item.get("id")) == document_id for item in results),
        description="created document to appear in search index",
    )
    assert initial_results

    updated_term = f"updated-{unique_term}"
    update_response = api_client.client.patch(
        f"/api/v1/documents/{document_id}",
        headers=user.auth_headers,
        json={"content": f"Updated searchable content {updated_term}"},
    )
    assert update_response.status_code == 200, update_response.text

    def search_for_updated_term():
        response = api_client.client.get(
            "/api/v1/search/",
            headers=user.auth_headers,
            params={"q": updated_term, "owner_id": user.id, "page": 1, "size": 10},
        )
        if response.status_code != 200:
            return []
        return response.json().get("items") or response.json().get("results") or []

    api_client.poll_until(
        search_for_updated_term,
        lambda results: any(str(item.get("id")) == document_id for item in results),
        description="updated document content to appear in search index",
    )

    delete_response = api_client.client.delete(f"/api/v1/documents/{document_id}", headers=user.auth_headers)
    assert delete_response.status_code == 204, delete_response.text
    api_client.created_documents.remove(document_id)

    api_client.poll_until(
        search_for_updated_term,
        lambda results: not any(str(item.get("id")) == document_id for item in results),
        description="deleted document to disappear from search index",
    )


def test_file_share_eventually_creates_notification_and_audit_signal(api_client):
    owner = api_client.register_user("side-effect-file-owner")
    collaborator = api_client.register_user("side-effect-file-collaborator")

    upload_response = api_client.client.post(
        "/api/v1/files/upload",
        headers=owner.auth_headers,
        files={"file": ("notify.txt", b"notification body", "text/plain")},
    )
    assert upload_response.status_code == 201, upload_response.text
    file_id = upload_response.json()["file"]["id"]
    api_client.created_files.append(file_id)

    share_response = api_client.client.post(
        f"/api/v1/files/{file_id}/share",
        headers=owner.auth_headers,
        json={"shared_with": collaborator.id, "permission": "viewer", "shared_by": owner.id},
    )
    assert share_response.status_code == 201, share_response.text

    def collaborator_notifications():
        response = api_client.client.get(
            "/api/v1/notifications",
            headers=collaborator.auth_headers,
            params={"user_id": collaborator.id, "page": 1, "page_size": 20},
        )
        if response.status_code != 200:
            return []
        return response.json().get("data", [])

    api_client.poll_until(
        collaborator_notifications,
        lambda notifications: any(file_id in str(notification) for notification in notifications),
        description="file share notification for collaborator",
    )

    def file_audit_events():
        response = api_client.client.get(
            "/api/v1/audit/events",
            headers=owner.auth_headers,
            params={"resource": file_id, "page": 1, "size": 20},
        )
        if response.status_code != 200:
            return []
        return response.json().get("events", [])

    api_client.poll_until(
        file_audit_events,
        lambda events: any(event.get("resourceId") == file_id for event in events),
        description="file share audit event",
    )


def test_report_generation_reaches_terminal_state(api_client):
    user = api_client.register_user("side-effect-report")
    create_response = api_client.client.post(
        "/api/v1/reports",
        headers=user.auth_headers,
        json={
            "reportName": f"Async Report {api_client.run_id}",
            "category": "USAGE_ANALYTICS",
            "reportType": "CSV",
            "requestedBy": user.id,
            "dateFrom": api_client.iso_time(-7),
            "dateTo": api_client.iso_time(),
            "parameters": {"source": "api-flow-test"},
        },
    )
    assert create_response.status_code == 202, create_response.text
    report_id = create_response.json()["id"]
    api_client.created_reports.append(report_id)

    def report_status():
        response = api_client.client.get(f"/api/v1/reports/{report_id}", headers=user.auth_headers)
        assert response.status_code == 200, response.text
        return response.json()

    terminal_report = api_client.poll_until(
        report_status,
        lambda report: report.get("status") in {"COMPLETED", "FAILED"},
        timeout_seconds=60,
        description="report terminal status",
    )
    assert terminal_report["status"] in {"COMPLETED", "FAILED"}
