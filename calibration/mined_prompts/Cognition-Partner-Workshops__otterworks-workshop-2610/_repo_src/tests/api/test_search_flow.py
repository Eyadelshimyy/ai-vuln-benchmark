import pytest
import uuid


pytestmark = pytest.mark.api_flow


def test_search_index_query_suggest_advanced_and_delete_flow(api_client):
    user = api_client.register_user("search-flow")
    document_id = str(uuid.uuid4())
    title = f"Unique Search Flow {api_client.run_id}"

    index_response = api_client.client.post(
        "/api/v1/search/index/document",
        headers=user.auth_headers,
        json={
            "id": document_id,
            "title": title,
            "content": "otterworks unique searchable content",
            "owner_id": user.id,
            "type": "document",
            "created_at": api_client.iso_time(),
            "updated_at": api_client.iso_time(),
        },
    )
    assert index_response.status_code == 201, index_response.text
    api_client.indexed_documents.append(document_id)

    search_response = api_client.client.get(
        "/api/v1/search/",
        headers=user.auth_headers,
        params={"q": "unique searchable", "owner_id": user.id, "page": 1, "size": 10},
    )
    assert search_response.status_code == 200, search_response.text
    search_data = search_response.json()
    assert search_data["total"] >= 1

    suggest_response = api_client.client.get(
        "/api/v1/search/suggest",
        headers=user.auth_headers,
        params={"q": "un"},
    )
    assert suggest_response.status_code == 200, suggest_response.text
    assert "suggestions" in suggest_response.json()

    advanced_response = api_client.client.post(
        "/api/v1/search/advanced",
        headers=user.auth_headers,
        json={"q": "otterworks", "owner_id": user.id, "page": 1, "size": 10},
    )
    assert advanced_response.status_code == 200, advanced_response.text

    delete_response = api_client.client.delete(
        f"/api/v1/search/index/document/{document_id}",
        headers=user.auth_headers,
    )
    assert delete_response.status_code in {200, 404}, delete_response.text
    api_client.indexed_documents.remove(document_id)


def test_search_validation_and_pagination_bounds(api_client):
    user = api_client.register_user("search-validation")

    missing_query = api_client.client.get("/api/v1/search/", headers=user.auth_headers)
    assert missing_query.status_code == 400

    invalid_page = api_client.client.get(
        "/api/v1/search/",
        headers=user.auth_headers,
        params={"q": "anything", "page": "not-a-number"},
    )
    assert invalid_page.status_code == 400

    short_suggest = api_client.client.get(
        "/api/v1/search/suggest",
        headers=user.auth_headers,
        params={"q": "a"},
    )
    assert short_suggest.status_code == 200, short_suggest.text
    assert short_suggest.json()["suggestions"] == []

    empty_index = api_client.client.post(
        "/api/v1/search/index/document",
        headers=user.auth_headers,
        json={},
    )
    assert empty_index.status_code == 400
