import os
import time

import pytest
import socketio


pytestmark = [pytest.mark.api_flow, pytest.mark.websocket]


def _collab_url(base_url: str) -> str:
    return os.getenv("OTTERWORKS_COLLAB_WS_URL", base_url.replace("http://", "ws://").replace("https://", "wss://"))


def test_socketio_rejects_missing_or_invalid_token(base_url):
    sio = socketio.Client(reconnection=False, request_timeout=3)
    with pytest.raises(socketio.exceptions.ConnectionError):
        sio.connect(_collab_url(base_url), transports=["websocket"])

    invalid = socketio.Client(reconnection=False, request_timeout=3)
    with pytest.raises(socketio.exceptions.ConnectionError):
        invalid.connect(
            _collab_url(base_url),
            auth={"token": "not-a-valid-token"},
            transports=["websocket"],
        )


def test_socketio_two_users_join_same_document_and_presence_updates(api_client, base_url):
    user_a = api_client.register_user("ws-user-a")
    user_b = api_client.register_user("ws-user-b")
    document = api_client.create_document(
        user_a,
        title=f"WebSocket Document {api_client.run_id}",
        content="collaboration body",
    )
    document_id = document["id"]
    received_by_b: list[dict] = []

    client_a = socketio.Client(reconnection=False, request_timeout=5)
    client_b = socketio.Client(reconnection=False, request_timeout=5)

    @client_b.on("document-update")
    def on_document_update(data):
        received_by_b.append(data)

    try:
        client_a.connect(_collab_url(base_url), auth={"token": user_a.access_token}, transports=["websocket"])
        client_b.connect(_collab_url(base_url), auth={"token": user_b.access_token}, transports=["websocket"])

        client_a.emit("join-document", {"documentId": document_id})
        client_b.emit("join-document", {"documentId": document_id})
        time.sleep(0.5)

        presence_response = api_client.client.get(
            f"/api/v1/collab/documents/{document_id}/presence",
            headers=user_a.auth_headers,
        )
        assert presence_response.status_code == 200, presence_response.text

        client_a.emit(
            "document-update",
            {"documentId": document_id, "update": {"text": f"hello {api_client.run_id}"}},
        )

        api_client.poll_until(
            lambda: received_by_b,
            lambda updates: len(updates) >= 1,
            timeout_seconds=10,
            interval_seconds=0.25,
            description="collaboration update fanout to second socket client",
        )
    finally:
        if client_a.connected:
            client_a.disconnect()
        if client_b.connected:
            client_b.disconnect()
