from unittest.mock import patch

from order_hooks import create_app


def test_create_order_notifies_subscribers():
    app = create_app()
    client = app.test_client()
    client.post("/subscriptions", json={"event": "order.created", "url": "http://merchant.test/hook"})

    with patch("order_hooks.app.send_webhook") as send:
        res = client.post("/orders", json={"sku": "ABC-1", "qty": 2})

    assert res.status_code == 201
    send.assert_called_once()
    url, payload = send.call_args.args
    assert url == "http://merchant.test/hook"
    assert payload["order"]["sku"] == "ABC-1"
