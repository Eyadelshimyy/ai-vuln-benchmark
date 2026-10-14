"""HTTP entry point: create an order, then tell every subscriber about it."""
import uuid

from flask import Flask, jsonify, request

from .store import OrderStore, SubscriptionStore
from .webhooks import send_webhook


def create_app(orders=None, subscriptions=None):
    app = Flask(__name__)
    orders = orders or OrderStore()
    subscriptions = subscriptions or SubscriptionStore()

    @app.post("/orders")
    def create_order():
        order = {"id": str(uuid.uuid4()), **request.get_json()}
        orders.add(order)
        for url in subscriptions.urls_for("order.created"):
            send_webhook(url, {"event": "order.created", "order": order})
        return jsonify(order), 201

    @app.post("/subscriptions")
    def subscribe():
        body = request.get_json()
        subscriptions.add(body["event"], body["url"])
        return "", 204

    return app
