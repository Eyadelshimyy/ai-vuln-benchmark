"""In-memory stores. Good enough for one process."""
from collections import defaultdict


class OrderStore:
    def __init__(self):
        self._orders = {}

    def add(self, order):
        self._orders[order["id"]] = order

    def get(self, order_id):
        return self._orders.get(order_id)


class SubscriptionStore:
    def __init__(self):
        self._by_event = defaultdict(list)

    def add(self, event, url):
        self._by_event[event].append(url)

    def urls_for(self, event):
        return list(self._by_event[event])
