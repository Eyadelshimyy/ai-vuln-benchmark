"""Outbound webhook delivery."""
import requests


def send_webhook(url, payload):
    # Fire and forget: no timeout, no retry. A failure is logged by nobody.
    try:
        requests.post(url, json=payload)
    except requests.RequestException:
        pass
