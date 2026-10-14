"""Best-effort MQTT broker connection test for the Settings UI — used to verify
broker/port/credentials/TLS before saving. Blocking; call it in a thread executor."""
import socket
import time

import paho.mqtt.client as mqtt

# Standard MQTT CONNACK return codes (v3.1.1) → human-readable reasons.
_CONNACK = {
    1: "Unacceptable protocol version",
    2: "Identifier rejected",
    3: "Broker unavailable",
    4: "Bad username or password",
    5: "Not authorized",
}


def check_connection(broker, port, username=None, password=None,
                     use_tls=False, tls_insecure=False, timeout=6):
    """Return (ok: bool, reason: str). Does a bounded TCP check first (so a wrong
    host/port fails fast instead of hanging), then a real MQTT CONNECT to surface
    auth/TLS/protocol errors via the CONNACK code."""
    broker = (broker or "").strip()
    if not broker:
        return False, "No broker host set"
    try:
        port = int(port or (8883 if use_tls else 1883))
    except (TypeError, ValueError):
        return False, "Invalid port"

    # 1) Bounded TCP reachability — turns the slow/hang cases into fast errors.
    try:
        socket.create_connection((broker, port), timeout=timeout).close()
    except socket.gaierror:
        return False, "Host not found"
    except (ConnectionRefusedError, TimeoutError, socket.timeout):
        return False, f"No response from {broker}:{port}"
    except OSError as exc:
        return False, f"Unreachable: {exc}"

    # 2) Real MQTT handshake — checks TLS + credentials via the CONNACK code.
    result = {"rc": None, "done": False}
    client = None
    try:
        try:
            from paho.mqtt.enums import CallbackAPIVersion
            client = mqtt.Client(CallbackAPIVersion.VERSION1)
        except ImportError:
            client = mqtt.Client()
        if username:
            client.username_pw_set(username, password or None)
        if use_tls:
            client.tls_set()
            if tls_insecure:
                client.tls_insecure_set(True)

        def _on_connect(c, u, flags, rc):
            result["rc"] = rc
            result["done"] = True

        client.on_connect = _on_connect
        client.connect(broker, port, keepalive=10)
        client.loop_start()
        deadline = time.time() + timeout
        while not result["done"] and time.time() < deadline:
            time.sleep(0.1)
    except Exception as exc:  # noqa: BLE001 — surface any TLS/socket error to the UI
        return False, str(exc)
    finally:
        if client is not None:
            try:
                client.loop_stop()
                client.disconnect()
            except Exception:  # noqa: BLE001
                pass

    if not result["done"]:
        return False, "Timed out waiting for the broker"
    rc = result["rc"]
    if rc == 0:
        return True, "Connected"
    return False, _CONNACK.get(rc, f"Refused (code {rc})")
