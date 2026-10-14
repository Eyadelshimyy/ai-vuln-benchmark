"""Local-lab isolation helpers."""

from __future__ import annotations

import ipaddress

_LAB_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0:0:0:0:0:0:0:1"})


def is_loopback_or_lab_host(hostname: str, extra: tuple[str, ...] = ()) -> bool:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return False
    if host in _LAB_HOSTS or host.endswith(".lab") or host.endswith(".localhost"):
        return True
    extra_norm = {item.strip().lower().rstrip(".") for item in extra if item.strip()}
    if host in extra_norm:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(addr.is_loopback)
