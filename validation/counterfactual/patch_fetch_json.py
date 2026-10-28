def fetch_json(url: str) -> dict | None:
    # SSRF fix (reasoned from intent: fetch club logos from EXTERNAL sources).
    # Allow only http(s) to public hosts; block private/loopback/link-local/reserved.
    import ipaddress, socket
    from urllib.parse import urlparse
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            return None
        for res in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(res[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return None
    except Exception:
        return None
    req = urllib.request.Request(url, headers={"User-Agent": "sada-scout/1.0 (club logos)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None
