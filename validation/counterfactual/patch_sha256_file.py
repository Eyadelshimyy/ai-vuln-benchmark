def sha256_file(path: Path) -> str:
    # Path-traversal fix: confine the resolved path to the project tree (cwd).
    import os
    base = Path(os.getcwd()).resolve()
    try:
        rp = Path(path).resolve()
        rp.relative_to(base)
    except (ValueError, OSError):
        raise PermissionError("path outside allowed base")
    h = hashlib.sha256()
    with rp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
