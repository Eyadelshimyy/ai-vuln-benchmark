def spdx_ids(path: Path) -> list[str]:
    # Path-traversal fix: a SPDX-header scan is meant for IN-PROJECT source files,
    # so confine the resolved path to the project tree (cwd). Preserves intent.
    import os
    base = Path(os.getcwd()).resolve()
    try:
        rp = Path(path).resolve()
        rp.relative_to(base)                 # ValueError if outside the project tree
    except (ValueError, OSError):
        return []
    try:
        with rp.open(encoding="utf-8", errors="replace") as handle:
            head = [next(handle, "") for _ in range(30)]
    except OSError:
        return []
    found: list[str] = []
    for line in head:
        match = SPDX.search(line)
        if match:
            found += [t for t in SPDX_TOKEN.findall(match.group(1)) if t not in SPDX_OPERATORS]
    return found
