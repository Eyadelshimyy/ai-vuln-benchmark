def submit(name: str, header: str, cmd: str, what: str, dry: bool) -> None:
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("name must be a single path component")
    body = (f"#!/usr/bin/env bash\n# fill: owner={OWNER} {header}\n# question: \"{QUESTION}\"\n# proofs-n2-guest (bc-c951b059): {what}\n"
            f"exec env PN2G_QUESTION='{QUESTION}' {cmd}\n")
    if dry:
        print("would submit", name, header, cmd)
        return
    p = G / "scripts" / name
    p.write_text(body)
    p.chmod(0o755)
    os.replace(p, FILL / "queue" / name)                  # the same filesystem: an atomic rename, never a half-written script
