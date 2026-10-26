#!/usr/bin/env python3
"""Populate the OtterWorks enterprise drive via the OtterWorks API gateway.

Creates a deep department/subfolder tree, uploads realistic multimodal files
(xlsx/docx/pptx/pdf/csv/txt/md/json/png/jpg/svg + committed mp4 clips), and
creates rich-text documents —
all owned by one shared "drive" account so the result is a single browsable
enterprise drive in the web UI.

The work is shardable by department (``--departments``) so it can be fanned out
across many parallel sessions/workers, all writing under the same owner without
collision (each department is an independent top-level subtree).

Example (one department):
    python generate_drive.py \
        --gateway http://<gw-host>:8080 \
        --email "$DRIVE_EMAIL" --password "$DRIVE_PASSWORD" \
        --departments Finance --scale 1.0

Example (all departments):
    python generate_drive.py --gateway ... --email ... --password ... \
        --departments all
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import re
import sys
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import filegen
import taxonomy

# Per-axis breadth at --scale 1.0 (bounded so volume stays reasonable).
AXIS_BASE = {
    "year": 4, "quarter": 4, "month": 6, "region": 6,
    "store": 8, "vendor": 8, "campaign": 8, "category": 8, "sku": 10,
}


def selected(axis: str, scale: float) -> list:
    values = taxonomy.AXES[axis]
    n = max(1, min(len(values), round(AXIS_BASE[axis] * scale)))
    return values[:n]


class Client:
    def __init__(self, gateway: str, email: str, password: str):
        self.gw = gateway.rstrip("/")
        self.s = requests.Session()
        self.email = email
        self.password = password
        self.token = None
        self.owner_id = None

    def register(self, display_name="OtterWorks Enterprise Drive"):
        """Create the shared drive account (no-op if it already exists)."""
        self.s.post(f"{self.gw}/api/v1/auth/register", json={
            "displayName": display_name, "email": self.email,
            "password": self.password, "confirmPassword": self.password,
        }, timeout=30)

    def login(self, register_if_missing=False):
        r = self.s.post(f"{self.gw}/api/v1/auth/login",
                        json={"email": self.email, "password": self.password}, timeout=30)
        if r.status_code >= 400 and register_if_missing:
            self.register()
            r = self.s.post(f"{self.gw}/api/v1/auth/login",
                            json={"email": self.email, "password": self.password}, timeout=30)
        r.raise_for_status()
        self.token = r.json()["accessToken"]
        # owner_id = JWT sub
        p = self.token.split(".")[1]
        p += "=" * (-len(p) % 4)
        self.owner_id = json.loads(base64.urlsafe_b64decode(p))["sub"]
        return self

    @property
    def h(self):
        return {"Authorization": f"Bearer {self.token}"}

    def _retry(self, fn, tries=4):
        last = None
        for i in range(tries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(0.5 * (2 ** i))
        raise last

    def list_child_folders(self, parent_id):
        params = {"owner_id": self.owner_id}
        if parent_id:
            params["parent_id"] = parent_id
        r = self.s.get(f"{self.gw}/api/v1/folders", headers=self.h, params=params, timeout=30)
        r.raise_for_status()
        return {f["name"]: f["id"] for f in r.json().get("folders", [])}

    def create_folder(self, name, parent_id):
        body = {"name": name, "owner_id": self.owner_id}
        if parent_id:
            body["parent_id"] = parent_id
        r = self.s.post(f"{self.gw}/api/v1/folders", headers=self.h, json=body, timeout=30)
        r.raise_for_status()
        return r.json()["id"]

    def upload(self, folder_id, filename, data: bytes, mime: str):
        files = {"file": (filename, data, mime)}
        form = {"owner_id": self.owner_id}
        if folder_id:
            form["folder_id"] = folder_id
        r = self.s.post(f"{self.gw}/api/v1/files/upload", headers=self.h,
                        data=form, files=files, timeout=120)
        r.raise_for_status()
        return r.json().get("id")

    def list_file_names(self, folder_id):
        # file-service caps page_size at 100 (default 50); paginate to get every
        # name so the idempotency check sees folders with >100 files in full.
        names: set = set()
        page = 1
        while True:
            params = {"owner_id": self.owner_id, "page": page, "page_size": 100}
            if folder_id:
                params["folder_id"] = folder_id
            r = self.s.get(f"{self.gw}/api/v1/files", headers=self.h, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            batch = data.get("files", [])
            names.update(f["name"] for f in batch)
            total = data.get("total", len(names))
            if not batch or len(names) >= total:
                break
            page += 1
        return names

    def list_document_titles(self):
        # document-service caps size at 100 (default 20); paginate all pages.
        titles: set = set()
        page = 1
        while True:
            params = {"owner_id": self.owner_id, "page": page, "size": 100}
            r = self.s.get(f"{self.gw}/api/v1/documents", headers=self.h,
                           params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            items = data.get("items", data.get("documents", data if isinstance(data, list) else []))
            titles.update(d.get("title") for d in items)
            total = data.get("total", len(titles))
            if not items or len(titles) >= total:
                break
            page += 1
        return titles

    def create_document(self, title, content, content_type="text/markdown"):
        body = {"title": title, "content": content, "content_type": content_type,
                "owner_id": self.owner_id}
        r = self.s.post(f"{self.gw}/api/v1/documents", headers=self.h, json=body, timeout=60)
        r.raise_for_status()
        return r.json().get("id")


class FolderCache:
    """Idempotent, thread-safe folder resolver (reuses existing folders)."""

    def __init__(self, client: Client):
        self.c = client
        self._ids = {}  # ("path/parts",) -> id
        self._lock = threading.Lock()

    def resolve(self, parts: tuple[str, ...]) -> str:
        with self._lock:
            key = ()
            parent = None
            for part in parts:
                key = key + (part,)
                if key in self._ids:
                    parent = self._ids[key]
                    continue
                existing = self.c.list_child_folders(parent)
                if part in existing:
                    fid = existing[part]
                else:
                    fid = self.c._retry(lambda: self.c.create_folder(part, parent))
                self._ids[key] = fid
                parent = fid
            return parent


def _axes_for(spec) -> list[str]:
    """Axes = every {placeholder} appearing in the folder path or file name.

    (The spec's ``expand`` list is advisory; deriving from placeholders keeps
    names and folders consistent and avoids KeyError on partial declarations.)
    """
    found = re.findall(r"{(\w+)}", spec["folder"] + " " + spec["name"])
    seen = []
    for a in found:
        if a in taxonomy.AXES and a not in seen:
            seen.append(a)
    return seen


def expand_specs(dept: str, scale: float):
    """Yield (folder_parts, filename, ext) for every file in a department."""
    for spec in taxonomy.DEPARTMENTS[dept]:
        axes = _axes_for(spec)
        value_lists = [selected(a, scale) for a in axes]
        combos = itertools.product(*value_lists) if axes else [()]
        for combo in combos:
            subs = dict(zip(axes, combo))
            folder = spec["folder"].format(**subs)
            name = spec["name"].format(**subs)
            parts = tuple([dept] + [p for p in folder.split("/") if p])
            yield parts, f"{name}.{spec['type']}", spec["type"], spec.get("kind")


def run_department(client: Client, cache: FolderCache, dept: str, scale: float,
                   seed: int, workers: int, dry_run: bool, make_docs: bool):
    files = list(expand_specs(dept, scale))
    if dry_run:
        return len(files), 0

    # Pre-create the full folder set (serialized via cache) to avoid races,
    # and prefetch existing file names per folder for idempotency.
    existing = {}
    for parts, _, _, _ in files:
        fid = cache.resolve(parts)
        if fid not in existing:
            existing[fid] = client._retry(lambda f=fid: client.list_file_names(f))
    lock = threading.Lock()

    def do_upload(item):
        parts, filename, ext, kind = item
        folder_id = cache.resolve(parts)
        with lock:  # skip if already present (re-run / overlapping shard safe)
            if filename in existing[folder_id]:
                return 0
            existing[folder_id].add(filename)
        data, mime = filegen.build(ext, filename.rsplit(".", 1)[0], seed, kind=kind)
        client._retry(lambda: client.upload(folder_id, filename, data, mime))
        return 1

    uploaded = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(do_upload, it) for it in files]
        for f in as_completed(futs):
            uploaded += f.result()

    docs = 0
    if make_docs:
        existing_docs = client._retry(client.list_document_titles)
        for title in taxonomy.DEPARTMENT_DOCS.get(dept, []):
            full = f"{dept} — {title}"
            if full in existing_docs:
                continue
            body = filegen.build("md", title, seed)[0].decode()
            client._retry(lambda t=full, b=body: client.create_document(t, b))
            docs += 1
    return uploaded, docs


ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def upload_assets(client: Client, cache: FolderCache, depts: list[str]) -> int:
    """Upload the committed media clips (idempotent, like generated files).

    Only placements whose top-level folder is in ``depts`` are uploaded, so
    per-department shard runs never write outside their own subtree.
    """
    uploaded = 0
    for filename, folder_parts in taxonomy.ASSET_PLACEMENTS:
        if folder_parts[0] not in depts:
            continue
        path = ASSETS_DIR / filename
        if not path.is_file():
            print(f"[drive] WARN missing committed asset: {path}")
            continue
        folder_id = cache.resolve(folder_parts)
        if filename in client._retry(lambda f=folder_id: client.list_file_names(f)):
            continue
        ext = filename.rsplit(".", 1)[-1].lower()
        mime = filegen.MIME.get(ext, "application/octet-stream")
        client._retry(lambda: client.upload(folder_id, filename, path.read_bytes(), mime))
        uploaded += 1
    return uploaded


def main(argv=None):
    ap = argparse.ArgumentParser(description="Populate the OtterWorks enterprise drive.")
    ap.add_argument("--gateway", required=True)
    ap.add_argument("--email", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--departments", default="all",
                    help="comma-separated department names, or 'all'")
    ap.add_argument("--scale", type=float, default=1.0, help="axis breadth multiplier")
    ap.add_argument("--seed", type=int, default=20240117)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-docs", action="store_true")
    ap.add_argument("--register", action="store_true",
                    help="create the drive account if it does not exist (seed-loader use)")
    ap.add_argument("--dry-run", action="store_true", help="print file counts, upload nothing")
    args = ap.parse_args(argv)

    if args.departments.strip().lower() == "all":
        depts = taxonomy.all_departments()
    else:
        depts = [d.strip() for d in args.departments.split(",") if d.strip()]
    unknown = [d for d in depts if d not in taxonomy.DEPARTMENTS]
    if unknown:
        ap.error(f"unknown departments: {unknown}\nvalid: {taxonomy.all_departments()}")

    if args.dry_run:
        total = 0
        for d in depts:
            n = len(list(expand_specs(d, args.scale)))
            total += n
            print(f"  {d:28s} {n:5d} files")
        print(f"  {'TOTAL':28s} {total:5d} files across {len(depts)} departments")
        return 0

    client = Client(args.gateway, args.email, args.password)
    client.login(register_if_missing=args.register)
    cache = FolderCache(client)
    print(f"[drive] owner_id={client.owner_id} departments={depts} scale={args.scale}")
    grand_files = grand_docs = 0
    for d in depts:
        t0 = time.time()
        nf, nd = run_department(client, cache, d, args.scale, args.seed,
                                args.workers, False, not args.no_docs)
        grand_files += nf
        grand_docs += nd
        print(f"[drive] {d:28s} files={nf:4d} docs={nd:2d} ({time.time()-t0:5.1f}s)")
    na = upload_assets(client, cache, depts)
    grand_files += na
    print(f"[drive] {'committed assets':28s} files={na:4d}")
    print(f"[drive] DONE files={grand_files} docs={grand_docs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
