"""Export and import projects for offline sharing / multi-user handoff.

Phase 1 scope: project record + contacts + notes (no attachment binaries).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .db import db_connection
from .project_links import get_links_for_project, link_project
from .projects import _ensure_docs_path, delete_project

_EXPORT_VERSION = 1


def export_project(project_id: int, output_path: str = "") -> dict:
    """Export a project with all contacts, notes, and linked projects as a JSON file.

    project_id: numeric project ID (use tool_get_project to find it)
    output_path: destination file path; defaults to
                 ~/.project-hub/exports/{slug}-{date}.json

    Linked projects (see tool_link_project) are exported by slug, not by DB ID —
    IDs are not stable across databases. Restoring a link on import requires the
    linked project to already exist in the target DB (see import_project).

    Returns {"path": str, "project": str, "contacts": int, "notes": int, "links": int}.
    Raises ValueError if project not found.
    """
    with db_connection() as conn:
        project_row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if not project_row:
            raise ValueError(f"Project {project_id} not found")

        project = dict(project_row)
        slug = project["slug"]

        contact_rows = conn.execute(
            "SELECT * FROM contacts WHERE project_id = ? ORDER BY id",
            (project_id,),
        ).fetchall()
        contacts = [dict(r) for r in contact_rows]

        note_rows = conn.execute(
            "SELECT * FROM notes WHERE project_id = ? ORDER BY id",
            (project_id,),
        ).fetchall()
        notes = [dict(r) for r in note_rows]

    links = [
        {
            "relation": link["relation"],
            "slug": link["project"]["slug"],
            "name": link["project"]["name"],
        }
        for link in get_links_for_project(project_id)
    ]

    payload = {
        "export_version": _EXPORT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "contacts": contacts,
        "notes": notes,
        "links": links,
    }

    if not output_path:
        date_str = datetime.now().strftime("%Y%m%d")
        export_dir = Path.home() / ".project-hub" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        dest = export_dir / f"{slug}-{date_str}.json"
    else:
        dest = Path(output_path).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)

    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[project-hub] Exported project '{project['name']}' → {dest}",
        file=sys.stderr,
    )

    return {
        "path": str(dest),
        "project": project["name"],
        "contacts": len(contacts),
        "notes": len(notes),
        "links": len(links),
    }


def import_project(json_path: str, merge_strategy: str = "skip") -> dict:
    """Import a project from a JSON export file.

    json_path: path to the .json file produced by export_project
    merge_strategy: what to do when a project with the same slug already exists
        - "skip"      — abort import, return {"imported": False, "reason": "exists"}
        - "rename"    — append _imported_{timestamp} suffix to slug/name and insert
        - "overwrite" — delete existing project (cascade) and re-insert

    Linked projects (see export_project) are restored by slug: if the linked
    project already exists in this DB, the link is recreated; otherwise it is
    skipped and listed under "links_not_restored" in the result.

    Returns summary dict.
    Raises ValueError on invalid file format or unsupported version.
    """
    if merge_strategy not in ("skip", "rename", "overwrite"):
        raise ValueError(f"merge_strategy must be skip|rename|overwrite, got '{merge_strategy}'")

    src = Path(json_path).expanduser()
    if not src.exists():
        raise ValueError(f"File not found: {src}")

    raw = json.loads(src.read_text(encoding="utf-8"))

    if raw.get("export_version") != _EXPORT_VERSION:
        raise ValueError(
            f"Unsupported export_version {raw.get('export_version')!r} (expected {_EXPORT_VERSION})"
        )

    project = raw["project"]
    contacts: list[dict] = raw.get("contacts", [])
    notes: list[dict] = raw.get("notes", [])

    # Phase 1: check for conflicts and handle merge strategy before inserting.
    # Overwrite calls delete_project() (which handles session-FK nullout, shared-contact
    # re-parenting, and docs-folder removal) rather than a raw DELETE that would bypass
    # those safeguards (#101).
    existing_found: bool = False
    with db_connection() as conn:
        existing_found = bool(
            conn.execute(
                "SELECT id FROM projects WHERE slug = ?", (project["slug"],)
            ).fetchone()
        )

    if existing_found:
        if merge_strategy == "skip":
            return {
                "imported": False,
                "reason": f"project with slug '{project['slug']}' already exists",
                "project": project["name"],
                "contacts": 0,
                "notes": 0,
                "links": 0,
                "links_not_restored": [],
            }
        elif merge_strategy == "overwrite":
            delete_project(project["slug"])
        elif merge_strategy == "rename":
            suffix = datetime.now().strftime("%Y%m%d%H%M%S")
            project["slug"] = f"{project['slug']}_imported_{suffix}"
            project["name"] = f"{project['name']} (imported {suffix})"

    # Strip the old primary key and docs_path — docs_path is created after insert (#96).
    project.pop("id", None)
    project.pop("docs_path", None)

    # Phase 2: insert the new project record and its contacts / notes.
    with db_connection() as conn:
        cols = ", ".join(project.keys())
        placeholders = ", ".join("?" for _ in project)
        conn.execute(
            f"INSERT INTO projects ({cols}) VALUES ({placeholders})",
            list(project.values()),
        )
        conn.commit()

        new_project_id: int = conn.execute(
            "SELECT id FROM projects WHERE slug = ?", (project["slug"],)
        ).fetchone()[0]

        # Create the docs folder and persist the real path (#96).
        real_docs_path = _ensure_docs_path(project["slug"])
        conn.execute(
            "UPDATE projects SET docs_path = ? WHERE id = ?",
            (real_docs_path, new_project_id),
        )
        conn.commit()

        imported_contacts = 0
        for c in contacts:
            c.pop("id", None)
            c["project_id"] = new_project_id
            cols_c = ", ".join(c.keys())
            ph_c = ", ".join("?" for _ in c)
            conn.execute(f"INSERT INTO contacts ({cols_c}) VALUES ({ph_c})", list(c.values()))
            imported_contacts += 1

        imported_notes = 0
        for n in notes:
            n.pop("id", None)
            n.pop("file_path", None)  # file_path from source DB is meaningless on this machine
            n["project_id"] = new_project_id
            cols_n = ", ".join(n.keys())
            ph_n = ", ".join("?" for _ in n)
            conn.execute(f"INSERT INTO notes ({cols_n}) VALUES ({ph_n})", list(n.values()))
            imported_notes += 1

        conn.commit()

    restored_links = 0
    links_not_restored: list[dict] = []
    for link in raw.get("links", []):
        try:
            link_project(project["slug"], link["slug"], link["relation"])
            restored_links += 1
        except (ValueError, KeyError) as e:
            links_not_restored.append(
                {
                    "slug": link.get("slug", "?"),
                    "name": link.get("name", link.get("slug", "?")),
                    "relation": link.get("relation", "?"),
                    "reason": str(e),
                }
            )

    print(
        f"[project-hub] Imported project '{project['name']}' "
        f"({imported_contacts} contacts, {imported_notes} notes, {restored_links} links)",
        file=sys.stderr,
    )

    return {
        "imported": True,
        "project": project["name"],
        "slug": project["slug"],
        "project_id": new_project_id,
        "contacts": imported_contacts,
        "notes": imported_notes,
        "links": restored_links,
        "links_not_restored": links_not_restored,
        "merge_strategy": merge_strategy,
    }
