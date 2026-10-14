"""Project-to-project relations: successor / predecessor / related."""
from __future__ import annotations

import sqlite3
from typing import Optional

from .db import db_connection

_VALID_RELATION_TYPES = ("successor", "predecessor", "related")


def _resolve_project(identifier: str) -> Optional[dict]:
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE slug = ? OR LOWER(name) = LOWER(?)",
            (identifier, identifier),
        ).fetchone()
        return dict(row) if row else None


def _project_stub(project: dict) -> dict:
    return {"id": project["id"], "slug": project["slug"], "name": project["name"]}


def get_links_for_project(project_id: int) -> list[dict]:
    """Return this project's links from its own perspective.

    Returns [{"relation": "successor" | "predecessor" | "related", "project": {id, slug, name}}, ...].
    The DB only ever stores 'successor'/'related' rows; 'predecessor' is inferred here as the
    inverse of a 'successor' row seen from the other side.
    """
    with db_connection() as conn:
        rows = conn.execute(
            """SELECT pl.relation_type, pl.project_id,
                      CASE WHEN pl.project_id = ? THEN pl.related_project_id ELSE pl.project_id END AS other_id
               FROM project_links pl
               WHERE pl.project_id = ? OR pl.related_project_id = ?""",
            (project_id, project_id, project_id),
        ).fetchall()

        links: list[dict] = []
        for row in rows:
            other_row = conn.execute(
                "SELECT id, slug, name FROM projects WHERE id = ?", (row["other_id"],)
            ).fetchone()
            if row["relation_type"] == "related":
                relation = "related"
            elif row["project_id"] == project_id:
                relation = "successor"
            else:
                relation = "predecessor"
            links.append({"relation": relation, "project": dict(other_row)})

    links.sort(key=lambda link: link["project"]["name"])
    return links


def _find_link(project_id: int, related_id: int) -> Optional[dict]:
    for link in get_links_for_project(project_id):
        if link["project"]["id"] == related_id:
            return link
    return None


def link_project(identifier: str, related_identifier: str, relation_type: str) -> dict:
    """Link two projects.

    relation_type: successor | predecessor | related
    "successor"/"predecessor" are directional as seen from `identifier`; "related" is symmetric.
    Raises ValueError for an unknown project, a self-link, an invalid relation_type, or a link
    that already exists between the two projects (in either direction) — unlink first to change it.
    """
    if relation_type not in _VALID_RELATION_TYPES:
        raise ValueError(
            f"relation_type must be one of: {', '.join(_VALID_RELATION_TYPES)} (got '{relation_type}')"
        )

    project = _resolve_project(identifier)
    if not project:
        raise ValueError(f"Project not found: '{identifier}'")
    related = _resolve_project(related_identifier)
    if not related:
        raise ValueError(f"Project not found: '{related_identifier}'")
    if project["id"] == related["id"]:
        raise ValueError("Cannot link a project to itself.")

    existing = _find_link(project["id"], related["id"])
    if existing:
        raise ValueError(
            f"'{project['name']}' is already linked to '{related['name']}' "
            f"(as {existing['relation']}). Use tool_unlink_project first to change it."
        )

    # Canonicalize storage: the DB only ever stores 'successor' or 'related' rows;
    # 'predecessor' is the inverse of 'successor' and gets flipped here so the read
    # side (get_links_for_project) never has to special-case a third stored value.
    if relation_type == "predecessor":
        store_project_id, store_related_id, store_type = related["id"], project["id"], "successor"
    elif relation_type == "successor":
        store_project_id, store_related_id, store_type = project["id"], related["id"], "successor"
    else:
        store_project_id, store_related_id = sorted((project["id"], related["id"]))
        store_type = "related"

    with db_connection() as conn:
        try:
            with conn:
                conn.execute(
                    "INSERT INTO project_links (project_id, related_project_id, relation_type) VALUES (?, ?, ?)",
                    (store_project_id, store_related_id, store_type),
                )
        except sqlite3.IntegrityError:
            # Duplicate insert raced past the _find_link check above (single-user tool, narrow window).
            raise ValueError(
                f"'{project['name']}' is already linked to '{related['name']}'. "
                f"Use tool_unlink_project first to change it."
            ) from None

    return {
        "relation_type": relation_type,
        "project": _project_stub(project),
        "related_project": _project_stub(related),
    }


def unlink_project(identifier: str, related_identifier: str) -> bool:
    """Remove the link between two projects, regardless of direction or stored relation type."""
    project = _resolve_project(identifier)
    related = _resolve_project(related_identifier)
    if not project or not related:
        return False

    with db_connection() as conn:
        with conn:
            result = conn.execute(
                """DELETE FROM project_links
                   WHERE (project_id = ? AND related_project_id = ?)
                      OR (project_id = ? AND related_project_id = ?)""",
                (project["id"], related["id"], related["id"], project["id"]),
            )
        return result.rowcount > 0
