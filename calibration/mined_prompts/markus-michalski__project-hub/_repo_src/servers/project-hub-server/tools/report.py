"""Static HTML report generation for project-hub (issue #26).

Renders a self-contained .html file (Jinja2 + Chart.js CDN) that can be
opened in any browser without a running server.
"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .db import db_connection
from .project_links import get_links_for_project

_REPORT_TYPES = {"full", "summary", "all-projects"}

_RELATION_LABELS = {
    "successor": "Nachfolger von",
    "predecessor": "Vorgänger von",
    "related": "Verknüpft mit",
}

_STATUS_LABELS = {
    "active": "Aktiv",
    "paused": "Pausiert",
    "completed": "Abgeschlossen",
    "cancelled": "Abgebrochen",
}

_NOTE_TYPE_LABELS = {
    "note": "Notiz",
    "meeting-notes": "Meeting",
    "email": "E-Mail",
    "decision": "Entscheidung",
    "action-item": "Aufgabe",
}

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(enabled_extensions=("j2", "html")),
    )


def _default_path(slug: str, report_type: str) -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    reports_dir = Path.home() / ".project-hub" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if report_type == "all-projects":
        filename = f"all-projects-{date_str}.html"
    else:
        filename = f"{slug}-{report_type}-{date_str}.html"
    return reports_dir / filename


def generate_report(
    project_id: int | None,
    report_type: str = "full",
    output_path: str = "",
    offline: bool = False,
) -> dict:
    """Render a static HTML report for one project or all projects.

    project_id: numeric project ID (None for all-projects report)
    report_type: "full" | "summary" | "all-projects"
    output_path: destination file; defaults to ~/.project-hub/reports/{slug}-{report_type}-{date}.html
    offline: reserved for future CDN-inlining support

    Returns {"path": str, "project": str}.
    Raises ValueError if project not found or report_type invalid.
    """
    if report_type not in _REPORT_TYPES:
        raise ValueError(f"report_type must be one of {sorted(_REPORT_TYPES)}, got '{report_type}'")

    env = _jinja_env()
    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")

    if report_type == "all-projects":
        return _render_all_projects(env, generated_at, output_path)

    # Single-project reports
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            raise ValueError(f"Project {project_id} not found")
        project = dict(row)

        contact_rows = conn.execute(
            "SELECT * FROM contacts WHERE project_id = ? ORDER BY type, name",
            (project_id,),
        ).fetchall()
        contacts = [dict(r) for r in contact_rows]

        note_rows = conn.execute(
            "SELECT * FROM notes WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
        notes = [dict(r) for r in note_rows]

    note_type_counts = dict(Counter(n["type"] for n in notes))
    links = get_links_for_project(project["id"])

    dest = Path(output_path).expanduser() if output_path else _default_path(project["slug"], report_type)
    dest.parent.mkdir(parents=True, exist_ok=True)

    template = env.get_template("report_full.html.j2")
    html = template.render(
        report_type=report_type,
        project=project,
        contacts=contacts,
        notes=notes,
        note_type_counts=note_type_counts,
        links=links,
        relation_labels=_RELATION_LABELS,
        status_labels=_STATUS_LABELS,
        note_type_labels=_NOTE_TYPE_LABELS,
        generated_at=generated_at,
    )

    dest.write_text(html, encoding="utf-8")

    print(f"[project-hub] Report '{report_type}' for '{project['name']}' → {dest}", file=sys.stderr)

    return {"path": str(dest), "project": project["name"]}


def _render_all_projects(env: Environment, generated_at: str, output_path: str) -> dict:
    with db_connection() as conn:
        project_rows = conn.execute(
            "SELECT * FROM projects ORDER BY name",
        ).fetchall()
        projects = [dict(r) for r in project_rows]

        for p in projects:
            p["contact_count"] = conn.execute(
                "SELECT COUNT(*) FROM contacts WHERE project_id = ?", (p["id"],)
            ).fetchone()[0]
            p["note_count"] = conn.execute(
                "SELECT COUNT(*) FROM notes WHERE project_id = ?", (p["id"],)
            ).fetchone()[0]

    for p in projects:
        p["links"] = get_links_for_project(p["id"])

    status_counts = dict(Counter(p["status"] for p in projects))
    type_counts = dict(Counter(p["type"] for p in projects))

    dest = Path(output_path).expanduser() if output_path else _default_path("all-projects", "all-projects")
    dest.parent.mkdir(parents=True, exist_ok=True)

    template = env.get_template("report_full.html.j2")
    html = template.render(
        report_type="all-projects",
        projects=projects,
        status_counts=status_counts,
        type_counts=type_counts,
        relation_labels=_RELATION_LABELS,
        status_labels=_STATUS_LABELS,
        note_type_labels=_NOTE_TYPE_LABELS,
        generated_at=generated_at,
    )

    dest.write_text(html, encoding="utf-8")

    print(f"[project-hub] All-projects report → {dest}", file=sys.stderr)

    return {"path": str(dest), "project": "all-projects"}
