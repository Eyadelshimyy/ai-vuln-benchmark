"""Project Hub MCP Server.

Provides persistent project management: projects, contacts, notes, and session tracking.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from tools.attachments import attach_file, attach_files, list_attachments, remove_attachment
from tools.contacts import add_contact, delete_contact, list_contacts, list_shared_contacts, update_contact
from tools.db import init_db
from tools.knowledge import (
    delete_knowledge,
    get_all_knowledge,
    get_knowledge,
    list_knowledge,
    save_knowledge,
    sync_knowledge_templates,
)
from tools.notes import add_note, delete_note, get_note, list_notes, update_note
from tools.project_links import link_project, unlink_project
from tools.project_types import (
    create_project_from_template,
    create_project_type,
    delete_project_type,
    get_project_template,
    get_project_type,
    list_project_types,
)
from tools.projects import (
    create_project,
    delete_project,
    get_project,
    get_project_by_id,
    list_docs,
    list_projects,
    update_project,
)
from tools.report import generate_report
from tools.search import search_contacts, search_notes
from tools.session import clear_session, get_session, set_session
from tools.transfer import export_project, import_project

# Initialize DB on startup
init_db()

mcp = MCPServer("project-hub-mcp")


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_get_session() -> dict:
    """Get current session — which project is active, last used skill."""
    return get_session()


@mcp.tool()
def tool_set_session(identifier: str, last_skill: str = "") -> dict:
    """Set the active project for this session by slug or name."""
    return set_session(identifier, last_skill)


@mcp.tool()
def tool_clear_session() -> dict:
    """Clear the active project (no project selected)."""
    return clear_session()


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_list_projects(status: str = "", limit: int = 50, offset: int = 0) -> dict:
    """List projects with pagination.

    status: active | paused | completed | cancelled (empty = all)
    Returns {"items": [...], "total": N, "limit": L, "offset": O}.
    """
    return list_projects(status, limit, offset)


@mcp.tool()
def tool_get_project(identifier: str) -> dict | None:
    """Get a project by slug or name.

    Includes a "links" list of related projects, see tool_link_project.
    """
    return get_project(identifier)


@mcp.tool()
def tool_get_project_by_id(project_id: int) -> dict | None:
    """Get a project by its numeric ID.

    Includes a "links" list of related projects, see tool_link_project.
    """
    return get_project_by_id(project_id)


@mcp.tool()
def tool_create_project(
    name: str,
    project_type: str = "generic",
    description: str = "",
    market: str = "",
    products: str = "",
    phase: str = "",
    go_live: str = "",
    budget: str = "",
    notes: str = "",
) -> dict:
    """Create a new project and its docs folder structure.

    project_type: any built-in or custom type name (use tool_list_project_types to discover available types)
    market: relevant market/country (e.g. DE, NL, SE — mainly for merchant-onboarding)
    products: integrated products (e.g. BNPL 30d, Pay in 3 — mainly for merchant-onboarding)
    phase: current project phase
    go_live: target go-live date
    budget: budget information
    Returns the created project on success, or {"error": ...} if a project with this name already exists.
    """
    return create_project(name, project_type, description, market, products, phase, go_live, budget, notes)


@mcp.tool()
def tool_update_project(
    identifier: str,
    name: str = "",
    project_type: str = "",
    status: str = "",
    description: str = "",
    market: str = "",
    products: str = "",
    phase: str = "",
    go_live: str = "",
    budget: str = "",
    notes: str = "",
) -> dict | None:
    """Update project fields. Only non-empty values are updated.

    status options: active | paused | completed | cancelled
    """
    fields = {
        k: v for k, v in {
            "name": name, "type": project_type, "status": status,
            "description": description, "market": market, "products": products,
            "phase": phase, "go_live": go_live, "budget": budget, "notes": notes,
        }.items() if v
    }
    return update_project(identifier, **fields)


@mcp.tool()
def tool_delete_project(identifier: str) -> bool:
    """Delete ANY project by slug or name — general-purpose, irreversible, no
    built-in scope restriction to test/sandbox data.

    Cascades to its non-shared contacts, notes, and project links; clears the
    active session and removes the project's docs folder from disk. Shared
    contacts (is_shared=True) are re-parented to another surviving project
    first — they are explicitly cross-project, so deleting whichever project
    happens to own the row must not remove it for every other project
    surfacing it too. Only if this is the last remaining project (nothing to
    re-parent to) are they cascade-deleted along with it. Callers (e.g. the
    delete-testdata skill) are responsible for their own safety checks before
    calling this — a materially larger blast radius than tool_delete_contact/
    tool_delete_note (which only ever remove one row each), even though none
    of the three have a built-in domain guard. Added for the
    create-testdata/reset-testdata/delete-testdata sandbox convention
    (project-hub#82) — there is no other way to fully remove a project, since
    its slug is UNIQUE and a soft-delete would block re-creation — but it is
    not restricted to that use.
    """
    return delete_project(identifier)


@mcp.tool()
def tool_list_docs(project_id: int) -> dict:
    """List all documents in a project's docs folder."""
    return list_docs(project_id)


@mcp.tool()
def tool_link_project(identifier: str, related_identifier: str, relation_type: str) -> dict:
    """Link two projects together, e.g. to mark a follow-up ("Joybuy 2" succeeds "Joybuy").

    relation_type: successor | predecessor | related
      successor    — identifier is the successor of related_identifier
      predecessor  — identifier is the predecessor of related_identifier
      related      — symmetric relation, no direction (e.g. two markets of the same rollout)

    Raises ValueError if either project is not found, if identifier == related_identifier,
    if relation_type is invalid, or if the two projects are already linked (unlink first to change it).
    Returns {"relation_type": ..., "project": {id, slug, name}, "related_project": {id, slug, name}}.
    """
    return link_project(identifier, related_identifier, relation_type)


@mcp.tool()
def tool_unlink_project(identifier: str, related_identifier: str) -> bool:
    """Remove the link between two projects, regardless of direction or relation type.

    Returns True if a link was removed, False if the two projects were not linked.
    """
    return unlink_project(identifier, related_identifier)


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_list_contacts(
    project_id: int, contact_type: str = "", limit: int = 50, offset: int = 0
) -> dict:
    """List contacts for a project with pagination.

    contact_type: internal | external (empty = all)
    Returns {"items": [...], "total": N, "limit": L, "offset": O}.
    """
    return list_contacts(project_id, contact_type, limit, offset)


@mcp.tool()
def tool_list_shared_contacts(limit: int = 50, offset: int = 0) -> dict:
    """List all shared/global contacts available across every project.

    Shared contacts are internal contacts defined once and reused everywhere.
    Returns {"items": [...], "total": N, "limit": L, "offset": O}.
    Each item includes project_name and project_slug showing where the contact was created.
    """
    return list_shared_contacts(limit, offset)


@mcp.tool()
def tool_add_contact(
    project_id: int,
    name: str,
    role: str = "",
    contact_type: str = "internal",
    email: str = "",
    phone: str = "",
    company: str = "",
    notes: str = "",
    is_shared: bool = False,
    force: bool = False,
) -> dict:
    """Add a contact to a project.

    contact_type: internal (own company) | external (merchant, client, partner, vendor)
    is_shared: set True for internal contacts available across all projects (e.g. own-company employees)
    role examples: Onboarding PM, Tech Lead, Account Manager, Legal, Merchant PM, etc.
    force: bypass a similar-name duplicate warning. Only set this after the user has
    explicitly confirmed it is a different person — never to silence the error on retry.
    """
    return add_contact(
        project_id, name, role, contact_type, email, phone, company, notes, is_shared,
        force=force,
    )


@mcp.tool()
def tool_update_contact(
    contact_id: int,
    name: str = "",
    role: str = "",
    contact_type: str = "",
    email: str = "",
    phone: str = "",
    company: str = "",
    notes: str = "",
    is_shared: bool | None = None,
    force: bool = False,
) -> dict | None:
    """Update an existing contact. Only provided (non-empty) values are changed.

    is_shared: True = make contact globally available; False = make project-specific again.
    force: bypass a similar-name duplicate warning. Only set this after the user has
    explicitly confirmed it is a different person — never to silence the error on retry.
    """
    fields: dict[str, str | int] = {k: v for k, v in {
        "name": name, "role": role, "type": contact_type,
        "email": email, "phone": phone, "company": company, "notes": notes,
    }.items() if v}
    if is_shared is not None:
        fields["is_shared"] = int(is_shared)
    return update_contact(contact_id, force=force, **fields)


@mcp.tool()
def tool_delete_contact(contact_id: int) -> bool:
    """Delete a contact by ID."""
    return delete_contact(contact_id)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_list_notes(
    project_id: int, note_type: str = "", limit: int = 50, offset: int = 0
) -> dict:
    """List notes for a project with pagination.

    note_type: note | meeting-notes | email | decision | action-item (empty = all)
    Returns {"items": [...], "total": N, "limit": L, "offset": O}, newest first.
    """
    return list_notes(project_id, note_type, limit, offset)


@mcp.tool()
def tool_get_note(note_id: int) -> dict | None:
    """Get a specific note by ID."""
    return get_note(note_id)


@mcp.tool()
def tool_add_note(
    project_id: int,
    title: str,
    content: str,
    note_type: str = "note",
    agenda: str = "",
    source_paths: list[str] | None = None,
) -> dict:
    """Add a note to a project.

    note_type: note | meeting-notes | email | decision | action-item
    agenda: optional agenda to compare against (used by /summarize for meeting-notes)
    content: the raw text (email body, meeting transcript, free-form notes, etc.)
    source_paths: absolute paths of local files `content` was extracted/imported from.
        Preserves each original under the project's attachments folder in the same
        call — always pass this when content came from a real file/folder the user
        referenced, not just when they explicitly ask to attach something
        (project-hub#134). A bad individual path never fails the note creation;
        check the returned "attachment_failures" field for per-file errors.
        Returns {"attachments_added": [...]} and, if any path failed,
        {"attachment_failures": [{"path": str, "error": str}, ...]}.
    """
    return add_note(project_id, title, content, note_type, agenda, source_paths)


@mcp.tool()
def tool_update_note(
    note_id: int,
    title: str = "",
    content: str = "",
    note_type: str = "",
    agenda: str = "",
) -> dict | None:
    """Update an existing note. Only non-empty values are changed.

    note_type: note | meeting-notes | email | decision | action-item
    """
    return update_note(note_id, title, content, note_type, agenda)


@mcp.tool()
def tool_delete_note(note_id: int) -> bool:
    """Delete a note by ID."""
    return delete_note(note_id)


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_attach_file(note_id: int, file_path: str) -> dict:
    """Copy a local file to the note's attachments folder and store the reference.

    file_path: absolute path to the source file (original is not moved/deleted).
    Returns {"name": str, "path": str, "size": int}.
    Raises ValueError if note not found or path traversal detected.
    Prints a warning to stderr if file is larger than 10 MB.
    """
    return attach_file(note_id, file_path)


@mcp.tool()
def tool_attach_files(note_id: int, file_paths: list[str]) -> dict:
    """Copy multiple local files to the note's attachments folder in one call.

    Use this whenever a user references several files or a whole folder in
    chat for analysis/import — list the folder first (ls/Glob), then pass
    every resulting path here so the originals are preserved, not just the
    extracted text (project-hub#134). A bad individual path never aborts the
    rest of the batch.

    file_paths: absolute paths to the source files (originals are not moved/deleted).
    Returns {"attached": [{"name", "path", "size"}, ...], "failed": [{"path", "error"}, ...]}.
    Raises ValueError if the note itself is not found, or if its project has no
    docs_path configured — both note-level problems, unlike a per-file bad path.
    """
    return attach_files(note_id, file_paths)


@mcp.tool()
def tool_list_attachments(note_id: int) -> list[dict]:
    """List all attachments for a note.

    Returns [{"name": str, "path": str, "size": int}, ...].
    """
    return list_attachments(note_id)


@mcp.tool()
def tool_remove_attachment(note_id: int, file_name: str) -> dict:
    """Remove an attachment from a note by file name.

    Deletes the file from disk and removes the reference from the DB.
    Raises ValueError if the note or attachment is not found.
    """
    remove_attachment(note_id, file_name)
    return {"removed": file_name}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_search_notes(query: str, project_id: int = 0) -> list[dict]:
    """Search notes by title or content (case-insensitive).

    project_id: limit to a specific project (0 = search all projects).
    Returns matches ordered newest first, including project_name for cross-project results.
    """
    return search_notes(query, project_id)


@mcp.tool()
def tool_search_contacts(query: str, project_id: int = 0) -> list[dict]:
    """Search contacts by name, role, email, or company (case-insensitive).

    project_id: limit to a specific project (0 = search all projects).
    Returns matches ordered by name, including project_name for cross-project results.
    """
    return search_contacts(query, project_id)


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_list_knowledge(project_type: str) -> list[dict]:
    """List all knowledge topics available for a given project type.

    Returns topic name, title (from H1), file path, and size.
    project_type examples: merchant-onboarding, it-project, generic
    """
    return list_knowledge(project_type)


@mcp.tool()
def tool_get_knowledge(project_type: str, topic: str) -> dict | None:
    """Read a knowledge file by project type and topic name.

    Returns full content as Markdown string plus metadata.
    Returns None if not found.
    """
    return get_knowledge(project_type, topic)


@mcp.tool()
def tool_get_all_knowledge(project_type: str) -> list[dict]:
    """Load ALL knowledge files for a project type in one call.

    Used by /resume to auto-load governance, process, and role knowledge
    when switching to a merchant-onboarding project.
    """
    return get_all_knowledge(project_type)


@mcp.tool()
def tool_save_knowledge(project_type: str, topic: str, content: str) -> dict:
    """Write or overwrite a knowledge file.

    content: full Markdown content (start with # Title)
    topic: filename without extension (e.g. 'governance', 'process', 'roles')
    Returns the saved knowledge entry with metadata.
    """
    return save_knowledge(project_type, topic, content)


@mcp.tool()
def tool_delete_knowledge(project_type: str, topic: str) -> bool:
    """Delete a knowledge file. Returns True if deleted, False if not found."""
    return delete_knowledge(project_type, topic)


# ---------------------------------------------------------------------------
# Project Types
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_list_project_types() -> list[dict]:
    """List all available project types: built-in and user-defined custom types.

    Each entry includes: name, label, description, source (built-in | custom), path.
    Custom types override built-ins when names collide.
    """
    return list_project_types()


@mcp.tool()
def tool_get_project_type(type_name: str) -> dict | None:
    """Get project type details by name.

    Checks user-defined types in ~/.project-hub/project-types/ first,
    then falls back to built-in types shipped with the plugin.
    Returns readme content plus metadata, or None if not found.
    """
    return get_project_type(type_name)


@mcp.tool()
def tool_create_project_type(
    name: str,
    description: str = "",
    phases: list[str] | None = None,
    contacts: list[str] | None = None,
) -> dict:
    """Create a custom project type scaffold in ~/.project-hub/project-types/{name}/.

    name: display name (will be slugified, e.g. "HR Onboarding" → "hr-onboarding")
    description: one-line description of this project type
    phases: optional list of standard phase names
    contacts: optional list of typical contact roles
    Returns error dict if the type already exists.
    """
    return create_project_type(name, description, phases, contacts)


@mcp.tool()
def tool_delete_project_type(type_name: str) -> dict:
    """Delete a custom project type.

    Built-in types shipped with the plugin cannot be deleted.
    Returns {"deleted": True, "name": ...} on success or {"error": ...} on failure.
    """
    return delete_project_type(type_name)


@mcp.tool()
def tool_get_project_template(project_type: str = "generic") -> dict:
    """Return the fillable template for a project type.

    The template contains YAML frontmatter with all fields for the type.
    Users can copy it, fill it in outside the chat, then import it via
    tool_create_project_from_template.

    Falls back to the generic template when no type-specific template exists.
    Returns {"project_type": ..., "template_content": ..., "fallback"?: true}.
    """
    return get_project_template(project_type)


@mcp.tool()
def tool_create_project_from_template(
    template_content: str = "",
    file_path: str = "",
) -> dict:
    """Create a project from a filled-out template.

    Accepts either:
    - template_content: the filled template pasted as text
    - file_path: absolute path to a saved .md template file

    Parses the YAML frontmatter, validates required fields, and creates the project.
    Returns the same result as tool_create_project on success, or {"error": ...} on failure.
    """
    return create_project_from_template(template_content=template_content, file_path=file_path)


# ---------------------------------------------------------------------------
# Reports  (static HTML export, issue #26)
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_generate_report(
    project_id: int | None = None,
    report_type: str = "full",
    output_path: str = "",
    offline: bool = False,
) -> dict:
    """Generate a static, self-contained HTML report for a project or all projects.

    project_id: numeric project ID (use tool_get_project to find it); None for all-projects
    report_type: "full" | "summary" | "all-projects"
      full          — all sections: header, contacts, charts, action items, full notes timeline
      summary       — header, open action items, last 5 activities (1-page executive view)
      all-projects  — cross-project table with status/type charts (project_id ignored)
    output_path: destination file path; defaults to ~/.project-hub/reports/{slug}-{report_type}-{date}.html
    offline: reserved for future offline CDN-inlining support (currently ignored)

    Returns {"path": str, "project": str}.
    Open the returned path in a browser; use Print → Save as PDF for PDF export.
    """
    return generate_report(project_id, report_type, output_path, offline)


# ---------------------------------------------------------------------------
# Export / Import  (multi-user / shared-DB handoff)
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_export_project(project_id: int, output_path: str = "") -> dict:
    """Export a project with all contacts, notes, and linked projects as a portable JSON file.

    Useful for sharing with a colleague or moving to a different DB.
    Attachments are NOT included in Phase 1. Linked projects (tool_link_project)
    are exported by slug, not DB ID.

    output_path: optional destination; defaults to ~/.project-hub/exports/{slug}-{date}.json
    Returns {"path": str, "project": str, "contacts": int, "notes": int, "links": int}.
    """
    return export_project(project_id, output_path)


@mcp.tool()
def tool_import_project(json_path: str, merge_strategy: str = "skip") -> dict:
    """Import a project from a JSON file created by tool_export_project.

    merge_strategy controls what happens when a project with the same slug exists:
      skip      — abort and return {"imported": False} (safe default)
      rename    — insert with a unique slug suffix
      overwrite — replace the existing project (destructive!)

    Linked projects are restored by slug if the linked project already exists in
    this DB; otherwise they are skipped and listed under "links_not_restored".

    Returns a summary with project name, slug, and counts.
    """
    return import_project(json_path, merge_strategy)


@mcp.tool()
def tool_sync_knowledge_templates(force: bool = False) -> dict:
    """Compare bundled plugin templates with the user's installed knowledge files.

    Reports new files and files where the plugin ships a different version.
    Set force=True to copy all new/changed templates to ~/.project-hub/knowledge/.
    User confirmation should be obtained before calling with force=True.

    Returns: {items: [{path, status, plugin_bytes, local_bytes?}], synced: [path, ...]}
    status values: "up-to-date" | "newer-version-available" | "new"
    """
    return sync_knowledge_templates(force=force)
