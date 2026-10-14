"""Contact CRUD operations."""
from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import Optional

from .db import db_connection

# Names this similar are reported as a near match. Deliberately below the 0.897 of
# "Ben Zimmermann" vs. "Sven Zimmermann" — distinct people do score this high, which is
# why a near match is forceable and an exact one is not.
_NEAR_MATCH_THRESHOLD = 0.88

# Characters with no NFKD decomposition, which would otherwise be dropped and split a
# token in half ("Søren" -> "s ren"). Transliterated to their conventional ASCII form.
_TRANSLIT_MAP = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "ẞ": "ss",
    "Ä": "ae", "Ö": "oe", "Ü": "ue",
    "ø": "oe", "Ø": "oe", "æ": "ae", "Æ": "ae", "å": "aa", "Å": "aa",
    "ł": "l", "Ł": "l", "đ": "d", "Đ": "d",
})


def list_contacts(
    project_id: int, contact_type: str = "", limit: int = 50, offset: int = 0
) -> dict:
    """List contacts for a project with pagination.

    Returns {"items": [...], "total": N, "limit": L, "offset": O}.
    """
    with db_connection() as conn:
        where = "WHERE project_id = ?"
        params: list = [project_id]
        if contact_type:
            where += " AND type = ?"
            params.append(contact_type)

        total: int = conn.execute(
            f"SELECT COUNT(*) FROM contacts {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"SELECT * FROM contacts {where} ORDER BY type, name LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }


def list_shared_contacts(limit: int = 50, offset: int = 0) -> dict:
    """List all shared contacts across all projects.

    Returns {"items": [...], "total": N, "limit": L, "offset": O}.
    Each item includes project_name and project_slug for context.
    """
    with db_connection() as conn:
        total: int = conn.execute(
            """SELECT COUNT(*) FROM contacts c
               JOIN projects p ON p.id = c.project_id
               WHERE c.is_shared = 1"""
        ).fetchone()[0]

        rows = conn.execute(
            """SELECT c.*, p.name as project_name, p.slug as project_slug
               FROM contacts c
               JOIN projects p ON p.id = c.project_id
               WHERE c.is_shared = 1
               ORDER BY c.type, c.name LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }


def _normalize_name(name: str) -> str:
    """Fold a person name to a comparable form, preserving token order.

    Handles the variants that occur when the same person is entered from different
    sources: casing, "Lastname, Firstname" order (Teams profile format), hyphens vs.
    spaces, umlaut transliteration and accents.

    Token order is preserved on purpose: two names folding to the same string here means
    they are the same person, which is what makes an exact match non-forceable. Order
    differences are a near match instead — see _match_kind.
    """
    n = name.strip()
    if n.count(",") == 1:
        last, first = (part.strip() for part in n.split(","))
        if last and first:
            n = f"{first} {last}"
    # NFC first: decomposed input ("u" + combining diaeresis) must become "ü" before
    # transliteration, or the combining mark is stripped below and it folds to "u".
    n = unicodedata.normalize("NFC", n).translate(_TRANSLIT_MAP)
    n = unicodedata.normalize("NFKD", n)
    n = "".join(ch for ch in n if not unicodedata.combining(ch))
    # isalnum() rather than [a-z0-9] so non-Latin scripts survive instead of folding to
    # an empty key, which would silently disable duplicate detection for them.
    n = "".join(ch if ch.isalnum() else " " for ch in n.lower())
    return " ".join(n.split())


def _match_kind(a: str, b: str) -> str:
    """Classify two raw names as "exact", "near" or "" (unrelated).

    "exact" means the names are the same once spelling variants are folded away — there
    is no legitimate reason to have both, so it cannot be forced.

    "near" covers heuristics that are usually but not always the same person: token
    permutations ("Thomas Michael" / "Michael Thomas"), subsets ("Jan Wulf" /
    "Jan Kalle Wulf"), and high overall similarity ("Mathias" / "Matthias"). These are
    forceable, because distinct people do legitimately land here.
    """
    na, nb = _normalize_name(a), _normalize_name(b)
    if not na or not nb:
        return ""
    if na == nb:
        return "exact"

    ta, tb = na.split(), nb.split()
    if sorted(ta) == sorted(tb):
        return "near"  # same tokens, different order
    shorter, longer = (set(ta), set(tb)) if len(ta) <= len(tb) else (set(tb), set(ta))
    if shorter < longer:
        return "near"  # dropped middle name, added double-barrelled surname
    if SequenceMatcher(None, na, nb).ratio() >= _NEAR_MATCH_THRESHOLD:
        return "near"
    return ""


def _find_duplicate_shared_contact(
    name: str, email: str = "", exclude_id: Optional[int] = None
) -> tuple[Optional[dict], str]:
    """Find a shared contact that is likely the same person as (name, email).

    Returns (contact, kind) where kind is "exact", "near" or "" — see _match_kind.
    An identical email always counts as exact, whatever the names look like.

    Comparison happens in Python rather than SQL because normalization (name order,
    punctuation, umlauts) cannot be expressed in SQLite's LOWER(). The shared-contact
    set is small enough that fetching it is cheaper than the alternative.
    exclude_id: ignore this contact id (needed when re-saving an already-shared contact).
    """
    sql = """
        SELECT c.*, p.name as project_name
        FROM contacts c
        JOIN projects p ON p.id = c.project_id
        WHERE c.is_shared = 1
    """
    params: list = []
    if exclude_id is not None:
        sql += " AND c.id != ?"
        params.append(exclude_id)

    with db_connection() as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    email_lower = email.strip().lower()
    best_near: Optional[dict] = None
    best_ratio = -1.0

    for row in rows:
        if email_lower and (row["email"] or "").strip().lower() == email_lower:
            return row, "exact"
        kind = _match_kind(name, row["name"])
        if kind == "exact":
            return row, "exact"  # an exact match anywhere wins over any near match
        if kind == "near":
            # Report the closest near match, not whichever row SQLite happened to return
            # first — the error names a specific contact, so it should name the likeliest.
            ratio = SequenceMatcher(None, _normalize_name(name), _normalize_name(row["name"])).ratio()
            if ratio > best_ratio:
                best_near, best_ratio = row, ratio

    return (best_near, "near") if best_near else (None, "")


def _duplicate_message(existing: dict, kind: str) -> str:
    where = f"(id={existing['id']}, project='{existing['project_name']}')"
    if kind == "exact":
        return (
            f"A shared contact with this name or email already exists: "
            f"'{existing['name']}' {where}. "
            f"This contact is available in all projects — no need to add it again."
        )
    return (
        f"A shared contact with a very similar name already exists: "
        f"'{existing['name']}' {where}. "
        f"If this is the same person, update that contact instead of creating a new one. "
        f"Do not retry with force=True unless the user has explicitly confirmed that this "
        f"is a different person."
    )


def add_contact(
    project_id: int,
    name: str,
    role: str = "",
    contact_type: str = "internal",
    email: str = "",
    phone: str = "",
    company: str = "",
    notes: str = "",
    is_shared: bool = False,
    *,
    force: bool = False,
) -> dict:
    """Add a contact to a project.

    is_shared: if True, the contact is available across all projects.
    Only internal contacts can be shared; external contacts are always project-specific.
    Raises ValueError if a shared contact with the same or a very similar name exists.
    force: allow creation despite a similar-name match. Exact matches are never forceable.
    """
    if is_shared and contact_type == "external":
        raise ValueError("External contacts cannot be shared across projects.")
    existing, kind = _find_duplicate_shared_contact(name, email)
    # A near match only blocks a contact that would itself be shared. Project-local
    # contacts are legitimately scoped to their project, so a heuristic name similarity
    # against a shared contact is not enough to refuse them.
    if kind == "near" and not is_shared:
        existing, kind = None, ""
    if existing and not (force and kind == "near"):
        raise ValueError(_duplicate_message(existing, kind))
    with db_connection() as conn:
        with conn:
            cursor = conn.execute(
                """INSERT INTO contacts (project_id, name, role, type, email, phone, company, notes, is_shared)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (project_id, name, role, contact_type, email, phone, company, notes, int(is_shared)),
            )
            contact_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,)).fetchone()
        return dict(row)


def update_contact(contact_id: int, *, force: bool = False, **fields) -> Optional[dict]:
    """Update fields on an existing contact.

    Raises ValueError if trying to share an external contact.
    force: allow the update despite a similar-name match against an existing shared
    contact. Exact matches are never forceable. Keyword-only, so a positional second
    argument stays a TypeError instead of being silently swallowed as a truthy force.
    """
    allowed = {"name", "role", "type", "email", "phone", "company", "notes", "is_shared"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}

    # Fetch current state once if needed for validation
    needs_current = updates.get("is_shared") or "name" in updates or "email" in updates
    current_dict: dict = {}
    if needs_current:
        with db_connection() as conn:
            row = conn.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,)).fetchone()
        current_dict = dict(row) if row else {}

    if updates.get("is_shared"):
        effective_type = updates.get("type") or current_dict.get("type", "internal")
        if effective_type == "external":
            raise ValueError("External contacts cannot be shared across projects.")

    # Block if effective name/email would collide with any existing shared contact
    if "name" in updates or "email" in updates or updates.get("is_shared"):
        effective_name = updates.get("name") or current_dict.get("name", "")
        effective_email = updates.get("email") or current_dict.get("email", "")
        existing, kind = _find_duplicate_shared_contact(
            effective_name, effective_email, exclude_id=contact_id
        )
        # Mirror add_contact: a near match only blocks a contact that ends up shared.
        effective_shared = updates.get("is_shared", current_dict.get("is_shared", 0))
        if kind == "near" and not effective_shared:
            existing, kind = None, ""
        if existing and not (force and kind == "near"):
            raise ValueError(_duplicate_message(existing, kind))

    if not updates:
        with db_connection() as conn:
            row = conn.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,)).fetchone()
            return dict(row) if row else None

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [contact_id]

    with db_connection() as conn:
        with conn:
            conn.execute(
                f"UPDATE contacts SET {set_clause} WHERE id = ?", values
            )
        row = conn.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,)).fetchone()
        return dict(row) if row else None


def delete_contact(contact_id: int) -> bool:
    with db_connection() as conn:
        with conn:
            result = conn.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
        return result.rowcount > 0
