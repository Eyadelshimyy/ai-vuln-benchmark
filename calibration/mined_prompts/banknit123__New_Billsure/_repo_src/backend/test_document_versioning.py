"""
Standalone tests for document_versioning.py. Same in-memory fake-DB
pattern as the other test_*.py files, no live credentials needed.

Run: python3 test_document_versioning.py
"""
import asyncio
import sys
import types
import uuid
from datetime import datetime, timezone, timedelta

# ---- in-memory fake of supabase_db's public interface ----
_tables = {}


def _matches(row, filters):
    for k, v in filters.items():
        if row.get(k) != v:
            return False
    return True


async def find_one(table, filters, exclude_fields=None):
    for row in _tables.get(table, []):
        if _matches(row, filters):
            return dict(row)
    return None


async def find_many(table, filters=None, exclude_fields=None, limit=10000):
    filters = filters or {}
    return [dict(r) for r in _tables.get(table, []) if _matches(r, filters)][:limit]


async def insert_one(table, data):
    row = dict(data)
    row.setdefault("id", str(uuid.uuid4()))
    _tables.setdefault(table, []).append(row)
    return dict(row)


async def update_one(table, filters, updates):
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(updates)
            return True
    return False


fake_sdb = types.SimpleNamespace(find_one=find_one, find_many=find_many, insert_one=insert_one, update_one=update_one)
sys.modules["supabase_db"] = fake_sdb

import document_versioning as dv   # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


async def main():
    # ---------------------------------------------------------------
    # Version creation: unknown types rejected, hash computed
    # ---------------------------------------------------------------
    try:
        await dv.create_document_version("not_a_real_doc_type", b"content", "2026-01-01", "legal1", False)
        check("rejects an unknown document_type", False)
    except dv.DocumentVersioningError:
        check("rejects an unknown document_type", True)

    v1 = await dv.create_document_version("credit_guide", b"Credit Guide v1 content", "2026-01-01", "legal1", is_material_change=False)
    check("first version of a document_type is version 1", v1["version"] == 1)
    check("new version starts as 'draft', not active", v1["status"] == "draft")
    check("version is stamped as a template by default, with the warning text", v1["is_template"] is True and "REQUIRES AUSTRALIAN LEGAL APPROVAL" in v1["template_warning"])
    check("content hash is computed correctly", v1["content_hash"] == dv.compute_document_hash(b"Credit Guide v1 content"))

    no_active = await dv.get_active_document("credit_guide")
    check("no active version exists before any approval", no_active is None)

    # ---------------------------------------------------------------
    # Approval: maker-checker, activates exactly one version
    # ---------------------------------------------------------------
    try:
        await dv.approve_document_version(v1["id"], approved_by="legal1")
        check("rejects approval by the same person who created the version", False)
    except dv.DocumentVersioningError:
        check("rejects approval by the same person who created the version", True)

    approved_v1 = await dv.approve_document_version(v1["id"], approved_by="compliance1")
    check("approval with a distinct approver succeeds", approved_v1["status"] == "approved")

    active = await dv.get_active_document("credit_guide")
    check("get_active_document now returns the approved version", active["id"] == v1["id"])

    try:
        await dv.approve_document_version(v1["id"], approved_by="compliance2")
        check("rejects re-approving an already-approved version", False)
    except dv.DocumentVersioningError:
        check("rejects re-approving an already-approved version", True)

    # ---------------------------------------------------------------
    # A second version approval archives the first
    # ---------------------------------------------------------------
    v2 = await dv.create_document_version("credit_guide", b"Credit Guide v2 content, updated fees section", "2026-03-01", "legal2", is_material_change=True)
    approved_v2 = await dv.approve_document_version(v2["id"], approved_by="compliance1")
    check("second version approval succeeds", approved_v2["status"] == "approved")

    active_now = await dv.get_active_document("credit_guide")
    check("the active version is now v2, not v1", active_now["id"] == v2["id"] and active_now["version"] == 2)

    v1_after = await find_one("document_versions", {"id": v1["id"]})
    check("the previously-active v1 is automatically archived when v2 is approved", v1_after["status"] == "archived")

    archived = await dv.list_archived_versions("credit_guide")
    check("list_archived_versions returns exactly the archived v1", len(archived) == 1 and archived[0]["id"] == v1["id"])

    # ---------------------------------------------------------------
    # Customer acceptance: can only accept the currently-approved version
    # ---------------------------------------------------------------
    draft_v3 = await dv.create_document_version("credit_guide", b"draft v3", "2026-06-01", "legal1", is_material_change=False)
    try:
        await dv.record_customer_acceptance("cust-1", "credit_guide", draft_v3["id"])
        check("rejects a customer accepting a draft (not-yet-approved) version", False)
    except dv.DocumentVersioningError:
        check("rejects a customer accepting a draft (not-yet-approved) version", True)

    try:
        await dv.record_customer_acceptance("cust-1", "credit_guide", v1["id"])
        check("rejects a customer accepting an already-archived version", False)
    except dv.DocumentVersioningError:
        check("rejects a customer accepting an already-archived version", True)

    acceptance = await dv.record_customer_acceptance("cust-1", "credit_guide", v2["id"], ip_address="203.0.113.5")
    check("accepting the currently-approved version succeeds", acceptance["version_number"] == 2)

    accepted_docs = await dv.get_customer_accepted_documents("cust-1")
    check("get_customer_accepted_documents returns the acceptance", len(accepted_docs) == 1)

    # ---------------------------------------------------------------
    # reproduce_accepted_document: exact content, integrity-checked
    # ---------------------------------------------------------------
    reproduced = await dv.reproduce_accepted_document(acceptance["id"])
    check("reproduced document content matches exactly what v2 contained", reproduced["content"] == "Credit Guide v2 content, updated fees section")
    check("reproduced document is reconstructable even though a newer version could later be published", reproduced["version_number"] == 2)

    # Simulate tampering: corrupt the stored content after the fact.
    await update_one("document_versions", {"id": v2["id"]}, {"content": "TAMPERED CONTENT"})
    try:
        await dv.reproduce_accepted_document(acceptance["id"])
        check("detects and refuses to serve tampered document content (hash mismatch)", False)
    except dv.DocumentVersioningError:
        check("detects and refuses to serve tampered document content (hash mismatch)", True)
    # restore for subsequent checks
    await update_one("document_versions", {"id": v2["id"]}, {"content": "Credit Guide v2 content, updated fees section"})

    # ---------------------------------------------------------------
    # requires_reacceptance: only material changes force it
    # ---------------------------------------------------------------
    never_accepted = await dv.requires_reacceptance("cust-never-accepted-anything", "credit_guide")
    check("a customer who never accepted anything for this document_type requires acceptance", never_accepted is True)

    up_to_date = await dv.requires_reacceptance("cust-1", "credit_guide")
    check("a customer who already accepted the currently-active version does not need to re-accept", up_to_date is False)

    # Publish a NON-material new version -- should not force re-acceptance.
    v3_minor = await dv.create_document_version("credit_guide", b"Credit Guide v2 content, updated fees section (typo fix)", "2026-07-01", "legal1", is_material_change=False)
    await dv.approve_document_version(v3_minor["id"], approved_by="compliance1")
    minor_change_reaccept = await dv.requires_reacceptance("cust-1", "credit_guide")
    check("a non-material change does NOT force re-acceptance, even though a newer version now exists", minor_change_reaccept is False)

    # Publish a MATERIAL new version -- should force re-acceptance.
    v4_material = await dv.create_document_version("credit_guide", b"Credit Guide v4, materially different fee structure", "2026-08-01", "legal1", is_material_change=True)
    await dv.approve_document_version(v4_material["id"], approved_by="compliance1")
    material_change_reaccept = await dv.requires_reacceptance("cust-1", "credit_guide")
    check("a material change DOES force re-acceptance", material_change_reaccept is True)

    # After re-accepting the material version, no longer required.
    await dv.record_customer_acceptance("cust-1", "credit_guide", v4_material["id"])
    after_reaccept = await dv.requires_reacceptance("cust-1", "credit_guide")
    check("re-accepting the material version clears the requirement", after_reaccept is False)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
