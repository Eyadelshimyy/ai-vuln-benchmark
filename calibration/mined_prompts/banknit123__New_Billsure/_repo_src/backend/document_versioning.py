"""
backend/document_versioning.py
=================================
Versioned document management and customer acceptance tracking for the
14 disclosure/agreement document types the ASIC ERS pilot needs (task
section 11).

Hard rule this module exists to enforce: **this codebase never drafts
final legal wording.** Every document type this module manages is
either (a) an operational artefact this module versions but does not
author (a real customer agreement, credit guide, etc. — drafted by
qualified Australian legal counsel, then fed into this module purely as
content to hash/version/track acceptance for), or (b) a clearly-marked
structural template with placeholder text only — see
`docs/asic-ers-readiness/document-templates/` for the 14 template
skeletons, each headed with an explicit "REQUIRES AUSTRALIAN LEGAL
APPROVAL" banner and containing section headings and bracketed
placeholders, never actual legal sentences.

Design principles:
- A document version is immutable once created — its content and hash
  never change after creation. A "change" is a new version, with the
  previous version archived, not edited. This mirrors pilot_config.py's
  and credit_ledger.py's "never mutate, only append" discipline.
- Approval is maker-checker: whoever authored/uploaded a version cannot
  also approve it.
- Every customer acceptance is a permanent snapshot of exactly which
  document_type + version + content hash was accepted, when — so
  `reproduce_accepted_document()` can always answer "what did this
  customer actually agree to" even after the document has since changed
  or been archived. This is the concrete implementation of "ability to
  reproduce the exact documents accepted by each customer."
- Only a version explicitly flagged `is_material_change=True` forces
  re-acceptance (`requires_reacceptance()`). A typo fix or formatting
  change is still a new version (so its own history exists), but it
  does not by itself force every existing customer to re-accept
  anything — only a human marking a version as materially different
  does that.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)

DOCUMENT_TYPES = (
    "ers_disclosure",
    "credit_guide",
    "credit_contract",
    "repayment_schedule_disclosure",
    "non_cash_payment_facility_terms",
    "product_disclosure_material",
    "target_market_determination",
    "privacy_policy",
    "privacy_collection_notice",
    "customer_funds_disclosure",
    "fees_and_remuneration_disclosure",
    "complaints_and_afca_information",
    "hardship_information",
    "exit_and_wind_down_disclosure",
)

VERSION_STATUSES = ("draft", "approved", "archived")

TEMPLATE_WARNING = (
    "TEMPLATE ONLY — REQUIRES AUSTRALIAN LEGAL APPROVAL. This is a "
    "structural skeleton (section headings and placeholders), not final "
    "legal wording. Do not present this to a customer or treat it as an "
    "approved disclosure until qualified Australian legal counsel has "
    "reviewed and approved the actual content."
)


class DocumentVersioningError(Exception):
    """Raised for any invalid document-versioning operation. Every raise
    path here is a refusal — there is no code path that activates or
    accepts a document version without the checks below having passed."""


def compute_document_hash(content: bytes) -> str:
    """SHA-256 of the exact document content — the immutable reference
    a customer acceptance snapshot points to. Same technique as
    bill_verification.compute_bill_hash(), same reasoning: this is what
    lets 'what did the customer actually accept' be answered later with
    certainty rather than by trusting a mutable pointer."""
    return hashlib.sha256(content).hexdigest()


async def create_document_version(document_type: str, content: bytes, effective_date: str,
                                   created_by: str, is_material_change: bool, is_template: bool = True) -> dict:
    """Creates a new DRAFT version. Does not activate it — see
    approve_document_version(). `is_template=True` (the default)
    stamps the TEMPLATE_WARNING onto the version record; set it to
    False only once real, legally-approved content is being versioned
    (this module trusts the caller's flag here rather than trying to
    detect "is this real legal text" itself, which isn't something
    code can determine)."""
    if document_type not in DOCUMENT_TYPES:
        raise DocumentVersioningError(f"unknown document_type: {document_type}")

    existing = await sdb.find_many("document_versions", {"document_type": document_type})
    next_version = max((v["version"] for v in existing), default=0) + 1

    return await sdb.insert_one("document_versions", {
        "document_type": document_type,
        "version": next_version,
        "content": content.decode("utf-8", errors="replace"),
        "content_hash": compute_document_hash(content),
        "effective_date": effective_date,
        "is_material_change": is_material_change,
        "is_template": is_template,
        "template_warning": TEMPLATE_WARNING if is_template else None,
        "status": "draft",
        "created_by": created_by,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": None,
        "approved_at": None,
    })


async def approve_document_version(version_id: str, approved_by: str) -> dict:
    """Maker-checker: approver must differ from the person who created
    the version. On approval, archives the currently-active version for
    the same document_type (if any) and activates this one — there is
    only ever one 'approved' (active) version per document_type at a
    time; everything else is 'draft' or 'archived'."""
    version = await sdb.find_one("document_versions", {"id": version_id})
    if not version:
        raise DocumentVersioningError(f"no document version {version_id}")
    if version.get("created_by") == approved_by:
        raise DocumentVersioningError("document approval requires a distinct approver from the creator (maker-checker)")
    if version.get("status") != "draft":
        raise DocumentVersioningError(f"version {version_id} is not in draft status (status={version.get('status')})")

    currently_active = await sdb.find_one("document_versions", {"document_type": version["document_type"], "status": "approved"})
    if currently_active:
        await sdb.update_one("document_versions", {"id": currently_active["id"]}, {"status": "archived"})

    updates = {"status": "approved", "approved_by": approved_by, "approved_at": datetime.now(timezone.utc).isoformat()}
    await sdb.update_one("document_versions", {"id": version_id}, updates)
    return {**version, **updates}


async def get_active_document(document_type: str) -> Optional[dict]:
    """The version a customer portal's 'view/download' action should
    serve — the one currently approved for this document_type."""
    return await sdb.find_one("document_versions", {"document_type": document_type, "status": "approved"})


async def list_archived_versions(document_type: str) -> list:
    return await sdb.find_many("document_versions", {"document_type": document_type, "status": "archived"})


async def record_customer_acceptance(customer_id: str, document_type: str, version_id: str,
                                      accepted_at: Optional[str] = None, ip_address: Optional[str] = None) -> dict:
    """Records that a customer accepted a SPECIFIC, immutable document
    version. Refuses to accept against anything other than the
    currently-approved version for that document_type — a customer
    should never be recorded as having 'accepted' a draft or an
    archived/superseded version, since that wouldn't reflect what they
    were actually shown."""
    version = await sdb.find_one("document_versions", {"id": version_id})
    if not version:
        raise DocumentVersioningError(f"no document version {version_id}")
    if version.get("document_type") != document_type:
        raise DocumentVersioningError("version_id does not belong to the specified document_type")
    if version.get("status") != "approved":
        raise DocumentVersioningError(f"cannot record acceptance of a non-approved version (status={version.get('status')})")

    return await sdb.insert_one("document_acceptances", {
        "customer_id": customer_id, "document_type": document_type, "version_id": version_id,
        "version_number": version["version"], "content_hash": version["content_hash"],
        "accepted_at": accepted_at or datetime.now(timezone.utc).isoformat(), "ip_address": ip_address,
    })


async def get_customer_accepted_documents(customer_id: str) -> list:
    return await sdb.find_many("document_acceptances", {"customer_id": customer_id})


async def reproduce_accepted_document(acceptance_id: str) -> dict:
    """Returns the EXACT document content and metadata a customer
    accepted, by content hash — reconstructable regardless of whether
    the document_type has since had newer versions published. Verifies
    the stored content still hashes to the value recorded at acceptance
    time (defence against any later tampering/corruption of the version
    row) and raises rather than silently returning content that no
    longer matches what was actually accepted."""
    acceptance = await sdb.find_one("document_acceptances", {"id": acceptance_id})
    if not acceptance:
        raise DocumentVersioningError(f"no acceptance record {acceptance_id}")
    version = await sdb.find_one("document_versions", {"id": acceptance["version_id"]})
    if not version:
        raise DocumentVersioningError(f"acceptance {acceptance_id} references a missing document version")

    actual_hash = compute_document_hash(version["content"].encode("utf-8"))
    if actual_hash != acceptance["content_hash"]:
        raise DocumentVersioningError(
            f"integrity check failed: stored document content hash ({actual_hash}) does not match "
            f"the hash recorded at acceptance time ({acceptance['content_hash']}) — content may have been altered"
        )

    return {
        "customer_id": acceptance["customer_id"], "document_type": acceptance["document_type"],
        "version_number": acceptance["version_number"], "content": version["content"],
        "content_hash": acceptance["content_hash"], "accepted_at": acceptance["accepted_at"],
    }


async def requires_reacceptance(customer_id: str, document_type: str) -> bool:
    """True if the customer's most recent acceptance for this
    document_type is NOT the currently-active version, AND the
    currently-active version is flagged as a material change. A
    non-material change (formatting, a typo fix) never forces
    re-acceptance by itself — only an explicit is_material_change=True
    on the active version does."""
    active = await get_active_document(document_type)
    if not active:
        return False

    acceptances = await sdb.find_many("document_acceptances", {"customer_id": customer_id, "document_type": document_type})
    if not acceptances:
        return True  # never accepted anything for this document_type at all

    latest_acceptance = max(acceptances, key=lambda a: a["accepted_at"])
    if latest_acceptance["version_id"] == active["id"]:
        return False

    return bool(active.get("is_material_change"))
