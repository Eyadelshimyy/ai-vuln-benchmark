"""Canonical HackerOne program-scope snapshot hashing."""

from __future__ import annotations

from app.adapters.hackerone.hashes import program_scope_content_hash
from app.adapters.hackerone.models import HackerOneProgram


def hash_program_scope(program: HackerOneProgram) -> str:
    scopes = [
        {
            "id": item.id,
            "asset_type": item.asset_type.value,
            "asset_type_raw": item.asset_type_raw,
            "asset_identifier": item.asset_identifier,
            "instruction": item.instruction,
            "eligible_for_submission": item.eligible_for_submission,
            "eligible_for_bounty": item.eligible_for_bounty,
        }
        for item in sorted(
            program.structured_scopes, key=lambda row: (row.id, row.asset_identifier)
        )
    ]
    exclusions = [
        {"id": item.id, "category": item.category, "details": item.details}
        for item in sorted(program.exclusions, key=lambda row: (row.id, row.category))
    ]
    weaknesses = [
        {"id": item.id, "name": item.name, "external_id": item.external_id}
        for item in sorted(program.weaknesses, key=lambda row: row.id)
    ]
    return program_scope_content_hash(
        handle=program.handle,
        scope_mode=program.scope_mode.value,
        instructions=program.instructions,
        active_testing_approved=program.active_testing_approved,
        requires_severity=program.requires_severity,
        scopes=scopes,
        exclusions=exclusions,
        weaknesses=weaknesses,
    )
