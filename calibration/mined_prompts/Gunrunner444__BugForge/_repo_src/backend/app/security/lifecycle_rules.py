"""Shared reproduction and verification predicates.

The rules are defined in :mod:`app.domain.lifecycle_policy`. This module
keeps the service and research-agent imports stable and adapts them to a
``SecurityFinding``.
"""

from __future__ import annotations

from app.domain.evidence import Evidence
from app.domain.findings import SecurityFinding
from app.domain.lifecycle_policy import (
    INDEPENDENT_VERIFICATION_KINDS,
    can_corroborate,
    positive_reproduction,
)
from app.domain.lifecycle_policy import (
    independent_verification_items as _independent_items,
)
from app.domain.target_identity import semantic_target_identity

__all__ = [
    "INDEPENDENT_VERIFICATION_KINDS",
    "can_corroborate",
    "independent_verification_items",
    "positive_reproduction",
]


def independent_verification_items(finding: SecurityFinding) -> tuple[Evidence, ...]:
    """Verification observations on a finding. See the domain policy."""
    return _independent_items(
        finding.evidence.items,
        target_id=semantic_target_identity(finding),
        finding_id=str(finding.id),
        finding_key=finding.finding_key,
        project_id=finding.project_id,
    )
