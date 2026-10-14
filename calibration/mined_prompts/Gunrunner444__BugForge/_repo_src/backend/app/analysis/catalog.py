"""Separate catalogs for code quality vs security observations."""

from __future__ import annotations

from enum import StrEnum


class FindingCatalog(StrEnum):
    CODE_QUALITY = "code_quality"
    SECURITY = "security"
