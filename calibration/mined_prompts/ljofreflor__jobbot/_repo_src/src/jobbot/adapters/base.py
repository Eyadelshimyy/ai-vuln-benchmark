"""Shared portal adapter protocols."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Protocol

from jobbot.models.candidate import Candidate
from jobbot.models.external_profile import ExternalProfile
from jobbot.models.job import JobPosting
from jobbot.models.sync import SyncPlan, SyncResult


class ProfileDiff(Protocol):
    def render(self) -> str: ...


class ProfileAdapter(Protocol):
    def login(self) -> None: ...

    def pull_profile(self) -> ExternalProfile: ...

    def build_diff(
        self,
        candidate: Candidate,
        external: ExternalProfile,
    ) -> object: ...


class WritableProfileAdapter(ProfileAdapter, Protocol):
    def build_sync_plan(
        self,
        candidate: Candidate,
        external: ExternalProfile,
    ) -> SyncPlan: ...

    def apply_sync_plan(self, plan: SyncPlan) -> SyncResult: ...


class ApplyMethod(StrEnum):
    EASY_APPLY = "easy_apply"
    EXTERNAL_ATS = "external_ats"
    EMAIL = "email"
    LINKEDIN_MESSAGE = "linkedin_message"
    UNKNOWN = "unknown"


class PrefillResult:
    def __init__(
        self,
        filled: list[str] | None = None,
        needs_review: list[str] | None = None,
    ) -> None:
        self.filled = filled or []
        self.needs_review = needs_review or []


class ApplicationPackage(Protocol):
    """Filesystem package prepared for a job application."""

    @property
    def directory(self) -> Path: ...

    @property
    def cv_pdf(self) -> Path | None: ...


class ApplicationPortalAdapter(Protocol):
    """Future portals inject Candidate + package — never invent answers."""

    name: str

    def detect_method(self, job: JobPosting) -> ApplyMethod: ...

    def prefill(
        self,
        candidate: Candidate,
        job: JobPosting,
        package: ApplicationPackage,
    ) -> PrefillResult: ...

    def attach_cv(self, package: ApplicationPackage) -> None: ...
