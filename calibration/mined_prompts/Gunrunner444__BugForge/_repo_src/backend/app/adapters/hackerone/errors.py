"""HackerOne API errors. Messages never include credentials."""

from __future__ import annotations

from app.security_testing.errors import SecurityTestingError


class HackerOneError(SecurityTestingError):
    def __init__(self, message: str, *, status_code: int | None = None, code: str = "") -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class HackerOneAuthError(HackerOneError):
    pass


class HackerOneRateLimitError(HackerOneError):
    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(message, status_code=429, code="rate_limited")


class HackerOneIdentityVerificationError(HackerOneError):
    def __init__(self, message: str = "HackerOne identity verification is required") -> None:
        super().__init__(message, status_code=403, code="identity_verification_required")


class HackerOneUrlRejectedError(HackerOneError):
    def __init__(self, message: str = "HackerOne client rejected an unauthorized URL") -> None:
        super().__init__(message, status_code=400, code="url_rejected")


class HackerOneSubmissionUnknownError(HackerOneError):
    def __init__(self, message: str = "HackerOne submission outcome is unknown") -> None:
        super().__init__(message, code="submission_outcome_unknown")


class HackerOneSubmissionInProgressError(HackerOneError):
    def __init__(
        self, message: str = "A HackerOne submission claim is already in progress"
    ) -> None:
        super().__init__(message, status_code=409, code="submission_in_progress")


class HackerOneReconciliationAmbiguousError(HackerOneError):
    def __init__(self, message: str = "Remote HackerOne report match is ambiguous") -> None:
        super().__init__(message, status_code=409, code="reconciliation_ambiguous")
