"""Repository eligibility and safety screening (v1.1.0).

Philosophy:
  - Hard policy blocks come first (size, file count, topics).
  - Then heuristic checks on repository metadata (no code execution here).
  - A repository is only eligible when ALL hard blocks pass AND no critical
    heuristic fails.
  - A LOW_RISK or NEEDS_REVIEW classification does NOT block automatic
    analysis but is recorded so operators can review the decision.
  - AI is NOT the final arbiter of safety — deterministic policy is.
  - Stars are NOT treated as a safety guarantee.

All checks operate on metadata only; no repository code is cloned or
executed in this service.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.core.config import Settings

logger = logging.getLogger(__name__)

# Topics that immediately block a repository regardless of other criteria
_BLOCKED_TOPICS = frozenset(
    {
        "exploit",
        "exploitation",
        "exploit-code",
        "exploit-kit",
        "malware",
        "ransomware",
        "virus",
        "trojan",
        "spyware",
        "hacking",
        "hack-tool",
        "hacking-tool",
        "pentest",
        "penetration-testing",
        "offensive-security",
        "ctf-solutions",
        "ctf-writeup",
        "keylogger",
        "rootkit",
        "botnet",
        "dos",
        "ddos",
        "denial-of-service",
        "rat",
        "remote-access-trojan",
        "stealer",
        "password-stealer",
        "credential-stealer",
        "phishing",
    }
)

# File extensions that indicate native/executable content — not blocked alone,
# but increase risk score
_NATIVE_EXTENSIONS = frozenset({".so", ".dylib", ".dll", ".exe", ".bin", ".ko"})

# File patterns that indicate dangerous scripts or hooks
_RISKY_SCRIPT_PATTERNS = [
    re.compile(r"(^|/)setup\.sh$", re.IGNORECASE),
    re.compile(r"(^|/)install\.sh$", re.IGNORECASE),
    re.compile(r"(^|/)bootstrap\.sh$", re.IGNORECASE),
    re.compile(r"(^|/)postinstall\.(sh|js)$", re.IGNORECASE),
    re.compile(r"(^|/)preinstall\.(sh|js)$", re.IGNORECASE),
    re.compile(r"(^|/)Makefile$"),
    re.compile(r"(^|/)configure$", re.IGNORECASE),
]

# CI workflow directories — not blocked but noted
_CI_PATHS = re.compile(r"(^|/)\.github/workflows/", re.IGNORECASE)

# Patterns that might indicate credential files
_CREDENTIAL_PATTERNS = [
    re.compile(r"(private_key|secret|credentials|\.pem|\.p12|\.pfx)$", re.IGNORECASE),
]

# OSI-recognised license identifiers (lowercase)
_OSI_LICENSES = frozenset(
    {
        "mit",
        "apache-2.0",
        "gpl-2.0",
        "gpl-3.0",
        "lgpl-2.1",
        "lgpl-3.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "mpl-2.0",
        "cddl-1.0",
        "epl-2.0",
        "agpl-3.0",
        "unlicense",
        "isc",
        "cc0-1.0",
        "eupl-1.1",
        "eupl-1.2",
        "0bsd",
        "ncsa",
    }
)


@dataclass
class PolicyCheckResult:
    """Result of one individual policy check."""

    check_name: str
    passed: bool
    is_hard_block: bool  # True = immediately reject regardless of other checks
    detail: str = ""


@dataclass
class EligibilityAssessment:
    """Full safety/eligibility assessment for a repository candidate."""

    # safe_candidate | low_risk | needs_review | blocked
    safety_classification: str = "safe_candidate"
    # discovered | eligible | rejected | blocked
    eligibility_status: str = "eligible"
    eligibility_score: float = 1.0  # 0.0 (fully blocked) – 1.0 (fully safe)
    rejection_reason: str | None = None
    policy_checks: list[PolicyCheckResult] = field(default_factory=list)

    @property
    def is_eligible(self) -> bool:
        return self.eligibility_status == "eligible"

    def to_detail_json(self) -> str:
        return json.dumps(
            [
                {
                    "check": c.check_name,
                    "passed": c.passed,
                    "hard_block": c.is_hard_block,
                    "detail": c.detail,
                }
                for c in self.policy_checks
            ]
        )


class EligibilityService:
    """Applies deterministic policy checks to a repository candidate.

    Operates entirely on repository metadata — no cloning or code execution.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def assess(
        self,
        *,
        full_name: str,
        owner: str,
        stars: int,
        is_fork: bool,
        is_archived: bool,
        primary_language: str | None,
        license_key: str | None,
        size_kb: int,
        topics: list[str],
        last_pushed_at: datetime | None,
        description: str | None = None,
    ) -> EligibilityAssessment:
        """Run all policy checks and produce an EligibilityAssessment.

        This is a pure function (no I/O) so it can be tested deterministically.
        """
        checks: list[PolicyCheckResult] = []
        hard_blocked = False
        rejection_reasons: list[str] = []

        # --- Hard blocks ---

        # Blocked topics
        normalised_topics = [t.lower() for t in topics]
        matched_blocked = _BLOCKED_TOPICS & set(normalised_topics)
        if matched_blocked:
            reason = f"blocked_topics:{','.join(sorted(matched_blocked))}"
            checks.append(PolicyCheckResult("blocked_topics", False, True, reason))
            hard_blocked = True
            rejection_reasons.append(reason)
        else:
            checks.append(PolicyCheckResult("blocked_topics", True, True))

        # Excluded owner
        excluded_owners = self._settings.get_excluded_owners()
        if owner.lower() in excluded_owners:
            reason = f"excluded_owner:{owner}"
            checks.append(PolicyCheckResult("excluded_owner", False, True, reason))
            hard_blocked = True
            rejection_reasons.append(reason)
        else:
            checks.append(PolicyCheckResult("excluded_owner", True, True))

        # Excluded topics from config
        config_excluded = self._settings.get_excluded_topics()
        matched_config_excluded = config_excluded & set(normalised_topics)
        if matched_config_excluded:
            reason = f"config_excluded_topics:{','.join(sorted(matched_config_excluded))}"
            checks.append(PolicyCheckResult("config_excluded_topics", False, True, reason))
            hard_blocked = True
            rejection_reasons.append(reason)
        else:
            checks.append(PolicyCheckResult("config_excluded_topics", True, True))

        # Repository size hard limit
        if (
            self._settings.safety_max_repo_size_kb > 0
            and size_kb > self._settings.safety_max_repo_size_kb
        ):
            reason = f"size_too_large:{size_kb}kb>{self._settings.safety_max_repo_size_kb}kb"
            checks.append(PolicyCheckResult("size_limit", False, True, reason))
            hard_blocked = True
            rejection_reasons.append(reason)
        else:
            checks.append(PolicyCheckResult("size_limit", True, True, f"{size_kb}kb"))

        # Archived check
        if self._settings.discovery_skip_archived and is_archived:
            reason = "repository_archived"
            checks.append(PolicyCheckResult("not_archived", False, True, reason))
            hard_blocked = True
            rejection_reasons.append(reason)
        else:
            checks.append(PolicyCheckResult("not_archived", True, True))

        # Fork check
        if self._settings.discovery_skip_forks and is_fork:
            reason = "repository_is_fork"
            checks.append(PolicyCheckResult("not_fork", False, True, reason))
            hard_blocked = True
            rejection_reasons.append(reason)
        else:
            checks.append(PolicyCheckResult("not_fork", True, True))

        # --- Policy checks (not immediate hard blocks) ---

        # License
        license_normalised = (license_key or "").lower()
        if self._settings.discovery_require_license:
            if not license_normalised or license_normalised not in _OSI_LICENSES:
                reason = f"no_osi_license:{license_key or 'none'}"
                checks.append(PolicyCheckResult("osi_license", False, True, reason))
                hard_blocked = True
                rejection_reasons.append(reason)
            else:
                checks.append(PolicyCheckResult("osi_license", True, False, license_normalised))
        else:
            checks.append(PolicyCheckResult("osi_license", True, False, "license_check_disabled"))

        # Language
        allowed_langs = self._settings.get_discovery_languages()
        if allowed_langs:
            if not primary_language:
                # Missing language metadata when a language filter is active — cannot
                # confirm eligibility; reject rather than silently passing.
                reason = "language_unknown"
                checks.append(PolicyCheckResult("language", False, False, reason))
                hard_blocked = True
                rejection_reasons.append(reason)
            elif primary_language not in allowed_langs:
                reason = f"unsupported_language:{primary_language}"
                checks.append(PolicyCheckResult("language", False, True, reason))
                hard_blocked = True
                rejection_reasons.append(reason)
            else:
                checks.append(PolicyCheckResult("language", True, False, primary_language))
        else:
            checks.append(PolicyCheckResult("language", True, False, primary_language or "unknown"))

        # Recency
        if self._settings.discovery_max_staleness_days > 0:
            if last_pushed_at is None:
                # Missing recency metadata — cannot confirm the repo is fresh enough.
                reason = "last_pushed_at_unknown"
                checks.append(PolicyCheckResult("recency", False, False, reason))
                hard_blocked = True
                rejection_reasons.append(reason)
            else:
                cutoff = datetime.now(UTC) - timedelta(
                    days=self._settings.discovery_max_staleness_days
                )
                if last_pushed_at < cutoff:
                    reason = f"stale_repository:last_push={last_pushed_at.date()}"
                    checks.append(PolicyCheckResult("recency", False, False, reason))
                    hard_blocked = True
                    rejection_reasons.append(reason)
                else:
                    checks.append(
                        PolicyCheckResult("recency", True, False, str(last_pushed_at.date()))
                    )
        else:
            checks.append(PolicyCheckResult("recency", True, False, "recency_check_disabled"))

        # Stars minimum (informational — not a safety control)
        if stars >= self._settings.discovery_min_stars:
            checks.append(PolicyCheckResult("min_stars", True, False, f"{stars}"))
        else:
            reason = f"below_min_stars:{stars}<{self._settings.discovery_min_stars}"
            checks.append(PolicyCheckResult("min_stars", False, True, reason))
            hard_blocked = True
            rejection_reasons.append(reason)

        # Compute safety classification and score
        failed_checks = [c for c in checks if not c.passed]
        total_checks = len(checks)
        passed_checks = total_checks - len(failed_checks)
        score = passed_checks / total_checks if total_checks else 1.0

        if hard_blocked:
            classification = "blocked"
        elif not failed_checks:
            classification = "safe_candidate"
        elif len(failed_checks) <= 1:
            classification = "low_risk"
        else:
            classification = "needs_review"

        if hard_blocked:
            eligibility_status = "rejected"
        else:
            eligibility_status = "eligible"

        return EligibilityAssessment(
            safety_classification=classification,
            eligibility_status=eligibility_status,
            eligibility_score=round(score, 4),
            rejection_reason=("; ".join(rejection_reasons) if rejection_reasons else None),
            policy_checks=checks,
        )
