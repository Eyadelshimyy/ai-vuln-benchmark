"""Identity providers offered on a login or signup page.

A portal that lets you continue with Google or LinkedIn is still HITL: JobBot
names the button so you can use an account you already have. It never starts
OAuth, never clicks the provider, and never treats a LinkedIn *profile URL*
field or a footer "Follow us" link as a sign-in option.
"""

from __future__ import annotations

import re
from enum import StrEnum

from bs4 import BeautifulSoup, Tag


# Order is display order, not preference.
class SsoProvider(StrEnum):
    """A third-party identity host, not an employer and not a job board."""

    GOOGLE = "google"
    LINKEDIN = "linkedin"
    MICROSOFT = "microsoft"
    APPLE = "apple"
    FACEBOOK = "facebook"
    GITHUB = "github"


_OAUTH_HOST_MARKERS: tuple[tuple[SsoProvider, tuple[str, ...]], ...] = (
    (SsoProvider.GOOGLE, ("accounts.google.com", "google.com/o/oauth2")),
    (SsoProvider.LINKEDIN, ("linkedin.com/oauth", "linkedin.com/uas/oauth")),
    (SsoProvider.MICROSOFT, ("login.microsoftonline.com", "login.live.com")),
    (SsoProvider.APPLE, ("appleid.apple.com",)),
    (SsoProvider.FACEBOOK, ("facebook.com/v", "facebook.com/dialog/oauth")),
    (SsoProvider.GITHUB, ("github.com/login/oauth",)),
)

# Visible copy that means "use this account to sign in", ES/EN.
_SIGNIN_PHRASE = (
    r"(?:sign[\s-]?in|log[\s-]?in|login|continue|sign[\s-]?up|"
    r"iniciar\s+sesi[oó]n|continuar|entrar|registr(?:arse|o)|"
    r"crear\s+cuenta)"
)

_PHRASE_BY_PROVIDER: tuple[tuple[SsoProvider, tuple[str, ...]], ...] = (
    (SsoProvider.GOOGLE, ("google", "gmail")),
    (SsoProvider.LINKEDIN, ("linkedin",)),
    (SsoProvider.MICROSOFT, ("microsoft", "azure", "office 365", "hotmail")),
    (SsoProvider.APPLE, ("apple", "apple id", "appleid")),
    (SsoProvider.FACEBOOK, ("facebook",)),
    (SsoProvider.GITHUB, ("github",)),
)

_CONTROL_TAGS = frozenset({"a", "button"})


def detect_sso_providers(html: str) -> tuple[SsoProvider, ...]:
    """Which identity providers the page offers as sign-in. Pure: no network."""
    soup = BeautifulSoup(html or "", "html.parser")
    found: list[SsoProvider] = []
    seen: set[SsoProvider] = set()
    for node in soup.find_all(_CONTROL_TAGS):
        if not isinstance(node, Tag):
            continue
        if _looks_like_footer_or_nav(node):
            continue
        provider = _provider_for_control(node)
        if provider is None or provider in seen:
            continue
        seen.add(provider)
        found.append(provider)
    return tuple(found)


def provider_label(provider: SsoProvider) -> str:
    """English label for the sheet: 'Sign in with Google'."""
    names = {
        SsoProvider.GOOGLE: "Google",
        SsoProvider.LINKEDIN: "LinkedIn",
        SsoProvider.MICROSOFT: "Microsoft",
        SsoProvider.APPLE: "Apple",
        SsoProvider.FACEBOOK: "Facebook",
        SsoProvider.GITHUB: "GitHub",
    }
    return f"Sign in with {names[provider]}"


def _provider_for_control(node: Tag) -> SsoProvider | None:
    href = str(node.get("href") or "").casefold()
    for provider, markers in _OAUTH_HOST_MARKERS:
        if any(marker in href for marker in markers):
            return provider
    blob = _control_blob(node)
    if not blob:
        return None
    for provider, names in _PHRASE_BY_PROVIDER:
        for name in names:
            # "Sign in with Google" / "Continuar con LinkedIn" / data-automation googleSignIn
            pattern = (
                rf"{_SIGNIN_PHRASE}.{{0,24}}{re.escape(name)}"
                rf"|{re.escape(name)}.{{0,16}}{_SIGNIN_PHRASE}"
            )
            if re.search(pattern, blob, re.IGNORECASE):
                return provider
            compact = blob.replace(" ", "")
            if f"{name}signin" in compact or f"signin{name}" in compact:
                return provider
    return None


def _control_blob(node: Tag) -> str:
    parts = [
        node.get_text(" ", strip=True),
        str(node.get("aria-label") or ""),
        str(node.get("title") or ""),
        str(node.get("id") or ""),
        str(node.get("class") or ""),
        str(node.get("data-automation-id") or ""),
        str(node.get("data-provider") or ""),
        str(node.get("data-testid") or ""),
        str(node.get("name") or ""),
    ]
    return " ".join(parts).casefold()


def _looks_like_footer_or_nav(node: Tag) -> bool:
    for parent in node.parents:
        if not isinstance(parent, Tag):
            continue
        name = (parent.name or "").lower()
        if name in {"footer", "nav", "header"}:
            return True
        role = str(parent.get("role") or "").casefold()
        if role in {"navigation", "contentinfo", "banner"}:
            return True
    return False
