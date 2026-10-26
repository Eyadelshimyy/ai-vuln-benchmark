"""Say, before a portal opens, that applying there needs a signed-in session.

JobBot never signs in for you, so on a platform that keeps a candidate profile the
only honest thing it can do is tell you. A session counts only with evidence from an
open signed-in page; a portal whose session cannot be checked is treated as not
signed in.
"""

from __future__ import annotations

from collections.abc import Sequence

from jobbot.browser.sessions import SessionState, SessionStatus, session_for, site_spec
from jobbot.companies.signup import AccountNeed, account_need
from jobbot.portals.detect import AtsKind

_PORTAL_NAMES: dict[AtsKind, str] = {
    AtsKind.TORRE: "Torre",
    AtsKind.GETONBOARD: "Get on Board",
    AtsKind.INDEED: "Indeed",
    AtsKind.LINKEDIN: "LinkedIn",
    AtsKind.WORKDAY: "Workday",
    AtsKind.SUCCESSFACTORS: "SuccessFactors",
    AtsKind.ORACLE: "Oracle",
}

_MUST_SIGN_IN = "no puedes postular si no inicias sesión"


def session_site(ats: AtsKind) -> str | None:
    """The `browser sessions` site that can evidence this portal, if any."""
    spec = site_spec(ats.value)
    return spec.site if spec is not None else None


def login_warning(ats: AtsKind, states: Sequence[SessionState]) -> str | None:
    """A warning when the portal needs an account and no session is evidenced."""
    if account_need(ats) is not AccountNeed.NEEDED:
        return None
    portal = _PORTAL_NAMES.get(ats, ats.value)
    site = session_site(ats)
    state = session_for(states, site) if site is not None else None
    if state is not None and state.ready:
        return None
    if state is None:
        return (
            f"{portal} exige una cuenta y JobBot no puede comprobar tu sesión ahí: "
            f"{_MUST_SIGN_IN} en {portal}."
        )
    if state.status is SessionStatus.NEEDS_LOGIN:
        return f"No iniciaste sesión en {portal} ({state.evidence}): {_MUST_SIGN_IN}."
    hint = f" {state.hint}" if state.hint else ""
    return (
        f"{portal} exige una cuenta y no hay evidencia de sesión ({state.evidence}): "
        f"{_MUST_SIGN_IN}.{hint}"
    )
