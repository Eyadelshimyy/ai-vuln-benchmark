"""Preflight for browser sessions: what JobBot can actually reach right now.

Same discipline as the company registry: a session is ``ready`` only when an open page
proves it — a port that answers proves nothing. JobBot never attaches on its own either;
it reports the ``--cdp`` command and lets you decide.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger("jobbot.browser.sessions")

DEFAULT_PORTS: tuple[int, ...] = (9222, 9223, 9224)
LOCAL_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost")


class SessionStatus(StrEnum):
    READY = "ready"
    NEEDS_LOGIN = "needs_login"
    UNKNOWN = "unknown"
    PROFILE_BUSY = "profile_busy"
    WRONG_ACCOUNT = "wrong_account"
    NO_ENDPOINT = "no_endpoint"


def session_blocks_open(status: SessionStatus) -> bool:
    """A busy profile or somebody else's login must not be driven."""
    return status in {SessionStatus.PROFILE_BUSY, SessionStatus.WRONG_ACCOUNT}


def refuse_to_open(status: SessionStatus, evidence: str) -> str | None:
    """Evidence to show instead of launching. Never includes the other account."""
    if not session_blocks_open(status):
        return None
    return evidence


@dataclass(frozen=True)
class SiteSpec:
    """Where a site's session lives and how an open page betrays its state."""

    site: str
    hosts: tuple[str, ...]
    signed_in: tuple[str, ...]
    auth: tuple[str, ...]
    start_url: str
    cdp_profile: str
    next_command: str


SITES: tuple[SiteSpec, ...] = (
    SiteSpec(
        site="indeed",
        hosts=("indeed.com",),
        signed_in=("profile.indeed.com", "/myjobs", "/resume"),
        auth=("/auth", "login", "signin", "secure.indeed.com/account"),
        start_url="https://profile.indeed.com/resume",
        cdp_profile="indeed-cdp",
        next_command="jobbot jobs search … --cdp {cdp}",
    ),
    SiteSpec(
        site="linkedin",
        hosts=("linkedin.com",),
        signed_in=("/feed", "/in/", "/mynetwork", "/details/"),
        auth=("login", "authwall", "checkpoint"),
        start_url="https://www.linkedin.com/feed/",
        cdp_profile="linkedin-cdp",
        next_command="jobbot linkedin sync --section publications --apply --cdp {cdp}",
    ),
    SiteSpec(
        site="gmail",
        hosts=("mail.google.com",),
        signed_in=("/mail/",),
        auth=("signin", "accounts.google.com"),
        start_url="https://mail.google.com/",
        cdp_profile="gmail-cdp",
        next_command="jobbot application apply J0001 --apply --cdp {cdp}",
    ),
    SiteSpec(
        site="getonboard",
        hosts=("getonbrd.com",),
        signed_in=("/webpros",),
        auth=("/login", "/users/sign_in"),
        start_url="https://www.getonbrd.com/webpros/edit",
        cdp_profile="getonboard-cdp",
        next_command="jobbot cv propagate --targets getonboard --apply --cdp {cdp}",
    ),
)

KNOWN_SITES: tuple[str, ...] = tuple(spec.site for spec in SITES)


def site_spec(site: str) -> SiteSpec | None:
    """The one registry both `browser sessions` and `chrome-debug` read."""
    key = (site or "").strip().casefold()
    return next((spec for spec in SITES if spec.site == key), None)


@dataclass(frozen=True)
class AccountExpectation:
    """Identity the workspace profile owns. Empty fields are not evidence."""

    email: str | None = None
    linkedin: str | None = None


@dataclass(frozen=True)
class CdpEndpoint:
    """A debugging port that answered. ``foreign`` ports are busy and never read."""

    url: str
    port: int
    browser: str | None = None
    page_urls: tuple[str, ...] = ()
    profile_dir: Path | None = None
    foreign: bool = False


@dataclass(frozen=True)
class ChromeProcess:
    pid: int
    profile_dir: Path
    cdp_port: int | None = None


@dataclass(frozen=True)
class SessionState:
    """Preflight verdict for one site: never claimed without evidence."""

    site: str
    status: SessionStatus
    evidence: str
    cdp_url: str | None = None
    hint: str | None = None
    holders: tuple[int, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status is SessionStatus.READY

    @property
    def profile_busy(self) -> bool:
        return self.status is SessionStatus.PROFILE_BUSY


class ProfileBusyError(RuntimeError):
    """Another Chrome already holds the persistent profile JobBot wants to launch."""


JsonFetcher = Callable[[str], object]


def fetch_local_json(url: str, *, timeout: float = 0.6) -> object:
    """GET a localhost CDP endpoint — localhost only, so this never leaves the machine."""
    if not any(url.startswith(f"http://{host}:") for host in LOCAL_HOSTS):
        msg = f"refusing non-local CDP url: {url}"
        raise ValueError(msg)
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - guarded above
        payload = response.read().decode("utf-8", "replace")
    parsed: object = json.loads(payload)
    return parsed


def discover_endpoints(
    ports: Sequence[int] = DEFAULT_PORTS,
    *,
    fetch: JsonFetcher = fetch_local_json,
    processes: Sequence[ChromeProcess] | None = None,
    own_root: Path | None = None,
) -> list[CdpEndpoint]:
    """Probe localhost debugging ports; silent ports are simply absent.

    With ``own_root`` (this workspace's browser data), a port whose Chrome runs a
    profile outside it is reported as foreign without fetching its tabs.
    """
    known = {
        proc.cdp_port: proc
        for proc in (processes if processes is not None else list_chrome_processes())
        if proc.cdp_port is not None
    }
    endpoints: list[CdpEndpoint] = []
    for port in ports:
        base = f"http://127.0.0.1:{port}"
        holder = known.get(port)
        if (
            own_root is not None
            and holder is not None
            and not _is_within(holder.profile_dir, own_root)
        ):
            endpoints.append(CdpEndpoint(url=base, port=port, foreign=True))
            continue
        try:
            version = fetch(f"{base}/json/version")
            pages = fetch(f"{base}/json/list")
        except (URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            logger.debug("no CDP endpoint on port %s: %s", port, exc)
            continue
        endpoints.append(
            CdpEndpoint(
                url=base,
                port=port,
                browser=_browser_name(version),
                page_urls=_page_urls(pages),
                profile_dir=holder.profile_dir if holder else None,
            )
        )
    return endpoints


def list_chrome_processes(runner: Callable[[], str] | None = None) -> list[ChromeProcess]:
    """Main browser processes only: helpers carry --type= and share the profile dir."""
    output = (runner or _ps_output)()
    processes: list[ChromeProcess] = []
    for line in output.splitlines():
        entry = line.strip()
        if "--user-data-dir=" not in entry or "--type=" in entry:
            continue
        pid_text, _, command = entry.partition(" ")
        directory = _USER_DATA_DIR.search(command)
        if not pid_text.isdigit() or directory is None:
            continue
        port = _CDP_PORT.search(command)
        processes.append(
            ChromeProcess(
                pid=int(pid_text),
                profile_dir=Path(directory.group("path").rstrip()),
                cdp_port=int(port.group("port")) if port else None,
            )
        )
    return processes


def profile_holders(
    profile_dir: Path,
    processes: Sequence[ChromeProcess] | None = None,
) -> tuple[int, ...]:
    """PIDs of Chrome instances holding this persistent profile (Chrome allows one)."""
    target = _resolved(profile_dir)
    found = processes if processes is not None else list_chrome_processes()
    return tuple(proc.pid for proc in found if _resolved(proc.profile_dir) == target)


def ensure_profile_free(
    profile_dir: Path,
    *,
    processes: Sequence[ChromeProcess] | None = None,
) -> None:
    """Fail fast instead of letting Chrome hand off to the running instance and die."""
    holders = profile_holders(profile_dir, processes)
    if not holders:
        return
    pids = ", ".join(str(pid) for pid in holders)
    msg = (
        f"{profile_dir} is already open in Chrome (pid {pids}); "
        "close that window or attach to it with --cdp"
    )
    raise ProfileBusyError(msg)


def inspect_sessions(
    root: Path,
    *,
    sites: Sequence[str] | None = None,
    ports: Sequence[int] = DEFAULT_PORTS,
    fetch: JsonFetcher = fetch_local_json,
    processes: Sequence[ChromeProcess] | None = None,
    account: AccountExpectation | None = None,
    page_text: dict[str, str] | None = None,
) -> list[SessionState]:
    """Session state per site, from open pages plus the persistent-profile locks."""
    from jobbot.config import load_config

    procs = list(processes if processes is not None else list_chrome_processes())
    browser_root = load_config(root=root).browser_data_dir
    endpoints = discover_endpoints(ports, fetch=fetch, processes=procs, own_root=browser_root)
    wanted = {site.casefold() for site in sites} if sites else None
    states: list[SessionState] = []
    for spec in SITES:
        if wanted is not None and spec.site not in wanted:
            continue
        holders = profile_holders(browser_root / spec.site, procs)
        states.append(
            _state_for(
                spec,
                endpoints,
                holders,
                ports,
                processes=procs,
                account=account,
                page_text=page_text,
            )
        )
    return states


def session_for(states: Sequence[SessionState], site: str) -> SessionState | None:
    return next((state for state in states if state.site == site), None)


def judge_account(site: str, html: str, expected: AccountExpectation) -> SessionStatus:
    """Match a signed-in page to this workspace. No identity on the page is unknown."""
    if site == "gmail":
        found = _EMAIL_RE.findall(html or "")
        if not found or not expected.email:
            return SessionStatus.UNKNOWN
        if any(item.casefold() == expected.email.casefold() for item in found):
            return SessionStatus.READY
        return SessionStatus.WRONG_ACCOUNT
    if site == "linkedin":
        found = _LINKEDIN_RE.findall(html or "")
        want = _linkedin_handle(expected.linkedin)
        if not found or not want:
            return SessionStatus.UNKNOWN
        if any(item.casefold() == want.casefold() for item in found):
            return SessionStatus.READY
        return SessionStatus.WRONG_ACCOUNT
    return SessionStatus.UNKNOWN


def _state_for(
    spec: SiteSpec,
    endpoints: Sequence[CdpEndpoint],
    holders: tuple[int, ...],
    ports: Sequence[int],
    *,
    processes: Sequence[ChromeProcess] = (),
    account: AccountExpectation | None = None,
    page_text: dict[str, str] | None = None,
) -> SessionState:
    launch = _debug_command(spec, endpoints, ports, processes)
    best: tuple[int, SessionState] | None = None
    weak: tuple[CdpEndpoint, str] | None = None
    for endpoint in endpoints:
        if endpoint.foreign:
            continue
        for url in endpoint.page_urls:
            if not _matches_host(url, spec):
                continue
            short = _short_url(url)
            found: SessionState | None = None
            if any(marker in url.casefold() for marker in spec.auth):
                found = SessionState(
                    site=spec.site,
                    status=SessionStatus.NEEDS_LOGIN,
                    evidence=f"open tab sits on the login wall: {short}",
                    cdp_url=endpoint.url,
                    hint=f"sign in at {endpoint.url}, then re-run with --cdp {endpoint.url}",
                    holders=holders,
                )
            elif any(marker in url.casefold() for marker in spec.signed_in):
                status, evidence = _signed_in_verdict(
                    spec, url, short, account=account, page_text=page_text
                )
                found = SessionState(
                    site=spec.site,
                    status=status,
                    evidence=evidence,
                    cdp_url=endpoint.url,
                    hint=f"--cdp {endpoint.url}",
                    holders=holders,
                )
            else:
                weak = weak or (endpoint, short)
                continue
            rank = _evidence_rank(found)
            if best is None or rank > best[0]:
                best = (rank, found)
    if best is not None:
        return best[1]
    if weak is not None:
        endpoint, short = weak
        return SessionState(
            site=spec.site,
            status=SessionStatus.UNKNOWN,
            evidence=f"tab on {short} proves nothing about the session",
            cdp_url=endpoint.url,
            hint=f"open {spec.start_url} in that Chrome to confirm the session",
            holders=holders,
        )
    if holders:
        pids = ", ".join(str(pid) for pid in holders)
        return SessionState(
            site=spec.site,
            status=SessionStatus.PROFILE_BUSY,
            evidence=f"browser-data/{spec.site} held by Chrome pid {pids}",
            hint="close that Chrome window, or point --cdp at it",
            holders=holders,
        )
    own = [endpoint for endpoint in endpoints if not endpoint.foreign]
    busy = _foreign_note(endpoints)
    if own:
        return SessionState(
            site=spec.site,
            status=SessionStatus.UNKNOWN,
            evidence=f"{len(own)} CDP endpoint(s) open, none with a {spec.site} tab{busy}",
            hint=f"open {spec.start_url} there, or launch: {launch}",
            holders=holders,
        )
    if busy:
        return SessionState(
            site=spec.site,
            status=SessionStatus.UNKNOWN,
            evidence=busy.removeprefix("; "),
            hint=launch,
            holders=holders,
        )
    return SessionState(
        site=spec.site,
        status=SessionStatus.NO_ENDPOINT,
        evidence="no debugging port answered on 127.0.0.1",
        hint=launch,
        holders=holders,
    )


_EVIDENCE_RANK: dict[SessionStatus, int] = {
    SessionStatus.READY: 4,
    SessionStatus.WRONG_ACCOUNT: 3,
    SessionStatus.UNKNOWN: 2,
    SessionStatus.NEEDS_LOGIN: 1,
}


def _evidence_rank(state: SessionState) -> int:
    """A signed-in tab on any endpoint outweighs a login wall on another one."""
    return _EVIDENCE_RANK.get(state.status, 0)


def free_debug_port(
    ports: Sequence[int],
    endpoints: Sequence[CdpEndpoint],
    processes: Sequence[ChromeProcess] = (),
) -> int:
    """First debugging port no Chrome answers on or holds; past the list if all are."""
    taken = {endpoint.port for endpoint in endpoints}
    taken.update(proc.cdp_port for proc in processes if proc.cdp_port is not None)
    candidates = list(ports) or list(DEFAULT_PORTS)
    for candidate in candidates:
        if candidate not in taken:
            return candidate
    port = max(candidates) + 1
    while port in taken:
        port += 1
    return port


def _debug_command(
    spec: SiteSpec,
    endpoints: Sequence[CdpEndpoint],
    ports: Sequence[int],
    processes: Sequence[ChromeProcess] = (),
) -> str:
    port = free_debug_port(ports, endpoints, processes)
    return f"jobbot browser chrome-debug --site {spec.site} --port {port}"


def _foreign_note(endpoints: Sequence[CdpEndpoint]) -> str:
    ports = ", ".join(str(endpoint.port) for endpoint in endpoints if endpoint.foreign)
    if not ports:
        return ""
    return f"; port {ports} held by a Chrome profile outside this workspace (not read)"


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/([A-Za-z0-9_-]+)", re.IGNORECASE)


def _linkedin_handle(value: str | None) -> str | None:
    if not value:
        return None
    match = _LINKEDIN_RE.search(value)
    if match:
        return match.group(1)
    tail = value.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def _signed_in_verdict(
    spec: SiteSpec,
    url: str,
    short: str,
    *,
    account: AccountExpectation | None,
    page_text: dict[str, str] | None,
) -> tuple[SessionStatus, str]:
    if account is None or page_text is None:
        return SessionStatus.READY, f"signed-in page open: {short}"
    status = judge_account(spec.site, page_text.get(url, ""), account)
    if status is SessionStatus.WRONG_ACCOUNT:
        return status, "signed-in account does not match this workspace"
    if status is SessionStatus.READY:
        return status, f"signed-in page matches this workspace: {short}"
    return SessionStatus.UNKNOWN, "signed-in page has no identity for this workspace"


def _matches_host(url: str, spec: SiteSpec) -> bool:
    lowered = url.casefold()
    return any(host in lowered for host in spec.hosts)


def _short_url(url: str) -> str:
    without_scheme = re.sub(r"^https?://", "", url)
    return without_scheme.split("?", 1)[0][:80]


def _browser_name(version: object) -> str | None:
    if isinstance(version, dict):
        name = version.get("Browser")
        if isinstance(name, str):
            return name
    return None


def _page_urls(pages: object) -> tuple[str, ...]:
    if not isinstance(pages, list):
        return ()
    urls: list[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        url = page.get("url")
        if isinstance(url, str) and url.startswith("http"):
            urls.append(url)
    return tuple(urls)


def _resolved(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:  # pragma: no cover - unreadable path
        return path


def _is_within(path: Path, root: Path) -> bool:
    return _resolved(path).is_relative_to(_resolved(root))


def _ps_output() -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["ps", "-Ao", "pid=,command="],  # noqa: S607 - ps resolved from PATH
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("ps failed: %s", exc)
        return ""
    return completed.stdout


# Path may contain spaces; take until the next flag (` --…`) or end of line.
_USER_DATA_DIR = re.compile(r"--user-data-dir=(?P<path>.+?)(?=\s+--|\s*$)")
_CDP_PORT = re.compile(r"--remote-debugging-port=(?P<port>\d+)")
