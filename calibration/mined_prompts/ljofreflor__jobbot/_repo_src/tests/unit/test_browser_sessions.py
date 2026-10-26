"""Session preflight: evidence-based, offline, and loud about a busy profile."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pytest

from jobbot.browser.sessions import (
    ChromeProcess,
    ProfileBusyError,
    SessionStatus,
    discover_endpoints,
    ensure_profile_free,
    fetch_local_json,
    inspect_sessions,
    list_chrome_processes,
    profile_holders,
    session_for,
)

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HELPER = "/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Helper"
PS_OUTPUT = "\n".join(
    [
        f"  30237 {CHROME} --user-data-dir=/repo/browser-data/indeed --remote-debugging-pipe",
        f"  30253 {HELPER} --type=renderer --user-data-dir=/repo/browser-data/indeed",
        f"  35028 {CHROME} --remote-debugging-port=9222"
        " --user-data-dir=/repo/browser-data/indeed-cdp",
    ]
)


class FakeCdp:
    """Answers /json/version and /json/list for the ports that are 'listening'."""

    def __init__(self, pages_by_port: dict[int, list[str]]) -> None:
        self.pages_by_port = pages_by_port
        self.requested: list[str] = []

    def __call__(self, url: str) -> object:
        self.requested.append(url)
        port = int(url.split(":")[2].split("/")[0])
        if port not in self.pages_by_port:
            raise URLError("connection refused")
        if url.endswith("/json/version"):
            return {"Browser": "Chrome/152.0.7977.83"}
        return [{"url": page} for page in self.pages_by_port[port]]


def _procs() -> list[ChromeProcess]:
    return list_chrome_processes(lambda: PS_OUTPUT)


def test_process_list_keeps_main_chrome_and_skips_helpers() -> None:
    processes = _procs()
    assert [(p.pid, p.cdp_port) for p in processes] == [(30237, None), (35028, 9222)]
    assert processes[0].profile_dir == Path("/repo/browser-data/indeed")


def test_process_list_keeps_user_data_dir_when_path_has_spaces() -> None:
    spaced = "/repo/nombre apellido/.local/browser-data/linkedin-cdp"
    ps = (
        f"  35028 {CHROME} --remote-debugging-port=9222"
        f" --user-data-dir={spaced} --remote-debugging-pipe"
    )
    processes = list_chrome_processes(lambda: ps)
    assert len(processes) == 1
    assert processes[0].profile_dir == Path(spaced)
    assert processes[0].cdp_port == 9222


def test_discovery_skips_silent_ports() -> None:
    fetch = FakeCdp({9223: ["https://www.linkedin.com/feed/"]})
    endpoints = discover_endpoints((9222, 9223), fetch=fetch, processes=[])
    assert [endpoint.port for endpoint in endpoints] == [9223]
    assert endpoints[0].browser == "Chrome/152.0.7977.83"


def test_discovery_links_the_endpoint_to_the_profile_that_opened_it() -> None:
    fetch = FakeCdp({9222: ["https://profile.indeed.com/resume"]})
    endpoints = discover_endpoints((9222,), fetch=fetch, processes=_procs())
    assert endpoints[0].profile_dir == Path("/repo/browser-data/indeed-cdp")


def test_signed_in_page_is_the_only_thing_that_makes_a_session_ready(tmp_path: Path) -> None:
    fetch = FakeCdp({9222: ["https://profile.indeed.com/resume"]})
    states = inspect_sessions(tmp_path, sites=["indeed"], ports=(9222,), fetch=fetch, processes=[])
    state = session_for(states, "indeed")
    assert state is not None and state.status is SessionStatus.READY
    assert state.cdp_url == "http://127.0.0.1:9222"
    assert state.hint == "--cdp http://127.0.0.1:9222"
    assert "profile.indeed.com/resume" in state.evidence


def test_home_page_alone_stays_unknown(tmp_path: Path) -> None:
    fetch = FakeCdp({9222: ["https://cl.indeed.com/"]})
    states = inspect_sessions(tmp_path, sites=["indeed"], ports=(9222,), fetch=fetch, processes=[])
    state = session_for(states, "indeed")
    assert state is not None and state.status is SessionStatus.UNKNOWN
    assert state.cdp_url == "http://127.0.0.1:9222"


def test_login_wall_reports_needs_login(tmp_path: Path) -> None:
    fetch = FakeCdp({9223: ["https://www.linkedin.com/checkpoint/lg/login"]})
    states = inspect_sessions(
        tmp_path, sites=["linkedin"], ports=(9223,), fetch=fetch, processes=[]
    )
    state = session_for(states, "linkedin")
    assert state is not None and state.status is SessionStatus.NEEDS_LOGIN
    assert state.hint is not None and "--cdp http://127.0.0.1:9223" in state.hint


@pytest.mark.parametrize(
    ("login_port", "signed_in_port"),
    [(9223, 9224), (9224, 9223)],
)
def test_signed_in_page_on_any_endpoint_beats_a_login_wall_on_another(
    tmp_path: Path, login_port: int, signed_in_port: int
) -> None:
    """Regression: a login tab on one port hid the signed-in tab on another."""
    fetch = FakeCdp(
        {
            login_port: ["https://www.getonbrd.com/login"],
            signed_in_port: ["https://www.getonbrd.com/webpros/edit"],
        }
    )
    states = inspect_sessions(
        tmp_path, sites=["getonboard"], ports=(9223, 9224), fetch=fetch, processes=[]
    )
    state = session_for(states, "getonboard")
    assert state is not None and state.status is SessionStatus.READY
    assert state.cdp_url == f"http://127.0.0.1:{signed_in_port}"
    assert state.hint == f"--cdp http://127.0.0.1:{signed_in_port}"
    assert "getonbrd.com/webpros/edit" in state.evidence


def test_login_wall_beats_a_tab_that_proves_nothing(tmp_path: Path) -> None:
    fetch = FakeCdp(
        {
            9222: ["https://www.getonbrd.com/empleos"],
            9223: ["https://www.getonbrd.com/users/sign_in"],
        }
    )
    states = inspect_sessions(
        tmp_path, sites=["getonboard"], ports=(9222, 9223), fetch=fetch, processes=[]
    )
    state = session_for(states, "getonboard")
    assert state is not None and state.status is SessionStatus.NEEDS_LOGIN
    assert state.cdp_url == "http://127.0.0.1:9223"


def test_matching_account_on_one_endpoint_beats_another_account_elsewhere(
    tmp_path: Path,
) -> None:
    from jobbot.browser.sessions import AccountExpectation

    other = "https://mail.google.com/mail/u/1/"
    mine = "https://mail.google.com/mail/u/0/"
    fetch = FakeCdp({9222: [other], 9223: [mine]})
    states = inspect_sessions(
        tmp_path,
        sites=["gmail"],
        ports=(9222, 9223),
        fetch=fetch,
        processes=[],
        account=AccountExpectation(email="ana.ejemplo@example.com"),
        page_text={
            other: "<div>otra.persona@example.com</div>",
            mine: "<div>ana.ejemplo@example.com</div>",
        },
    )
    state = session_for(states, "gmail")
    assert state is not None and state.status is SessionStatus.READY
    assert state.cdp_url == "http://127.0.0.1:9223"


def test_no_endpoint_suggests_chrome_debug(tmp_path: Path) -> None:
    fetch = FakeCdp({})
    states = inspect_sessions(tmp_path, sites=["indeed"], ports=(9222,), fetch=fetch, processes=[])
    state = session_for(states, "indeed")
    assert state is not None and state.status is SessionStatus.NO_ENDPOINT
    assert state.hint == "jobbot browser chrome-debug --site indeed --port 9222"


def _foreign_chrome(port: int, pid: int = 4242) -> ChromeProcess:
    """A Chrome somebody else opened, with its own profile outside this workspace."""
    return ChromeProcess(pid=pid, profile_dir=Path("/elsewhere/other-profile"), cdp_port=port)


def test_debug_hint_never_suggests_a_port_every_probe_found_taken(tmp_path: Path) -> None:
    """Regression: with every probed port answering, the hint fell back to 9222."""
    fetch = FakeCdp(
        {
            9222: ["https://mail.google.com/mail/u/0/"],
            9223: ["https://www.linkedin.com/feed/"],
            9224: ["https://mail.google.com/mail/u/0/"],
        }
    )
    states = inspect_sessions(
        tmp_path,
        sites=["indeed"],
        ports=(9222, 9223, 9224),
        fetch=fetch,
        processes=[_foreign_chrome(9222)],
    )
    state = session_for(states, "indeed")
    assert state is not None and state.hint is not None
    for taken in (9222, 9223, 9224):
        assert f"--port {taken}" not in state.hint
    assert "--port 9225" in state.hint


def test_debug_hint_skips_a_port_held_by_a_chrome_that_does_not_answer(tmp_path: Path) -> None:
    """A Chrome holding the port per the process table is busy even if CDP is silent."""
    states = inspect_sessions(
        tmp_path,
        sites=["indeed"],
        ports=(9222, 9223),
        fetch=FakeCdp({}),
        processes=[_foreign_chrome(9222)],
    )
    state = session_for(states, "indeed")
    assert state is not None and state.hint is not None
    assert "--port 9222" not in state.hint
    assert state.hint == "jobbot browser chrome-debug --site indeed --port 9223"


def test_foreign_chrome_tabs_are_never_read(tmp_path: Path) -> None:
    """A port held by a profile outside this workspace is busy, not a session."""
    fetch = FakeCdp({9222: ["https://profile.indeed.com/resume"]})
    states = inspect_sessions(
        tmp_path,
        sites=["indeed"],
        ports=(9222,),
        fetch=fetch,
        processes=[_foreign_chrome(9222)],
    )
    state = session_for(states, "indeed")
    assert state is not None
    assert state.status is not SessionStatus.READY
    assert state.cdp_url is None
    assert "indeed.com" not in state.evidence
    assert "9222" in state.evidence and "not read" in state.evidence
    assert not [url for url in fetch.requested if ":9222/" in url]
    assert state.hint is not None and "--port 9222" not in state.hint


def test_foreign_signed_in_tab_never_outranks_a_login_wall_on_an_own_port(
    tmp_path: Path,
) -> None:
    fetch = FakeCdp(
        {
            9222: ["https://www.getonbrd.com/webpros/edit"],
            9223: ["https://www.getonbrd.com/login"],
        }
    )
    states = inspect_sessions(
        tmp_path,
        sites=["getonboard"],
        ports=(9222, 9223),
        fetch=fetch,
        processes=[_foreign_chrome(9222)],
    )
    state = session_for(states, "getonboard")
    assert state is not None and state.status is SessionStatus.NEEDS_LOGIN
    assert state.cdp_url == "http://127.0.0.1:9223"
    assert not [url for url in fetch.requested if ":9222/" in url]


def test_foreign_endpoint_is_skipped_even_when_it_carries_tabs() -> None:
    from jobbot.browser.sessions import SITES, CdpEndpoint, _state_for

    spec = next(site for site in SITES if site.site == "getonboard")
    foreign = CdpEndpoint(
        url="http://127.0.0.1:9222",
        port=9222,
        page_urls=("https://www.getonbrd.com/webpros/edit",),
        foreign=True,
    )
    state = _state_for(spec, [foreign], (), (9222,))
    assert state.status is SessionStatus.UNKNOWN
    assert state.cdp_url is None
    assert "not read" in state.evidence


def test_jobbot_profile_endpoint_is_still_read(tmp_path: Path) -> None:
    own = ChromeProcess(pid=7, profile_dir=tmp_path / "browser-data" / "indeed-cdp", cdp_port=9222)
    fetch = FakeCdp({9222: ["https://profile.indeed.com/resume"]})
    states = inspect_sessions(
        tmp_path, sites=["indeed"], ports=(9222,), fetch=fetch, processes=[own]
    )
    state = session_for(states, "indeed")
    assert state is not None and state.status is SessionStatus.READY
    assert state.cdp_url == "http://127.0.0.1:9222"


def test_discovery_marks_foreign_endpoint_without_its_contents(tmp_path: Path) -> None:
    fetch = FakeCdp({9222: ["https://profile.indeed.com/resume"]})
    endpoints = discover_endpoints(
        (9222,),
        fetch=fetch,
        processes=[_foreign_chrome(9222)],
        own_root=tmp_path / "browser-data",
    )
    assert len(endpoints) == 1
    assert endpoints[0].foreign is True
    assert endpoints[0].page_urls == ()
    assert endpoints[0].profile_dir is None
    assert fetch.requested == []


def test_discovery_owns_workspace_profile_when_path_has_spaces(tmp_path: Path) -> None:
    """#193: \\S+ truncated at the first space and misclassified the profile as foreign."""
    own_root = tmp_path / "nombre apellido" / ".local" / "browser-data"
    profile = own_root / "linkedin-cdp"
    profile.mkdir(parents=True)
    ps = (
        f"  35028 {CHROME} --remote-debugging-port=9222"
        f" --user-data-dir={profile} --remote-debugging-pipe"
    )
    processes = list_chrome_processes(lambda: ps)
    fetch = FakeCdp({9222: ["https://www.linkedin.com/feed/"]})
    endpoints = discover_endpoints(
        (9222,),
        fetch=fetch,
        processes=processes,
        own_root=own_root,
    )
    assert len(endpoints) == 1
    assert endpoints[0].foreign is False
    assert endpoints[0].profile_dir == profile
    assert fetch.requested  # own profile: tabs are read


def test_discovery_marks_foreign_when_spaced_path_outside_own_root(tmp_path: Path) -> None:
    own_root = tmp_path / "browser-data"
    own_root.mkdir()
    foreign = tmp_path / "other person" / "browser-data" / "linkedin-cdp"
    foreign.mkdir(parents=True)
    ps = (
        f"  35028 {CHROME} --remote-debugging-port=9222"
        f" --user-data-dir={foreign}"
    )
    processes = list_chrome_processes(lambda: ps)
    fetch = FakeCdp({9222: ["https://www.linkedin.com/feed/"]})
    endpoints = discover_endpoints(
        (9222,),
        fetch=fetch,
        processes=processes,
        own_root=own_root,
    )
    assert len(endpoints) == 1
    assert endpoints[0].foreign is True
    assert endpoints[0].page_urls == ()
    assert fetch.requested == []


def test_free_debug_port_skips_answering_and_held_ports() -> None:
    from jobbot.browser.sessions import CdpEndpoint, free_debug_port

    endpoints = [CdpEndpoint(url="http://127.0.0.1:9223", port=9223)]
    processes = [_foreign_chrome(9222), _foreign_chrome(9225, pid=5)]
    assert free_debug_port((9222, 9223, 9224), endpoints, processes) == 9224
    assert free_debug_port((9222, 9223), endpoints, processes) == 9224
    assert free_debug_port((9222,), [], [_foreign_chrome(9222)]) == 9223


def test_busy_persistent_profile_is_reported_instead_of_a_launch_crash(tmp_path: Path) -> None:
    """Regression F0002: a leftover Chrome on browser-data/indeed killed the launch."""
    profile_dir = tmp_path / "browser-data" / "indeed"
    processes = [ChromeProcess(pid=30237, profile_dir=profile_dir)]
    states = inspect_sessions(
        tmp_path, sites=["indeed"], ports=(9222,), fetch=FakeCdp({}), processes=processes
    )
    state = session_for(states, "indeed")
    assert state is not None and state.status is SessionStatus.PROFILE_BUSY
    assert state.holders == (30237,)
    assert "30237" in state.evidence


def test_ensure_profile_free_raises_with_the_pid(tmp_path: Path) -> None:
    profile_dir = tmp_path / "browser-data" / "indeed"
    processes = [ChromeProcess(pid=30237, profile_dir=profile_dir)]
    ensure_profile_free(tmp_path / "browser-data" / "linkedin", processes=processes)
    with pytest.raises(ProfileBusyError, match="30237"):
        ensure_profile_free(profile_dir, processes=processes)


def test_profile_holders_matches_through_relative_paths(tmp_path: Path) -> None:
    profile_dir = tmp_path / "browser-data" / "indeed"
    profile_dir.mkdir(parents=True)
    processes = [ChromeProcess(pid=1, profile_dir=profile_dir / "." / "")]
    assert profile_holders(profile_dir, processes) == (1,)


def test_fetch_refuses_remote_urls() -> None:
    with pytest.raises(ValueError, match="non-local"):
        fetch_local_json("http://example.com:9222/json/list")


def test_bad_json_is_treated_as_no_endpoint(tmp_path: Path) -> None:
    def broken(_url: str) -> object:
        raise json.JSONDecodeError("boom", "", 0)

    assert discover_endpoints((9222,), fetch=broken, processes=[]) == []


def test_cli_reports_status_and_the_command_to_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import jobbot.browser.sessions as sessions
    from jobbot.cli import run_cli
    from jobbot.exit_codes import SUCCESS

    monkeypatch.chdir(tmp_path)
    fetch = FakeCdp({9222: ["https://profile.indeed.com/resume"]})
    monkeypatch.setattr(
        sessions,
        "inspect_sessions",
        lambda root, **kwargs: inspect_sessions(root, ports=(9222,), fetch=fetch, processes=[]),
    )

    assert run_cli(["browser", "sessions"], standalone_mode=False) == SUCCESS
    out = capsys.readouterr().out
    assert "ready" in out
    assert "indeed: --cdp http://127.0.0.1:9222" in out
    # An endpoint is open but proves nothing about the other sites.
    assert "unknown" in out


def test_other_accounts_session_is_wrong_account_and_blocks_open(tmp_path: Path) -> None:
    from jobbot.browser.sessions import AccountExpectation, refuse_to_open

    url = "https://mail.google.com/mail/u/0/"
    fetch = FakeCdp({9222: [url]})
    html = "<div>ana.real@example.com</div><div>otra.persona@example.com</div>"
    states = inspect_sessions(
        tmp_path,
        sites=["gmail"],
        ports=(9222,),
        fetch=fetch,
        processes=[],
        account=AccountExpectation(email="ana.ejemplo@example.com"),
        page_text={url: html},
    )
    state = session_for(states, "gmail")
    assert state is not None
    assert state.status is SessionStatus.WRONG_ACCOUNT
    reason = refuse_to_open(state.status, state.evidence)
    assert reason
    assert "otra.persona@example.com" not in reason
    assert "ana.real@example.com" not in reason


def test_matching_account_stays_ready(tmp_path: Path) -> None:
    from jobbot.browser.sessions import AccountExpectation

    url = "https://www.linkedin.com/in/ana-ejemplo/"
    fetch = FakeCdp({9222: [url]})
    states = inspect_sessions(
        tmp_path,
        sites=["linkedin"],
        ports=(9222,),
        fetch=fetch,
        processes=[],
        account=AccountExpectation(linkedin="https://www.linkedin.com/in/ana-ejemplo"),
        page_text={url: "profile https://www.linkedin.com/in/ana-ejemplo"},
    )
    state = session_for(states, "linkedin")
    assert state is not None and state.status is SessionStatus.READY


def test_signed_in_page_without_identity_is_unknown(tmp_path: Path) -> None:
    from jobbot.browser.sessions import AccountExpectation

    url = "https://mail.google.com/mail/u/0/"
    fetch = FakeCdp({9222: [url]})
    states = inspect_sessions(
        tmp_path,
        sites=["gmail"],
        ports=(9222,),
        fetch=fetch,
        processes=[],
        account=AccountExpectation(email="ana.ejemplo@example.com"),
        page_text={url: "<div>inbox</div>"},
    )
    state = session_for(states, "gmail")
    assert state is not None and state.status is SessionStatus.UNKNOWN
