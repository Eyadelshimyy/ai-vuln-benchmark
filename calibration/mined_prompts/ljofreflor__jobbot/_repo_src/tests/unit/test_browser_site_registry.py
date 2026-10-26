"""Regression: a hint JobBot prints must be a command JobBot accepts.

`browser sessions` suggested `chrome-debug --site getonboard`, and chrome-debug
rejected it because each command carried its own list of sites (failure F0005).
"""

from __future__ import annotations

import pytest

from jobbot.browser.sessions import SITES, site_spec
from jobbot.exit_codes import GENERIC_FAILURE, SUCCESS


@pytest.mark.parametrize("spec", SITES, ids=[spec.site for spec in SITES])
def test_every_known_site_can_be_launched(
    spec: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jobbot.browser.sessions import SiteSpec
    from jobbot.cli import run_cli

    assert isinstance(spec, SiteSpec)
    code = run_cli(
        ["browser", "chrome-debug", "--site", spec.site, "--port", "9224", "--print-only"],
        standalone_mode=False,
    )
    assert code == SUCCESS, capsys.readouterr().err


def test_an_unknown_site_lists_the_ones_that_exist(capsys: pytest.CaptureFixture[str]) -> None:
    from jobbot.cli import run_cli

    code = run_cli(
        ["browser", "chrome-debug", "--site", "monster", "--print-only"],
        standalone_mode=False,
    )
    assert code == GENERIC_FAILURE
    message = capsys.readouterr().err
    for spec in SITES:
        assert spec.site in message


def test_site_lookup_is_case_and_space_tolerant() -> None:
    assert site_spec(" GetOnBoard ") is site_spec("getonboard")
    assert site_spec("nope") is None
