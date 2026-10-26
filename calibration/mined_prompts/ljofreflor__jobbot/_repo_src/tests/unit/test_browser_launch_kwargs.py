"""Regression: prefer installed Chrome so Turnstile is less likely to loop."""

from pathlib import Path

from jobbot.browser.session import launch_persistent_kwargs


def test_launch_kwargs_include_chrome_channel_by_default() -> None:
    kwargs = launch_persistent_kwargs(profile_dir=Path("/tmp/jobbot-browser"))
    assert kwargs["channel"] == "chrome"
    assert kwargs["user_data_dir"] == "/tmp/jobbot-browser"
    assert kwargs["headless"] is False


def test_launch_kwargs_omit_channel_when_disabled() -> None:
    kwargs = launch_persistent_kwargs(
        profile_dir=Path("/tmp/jobbot-browser"),
        channel=None,
    )
    assert "channel" not in kwargs
