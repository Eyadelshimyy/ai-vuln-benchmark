"""Regression: manual pause must not skip when stdin is unavailable."""

from __future__ import annotations

from jobbot.adapters.indeed.jobs import _looks_like_challenge_html
from jobbot.browser.manual import wait_for_manual_clear


def test_wait_for_manual_clear_polls_without_stdin() -> None:
    calls = {"n": 0}

    def is_clear() -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    sleeps: list[float] = []
    ok = wait_for_manual_clear(
        is_clear=is_clear,
        timeout_seconds=10.0,
        poll_seconds=0.01,
        stdin_available=False,
        sleep=sleeps.append,
        now=lambda: 0.0 if calls["n"] < 3 else 100.0,
    )
    # With our now() jumping to 100 after clear, loop should succeed on poll
    assert ok is True
    assert calls["n"] >= 3
    assert sleeps  # polled at least once


def test_wait_for_manual_clear_times_out() -> None:
    clock = {"t": 0.0}

    def now() -> float:
        return clock["t"]

    def sleep(dt: float) -> None:
        clock["t"] += dt

    ok = wait_for_manual_clear(
        is_clear=lambda: False,
        timeout_seconds=5.0,
        poll_seconds=2.0,
        stdin_available=False,
        sleep=sleep,
        now=now,
    )
    assert ok is False
    assert clock["t"] >= 5.0


def test_wait_for_manual_clear_uses_enter_on_tty() -> None:
    entered = {"done": False}

    def read_enter() -> None:
        entered["done"] = True

    ok = wait_for_manual_clear(
        is_clear=lambda: True,
        stdin_available=True,
        read_enter=read_enter,
    )
    assert ok is True
    assert entered["done"] is True


def test_challenge_html_detects_cloudflare_spanish_page() -> None:
    html = """
    <title>Security Check - Indeed.com</title>
    <script>window.INDEED_CLOUDFLARE_STATIC_PAGE={PAGE_TYPE:"captcha"};</script>
    <h1>Verificación adicional requerida</h1>
    """
    assert _looks_like_challenge_html(html) is True


def test_challenge_html_ignores_normal_job_results() -> None:
    html = """
    <div data-jk="abc123def"><h2>Senior Data Scientist</h2></div>
    """
    assert _looks_like_challenge_html(html) is False
