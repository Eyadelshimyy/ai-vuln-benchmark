"""Minimal continuous runner that records failures and continues."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from jobbot.config import JobbotConfig, load_config
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.exit_codes import (
    AUTH_REQUIRED,
    GENERIC_FAILURE,
    MANUAL_CHALLENGE,
    SUCCESS,
    UI_CHANGED,
)
from jobbot.ops.failures import FailureRecord, record_failure

LOOP_COMMANDS = frozenset({"getonboard-prepare", "getonboard-sync"})


@dataclass(frozen=True)
class LoopTickResult:
    ok: bool
    exit_code: int
    failure: FailureRecord | None = None
    detail: str = ""


def _run_getonboard_prepare(config: JobbotConfig) -> None:
    from jobbot.adapters.getonboard.client import GetOnBoardProfileClient

    GetOnBoardProfileClient.from_config(config).prepare_package()


def _run_getonboard_sync(config: JobbotConfig) -> None:
    from jobbot.adapters.getonboard.client import GetOnBoardProfileClient

    GetOnBoardProfileClient.from_config(config).prepare_package()


_RUNNERS: dict[str, Callable[[JobbotConfig], None]] = {
    "getonboard-prepare": _run_getonboard_prepare,
    "getonboard-sync": _run_getonboard_sync,
}


def run_loop_tick(
    cmd: str,
    *,
    config: JobbotConfig | None = None,
) -> LoopTickResult:
    """Execute one loop step; on error persist failure and return codes."""
    if cmd not in LOOP_COMMANDS:
        msg = f"unknown loop cmd {cmd!r}; expected one of {sorted(LOOP_COMMANDS)}"
        raise ValueError(msg)
    cfg = config or load_config()
    runner = _RUNNERS[cmd]
    try:
        runner(cfg)
        return LoopTickResult(ok=True, exit_code=SUCCESS, detail="ok")
    except Exception as exc:  # noqa: BLE001 — loop must survive and record
        exit_code = _classify_exception(exc)
        engine = make_engine(cfg.database_path)
        session = make_session_factory(engine)()
        try:
            failure = record_failure(
                session,
                config=cfg,
                exit_code=exit_code,
                argv=["jobbot", "ops", "loop", "--cmd", cmd],
                exc=exc,
                component="getonboard" if cmd.startswith("getonboard") else "ops",
                context={"loop_cmd": cmd},
            )
        finally:
            session.close()
        return LoopTickResult(
            ok=False,
            exit_code=exit_code,
            failure=failure,
            detail=str(exc),
        )


def _classify_exception(exc: BaseException) -> int:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "auth" in name or "login" in text or "not authenticated" in text:
        return AUTH_REQUIRED
    if "captcha" in text or "challenge" in text or "2fa" in text:
        return MANUAL_CHALLENGE
    if "ui_changed" in text or "selector" in text and "not found" in text:
        return UI_CHANGED
    return GENERIC_FAILURE


def run_loop(
    cmd: str,
    *,
    interval_sec: float = 300.0,
    fail_fast: bool = False,
    max_ticks: int | None = None,
    config: JobbotConfig | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_tick: Callable[[LoopTickResult], None] | None = None,
    hitl_pause: Callable[[str], None] | None = None,
) -> int:
    """
    Run ``cmd`` repeatedly.

    On failure: record Fxxxx and continue (unless ``fail_fast`` or auth/challenge).
    Returns last non-success exit code, or SUCCESS if never failed.
    """
    cfg = config or load_config()
    last_code = SUCCESS
    ticks = 0
    while max_ticks is None or ticks < max_ticks:
        ticks += 1
        result = run_loop_tick(cmd, config=cfg)
        if on_tick is not None:
            on_tick(result)
        if not result.ok:
            last_code = result.exit_code
            if result.exit_code in {AUTH_REQUIRED, MANUAL_CHALLENGE}:
                reason = (
                    f"HITL required (exit {result.exit_code}); "
                    f"failure={result.failure.id if result.failure else '?'}"
                )
                if hitl_pause is not None:
                    hitl_pause(reason)
                else:
                    break
            if fail_fast:
                break
        if max_ticks is not None and ticks >= max_ticks:
            break
        sleep_fn(interval_sec)
    return last_code


def describe_loop_commands() -> dict[str, str]:
    return {
        "getonboard-prepare": "Cumulative GoB permanent profile prepare",
        "getonboard-sync": "Same as prepare (dry maintainer tick)",
    }
