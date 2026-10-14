"""GET /api/status — what's wrong with this install.

Health stays a cheap liveness probe. This route is the one an operator
reads. SMTP values never leave the process: the email check is booleans
and variable names.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Request

router = APIRouter(tags=["status"])

_SMTP_VARS = ("TRACEBI_SMTP_URL", "TRACEBI_SMTP_FROM")


def _check(name: str, ok: bool, detail) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _output_writable() -> dict:
    # Same probe the startup warning uses. Kept here so the router does not
    # call the private helper on the reports router.
    out_dir = os.path.join(os.getcwd(), "output")
    probe = os.path.join(out_dir, ".tracebi-write-probe")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("")
        os.unlink(probe)
    except OSError as exc:
        reason = exc.strerror or str(exc)
        return _check("output_writable", False, f"cannot write {out_dir}: {reason}")
    return _check("output_writable", True, out_dir)


def _discovery() -> dict:
    from tracebi.web.discovery import discovery_report

    names = [
        entry.get("file") or entry.get("module")
        for entry in discovery_report()
        if entry.get("status") == "failed"
    ]
    return _check("discovery", not names, {"failed": len(names), "names": names})


def _models_loaded() -> dict:
    from tracebi import model_registry

    loaded: set[int] = set()
    failed: list[str] = []
    for name in model_registry.list_models():
        try:
            model = model_registry.get_model(name)
        except Exception:  # noqa: BLE001 — a broken file is the check's answer
            failed.append(name)
            continue
        loaded.add(id(model))
    return _check(
        "models_loaded", not failed, {"count": len(loaded), "failed": failed},
    )


def _email_configured() -> dict:
    present = {
        name: bool(os.environ.get(name, "").strip()) for name in _SMTP_VARS
    }
    return _check("email_configured", all(present.values()), present)


def _schedules() -> dict:
    from tracebi.schedule import discover_library_schedules

    on = os.environ.get("TRACEBI_SCHEDULES_IN_SERVER") == "1"
    schedules, _errors = discover_library_schedules()
    return _check(
        "schedules_in_server", True, {"on": on, "count": len(schedules)},
    )


def _auth() -> dict:
    from tracebi.web.api.auth import posture_line

    return _check("auth", True, posture_line())


def collect_checks() -> list[dict]:
    return [
        _output_writable(),
        _discovery(),
        _models_loaded(),
        _email_configured(),
        _schedules(),
        _auth(),
    ]


@router.get("/status")
def status(request: Request) -> dict:
    from tracebi import _updates

    # From the cached release check only; a stale cache refreshes in the
    # background, so this never waits on GitHub.
    update = _updates.status(wait=False)
    update.pop("notes", None)
    return {"version": request.app.version, "checks": collect_checks(),
            "update": update,
            # Build mode: point at a figure and the agent can see what you mean
            # (dev-state only; see /api/reports/{name}/workbench/pointing).
            "build_mode": os.environ.get("TRACEBI_DEV_MODE") == "1"}
