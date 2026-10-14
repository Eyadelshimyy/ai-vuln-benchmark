"""
Background report runs, recorded in the shared state store.

The HTML stays in the file ``output_path`` names. Polling reads it back,
so a second worker that shares the database can return the same result.
The response shape the UI already polls is unchanged.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Optional

from tracebi.web.api.errors import error_detail


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detail(row: dict) -> dict:
    raw = row.get("detail")
    if isinstance(raw, str) and raw:
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _result(row: dict, detail: dict):
    if row.get("status") != "succeeded":
        return None
    path = row.get("output_path")
    manifest_path = detail.get("manifest_path")
    if not manifest_path and path:
        manifest_path = path + ".manifest.json"
    html = ""
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
    manifest = None
    if manifest_path and os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    return {
        "name": detail.get("name") or row.get("target"),
        "html": html,
        "manifest": manifest,
        "retained": detail.get("retained"),
        "html_path": path,
        "manifest_path": manifest_path,
    }


def _api(row: dict, *, with_result: bool) -> dict:
    detail = _detail(row)
    body = {
        "run_id": str(row["id"]),
        "kind": row.get("kind"),
        "name": row.get("target") or row.get("layer_name"),
        "status": row.get("status"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished") or row.get("completed_at"),
        "error": detail.get("error"),
    }
    if with_result:
        body["result"] = _result(row, detail)
    return body


class RunStore:
    def __init__(self, max_workers: int = 4) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tracebi-run"
        )

    def start(self, kind: str, name: str, fn: Callable[[], dict]) -> dict:
        """Submit ``fn`` and return the new run record. The row exists before return."""
        from tracebi.state import get_run, install_extra, record_run

        started = _now()
        with install_extra("web"):
            run_id = record_run(kind=kind, target=name, status="running", started=started)
            self._pool.submit(self._execute, run_id, fn)
            return _api(get_run(run_id), with_result=True)

    def _execute(self, run_id: int, fn: Callable[[], dict]) -> None:
        from tracebi.state import install_extra, update_run

        output_path = None
        try:
            result = fn()
            status, error = "succeeded", None
        except Exception as exc:  # noqa: BLE001 — failures are the payload here
            result, status = None, "failed"
            error = error_detail("Run failed", exc)
        detail: dict = {}
        if error is not None:
            detail["error"] = error
        if isinstance(result, dict):
            output_path = result.get("html_path")
            detail["name"] = result.get("name")
            detail["retained"] = result.get("retained")
            detail["manifest_path"] = result.get("manifest_path")
        finished = _now()
        with install_extra("web"):
            update_run(
                None, run_id, status=status, finished=finished,
                completed_at=finished, output_path=output_path, detail=detail,
            )

    def get(self, run_id: str) -> Optional[dict]:
        from tracebi.state import get_run, install_extra

        if not str(run_id).isdigit():
            return None
        with install_extra("web"):
            row = get_run(int(run_id))
        if row is None:
            return None
        return _api(row, with_result=True)

    def list_for(self, kind: str, name: str, limit: int = 10) -> list[dict]:
        """Newest-first run summaries (without the result payload)."""
        from tracebi.state import install_extra, list_runs

        with install_extra("web"):
            rows = list_runs(kind=kind, target=name, limit=limit)
        # list_runs returns the public shape; rebuild from the same fields.
        out = []
        for row in rows:
            detail = row.get("detail") or {}
            if not isinstance(detail, dict):
                detail = {}
            out.append({
                "run_id": str(row["id"]),
                "kind": row.get("kind"),
                "name": row.get("target"),
                "status": row.get("status"),
                "started_at": row.get("started"),
                "finished_at": row.get("finished"),
                "error": detail.get("error"),
            })
        return out


run_store = RunStore()
