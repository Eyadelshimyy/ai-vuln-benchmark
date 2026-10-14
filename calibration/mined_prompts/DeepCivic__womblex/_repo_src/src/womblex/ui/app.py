"""FastAPI app factory for the console sidecar.

Serves the read API, and the built SPA (``ui/``, docs/ui-plan.md merge 4)
when one has been built alongside it.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from womblex.ui.deps import UISettings
from womblex.ui.routes import composer, dashboard, execute, resources, runs

# `ui/` is not vendored into the wheel — only Dockerfile.ui's builder stage
# produces this directory, at the container's WORKDIR. Resolved from cwd
# rather than `__file__`: a normal
# (non-editable) `pip install` copies this module into site-packages, where
# there is no sibling `ui/` to find, so cwd is the only path that agrees
# with where the image actually places the build.
DEFAULT_SPA_DIR = Path("ui/build")


def create_app(
    *,
    output_root: Path | None = None,
    store_uri: str | None = None,
    db_dsn: str | None = None,
    presets_dir: Path | None = None,
    ingest_uri: str | None = None,
    spa_dir: Path | None = DEFAULT_SPA_DIR,
) -> FastAPI:
    """Build the console app, bound to one run source for its lifetime.

    Exactly one of ``output_root`` / ``store_uri`` must be given;
    :class:`~womblex.ui.deps.UISettings` enforces that and raises
    ``ValueError`` otherwise. Binding at construction rather than per request
    is what keeps the run source out of the URL space — no endpoint can be
    talked into reading a directory the operator did not mount.

    ``db_dsn`` is the optional job queue the Dashboard reads. Omitted means
    no queue, which is a normal local deployment — the dashboard falls back
    to the run's own per-stage checkpoints.

    ``presets_dir`` is where the Pipeline Composer saves operator-authored
    presets (one JSON file each). Omitted disables saving — the built-in
    presets still serve, but ``POST /api/composer/presets`` refuses with 409.

    ``ingest_uri`` names where source documents arrive from.
    Omitted, the Execution Controls cannot dispatch. Raises ``ValueError``
    when it is not disjoint from ``store_uri``'s effective ``runs/`` output.

    ``spa_dir`` is mounted only if it exists — a bare ``womblex[ui]`` install
    with no SvelteKit build alongside it still serves the read API, just
    without the frontend.
    """
    settings = UISettings(
        output_root=output_root, store_uri=store_uri,
        db_dsn=db_dsn,
        presets_dir=presets_dir, ingest_uri=ingest_uri,
    )
    app = FastAPI(title="Womblex Console")
    app.state.settings = settings
    app.include_router(runs.router)
    app.include_router(dashboard.router)
    app.include_router(composer.router)
    app.include_router(resources.router)
    app.include_router(execute.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "mode": "remote" if settings.is_remote else "local"}

    if spa_dir is not None and spa_dir.is_dir():
        _mount_spa(app, spa_dir)

    return app


def resolve_spa_path(spa_dir: Path, full_path: str) -> Path:
    """A real file under `spa_dir` for `full_path`, else `spa_dir/index.html`.

    `full_path` is attacker-controlled (e.g. `../../etc/passwd`) — resolved
    and checked for containment before it ever reaches the filesystem, since
    `spa_dir / full_path` alone does not stop it walking outside `spa_dir`.
    """
    root = spa_dir.resolve()
    candidate = (spa_dir / full_path).resolve()
    if candidate.is_relative_to(root) and candidate.is_file():
        return candidate
    return root / "index.html"


def _is_api_path(full_path: str) -> bool:
    """Whether the catch-all was reached by a request into the API namespace."""
    return full_path == "api" or full_path.startswith("api/")


def _mount_spa(app: FastAPI, spa_dir: Path) -> None:
    """Serve the SvelteKit build, with an SPA fallback for client-side routes.

    Registered after the API routes above, so `/api/*` still matches them
    first — Starlette tries routes in registration order, and only falls
    through to this catch-all when nothing more specific matched. `_app/` is
    the SvelteKit asset prefix; anything else resolves to a real file under
    `spa_dir` or, for a client route like `/corpus`, to `index.html` so the
    SPA's own router can take over.
    """
    app.mount("/_app", StaticFiles(directory=spa_dir / "_app"), name="spa-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str) -> FileResponse:
        # An unmatched path under /api/ is a wrong endpoint, not a client
        # route: serving index.html there would answer a bad API call with
        # 200 and an HTML body, which a JSON client reports as a parse error
        # rather than the 404 it actually is.
        if _is_api_path(full_path):
            raise HTTPException(status_code=404, detail=f"no such endpoint: /{full_path}")
        return FileResponse(resolve_spa_path(spa_dir, full_path))
