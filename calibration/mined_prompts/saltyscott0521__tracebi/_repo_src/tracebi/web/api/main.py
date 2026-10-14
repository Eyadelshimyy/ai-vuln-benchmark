"""
TraceBi Web API — FastAPI application.

Startup loads the app module, if one is named, which populates the
registry with connectors, models, reports, and pipeline runners. Set the
TRACEBI_APP environment variable to point at a different module.

    TRACEBI_APP=myproject.tracebi_config uvicorn tracebi.web.api.main:app --reload

Project-root directories are also auto-discovered at startup so you can define
artifacts outside of the app module package:

    models/       DataModel definitions (each file exposes a ``model`` variable)
    pipelines/    PipelineRunner definitions (each file exposes a ``runner`` variable)
    reports/      Report packages — <name>/ (report.json + template.html) or
                  a <name>.json spec compiled into one
    scheduled/    Scheduled report scripts

Environment switches:
    TRACEBI_APP                 — app module to import (default: none; set
                                  tracebi.web.demo_app for the bundled demo)
    TRACEBI_MODELS_DIR          — model definitions folder (default: models)
    TRACEBI_PIPELINES_DIR       — pipeline definitions folder (default: pipelines)
    TRACEBI_REPORTS_DIR         — reports folder: specs, packages, factories (default: reports)
    TRACEBI_SCHEDULED_DIR       — scheduled scripts folder (default: scheduled)
    TRACEBI_DEV_MODE=1          — mount /_dev/reload; Build mode and the project
                                  feed (/api/workbench/project) answer, and the
                                  feed's heartbeat keeps show() posting
    TRACEBI_AUTH_USER / _PASS   — enable HTTP Basic auth
    TRACEBI_AUTH_PROXY_HEADER   — enable proxy header-trust auth
    TRACEBI_SCHEDULES_IN_SERVER=1
                            — run report.json schedules in this process
                              (off by default; one process only)
"""

import importlib
import os
import sys
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tracebi.web.api.errors import error_detail

from tracebi.web.api.routers import (
    connectors, models, reports, pipelines, docs, verify, desk, status, runs, drafts,
)
from tracebi.web.api.auth import install_if_configured as _install_auth
from tracebi.web.api.sso import SessionAuthMiddleware as _SessionAuthMiddleware
from tracebi.web.api.csrf import CSRFMiddleware as _CSRFMiddleware
from tracebi.web.api.csrf import allowed_origins as _allowed_origins

from contextlib import asynccontextmanager

from tracebi._version import get_version as _tracebi_version
from tracebi.schedule import server_lifespan


@asynccontextmanager
async def _lifespan(app):
    """In-server schedules, plus live discovery: a thread that picks up
    report packages, specs, model files and pipeline files added while the
    server runs.
    TRACEBI_DISCOVERY_INTERVAL sets the seconds between scans (default 5;
    0 turns it off)."""
    from tracebi.connections import connections_dir
    from tracebi.web.discovery import start_watcher

    try:
        interval = float(os.environ.get("TRACEBI_DISCOVERY_INTERVAL", "5"))
    except ValueError:
        interval = 5.0
    stop = None
    if interval > 0:
        stop = start_watcher(os.environ.get("TRACEBI_REPORTS_DIR", "reports"),
                             os.environ.get("TRACEBI_MODELS_DIR", "models"),
                             interval,
                             os.environ.get("TRACEBI_PIPELINES_DIR", "pipelines"),
                             str(connections_dir()))
    stop_beat = _start_feed_heartbeat() if os.environ.get("TRACEBI_DEV_MODE") == "1" else None
    try:
        async with server_lifespan(app), _mcp_sessions():
            yield
    finally:
        if stop is not None:
            stop.set()
        if stop_beat is not None:
            stop_beat.set()


@asynccontextmanager
async def _mcp_sessions():
    """Run the gateway's session manager while the app runs, when /mcp is served."""
    if _mcp_server is None:
        yield
        return
    async with _mcp_server.session_manager.run():
        yield


def _start_feed_heartbeat() -> threading.Event:
    """Keep the project feed live while the app runs in dev mode, so any script
    in the project can ``tracebi.workbench.show(...)`` with no setup and the
    exhibit lands where the person is looking. Stops (and the marker goes stale
    in seconds) when the server does. Returns the event that stops it."""
    from tracebi import workbench

    wb = workbench.discovery_dir(os.getcwd())
    stop = threading.Event()

    def beat() -> None:
        workbench.heartbeat(wb)
        while not stop.wait(workbench.HEARTBEAT_WINDOW / 3):
            workbench.heartbeat(wb)

    threading.Thread(target=beat, daemon=True).start()
    return stop


app = FastAPI(
    title="TraceBi API",
    description=("The trust layer for AI-generated analytics: a code-first BI "
                 "framework where every number has a receipt."),
    version=_tracebi_version(),
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In dev mode the app is beside an agent that re-runs transforms from the
# terminal, which needs the warehouse read-write; a read-only handle kept open
# between requests would block it. So in dev mode the app lets go of every
# warehouse after each request (the next query reopens it, which is cheap).
@app.middleware("http")
async def _release_warehouses_in_dev(request, call_next):
    response = await call_next(request)
    if os.environ.get("TRACEBI_DEV_MODE") == "1":
        from starlette.concurrency import run_in_threadpool

        from tracebi import model_registry
        from tracebi.registry import registry

        def release() -> None:
            registry.release_all()
            model_registry.release_all()

        await run_in_threadpool(release)
    return response


# Same-origin CSRF guard: a browser cross-site POST carries an Origin and is
# refused unless allowed; a request with no Origin (curl, CLI) is not a browser
# CSRF and passes. CORS alone does not stop a cross-site POST reaching us.
app.add_middleware(_CSRFMiddleware)

# Optional auth — Basic or reverse-proxy header trust, depending on env.
_auth_mode = _install_auth(app)
# App sign-in (OIDC): always in the stack, inert until serve_sso() turns it on,
# so a late start (tests, an app module) needs no middleware rebuild.
app.add_middleware(_SessionAuthMiddleware)
if _auth_mode:
    print(f"[tracebi] auth mode: {_auth_mode}")
else:
    print(
        "[tracebi] WARNING: no auth configured — the API (including pipeline "
        "run endpoints) is open to anyone who can reach this server. Set "
        "TRACEBI_AUTH_USER/TRACEBI_AUTH_PASS or TRACEBI_AUTH_PROXY_HEADER "
        "before exposing it beyond localhost."
    )

# Report builds are kept in output/ beside the working directory. When the
# server cannot write there, the page goes under the system temp dir and the
# path is recorded in the run store — worth one loud line, since a wiped
# temp dir looks like "reports keep re-running".
if reports._writable_output_html("tracebi-startup-check") is None:
    print(
        f"[tracebi] WARNING: cannot write report builds to "
        f"{os.path.join(os.getcwd(), 'output')} — builds are kept under the "
        f"system temp directory and the path is recorded in the run store. "
        f"Make the folder writable by this process (in Docker, see the "
        f"output-perms service in docker-compose.yml).",
        file=sys.stderr,
    )

app.include_router(connectors.router, prefix="/api")
app.include_router(desk.router,       prefix="/api")
app.include_router(models.router,     prefix="/api")
app.include_router(reports.router,    prefix="/api")
app.include_router(reports.share_router)                # /r/<name>: the share link
app.include_router(reports.workbench_router, prefix="/api")   # dev mode: the project feed
app.include_router(pipelines.router,  prefix="/api")
app.include_router(docs.router,       prefix="/api")
app.include_router(verify.router,     prefix="/api")
app.include_router(status.router,     prefix="/api")
app.include_router(runs.router,       prefix="/api")
app.include_router(drafts.router,     prefix="/api")

# The agent gateway, served by this app at /mcp when TRACEBI_MCP_TOKEN is set,
# or when TRACEBI_OIDC_ISSUER is set (a sign-in per person, TraceBi's own OAuth
# endpoints beside /mcp): one process, so an agent's drafts are the app's
# drafts. The bearer token is the gate (/mcp and the OAuth endpoints are outside
# the app's Basic/proxy auth, which guards /api). Without either there is no
# /mcp here; `tracebi mcp --transport http` still serves it on its own.
_mcp_server = None
_gateway_routes: list = []


def serve_gateway(token=None, oauth=None):
    """Serve the gateway at /mcp behind *token* and/or per-person *oauth*
    (a ``tracebi.mcp_oauth.GatewayOAuth``). Returns the /mcp route; the OAuth
    routes are added with it (``unserve_gateway`` removes them all).
    The app's lifespan runs its sessions, so call this before the app starts."""
    global _mcp_server
    from starlette.routing import Route

    from tracebi.mcp_server import build_server

    _mcp_server = build_server(token=token, oauth=oauth)
    # host is not loopback: behind a proxy the Host header is the public name,
    # and the bearer token, not a Host check, is what keeps strangers out.
    route = Route("/mcp", endpoint=_mcp_server.streamable_http_app(
        streamable_http_path="/mcp", host="0.0.0.0"))
    added = [route] + (oauth.routes() if oauth is not None else [])
    _gateway_routes[:] = added
    app.router.routes[0:0] = added
    return route


def unserve_gateway() -> None:
    """Take the gateway's routes back out (tests; the app never does this)."""
    global _mcp_server
    for route in _gateway_routes:
        if route in app.router.routes:
            app.router.routes.remove(route)
    _gateway_routes.clear()
    _mcp_server = None


def serve_sso(oauth) -> None:
    """Turn on the app's own sign-in (/login, /logout, session cookies) over
    *oauth*, the ``GatewayOAuth`` that holds the OIDC settings. Role
    enforcement is on from here: the identity provider's groups decide."""
    from tracebi.web.api import sso

    app_sso = sso.AppSSO(oauth)
    routes = app_sso.routes()
    sso.enable(app_sso)
    _sso_routes[:] = routes
    app.router.routes[0:0] = routes


def unserve_sso() -> None:
    """Take app sign-in back out (tests; the app never does this)."""
    from tracebi.web.api import sso

    sso.enable(None)
    for route in _sso_routes:
        if route in app.router.routes:
            app.router.routes.remove(route)
    _sso_routes.clear()


_sso_routes: list = []

if (os.environ.get("TRACEBI_LOCAL_USERS", "").strip()
        or os.environ.get("TRACEBI_DEMO_LOGIN", "").strip()):
    # The built-in sign-in (a demo user and/or named users): the gateway signs
    # people in on a page of its own. The app's own pages stay as they are.
    from tracebi.mcp_oauth import GatewayOAuth, OAuthConfig

    _static = os.environ.get("TRACEBI_MCP_TOKEN", "").strip() or None
    serve_gateway(_static, oauth=GatewayOAuth(OAuthConfig.from_env()))
    print("[tracebi] mcp gateway: /mcp (built-in sign-in via TRACEBI_LOCAL_USERS / "
          "TRACEBI_DEMO_LOGIN" + ("; static TRACEBI_MCP_TOKEN also accepted)" if _static else ")"))
elif os.environ.get("TRACEBI_OIDC_ISSUER", "").strip():
    from tracebi.mcp_oauth import GatewayOAuth, OAuthConfig

    _static = os.environ.get("TRACEBI_MCP_TOKEN", "").strip() or None
    _oauth = GatewayOAuth(OAuthConfig.from_env())
    serve_gateway(_static, oauth=_oauth)
    serve_sso(_oauth)
    print("[tracebi] app sign-in: /login (OIDC; roles enforced from the provider's groups)")
    print("[tracebi] mcp gateway: /mcp (sign-in per person via TRACEBI_OIDC_ISSUER"
          + ("; static TRACEBI_MCP_TOKEN also accepted)" if _static else ")"))
elif os.environ.get("TRACEBI_MCP_TOKEN", "").strip():
    serve_gateway(os.environ["TRACEBI_MCP_TOKEN"].strip())
    print("[tracebi] mcp gateway: /mcp (bearer TRACEBI_MCP_TOKEN)")

# Dev-mode reload endpoint — opt-in via TRACEBI_DEV_MODE=1.
if os.environ.get("TRACEBI_DEV_MODE") == "1":
    from tracebi.web.api.routers import dev
    app.include_router(dev.router, prefix="/api")
    print("[tracebi] dev mode: /api/_dev/reload mounted")


@app.get("/api/health")
def health():
    # Resolved once at startup (FastAPI's app.version). Probes call this
    # constantly, so don't re-read package metadata on each request.
    return {"status": "ok", "version": app.version}


@app.get("/api/me")
def me(request: Request):
    """Who the app thinks is asking, and how they signed in. The UI shows the
    person (and a sign-out) only for OIDC."""
    how = getattr(request.state, "sign_in", None) or _auth_mode or "none"
    return {"actor": getattr(request.state, "user", None),
            "role": getattr(request.state, "role", "admin"), "sign_in": how}


@app.get("/api/discovery")
def discovery():
    """
    What happened to every file auto-discovery looked at: registered,
    skipped (and why), or failed (with the error).

    Discovery is convention-based and quiet — a file in the wrong place or
    one that raises on import simply never appears. This answers "why isn't
    my report showing up?" without reading the server's stderr.
    """
    from tracebi.web.discovery import discovery_report

    entries = discovery_report()
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    return {"summary": counts, "entries": entries}


@app.get("/api/spec/schema")
def spec_schema():
    """JSON Schema for a report spec, generated from the section dataclasses."""
    from tracebi.spec import json_schema
    return json_schema()


@app.post("/api/spec/validate")
def spec_validate(body: dict):
    """
    Check a report spec without executing it.

    Validates section types, field names, enum values, and — against the
    registered models — that each referenced model, fact, measure and
    dimension exists. Returns field-scoped errors like
    ``sections[2].data.query.fact``, so an author can fix the spec before
    anything runs.
    """
    from tracebi.spec import ReportSpec

    try:
        spec = ReportSpec.from_dict(body)
    except Exception as exc:  # noqa: BLE001 — a malformed spec is a 400
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = spec.validate(_registered_models())
    result["data_coverage"] = spec.data_coverage()
    return result


@app.post("/api/spec/render")
def spec_render(body: dict):
    """Compile a report spec to the artifact package and render it to HTML.

    One report form: a spec renders through the same path as a hand-authored
    package, so it gets figures, badges, the receipt drawer, and a schema-2
    manifest — not the legacy renderer's bare HTML.
    """
    import os
    import tempfile
    from pathlib import Path

    from tracebi.reports.compile_spec import compile_spec
    from tracebi.reports.template_package import TemplatePackage
    from tracebi.spec import ReportSpec

    try:
        spec = ReportSpec.from_dict(body)
        compiled = compile_spec(spec)
        models = _registered_models()
        with tempfile.TemporaryDirectory() as d:
            for fname, content in compiled.files.items():
                (Path(d) / fname).write_text(content, encoding="utf-8")
            out = os.path.join(d, "__render.html")
            manifest = TemplatePackage(d).render(models, out, save_manifest=False)
            html = Path(out).read_text(encoding="utf-8")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=error_detail("Spec render failed", exc)
        ) from exc

    return {"html": html, "manifest": manifest.to_dict()}


def _registered_models() -> dict:
    """Registered models keyed by name, for spec validation and building."""
    from tracebi.web.api.registry import registry as _reg
    return {m["name"]: _reg.get_model(m["name"]) for m in _reg.list_models()}


@app.get("/api/schema")
def schema():
    """
    The framework's vocabulary as data: every report section with its
    fields, types, defaults and allowed values, the DataSet
    verbs, measure kinds, filter operators, and the discovery conventions.

    Generated from the code rather than hand-maintained, so it stays
    correct. Intended for tools — an agent authoring a project, an editor
    completing a constructor, or a UI building a form.
    """
    from tracebi.capabilities import describe
    return describe()


# ── Load app module ────────────────────────────────────────────────────────

# An app module wires up connectors, which cannot be
# expressed as a file convention. The default is no app module: a project
# that only uses the models/ pipelines/ reports/ directories needs
# none, and a serve should show *your* project, not the bundled demo. Opt
# into the demo explicitly with TRACEBI_APP=tracebi.web.demo_app (it is
# self-contained and runs from any working directory).
_app_module = os.environ.get("TRACEBI_APP", "").strip()

if _app_module:
    # Checked *before* importing, not in the handler below, because the legacy
    # spelling may still import and do nothing: `git pull` leaves the old
    # web/api and web/demo_app directories behind whenever an untracked file
    # (a stale __pycache__) is in them, and a directory with no modules is a
    # perfectly importable namespace package. That registers nothing, raises
    # nothing, and boots a server that passes every healthcheck with an empty
    # registry — so refuse the spelling itself rather than waiting to fail.
    if _app_module == "web" or _app_module.startswith("web."):
        raise ImportError(
            f"TRACEBI_APP={_app_module!r} predates the namespace move: the web "
            f"app now ships inside the tracebi package. Use "
            f"'tracebi.{_app_module}' (e.g. TRACEBI_APP=tracebi.web.demo_app). "
            f"If you upgraded in place, delete any leftover top-level "
            f"web/api and web/demo_app directories too."
        )
    try:
        importlib.import_module(_app_module)
    except Exception as exc:
        # Deliberately broad: an app module can fail with anything (a KeyError
        # from a missing model, say), and a broken app module must start the
        # server empty with a warning rather than crash it.
        import warnings
        warnings.warn(
            f"TRACEBI_APP module '{_app_module}' could not be imported: {exc}. "
            "The API will start with an empty registry.",
            stacklevel=1,
        )

# Folder-based auto-discovery — decorator-based artifacts fire registry side
# effects on import (reports use @register.report).
_scheduled = os.environ.get("TRACEBI_SCHEDULED_DIR", "scheduled")
if os.path.isdir(_scheduled):
    from tracebi.web.discovery import auto_discover as _auto_discover
    if any(name.endswith((".py", ".ipynb"))
           for name in os.listdir(_scheduled) if not name.startswith(".")):
        print(
            f"[tracebi] {_scheduled} is deprecated and never ran these "
            f"reports. Put a \"schedule\" block in report.json. "
            f"See `tracebi schedule --help`.",
            file=sys.stderr,
        )
    _discovered = _auto_discover(_scheduled)
    if _discovered:
        print(f"[tracebi] auto-discovered {len(_discovered)} module(s) "
              f"from {_scheduled}")

from tracebi.web.discovery import discover_library as _discover_library
_lib_discovered = _discover_library()
if _lib_discovered:
    print(f"[tracebi] auto-discovered {len(_lib_discovered)} module(s) "
          f"from the report library")

# Connections discovery — each connections/<name>.yaml declares a source. Done
# before models so the Sources page lists them; nothing is resolved or opened.
from tracebi.connections import connections_dir as _connections_dir
_conn_dir = str(_connections_dir())
if os.path.isdir(_conn_dir):
    from tracebi.web.discovery import register_connections as _register_connections
    _disc_conns = _register_connections(_conn_dir)
    if _disc_conns:
        print(f"[tracebi] auto-discovered {len(_disc_conns)} connection(s) from {_conn_dir}")

# Models discovery — each models/<name>.py exposes a `model` variable.
_models_dir = os.environ.get("TRACEBI_MODELS_DIR", "models")
if os.path.isdir(_models_dir):
    # The same registration live discovery repeats while the server runs.
    from tracebi.web.discovery import register_models as _register_models
    _disc_models = _register_models(_models_dir)
    if _disc_models:
        print(f"[tracebi] auto-discovered {len(_disc_models)} model(s) from {_models_dir}")

# Pipelines discovery — each pipelines/<name>.py exposes a `runner` variable;
# a pipelines/<name>.yaml compiles to one.
_pipelines_dir = os.environ.get("TRACEBI_PIPELINES_DIR", "pipelines")
if os.path.isdir(_pipelines_dir):
    # The same registration live discovery repeats while the server runs.
    from tracebi.web.discovery import register_pipelines as _register_pipelines
    _disc_pipes = _register_pipelines(_pipelines_dir)
    if _disc_pipes:
        print(f"[tracebi] auto-discovered {len(_disc_pipes)} pipeline(s) from {_pipelines_dir}")


# ── Serve built React UI (production) ──────────────────────────────────────

# tracebi/web/ui/dist — inside the installed package, which is why the wheel
# can carry the bundle at all. The Node workspace stays at the repo root
# (web/ui/) and vite's build.outDir writes here.
_ui_dist = os.path.join(os.path.dirname(__file__), "..", "ui", "dist")

# The bundle is index.html, not the directory: `npm run build` empties dist
# before it writes (vite's emptyOutDir), so a build that fails leaves the
# directory there and nothing in it. Mounting that serves the bare 404 this
# branch exists to prevent.
if os.path.isfile(os.path.join(_ui_dist, "index.html")):
    from starlette.exceptions import HTTPException as _StarletteHTTPException

    class _SPAFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            try:
                return await super().get_response(path, scope)
            except _StarletteHTTPException as exc:
                # A missing hashed asset (a deploy swapped the bundle) is a
                # 404 no cache keeps: index.html at 200 under a .js URL got
                # cached by the CDN and blanked the app. Page routes fall back.
                if exc.status_code == 404 and path.startswith("assets/"):
                    raise _StarletteHTTPException(
                        404, headers={"Cache-Control": "no-store"}) from None
                if exc.status_code == 404:
                    return await super().get_response("index.html", scope)
                raise

    app.mount("/", _SPAFiles(directory=_ui_dist, html=True), name="ui")
else:
    # tracebi/web/ui/dist is gitignored, so a fresh clone has no bundle. Without this
    # branch "/" is a bare 404 and nothing says why — say why instead.
    from fastapi import Request as _Request
    from fastapi.responses import HTMLResponse as _HTMLResponse, JSONResponse as _JSONResponse

    # The remedy depends on which tree we are running from. A checkout can
    # build the bundle; an installed package cannot — telling someone to run
    # npm inside site-packages is not an instruction they can follow.
    # tracebi/web/api/ → three levels up is the repo root (or site-packages,
    # which has no pyproject.toml and so takes the installed-package branch).
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if os.path.isfile(os.path.join(_repo_root, "pyproject.toml")):
        _UI_REMEDY = (
            f"cd {os.path.join(_repo_root, 'web', 'ui')} && npm ci && npm run build"
        )
    else:
        _UI_REMEDY = (
            "This installed package carries no UI bundle. Install a wheel built "
            "with one (see .github/workflows/release.yml), or run the server from "
            "a clone of https://github.com/saltyscott0521/tracebi"
        )
    _UI_MISSING_MSG = (
        "The TraceBi web UI has not been built, so there is no page to serve here. "
        f"The API is running normally — try /api/health. {_UI_REMEDY}"
    )

    print(f"[tracebi] WARNING: no built UI at {_ui_dist} — the API works but / "
          f"has no page. {_UI_REMEDY}", file=sys.stderr)

    def _accepts_html(request: _Request) -> bool:
        """True when the client listed text/html as acceptable.

        A substring test on the raw header reads ``text/html;q=0`` (an
        explicit refusal) as a request for HTML and ``TEXT/HTML`` as a
        refusal, so parse the list instead. Multiple Accept headers count.
        """
        for entry in ",".join(request.headers.getlist("accept")).split(","):
            media, _, params = entry.partition(";")
            if media.strip().lower() != "text/html":
                continue
            for param in params.split(";"):
                key, _, value = param.partition("=")
                if key.strip().lower() == "q":
                    try:
                        return float(value) > 0
                    except ValueError:
                        return False
            return True
        return False

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def ui_not_built(request: _Request):
        if not _accepts_html(request):
            return _JSONResponse({"detail": _UI_MISSING_MSG, "remedy": _UI_REMEDY})
        return _HTMLResponse(
            "<!doctype html><meta charset='utf-8'>"
            "<title>TraceBi — UI not built</title>"
            "<h1>TraceBi API is running</h1>"
            "<p>The web UI has not been built, so there is no page to serve here.</p>"
            f"<pre>{_UI_REMEDY}</pre>"
            "<p>Restart the server once that is in place. The API is unaffected — "
            "see <a href='/api/health'>/api/health</a> and "
            "<a href='/docs'>/docs</a>.</p>"
        )
