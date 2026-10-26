# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml", "tabulate"]
# ///
"""Derive the edge-reachable route inventory from OtterWorks source.

A black-box scanner only finds what it can crawl, so an endpoint nothing links
to is an endpoint nothing tests. Reading the code instead gives the *intended*
surface — including routes added yesterday — which is what the DAST suite's
coverage is measured against.

The gateway's own route table is the authority on what is reachable at the edge:
a backend route whose prefix the gateway does not proxy is not attackable from
outside, and one it does proxy is, whatever the frontend happens to link to.

Extraction is per-language and deliberately literal: a route is reported only
when its path is a string in the source. Anything assembled at runtime is not
reported, and a service with no extractor is reported as *unknown* rather than
as covered — an unverifiable claim about coverage is the failure mode this whole
module exists to prevent.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES = REPO_ROOT / "services"
GATEWAY_CONFIG = SERVICES / "api-gateway" / "internal" / "config" / "config.go"
GATEWAY_JWT = SERVICES / "api-gateway" / "internal" / "middleware" / "jwt.go"
ATTACK_SURFACE = REPO_ROOT / "security" / "dast" / "attack-surface.yaml"

#: The bodies of DefaultPublicPaths() and DefaultPrefixPaths() in the gateway's
#: JWT middleware — the only paths it serves without a validated token.
_GO_PATH_LIST = re.compile(
    r"func Default(?P<kind>Public|Prefix)Paths\(\)\s*\[\]string\s*\{.*?\{(?P<body>[^}]*)\}",
    re.DOTALL,
)


def gateway_public_paths(source: Path = GATEWAY_JWT) -> tuple[set[str], tuple[str, ...]] | None:
    """(exact paths, path prefixes) the gateway serves without a token, or None.

    Hand-maintaining this list is how a suppression quietly stops matching what
    the application does: a route that used to be public keeps its pass long
    after the middleware started protecting it. Reading the middleware means the
    sweep's idea of "expected to answer anonymously" cannot drift from the
    gateway's, and None — rather than an empty set — when the middleware cannot
    be read, because guessing either way produces a wrong verdict.
    """
    if not source.exists():
        return None
    found = {
        match["kind"]: set(re.findall(r'"([^"]+)"', match["body"]))
        for match in _GO_PATH_LIST.finditer(source.read_text())
    }
    if not found.get("Public"):
        return None
    return found["Public"], tuple(sorted(found.get("Prefix", set())))


#: Methods whose request, if the route turns out to be unauthenticated, is not a
#: probe but the operation itself.
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def may_sweep_unsafely() -> bool:
    """Has the operator declared this run's target theirs to destroy?

    Sweeping with the route's real method is the point — a GET-only sweep cannot
    find an unauthenticated DELETE — but a route that answers has been carried out,
    and :func:`sweep_exclusions` only covers the tenant-wide operations someone
    thought of. So it takes ``DAST_SWEEP_UNSAFE_METHODS``, and nothing else: an
    address is not evidence of what is behind it, and this repo's own runbook
    reaches a live shared tenant at ``localhost:8080`` through ``kubectl
    port-forward``. Without the declaration those routes are reported unswept
    rather than performed — here rather than in the probe, because the coverage
    gate has to name the same routes as unswept for the hole to stay visible.
    """
    return os.getenv("DAST_SWEEP_UNSAFE_METHODS", "").strip().lower() in {"1", "true", "yes"}


def sweep_exclusions(path: Path = ATTACK_SURFACE) -> dict[str, str]:
    """`METHOD /path` -> why the anonymous sweep must not send this route.

    The sweep issues each route's own method, so a route that is served without
    a token is not only reported but *performed*. For a route that creates an
    object under a scan-owned id that is the point; for one that acts on the
    whole tenant (reindex everything, mark every notification read) it is
    collateral damage on the environment being scanned. Those are named here and
    left to hand-written probes, and the coverage gate treats the exclusion as
    the reason a route was not swept rather than pretending it was attacked.
    """
    if not path.exists():
        return {}
    spec = yaml.safe_load(path.read_text()) or {}
    return {
        f"{entry['method'].upper()} {entry['path']}": entry.get("reason", "")
        for entry in spec.get("sweep_exclusions") or []
        if entry.get("method") and entry.get("path")
    }


#: `"/api/v1/auth": c.AuthServiceURL,` inside ServiceRoutes().
_GATEWAY_ROUTE = re.compile(r'"(?P<prefix>/[^"]+)":\s*c\.(?P<field>\w+)ServiceURL')
_METHODS = ("get", "post", "put", "patch", "delete")


@dataclass(frozen=True)
class Route:
    """One edge-reachable endpoint, as declared in a service's source."""

    method: str
    path: str
    service: str
    source: str

    @property
    def key(self) -> str:
        return f"{self.method} {self.path}"


def _camel_to_kebab(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


def gateway_prefixes(config: Path = GATEWAY_CONFIG) -> dict[str, str]:
    """Proxied prefix -> service directory name, from the gateway's route table."""
    if not config.exists():
        return {}
    return {
        match["prefix"]: f"{_camel_to_kebab(match['field'])}-service"
        for match in _GATEWAY_ROUTE.finditer(config.read_text())
    }


def repo_relative(path: Path) -> str:
    """Where a route was declared, repo-relative when the file is in the repo."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _join(*parts: str) -> str:
    joined = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"/{joined}" if joined else "/"


def _normalize(path: str) -> str:
    """Rewrite every path-parameter syntax to `{}` so paths compare across languages."""
    path = re.sub(r"\{[^}/]*\}", "{}", path)  # FastAPI, actix, Spring
    path = re.sub(r"<[^>/]*>", "{}", path)  # Flask
    path = re.sub(r":[^/]+", "{}", path)  # Rails, Express
    return path.rstrip("/") or "/"


# ── per-language extractors ───────────────────────────────────────────────────


def _fastapi(service: Path, name: str) -> list[Route]:
    """FastAPI: `include_router(mod.router, prefix=...)` plus `@router.<method>`."""
    main = service / "app" / "main.py"
    if not main.exists():
        return []
    prefixes: dict[str, list[str]] = {}
    for match in re.finditer(
        r"include_router\(\s*(\w+)\.router(?:\s*,\s*prefix=\"([^\"]*)\")?", main.read_text()
    ):
        prefixes.setdefault(match[1], []).append(match[2] or "")
    routes: list[Route] = []
    for module, mounts in prefixes.items():
        path = service / "app" / "api" / f"{module}.py"
        if not path.exists():
            continue
        text = path.read_text()
        for match in re.finditer(
            rf"@router\.({'|'.join(_METHODS)})\(\s*\"([^\"]*)\"", text, re.IGNORECASE
        ):
            for mount in mounts:
                routes.append(
                    Route(
                        match[1].upper(),
                        _normalize(_join(mount, match[2])),
                        name,
                        repo_relative(path),
                    )
                )
    return routes


def _flask(service: Path, name: str) -> list[Route]:
    """Flask: `register_blueprint(bp, url_prefix=...)` plus `@bp.route`."""
    main = service / "app" / "main.py"
    if not main.exists():
        return []
    prefixes: dict[str, list[str]] = {}
    for match in re.finditer(
        r"register_blueprint\(\s*(\w+)(?:\s*,\s*url_prefix=\"([^\"]*)\")?", main.read_text()
    ):
        prefixes.setdefault(match[1], []).append(match[2] or "")
    routes: list[Route] = []
    for path in sorted((service / "app" / "api").glob("*.py")):
        text = path.read_text()
        for match in re.finditer(
            r"@(\w+)\.route\(\s*\"([^\"]*)\"(?:[^)]*methods=\[([^\]]*)\])?", text
        ):
            methods = (
                [m.strip().strip("\"'").upper() for m in match[3].split(",")]
                if match[3]
                else ["GET"]
            )
            for mount in prefixes.get(match[1], []):
                for method in methods:
                    routes.append(
                        Route(
                            method,
                            _normalize(_join(mount, match[2])),
                            name,
                            repo_relative(path),
                        )
                    )
    return routes


def _actix(service: Path, name: str) -> list[Route]:
    """Actix-web: `web::scope("…")` blocks containing `.route("…", web::<method>()…)`."""
    main = service / "src" / "main.rs"
    if not main.exists():
        return []
    text = main.read_text()
    routes: list[Route] = []
    scopes: list[tuple[int, str]] = [
        (match.end(), match[1]) for match in re.finditer(r'web::scope\("([^"]*)"\)', text)
    ]
    for match in re.finditer(
        rf'\.route\(\s*"([^"]*)"\s*,\s*web::({"|".join(_METHODS)})\(\)', text, re.DOTALL
    ):
        # The nearest scope opened before this route is the one it is mounted in;
        # actix nests scopes textually, so source order is the mounting order.
        mount = next((prefix for start, prefix in reversed(scopes) if start < match.start()), "")
        routes.append(
            Route(
                match[2].upper(),
                _normalize(_join(mount, match[1])),
                name,
                repo_relative(main),
            )
        )
    return routes


_SPRING_CLASS = re.compile(r'@RequestMapping\(\s*"([^"]*)"')
_SPRING_METHOD = re.compile(r'@(Get|Post|Put|Patch|Delete)Mapping(?:\(\s*"([^"]*)")?')


def _spring(service: Path, name: str) -> list[Route]:
    """Spring MVC: class-level `@RequestMapping` plus method-level `@<Verb>Mapping`."""
    routes: list[Route] = []
    for suffix in ("*.java", "*.kt"):
        for path in sorted((service / "src" / "main").rglob(suffix)):
            text = path.read_text()
            class_match = _SPRING_CLASS.search(text)
            mount = class_match[1] if class_match else ""
            for match in _SPRING_METHOD.finditer(text):
                routes.append(
                    Route(
                        match[1].upper(),
                        _normalize(_join(mount, match[2] or "")),
                        name,
                        repo_relative(path),
                    )
                )
    return routes


_KTOR_ROUTE = re.compile(r'\broute\(\s*"([^"]*)"\s*\)\s*\{')
_KTOR_METHOD = re.compile(rf'\b({"|".join(_METHODS)})\s*(?:\(\s*"([^"]*)"\s*\))?\s*\{{')
_KTOR_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')


def _ktor(service: Path, name: str) -> list[Route]:
    """Ktor: nested `route("…") { }` blocks containing `get { }` / `put("…") { }`.

    Ktor's routing is a DSL rather than annotations, so a path is the
    concatenation of the blocks enclosing it and nesting has to be tracked by
    brace depth.
    """
    routes: list[Route] = []
    for path in sorted((service / "src" / "main").rglob("*.kt")):
        text = path.read_text()
        # A path parameter is written `{id}`, so braces inside string literals
        # would corrupt the depth count. Blank the literals for counting only,
        # keeping every offset (and therefore every match position) intact.
        depth_text = _KTOR_STRING.sub(lambda m: " " * len(m[0]), text)
        openers: dict[int, tuple[str, str]] = {}
        for match in _KTOR_ROUTE.finditer(text):
            openers[match.end() - 1] = ("route", match[1])
        for match in _KTOR_METHOD.finditer(text):
            openers.setdefault(match.end() - 1, (match[1].upper(), match[2] or ""))
        stack: list[str] = []
        for index, char in enumerate(depth_text):
            if char == "{":
                kind, declared = openers.get(index, ("", ""))
                if kind and kind != "route":
                    routes.append(
                        Route(
                            kind,
                            _normalize(_join(*stack, declared)),
                            name,
                            repo_relative(path),
                        )
                    )
                stack.append(declared if kind == "route" else "")
            elif char == "}" and stack:
                stack.pop()
    return routes


#: Only services whose routes can be read literally are extracted. The rest are
#: reported as unknown coverage, which is the honest answer: see module docstring.
EXTRACTORS = {
    "auth-service": _spring,
    "document-service": _fastapi,
    "file-service": _actix,
    "search-service": _flask,
    "report-service": _spring,
    "notification-service": _ktor,
}


def service_routes(name: str) -> list[Route] | None:
    """Routes declared by a service, or None when its routes cannot be read.

    An extractor that matches nothing is not evidence that a service has no
    routes — far more likely it was pointed at the wrong framework, or the
    source moved. Since the gateway only proxies a prefix for a service that
    serves something there, an empty extraction is reported as *unknown*: a
    parser that quietly stops matching has to make the gate louder, not
    shrink the surface it measures.
    """
    extractor = EXTRACTORS.get(name)
    if extractor is None:
        return None
    service = SERVICES / name
    if not service.exists():
        return None
    return extractor(service, name) or None


def edge_routes() -> tuple[list[Route], dict[str, str]]:
    """(routes reachable through the gateway, prefixes whose routes cannot be read).

    A backend route only counts when the gateway proxies its prefix: the suite
    attacks the deployed edge, and a route behind no proxied prefix is not part
    of that surface.
    """
    prefixes = gateway_prefixes()
    routes: list[Route] = []
    unknown: dict[str, str] = {}
    by_service: dict[str, list[Route] | None] = {}
    for prefix, service in sorted(prefixes.items()):
        if service not in by_service:
            by_service[service] = service_routes(service)
        declared = by_service[service]
        if declared is None:
            unknown[prefix] = service
            continue
        routes.extend(route for route in declared if route.path.startswith(prefix))
    return sorted(set(routes), key=lambda r: (r.path, r.method)), unknown


if __name__ == "__main__":
    from tabulate import tabulate

    inventory, unmapped = edge_routes()
    print(
        tabulate(
            [[route.method, route.path, route.service, route.source] for route in inventory],
            headers=["method", "path", "service", "declared in"],
        )
    )
    if unmapped:
        print(
            "\nProxied prefixes with no route extractor (coverage unmeasured):\n  "
            + "\n  ".join(f"{prefix} ({service})" for prefix, service in sorted(unmapped.items()))
        )
