"""Perimeter attack cases: reaching a backend service without the gateway.

Every other probe attacks the API gateway, which is what a scanner pointed at
the deployed URL can see. But identity in OtterWorks is a header: the gateway
validates the JWT and forwards ``X-User-ID``, and the backends behind it trust
that header on its own. So the gateway is not merely the front door — it is the
*only* thing authenticating anyone, and any origin that reaches a backend
directly is an unauthenticated impersonation endpoint.

Whether such an origin exists is a deployment fact, not a runtime one: it is in
``docker-compose.yml`` (published container ports) and in the services' Helm
values (``ingress.enabled``). This module reads those, then attacks whatever
they say is reachable — the finding is dynamic, but a crawler starting at the
gateway would never have a URL to crawl to.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml
from httpx import HTTPError

from .base import SEVERITY_ORDER, Evidence, Result, Severity, Verdict, probe, unavailable
from .context import ScanContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from route_inventory import (  # noqa: E402
    UNSAFE_METHODS,
    edge_routes,
    gateway_public_paths,
    may_sweep_unsafely,
    repo_relative,
    sweep_exclusions,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
COMPOSE = REPO_ROOT / "docker-compose.yml"
HELM = REPO_ROOT / "infrastructure" / "helm"

LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"}

#: A per-service route that the gateway protects and that scopes its response to
#: the caller, so serving it to a client-chosen identity is impersonation. Only
#: services that read X-User-ID are worth attacking this way.
HEADER_TRUSTING: dict[str, tuple[str, str, dict[str, str]]] = {
    "file-service": ("GET", "/api/v1/files", {}),
    "search-service": ("GET", "/api/v1/search/", {"q": "otterworks"}),
}


#: Stand-in for a path parameter in a swept route. A random uuid belongs to
#: nobody, so a 2xx for it is an authorization failure and not a real object.
SWEEP_ID = "00000000-0000-4000-8000-000000000000"


def follow_once(ctx: ScanContext, method: str, response: httpx.Response) -> httpx.Response | None:
    """Re-send an anonymous request to where a 3xx pointed, if it stays on the target.

    Only one hop, and only within the target's own origin: chasing a redirect off
    the host under test would attack somewhere nobody authorized, and a chain
    means the route being swept is no longer the route being answered.
    """
    location = response.headers.get("location", "")
    if not location:
        return None
    destination = response.url.join(location)
    if (destination.scheme, destination.host, destination.port) != (
        response.url.scheme,
        response.url.host,
        response.url.port,
    ):
        return None
    # Absolute, so httpx does not merge the target's base path onto a path that
    # already contains it: a path-routed tenant would be sent `/t-abc/t-abc/...`,
    # get a 404, and the route would be graded as refusing anonymous callers.
    return ctx.request(method, str(destination))


def anonymous_by_design(path: str) -> bool | None:
    """Does the gateway's own JWT middleware serve this path without a token?

    The alternative is a hand-written list of public routes, which is a
    suppression that ages: the day the middleware starts protecting a route, the
    list keeps excusing it, and the sweep goes quiet on exactly the regression it
    exists to catch. Returns None when the middleware cannot be read — with no
    idea which routes are meant to answer, every 2xx is equally suspicious and
    equally unprovable.
    """
    declared = gateway_public_paths()
    if declared is None:
        return None
    exact, prefixes = declared
    return path in exact or path.startswith(prefixes)


@dataclass(frozen=True)
class Origin:
    """A backend base URL that is not the gateway, and where it came from."""

    service: str
    url: str
    source: str
    #: A published compose port is a development convenience on a host the developer
    #: already owns; a chart that publishes its own ingress puts the same unguarded
    #: backend on the public internet. Same attack, very different blast radius, so
    #: the severity of a reproduction depends on which origin produced it.
    severity: Severity = Severity.CRITICAL


def published_port(mapping: object) -> tuple[str | None, str] | None:
    """(bound address, host port) for a compose port entry, or None if there is none.

    Compose accepts `CONTAINER`, `HOST:CONTAINER`, `IP:HOST:CONTAINER` and the long
    mapping form, and this is the only input to a security verdict: reading
    `IP:HOST:CONTAINER` as unpublished would report a reachable backend `SECURE`,
    and reading a bare `CONTAINER` port as a host port would invent an origin that
    is never reachable and leave the probe inconclusive forever.
    """
    if isinstance(mapping, dict):
        published = mapping.get("published")
        return (mapping.get("host_ip"), str(published)) if published else None
    fields = str(mapping).split(":")
    if len(fields) == 3:
        address, host_port = fields[0].strip(), fields[1].strip()
    elif len(fields) == 2:
        address, host_port = None, fields[0].strip()
    else:
        # A container port on its own is published on an ephemeral host port, so
        # there is no address to attack.
        return None
    # A range (`8080-8090`) publishes several; the first is enough to prove reach.
    host_port = host_port.split("-")[0]
    return (address, host_port) if host_port.isdigit() else None


def compose_origins(target: str, compose: Path = COMPOSE) -> list[Origin]:
    """Backends published on the host by docker compose, for a local target.

    Only meaningful when the target is the local stack: a published port is a
    hole in the same host the gateway is on.
    """
    host = urlparse(target).hostname or ""
    if host not in LOCAL_HOSTS or not compose.exists():
        return []
    spec = yaml.safe_load(compose.read_text()) or {}
    origins = []
    for service, definition in (spec.get("services") or {}).items():
        if service not in HEADER_TRUSTING or not isinstance(definition, dict):
            continue
        for mapping in definition.get("ports") or []:
            entry = published_port(mapping)
            if entry is None:
                continue
            address, port = entry
            origins.append(
                Origin(
                    service,
                    # A port bound to one interface is only reachable there, so the
                    # origin has to use that address rather than the target's host.
                    f"http://{address or host}:{port}",
                    f"{compose.name} ports",
                    severity=Severity.LOW,
                )
            )
            break
    return origins


def in_scope(declared: str, target_host: str) -> bool:
    """Is a chart-declared host covered by the target the operator named?

    Only the target itself and hosts beneath it. Anything wider means deciding
    what counts as "the same site", which needs the public suffix list —
    approximating it by counting labels makes every host under a multi-label
    suffix (``example.co.uk``, ``eu-west-1.amazonaws.com``) a sibling of every
    other, i.e. permission to attack strangers. A chart host outside the target
    has to be named by the operator, via ``DAST_ALLOW_ORIGIN_HOSTS``.
    """
    declared, target_host = declared.lower().rstrip("."), target_host.lower().rstrip(".")
    if declared == target_host or declared.endswith(f".{target_host}"):
        return True
    allowed = {h.strip().lower() for h in os.getenv("DAST_ALLOW_ORIGIN_HOSTS", "").split(",")}
    return declared in allowed - {""}


def declared_ingress_hosts(helm: Path = HELM) -> list[str]:
    """Every backend host the charts publish, whatever the scan is aimed at.

    Kept separate from :func:`ingress_origins` so that a chart origin dropped for
    being outside the target is still visible to the probe: not attacking a host
    is a reason to withhold a verdict, never a reason to issue a clean one.
    """
    hosts = []
    for values in sorted(helm.glob("*/values.yaml")):
        if values.parent.name not in HEADER_TRUSTING:
            continue
        spec = yaml.safe_load(values.read_text()) or {}
        ingress = spec.get("ingress") or {}
        if not ingress.get("enabled"):
            continue
        hosts += [
            (entry or {}).get("host")
            for entry in ingress.get("hosts") or []
            if (entry or {}).get("host")
        ]
    return hosts


def ingress_origins(target: str, helm: Path = HELM) -> list[Origin]:
    """Backends whose chart publishes its own ingress hostname, under the target.

    The tenant deploy disables these, but the chart defaults do not: a plain
    ``helm install`` of a backend puts it on the public ingress controller,
    beside the gateway rather than behind it.

    Authorization to scan is authorization to scan *something*, so a chart host
    is only attacked when the target covers it (see :func:`in_scope`): a run
    aimed at localhost or at one tenant never sends traffic to whatever host a
    chart happens to mention.
    """
    host = urlparse(target).hostname or ""
    if host in LOCAL_HOSTS or not host:
        return []
    origins = []
    for values in sorted(helm.glob("*/values.yaml")):
        service = values.parent.name
        if service not in HEADER_TRUSTING:
            continue
        spec = yaml.safe_load(values.read_text()) or {}
        ingress = spec.get("ingress") or {}
        if not ingress.get("enabled"):
            continue
        for entry in ingress.get("hosts") or []:
            declared = (entry or {}).get("host")
            if declared and in_scope(declared, host):
                origins.append(
                    Origin(
                        service,
                        f"https://{declared}",
                        f"{repo_relative(values)} ingress.enabled",
                    )
                )
    return origins


def reachable(origin: Origin, timeout: float = 3.0) -> bool:
    try:
        return httpx.get(f"{origin.url}/health", timeout=timeout).status_code < 500
    except httpx.HTTPError:
        return False


@probe(
    finding_id="DAST-GATEWAY-BYPASS-IDENTITY",
    title="A backend is reachable outside the gateway and trusts X-User-ID",
    severity=Severity.CRITICAL,
    owasp="API8:2023 Security Misconfiguration",
    cwe="CWE-290",
    service="infrastructure",
    remediation=(
        "Keep backend services ClusterIP-only behind the gateway (ingress.enabled=false in "
        "each backend chart, no published ports on any shared host), and have services "
        "authenticate the request themselves rather than trusting a forwarded X-User-ID."
    ),
)
def gateway_bypass_identity(ctx: ScanContext) -> Result:
    """Assert the victim's identity straight at a backend, with no token at all."""
    self = gateway_bypass_identity.probe
    origins = compose_origins(ctx.base_url) + ingress_origins(ctx.base_url)
    if not origins:
        target_host = urlparse(ctx.base_url).hostname or ""
        elsewhere = [h for h in declared_ingress_hosts() if not in_scope(h, target_host)]
        if elsewhere:
            # The charts do publish backends; this run just is not allowed to touch
            # the hosts they name. Untested is not the same as not exposed.
            return self.result(
                Verdict.INCONCLUSIVE,
                "the service charts publish backend ingress hosts, but none of them is under "
                f"the scan target {target_host}, so none was attacked: "
                + ", ".join(sorted(set(elsewhere)))
                + ". Set DAST_ALLOW_ORIGIN_HOSTS to the ones you are authorized to scan.",
            )
        return self.result(
            Verdict.SECURE,
            "no backend origin is published outside the gateway by the compose file or the "
            "service charts, so there is nowhere to bypass it",
        )

    live = [origin for origin in origins if reachable(origin)]
    declared = ", ".join(f"{o.service} ({o.source})" for o in origins)
    if not live:
        # Declared and unreachable is not the same as not declared: the origin may
        # simply be unroutable from this host, so the perimeter is untested rather
        # than proven.
        return self.result(
            Verdict.INCONCLUSIVE,
            f"backend origins are declared but none answered from here: {declared}",
        )

    evidence: list[Evidence] = []
    inconclusive: list[str] = []
    reproduced: list[Result] = []
    for origin in live:
        method, path, params = HEADER_TRUSTING[origin.service]
        url = f"{origin.url}{path}"
        # Control: the gateway must refuse this route without a token, or it is a
        # public endpoint and reaching it directly proves nothing.
        through_gateway = ctx.request(method, path, params=params)
        if through_gateway.status_code == 200:
            # No bypass to demonstrate with this route, and no attack made either:
            # the probe learned nothing about this origin.
            inconclusive.append(
                f"{origin.service}: the gateway serves {path} unauthenticated, so reaching "
                "it directly would prove nothing and the backend was not attacked"
            )
            evidence.append(
                Evidence.from_response(
                    through_gateway,
                    note=f"{path} is served unauthenticated by the gateway itself",
                )
            )
            continue
        if unavailable(through_gateway):
            inconclusive.append(
                f"{origin.service}: the gateway returned {through_gateway.status_code} for "
                f"{path}, so there is no protected baseline to compare against"
            )
            continue
        try:
            spoofed = httpx.request(
                method,
                url,
                params=params,
                headers={"X-User-ID": ctx.victim.user_id},
                timeout=10.0,
            )
            anonymous = httpx.request(method, url, params=params, timeout=10.0)
        except httpx.HTTPError as exc:
            inconclusive.append(f"{origin.service}: {type(exc).__name__} {exc}")
            continue
        if unavailable(spoofed) or unavailable(anonymous):
            inconclusive.append(
                f"{origin.service}: direct requests returned "
                f"{spoofed.status_code}/{anonymous.status_code}"
            )
            continue
        # Either one answering is the bypass. A backend that rejects an identity it
        # cannot resolve but serves a caller who claims none is exposed just as
        # thoroughly, and reading only the spoofed request would call it protected.
        if spoofed.status_code != 200 and anonymous.status_code != 200:
            evidence.append(
                Evidence.from_response(
                    spoofed,
                    note=f"{origin.service} direct via {origin.source}: refused the header",
                )
            )
            continue
        # Both are the same bypass; which one it is decides where the fix goes.
        if anonymous.status_code == 200:
            cause = (
                "and requires no identity at all: the gateway is the only thing "
                "authenticating this route"
            )
        else:
            cause = (
                f"and served it to a client-chosen X-User-ID with no token (the same request "
                f"without the header: {anonymous.status_code}), so the caller picks who they are"
            )
        result = self.result(
            Verdict.VULNERABLE,
            f"{origin.service} answers {method} {path} at {origin.url}, outside the gateway "
            f"— which refuses the same request with {through_gateway.status_code} — "
            f"{cause}; the origin exists because of {origin.source}. Declared backend "
            f"origins: {declared}",
            [
                Evidence.from_response(
                    through_gateway, note="same route through the gateway, unauthenticated"
                ),
                Evidence.from_response(
                    spoofed,
                    note=f"direct to the backend, X-User-ID: {ctx.victim.user_id}, no token",
                ),
                Evidence.from_response(anonymous, note="direct to the backend, no header"),
            ],
        )
        result.severity = origin.severity
        reproduced.append(result)

    if reproduced:
        # Report the worst origin: a publicly published chart outranks a local port.
        return max(reproduced, key=lambda r: SEVERITY_ORDER[r.severity])

    if inconclusive:
        return self.result(
            Verdict.INCONCLUSIVE,
            "a backend origin was reachable but the bypass could not be assessed: "
            + "; ".join(inconclusive),
            evidence,
        )
    return self.result(
        Verdict.SECURE,
        f"backend origins are reachable ({declared}) and every one of them refused a "
        "gateway-protected route sent directly",
        evidence,
    )


@probe(
    finding_id="DAST-ANONYMOUS-ROUTE-SWEEP",
    title="A gateway-proxied route answers a caller with no token",
    severity=Severity.HIGH,
    owasp="API2:2023 Broken Authentication",
    cwe="CWE-306",
    service="api-gateway",
    remediation=(
        "Require a validated token on every proxied route except the declared public "
        "ones, and reject the request at the gateway rather than in each service."
    ),
    requires_identity=False,
)
def anonymous_route_sweep(ctx: ScanContext) -> Result:
    """Send every route the source declares, with no credentials at all.

    The hand-written probes attack the endpoints someone thought to attack. This
    one attacks the endpoints that *exist*: the inventory comes from the services'
    route definitions, so a route added tomorrow is swept the day it lands, with
    no crawler having to find a link to it.
    """
    self = anonymous_route_sweep.probe
    routes, unknown = edge_routes()
    if not routes:
        return self.result(
            Verdict.INCONCLUSIVE, "no routes could be read from the services' source"
        )

    if anonymous_by_design("/") is None:
        return self.result(
            Verdict.INCONCLUSIVE,
            "the gateway's JWT middleware could not be read, so which routes are meant "
            "to answer anonymously is unknown and no response can be judged",
        )

    excluded = sweep_exclusions()
    unsafe_allowed = may_sweep_unsafely()
    withheld = 0
    served: list[Evidence] = []
    swept = 0
    unreachable: list[str] = []
    for route in routes:
        if route.key in excluded:
            # Sending this one would perform it. The coverage gate reports the route
            # as unswept with the reason, so the hole stays visible.
            continue
        if route.method in UNSAFE_METHODS and not unsafe_allowed:
            withheld += 1
            continue
        path = route.path.replace("{}", SWEEP_ID)
        try:
            response = ctx.request(route.method, path)
            if response.is_redirect:
                # The inventory holds paths without a trailing slash, and a framework
                # that mounts the route with one answers 3xx without running the
                # handler. A redirect is neither a refusal nor an answer: follow it
                # once, or the route counts as swept having never been delivered.
                response = follow_once(ctx, route.method, response)
        except HTTPError as exc:
            unreachable.append(f"{route.key}: {type(exc).__name__}")
            continue
        if response is None or response.is_redirect:
            unreachable.append(f"{route.key}: redirected away from the route")
            continue
        if unavailable(response):
            unreachable.append(f"{route.key}: {response.status_code}")
            continue
        swept += 1
        # The declared-public routes are still requested, so the coverage report can
        # see them; only an unexpected 2xx on a protected route is a finding.
        if response.is_success and not anonymous_by_design(route.path):
            served.append(
                Evidence.from_response(
                    response,
                    note=f"{route.key} declared in {route.source}, requested with no token",
                )
            )
    scope = f"{swept} route(s) read from source"
    if withheld:
        scope += (
            f"; {withheld} route(s) were not sent because their method would write to a "
            "target that has not been declared disposable (DAST_SWEEP_UNSAFE_METHODS)"
        )
    if excluded:
        scope += (
            f"; {len(excluded)} route(s) are excluded from the sweep because sending them "
            "would carry out a tenant-wide operation"
        )
    if unknown:
        scope += (
            f"; {len(unknown)} proxied prefix(es) have no route extractor and were not swept: "
            + ", ".join(sorted(unknown))
        )
    if served:
        return self.result(
            Verdict.VULNERABLE,
            f"{len(served)} of {scope} answered an unauthenticated request",
            served[:5],
        )
    if not swept:
        return self.result(
            Verdict.INCONCLUSIVE,
            "every route was unavailable, so nothing was assessed: " + "; ".join(unreachable[:5]),
        )
    if unreachable:
        # Reported, not fatal: the routes that did answer still refused, and the
        # ones that did not are named so the gap is visible rather than implied.
        scope += f"; {len(unreachable)} route(s) were unavailable: " + "; ".join(unreachable[:5])
    return self.result(Verdict.SECURE, f"every one of {scope} refused an anonymous caller")
