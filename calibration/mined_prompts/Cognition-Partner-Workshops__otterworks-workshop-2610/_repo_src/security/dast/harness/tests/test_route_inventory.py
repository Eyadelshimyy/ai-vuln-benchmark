"""The route inventory is read from source, so its parsers are the thing to test.

These build tiny service trees rather than asserting against the real ones: a
test that pins the live route list fails every time somebody adds an endpoint,
which is the opposite of what the coverage gate is for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import route_inventory  # noqa: E402
from route_inventory import (  # noqa: E402
    EXTRACTORS,
    Route,
    _normalize,
    edge_routes,
    gateway_prefixes,
    gateway_public_paths,
)


def declared(routes: list[Route], filename: str) -> list[tuple[str, str]]:
    """The (method, path) pairs, having checked each one names its source file."""
    assert all(route.source.endswith(filename) for route in routes), routes
    return [(route.method, route.path) for route in routes]


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_gateway_prefixes_map_to_service_directories(tmp_path: Path) -> None:
    config = write(
        tmp_path / "config.go",
        """
func (c *Config) ServiceRoutes() map[string]string {
	return map[string]string{
		"/api/v1/auth":      c.AuthServiceURL,
		"/api/v1/documents": c.DocumentServiceURL,
		"/socket.io":        c.CollabServiceURL,
	}
}
""",
    )
    assert gateway_prefixes(config) == {
        "/api/v1/auth": "auth-service",
        "/api/v1/documents": "document-service",
        "/socket.io": "collab-service",
    }


def test_gateway_prefixes_are_empty_when_the_config_is_missing(tmp_path: Path) -> None:
    assert gateway_prefixes(tmp_path / "nope.go") == {}


def test_fastapi_routes_take_the_prefix_from_the_router_mount(tmp_path: Path) -> None:
    service = tmp_path / "document-service"
    write(
        service / "app" / "main.py",
        'app.include_router(documents.router, prefix="/api/v1/documents")\n',
    )
    write(
        service / "app" / "api" / "documents.py",
        '@router.get("/")\n'
        "async def list_documents(): ...\n"
        '@router.patch("/{document_id}")\n'
        "async def update(document_id: str): ...\n",
    )
    routes = EXTRACTORS["document-service"](service, "document-service")
    assert declared(routes, "api/documents.py") == [
        ("GET", "/api/v1/documents"),
        ("PATCH", "/api/v1/documents/{}"),
    ]


def test_flask_routes_expand_every_declared_method(tmp_path: Path) -> None:
    service = tmp_path / "search-service"
    write(
        service / "app" / "main.py",
        'app.register_blueprint(search_bp, url_prefix="/api/v1/search")\n',
    )
    write(
        service / "app" / "api" / "search.py",
        "search_bp = Blueprint('search', __name__)\n"
        '@search_bp.route("/index/<doc_type>/<doc_id>", methods=["DELETE"])\n'
        "def drop(doc_type, doc_id): ...\n",
    )
    routes = EXTRACTORS["search-service"](service, "search-service")
    assert declared(routes, "api/search.py") == [("DELETE", "/api/v1/search/index/{}/{}")]


def test_actix_routes_are_read_from_the_scope(tmp_path: Path) -> None:
    service = tmp_path / "file-service"
    write(
        service / "src" / "main.rs",
        """
web::scope("/api/v1/files")
    .route("", web::get().to(handlers::list_files))
    .route("/{id}/share", web::post().to(handlers::share))
""",
    )
    routes = EXTRACTORS["file-service"](service, "file-service")
    assert declared(routes, "src/main.rs") == [
        ("GET", "/api/v1/files"),
        ("POST", "/api/v1/files/{}/share"),
    ]


def test_spring_routes_combine_the_class_and_method_mappings(tmp_path: Path) -> None:
    service = tmp_path / "auth-service"
    write(
        service / "src/main/java/com/otterworks/auth/controller/AuthController.java",
        """
@RestController
@RequestMapping("/api/v1/auth")
public class AuthController {
    @PostMapping("/login")
    public ResponseEntity<?> login() { return null; }

    @GetMapping("/users/by-id/{userId}")
    public ResponseEntity<?> byId(@PathVariable String userId) { return null; }
}
""",
    )
    routes = EXTRACTORS["auth-service"](service, "auth-service")
    assert declared(routes, "AuthController.java") == [
        ("POST", "/api/v1/auth/login"),
        ("GET", "/api/v1/auth/users/by-id/{}"),
    ]


def test_ktor_routes_nest_through_the_routing_dsl(tmp_path: Path) -> None:
    service = tmp_path / "notification-service"
    write(
        service / "src/main/kotlin/com/otterworks/notification/routes/Routes.kt",
        """
routing {
    get("/health") { call.respond(HealthResponse()) }

    route("/api/v1/notifications") {
        get {
            call.respond(notificationService.list())
        }
        get("/unread-count") { call.respond(count) }
        put("/{id}/read") { call.respond(HttpStatusCode.NoContent) }
    }

    route("/api/v1/preferences") {
        get { call.respond(prefs) }
    }
}
""",
    )
    routes = EXTRACTORS["notification-service"](service, "notification-service")
    assert declared(routes, "Routes.kt") == [
        ("GET", "/health"),
        ("GET", "/api/v1/notifications"),
        ("GET", "/api/v1/notifications/unread-count"),
        # `{id}` lives inside a string literal, so it must not be counted as a
        # block opening — the routes after it would otherwise nest one too deep.
        ("PUT", "/api/v1/notifications/{}/read"),
        ("GET", "/api/v1/preferences"),
    ]


def test_a_service_whose_extractor_matches_nothing_is_unknown_not_covered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The parser pointed at the wrong framework must make the gate louder, not quieter."""
    service = tmp_path / "notification-service"
    write(service / "src" / "main" / "kotlin" / "Routes.kt", "routing { }")
    monkeypatch.setattr(route_inventory, "SERVICES", tmp_path)
    monkeypatch.setitem(route_inventory.EXTRACTORS, "notification-service", route_inventory._spring)

    assert route_inventory.service_routes("notification-service") is None


def test_path_parameters_normalize_across_frameworks() -> None:
    assert _normalize("/api/v1/documents/{document_id}") == "/api/v1/documents/{}"
    assert _normalize("/api/v1/search/<doc_type>/<doc_id>") == "/api/v1/search/{}/{}"
    assert _normalize("/api/v1/admin/users/:id") == "/api/v1/admin/users/{}"
    assert _normalize("/api/v1/files/") == "/api/v1/files"
    assert _normalize("/") == "/"


def test_the_real_inventory_only_reports_proxied_routes() -> None:
    routes, unknown = edge_routes()
    prefixes = tuple(gateway_prefixes())
    assert routes, "the repository's own services should yield routes"
    assert all(route.path.startswith(prefixes) for route in routes)
    # A service with no extractor must surface as unknown rather than as covered:
    # silently reporting an unparsed service as fully attacked is the failure this
    # module exists to prevent.
    assert set(unknown) <= set(prefixes)
    assert not set(unknown) & {route.path for route in routes}


def test_the_public_paths_come_from_the_gateway_middleware(tmp_path: Path) -> None:
    """A hand-kept list of public routes excuses them long after they stop being public."""
    jwt = write(
        tmp_path / "jwt.go",
        """
func DefaultPublicPaths() []string {
	return []string{
		"/api/v1/auth/login",
		"/api/v1/auth/register",
	}
}

func DefaultPrefixPaths() []string {
	return []string{
		"/health",
		"/metrics",
	}
}
""",
    )
    assert gateway_public_paths(jwt) == (
        {"/api/v1/auth/login", "/api/v1/auth/register"},
        ("/health", "/metrics"),
    )


def test_an_unreadable_middleware_is_not_an_empty_public_list(tmp_path: Path) -> None:
    """None makes the sweep withhold a verdict; an empty set would invent findings."""
    assert gateway_public_paths(tmp_path / "absent.go") is None
    assert gateway_public_paths(write(tmp_path / "other.go", "package middleware")) is None


def test_the_repository_gateway_serves_only_login_and_register_anonymously() -> None:
    exact, prefixes = gateway_public_paths()
    assert exact == {"/api/v1/auth/login", "/api/v1/auth/register"}
    assert "/api/v1/auth/refresh" not in exact
    assert "/metrics" in prefixes
