from collections.abc import Awaitable, Callable

from fastapi import APIRouter, FastAPI

from gvas.config import Settings
from gvas.interfaces.http.public import SiteOriginCorsMiddleware


def create_app(
    settings: Settings | None = None,
    routers: tuple[APIRouter, ...] = (),
    *,
    cors_origins: Callable[[], Awaitable[frozenset[str]]] | None = None,
) -> FastAPI:
    app = FastAPI(title="Güd Vector Agent Suite")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    for router in routers:
        app.include_router(router)
    if cors_origins is not None:
        app.add_middleware(SiteOriginCorsMiddleware, origins=cors_origins)
    return app


app = create_app()
