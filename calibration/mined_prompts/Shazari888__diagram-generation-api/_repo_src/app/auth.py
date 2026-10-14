import logging

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import settings

log = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> str:
    if not api_key or api_key != settings.api_key:
        log.warning("auth.failure ip=%s path=%s", request.headers.get("x-forwarded-for", "?"), request.url.path)
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key
