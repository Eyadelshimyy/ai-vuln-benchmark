import logging
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from x402 import x402ResourceServer
from x402.http import FacilitatorConfig, HTTPFacilitatorClient
from x402.http.middleware.fastapi import payment_middleware
from x402.http.types import PaymentOption, RouteConfig
from x402.mechanisms.evm.exact import ExactEvmServerScheme

from app.api.routes import router
from app.config import settings
from app.security import (
    audit,
    body_size_limit_middleware,
    https_redirect_middleware,
    install_log_filter,
    limiter,
    record_payment_failure,
    request_id_middleware,
    security_headers_middleware,
    validate_secrets,
)
from app.services import cache, db

install_log_filter()

# Better Stack / Logtail structured log streaming (active: 2024-08-20)
if settings.logtail_token:
    from logtail import LogtailHandler
    _logtail_handler = LogtailHandler(source_token=settings.logtail_token)
    _logtail_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(_logtail_handler)
    logging.getLogger(__name__).info("Logtail logging active")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_secrets()
    await db.init_db()
    await cache.connect()
    yield
    await cache.disconnect()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

# CORS — explicit allowlist only; empty string blocks all cross-origin requests
_cors_origins = [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID", "PAYMENT-SIGNATURE"],
)

# Rate limiter
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Please slow down."})


app.include_router(router)


def _build_cdp_facilitator_config() -> dict[str, object]:
    if not settings.cdp_api_key_id or not settings.cdp_api_key_secret:
        raise ValueError(
            "CDP_API_KEY_ID and CDP_API_KEY_SECRET are required for CDP x402 facilitator."
        )

    try:
        from cdp.auth.utils.jwt import JwtOptions, generate_jwt
    except ImportError as exc:
        raise ValueError(
            "cdp-sdk is required to authenticate with CDP x402 facilitator."
        ) from exc

    parsed = urlparse(settings.x402_facilitator_url)
    request_host = parsed.netloc
    base_path = parsed.path.rstrip("/")
    if not request_host or not base_path:
        raise ValueError("X402_FACILITATOR_URL must include host and path.")

    key_secret = settings.cdp_api_key_secret.replace("\\n", "\n")

    def _create_headers() -> dict[str, dict[str, str]]:
        return {
            "supported": {"Authorization": f"Bearer {generate_jwt(JwtOptions(api_key_id=settings.cdp_api_key_id, api_key_secret=key_secret, request_method='GET', request_host=request_host, request_path=f'{base_path}/supported', expires_in=120))}"},
            "verify":    {"Authorization": f"Bearer {generate_jwt(JwtOptions(api_key_id=settings.cdp_api_key_id, api_key_secret=key_secret, request_method='POST', request_host=request_host, request_path=f'{base_path}/verify', expires_in=120))}"},
            "settle":    {"Authorization": f"Bearer {generate_jwt(JwtOptions(api_key_id=settings.cdp_api_key_id, api_key_secret=key_secret, request_method='POST', request_host=request_host, request_path=f'{base_path}/settle', expires_in=120))}"},
        }

    return {"url": settings.x402_facilitator_url, "create_headers": _create_headers}


if settings.x402_enabled:
    if not settings.x402_pay_to:
        raise ValueError("X402_PAY_TO is required when X402_ENABLED=true")

    # Patch Base mainnet USDC to use Permit2 (Dexter facilitator migrated from EIP-3009)
    try:
        from x402.mechanisms.evm.constants import NETWORK_CONFIGS
        if settings.x402_network in NETWORK_CONFIGS:
            asset = NETWORK_CONFIGS[settings.x402_network].get("default_asset")
            if asset:
                if "asset_transfer_method" not in asset:
                    asset["asset_transfer_method"] = "permit2"
                if asset.get("name") == "USD Coin":
                    asset["name"] = "USDC"
    except Exception:
        pass  # Non-fatal: falls back to EIP-3009 which may still work

    # Explicit extra — leave empty so facilitator negotiates the transfer method
    _PERMIT2_EXTRA = None

    if "api.cdp.coinbase.com" in settings.x402_facilitator_url:
        facilitator = HTTPFacilitatorClient(_build_cdp_facilitator_config())
    else:
        facilitator = HTTPFacilitatorClient(
            FacilitatorConfig(url=settings.x402_facilitator_url)
        )
    x402_server = x402ResourceServer(facilitator).register(
        settings.x402_network, ExactEvmServerScheme()
    )

    _format_prices = {
        "svg": settings.x402_price_svg,
        "png": settings.x402_price_png,
    }

    x402_routes = {
        f"POST /paid/diagrams/generate/{fmt}": RouteConfig(
            accepts=PaymentOption(
                scheme="exact",
                network=settings.x402_network,
                pay_to=settings.x402_pay_to,
                price=price,
                extra=_PERMIT2_EXTRA if _PERMIT2_EXTRA else None,
            ),
            description=f"Generate and store a diagram ({fmt} format)",
            mime_type="application/json",
        )
        for fmt, price in _format_prices.items()
    }

    # Add edit endpoints (same pricing as generate)
    x402_routes.update({
        f"POST /paid/diagrams/edit/{fmt}": RouteConfig(
            accepts=PaymentOption(
                scheme="exact",
                network=settings.x402_network,
                pay_to=settings.x402_pay_to,
                price=price,
                extra=_PERMIT2_EXTRA if _PERMIT2_EXTRA else None,
            ),
            description=f"Edit and render diagram ({fmt} format)",
            mime_type="application/json",
        )
        for fmt, price in _format_prices.items()
    })

    _x402_middleware = payment_middleware(x402_routes, x402_server)

    _PAID_PATHS = {
        f"/paid/diagrams/generate/{fmt}" for fmt in _format_prices
    } | {
        f"/paid/diagrams/edit/{fmt}" for fmt in _format_prices
    }

    @app.middleware("http")
    async def x402_http_middleware(request: Request, call_next):
        client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown").split(",")[0].strip()

        # Only run x402 logic on paid paths; all other routes go directly to call_next
        if not (request.method == "POST" and request.url.path in _PAID_PATHS):
            return await call_next(request)

        # Idempotency: if client sends X-Idempotency-Key, return cached result for duplicates
        idem_key = request.headers.get("x-idempotency-key")
        if idem_key:
            from app.services.cache import get_cached
            cached = await get_cached("idem", {"key": idem_key, "path": request.url.path})
            if cached:
                import json
                from fastapi.responses import JSONResponse as _JSONResponse
                audit("payment.idempotent_replay", request, idem_key=idem_key[:16])
                return _JSONResponse(content=json.loads(cached))

        # Rate limit: 20 requests per IP per 60 seconds
        from app.services.cache import increment_counter
        count = await increment_counter(f"ratelimit:paid:{client_ip}", 60)
        if count > 20:
            audit("rate_limit.exceeded", request)
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Please slow down."})

        response = await _x402_middleware(request, call_next)

        if response.status_code == 402:
            record_payment_failure(client_ip, request.url.path)
            audit("payment.challenge_issued", request)
        elif response.status_code == 200:
            audit("payment.verified_success", request)

        return response


# Security + utility middleware — registered last so they run outermost (first in, last out)
@app.middleware("http")
async def _security_headers(request: Request, call_next):
    return await security_headers_middleware(request, call_next)

@app.middleware("http")
async def _body_size(request: Request, call_next):
    return await body_size_limit_middleware(request, call_next)

@app.middleware("http")
async def _request_id(request: Request, call_next):
    return await request_id_middleware(request, call_next)

@app.middleware("http")
async def _https(request: Request, call_next):
    return await https_redirect_middleware(request, call_next)
