"""Demo sandbox routes: a visitor opens their own copy of the demo and signs
in to its owner dashboard without e-mail. Mounted on demo deployments only.

The sandbox token is the visitor's only credential. The sign-in route turns
it into the same single-use owner sign-in token an e-mailed link carries,
which the dashboard exchanges at ``/v1/portal/sessions``.
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from gvas.application.owner_assistant import OwnerAssistant
from gvas.domain.identifiers import BusinessId
from gvas.interfaces.demo_sandboxes import (
    DemoSandboxes,
    SandboxAuthenticationError,
    SandboxCapacityError,
)
from gvas.interfaces.http.portal import bearer_token
from gvas.interfaces.http.public import PerIpRateLimiter, client_ip

GENERIC_UNAUTHORIZED = "unauthorized"
OWNER_TEXT_MAX_CHARS = 500
OWNER_TEXT_COUNTERS_MAX = 5000
OWNER_TEXTS_USED_UP_REPLY = (
    "That's all the texts for this demo copy. Start a new demo visit to keep texting Gus."
)
OWNER_TEXT_HISTORY_MAX = 10


class _Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["owner", "assistant"]
    text: str = Field(min_length=1, max_length=1000)


class _OwnerText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=OWNER_TEXT_MAX_CHARS)
    history: list[_Turn] = Field(default_factory=list, max_length=OWNER_TEXT_HISTORY_MAX)


def create_sandbox_router(
    sandboxes: DemoSandboxes,
    *,
    per_ip_per_hour: int,
    rate_limiter: PerIpRateLimiter | None = None,
    owner_assistant: OwnerAssistant | None = None,
    owner_text_limiter: PerIpRateLimiter | None = None,
    owner_texts_per_sandbox: int = 30,
) -> APIRouter:
    router = APIRouter()
    limiter = rate_limiter or PerIpRateLimiter(per_minute=120, burst=30)
    create_limiter = PerIpRateLimiter(per_minute=per_ip_per_hour / 60, burst=per_ip_per_hour)
    # Each owner text is a model call: a much tighter limit than page loads.
    text_limiter = owner_text_limiter or PerIpRateLimiter(per_minute=6, burst=10)
    # And a budget per sandbox copy, like Gus's per-visitor chat cap.
    texts_used: dict[BusinessId, int] = {}

    async def rate_limit(request: Request) -> None:
        if not limiter.allow(client_ip(request)):
            raise HTTPException(status_code=429, detail="rate limited")

    limited = [Depends(rate_limit)]

    @router.post("/v1/demo/sandboxes", dependencies=limited, status_code=201)
    async def create_sandbox(request: Request) -> JSONResponse:
        if not create_limiter.allow(client_ip(request)):
            return JSONResponse({"detail": "rate limited"}, status_code=429)
        try:
            created = await sandboxes.create()
        except SandboxCapacityError:
            return JSONResponse({"detail": "the demo is full right now"}, status_code=503)
        return JSONResponse(
            {
                "publicKey": created.public_key,
                "sandboxToken": created.sandbox_token,
                "idleMinutes": int(sandboxes.idle.total_seconds() // 60),
            },
            status_code=201,
        )

    @router.post("/v1/demo/sandboxes/sign-in", dependencies=limited)
    async def sign_in(request: Request) -> JSONResponse:
        try:
            token = await sandboxes.sign_in(bearer_token(request))
        except SandboxAuthenticationError:
            return JSONResponse({"detail": GENERIC_UNAUTHORIZED}, status_code=401)
        return JSONResponse({"signInToken": token})

    @router.post("/v1/demo/sandboxes/owner-texts", dependencies=limited)
    async def owner_text(body: _OwnerText, request: Request) -> JSONResponse:
        """The demo's in-page Texts view: the owner texts Gus about their own
        sandbox copy. Read only, and nothing is sent anywhere."""

        if owner_assistant is None:
            return JSONResponse({"detail": "texts are off on this demo"}, status_code=503)
        if not text_limiter.allow(client_ip(request)):
            return JSONResponse({"detail": "rate limited"}, status_code=429)
        try:
            business_id = await sandboxes.business_id(bearer_token(request))
        except SandboxAuthenticationError:
            return JSONResponse({"detail": GENERIC_UNAUTHORIZED}, status_code=401)
        if texts_used.get(business_id, 0) >= owner_texts_per_sandbox:
            return JSONResponse({"reply": OWNER_TEXTS_USED_UP_REPLY})
        texts_used[business_id] = texts_used.get(business_id, 0) + 1
        while len(texts_used) > OWNER_TEXT_COUNTERS_MAX:
            # Oldest first: long-expired sandboxes go before live ones.
            texts_used.pop(next(iter(texts_used)))
        reply = await owner_assistant.answer(
            business_id,
            body.text.strip(),
            tuple((turn.role, turn.text) for turn in body.history),
        )
        return JSONResponse({"reply": reply})

    return router
