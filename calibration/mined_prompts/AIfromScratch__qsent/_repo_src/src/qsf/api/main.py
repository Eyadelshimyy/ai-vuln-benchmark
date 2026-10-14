import time
import uuid
from pathlib import Path

import httpx
import structlog
import structlog.contextvars
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from qsf.agents.news_comparison import run_news_comparison, run_news_comparison_stream
from qsf.agents.workflow import pipeline
from qsf.api.auth import COOKIE_NAME, decode_session_token, get_current_user
from qsf.api.auth import router as auth_router
from qsf.common.logging import configure_logging, flush_langfuse, get_langfuse, set_current_trace, get_logger

configure_logging()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

log = get_logger(__name__)

app = FastAPI()


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = uuid.uuid4().hex

        user_id = "anonymous"
        try:
            token = request.cookies.get(COOKIE_NAME)
            if token:
                payload = decode_session_token(token)
                user_id = payload.get("google_sub") or "anonymous"
        except Exception:
            pass

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(trace_id=trace_id, user_id=user_id)

        t0 = time.monotonic()
        log.info("request.received", method=request.method, path=request.url.path)

        response = await call_next(request)

        duration_ms = round((time.monotonic() - t0) * 1000)
        log.info("request.completed", status=response.status_code, duration_ms=duration_ms)

        response.headers["X-Trace-ID"] = trace_id
        return response


app.add_middleware(TraceMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


class AnalyzeRequest(BaseModel):
    ticker: str


class NewsComparisonRequest(BaseModel):
    tickers: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest, user: dict = Depends(get_current_user)):
    ticker = request.ticker.upper().strip()
    trace_id = structlog.contextvars.get_contextvars().get("trace_id", uuid.uuid4().hex)

    lf = get_langfuse()
    if lf:
        trace = lf.trace(
            id=trace_id,
            name="analyze",
            user_id=user.get("google_sub") or "anonymous",
            session_id=ticker,
            input={"ticker": ticker},
        )
        set_current_trace(trace)

    state = pipeline.invoke({"ticker": ticker, "trace_id": trace_id})

    if lf and not state.get("error") and state.get("result"):
        lf.trace(id=trace_id, output={"sentiment_score": state["result"].get("sentiment_score")})

    flush_langfuse()

    if state.get("error"):
        raise HTTPException(
            status_code=404,
            detail={"error": state["error"], "source_status": state.get("source_status", {})},
        )
    return state["result"]


@app.post("/diagnostics/news-comparison")
def news_comparison(request: NewsComparisonRequest, user: dict = Depends(get_current_user)):
    tickers = [t.upper().strip() for t in request.tickers if t.strip()]
    if not tickers:
        raise HTTPException(status_code=422, detail="At least one ticker is required")
    return run_news_comparison(tickers)


@app.get("/diagnostics/news-comparison/stream")
def news_comparison_stream(tickers: list[str] = Query(...)):
    clean = [t.upper().strip() for t in tickers if t.strip()]
    if not clean:
        raise HTTPException(status_code=422, detail="At least one ticker is required")
    return StreamingResponse(
        run_news_comparison_stream(clean),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/auth/avatar")
async def proxy_avatar(user: dict = Depends(get_current_user)):
    picture = user.get("picture", "")
    if not picture:
        raise HTTPException(status_code=404, detail="No avatar")
    async with httpx.AsyncClient() as client:
        r = await client.get(picture, follow_redirects=True, timeout=5)
    if r.status_code != 200:
        raise HTTPException(status_code=404, detail="Avatar unavailable")
    return Response(content=r.content, media_type=r.headers.get("content-type", "image/jpeg"))


@app.get("/login.html")
def serve_login():
    return FileResponse(PROJECT_ROOT / "login.html")


@app.get("/")
def serve_index():
    return FileResponse(PROJECT_ROOT / "index.html")
