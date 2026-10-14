import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    DiagramResponse,
    GenerateDiagramRequest,
    GenerateDiagramResponse,
    HealthResponse,
)
from app.api.edit_schemas import (
    EditDiagramRequest,
    EditDiagramResponse,
)
from app.services.edit_engine import DiagramEditEngine, EditEngineException
from app.auth import verify_api_key
from app.config import settings
from app.security import audit, check_admin_ip, limiter
from app.services import cache, db, llm, renderer

log = logging.getLogger(__name__)

router = APIRouter()


def _base_meta_tag() -> str:
    if not settings.base_app_id:
        return ""
    return f'<meta name="base:app_id" content="{settings.base_app_id}" />'


@router.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    return HTMLResponse(
        content=f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  {_base_meta_tag()}
  <title>Diagram Generation API</title>
</head>
<body>
  <h1>Diagram Generation API</h1>
  <p>Service is running.</p>
</body>
</html>"""
    )


VISUALIZER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Diagram API Visualizer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; color: #111; }
    .row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
    input, select, textarea, button { font-size: 14px; padding: 8px; }
    textarea { width: 100%; min-height: 90px; }
    #preview { border: 1px solid #ddd; border-radius: 6px; padding: 12px; min-height: 200px; background: #fafafa; }
    #status { margin: 10px 0; white-space: pre-wrap; }
    #source { margin-top: 12px; width: 100%; min-height: 120px; font-family: Consolas, monospace; }
    .hint { margin: 8px 0 12px; padding: 10px; border-radius: 6px; background: #eef6ff; border: 1px solid #c8ddff; }
    .hint a { margin-right: 12px; }
    iframe, img, svg { max-width: 100%; }
  </style>
</head>
<body>
  <h2>Diagram API Visualizer</h2>
  <div class="row">
    <input id="apiKey" placeholder="X-API-Key" style="min-width: 280px;" />
    <select id="diagramType">
      <option value="mermaid">mermaid</option>
      <option value="d2">d2</option>
      <option value="plantuml">plantuml</option>
      <option value="graphviz">graphviz</option>
    </select>
    <select id="format">
      <option value="svg">svg</option>
      <option value="png">png</option>
      <option value="pdf">pdf</option>
    </select>
    <button id="generateBtn">Generate</button>
  </div>
  <textarea id="prompt">Create a login flow with success and retry paths.</textarea>
  <div id="status"></div>
  <div id="preview"></div>
  <textarea id="source" readonly placeholder="Generated diagram source appears here"></textarea>

  <script>
    const statusEl = document.getElementById("status");
    const previewEl = document.getElementById("preview");
    const sourceEl = document.getElementById("source");
    let currentPdfObjectUrl = null;

    function base64ToBlobUrl(base64Data, mimeType) {
      const raw = atob(base64Data);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) {
        bytes[i] = raw.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: mimeType });
      return URL.createObjectURL(blob);
    }

    function setStatus(text, isError) {
      statusEl.style.color = isError ? "#b00020" : "#0b5";
      statusEl.textContent = text;
    }

    async function generate() {
      setStatus("Generating...", false);
      previewEl.innerHTML = "";
      sourceEl.value = "";
      if (currentPdfObjectUrl) {
        URL.revokeObjectURL(currentPdfObjectUrl);
        currentPdfObjectUrl = null;
      }

      const body = {
        prompt: document.getElementById("prompt").value,
        diagram_type: document.getElementById("diagramType").value,
        format: document.getElementById("format").value,
      };

      try {
        const res = await fetch("/diagrams/generate", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": document.getElementById("apiKey").value,
          },
          body: JSON.stringify(body),
        });

        const raw = await res.text();
        let data = null;
        try {
          data = JSON.parse(raw);
        } catch {
          data = null;
        }

        if (!res.ok) {
          const errorText = (data && data.detail)
            ? data.detail
            : (raw || ("HTTP " + res.status));
          throw new Error(errorText);
        }

        if (!data) {
          throw new Error("Server returned a non-JSON success response.");
        }

        sourceEl.value = data.diagram.source || "";

        if (data.format === "svg") {
          previewEl.innerHTML = data.rendered;
        } else if (data.format === "png") {
          previewEl.innerHTML = '<img alt="PNG diagram preview" src="data:image/png;base64,' + data.rendered + '" />';
        } else if (data.format === "pdf") {
          const pdfUrl = base64ToBlobUrl(data.rendered, "application/pdf");
          currentPdfObjectUrl = pdfUrl;
          previewEl.innerHTML =
            '<div class="hint">' +
            'PDF generated successfully. If preview is blank in this embedded browser, use this link: ' +
            '<a href="' + pdfUrl + '" download="diagram.pdf">Download PDF</a>' +
            '</div>' +
            '<iframe title="PDF diagram preview" style="width:100%;height:700px;border:0;" src="' + pdfUrl + '"></iframe>';
        }

        setStatus("Success: " + data.diagram.id, false);
      } catch (err) {
        setStatus("Error: " + err.message, true);
      }
    }

    document.getElementById("generateBtn").addEventListener("click", generate);
  </script>
</body>
</html>
"""


@router.get("/visualizer", response_class=HTMLResponse)
async def visualizer() -> HTMLResponse:
    return HTMLResponse(content=VISUALIZER_HTML)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


async def _generate_diagram(
    body: GenerateDiagramRequest,
    session: AsyncSession,
) -> GenerateDiagramResponse:
    cache_payload = {
        "prompt": body.prompt,
        "diagram_type": body.diagram_type,
        "format": body.format,
    }

    cached = await cache.get_cached("diagram", cache_payload)
    if cached:
        data = json.loads(cached)
        diagram = await db.get_diagram(session, UUID(data["id"]))
        if diagram:
            return GenerateDiagramResponse(
                diagram=DiagramResponse.model_validate(diagram),
                rendered=data["rendered"],
                format=body.format,
            )

    source = await llm.generate_source(body.prompt, body.diagram_type)
    try:
        rendered = await renderer.render(source, body.diagram_type, body.format)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    diagram = await db.create_diagram(
        session,
        prompt=body.prompt,
        source=source,
        diagram_type=body.diagram_type,
    )

    await cache.set_cached(
        "diagram",
        cache_payload,
        json.dumps({"id": str(diagram.id), "rendered": rendered}),
    )

    return GenerateDiagramResponse(
        diagram=DiagramResponse.model_validate(diagram),
        rendered=rendered,
        format=body.format,
    )


@router.post(
    "/diagrams/generate",
    response_model=GenerateDiagramResponse,
    include_in_schema=False,
)
@limiter.limit(settings.rate_limit_free)
async def generate_diagram(
    request: Request,
    body: GenerateDiagramRequest,
    session: AsyncSession = Depends(db.get_session),
    _: str = Depends(verify_api_key),
) -> GenerateDiagramResponse:
    if not check_admin_ip(request):
        audit("admin.ip_blocked", request)
        raise HTTPException(status_code=403, detail="Forbidden")
    audit("admin.generate", request)
    return await _generate_diagram(body, session)


@router.post("/paid/diagrams/generate/svg", response_model=GenerateDiagramResponse)
@limiter.limit(settings.rate_limit_paid)
async def generate_diagram_paid_svg(
    request: Request,
    body: GenerateDiagramRequest,
    session: AsyncSession = Depends(db.get_session),
) -> GenerateDiagramResponse:
    body.format = "svg"
    return await _generate_diagram(body, session)


@router.post("/paid/diagrams/generate/png", response_model=GenerateDiagramResponse)
@limiter.limit(settings.rate_limit_paid)
async def generate_diagram_paid_png(
    request: Request,
    body: GenerateDiagramRequest,
    session: AsyncSession = Depends(db.get_session),
) -> GenerateDiagramResponse:
    body.format = "png"
    return await _generate_diagram(body, session)


@router.get(
    "/diagrams/{diagram_id}",
    response_model=DiagramResponse,
    include_in_schema=False,
)
async def get_diagram(
    diagram_id: UUID,
    request: Request,
    session: AsyncSession = Depends(db.get_session),
    _: str = Depends(verify_api_key),
) -> DiagramResponse:
    if not check_admin_ip(request):
        audit("admin.ip_blocked", request)
        raise HTTPException(status_code=403, detail="Forbidden")
    audit("admin.get_diagram", request, diagram_id=str(diagram_id))
    diagram = await db.get_diagram(session, diagram_id)
    if not diagram:
        raise HTTPException(status_code=404, detail="Diagram not found")
    return DiagramResponse.model_validate(diagram)


# ============================================================================
# Edit Endpoints (Free and Paid)
# ============================================================================


async def _edit_diagram(
    body: EditDiagramRequest,
    session: AsyncSession,
) -> EditDiagramResponse:
    """
    Apply edit operations to diagram source and optionally render.
    """
    try:
        # Apply all operations in sequence
        engine = DiagramEditEngine()
        operations_list = [op.model_dump(by_alias=True) for op in body.operations]
        edited_source, ops_count = engine.apply_operations(
            body.diagram_source, operations_list
        )
    except EditEngineException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # Optionally render after edit
    rendered = None
    if body.render_after_edit:
        try:
            rendered = await renderer.render(
                edited_source, body.diagram_type, body.format
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return EditDiagramResponse(
        ok=True,
        endpoint="/diagrams/edit",
        diagram_type=body.diagram_type,
        format=body.format,
        edited_diagram_source=edited_source,
        rendered=rendered,
        metadata={
            "operations_applied": ops_count,
            "render_after_edit": body.render_after_edit,
        },
    )


@router.post(
    "/diagrams/edit",
    response_model=EditDiagramResponse,
    include_in_schema=False,
)
@limiter.limit(settings.rate_limit_free)
async def edit_diagram(
    request: Request,
    body: EditDiagramRequest,
    session: AsyncSession = Depends(db.get_session),
    _: str = Depends(verify_api_key),
) -> EditDiagramResponse:
    """
    Free endpoint: Apply edit operations to diagram source.
    Rate limited to 5 edits per day per IP.
    """
    if not check_admin_ip(request):
        audit("admin.ip_blocked", request)
        raise HTTPException(status_code=403, detail="Forbidden")
    audit("admin.edit", request)
    return await _edit_diagram(body, session)


@router.post("/paid/diagrams/edit/svg", response_model=EditDiagramResponse)
@limiter.limit(settings.rate_limit_paid)
async def edit_diagram_paid_svg(
    request: Request,
    body: EditDiagramRequest,
    session: AsyncSession = Depends(db.get_session),
) -> EditDiagramResponse:
    """
    Paid endpoint: Apply edit operations and render as SVG.
    Requires x402 payment (0.05 USDC).
    """
    body.format = "svg"
    return await _edit_diagram(body, session)


@router.post("/paid/diagrams/edit/png", response_model=EditDiagramResponse)
@limiter.limit(settings.rate_limit_paid)
async def edit_diagram_paid_png(
    request: Request,
    body: EditDiagramRequest,
    session: AsyncSession = Depends(db.get_session),
) -> EditDiagramResponse:
    """
    Paid endpoint: Apply edit operations and render as PNG.
    Requires x402 payment (0.05 USDC).
    """
    body.format = "png"
    return await _edit_diagram(body, session)
