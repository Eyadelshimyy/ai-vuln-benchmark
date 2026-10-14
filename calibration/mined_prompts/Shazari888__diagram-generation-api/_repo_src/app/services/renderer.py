import asyncio
import base64
import io
from urllib.parse import quote

import httpx

from app.config import settings


def _convert_svg_to_pdf_without_cairo(svg_bytes: bytes) -> bytes:
    try:
        from reportlab.graphics import renderPDF
        from svglib.svglib import svg2rlg
    except Exception as exc:
        raise RuntimeError(
            "SVG conversion to PDF requires either CairoSVG with system cairo libraries or the Python packages reportlab and svglib."
        ) from exc

    drawing = svg2rlg(io.BytesIO(svg_bytes))
    if drawing is None:
        raise RuntimeError("SVG conversion to PDF failed: could not parse SVG content.")
    return renderPDF.drawToString(drawing)


def _convert_svg_to_png_without_cairo(svg_bytes: bytes) -> bytes:
    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        raise RuntimeError(
            "SVG conversion to PNG requires CairoSVG with system cairo libraries or the Python package pypdfium2."
        ) from exc

    # Convert SVG -> PDF with the cairo-free path, then rasterize first page to PNG.
    pdf_bytes = _convert_svg_to_pdf_without_cairo(svg_bytes)
    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf[0]
    bitmap = page.render(scale=2)
    pil_image = bitmap.to_pil()
    output = io.BytesIO()
    pil_image.save(output, format="PNG")
    return output.getvalue()


def _convert_svg(svg_bytes: bytes, output_format: str) -> bytes:
    try:
        import cairosvg
    except Exception:
        cairosvg = None

    if output_format == "png":
        if cairosvg is not None:
            try:
                return cairosvg.svg2png(bytestring=svg_bytes)
            except Exception:
                # Some environments have CairoSVG installed but lack cairo runtime libs.
                pass
        return _convert_svg_to_png_without_cairo(svg_bytes)

    if output_format == "pdf":
        if cairosvg is not None:
            try:
                return cairosvg.svg2pdf(bytestring=svg_bytes)
            except Exception:
                # Some environments have CairoSVG installed but lack cairo runtime libs.
                pass
        return _convert_svg_to_pdf_without_cairo(svg_bytes)

    raise ValueError(f"Unsupported conversion format: {output_format}")


def _is_retryable_kroki_failure(response: httpx.Response) -> bool:
    if response.status_code in {429, 500, 502, 503, 504}:
        return True
    if response.status_code != 400:
        return False
    body = response.text.lower()
    return "failed to launch the browser process" in body or "resource temporarily unavailable" in body


def _is_unsupported_output_format(response: httpx.Response) -> bool:
    return response.status_code == 400 and "unsupported output format" in response.text.lower()


def _is_mermaid_launch_failure(response: httpx.Response) -> bool:
    if response.status_code not in {400, 500, 502, 503, 504}:
        return False
    body = response.text.lower()
    return "failed to launch the browser process" in body or "resource temporarily unavailable" in body


def _encode_mermaid_ink_source(source: str) -> str:
    # Mermaid Ink accepts base64 source directly in the URL path.
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return quote(encoded, safe="")


async def _render_mermaid_ink(source: str, output_format: str) -> str:
    encoded = _encode_mermaid_ink_source(source)
    if output_format == "svg":
        path = "svg"
    elif output_format == "png":
        path = "img"
    elif output_format == "pdf":
        path = "pdf"
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    url = f"{settings.mermaid_ink_base_url.rstrip('/')}/{path}/{encoded}"
    params = None
    if output_format == "png":
        params = {"type": "png"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=30.0)
    response.raise_for_status()

    if output_format == "svg":
        return response.text
    return base64.b64encode(response.content).decode("ascii")


async def _post_kroki(client: httpx.AsyncClient, url: str, source: str) -> httpx.Response:
    delays = [0.4, 1.0, 2.0, 3.0, 5.0]
    attempt = 0
    while True:
        try:
            response = await client.post(
                url,
                content=source.encode(),
                headers={"Content-Type": "text/plain"},
                timeout=30.0,
            )
        except httpx.TransportError:
            if attempt >= len(delays):
                raise
            await asyncio.sleep(delays[attempt])
            attempt += 1
            continue

        if response.is_success or not _is_retryable_kroki_failure(response) or attempt >= len(delays):
            return response

        await asyncio.sleep(delays[attempt])
        attempt += 1


async def render(source: str, diagram_type: str, output_format: str = "svg") -> str:
    # Mermaid Ink is fast and purpose-built for mermaid; try it first to avoid
    # the long Kroki retry chain (which uses a headless browser that often fails
    # on shared cloud infra).
    if diagram_type.lower() == "mermaid":
        try:
            return await _render_mermaid_ink(source, output_format)
        except httpx.HTTPError:
            pass  # fall through to Kroki

    url = f"{settings.kroki_base_url}/{diagram_type}/{output_format}"
    async with httpx.AsyncClient() as client:
        response = await _post_kroki(client, url, source)
    if response.is_success:
        if output_format == "svg":
            return response.text
        return base64.b64encode(response.content).decode("ascii")

    if output_format in {"png", "pdf"} and _is_unsupported_output_format(response):
        svg_url = f"{settings.kroki_base_url}/{diagram_type}/svg"
        async with httpx.AsyncClient() as client:
            svg_response = await _post_kroki(client, svg_url, source)

        if diagram_type.lower() == "mermaid" and _is_mermaid_launch_failure(svg_response):
            try:
                return await _render_mermaid_ink(source, output_format)
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Mermaid Ink fallback failed: {exc}") from exc

        if not svg_response.is_success:
            raise RuntimeError(
                f"Kroki SVG fallback failed with status {svg_response.status_code}: {svg_response.text[:300]}"
            )

        converted_bytes = _convert_svg(svg_response.content, output_format)
        return base64.b64encode(converted_bytes).decode("ascii")

    if diagram_type.lower() == "mermaid" and _is_mermaid_launch_failure(response):
        try:
            return await _render_mermaid_ink(source, output_format)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Mermaid Ink fallback failed: {exc}") from exc

    raise RuntimeError(
        f"Kroki render failed with status {response.status_code}: {response.text[:300]}"
    )
