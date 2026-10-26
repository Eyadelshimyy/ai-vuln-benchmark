"""Excel export endpoint — full workbook per PRODUCT_REQUIREMENTS.md §15."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..brand import EXPORT_FILENAME
from ..providers import ExposureDataProvider, get_provider
from ..services.export_excel import build_export_xlsx

router = APIRouter(prefix="/exports", tags=["exports"])

_XLSX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


@router.post("/excel")
def export_excel(
    payload: dict[str, Any],
    provider: ExposureDataProvider = Depends(get_provider),
) -> StreamingResponse:
    """Build the workbook synchronously and stream it back."""
    xlsx_bytes = build_export_xlsx(payload, provider)
    filename = EXPORT_FILENAME
    return StreamingResponse(
        iter([xlsx_bytes]),
        media_type=_XLSX_CONTENT_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


__all__ = ["router"]
