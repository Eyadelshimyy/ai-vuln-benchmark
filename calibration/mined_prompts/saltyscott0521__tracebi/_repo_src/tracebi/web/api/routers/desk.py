"""The desk review list. GET, so a viewer may read it."""

import os

from fastapi import APIRouter

from tracebi.desk import review

router = APIRouter(tags=["desk"])


def _loaded_models() -> dict:
    from tracebi.model_registry import get_model, list_models

    models: dict = {}
    for name in list_models():
        try:
            model = get_model(name)
        except Exception:  # noqa: BLE001 — one broken model stays off the desk
            continue
        models[name] = model
        named = getattr(model, "name", None)
        if named:
            models[named] = model
    return models


@router.get("/desk")
def desk():
    """Pins, exploration drafts, receipts that do not reproduce, sink status."""
    return review(os.getcwd(), _loaded_models())
