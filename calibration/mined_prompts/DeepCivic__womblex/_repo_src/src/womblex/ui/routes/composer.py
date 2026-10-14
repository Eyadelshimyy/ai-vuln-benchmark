"""``/api/composer`` — the Pipeline Composer's graph, form schema, presets and
config validation (docs/ui-plan.md merge 9).

The read surface (`graph`, `schema`, `validate`, `yaml`, `GET /presets`) touches
no run and no filesystem — it builds a `WomblexConfig` in memory. Saving a
preset is the one write: `POST`/`DELETE /presets` file one JSON per preset —
under the store's own `presets/` prefix in remote mode (a sibling of `runs/`,
so no writable mount), or a local `presets_dir`. They refuse
with 409 where the console cannot write (local mode with no presets dir), like
the Execution Controls without a queue. `readers` owns the local-vs-store
split.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, ValidationError

from womblex.ui import composer, readers
from womblex.ui.deps import UISettings, get_settings
from womblex.ui.readers import StoreUnreachable

router = APIRouter(prefix="/api/composer", tags=["composer"])


@router.get("/graph")
def get_graph() -> dict:
    """The pipeline DAG `STAGE_CONTRACTS` implies — nodes, required-input edges."""
    return composer.get_stage_graph()


@router.get("/presets")
def get_presets(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """Named pre-configured pipelines the form offers as starting points.

    Built-in presets first, then any the operator has saved — under the store's
    ``presets/`` prefix in remote mode, or the local presets dir. Each is a
    *partial* `WomblexConfig` (no `dataset` / `paths` — the operator still
    supplies the run's identity and paths). `DEFAULT-Isaacus` is the reference
    extract → chunk → enrich → build_graph → money shape. Each carries `source`
    (`builtin` | `saved`) so the form can offer delete only on the ones it can
    delete.
    """
    try:
        presets = readers.list_all_presets(settings)
    except StoreUnreachable as e:
        raise HTTPException(status_code=503, detail=f"preset store unreachable: {e}") from e
    return {"presets": presets}


@router.get("/presets/{name}")
def get_preset(name: str, settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """One preset by name (a saved one shadowing a built-in); 404 if none."""
    try:
        preset = readers.get_any_preset(settings, name)
    except StoreUnreachable as e:
        raise HTTPException(status_code=503, detail=f"preset store unreachable: {e}") from e
    if preset is None:
        raise HTTPException(status_code=404, detail=f"no such preset: {name!r}")
    return preset.as_dict()


class SavePresetRequest(BaseModel):
    """Save-a-preset form. `config` is a whole composed config; `dataset`/`paths`
    are stripped on save (a preset is an overlay). `description`/`formats` optional."""

    name: str = Field(..., min_length=1)
    description: str = ""
    formats: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


@router.post("/presets", status_code=201)
def post_preset(
    body: SavePresetRequest, settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """Save *body* as a named preset (docs/ui-plan.md merge 9).

    Remote mode writes to the store's own ``presets/`` prefix (a sibling of
    ``runs/``), so a store-backed console always saves and
    needs no writable mount. Local mode needs a writable ``presets_dir``:
    409 when none is configured (wire up `--presets-dir` /
    `$WOMBLEX_UI_PRESETS_DIR`). 400 on an unsafe name or a config that would
    not load as an overlay (the same `WomblexConfig(**raw)` construction the
    built-ins are validated with). Overwriting a saved preset of the same name
    is allowed — it is the operator's own to replace.
    """
    if not settings.presets_writable:
        raise HTTPException(
            status_code=409,
            detail="This console has no presets dir configured; saving is disabled. "
                   "Set --presets-dir (or $WOMBLEX_UI_PRESETS_DIR) to save presets.",
        )
    try:
        preset = readers.save_preset(
            settings,
            name=body.name, description=body.description,
            formats=tuple(body.formats), config=body.config,
        )
    except ValidationError as e:
        detail = e.errors(include_url=False, include_context=False, include_input=False)
        raise HTTPException(status_code=400, detail=detail) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return preset.as_dict()


@router.delete("/presets/{name}")
def delete_preset(
    name: str, settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """Delete a saved preset by name.

    409 when this console cannot write presets at all (local mode with no
    presets dir), 404 when no *saved* preset by that name exists — a built-in
    is code and cannot be deleted, so a name that matches only a built-in is a
    404 here, not a silent success.
    """
    if not settings.presets_writable:
        raise HTTPException(
            status_code=409,
            detail="This console has no presets dir configured; nothing to delete.",
        )
    if not readers.delete_saved_preset(settings, name):
        raise HTTPException(status_code=404, detail=f"no such saved preset: {name!r}")
    return {"deleted": name}


@router.get("/schema")
def get_schema() -> dict:
    """`WomblexConfig`'s JSON Schema, for the composer form to render fields from."""
    return composer.get_config_schema()


@router.post("/validate")
def post_validate(
    raw: dict[str, Any], settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """Whether *raw* builds a valid `WomblexConfig`, plus Pydantic's own errors.

    A key the schema does not claim is a validation error, as it is for the
    CLI. `paths` is filled in from this deployment's ingest/output locations —
    the schema does not offer it (`get_config_schema`), so *raw* never carries one.
    """
    return composer.validate_config(raw, settings)


@router.post("/yaml")
def post_yaml(
    raw: dict[str, Any], settings: UISettings = Depends(get_settings),  # noqa: B008
) -> Response:
    """Validate *raw* and return it as a downloadable YAML file.

    422 with Pydantic's own errors on an invalid config — the same shape
    `/validate` reports — there is no separate "download anyway" path for a
    config the CLI would reject, a stray key included.
    """
    try:
        text = composer.render_yaml(raw, settings)
    except ValidationError as e:
        detail = e.errors(include_url=False, include_context=False, include_input=False)
        raise HTTPException(status_code=422, detail=detail) from e
    return Response(
        content=text,
        media_type="application/yaml",
        headers={"Content-Disposition": "attachment; filename=womblex.yaml"},
    )
