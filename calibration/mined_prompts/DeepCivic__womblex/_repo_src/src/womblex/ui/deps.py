"""Where the console reads run state from.

The sidecar reaches run state exactly the way a worker does, so it adds no
configuration surface of its own: a local deployment points at an
``output_root`` that is bind-mounted
read-only, and a cloud deployment sets ``WOMBLEX_STORE_URI`` — the same
variable ``womblex-cloud`` already reads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request

from womblex.store.remote import assert_disjoint_locations


@dataclass(frozen=True)
class UISettings:
    """Where the console reads run state from. Exactly one of the two is set.

    ``db_dsn`` is the optional job queue the dashboard reads. It is
    orthogonal to the run source rather than paired with it: a queue is
    present whenever one is configured (env or argument) and absent
    otherwise, and a deployment with no queue falls back to the per-stage
    checkpoints inside the run itself.

    ``presets_dir`` is where the Pipeline Composer *saves* operator-authored
    presets (docs/ui-plan.md merge 9) — one JSON file per preset. It applies
    to *local* mode only, it is local-mode only: remote mode has no
    equivalent field because it always writes presets under the store's own
    ``presets/`` prefix, a sibling of ``runs/`` in the same
    bucket — so a store-backed console needs no writable mount to save presets.
    In local mode, ``None`` disables saving: the built-in presets still serve,
    but ``POST /api/composer/presets`` refuses with 409.

    ``ingest_uri`` names where source documents arrive from, distinct from
    ``store_uri``/``output_root``. ``None`` means the Execution Controls
    cannot dispatch; everything read-only still serves. When both are set they
    must be disjoint of the output store's effective ``runs/`` prefix,
    enforced here at construction.
    """

    output_root: Path | None
    store_uri: str | None
    db_dsn: str | None = None
    presets_dir: Path | None = None
    ingest_uri: str | None = None

    def __post_init__(self) -> None:
        if bool(self.output_root) == bool(self.store_uri):
            raise ValueError("UISettings needs exactly one of output_root or store_uri")
        if self.ingest_uri and self.store_uri:
            assert_disjoint_locations(self.ingest_uri, self.store_uri)

    @property
    def is_remote(self) -> bool:
        return self.store_uri is not None

    @property
    def presets_writable(self) -> bool:
        """Whether this deployment can save presets at all.

        Remote mode always can (it writes to the store's ``presets/`` prefix); local mode can only when a writable ``presets_dir`` was
        configured. The composer's save/delete routes 409 when this is false.
        """
        return self.is_remote or self.presets_dir is not None


def resolve_settings(
    output_root: Path | None,
    store_uri: str | None,
    *,
    db_dsn: str | None = None,
    presets_dir: Path | None = None,
    ingest_uri: str | None = None,
) -> UISettings:
    """Resolve settings from explicit arguments, falling back to env vars.

    ``$WOMBLEX_UI_OUTPUT_ROOT`` names a local run root;
    ``$WOMBLEX_STORE_URI`` names an object store. Raises ``ValueError`` when
    both or neither resolve — the console reads exactly one run source, and
    silently preferring one over the other would hide a misconfigured
    deployment behind an empty run list.
    ``$WOMBLEX_UI_PRESETS_DIR`` is the env fallback for ``presets_dir`` — the
    directory the composer saves operator-authored presets into. Absent means
    saving is disabled (built-in presets still serve).

    ``$WOMBLEX_DB_DSN`` / ``$DATABASE_URL`` name the job queue, the same
    pair ``womblex worker`` reads. Absent is not an error: the dashboard
    falls back to checkpoints.

    ``$WOMBLEX_INGEST_URI`` is the env fallback for ``ingest_uri`` — the same
    variable ``womblex enqueue``/``worker`` read. Absent means no ingest is
    configured. When resolved alongside a store, ``UISettings`` itself checks
    the two are disjoint and raises ``ValueError`` naming both.
    """
    root = output_root
    if root is None and "WOMBLEX_UI_OUTPUT_ROOT" in os.environ:
        root = Path(os.environ["WOMBLEX_UI_OUTPUT_ROOT"])
    store = store_uri or os.environ.get("WOMBLEX_STORE_URI")
    if root and store:
        raise ValueError("pass only one of --output-root / --store (or their env vars)")
    if not root and not store:
        raise ValueError(
            "no run source: pass --output-root or --store "
            "(or set $WOMBLEX_UI_OUTPUT_ROOT / $WOMBLEX_STORE_URI)"
        )
    dsn = db_dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")
    presets = presets_dir
    if presets is None and "WOMBLEX_UI_PRESETS_DIR" in os.environ:
        presets = Path(os.environ["WOMBLEX_UI_PRESETS_DIR"])
    ingest = ingest_uri or os.environ.get("WOMBLEX_INGEST_URI")
    return UISettings(
        output_root=root, store_uri=store,
        db_dsn=dsn, presets_dir=presets, ingest_uri=ingest,
    )


def get_settings(request: Request) -> UISettings:
    """FastAPI dependency: this deployment's settings (flags + env at start-up)."""
    settings: UISettings = request.app.state.settings
    return settings
