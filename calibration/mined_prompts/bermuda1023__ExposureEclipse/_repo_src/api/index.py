"""Vercel Python serverless entrypoint.

Re-exports the FastAPI app from `backend/app/main.py` so Vercel's
``@vercel/python`` runtime can serve it. Routes registered with prefix
``/api/...`` resolve as-is because Vercel rewrites ``/api/(.*)`` to this
function while preserving the original request path.

The mock provider's fixtures live at ``mockdata/`` relative to the repo root
(bundled with the function via ``includeFiles`` in ``vercel.json``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make `from app.main import app` resolve against backend/app.
sys.path.insert(0, str(REPO_ROOT / "backend"))

# Point the mock provider at the bundled fixtures unless the caller overrides.
os.environ.setdefault("MOCK_DATA_DIR", str(REPO_ROOT / "mockdata"))

# CORS only needs to allow the Vercel domain itself; both share an origin so
# this is mostly a safety net for preview deploys.
os.environ.setdefault(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:5173,http://localhost:4173,https://*.vercel.app",
)

from app.main import app  # noqa: E402  (must come after sys.path setup)
