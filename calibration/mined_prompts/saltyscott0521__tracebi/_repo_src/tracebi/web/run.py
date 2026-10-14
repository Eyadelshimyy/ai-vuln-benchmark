"""
Development entrypoint for the TraceBi web API.

    python -m tracebi.web.run            # default port 8000, hot-reload on
    python -m tracebi.web.run --port 9000

Production (e.g. Docker):
    uvicorn tracebi.web.api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import argparse
import sys
import os


def _ensure_importable() -> None:
    """Make ``tracebi.web.api.main`` resolvable when run as a plain script.

    ``python -m tracebi.web.run`` already has the package on sys.path.
    ``python tracebi/web/run.py`` does not — sys.path[0] is this directory —
    so add the repo root, which is three levels up from here.
    """
    if "tracebi" in sys.modules:
        return
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.insert(0, root)


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceBi API dev server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-reload", action="store_true")
    args = parser.parse_args()

    _ensure_importable()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required:  pip install uvicorn")
        sys.exit(1)

    uvicorn.run(
        "tracebi.web.api.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )


if __name__ == "__main__":
    main()
