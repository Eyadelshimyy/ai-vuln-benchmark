"""Web service entrypoint.

Serves the channel ingress route and the health check from the production app,
so the deployed process is never the provider-free app used by tests.

    gvas-web
"""

import argparse
import os
from collections.abc import Sequence

import uvicorn

APP_FACTORY = "gvas.composition.production:create_production_app"
DEFAULT_PORT = 8000


def default_port() -> int:
    """Hosts that assign a port publish it as ``PORT``."""

    value = os.environ.get("PORT", "").strip()
    return int(value) if value.isdigit() else DEFAULT_PORT


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the GVAS web service")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=default_port())
    arguments = parser.parse_args(argv)
    uvicorn.run(APP_FACTORY, factory=True, host=arguments.host, port=arguments.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
